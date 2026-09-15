"""
title: ChaBo RAG
author: ChaBo
version: 0.2.0
license: Apache-2.0
description: Routes OpenWebUI chats to a ChaBo RAG orchestrator, forwarding attachment text.
requirements: aiohttp
"""
# ---------------------------------------------------------------------------------------
# This file does NOT run inside ChaBo. It is pasted into OpenWebUI (Workspace -> Functions ->
# "+" -> paste -> save, then enable it and set the valves). It lives in this repo so the
# bridge is versioned alongside the API it talks to.
#
# What it does:
#   chat        -> POST {CHABO_URL}/v1/chat/completions (streamed, SSE)
#   attachments -> the text OpenWebUI already extracted, forwarded on that same request as
#                  `files: [{"name", "content"}]`
#
# Why a bridge is needed at all: OpenWebUI does its own file extraction and does not forward
# uploaded files to a custom backend over the chat call (open-webui#17293).
#
# What it deliberately does NOT do: fetch the original bytes and make ChaBo re-extract them.
# OpenWebUI has already done that work, so re-doing it costs a round trip and a parse per
# turn and gains nothing.
#
# STATUS - the chat path is straightforward; the attachment path is NOT yet verified against
# a live OpenWebUI. What a Pipe can actually see of an upload is unresolved upstream
# (open-webui#19963). Set DEBUG_FILES to log the shape of what arrives; that log IS the spike.
# ---------------------------------------------------------------------------------------
import json
import logging
from typing import Any, AsyncGenerator, Dict, List

import aiohttp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Pipe:
    class Valves(BaseModel):
        CHABO_URL: str = Field(
            default="http://chabo:7860",
            description="Base URL of the ChaBo orchestrator (no trailing slash).",
        )
        MODEL_ID: str = Field(
            default="chabo",
            description="Model id to request — must match [api] model_name in params.cfg.",
        )
        MODEL_NAME: str = Field(
            default="ChaBo RAG",
            description="Display name shown in the OpenWebUI model picker.",
        )
        REQUEST_TIMEOUT: int = Field(
            default=300, description="Seconds to wait for a full answer."
        )
        FORWARD_UPLOADS: bool = Field(
            default=True,
            description="Forward the text of uploaded files to ChaBo as chat attachments.",
        )
        DEBUG_FILES: bool = Field(
            default=False,
            description="Log the structure of incoming file payloads (use this to run the spike).",
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> List[dict]:
        """One entry: this ChaBo deployment."""
        return [{"id": self.valves.MODEL_ID, "name": self.valves.MODEL_NAME}]

    def _collect_attachments(self, body: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Turn OpenWebUI's file payload into ChaBo `files` entries.

        Never raises: an attachment we can't read degrades to a normal (corpus-only) answer
        rather than a failed chat.
        """
        files = body.get("files") or []
        if self.valves.DEBUG_FILES:
            logger.info("ChaBo pipe — incoming file payload: %s", json.dumps(files, default=str)[:4000])
        if not files:
            return []

        attachments: List[Dict[str, str]] = []
        for entry in files:
            payload = entry.get("file", entry) if isinstance(entry, dict) else {}
            name = payload.get("filename") or payload.get("name") or "attachment"
            data = payload.get("data") or {}
            text = data.get("content") if isinstance(data, dict) else None
            if isinstance(text, str) and text.strip():
                attachments.append({"name": name, "content": text})
            else:
                logger.info("ChaBo pipe: no extracted text for attachment %s — skipping", name)
        return attachments

    # -- chat ----------------------------------------------------------------------------

    async def pipe(self, body: Dict[str, Any], **kwargs) -> AsyncGenerator[str, None]:
        """Stream one ChaBo answer back to OpenWebUI."""
        payload: Dict[str, Any] = {
            "model": self.valves.MODEL_ID,
            "messages": body.get("messages", []),
            "stream": True,
        }

        if self.valves.FORWARD_UPLOADS:
            attachments = self._collect_attachments(body)
            if attachments:
                payload["files"] = attachments

        timeout = aiohttp.ClientTimeout(total=self.valves.REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.valves.CHABO_URL}/v1/chat/completions", json=payload
            ) as resp:
                if resp.status != 200:
                    yield f"Error: ChaBo returned HTTP {resp.status} — {await resp.text()}"
                    return

                async for raw in resp.content:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
