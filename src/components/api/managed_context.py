"""Opt-in, backend-only adapter for externally authorized retrieval results.

It reuses the graph and output guards; default public request schemas cannot
inject documents. No network retrieval credentials or human identity live here.
"""
import hmac
import json
import re

from fastapi import APIRouter, HTTPException, Request
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from components.orchestration.renderers import MarkdownRenderer
from components.orchestration.streaming import (
    consume_stream, make_output_classifier, make_output_filter,
    prepare_conversation, process_query_streaming,
)


class ContextDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1, max_length=8000)
    title: str = Field(min_length=1, max_length=200)
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    ordinal: int = Field(ge=0, le=100000)


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str = Field(pattern=r"^(user|assistant)$")
    content: str = Field(min_length=1, max_length=16000)


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[HistoryMessage] = Field(min_length=1, max_length=9)
    documents: list[ContextDocument] = Field(max_length=5)


class StrictRenderer(MarkdownRenderer):
    def error(self, content):
        # Provider failure is an HTTP failure, never a successful answer string.
        raise HTTPException(503, "Generation service temporarily unavailable.")


def build_managed_context_router(graph, *, key, blocklist=None,
                                 blocklist_notice="[response withheld]",
                                 classification_config=None):
    if not re.fullmatch(r"[a-f0-9]{64}", key or ""):
        raise ValueError("Managed context requires a dedicated 256-bit key.")
    router = APIRouter()

    @router.post("/internal/v1/managed-answer", include_in_schema=False)
    async def answer(request: Request):
        expected = "Bearer " + key
        if not hmac.compare_digest(request.headers.get("authorization", ""), expected):
            raise HTTPException(401, "Service credential required.")
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 65536:
                raise HTTPException(413, "Request too large.")
        try:
            value = ContextRequest.model_validate(json.loads(raw))
        except (ValidationError, ValueError, UnicodeError):
            raise HTTPException(400, "Invalid managed context.") from None
        if (value.messages[-1].role != "user"
                or sum(len(d.content.encode()) for d in value.documents) > 24000
                or sum(len(m.content.encode()) for m in value.messages) > 24000):
            raise HTTPException(400, "Invalid managed context.")
        messages = [m.model_dump() for m in value.messages]
        query, context, user_history = prepare_conversation(messages, max_turns=4, max_chars=16000)
        documents = [Document(page_content=d.content, metadata={
            "filename": d.title, "source": "managed:" + d.id,
            "artifact_id": d.id, "segment": d.ordinal,
        }) for d in value.documents]
        renderer = StrictRenderer(trailing_flush_delay=0)
        chunks = []
        async for chunk in consume_stream(
            process_query_streaming(graph, query, conversation_context=context,
                                    user_messages_history=user_history,
                                    supplied_documents=documents, session_type="managed"),
            make_output_filter(blocklist, blocklist_notice),
            make_output_classifier(classification_config), renderer,
        ):
            chunks.append(chunk)
            if len("".join(chunks).encode()) > 16000:
                raise HTTPException(503, "Generation exceeded response limit.")
        content = "".join(chunks)
        if not content.strip():
            raise HTTPException(503, "Empty generation response.")
        return {"answer": content}

    return router
