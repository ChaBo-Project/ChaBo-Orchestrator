"""
OpenAI-compatible chat surface: POST /v1/chat/completions, GET /v1/models.

Endpoint for standardized chat frontends (OpenWebUI, LibreChat etc.)
Uses the same graph as Chabo-ChatUI - i.e. `consume_stream` (only the renderer differs).

Deviations from OpenAI's API:
- Sampling parameters (`temperature`, `top_p`, `max_tokens`, …) are ignored.
- `usage` is omitted. Token accounting isn't available uniformly across inference providers
- Client-supplied `system` message is ignored. The system prompt is set in Chabo `prompts.py`
- `files1 field carries attachments:
   - `files`: [{"name": ..., "content": "<extracted text>"}] 
   - generic frontends process files themselves and send raw text. 
   - Raw-file processing still exists on the Chabo-ChatUI route.
- Responses may carry a top-level `citations` array (Perplexity-style). Sources are always *also*
  rendered as markdown in the message body.
"""
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from components.guardrails.output_classification import OutputClassificationConfig
from components.ingestor.ingestor import process_text
from components.orchestration.renderers import MarkdownRenderer, OpenAIChunkRenderer
from components.orchestration.streaming import (
    consume_stream,
    make_output_classifier,
    make_output_filter,
    prepare_conversation,
    process_query_streaming,
)

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """
    One OpenAI-shaped message.

    `content` may be a plain string or the multimodal parts array; extra keys (tool calls,
    `name`, UI-specific annotations) are ignored.
    """
    model_config = ConfigDict(extra="allow")

    role: str
    content: Union[str, List[Dict[str, Any]], None] = None


class ChatCompletionRequest(BaseModel):
    """
    A /v1/chat/completions request. Unknown fields are accepted (see module docstring).
    """
    model_config = ConfigDict(extra="allow")

    messages: List[ChatMessage] = Field(default_factory=list)
    model: Optional[str] = None
    stream: bool = False
    #: This turn's attachments: [{"name": "report.pdf", "content": "<extracted text>"}].
    files: Optional[List[Dict[str, Any]]] = None


def extract_text(content: Union[str, List[Dict[str, Any]], None]) -> str:
    """
    Flatten OpenAI message content to plain text.

    Keeps only `text` — images and other modalities are dropped, since the RAG pipeline is text-only.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
        elif isinstance(part, str):
            parts.append(part)
    return "\n".join(p for p in parts if p)


def resolve_attachments(request: ChatCompletionRequest, config) -> tuple[Optional[str], Optional[str]]:
    """
    Resolve this turn's attachments to (ingestor_context, filename).

    Text is chunked and capped exactly as an uploaded file would be, so an attachment can't
    blow the context window just because the frontend did the extraction. `filename` is the
    citation label only - nothing is parsed from it.

    config is passed through to process_text() from main.py

    MAX 1 ATTACHMENT PER TURN
    """
    if request.files and len(request.files) > 1:
        raise HTTPException(
            status_code=400,
            detail="Only one attachment per turn is supported. Send a separate turn per file.",
        )

    chunks: List[str] = []
    names: List[str] = []

    for entry in request.files or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "attachment")
        text = entry.get("content")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Attachment '{name}' carries no text. `files` entries must be "
                       "{'name': ..., 'content': '<extracted text>'}.",
            )
        try:
            chunks.append(process_text(text, name, config))
        except Exception as e:
            logger.error("Attachment processing failed for %s: %s", name, e, exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to process attachment '{name}': {e}"
            ) from e
        names.append(name)

    if not chunks:
        return None, None
    return "\n\n".join(chunks), ", ".join(names)


def build_openai_router(
    compiled_graph,
    *,
    config,
    model_name: str = "chabo",
    max_turns: int = 3,
    max_chars: int = 8000,
    blocklist=None,
    blocklist_notice: str = "[response withheld]",
    classification_config: Optional[OutputClassificationConfig] = None,
    include_citations: bool = True,
) -> APIRouter:
    """
    Build the OpenAI-compatible router.

    Everything the graph needs is injected here (same pattern as the LangServe adapters), so
    the route handlers stay free of module-level state.
    """
    router = APIRouter(prefix="/v1", tags=["openai"])

    def _run_pipeline(request: ChatCompletionRequest):
        """Common request unpacking -> the internal event stream for this turn."""
        if not request.messages:
            raise HTTPException(status_code=400, detail="`messages` must not be empty.")

        # Normalise to the {role, content-as-text} shape the conversation helpers expect.
        messages = [
            {"role": m.role, "content": extract_text(m.content)} for m in request.messages
        ]
        query, conversation_context, user_messages_history = prepare_conversation(
            messages, max_turns=max_turns, max_chars=max_chars
        )
        if not (query or "").strip():
            raise HTTPException(status_code=400, detail="No user message found in `messages`.")

        ingestor_context, filename = resolve_attachments(request, config)

        return process_query_streaming(
            compiled_graph=compiled_graph,
            query=query,
            conversation_context=conversation_context,
            user_messages_history=user_messages_history,
            file_content=None,
            filename=filename,
            ingestor_context=ingestor_context,
            session_type="openai",
        )

    @router.get("/models")
    async def list_models():
        """
        Model listing — necessary for OpenWebUI/LibreChat to initiate chat
        - They use it to populate their model selector
        - Has no bearing on the actual LLM used (defined in config)
        """
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "chabo",
                }
            ],
        }

    @router.post("/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """Chat completion, streamed as SSE when `stream: true`, otherwise a single JSON body."""
        response_model = request.model or model_name
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        process_iter = _run_pipeline(request)

        output_filter = make_output_filter(blocklist, blocklist_notice)
        classifier = make_output_classifier(classification_config)

        if request.stream:
            renderer = OpenAIChunkRenderer(
                model=response_model,
                request_id=request_id,
                include_citations=include_citations,
            )

            async def event_stream():
                try:
                    async for frame in consume_stream(
                        process_iter, output_filter, classifier, renderer
                    ):
                        yield frame
                except Exception as e:  # transport-level failure after headers were sent
                    logger.error("OpenAI stream failed: %s", e, exc_info=True)
                    yield renderer.error(str(e))
                    yield renderer.finish()

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # don't let a proxy buffer the stream
                },
            )

        # Non-streaming: same pipeline, same renderer output, collected into one message.
        renderer = MarkdownRenderer(trailing_flush_delay=0.0)
        parts = []
        try:
            async for piece in consume_stream(process_iter, output_filter, classifier, renderer):
                parts.append(piece)
        except Exception as e:
            logger.error("OpenAI non-streaming request failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e
        content = "".join(parts)

        response: Dict[str, Any] = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        if include_citations and renderer.citations:
            response["citations"] = renderer.citations
        return response

    return router
