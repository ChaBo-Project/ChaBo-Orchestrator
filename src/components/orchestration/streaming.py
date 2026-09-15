"""
Frontend-agnostic components of the LangGraph workflow stream.

Layering:
- process_query_streaming: runs the graph and normalises its custom events into ChaBo's internal event stream 
(data / filters_applied / sources / error / end).
- consume_stream: the single consumer of that stream: applies the output guards and renders each event through
a `BaseRenderer` (renderers.py).
- prepare_conversation / normalize_messages / decode_base64_file request unpacking shared by every frontend.

Every frontend entry point (ChatUI's adapters in ui_adapters.py, the OpenAI-compatible
routes in components/api/openai_compat.py) is built on these. Only the request shape and
the renderer differ per frontend.
"""
import base64
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple

from components.utils import build_conversation_context
from components.guardrails.output_guard import StreamingBlocklistFilter
from components.guardrails.output_classification import StreamingClassifier, OutputClassificationConfig
from .renderers import BaseRenderer, MarkdownRenderer

logger = logging.getLogger(__name__)


def normalize_messages(messages_value) -> List[Any]:
    """
    Accept dicts or objects for conversation messages and return objects with .role/.content.

    Frontends differ here:
    - LangServe hands over pydantic models
    - OpenAI routes hand over plain dicts
    """
    messages = []
    for msg in messages_value or []:
        if isinstance(msg, dict):
            messages.append(type('Message', (), {
                'role': msg.get('role', 'unknown'),
                'content': msg.get('content', '')
            })())
        else:
            messages.append(msg)
    return messages


def prepare_conversation(
    messages_value,
    fallback_text: str = "",
    max_turns: int = 3,
    max_chars: int = 8000,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Turn an incoming message list into the three things the graph needs.

    Returns (query, conversation_context, user_messages_history):
    - query: the latest user turn (or `fallback_text` when there are none)
    - conversation_context: user+assistant history for generation (last N turns)
    - user_messages_history user-turn-only history for filter extraction (assistant
                            responses and retrieved doc content are deliberately excluded,
                            since they cause spurious filter matches)
    """
    messages = normalize_messages(messages_value)
    if not messages:
        return fallback_text, None, None

    user_messages = [msg for msg in messages if msg.role == 'user']
    query = user_messages[-1].content if user_messages else fallback_text

    msg_metadata = {
        'total': len(messages),
        'user': len(user_messages),
        'assistant': len([m for m in messages if m.role == 'assistant']),
        'msg_lengths': [len(m.content) for m in messages]
    }
    logger.info(f"Processing query: {str(query)[:20]}... | Conversation: {msg_metadata}")

    conversation_context = build_conversation_context(messages, max_turns=max_turns, max_chars=max_chars)
    user_messages_history = "\n".join(
        f"USER: {msg.content}" for msg in user_messages[-max_turns:]
    ) if user_messages else None

    return query, conversation_context, user_messages_history


def decode_base64_file(file_info: Dict[str, Any]) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Decode one inline base64 file entry ({name, type: 'base64', content}) to (bytes, filename).
    """
    if file_info.get('type') != 'base64' or not file_info.get('content'):
        return None, None
    try:
        return base64.b64decode(file_info['content']), file_info.get('name', 'uploaded_file')
    except Exception as e:
        raise ValueError(f"Failed to decode uploaded file - {str(e)}") from e


async def consume_stream(
    process_iter,
    output_filter: Optional[StreamingBlocklistFilter] = None,
    classifier: Optional[StreamingClassifier] = None,
    renderer: Optional[BaseRenderer] = None,
):
    """
    Shared event consumer for every frontend.

    Maps process_query_streaming events through `renderer` (markdown by default) and appends
    the filters footnote + sources on `end`.

    Output guards (independent, both optional):
      - `classifier` (LLM classifier): observes the raw answer text and classifies it in
        windows. On a hit, the stream stops and the classifier message is displayed.
      - `output_filter` (blocklist): every token is routed through the streaming
        blocklist filter. On a hit the stream stops and the blocklist message is displayed.
    In either case the footnote/sources are suppressed on a hit.

    Guards see the raw generated text (not text formatted for UI) — the renderer is
    applied after a chunk has passed both guards.
    """
    renderer = renderer or MarkdownRenderer()
    delay = renderer.trailing_flush_delay
    filters_footnote = None
    sources_collected = None
    blocked = False

    try:
        out = renderer.prelude()
        if out:
            yield out

        async for result in process_iter:
            if not isinstance(result, dict):
                out = renderer.text(str(result))
                if out:
                    yield out
                await asyncio.sleep(0)
                continue

            result_type = result.get("type", "data")
            content = result.get("content", "")

            if result_type == "data":
                # LLM classifier observes the raw generated text first: a verdict from an
                # earlier window truncates BEFORE this chunk is shown (non-blocking check).
                if classifier is not None and classifier.feed(content):
                    out = renderer.notice(classifier.cfg.notice)
                    if out:
                        yield out
                    await asyncio.sleep(delay)
                    blocked = True
                    break
                if output_filter is not None:
                    # On a hit `emit` IS the blocklist notice (the buffer is dropped), so it
                    # is rendered as a notice rather than as answer text.
                    emit, hit = output_filter.feed(content)
                    if emit:
                        out = renderer.notice(emit) if hit else renderer.text(emit)
                        if out:
                            yield out
                    if hit:
                        blocked = True
                        break  # stop streaming the (now-blocked) answer
                else:
                    out = renderer.text(content)
                    if out:
                        yield out
            elif result_type == "filters_applied":
                filters_footnote = (content.get("filters", {}), content.get("narrowed", False))
            elif result_type == "sources":
                sources_collected = content
            elif result_type == "end":
                if output_filter is not None and not blocked:
                    tail, hit = output_filter.flush_final()
                    if tail:
                        out = renderer.notice(tail) if hit else renderer.text(tail)
                        if out:
                            yield out
                        await asyncio.sleep(delay)
                    if hit:
                        blocked = True
                if classifier is not None and not blocked:
                    notice, hit = await classifier.flush_final()
                    if hit:
                        out = renderer.notice(notice)
                        if out:
                            yield out
                        await asyncio.sleep(delay)
                        blocked = True
                if blocked:
                    break  # suppress footnote + sources on a blocked answer
                if filters_footnote:
                    out = renderer.footnote(*filters_footnote)
                    if out:
                        yield out
                    await asyncio.sleep(delay)
                if sources_collected:
                    logger.info("Appending markdown sources block")
                    out = renderer.sources(sources_collected)
                    if out:
                        yield out
            elif result_type == "error":
                out = renderer.error(content)
                if out:
                    yield out

            await asyncio.sleep(0)
    finally:
        if classifier is not None:
            await classifier.aclose()  # cancel any incoming (in-progress) classifications

    tail = renderer.finish()
    if tail:
        yield tail


def make_output_filter(blocklist, blocklist_notice: str) -> Optional[StreamingBlocklistFilter]:
    """
    Construct a fresh per-request blocklist filter instance, or None when the blocklist is off.
    """
    if blocklist is None:
        return None
    return StreamingBlocklistFilter(blocklist, blocklist_notice)


def make_output_classifier(classification_config: Optional[OutputClassificationConfig]) -> Optional[StreamingClassifier]:
    """
    Construct a fresh per-request LLM classifier instance, or None when the classifier is off.
    """
    if classification_config is None:
        return None
    return StreamingClassifier(classification_config)


async def process_query_streaming(
    compiled_graph,
    query: str,
    file_upload=None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    conversation_context: str = None,
    user_messages_history: str = None,
    file_content: bytes = None,
    filename: str = None,
    ingestor_context: str = None,
    session_type: str = "chatui",
):
    """
    Process a query through the LangGraph workflow with streaming.

    `ingestor_context` is already-chunked attachment text (from UIs that extract it themselves
    and send it on the chat request). `ingest_node` leaves it untouched when no raw
    `file_content` is supplied.
    """
    initial_state = {
        "query": query,
        "metadata": {"session_type": session_type},
        "raw_documents": [],
        "conversation_context": conversation_context,
        "metadata_filters": metadata_filters,
        "user_messages_history": user_messages_history,
    }

    # Add file content if present
    if file_content and filename:
        initial_state["file_content"] = file_content

    # Already-chunked attachment text (the frontend did the extraction)
    if ingestor_context:
        initial_state["ingestor_context"] = ingestor_context

    # The attachment's name - 2 considerations here:
    # `ingest_node` reads it to pick a parser (based on file type)
    # `generate_node_streaming` uses it as the attachment's citation label (which is why it
    # is maintained for pre-extracted text as well as files (otherwise a text attachment
    # is cited as "unknown").
    if filename and (file_content or ingestor_context):
        initial_state["filename"] = filename

    try:
        async for output in compiled_graph.astream(initial_state, stream_mode="custom"):
            if output.get("event") == "data":
                yield {"type": "data", "content": output["data"]}
            elif output.get("event") == "filters_applied":
                yield {"type": "filters_applied", "content": output["data"]}
            elif output.get("event") == "final_answer":
                # Handle final_answer event with webSources
                sources = output["data"].get("webSources", [])
                if sources:
                    yield {"type": "sources", "content": sources}
            elif output.get("event") == "error":
                yield {"type": "error", "content": output["data"].get("error", "Unknown error")}

        yield {"type": "end", "content": ""}

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        yield {"type": "error", "content": str(e)}
