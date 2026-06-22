"""
ChatUI Adapters for LangGraph Workflow Streaming
"""
import logging
import asyncio
import json
from typing import AsyncGenerator, Dict, Any, Optional

from components.utils import build_conversation_context
from components.guardrails.output_guard import StreamingBlocklistFilter

logger = logging.getLogger(__name__)


def _build_filters_footnote(filters: Dict, narrowed: bool) -> str:
    """Build a subtle italic footnote showing which filters were applied during retrieval."""
    parts = [
        f"{k}: {', '.join(v) if isinstance(v, list) else v}"
        for k, v in filters.items()
    ]
    base = "🔍 Searched within: " + " · ".join(parts)
    if narrowed:
        base += " (narrowed — combined filter returned no results)"
    return "*" + base + "*"


def _render_sources(sources_collected) -> str:
    """
    Render collected sources as markdown with doc:// URLs for ChatUI to parse.
    """
    sources_text = "\n\n**Sources:**\n"
    for i, source in enumerate(sources_collected, 1):
        if isinstance(source, dict):
            title = source.get('title', 'Unknown')
            uri = source.get('uri') or 'doc://#'
            sources_text += f"{i}. [{title}]({uri})\n"
        else:
            sources_text += f"{i}. {str(source)}\n"
    return sources_text


async def _consume_stream(process_iter, output_filter: Optional[StreamingBlocklistFilter] = None):
    """
    Shared event consumer for both ChatUI adapters.

    Maps process_query_streaming events to plain text yielded to the client
    Appends the filters footnote + sources on `end` (for Chatui)
    
    Output guard: when `output_filter` is enabled, every token is routed through 
    the streaming blocklist filter. When there is a hit the stream stops, 
    the notice is displayed, and the footnote/sources are suppressed.
    """
    filters_footnote = None
    sources_collected = None
    blocked = False

    async for result in process_iter:
        if not isinstance(result, dict):
            yield str(result)
            await asyncio.sleep(0)
            continue

        result_type = result.get("type", "data")
        content = result.get("content", "")

        if result_type == "data":
            if output_filter is not None:
                emit, hit = output_filter.feed(content)
                if emit:
                    yield emit
                if hit:
                    blocked = True
                    break  # stop streaming the (now-blocked) answer
            else:
                yield content
        elif result_type == "filters_applied":
            filters_footnote = _build_filters_footnote(
                content.get("filters", {}), content.get("narrowed", False)
            )
        elif result_type == "sources":
            sources_collected = content
        elif result_type == "end":
            if output_filter is not None and not blocked:
                tail, hit = output_filter.flush_final()
                if tail:
                    yield tail
                if hit:
                    blocked = True
            if blocked:
                return  # suppress footnote + sources on a blocked answer
            if filters_footnote:
                yield f"\n\n---\n{filters_footnote}"
            if sources_collected:
                logger.info("Sending markdown sources with doc:// scheme")
                yield _render_sources(sources_collected)
        elif result_type == "error":
            yield f"Error: {content}"

        await asyncio.sleep(0)


def _make_output_filter(blocklist, output_notice: str) -> Optional[StreamingBlocklistFilter]:
    """
    Construct a fresh per-request output filter instance, or None when the guard is off.
    """
    if blocklist is None:
        return None
    return StreamingBlocklistFilter(blocklist, output_notice)


async def process_query_streaming(
    compiled_graph,
    query: str,
    file_upload=None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    conversation_context: str = None,
    user_messages_history: str = None,
    file_content: bytes = None,
    filename: str = None
):
    """
    Process a query through the LangGraph workflow with streaming.

    COPIED FROM ORIGINAL ORCHESTRATOR. TO BE REPLACED WITH AGENTIC WORFLOW
    """
    initial_state = {
        "query": query,
        "metadata": {"session_type": "chatui"},
        "raw_documents": [],
        "conversation_context": conversation_context,
        "metadata_filters": metadata_filters,
        "user_messages_history": user_messages_history,
    }

    # Add file content if present
    if file_content and filename:
        initial_state["file_content"] = file_content
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


async def chatui_adapter(data, compiled_graph, max_turns: int = 3, max_chars: int = 8000,
                         blocklist=None, output_notice: str = "[response withheld]"):
    """Text-only adapter for ChatUI with structured message support"""
    logger.debug(f"ChatUI adapter called with data type: {type(data)}")

    try:
        # Handle both dict and object access patterns
        if isinstance(data, dict):
            text_value = data.get('text', '')
            messages_value = data.get('messages', None)
            preprompt_value = data.get('preprompt', None)
        else:
            text_value = getattr(data, 'text', '')
            messages_value = getattr(data, 'messages', None)
            preprompt_value = getattr(data, 'preprompt', None)

        # Convert dict messages to objects if needed
        messages = []
        if messages_value:
            for msg in messages_value:
                if isinstance(msg, dict):
                    messages.append(type('Message', (), {
                        'role': msg.get('role', 'unknown'),
                        'content': msg.get('content', '')
                    })())
                else:
                    messages.append(msg)

        # Extract latest user query
        user_messages = [msg for msg in messages if msg.role == 'user']
        query = user_messages[-1].content if user_messages else text_value

        # Conversation metadata (troubleshooting purposes)
        msg_metadata = {
            'total': len(messages),
            'user': len(user_messages),
            'assistant': len([m for m in messages if m.role == 'assistant']),
            'msg_lengths': [len(m.content) for m in messages]
        }
        logger.info(f"Processing query: {query[:20]}... | Conversation: {msg_metadata}")

        # Build conversation context for generation (last N turns)
        conversation_context = build_conversation_context(messages, max_turns=max_turns, max_chars=max_chars)

        # User-only history for filter extraction (no assistant responses / retrieved doc content)
        user_only = [msg for msg in messages if msg.role == 'user']
        user_messages_history = "\n".join(
            f"USER: {msg.content}" for msg in user_only[-max_turns:]
        ) if user_only else None

        output_filter = _make_output_filter(blocklist, output_notice)
        async for result in _consume_stream(
            process_query_streaming(
                compiled_graph=compiled_graph,
                query=query,
                file_upload=None,
                conversation_context=conversation_context,
                user_messages_history=user_messages_history,
            ),
            output_filter,
        ):
            yield result

    except Exception as e:
        logger.error(f"ChatUI error: {str(e)}")
        logger.error(f"Full traceback:", exc_info=True)
        yield f"Error: {str(e)}"


async def chatui_file_adapter(data, compiled_graph, max_turns: int = 3, max_chars: int = 8000,
                              blocklist=None, output_notice: str = "[response withheld]"):
    """File upload adapter for ChatUI with structured message support"""
    try:
        # Handle both dict and object access patterns
        if isinstance(data, dict):
            text_value = data.get('text', '')
            messages_value = data.get('messages', None)
            files_value = data.get('files', None)
            preprompt_value = data.get('preprompt', None)
        else:
            text_value = getattr(data, 'text', '')
            messages_value = getattr(data, 'messages', None)
            files_value = getattr(data, 'files', None)
            preprompt_value = getattr(data, 'preprompt', None)

        # Extract query - prefer structured messages
        conversation_context = None
        if messages_value and len(messages_value) > 0:
            # Convert dict messages to objects
            messages = []
            for msg in messages_value:
                if isinstance(msg, dict):
                    messages.append(type('Message', (), {
                        'role': msg.get('role', 'unknown'),
                        'content': msg.get('content', '')
                    })())
                else:
                    messages.append(msg)

            user_messages = [msg for msg in messages if msg.role == 'user']
            query = user_messages[-1].content if user_messages else text_value

            # Conversation metadata (troubleshooting purposes)
            msg_metadata = {
                'total': len(messages),
                'user': len(user_messages),
                'assistant': len([m for m in messages if m.role == 'assistant']),
                'msg_lengths': [len(m.content) for m in messages]
            }
            logger.info(f"Processing query with file: {query[:20]}... | Conversation: {msg_metadata}")

            conversation_context = build_conversation_context(messages, max_turns=max_turns, max_chars=max_chars)

            # User-only history for filter extraction (no assistant responses / retrieved doc content)
            user_only = [msg for msg in messages if msg.role == 'user']
            user_messages_history = "\n".join(
                f"USER: {msg.content}" for msg in user_only[-max_turns:]
            ) if user_only else None
        else:
            query = text_value
            user_messages_history = None

        file_content = None
        filename = None

        if files_value and len(files_value) > 0:
            file_info = files_value[0]
            logger.info(f"Processing file: {file_info.get('name', 'unknown')}")

            if file_info.get('type') == 'base64' and file_info.get('content'):
                try:
                    import base64
                    file_content = base64.b64decode(file_info['content'])
                    filename = file_info.get('name', 'uploaded_file')
                except Exception as e:
                    logger.error(f"Error decoding base64 file: {str(e)}")
                    yield f"Error: Failed to decode uploaded file - {str(e)}"
                    return

        output_filter = _make_output_filter(blocklist, output_notice)
        async for result in _consume_stream(
            process_query_streaming(
                compiled_graph=compiled_graph,
                query=query,
                file_upload=None,
                conversation_context=conversation_context,
                user_messages_history=user_messages_history,
                file_content=file_content,
                filename=filename
            ),
            output_filter,
        ):
            yield result

    except Exception as e:
        logger.error(f"ChatUI file adapter error: {str(e)}")
        yield f"Error: {str(e)}"