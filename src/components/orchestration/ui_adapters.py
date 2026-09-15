"""
Frontend adapter for the LangGraph workflow stream.

`chatui_adapter` unpacks an incoming LangServe request into (query, conversation
context, history, optional file), and drives the shared streaming pipeline
(`process_query_streaming` + `consume_stream` in `streaming.py`) through a Chabo-ChatUI customized
`MarkdownRenderer` (`_chatui_renderer`). One function serves both LangServe routes
(`/chatfed-ui-stream`, `/chatfed-with-file-stream` in main.py) — file handling is a no-op when
the request carries no `files`, so there is no separate text-only variant.

- the OpenAI-compatible routes in components/api reuse the same two layers with a different renderer
"""
import logging
from typing import Optional

from .renderers import CHATUI_PLACEHOLDER_URI, MarkdownRenderer
from .streaming import (
    consume_stream,
    decode_base64_file,
    make_output_classifier,
    make_output_filter,
    prepare_conversation,
    process_query_streaming,
)
from components.guardrails.output_classification import OutputClassificationConfig

logger = logging.getLogger(__name__)


def _field(data, name: str, default=None):
    """Read one field from a LangServe payload, whether it's a dict or a plain object."""
    return data.get(name, default) if isinstance(data, dict) else getattr(data, name, default)


def _chatui_renderer() -> MarkdownRenderer:
    """
    The markdown renderer as per Chabo-ChatUI.

    Two frontend-specific settings:
     1. Trailing flush delay (langserve-streaming parser drops coalesced trailing chunks)
     2. Placeholder links (only renders a citation whose URL matches doc:// | http:// | https://, so a
    source with no URL needs a prefix).
    """
    return MarkdownRenderer(placeholder_uri=CHATUI_PLACEHOLDER_URI)


async def chatui_adapter(data, compiled_graph, max_turns: int = 3, max_chars: int = 8000,
                                blocklist=None, blocklist_notice: str = "[response withheld]",
                                classification_config: Optional[OutputClassificationConfig] = None):
    """
    ChatUI adapter with structured message support.

    Serves both `/chatfed-ui-stream` and `/chatfed-with-file-stream`: a request with no `files`
    behaves exactly like the old text-only adapter, since the file-decoding step is skipped.
    """
    logger.debug(f"ChatUI adapter called with data type: {type(data)}")

    try:
        text_value = _field(data, 'text', '')
        messages_value = _field(data, 'messages')
        files_value = _field(data, 'files')

        query, conversation_context, user_messages_history = prepare_conversation(
            messages_value, fallback_text=text_value, max_turns=max_turns, max_chars=max_chars
        )

        file_content = None
        filename = None

        if files_value and len(files_value) > 0:
            file_info = files_value[0]
            logger.info(f"Processing file: {file_info.get('name', 'unknown')}")
            try:
                file_content, filename = decode_base64_file(file_info)
            except ValueError as e:
                logger.error(str(e))
                yield f"Error: {str(e)}"
                return

        output_filter = make_output_filter(blocklist, blocklist_notice)
        classifier = make_output_classifier(classification_config)
        async for result in consume_stream(
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
            classifier,
            _chatui_renderer(),
        ):
            yield result

    except Exception as e:
        logger.error(f"ChatUI adapter error: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        yield f"Error: {str(e)}"
