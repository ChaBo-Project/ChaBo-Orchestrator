"""
Frontend renderers: turn ChaBo's internal event stream into a wire format.

The graph emits a small typed event stream (data / filters_applied / sources / error / end).
`consume_stream` (streaming.py) is the single consumer of that stream and owns the output
guards; a *renderer* owns the separate question of what those semantic outputs look like on
the wire. Adding a frontend is therefore a new renderer, not a fork of the streaming logic.

Two renderers ship today:

  MarkdownRenderer     plain markdown text, one string per event. Used by the LangServe
                       routes (ChatUI) and, with `trailing_flush_delay=0`, to build the
                       message body of a non-streaming OpenAI response.
  OpenAIChunkRenderer  `chat.completion.chunk` SSE frames for /v1/chat/completions.

Both render sources the same way — inline `[N]` markers in the answer plus a markdown source
list — because that is the only citation rendering that works in every frontend. Anything
richer (the `citations` array below) is layered on top as a best-effort enhancement.

Renderer methods return the wire text to emit, or None for "nothing to emit". A renderer is
per-request: it carries stream identity (OpenAI id/created) and collects `citations`.
"""
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ChatUI's langserve-streaming parser reads the event type from the chunk prefix, so the
# trailing events (answer tail / footnote / sources) must not coalesce into one SSE chunk —
# this sleep forces them apart. It is a transport workaround for the plain-string LangServe
# stream, NOT a property of the pipeline: renderers with self-delimiting frames (see
# OpenAIChunkRenderer) set the delay to 0.
_TRAILING_FLUSH_DELAY = float(os.getenv("TRAILING_FLUSH_DELAY", "0.05"))

# Our Chabo-ChatUI validates source link schemes and only accepts doc://, http://, https://,
# so a source with no URL has to be given a placeholder scheme there or its citation does not
# render. That is a property of THAT frontend, so it lives here rather than in
# generator/sources.py (which reports `uri: None` and lets each renderer decide). Renderers
# serving other frontends leave `placeholder_uri` unset and print an unlinked title.
CHATUI_PLACEHOLDER_URI = "doc://#"


def format_filters_footnote(filters: Dict[str, Any], narrowed: bool) -> str:
    """Build a subtle italic footnote showing which filters were applied during retrieval."""
    parts = [
        f"{k}: {', '.join(v) if isinstance(v, list) else v}"
        for k, v in filters.items()
    ]
    base = "🔍 Searched within: " + " · ".join(parts)
    if narrowed:
        base += " (narrowed — combined filter returned no results)"
    return "*" + base + "*"


def format_sources_markdown(sources_collected, placeholder_uri: Optional[str] = None) -> str:
    """
    Render collected sources as a markdown list.

    `create_sources_list()` reports `uri=None` for a source with no link, and what that looks
    like on the wire is the renderer's call: pass `placeholder_uri` for a frontend that only
    renders a citation when the link carries an accepted scheme (our Chabo-ChatUI — see
    CHATUI_PLACEHOLDER_URI), leave it unset to print an unlinked title.
    """
    sources_text = "\n\n**Sources:**\n"
    for i, source in enumerate(sources_collected, 1):
        if isinstance(source, dict):
            title = source.get("title", "Unknown")
            uri = source.get("uri") or placeholder_uri
            sources_text += f"{i}. [{title}]({uri})\n" if uri else f"{i}. {title}\n"
        else:
            sources_text += f"{i}. {str(source)}\n"
    return sources_text


def citation_list(sources_collected) -> List[str]:
    """
    Flatten sources to the string list used by the non-standard `citations` field.

    Perplexity-style `citations` is an array of strings and is what OpenWebUI's ad hoc
    support reads. Sources without a URI contribute their title so the array stays aligned
    positionally with the `[N]` markers in the answer.
    """
    citations = []
    for source in sources_collected or []:
        if isinstance(source, dict):
            citations.append(source.get("uri") or source.get("title", "Unknown"))
        else:
            citations.append(str(source))
    return citations


class BaseRenderer:
    """
    Interface `consume_stream` renders through.

    Each hook returns the text to put on the wire, or None to emit nothing. Renderers are
    constructed per request.
    """

    #: Seconds to sleep around trailing events so a fragile client parser sees them apart.
    trailing_flush_delay: float = 0.0

    def __init__(self) -> None:
        #: Sources seen on this stream, in citation order (read by non-streaming callers).
        self.citations: List[str] = []

    def prelude(self) -> Optional[str]:
        """Emitted once before any answer text."""
        return None

    def text(self, chunk: str) -> Optional[str]:
        """A chunk of generated answer text."""
        raise NotImplementedError

    def notice(self, message: str) -> Optional[str]:
        """A guardrail notice replacing/terminating the answer."""
        raise NotImplementedError

    def footnote(self, filters: Dict[str, Any], narrowed: bool) -> Optional[str]:
        """The retrieval-filters footnote, appended after the answer."""
        raise NotImplementedError

    def sources(self, sources_collected) -> Optional[str]:
        """The cited-sources block, appended last."""
        raise NotImplementedError

    def error(self, message: str) -> Optional[str]:
        """A pipeline error surfaced to the user."""
        raise NotImplementedError

    def finish(self) -> Optional[str]:
        """Emitted once after everything else, including on a guard-blocked stream."""
        return None


class MarkdownRenderer(BaseRenderer):
    """
    Plain markdown text.

    Used by the LangServe routes (`output_type=str`, where it is constructed with the ChatUI
    fork's `placeholder_uri`) and reused as the body builder for non-streaming OpenAI
    responses (where it is not). The frontend-specific part is those two constructor
    arguments — the rendering itself is shared.
    """

    def __init__(
        self,
        trailing_flush_delay: Optional[float] = None,
        placeholder_uri: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.trailing_flush_delay = (
            _TRAILING_FLUSH_DELAY if trailing_flush_delay is None else trailing_flush_delay
        )
        #: Link substituted for a source with no URL; None renders an unlinked title.
        self.placeholder_uri = placeholder_uri

    def text(self, chunk: str) -> Optional[str]:
        return chunk or None

    def notice(self, message: str) -> Optional[str]:
        return message

    def footnote(self, filters: Dict[str, Any], narrowed: bool) -> Optional[str]:
        return "\n\n---\n" + format_filters_footnote(filters, narrowed)

    def sources(self, sources_collected) -> Optional[str]:
        self.citations = citation_list(sources_collected)
        return format_sources_markdown(sources_collected, self.placeholder_uri)

    def error(self, message: str) -> Optional[str]:
        return f"Error: {message}"


class OpenAIChunkRenderer(BaseRenderer):
    """
    OpenAI-compatible `chat.completion.chunk` SSE frames for /v1/chat/completions.

    Frames are self-delimiting, so no flush delay is needed. Sources are rendered into the
    message body as markdown (the universal path) and, when `include_citations` is set, are
    additionally attached to that frame as a top-level `citations` array — a non-standard
    field some UIs read, ignored by everything else. Never load-bearing.
    """

    trailing_flush_delay = 0.0

    def __init__(
        self,
        model: str,
        request_id: Optional[str] = None,
        include_citations: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.id = request_id or f"chatcmpl-{uuid.uuid4().hex}"
        self.created = int(time.time())
        self.include_citations = include_citations

    def _frame(
        self,
        delta: Dict[str, Any],
        finish_reason: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        chunk = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if extra:
            chunk.update(extra)
        return "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"

    def prelude(self) -> Optional[str]:
        return self._frame({"role": "assistant", "content": ""})

    def text(self, chunk: str) -> Optional[str]:
        return self._frame({"content": chunk}) if chunk else None

    def notice(self, message: str) -> Optional[str]:
        return self._frame({"content": message})

    def footnote(self, filters: Dict[str, Any], narrowed: bool) -> Optional[str]:
        return self._frame({"content": "\n\n---\n" + format_filters_footnote(filters, narrowed)})

    def sources(self, sources_collected) -> Optional[str]:
        self.citations = citation_list(sources_collected)
        extra = {"citations": self.citations} if self.include_citations else None
        return self._frame({"content": format_sources_markdown(sources_collected)}, extra=extra)

    def error(self, message: str) -> Optional[str]:
        # Surfaced as message content so it is visible in any client, rather than as a
        # transport-level error the UI may swallow.
        return self._frame({"content": f"\n\nError: {message}"})

    def finish(self) -> Optional[str]:
        return self._frame({}, finish_reason="stop") + "data: [DONE]\n\n"
