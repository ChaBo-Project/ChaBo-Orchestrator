"""
Tests for the shared event consumer (`consume_stream` in orchestration/streaming.py).

*No network required (.env / services etc.)

`consume_stream` is the single place the internal event stream is turned into output, for
every frontend: it owns the output guards and delegates formatting to a renderer. These
tests pin the parts that must hold whichever renderer is in use — in particular that a
guard-blocked stream still terminates cleanly on the wire.
"""
import asyncio
import json

from components.guardrails.output_guard import StreamingBlocklistFilter, compile_blocklist
from components.orchestration.renderers import MarkdownRenderer, OpenAIChunkRenderer
from components.orchestration.streaming import consume_stream

SOURCES = [{"title": "Wheat guide", "uri": "https://example.org/wheat"}]
FILTERS = {"filters": {"crop_type": ["wheat"]}, "narrowed": False}


async def _events(*events):
    for event in events:
        yield event


def _answer(*chunks, sources=True, filters=True):
    """The internal event stream for one successful answer."""
    events = [{"type": "data", "content": c} for c in chunks]
    if filters:
        events.append({"type": "filters_applied", "content": FILTERS})
    if sources:
        events.append({"type": "sources", "content": SOURCES})
    events.append({"type": "end", "content": ""})
    return _events(*events)


def _drain(process_iter, output_filter=None, classifier=None, renderer=None):
    async def run():
        return [piece async for piece in consume_stream(
            process_iter, output_filter, classifier, renderer
        )]
    return asyncio.run(run())


def _blocklist_filter(term="badword", notice="[blocked]"):
    return StreamingBlocklistFilter(compile_blocklist({"en": [term]}), notice)


class FlaggingClassifier:
    """Minimal stand-in for StreamingClassifier: flags once `trigger` has been seen."""

    class _Cfg:
        notice = "[withheld]"

    def __init__(self, trigger):
        self.cfg = self._Cfg()
        self.trigger = trigger
        self.seen = ""
        self.closed = False

    def feed(self, chunk):
        flagged = self.trigger in self.seen  # verdicts land a window behind the text
        self.seen += chunk
        return flagged

    async def flush_final(self):
        return self.cfg.notice, self.trigger in self.seen

    async def aclose(self):
        self.closed = True


# --- markdown (LangServe / ChatUI) path ---

def test_answer_text_footnote_and_sources_are_emitted_in_order():
    out = "".join(_drain(_answer("Wheat is sown ", "in November [1]."), renderer=MarkdownRenderer(0)))
    assert out.index("Wheat is sown") < out.index("Searched within") < out.index("**Sources:**")


def test_a_pipeline_error_is_rendered_for_the_user():
    out = "".join(_drain(_events({"type": "error", "content": "boom"}), renderer=MarkdownRenderer(0)))
    assert out == "Error: boom"


# --- output guards ---

def test_a_blocked_term_stops_the_answer_and_suppresses_footnote_and_sources():
    out = "".join(_drain(
        _answer("all fine so far ", "then badword appears"),
        output_filter=_blocklist_filter(),
        renderer=MarkdownRenderer(0),
    ))
    assert "[blocked]" in out
    assert "badword" not in out
    assert "**Sources:**" not in out and "Searched within" not in out


def test_the_blocklist_notice_is_emitted_exactly_once():
    out = "".join(_drain(
        _answer("badword right away", "more text"),
        output_filter=_blocklist_filter(),
        renderer=MarkdownRenderer(0),
    ))
    assert out.count("[blocked]") == 1


def test_a_clean_answer_passes_the_blocklist_untouched():
    out = "".join(_drain(
        _answer("wheat is sown in November [1]."),
        output_filter=_blocklist_filter(),
        renderer=MarkdownRenderer(0),
    ))
    assert "wheat is sown in November [1]." in out
    assert "[blocked]" not in out and "**Sources:**" in out


def test_a_classifier_verdict_truncates_the_stream_and_suppresses_sources():
    out = "".join(_drain(
        _answer("safe opening ", "flagme now ", "tail that should not appear"),
        classifier=FlaggingClassifier("flagme"),
        renderer=MarkdownRenderer(0),
    ))
    assert "[withheld]" in out
    assert "tail that should not appear" not in out
    assert "**Sources:**" not in out


def test_in_flight_classifications_are_cancelled_when_the_stream_ends():
    classifier = FlaggingClassifier("never-appears")
    _drain(_answer("all clean"), classifier=classifier, renderer=MarkdownRenderer(0))
    assert classifier.closed is True


# --- renderer independence: the same guard behaviour on the OpenAI wire ---

def _frames(pieces):
    return [
        json.loads(line[len("data: "):])
        for line in "".join(pieces).splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def test_a_blocked_openai_stream_still_terminates_with_stop_and_done():
    # Without a terminating frame an OpenAI client waits forever on a blocked answer.
    pieces = _drain(
        _answer("fine ", "badword here"),
        output_filter=_blocklist_filter(),
        renderer=OpenAIChunkRenderer("chabo"),
    )
    body = "".join(pieces)
    assert "[blocked]" in body
    assert _frames(pieces)[-1]["choices"][0]["finish_reason"] == "stop"
    assert body.rstrip().endswith("data: [DONE]")


def test_a_blocked_openai_stream_carries_no_citations_frame():
    pieces = _drain(
        _answer("fine ", "badword here"),
        output_filter=_blocklist_filter(),
        renderer=OpenAIChunkRenderer("chabo"),
    )
    assert not any("citations" in frame for frame in _frames(pieces))
