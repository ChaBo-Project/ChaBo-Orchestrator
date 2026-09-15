"""
Tests for the frontend renderers (components/orchestration/renderers.py).

*No network required (.env / services etc.)

Covers:
- markdown rendering of sources, with and without a real link
- the Chabo-ChatUI's placeholder link scheme, and its absence everywhere else
- the filters footnote
- OpenAI chat.completion.chunk framing, and the citations array
"""
import json
import re

from components.orchestration.renderers import (
    CHATUI_PLACEHOLDER_URI,
    MarkdownRenderer,
    OpenAIChunkRenderer,
    citation_list,
    format_filters_footnote,
    format_sources_markdown,
)
from components.orchestration.ui_adapters import _chatui_renderer


#: Copied verbatim from the Chabo-ChatUI's langserve-streaming endpoint (`sourceRegex`). If
#: the fork's parser changes, change this and watch which of our renderers stop matching.
CHATUI_SOURCE_REGEX = re.compile(r"\[([^\]]+)\]\(((?:doc|https?)://[^)]+)\)")


def _frames(raw: str):
    """Parse the JSON payload of every `data:` frame in an SSE string, skipping [DONE]."""
    out = []
    for line in raw.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            out.append(json.loads(line[len("data: "):]))
    return out


# --- markdown source rendering ---

def test_source_with_a_link_is_rendered_as_a_markdown_link():
    text = format_sources_markdown([{"title": "Report", "uri": "https://example.org/a"}])
    assert "1. [Report](https://example.org/a)" in text


def test_source_without_a_link_is_rendered_as_plain_text_by_default():
    # The core no longer invents a placeholder (create_sources_list reports uri=None); a
    # renderer that wants one asks for it — see the ChatUI cases below.
    text = format_sources_markdown([{"title": "Report", "uri": None}])
    assert "1. Report" in text
    assert "doc://" not in text
    assert "(" not in text


def test_a_placeholder_uri_is_substituted_for_a_source_with_no_link():
    # Our Chabo-ChatUI only renders a citation whose URL matches doc:// | http:// | https://,
    # so on that path an absent link must still produce a scheme-carrying markdown link.
    text = format_sources_markdown([{"title": "Report", "uri": None}], CHATUI_PLACEHOLDER_URI)
    assert "1. [Report](doc://#)" in text


def test_a_placeholder_never_overrides_a_real_link():
    text = format_sources_markdown(
        [{"title": "Report", "uri": "https://example.org/a"}], CHATUI_PLACEHOLDER_URI
    )
    assert "1. [Report](https://example.org/a)" in text
    assert "doc://" not in text


def test_the_chatui_renderer_is_the_one_that_carries_the_placeholder():
    # Guards the wiring, not just the helper: the LangServe/ChatUI path must construct the
    # renderer with the placeholder, and every other frontend must not inherit it.
    assert CHATUI_PLACEHOLDER_URI in _chatui_renderer().sources([{"title": "A", "uri": None}])
    assert "doc://" not in MarkdownRenderer(0).sources([{"title": "A", "uri": None}])


def test_the_chatui_source_block_parses_under_the_forks_own_regex():
    """
    Pin our wire format against the consumer that has to read it.

    The fork's langserve-streaming endpoint finds sources by running CHATUI_SOURCE_REGEX over
    any streamed chunk containing "**Sources:**", and *also* truncates the displayed text at
    that marker. So a source whose link doesn't carry one of those schemes isn't merely
    unlinked — it is dropped from the sources panel AND from the message body. Both halves
    have to hold in the same single chunk.
    """
    block = _chatui_renderer().sources([
        {"title": "Report", "uri": None},
        {"title": "Guide", "uri": "https://example.org/g"},
    ])
    assert "**Sources:**" in block
    assert CHATUI_SOURCE_REGEX.findall(block) == [
        ("Report", "doc://#"),
        ("Guide", "https://example.org/g"),
    ]


def test_sources_are_numbered_in_citation_order():
    text = format_sources_markdown([{"title": "A", "uri": None}, {"title": "B", "uri": None}])
    assert "1. A" in text and "2. B" in text


def test_citation_list_falls_back_to_the_title_when_there_is_no_uri():
    assert citation_list([{"title": "A", "uri": None}, {"title": "B", "uri": "u"}]) == ["A", "u"]


# --- filters footnote ---

def test_footnote_lists_each_applied_filter():
    note = format_filters_footnote({"crop_type": ["wheat", "maize"], "year": 2024}, False)
    assert "crop_type: wheat, maize" in note and "year: 2024" in note


def test_footnote_flags_a_narrowed_filter():
    assert "narrowed" in format_filters_footnote({"crop_type": ["wheat"]}, True)


# --- MarkdownRenderer ---

def test_markdown_renderer_passes_answer_text_through_unchanged():
    assert MarkdownRenderer(0).text("hello") == "hello"


def test_markdown_renderer_records_citations_for_non_streaming_callers():
    renderer = MarkdownRenderer(0)
    renderer.sources([{"title": "A", "uri": "https://x"}])
    assert renderer.citations == ["https://x"]


def test_markdown_renderer_emits_no_prelude_or_finish_frames():
    renderer = MarkdownRenderer(0)
    assert renderer.prelude() is None
    assert renderer.finish() is None


# --- OpenAIChunkRenderer ---

def test_openai_prelude_opens_the_stream_with_the_assistant_role():
    renderer = OpenAIChunkRenderer("chabo")
    frame = _frames(renderer.prelude())[0]
    assert frame["object"] == "chat.completion.chunk"
    assert frame["choices"][0]["delta"]["role"] == "assistant"


def test_openai_text_chunk_carries_the_content_delta():
    renderer = OpenAIChunkRenderer("my-model")
    frame = _frames(renderer.text("hello"))[0]
    assert frame["model"] == "my-model"
    assert frame["choices"][0]["delta"]["content"] == "hello"
    assert frame["choices"][0]["finish_reason"] is None


def test_every_frame_of_one_response_shares_the_same_id_and_created():
    renderer = OpenAIChunkRenderer("chabo")
    frames = _frames(renderer.prelude() + renderer.text("a") + renderer.text("b"))
    assert len({f["id"] for f in frames}) == 1
    assert len({f["created"] for f in frames}) == 1


def test_openai_finish_sends_a_stop_frame_then_the_done_sentinel():
    raw = OpenAIChunkRenderer("chabo").finish()
    assert _frames(raw)[0]["choices"][0]["finish_reason"] == "stop"
    assert raw.endswith("data: [DONE]\n\n")


def test_openai_sources_frame_carries_markdown_and_the_citations_array():
    renderer = OpenAIChunkRenderer("chabo", include_citations=True)
    frame = _frames(renderer.sources([{"title": "A", "uri": "https://x"}]))[0]
    assert "**Sources:**" in frame["choices"][0]["delta"]["content"]
    assert frame["citations"] == ["https://x"]


def test_citations_array_is_omitted_when_the_enhancement_is_off():
    renderer = OpenAIChunkRenderer("chabo", include_citations=False)
    frame = _frames(renderer.sources([{"title": "A", "uri": "https://x"}]))[0]
    # The markdown source list is the universal path and stays either way.
    assert "**Sources:**" in frame["choices"][0]["delta"]["content"]
    assert "citations" not in frame


def test_openai_frames_are_self_delimiting_so_no_flush_delay_is_needed():
    assert OpenAIChunkRenderer("chabo").trailing_flush_delay == 0


def test_openai_errors_are_surfaced_as_message_content():
    frame = _frames(OpenAIChunkRenderer("chabo").error("boom"))[0]
    assert "boom" in frame["choices"][0]["delta"]["content"]


def test_non_ascii_answer_text_survives_json_framing():
    frame = _frames(OpenAIChunkRenderer("chabo").text("زراعة القمح"))[0]
    assert frame["choices"][0]["delta"]["content"] == "زراعة القمح"
