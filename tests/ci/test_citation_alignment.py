"""
End-to-end alignment between the `[n]` citations in an answer and the source list.

*No network required (.env / services etc.)

Every citation number in the answer must reach the client attached to the document it
refers to, whatever gaps there are between the cited numbers (issue #63).
"""
import re

import pytest

from components.generator.sources import (
    create_sources_list,
    extract_sources,
    parse_citations,
)
from components.orchestration.renderers import (
    CHATUI_PLACEHOLDER_URI,
    citation_list,
    format_sources_markdown,
)

FILENAMES = ["grant.txt", "unused.txt", "induction.txt"]

CASES = [
    ("Grant [1].", [1]),
    ("Grant [1]. Induction [3].", [1, 3]),
    ("Induction [3].", [3]),
    ("Both documents [1, 3].", [1, 3]),
    ("No citations.", []),
]


def build_sources(answer, with_links):
    documents = [
        {
            "__all_metadata__": {
                "filename": filename,
                "url": f"https://example.org/{filename}" if with_links else None,
            }
        }
        for filename in FILENAMES
    ]
    selected = extract_sources(documents, parse_citations(answer))
    return create_sources_list(selected, ["filename"], "url")


@pytest.mark.parametrize("answer, cited", CASES)
@pytest.mark.parametrize("with_links", [False, True])
def test_each_citation_number_maps_to_its_own_document(answer, cited, with_links):
    sources = build_sources(answer, with_links)

    assert [source["number"] for source in sources] == cited
    for source in sources:
        filename = FILENAMES[source["number"] - 1]
        assert source["title"] == filename
        assert source["uri"] == (f"https://example.org/{filename}" if with_links else None)


@pytest.mark.parametrize("answer, cited", CASES)
@pytest.mark.parametrize("with_links", [False, True])
def test_markdown_source_lines_use_the_citation_numbers(answer, cited, with_links):
    markdown = format_sources_markdown(build_sources(answer, with_links))

    listed = {
        int(number): entry
        for number, entry in re.findall(r"(?m)^- \[(\d+)\] (.+)$", markdown)
    }
    assert sorted(listed) == cited
    for number, entry in listed.items():
        assert FILENAMES[number - 1] in entry


def chatui_chip_labels(answer, block):
    """
    Port of how ChaBo-ChatUI (up to 0.9.3) labels source chips: its langserve-streaming
    endpoint keeps only the [title](uri) pairs from the Sources block, then ChatMessage.svelte
    labels chip i with the i-th sorted single [n] found in the answer text.
    """
    titles = [t for t, _ in re.findall(r"\[([^\]]+)\]\(((?:doc|https?)://[^)]+)\)", block)]
    unique = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    return [
        (unique[i] if i < len(unique) and unique[i] else i + 1, title)
        for i, title in enumerate(titles)
    ]


@pytest.mark.parametrize("answer", ["Grant [1].", "Grant [1]. Induction [3].", "Induction [3]."])
def test_existing_chatui_still_labels_chips_correctly(answer):
    # The cited-only list keeps working with ChatUI releases that pair sources with the
    # cited numbers instead of reading the number from each line.
    block = format_sources_markdown(build_sources(answer, False), CHATUI_PLACEHOLDER_URI)

    for label, title in chatui_chip_labels(answer, block):
        assert title == FILENAMES[label - 1]


@pytest.mark.parametrize("answer, cited", CASES)
@pytest.mark.parametrize("with_links", [False, True])
def test_citations_array_resolves_each_number_to_its_document(answer, cited, with_links):
    citations = citation_list(build_sources(answer, with_links))

    assert len(citations) == (max(cited) if cited else 0)
    for number in cited:
        filename = FILENAMES[number - 1]
        assert citations[number - 1] == (f"https://example.org/{filename}" if with_links else filename)
