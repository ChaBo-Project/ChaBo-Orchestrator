import pytest

from components.generator.sources import (
    create_sources_list,
    extract_sources,
    parse_citations,
)
from components.orchestration.renderers import (
    CHATUI_PLACEHOLDER_URI,
    MarkdownRenderer,
    citation_list,
    format_sources_markdown,
)


@pytest.mark.parametrize(
    "answer, expected_count",
    [
        ("Grant [1].", 1),
        ("Grant [1]. Induction [3].", 3),
        ("Induction [3].", 3),
        ("Both documents [1, 3].", 3),
        ("No citations.", 0),
    ],
)
@pytest.mark.parametrize("with_links", [False, True])
def test_answer_citations_match_source_positions(answer, expected_count, with_links):
    filenames = ["grant.txt", "unused.txt", "induction.txt"]
    documents = [
        {
            "__all_metadata__": {
                "filename": filename,
                "url": f"https://example.org/{filename}" if with_links else None,
            }
        }
        for filename in filenames
    ]

    numbers = parse_citations(answer)
    selected = extract_sources(documents, numbers, preserve_positions=True)
    sources = create_sources_list(selected, ["filename"], "url")

    assert [source["title"] for source in sources] == filenames[:expected_count]

    citations = citation_list(sources)
    assert len(citations) == expected_count

    markdown = format_sources_markdown(sources)
    chatui = MarkdownRenderer(0, placeholder_uri=CHATUI_PLACEHOLDER_URI)
    chatui_block = chatui.sources(sources)

    assert chatui.citations == citations

    for number in numbers:
        filename = filenames[number - 1]
        expected_uri = f"https://example.org/{filename}" if with_links else None

        assert sources[number - 1] == {"title": filename, "uri": expected_uri}
        assert citations[number - 1] == (expected_uri or filename)

        if with_links:
            assert f"{number}. [{filename}]({expected_uri})" in markdown
        else:
            assert f"{number}. {filename}\n" in markdown

        chatui_uri = expected_uri or CHATUI_PLACEHOLDER_URI
        assert f"{number}. [{filename}]({chatui_uri})" in chatui_block
