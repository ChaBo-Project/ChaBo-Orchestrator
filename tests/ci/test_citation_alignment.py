"""
End-to-end alignment between the `[n]` citations in an answer and the source list.

*No network required (.env / services etc.)

Every citation number in the answer must reach the client attached to the document it
refers to, whatever gaps there are between the cited numbers (issue #63).
"""
import pytest

from components.generator.sources import (
    create_sources_list,
    extract_sources,
    parse_citations,
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
