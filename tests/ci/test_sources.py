"""
Tests for the citation and source helpers.

*No network required (.env / services etc.)

Covers:
- finding citation numbers
- picking out cited sources and keeping their citation numbers
- normalising the various citation styles into plain bracket numbers
"""

from components.generator.sources import (
    parse_citations,
    extract_sources,
    create_sources_list,
    clean_citations,
)


# --- parse_citations: finding the cited numbers in the answer ---

def test_finds_single_and_grouped_citation_numbers():
    assert parse_citations("As shown [1] and also [2,3].") == [1, 2, 3]


def test_removes_duplicates_and_sorts_the_numbers():
    assert parse_citations("[3] text [1] more [3]") == [1, 3]


def test_text_without_citations_returns_an_empty_list():
    assert parse_citations("no citations here") == []


def test_expands_citation_ranges():
    assert parse_citations("See [1-3] and [5–6].") == [1, 2, 3, 5, 6]


def test_expands_ranges_inside_grouped_citations():
    assert parse_citations("See [1, 4-5].") == [1, 4, 5]


def test_ignores_reversed_and_overly_wide_ranges():
    assert parse_citations("Between [5-2] and [1900-2024].") == []


def test_unmatched_citations_are_logged(caplog):
    with caplog.at_level("WARNING"):
        extract_sources([{"title": "A"}], [1, 7])
    assert "[7]" in caplog.text


# --- extract_sources: keeping only the sources that were cited ---

def test_returns_only_the_cited_sources_with_their_number():
    results = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
    sources = extract_sources(results, [1, 3])
    assert [s["title"] for s in sources] == ["A", "C"]
    assert sources[0]["_citation_number"] == 1


def test_out_of_range_citation_numbers_are_skipped():
    assert extract_sources([{"title": "A"}], [5]) == []


def test_no_citations_returns_an_empty_list():
    assert extract_sources([{"title": "A"}], []) == []


# --- clean_citations: normalising citation styles to [x] ---

def test_normalises_document_style_references_to_bracket_numbers():
    assert clean_citations("(Document 2)") == "[2]"


def test_collapses_doubled_brackets():
    assert clean_citations("text [[1]] here") == "text [1] here"


def test_gapped_citations_keep_their_own_numbers():
    results = [{"title": "A"}, {"title": "B"}, {"title": "C"}]

    sources = extract_sources(results, [3, 1])

    assert [(s["_citation_number"], s["title"]) for s in sources] == [(1, "A"), (3, "C")]
    assert all("_citation_number" not in result for result in results)


def test_only_valid_citation_numbers_are_kept():
    results = [{"title": "A"}, {"title": "B"}]

    assert [s["_citation_number"] for s in extract_sources(results, [0, 2, 99])] == [2]


# --- create_sources_list: the per-source entries sent to every client ---

def test_source_entries_carry_their_citation_number():
    cited = [
        {"__all_metadata__": {"filename": "a.pdf", "url": "https://example.org/a"}, "_citation_number": 2},
        {"__all_metadata__": {"filename": "b.pdf"}, "_citation_number": 4},
    ]

    assert create_sources_list(cited, ["filename"], "url") == [
        {"number": 2, "uri": "https://example.org/a", "title": "a.pdf"},
        {"number": 4, "uri": None, "title": "b.pdf"},
    ]


def test_untitled_source_falls_back_to_its_number():
    cited = [{"__all_metadata__": {}, "_citation_number": 3}]

    assert create_sources_list(cited, ["filename"], "url")[0]["title"] == "Source 3"
