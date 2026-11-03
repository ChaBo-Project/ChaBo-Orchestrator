"""
Tests for the citation and source helpers.

*No network required (.env / services etc.)

Covers:
- finding citation numbers
- picking out cited sources
- normalising the various citation styles into plain bracket numbers
"""

from components.generator.sources import (
    parse_citations,
    extract_sources,
    clean_citations,
)


# --- parse_citations: finding the cited numbers in the answer ---

def test_finds_single_and_grouped_citation_numbers():
    assert parse_citations("As shown [1] and also [2,3].") == [1, 2, 3]


def test_removes_duplicates_and_sorts_the_numbers():
    assert parse_citations("[3] text [1] more [3]") == [1, 3]


def test_text_without_citations_returns_an_empty_list():
    assert parse_citations("no citations here") == []


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

def test_preserves_source_positions_when_citations_have_gaps():
    results = [{"title": "A"}, {"title": "B"}, {"title": "C"}]

    sources = extract_sources(results, [1, 3], preserve_positions=True)

    assert [source["title"] for source in sources] == ["A", "B", "C"]
    assert [source["_citation_number"] for source in sources] == [1, 2, 3]
    assert all("_citation_number" not in result for result in results)


def test_preserves_source_positions_when_only_third_source_is_cited():
    results = [{"title": "A"}, {"title": "B"}, {"title": "C"}]

    sources = extract_sources(results, [3], preserve_positions=True)

    assert [source["title"] for source in sources] == ["A", "B", "C"]


def test_preserving_positions_stops_at_highest_valid_citation():
    results = [{"title": "A"}, {"title": "B"}, {"title": "C"}]

    sources = extract_sources(results, [1, 99], preserve_positions=True)

    assert [source["title"] for source in sources] == ["A"]


def test_preserving_positions_without_valid_citations_returns_no_sources():
    results = [{"title": "A"}]

    assert extract_sources(results, [], preserve_positions=True) == []
    assert extract_sources(results, [0, 99], preserve_positions=True) == []
