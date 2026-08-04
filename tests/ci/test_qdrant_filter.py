"""
Tests for building a Qdrant metadata search filter and a result Document.

*No network required (.env / services etc.)

Covers how a plain filter dictionary is turned into the
Qdrant filter structure, and how one search result becomes a Document.
"""

from components.retriever import retriever_orchestrator as ro

_build_qdrant_filter = ro._build_qdrant_filter
_make_document = ro._make_document
rest = ro.rest


# --- _build_qdrant_filter: dict of filters to a Qdrant filter ---

def test_no_filters_returns_none():
    assert _build_qdrant_filter({}) is None
    assert _build_qdrant_filter(None) is None


def test_a_scalar_value_becomes_an_exact_match():
    built = _build_qdrant_filter({"title": "Guide"})
    condition = built.must[0]
    assert condition.key == "metadata.title"
    assert isinstance(condition.match, rest.MatchValue)
    assert condition.match.value == "Guide"


def test_a_list_value_becomes_a_match_any():
    built = _build_qdrant_filter({"crop_type": ["wheat", "maize"]})
    condition = built.must[0]
    assert condition.key == "metadata.crop_type"
    assert isinstance(condition.match, rest.MatchAny)
    assert condition.match.any == ["wheat", "maize"]


def test_several_fields_are_combined_together():
    built = _build_qdrant_filter({"title": "Guide", "crop_type": ["wheat"]})
    assert {c.key for c in built.must} == {"metadata.title", "metadata.crop_type"}


# --- _make_document: one search result to a Document ---

def test_builds_a_document_and_records_the_scores():
    candidate = {"answer": "some text", "answer_metadata": {"title": "A"}, "score": 0.42}
    doc = _make_document(candidate, rerank_score=0.9)
    assert doc.page_content == "some text"
    assert doc.metadata["title"] == "A"
    assert doc.metadata["retriever_score"] == 0.42
    assert doc.metadata["rerank_score"] == 0.9


def test_falls_back_to_alternate_field_names():
    candidate = {"page_content": "text", "metadata": {"x": 1}, "score": 0.1}
    doc = _make_document(candidate, rerank_score=None)
    assert doc.page_content == "text"
    assert doc.metadata["x"] == 1
    assert doc.metadata["rerank_score"] is None
