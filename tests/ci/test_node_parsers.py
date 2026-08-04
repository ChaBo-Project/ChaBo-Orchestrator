"""
Tests for parsing and routing helpers in the orchestration nodes.

*No network required (.env / services etc.)

Covers:
- model output string reads and graph state dictionaries.
- filter extraction parsing (type casting, unknown fields, uncastable
values, code fences, empty result)
- query-rewrite parsing
- guard router
"""

from components.orchestration.nodes import (
    _parse_filter_response,
    _parse_rewrite_response,
    _route_after_guard,
)

FIELDS = {"crop_type": "list", "year": "int", "title": "str"}


# --- _parse_filter_response: reading the filter-extraction answer ---

def test_reads_and_casts_each_field_to_its_declared_type():
    raw = '{"crop_type": "wheat", "year": "2020", "title": "Guide"}'
    assert _parse_filter_response(raw, FIELDS) == {
        "crop_type": ["wheat"],
        "year": 2020,
        "title": "Guide",
    }


def test_a_list_field_keeps_all_values():
    assert _parse_filter_response('{"crop_type": ["wheat", "maize"]}', FIELDS) == {
        "crop_type": ["wheat", "maize"]
    }


def test_fields_not_in_the_allowed_set_are_ignored():
    assert _parse_filter_response('{"crop_type": "wheat", "unknown": "x"}', FIELDS) == {
        "crop_type": ["wheat"]
    }


def test_a_value_that_cannot_be_cast_is_dropped():
    # "year" expects an integer; a non-numeric value is dropped, not fatal.
    assert _parse_filter_response('{"year": "not-a-number", "title": "Guide"}', FIELDS) == {
        "title": "Guide"
    }


def test_json_wrapped_in_code_fences_is_still_read():
    assert _parse_filter_response('```json\n{"title": "Guide"}\n```', FIELDS) == {"title": "Guide"}


def test_no_usable_filters_returns_none():
    assert _parse_filter_response('{"unknown": "x"}', FIELDS) is None


def test_invalid_json_returns_none():
    assert _parse_filter_response("not json", FIELDS) is None


# --- _parse_rewrite_response: reading the query-rewrite answer ---

def test_reads_the_rewritten_query_and_notes():
    raw = '{"query_rewrite": "wheat lodging", "notes": {"detected_language": "en"}}'
    result = _parse_rewrite_response(raw)
    assert result["query_rewrite"] == "wheat lodging"
    assert result["notes"] == {"detected_language": "en"}


def test_missing_notes_defaults_to_an_empty_dict():
    assert _parse_rewrite_response('{"query_rewrite": "wheat lodging"}')["notes"] == {}


def test_an_empty_rewrite_returns_none():
    assert _parse_rewrite_response('{"query_rewrite": "   "}') is None


def test_rewrite_invalid_json_returns_none():
    assert _parse_rewrite_response("not json") is None


# --- _route_after_guard: deciding where to go after the input guard ---

def test_a_blocked_input_is_routed_to_the_blocked_branch():
    assert _route_after_guard({"guard_blocked": True}) == "blocked"


def test_a_safe_input_continues():
    assert _route_after_guard({"guard_blocked": False}) == "continue"
    assert _route_after_guard({}) == "continue"
