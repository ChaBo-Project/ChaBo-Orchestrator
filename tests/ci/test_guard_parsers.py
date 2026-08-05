"""
Tests for the guardrail helper functions that read a safety model's answer.

*No network required (.env / services etc.)

Covers:
- block decisions
- two answer readers (JSON and Qwen text)
- backend configuration checks.
"""

import json

import pytest

from components.guardrails.llm_guard import (
    decide_safe,
    parse_llm_guard_verdict,
    parse_qwen_verdict,
    build_guard_backend,
)


# --- decide_safe: which severities lead to a blocked answer ---

def test_unsafe_is_always_blocked():
    assert decide_safe("unsafe", block_controversial=False) is False


def test_safe_is_allowed():
    assert decide_safe("safe", block_controversial=False) is True


def test_controversial_is_blocked_only_when_configured():
    assert decide_safe("controversial", block_controversial=False) is True
    assert decide_safe("controversial", block_controversial=True) is False


# --- parse_llm_guard_verdict: reading the JSON answer from the guard model ---

def test_parses_a_plain_safe_verdict():
    verdict = parse_llm_guard_verdict('{"verdict": "safe", "category": "None"}', block_controversial=True)
    assert verdict.safe is True
    assert verdict.severity == "safe"


def test_parses_an_unsafe_verdict_and_blocks():
    verdict = parse_llm_guard_verdict('{"verdict": "unsafe", "category": "Violent"}', block_controversial=False)
    assert verdict.safe is False
    assert verdict.category == "Violent"


def test_ignores_markdown_code_fences_around_the_json():
    raw = '```json\n{"verdict": "safe"}\n```'
    verdict = parse_llm_guard_verdict(raw, block_controversial=True)
    assert verdict.severity == "safe"


def test_reads_the_optional_guideline_fields_when_present():
    raw = '{"verdict": "safe", "guideline_compliant": false, "guideline_note": "off topic"}'
    verdict = parse_llm_guard_verdict(raw, block_controversial=True)
    assert verdict.guideline_compliant is False
    assert verdict.guideline_note == "off topic"


def test_a_missing_guideline_field_means_it_was_not_checked():
    verdict = parse_llm_guard_verdict('{"verdict": "safe"}', block_controversial=True)
    assert verdict.guideline_compliant is None


def test_rejects_an_unrecognised_verdict_value():
    with pytest.raises(ValueError):
        parse_llm_guard_verdict('{"verdict": "maybe"}', block_controversial=True)


def test_rejects_text_that_is_not_json():
    with pytest.raises(json.JSONDecodeError):
        parse_llm_guard_verdict("not json at all", block_controversial=True)


# --- parse_qwen_verdict: reading the Qwen guard model's text answer ---

def test_reads_severity_and_category_from_qwen_output():
    verdict = parse_qwen_verdict("Safety: Unsafe\nCategories: Violent", block_controversial=False)
    assert verdict.safe is False
    assert verdict.severity == "unsafe"
    assert verdict.category == "Violent"


def test_qwen_output_without_a_safety_line_is_rejected():
    with pytest.raises(ValueError):
        parse_qwen_verdict("no verdict here", block_controversial=False)


# --- build_guard_backend: configuration checks ---

def test_classifier_mode_needs_an_endpoint():
    with pytest.raises(ValueError):
        build_guard_backend("classifier", block_controversial=False)


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        build_guard_backend("something-else", block_controversial=False)
