"""
Tests for run_record.py's summarize()/write_cases_jsonl()/write_cases_csv() - pure file I/O
and arithmetic, no judge or graph needed.

Covers:
- averages are computed over "ok" cases only, never over failed/blocked/no_context cases
- an average is None (not 0) when there are zero "ok" cases
- status counts include every status, even ones with zero cases
- cases.jsonl round-trips one JSON object per line, chunks included
- cases.csv flattens scores into columns and leaves them blank for unscored cases
"""
import csv
import json

import pytest

from run_record import summarize, write_cases_csv, write_cases_jsonl


def _ok(case_id, faithfulness, context_recall, context_precision, factual_correctness):
    return {
        "case_id": case_id,
        "question": f"question {case_id}?",
        "answer": f"answer {case_id}",
        "expected_answer": f"expected {case_id}",
        "chunks": [{"text": "chunk text", "metadata": {"source": "a.pdf"}}],
        "status": "ok",
        "error": None,
        "scores": {
            "faithfulness": faithfulness,
            "context_recall": context_recall,
            "context_precision": context_precision,
            "factual_correctness": factual_correctness,
        },
    }


def _unscored(case_id, status):
    return {
        "case_id": case_id,
        "question": f"question {case_id}?",
        "answer": None if status == "error" else f"answer {case_id}",
        "expected_answer": f"expected {case_id}",
        "chunks": [],
        "status": status,
        "error": "something failed" if status == "error" else None,
        "scores": None,
    }


def test_summarize_averages_only_ok_cases():
    results = [
        _ok(1, faithfulness=1.0, context_recall=0.8, context_precision=0.6, factual_correctness=0.4),
        _ok(2, faithfulness=0.0, context_recall=0.4, context_precision=0.8, factual_correctness=0.6),
        _unscored(3, "error"),
        _unscored(4, "no_context"),
        _unscored(5, "blocked"),
    ]

    summary = summarize(results)

    assert summary["total_cases"] == 5
    assert summary["counts"] == {"ok": 2, "no_context": 1, "blocked": 1, "error": 1}
    assert summary["averages"]["faithfulness"] == pytest.approx(0.5)
    assert summary["averages"]["context_recall"] == pytest.approx(0.6)
    assert summary["averages"]["context_precision"] == pytest.approx(0.7)
    assert summary["averages"]["factual_correctness"] == pytest.approx(0.5)


def test_summarize_average_is_none_when_no_ok_cases():
    results = [_unscored(1, "error"), _unscored(2, "blocked")]

    summary = summarize(results)

    assert summary["counts"] == {"ok": 0, "no_context": 0, "blocked": 1, "error": 1}
    assert all(v is None for v in summary["averages"].values())


def test_write_cases_jsonl_round_trips(tmp_path):
    results = [_ok(1, 1.0, 0.9, 0.8, 0.7), _unscored(2, "no_context")]
    path = tmp_path / "run-1" / "cases.jsonl"

    write_cases_jsonl(results, str(path))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    loaded = [json.loads(line) for line in lines]
    assert loaded[0]["case_id"] == 1
    assert loaded[0]["scores"]["faithfulness"] == 1.0
    assert loaded[0]["chunks"] == [{"text": "chunk text", "metadata": {"source": "a.pdf"}}]
    assert loaded[1]["case_id"] == 2
    assert loaded[1]["scores"] is None


def test_write_cases_csv_flattens_scores_and_blanks_unscored(tmp_path):
    results = [_ok(1, 1.0, 0.9, 0.8, 0.7), _unscored(2, "error")]
    path = tmp_path / "run-1" / "cases.csv"

    write_cases_csv(results, str(path))

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["case_id"] == "1"
    assert rows[0]["faithfulness"] == "1.0"
    assert rows[0]["status"] == "ok"

    assert rows[1]["case_id"] == "2"
    assert rows[1]["faithfulness"] == ""
    assert rows[1]["status"] == "error"
    assert rows[1]["error"] == "something failed"
    # cases.csv is deliberately answer/score-only - no chunk text column at all.
    assert "chunks" not in rows[1]
