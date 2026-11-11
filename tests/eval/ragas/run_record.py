"""
Turns a scored run (run_eval.py's HTTP-based run_testset() + scoring.score_testset() output)
into the per-run record files: summary.json, cases.jsonl/cases.csv, manifest.json, and an
appended runs.csv row.

manifest.json/runs.csv are intentionally generic here: the caller (run_eval.py) supplies a
run_context dict with whatever it actually knows for that run - Mode A (an already-running
instance) and Mode B (a disposable container with an overridden config) know different
things, so this module doesn't assume either shape, it just records what it's given.

Known gap: cases.jsonl is meant to also carry duration, the rewritten query, and applied
filters per case. The /eval-query route doesn't capture those yet (the query rewriter and
metadata filters aren't surfaced in its response), so they're simply absent from these
records for now rather than filled with placeholders.
"""
import csv
import hashlib
import json
import os
from collections import Counter
from typing import Any, Dict, List

import pandas as pd

from scoring import METRIC_NAMES


def test_set_hash(path: str, length: int = 12) -> str:
    """SHA-256 of the test-set Excel file's raw bytes, truncated for readability. Any edit to
    the file changes this - lets two runs be confirmed to have used byte-identical test data
    with zero manual version bookkeeping. Not a security hash, just a change fingerprint."""
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return digest[:length]


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    The four metric averages plus average answer length over "ok" cases only, a count of
    cases per status, and blocked-correctness over all cases.

    An average is None (not 0) when there are zero "ok" cases to average - a missing number
    reads as "nothing was scored", not as "everything scored zero".
    """
    ok_results = [r for r in results if r["status"] == "ok"]

    averages: Dict[str, Any] = {}
    for metric in METRIC_NAMES:
        values = [r["scores"][metric] for r in ok_results if r.get("scores")]
        averages[metric] = (sum(values) / len(values)) if values else None

    answer_lengths = [len(r["answer"]) for r in ok_results if r.get("answer")]
    averages["answer_length"] = (sum(answer_lengths) / len(answer_lengths)) if answer_lengths else None

    status_counts = Counter(r["status"] for r in results)

    # Did the guard block exactly the cases that shouldn't be answered, and answer exactly
    # the ones that should? Checked over every case, not just "ok" ones.
    blocked_matches = [
        (r["status"] == "blocked") == (not r["should_be_answered"]) for r in results
    ]
    correct = sum(blocked_matches)
    total = len(blocked_matches)

    return {
        "total_cases": len(results),
        "counts": {status: status_counts.get(status, 0) for status in ("ok", "no_context", "blocked", "error")},
        "averages": averages,
        "blocked_correctness": {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 3) if total else None,
        },
    }


def build_manifest(run_context: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    """Combine caller-supplied run context (whatever that mode actually knows - endpoint URL
    and judge/test-set info for Mode A, plus rendered-config/swept-field info for Mode B) with
    this run's summary into one manifest record."""
    return {**run_context, "summary": summary}


def append_runs_csv(path: str, run_context: Dict[str, Any], summary: Dict[str, Any]) -> None:
    """
    Append one flattened row to the cross-run history CSV: run_context fields, the metric
    averages, and a count_<status> column per status. Tolerates a column set that changes
    across runs (Mode A vs Mode B carry different context fields, or a metric gets added
    later) via pandas' outer-join concat - same pattern v0 used for ragas_history.csv.
    """
    row = {
        **run_context,
        **summary.get("averages", {}),
        **{f"count_{status}": n for status, n in summary.get("counts", {}).items()},
    }
    new_df = pd.DataFrame([row])
    existing_df = None
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            existing_df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            existing_df = None

    combined_df = (
        pd.concat([existing_df, new_df], ignore_index=True, sort=False)
        if existing_df is not None
        else new_df
    )

    # Write atomically (temp file + rename) so a crash mid-write can't leave a
    # truncated/empty CSV behind for the next run to trip over.
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    combined_df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def write_cases_jsonl(results: List[Dict[str, Any]], path: str) -> None:
    """One self-contained JSON object per line - holds the full record, including retrieved
    chunk text, so a run stays readable even if interrupted partway through."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def write_cases_csv(results: List[Dict[str, Any]], path: str) -> None:
    """Flat, Excel-friendly view derived from the same results: question/answer/expected
    answer/status plus one column per metric score. No chunk text - that's cases.jsonl only."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["case_id", "question", "answer", "answer_length", "expected_answer", "status", "error", *METRIC_NAMES]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            scores = result.get("scores") or {}
            answer = result["answer"]
            writer.writerow(
                {
                    "case_id": result["case_id"],
                    "question": result["question"],
                    "answer": answer,
                    "answer_length": len(answer) if answer else None,
                    "expected_answer": result["expected_answer"],
                    "status": result["status"],
                    "error": result["error"],
                    **{metric: scores.get(metric) for metric in METRIC_NAMES},
                }
            )
