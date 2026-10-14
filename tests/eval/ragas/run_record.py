"""
Turns a scored run (graph_runner.run_testset() + scoring.score_testset() output) into the
per-run record files from Step 4 of the RAGAS eval plan.

Only summary.json's numbers and cases.jsonl/cases.csv are built here. manifest.json and the
appended runs.csv row need real run context (ChaBo's git commit/tag, the instance's resolved
config, generator/judge model) that only exists once the Step 5 CLI wires an instance repo in
- deliberately not invented here.

Known gap vs. the full plan: cases.jsonl is meant to also carry duration, the rewritten
query, and applied filters per case. graph_runner.run_case() doesn't capture those yet (the
query rewriter and metadata filters aren't wired into the standalone-question runner built so
far), so they're simply absent from these records for now rather than filled with placeholders.
"""
import csv
import json
import os
from collections import Counter
from typing import Any, Dict, List

from scoring import METRIC_NAMES


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    The four metric averages over "ok" cases only, plus a count of cases per status.

    An average is None (not 0) when there are zero "ok" cases to average - a missing number
    reads as "nothing was scored", not as "everything scored zero".
    """
    ok_results = [r for r in results if r["status"] == "ok"]

    averages: Dict[str, Any] = {}
    for metric in METRIC_NAMES:
        values = [r["scores"][metric] for r in ok_results if r.get("scores")]
        averages[metric] = (sum(values) / len(values)) if values else None

    status_counts = Counter(r["status"] for r in results)

    return {
        "total_cases": len(results),
        "counts": {status: status_counts.get(status, 0) for status in ("ok", "no_context", "blocked", "error")},
        "averages": averages,
    }


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
    fieldnames = ["case_id", "question", "answer", "expected_answer", "status", "error", *METRIC_NAMES]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            scores = result.get("scores") or {}
            writer.writerow(
                {
                    "case_id": result["case_id"],
                    "question": result["question"],
                    "answer": result["answer"],
                    "expected_answer": result["expected_answer"],
                    "status": result["status"],
                    "error": result["error"],
                    **{metric: scores.get(metric) for metric in METRIC_NAMES},
                }
            )
