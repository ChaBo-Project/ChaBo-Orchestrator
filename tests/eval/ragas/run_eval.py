"""
CLI entrypoint for the RAGAS v1 eval. Mode A only for now (point at an already-running
orchestrator instance over HTTP via its /eval-query route) - Mode B (spin up a disposable
local container with an overridden config) lands in a later step.

Usage:
    python tests/eval/ragas/run_eval.py --endpoint http://localhost:7860 \
        --testset path/to/testset.xlsx [--eval-token ...] [--limit 2] [--out-dir ...]

If --eval-token is omitted, EVAL_API_KEY is read from the environment.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

# Add src/ to path so imports match how main.py resolves them (needed by judge.py's own
# import of components.utils, and so this script could do the same if it ever needs to).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from components.utils import getconfig  # noqa: E402  (after sys.path setup, same pattern as judge.py)

from judge import build_judge, UnsupportedJudgeProviderError  # noqa: E402
from run_record import (  # noqa: E402
    append_runs_csv, build_manifest, summarize, test_set_hash, write_cases_csv, write_cases_jsonl,
)
from scoring import build_metrics, score_testset  # noqa: E402
from testset import load_testset  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "results")


def _git_commit() -> Optional[str]:
    """Short commit hash of the local chabo checkout running this tooling - best-effort,
    None if git isn't available or this isn't a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


async def _call_eval_query(
    client: httpx.AsyncClient, endpoint: str, token: str, question: str
) -> Dict[str, Any]:
    """One HTTP call to /eval-query. Raises on transport/HTTP failure - callers decide retry."""
    response = await client.post(
        f"{endpoint.rstrip('/')}/eval-query",
        json={"question": question},
        headers={"X-Eval-Token": token},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


async def run_case(
    client: httpx.AsyncClient, endpoint: str, token: str, question: str
) -> Dict[str, Any]:
    """
    Call /eval-query for one question, with one retry only when the first attempt fails to
    even complete (covers transient failures like an endpoint scaling up from cold) - a
    clean "no_context"/"blocked" response is never retried, those are real outcomes.
    """
    try:
        return await _call_eval_query(client, endpoint, token, question)
    except (httpx.HTTPError, ValueError) as exc:
        try:
            return await _call_eval_query(client, endpoint, token, question)
        except (httpx.HTTPError, ValueError) as retry_exc:
            return {"answer": None, "retrieved_contexts": [], "status": "error", "error": str(retry_exc)}


async def run_testset(endpoint: str, token: str, cases: List) -> List[Dict[str, Any]]:
    """
    Run every case through /eval-query, sequentially - eval shares endpoints (Qdrant,
    embedding, LLM) with the live chatbot in Mode A, so cases are not parallelised.
    """
    results: List[Dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        for case in cases:
            print(f"Processing #{case.case_id}: {case.question[:60]}...")
            result = await run_case(client, endpoint, token, case.question)
            results.append({
                "case_id": case.case_id,
                "question": case.question,
                "expected_answer": case.expected_answer,
                "should_be_answered": case.should_be_answered,
                "answer": result.get("answer"),
                "chunks": [
                    {"text": c.get("text"), "metadata": c.get("metadata", {})}
                    for c in result.get("retrieved_contexts", [])
                ],
                "status": result.get("status", "error"),
                "error": result.get("error"),
            })
    return results


async def main_async(args: argparse.Namespace) -> int:
    token = args.eval_token or os.getenv("EVAL_API_KEY")
    if not token:
        print("No eval token - pass --eval-token or set EVAL_API_KEY.")
        return 1

    try:
        cases = load_testset(args.testset)
    except Exception as exc:
        print(f"Failed to load test set: {exc}")
        return 1

    if args.limit:
        cases = cases[: args.limit]
    print(f"Loaded {len(cases)} case(s) from {args.testset}.")

    results = await run_testset(args.endpoint, token, cases)

    config = getconfig("params.cfg")
    try:
        judge = build_judge(config)
    except UnsupportedJudgeProviderError as exc:
        print(f"Judge configuration error: {exc}")
        return 1
    metrics = build_metrics(judge)

    print("Scoring with RAGAS...")
    scored = await score_testset(metrics, results)

    os.makedirs(args.out_dir, exist_ok=True)
    summary = summarize(scored)
    write_cases_jsonl(scored, os.path.join(args.out_dir, "cases.jsonl"))
    write_cases_csv(scored, os.path.join(args.out_dir, "cases.csv"))

    # Mode A run context: only what's actually known without a rendered instance config -
    # generator_model is omitted since Mode A never sees the remote instance's own config.
    run_context = {
        "mode": "A",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.endpoint,
        "chabo_commit": _git_commit(),
        "testset_path": args.testset,
        "testset_hash": test_set_hash(args.testset),
        "judge_model": config.get("ragas", "JUDGE_MODEL", fallback=None),
        "num_cases": len(scored),
    }
    manifest = build_manifest(run_context, summary)
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    append_runs_csv(os.path.join(args.out_dir, "runs.csv"), run_context, summary)

    print(f"\nDone. {summary['total_cases']} case(s), counts={summary['counts']}")
    print(f"Averages: {summary['averages']}")
    print(f"Blocked-correctness: {summary['blocked_correctness']}")
    print(f"Results written to {args.out_dir}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="ChaBo RAGAS v1 eval (Mode A: run against a live instance)")
    parser.add_argument("--endpoint", required=True, help="Base URL of the running orchestrator instance")
    parser.add_argument("--testset", required=True, help="Path to the Questions Excel file")
    parser.add_argument("--eval-token", default=None, help="Defaults to the EVAL_API_KEY env var")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases (cheap dry run)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Where to write summary.json/cases.jsonl/cases.csv")
    args = parser.parse_args()

    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
