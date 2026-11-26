"""
CLI entrypoint for the RAGAS v1 eval. Two modes, both ending up calling the same /eval-query
contract on some orchestrator base URL:

  Mode A (baseline) - point at an already-running real instance:
    python tests/eval/ragas/run_eval.py --endpoint http://localhost:7860 \
        --testset path/to/testset.xlsx [--eval-token ...] [--limit 2] [--out-dir ...]

  Mode B (parameter sweep) - spin up a disposable local copy of an instance with an
  overridden config, run against that, tear it down afterward:
    python tests/eval/ragas/run_eval.py --instance-repo path/to/instance-checkout \
        --testset path/to/testset.xlsx --override path/to/override.cfg [--host-port 18860]

Judge settings are read from eval.cfg next to this script (RAGAS_* env vars override it).
Self-contained: imports nothing from the orchestrator's src/, so it can run from any folder.

If --eval-token is omitted, EVAL_API_KEY is read from the environment (Mode A) or generated
implicitly by reusing that same value as the temp container's own EVAL_API_KEY (Mode B).
"""
import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import configparser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from config_render import ConfigRenderError, apply_override, render_config
from container_runner import ContainerError, temp_container
from eval_config import load_config
from judge import build_judge, resolve_judge_model, UnsupportedJudgeProviderError
from run_record import (
    append_runs_csv, build_manifest, summarize, test_set_hash, write_cases_csv, write_cases_jsonl,
)
from scoring import build_metrics, score_testset
from testset import load_testset

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
EVAL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.cfg")
DEFAULT_HOST_PORT = 18860

# Forwarded to a Mode B temp container if set in this process's own environment - the user
# loads these locally themselves beforehand (no automatic secret-fetching), same as any other
# eval run. Not an exhaustive list of every possible provider key, just the common ones.
SECRET_ENV_VARS = (
    "HF_TOKEN", "QDRANT_API_KEY", "AZURE_API_KEY",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY",
)


def _git_commit() -> Optional[str]:
    """Short commit hash of the local chabo checkout running this tooling - best-effort,
    None if git isn't available or this isn't a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return result.stdout.strip()
    except Exception:
        return None


def _gather_secrets() -> Dict[str, str]:
    """Whichever of SECRET_ENV_VARS are actually set in this process's environment."""
    return {name: os.environ[name] for name in SECRET_ENV_VARS if os.environ.get(name)}


class EvalAuthError(RuntimeError):
    """The orchestrator rejected the eval token (or has no EVAL_API_KEY set at all)."""


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def check_endpoint(endpoint: str, allow_http: bool = False) -> None:
    """Refuse an endpoint the eval token should not be sent to: anything that is not an
    http(s) URL, and plain http:// to a non-local host (the token would travel unencrypted)
    unless allow_http is set for a trusted private network."""
    url = httpx.URL(endpoint)
    if url.scheme not in ("http", "https") or not url.host:
        raise ValueError(f"--endpoint must be an http(s) URL, e.g. https://... - got {endpoint!r}")
    if url.scheme == "http" and url.host not in LOCAL_HOSTS and not allow_http:
        raise ValueError(
            f"Refusing plain http:// to {url.host}: the eval token would be sent unencrypted. "
            "Use https://, or pass --allow-http for a trusted private network."
        )


def request_headers(endpoint: str, token: str) -> Dict[str, str]:
    """Headers for /eval-query. A private Hugging Face Space also needs the HF token as a
    Bearer header - sent only to *.hf.space hosts, never to any other endpoint."""
    headers = {"X-Eval-Token": token}
    hf_token = os.getenv("HF_TOKEN")
    url = httpx.URL(endpoint)
    if hf_token and url.scheme == "https" and (url.host or "").endswith(".hf.space"):
        headers["Authorization"] = f"Bearer {hf_token}"
    return headers


async def _call_eval_query(
    client: httpx.AsyncClient, endpoint: str, token: str, question: str
) -> Dict[str, Any]:
    """One HTTP call to /eval-query. Raises on transport/HTTP failure - callers decide retry."""
    response = await client.post(
        f"{endpoint.rstrip('/')}/eval-query",
        json={"question": question},
        headers=request_headers(endpoint, token),
        timeout=120.0,
    )
    # /eval-query answers 404 (not 401) on a bad token, so both mean "auth problem".
    # EvalAuthError is not an httpx.HTTPError, so run_case's retry does not swallow it.
    if response.status_code in (401, 404):
        raise EvalAuthError(
            f"{endpoint} rejected the eval token (HTTP {response.status_code}). "
            "Check that --eval-token / EVAL_API_KEY matches the EVAL_API_KEY secret "
            "set on that instance, that the instance runs a version that has /eval-query, "
            "and, for a private Hugging Face Space, that HF_TOKEN is set."
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
    except (httpx.HTTPError, ValueError):
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


async def _score_and_record(
    endpoint: str, token: str, args: argparse.Namespace, extra_run_context: Dict[str, Any]
) -> int:
    """Shared tail end of both modes: load test set, run it, score it, write records."""
    try:
        cases = load_testset(args.testset)
    except Exception as exc:
        print(f"Failed to load test set: {exc}")
        return 1

    if args.limit:
        cases = cases[: args.limit]
    print(f"Loaded {len(cases)} case(s) from {args.testset}.")

    # Build the judge before running any case: a judge config error (missing key, endpoint,
    # or model) must stop the run before it spends on endpoint queries.
    try:
        config = load_config(EVAL_CONFIG_PATH)
        judge = build_judge(config)
    except configparser.Error:
        print("Judge configuration error: check the INI syntax in eval.cfg.")
        return 1
    except (UnsupportedJudgeProviderError, ValueError, RuntimeError) as exc:
        print(f"Judge configuration error: {exc}")
        return 1
    metrics = build_metrics(judge)

    try:
        results = await run_testset(endpoint, token, cases)
    except EvalAuthError as exc:
        print(f"Aborting: {exc}")
        return 1

    # Save the raw answers before scoring, so a crash during scoring does not lose them.
    # Overwritten below with the scored version.
    os.makedirs(args.run_dir, exist_ok=True)
    cases_jsonl_path = os.path.join(args.run_dir, "cases.jsonl")
    write_cases_jsonl(results, cases_jsonl_path)

    print("Scoring with RAGAS...")
    scored = await score_testset(metrics, results)

    summary = summarize(scored)
    write_cases_jsonl(scored, cases_jsonl_path)
    write_cases_csv(scored, os.path.join(args.run_dir, "cases.csv"))
    with open(os.path.join(args.run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    run_context = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "chabo_commit": _git_commit(),
        # File names only: results get shared, and absolute paths expose the local machine.
        "testset_path": os.path.basename(args.testset),
        "testset_hash": test_set_hash(args.testset),
        "judge_model": resolve_judge_model(config),
        "num_cases": len(scored),
        "run_dir": os.path.basename(args.run_dir),
        **extra_run_context,
    }
    manifest = build_manifest(run_context, summary)
    with open(os.path.join(args.run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    append_runs_csv(os.path.join(args.out_dir, "runs.csv"), run_context, summary)

    print(f"\nDone. {summary['total_cases']} case(s), counts={summary['counts']}")
    print(f"Averages: {summary['averages']}")
    print(f"Blocked-correctness: {summary['blocked_correctness']}")
    if summary["scoring_errors"]:
        print(f"Warning: {summary['scoring_errors']} case(s) could not be scored - see scoring_error in cases.jsonl")
    print(f"Results written to {args.run_dir}")
    return 0


async def _run_mode_a(args: argparse.Namespace, token: str) -> int:
    return await _score_and_record(args.endpoint, token, args, {"mode": "A"})


async def _run_mode_b(args: argparse.Namespace, token: str) -> int:
    instance_repo_dir = Path(args.instance_repo).resolve()
    print(f"Rendering config from {instance_repo_dir}...")
    rendered_dir, chabo_tag = render_config(instance_repo_dir)
    try:
        override_path = Path(args.override) if args.override else None
        swept = apply_override(rendered_dir / "params.override.cfg", override_path)

        os.makedirs(args.run_dir, exist_ok=True)
        rendered_copy_path = os.path.join(args.run_dir, "params.override.cfg")
        shutil.copy(rendered_dir / "params.override.cfg", rendered_copy_path)

        secrets = _gather_secrets()
        secrets["EVAL_API_KEY"] = token  # our own disposable container - we set its token too

        print(f"Starting temporary orchestrator container (chabo_tag={chabo_tag}, port={args.host_port})...")
        async with temp_container(chabo_tag, rendered_dir, args.host_port, secrets) as endpoint:
            print(f"Container healthy at {endpoint}.")
            return await _score_and_record(endpoint, token, args, {
                "mode": "B",
                "chabo_tag": chabo_tag,
                "override_path": os.path.basename(args.override) if args.override else None,
                "swept_fields": swept,
                "rendered_config_copy": os.path.basename(rendered_copy_path),  # inside the run folder
            })
    finally:
        # rendered_dir is a disposable temp directory (see config_render.render_config's
        # docstring) - the permanent record already landed at rendered_copy_path above.
        shutil.rmtree(rendered_dir, ignore_errors=True)


async def main_async(args: argparse.Namespace) -> int:
    token = args.eval_token or os.getenv("EVAL_API_KEY")
    if not token:
        print("No eval token - pass --eval-token or set EVAL_API_KEY.")
        return 1

    if args.endpoint:
        try:
            check_endpoint(args.endpoint, args.allow_http)
        except httpx.InvalidURL:
            print("Invalid --endpoint URL. Check the host and port.")
            return 1
        except ValueError as exc:
            print(exc)
            return 1


    # One folder per run so runs never overwrite each other. No ":" - invalid in Windows paths.
    args.run_dir = os.path.join(args.out_dir, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"))


    if args.instance_repo:
        # Check the override before rendering, so a typo fails in a second, not after a render.
        if args.override and not Path(args.override).is_file():
            print(f"Override file not found: {args.override}")
            return 1
        try:
            return await _run_mode_b(args, token)
        except configparser.Error:
            print("Mode B failed: check the INI syntax in the rendered config or override file.")
            return 1
        except (ConfigRenderError, ContainerError, FileNotFoundError) as exc:
            print(f"Mode B failed: {exc}")
            return 1
    return await _run_mode_a(args, token)


def main() -> None:
    parser = argparse.ArgumentParser(description="ChaBo RAGAS v1 eval")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--endpoint", help="Mode A: base URL of an already-running orchestrator instance")
    mode.add_argument("--instance-repo", help="Mode B: local path to an already-checked-out instance repo")

    parser.add_argument("--testset", required=True, help="Path to the Questions Excel file")
    parser.add_argument("--eval-token", default=None, help="Defaults to the EVAL_API_KEY env var")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases (cheap dry run)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Base folder for results: each run gets its own timestamped subfolder; runs.csv stays here")
    parser.add_argument("--override", default=None, help="Mode B only: config override file to layer on top")
    parser.add_argument("--host-port", type=int, default=DEFAULT_HOST_PORT, help="Mode B only: local port for the temp container")
    parser.add_argument("--allow-http", action="store_true", help="Mode A only: allow plain http:// to a non-local host (trusted private network)")
    args = parser.parse_args()

    if args.override and not args.instance_repo:
        parser.error("--override only makes sense with --instance-repo (Mode B)")

    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
