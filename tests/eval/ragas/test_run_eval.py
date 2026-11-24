import asyncio
import argparse, configparser, json
import httpx
import pytest

from run_eval import EvalAuthError, run_case


def _client(status_code: int, calls: list) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status_code, json={"status": "ok", "answer": "a", "retrieved_contexts": []})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("status_code", [401, 404])
def test_auth_rejection_raises_without_retry(status_code):
    calls = []

    async def go():
        async with _client(status_code, calls) as client:
            await run_case(client, "http://x", "bad-token", "q?")

    with pytest.raises(EvalAuthError):
        asyncio.run(go())
    assert len(calls) == 1  # failed fast, no retry


def test_success_passes_through():
    calls = []

    async def go():
        async with _client(200, calls) as client:
            return await run_case(client, "http://x", "tok", "q?")

    assert asyncio.run(go())["status"] == "ok"


def _patch_pipeline(monkeypatch):
    import run_eval

    async def fake_run_testset(endpoint, token, cases):
        return [{
            "case_id": 1, "question": "q?", "expected_answer": "e", "should_be_answered": True,
            "answer": "a", "chunks": [], "status": "no_context", "error": None,
        }]

    async def fake_score_testset(metrics, results):
        return [{**r, "scores": None} for r in results]

    monkeypatch.setattr(run_eval, "load_testset", lambda path: [object()])
    monkeypatch.setattr(run_eval, "run_testset", fake_run_testset)
    monkeypatch.setattr(run_eval, "getconfig", lambda path: configparser.ConfigParser())
    monkeypatch.setattr(run_eval, "build_judge", lambda config: object())
    monkeypatch.setattr(run_eval, "build_metrics", lambda judge: {})
    monkeypatch.setattr(run_eval, "score_testset", fake_score_testset)


def test_each_run_gets_its_own_folder_and_summary_json(tmp_path, monkeypatch):
    import run_eval

    _patch_pipeline(monkeypatch)
    testset = tmp_path / "testset.xlsx"
    testset.write_bytes(b"x")  # only hashed, never parsed (load_testset is patched)

    for stamp in ("run-1", "run-2"):
        args = argparse.Namespace(
            testset=str(testset), limit=None,
            out_dir=str(tmp_path / "results"), run_dir=str(tmp_path / "results" / stamp),
        )
        assert asyncio.run(run_eval._score_and_record("http://x", "tok", args, {"mode": "A"})) == 0

    for stamp in ("run-1", "run-2"):
        run_dir = tmp_path / "results" / stamp
        for name in ("cases.jsonl", "cases.csv", "manifest.json", "summary.json"):
            assert (run_dir / name).exists()
        assert json.loads((run_dir / "summary.json").read_text())["total_cases"] == 1

    runs_csv = (tmp_path / "results" / "runs.csv").read_text().splitlines()
    assert len(runs_csv) == 3  # header + one row per run
    assert "run_dir" in runs_csv[0]


def _args(tmp_path):
    testset = tmp_path / "testset.xlsx"
    testset.write_bytes(b"x")
    return argparse.Namespace(
        testset=str(testset), limit=None,
        out_dir=str(tmp_path / "results"), run_dir=str(tmp_path / "results" / "run-1"),
    )


@pytest.mark.parametrize("error", [ValueError("missing JUDGE_MODEL"), RuntimeError("missing AZURE_API_KEY")])
def test_judge_config_error_stops_before_any_query(tmp_path, monkeypatch, error):
    import run_eval

    _patch_pipeline(monkeypatch)
    queried = []

    async def tracking_run_testset(endpoint, token, cases):
        queried.append(cases)
        return []

    def failing_build_judge(config):
        raise error

    monkeypatch.setattr(run_eval, "run_testset", tracking_run_testset)
    monkeypatch.setattr(run_eval, "build_judge", failing_build_judge)

    assert asyncio.run(run_eval._score_and_record("http://x", "tok", _args(tmp_path), {"mode": "A"})) == 1
    assert queried == []  # no endpoint query was spent


def test_raw_answers_saved_even_if_scoring_crashes(tmp_path, monkeypatch):
    import run_eval

    _patch_pipeline(monkeypatch)

    async def crashing_score_testset(metrics, results):
        raise RuntimeError("judge exploded")

    monkeypatch.setattr(run_eval, "score_testset", crashing_score_testset)
    args = _args(tmp_path)

    with pytest.raises(RuntimeError):
        asyncio.run(run_eval._score_and_record("http://x", "tok", args, {"mode": "A"}))

    lines = (tmp_path / "results" / "run-1" / "cases.jsonl").read_text().splitlines()
    assert [json.loads(line)["answer"] for line in lines] == ["a"]
