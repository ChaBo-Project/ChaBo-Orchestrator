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
    monkeypatch.setattr(run_eval, "load_config", lambda path: configparser.ConfigParser())
    monkeypatch.setattr(run_eval, "build_judge", lambda config: object())
    monkeypatch.setattr(run_eval, "resolve_judge_model", lambda config: "judge-x")
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


@pytest.mark.parametrize("endpoint", [
    "https://my-space.hf.space", "http://localhost:7860", "http://127.0.0.1:18860", "http://[::1]:7860",
])
def test_check_endpoint_accepts_https_and_local_http(endpoint):
    from run_eval import check_endpoint

    check_endpoint(endpoint)


@pytest.mark.parametrize("endpoint", ["http://10.0.0.5:7860", "http://my-space.hf.space", "ftp://x", "localhost:7860"])
def test_check_endpoint_rejects_remote_http_and_non_urls(endpoint):
    from run_eval import check_endpoint

    with pytest.raises(ValueError):
        check_endpoint(endpoint)


def test_check_endpoint_allow_http_for_private_network():
    from run_eval import check_endpoint

    check_endpoint("http://10.0.0.5:7860", allow_http=True)


def test_hf_token_sent_only_to_hf_spaces(monkeypatch):
    from run_eval import request_headers

    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    assert request_headers("https://my-space.hf.space", "tok")["Authorization"] == "Bearer hf_secret"
    for endpoint in ("https://example.org", "https://hf.space.evil.org", "http://localhost:7860"):
        headers = request_headers(endpoint, "tok")
        assert "Authorization" not in headers
        assert headers["X-Eval-Token"] == "tok"


def test_eval_query_request_carries_hf_bearer(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    calls = []

    async def go():
        async with _client(200, calls) as client:
            await run_case(client, "https://my-space.hf.space", "tok", "q?")

    asyncio.run(go())
    assert calls[0].headers["Authorization"] == "Bearer hf_secret"
    assert calls[0].headers["X-Eval-Token"] == "tok"


def _mode_b_args(tmp_path, **overrides):
    values = dict(
        eval_token="tok", endpoint=None, instance_repo=str(tmp_path), override=None,
        allow_http=False, out_dir=str(tmp_path / "results"), host_port=18860,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_missing_override_fails_before_render(tmp_path, monkeypatch):
    import run_eval

    rendered = []
    monkeypatch.setattr(run_eval, "render_config", lambda repo: rendered.append(repo))

    args = _mode_b_args(tmp_path, override=str(tmp_path / "typo.cfg"))
    assert asyncio.run(run_eval.main_async(args)) == 1
    assert rendered == []


def test_mode_b_errors_print_a_message_not_a_traceback(tmp_path, monkeypatch, capsys):
    import run_eval
    from config_render import ConfigRenderError

    def failing_render(repo):
        raise ConfigRenderError("validate-config.py failed: bad value")

    monkeypatch.setattr(run_eval, "render_config", failing_render)

    assert asyncio.run(run_eval.main_async(_mode_b_args(tmp_path))) == 1
    assert "Mode B failed: validate-config.py failed" in capsys.readouterr().out


def test_manifest_and_runs_csv_hold_no_absolute_paths(tmp_path, monkeypatch):
    import run_eval

    _patch_pipeline(monkeypatch)
    args = _args(tmp_path)
    assert asyncio.run(run_eval._score_and_record("http://x", "tok", args, {"mode": "A"})) == 0

    manifest = (tmp_path / "results" / "run-1" / "manifest.json").read_text()
    runs_csv = (tmp_path / "results" / "runs.csv").read_text()
    assert json.loads(manifest)["testset_path"] == "testset.xlsx"
    assert str(tmp_path) not in manifest
    assert str(tmp_path) not in runs_csv



def test_http_space_does_not_receive_hf_token(monkeypatch):
    from run_eval import request_headers

    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    headers = request_headers("http://example.hf.space", "dummy-eval-token")

    assert headers["X-Eval-Token"] == "dummy-eval-token"
    assert "Authorization" not in headers


def test_malformed_endpoint_returns_a_cli_error(capsys):
    from run_eval import main_async

    args = argparse.Namespace(
        endpoint="https://example.invalid:bad-port",
        eval_token="dummy-eval-token",
        allow_http=False,
    )

    assert asyncio.run(main_async(args)) == 1
    assert "endpoint" in capsys.readouterr().out.lower()


def test_malformed_override_returns_a_mode_b_error(tmp_path, monkeypatch, capsys):
    import run_eval

    rendered_dir = tmp_path / "rendered"
    rendered_dir.mkdir()
    (rendered_dir / "params.override.cfg").write_text("[ragas]\n", encoding="utf-8")
    override = tmp_path / "broken.cfg"
    override.write_text("key = value\n", encoding="utf-8")
    monkeypatch.setattr(run_eval, "render_config", lambda _: (rendered_dir, "test"))

    args = _mode_b_args(tmp_path, override=str(override))

    assert asyncio.run(run_eval.main_async(args)) == 1
    assert "Mode B failed:" in capsys.readouterr().out


def test_malformed_judge_config_returns_a_cli_error(tmp_path, monkeypatch, capsys):
    import run_eval

    config = tmp_path / "eval.cfg"
    config.write_text("key = value\n", encoding="utf-8")
    monkeypatch.setattr(run_eval, "EVAL_CONFIG_PATH", str(config))
    monkeypatch.setattr(run_eval, "load_testset", lambda _: [object()])
    args = argparse.Namespace(testset="unused.xlsx", limit=None)

    assert asyncio.run(
        run_eval._score_and_record(
            "https://example.invalid", "dummy-eval-token", args, {}
        )
    ) == 1
    assert "Judge configuration error:" in capsys.readouterr().out


def test_instance_info_from_responses():
    from run_eval import instance_info

    same = {"version": "0.9.6", "commit": "abc"}
    assert instance_info([{"instance": same}, {"instance": same}]) == {
        "version": "0.9.6", "commit": "abc", "mixed": False,
    }
    assert instance_info([{"instance": None}]) == {"version": "unknown", "commit": "unknown", "mixed": False}
    assert instance_info([{"instance": same}, {"instance": {"version": "0.9.7", "commit": "def"}}])["mixed"]


@pytest.mark.parametrize("instance, tool, expected", [
    ({"version": "0.9.6", "mixed": False}, "0.9.6", None),
    ({"version": "0.9.5", "mixed": False}, "0.9.6", "chabo-rag-orchestrator-v0.9.5"),
    ({"version": "dev", "mixed": False}, "0.9.6", "can't verify"),
    ({"version": "unknown", "mixed": False}, "0.9.6", "can't verify"),
    ({"version": "0.9.6", "mixed": False}, None, "can't verify"),
    ({"version": "0.9.6", "mixed": True}, "0.9.6", "changed during the run"),
])
def test_version_warning(instance, tool, expected):
    from run_eval import version_warning

    warning = version_warning(instance, tool)
    if expected is None:
        assert warning is None
    else:
        assert expected in warning


def test_tool_version_reads_release_manifest():
    import run_eval

    with open(run_eval.RELEASE_MANIFEST_PATH, encoding="utf-8") as f:
        assert run_eval._tool_version() == json.load(f)["."]


def test_manifest_records_instance_and_eval_tool_versions(tmp_path, monkeypatch):
    import run_eval

    _patch_pipeline(monkeypatch)

    async def fake_run_testset(endpoint, token, cases):
        return [{
            "case_id": 1, "question": "q?", "expected_answer": "e", "should_be_answered": True,
            "answer": "a", "chunks": [], "status": "no_context", "error": None,
            "instance": {"version": "0.9.6", "commit": "abc1234"},
        }]

    monkeypatch.setattr(run_eval, "run_testset", fake_run_testset)
    monkeypatch.setattr(run_eval, "_tool_version", lambda: "0.9.6")
    assert asyncio.run(run_eval._score_and_record("http://x", "tok", _args(tmp_path), {"mode": "A"})) == 0

    manifest = json.loads((tmp_path / "results" / "run-1" / "manifest.json").read_text())
    assert manifest["instance_version"] == "0.9.6"
    assert manifest["instance_commit"] == "abc1234"
    assert manifest["eval_tool_version"] == "0.9.6"
    assert "eval_tool_commit" in manifest and "chabo_commit" not in manifest

@pytest.mark.parametrize("missing_first", [False, True])
def test_instance_info_marks_known_and_missing_metadata_as_mixed(missing_first):
    from run_eval import instance_info

    known = {"instance": {"version": "0.9.6", "commit": "abc1234"}}
    missing = {}
    results = [missing, known] if missing_first else [known, missing]

    assert instance_info(results)["mixed"] is True

def test_mixed_run_records_flag_and_warns_once(tmp_path, monkeypatch, capsys):
    import csv
    import run_eval

    _patch_pipeline(monkeypatch)

    async def fake_run_testset(endpoint, token, cases):
        known = {
            "case_id": 1,
            "question": "First question?",
            "expected_answer": "Expected answer",
            "should_be_answered": True,
            "answer": "Answer",
            "chunks": [],
            "status": "no_context",
            "error": None,
            "instance": {"version": "0.9.6", "commit": "abc1234"},
        }
        missing = {**known, "case_id": 2, "instance": None}
        return [known, missing]

    monkeypatch.setattr(run_eval, "run_testset", fake_run_testset)
    monkeypatch.setattr(run_eval, "_tool_version", lambda: "0.9.6")

    assert asyncio.run(
        run_eval._score_and_record(
            "http://x", "tok", _args(tmp_path), {"mode": "A"}
        )
    ) == 0

    results_dir = tmp_path / "results"
    manifest = json.loads((results_dir / "run-1" / "manifest.json").read_text())
    with (results_dir / "runs.csv").open(encoding="utf-8", newline="") as f:
        history_row = next(csv.DictReader(f))

    output = capsys.readouterr().out
    assert {
        "manifest_mixed": manifest.get("instance_mixed"),
        "history_mixed": history_row.get("instance_mixed", "").lower(),
        "version_warnings": output.count("Warning: the instance version"),
    } == {
        "manifest_mixed": True,
        "history_mixed": "true",
        "version_warnings": 1,
    }
