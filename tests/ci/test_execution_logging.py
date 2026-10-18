import asyncio
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from components import logging_setup as logs

SRC_DIR = Path(logs.__file__).resolve().parents[1]
GENERATED_ID_RE = re.compile(r"[0-9a-f]{32}")
OLD = time.time() - 8 * 24 * 60 * 60


@pytest.fixture
def isolated_logging(tmp_path, monkeypatch):
    """Use temporary logs without replacing pytest's logging configuration."""
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_TO_FILE", "true")
    logs._last_retention_check.clear()

    root = logging.getLogger()
    previous_level = root.level
    router = logs.ExecutionFileRouter()
    root.addHandler(router)
    root.setLevel(logging.INFO)

    try:
        yield tmp_path
    finally:
        root.removeHandler(router)
        router.close()
        root.setLevel(previous_level)


async def receive():
    return {
        "type": "http.request",
        "body": b"",
        "more_body": False,
    }


def scope(path="/test", request_id="test-request", headers=None):
    if headers is None:
        headers = [(b"x-request-id", request_id.encode("ascii"))]
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers,
    }


async def ok_app(request_scope, request_receive, send):
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [],
    })
    await send({"type": "http.response.body", "body": b"OK"})


def run_request(app, request_scope=None, request_receive=receive):
    """Run one request through the middleware; return the sent messages."""
    responses = []

    async def send(message):
        responses.append(message)

    asyncio.run(
        logs.ExecutionLoggingMiddleware(app)(
            request_scope or scope(), request_receive, send
        )
    )
    return responses


def read_single_log(log_dir):
    files = list(log_dir.glob("run-*/execution.log"))
    assert len(files) == 1
    return files[0].read_text(encoding="utf-8")


def summary_line(content):
    return next(
        line for line in content.splitlines()
        if "Request summary:" in line
    )


def response_request_id(responses):
    headers = dict(responses[0]["headers"])
    return headers[b"x-request-id"].decode("ascii")


def run_python(code, env_updates, tmp_path):
    env = os.environ.copy()
    env.pop("LOG_TO_FILE", None)
    env["LOG_LEVEL"] = "INFO"
    env["PYTHONPATH"] = str(SRC_DIR)
    env.update(env_updates)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
    )


def test_stream_logging_stays_active_until_response_finishes(isolated_logging):
    responses = []

    async def app(request_scope, request_receive, send):
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [],
        })
        await send({
            "type": "http.response.body",
            "body": b"first",
            "more_body": True,
        })

        assert logs.execution_log_context.get().active
        assert not list(isolated_logging.glob("run-*/.finished"))

        await asyncio.sleep(0)
        logging.getLogger("test").info("SECOND_CHUNK")

        await send({
            "type": "http.response.body",
            "body": b"second",
            "more_body": False,
        })

    async def send(message):
        responses.append(message)

    asyncio.run(
        logs.ExecutionLoggingMiddleware(app)(scope(), receive, send)
    )

    files = list(isolated_logging.glob("run-*/execution.log"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "SECOND_CHUNK" in content
    assert "status=200" in content
    assert "response_complete=True outcome=finished" in content
    assert content.index("Request summary:") < content.index(
        "Execution scope finished."
    )
    assert (files[0].parent / ".finished").is_file()
    assert (b"x-request-id", b"test-request") in responses[0]["headers"]
    assert logs.run_id_context.get() == "-"
    assert logs.execution_log_context.get() is None


def test_file_logging_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.delenv("LOG_TO_FILE", raising=False)

    responses = run_request(ok_app)

    assert responses[0]["status"] == 200
    assert response_request_id(responses) == "test-request"
    assert not list(tmp_path.iterdir())


def test_configure_logging_without_log_to_file_creates_no_files(tmp_path):
    result = run_python(
        "from components.logging_setup import configure_logging; "
        "import logging; configure_logging(); "
        "logging.info('PROCESS_TEST_OK')",
        {"LOG_DIR": str(tmp_path / "logs")},
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "PROCESS_TEST_OK" in result.stderr
    assert "File logging disabled" in result.stderr
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("value", ["sometimes", "2"])
def test_invalid_log_to_file_is_rejected(monkeypatch, value):
    monkeypatch.setenv("LOG_TO_FILE", value)
    with pytest.raises(ValueError, match="LOG_TO_FILE"):
        logs.file_logging_enabled()


def test_empty_log_dir_uses_default(monkeypatch):
    monkeypatch.setenv("LOG_DIR", "  ")
    assert logs.log_directory() == Path("logs")


def test_unhandled_exception_returns_500_with_request_id(isolated_logging):
    async def app(request_scope, request_receive, send):
        raise ValueError("deliberate test")

    responses = run_request(app)

    assert responses[0]["status"] == 500
    assert response_request_id(responses) == "test-request"
    assert responses[1]["body"] == b"Internal Server Error"

    content = read_single_log(isolated_logging)
    summary = summary_line(content)
    assert "status=500" in summary
    assert "response_complete=True outcome=failed" in summary
    assert content.count("ValueError: deliberate test") == 1
    assert "Unhandled exception; returning HTTP 500." in content
    assert "Execution failed." not in content
    assert logs.run_id_context.get() == "-"
    assert logs.execution_log_context.get() is None


def test_exception_after_response_start_is_raised(isolated_logging):
    async def app(request_scope, request_receive, send):
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [],
        })
        raise ValueError("deliberate test")

    with pytest.raises(ValueError):
        run_request(app)

    content = read_single_log(isolated_logging)
    assert "status=200" in summary_line(content)
    assert "outcome=failed" in summary_line(content)
    assert content.count("ValueError: deliberate test") == 1
    assert "Execution failed." in content


def test_cancellation_preserves_exception_and_summary(isolated_logging):
    async def app(request_scope, request_receive, send):
        raise asyncio.CancelledError("deliberate test")

    with pytest.raises(asyncio.CancelledError):
        run_request(app)

    content = read_single_log(isolated_logging)
    assert "response_complete=False outcome=cancelled" in content
    assert (isolated_logging.glob("run-*/.finished").__next__()).is_file()
    assert logs.run_id_context.get() == "-"
    assert logs.execution_log_context.get() is None


def test_client_disconnect_is_reported(isolated_logging):
    async def disconnect():
        return {"type": "http.disconnect"}

    async def app(request_scope, request_receive, send):
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [],
        })
        await send({
            "type": "http.response.body",
            "body": b"partial",
            "more_body": True,
        })
        # A streaming response stops when it sees the disconnect.
        assert (await request_receive())["type"] == "http.disconnect"

    run_request(app, request_receive=disconnect)

    summary = summary_line(read_single_log(isolated_logging))
    assert "response_complete=False outcome=client_disconnected" in summary


@pytest.mark.parametrize(
    "headers",
    [
        [(b"x-request-id", b"bad id with spaces")],
        [(b"x-request-id", b"x" * 129)],
        [(b"x-request-id", b"first"), (b"x-request-id", b"second")],
        [],
    ],
    ids=["spaces", "too-long", "duplicate", "missing"],
)
def test_invalid_or_duplicate_request_id_gets_generated_id(
    isolated_logging, headers
):
    responses = run_request(ok_app, scope(headers=headers))

    request_id = response_request_id(responses)
    assert GENERATED_ID_RE.fullmatch(request_id)
    assert f"[RUN={request_id}]" in read_single_log(isolated_logging)


def test_parallel_requests_with_reused_id_have_separate_files(
    isolated_logging,
):
    async def app(request_scope, request_receive, send):
        label = request_scope["path"]
        logging.getLogger("test").info("BEGIN %s", label)
        await asyncio.to_thread(
            logging.getLogger("test").info, "WORKER %s", label
        )
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [],
        })
        await send({"type": "http.response.body", "body": b"OK"})

    async def send(message):
        pass

    async def run():
        middleware = logs.ExecutionLoggingMiddleware(app)
        await asyncio.gather(
            middleware(scope("/alpha"), receive, send),
            middleware(scope("/beta"), receive, send),
        )

    asyncio.run(run())

    files = list(isolated_logging.glob("run-*/execution.log"))
    assert len(files) == 2
    contents = [file.read_text(encoding="utf-8") for file in files]

    for label, other in (("/alpha", "/beta"), ("/beta", "/alpha")):
        content = next(text for text in contents if f"BEGIN {label}" in text)
        assert f"WORKER {label}" in content
        assert f"BEGIN {other}" not in content
        assert f"WORKER {other}" not in content
        assert "[RUN=test-request]" in content


@pytest.mark.parametrize("path", ["/health", "/health/"])
def test_health_executes_without_request_log(isolated_logging, path):
    called = []

    async def app(request_scope, request_receive, send):
        called.append(True)

    async def send(message):
        pass

    asyncio.run(
        logs.ExecutionLoggingMiddleware(app)(
            scope(path), receive, send
        )
    )

    assert called == [True]
    assert not list(isolated_logging.glob("run-*"))


@pytest.mark.parametrize("prefix", ["run", "process"])
def test_retention_keeps_recent_unfinished_and_unrelated_logs(
    isolated_logging, prefix
):
    base = isolated_logging
    old = base / f"{prefix}-1-{'a' * 32}"
    recent = base / f"{prefix}-1-{'b' * 32}"
    unfinished = base / f"{prefix}-1-{'c' * 32}"
    unrelated = base / "keep-this-folder"

    for directory in (old, recent, unfinished, unrelated):
        directory.mkdir()
        (directory / "example.log").write_text("test", encoding="utf-8")

    marker = old / ".finished"
    marker.touch()
    os.utime(marker, (OLD, OLD))
    (recent / ".finished").touch()

    # Even an old marker must not authorize deleting an unrelated folder.
    unrelated_marker = unrelated / ".finished"
    unrelated_marker.touch()
    os.utime(unrelated_marker, (OLD, OLD))

    logs.cleanup_completed_logs(base)

    assert not old.exists()
    assert recent.exists()
    assert unfinished.exists()
    assert unrelated.exists()


def _old_unfinished(base, pid, suffix):
    directory = base / f"process-{pid}-{suffix * 32}"
    directory.mkdir()
    log_file = directory / "process.log"
    log_file.write_text("test", encoding="utf-8")
    for path in (log_file, directory):
        os.utime(path, (OLD, OLD))
    return directory


def test_retention_removes_old_unfinished_logs_of_dead_processes(
    isolated_logging, monkeypatch
):
    base = isolated_logging
    dead = _old_unfinished(base, 999999, "a")
    alive = _old_unfinished(base, 4242, "b")
    restarted = _old_unfinished(base, os.getpid(), "c")
    active = _old_unfinished(base, os.getpid(), "d")
    logs._mark_active(active)

    monkeypatch.setattr(logs, "_pid_running", lambda pid: pid == 4242)
    try:
        logs.cleanup_completed_logs(base)
    finally:
        logs._unmark_active(active)

    assert not dead.exists()
    # Another live process may still write to its directory.
    assert alive.exists()
    # Our PID but not our directory: left by a previous process.
    assert not restarted.exists()
    assert active.exists()


def test_request_continues_when_run_log_cannot_be_created(
    isolated_logging, monkeypatch
):
    invalid = isolated_logging / "regular-file"
    invalid.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("LOG_DIR", str(invalid))

    async def app(request_scope, request_receive, send):
        assert logs.run_id_context.get() == "test-request"
        await ok_app(request_scope, request_receive, send)

    responses = run_request(app)

    assert responses[0]["status"] == 200
    assert responses[1]["body"] == b"OK"
    assert invalid.is_file()
    assert logs.run_id_context.get() == "-"
    assert logs.execution_log_context.get() is None


def test_startup_fails_fast_when_log_directory_is_unwritable(tmp_path):
    invalid = tmp_path / "regular-file"
    invalid.write_text("not a directory", encoding="utf-8")

    result = run_python(
        "from components.logging_setup import configure_logging; "
        "configure_logging()",
        {"LOG_DIR": str(invalid), "LOG_TO_FILE": "true"},
        tmp_path,
    )

    assert result.returncode != 0
    assert "Uncaught exception; process terminating." in result.stderr
    assert "LOG_TO_FILE=true but log directory" in result.stderr
    assert invalid.is_file()


@pytest.mark.parametrize("fail_at_startup", [False, True])
def test_process_exit_marks_logs_and_records_uncaught_exception(
    tmp_path, fail_at_startup
):
    code = (
        "from components.logging_setup import configure_logging; "
        "import logging; configure_logging(); "
        "logging.info('PROCESS_TEST_OK'); "
    )
    if fail_at_startup:
        code += "raise RuntimeError('STARTUP_TEST_FAILURE')"

    result = run_python(
        code, {"LOG_DIR": str(tmp_path), "LOG_TO_FILE": "true"}, tmp_path
    )

    if fail_at_startup:
        assert result.returncode != 0
    else:
        assert result.returncode == 0, result.stderr

    files = list(tmp_path.glob("process-*/process.log"))
    assert len(files) == 1
    assert (files[0].parent / ".finished").is_file()

    content = files[0].read_text(encoding="utf-8")
    assert "PROCESS_TEST_OK" in content
    if fail_at_startup:
        assert "Uncaught exception; process terminating." in content
        assert "Traceback" in content
        assert "RuntimeError: STARTUP_TEST_FAILURE" in content


def _generator_with_fake_model(scenario):
    from components.generator.generator_orchestrator import Generator
    from components.llm.llm_client import LLMClient

    class FakeChatModel:
        async def ainvoke(self, messages):
            if scenario == "failure":
                raise RuntimeError("TEST_PROVIDER_FAILURE")
            return SimpleNamespace(content="TEST_ANSWER")

        async def astream(self, messages):
            yield SimpleNamespace(content="FIRST_CHUNK")
            if scenario == "failure":
                raise RuntimeError("TEST_PROVIDER_FAILURE")
            if scenario == "cancel":
                raise asyncio.CancelledError()
            yield SimpleNamespace(content="LAST_CHUNK")

    # Exercise real inference methods without credentials or network calls.
    client = object.__new__(LLMClient)
    client.provider = "huggingface"
    client.inference_provider = "test-provider"
    client.model = "test-model"
    client.task = "generation"
    client.chat_model = FakeChatModel()

    generator = object.__new__(Generator)
    generator.llm_client = client
    return generator


@pytest.mark.parametrize(
    ("streaming", "scenario", "expected"),
    [
        (True, "success", "succeeded"),
        (True, "failure", "failed"),
        (True, "cancel", "cancelled"),
        (True, "stop_early", "stopped_early"),
        (False, "success", "succeeded"),
        (False, "failure", "failed"),
    ],
)
def test_generation_outcome_in_request_summary(
    isolated_logging, streaming, scenario, expected
):
    generator = _generator_with_fake_model(scenario)
    output = []

    async def app(request_scope, request_receive, send):
        async def generate():
            if streaming:
                stream = generator.astream([])
                async for chunk in stream:
                    output.append(chunk)
                    if scenario == "stop_early":
                        # Like an output guard block: stop reading.
                        await stream.aclose()
                        break
            else:
                try:
                    output.append(await generator.ainvoke([]))
                except RuntimeError:
                    # Simulate an application handling a generation failure.
                    output.append("HANDLED_FAILURE")

        # Verify that a child task updates the middleware's shared state.
        await asyncio.create_task(generate())
        await ok_app(request_scope, request_receive, send)

    async def send(message):
        pass

    request = logs.ExecutionLoggingMiddleware(app)(scope(), receive, send)

    if scenario == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(request)
    else:
        asyncio.run(request)

    content = read_single_log(isolated_logging)
    summary = summary_line(content)
    assert f"generation_outcome={expected}" in summary

    if scenario == "failure":
        assert "Request ended with generation failure:" in content
        assert "status=200" in summary
        assert "response_complete=True outcome=finished" in summary
        assert "task=generation" in content
        assert "inference_provider=test-provider" in content
        if streaming:
            assert output[0] == "FIRST_CHUNK"
            assert output[-1] == "Error: TEST_PROVIDER_FAILURE"
            assert "LLM streaming failed: task=generation" in content
    elif scenario == "success":
        assert output == (
            ["FIRST_CHUNK", "LAST_CHUNK"]
            if streaming else ["TEST_ANSWER"]
        )
        assert "Request ended with generation failure:" not in content
        assert "LLM call completed: task=generation" in content
        assert "inference_provider=test-provider" in content
        if streaming:
            assert re.search(r"streaming=True duration_s=\S+ first_chunk_s=\d", content)
    elif scenario == "stop_early":
        assert output == ["FIRST_CHUNK"]
        assert (
            "LLM streaming ended early: task=generation "
            "reason=consumer_closed" in content
        )
        assert "output_chars=11" in content
        assert re.search(r"reason=consumer_closed duration_s=\S+ first_chunk_s=\d", content)
    else:
        assert "response_complete=False outcome=cancelled" in summary
        assert "LLM streaming ended early: task=generation reason=cancelled" in content

    assert logs.generation_log_context.get() is None


def test_build_llm_client_passes_task_name(monkeypatch):
    import configparser

    from components.llm import build_llm_client

    config = configparser.ConfigParser()
    config.read_dict({"query_rewriter": {
        "llm_provider": "azure",
        "llm_model": "m",
        "llm_azure_endpoint": "https://res.example/openai/v1/",
    }})
    monkeypatch.setenv("AZURE_API_KEY", "k")

    assert build_llm_client(config, "query_rewrite").task == "query_rewrite"


@pytest.mark.parametrize(
    "target",
    [
        "/",
        "/v1/models",
        "/?__theme=light&__sign=test-secret",
        "/chatfed-ui-stream/stream?token=test-secret",
    ],
)
def test_access_logs_remove_query_strings(target):
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1234", "GET", target, "1.1", 200),
        None,
    )

    assert logs.AccessLogQueryFilter().filter(record)

    assert record.args == (
        "127.0.0.1:1234",
        "GET",
        target.partition("?")[0],
        "1.1",
        200,
    )
    assert "test-secret" not in record.getMessage()
    assert "?" not in record.getMessage()


def test_summary_defaults_for_retrieval_and_blocking(isolated_logging):
    run_request(ok_app)

    summary = summary_line(read_single_log(isolated_logging))
    assert (
        "generation_outcome=not_started retrieval_outcome=not_started "
        "documents=- blocked_by=-"
    ) in summary


def _fake_retriever(embed=None, points=None, search_error=None, rerank=None):
    """A retriever without config or network: real retrieval methods, fake I/O."""
    from components.retriever.retriever_orchestrator import ChaBoHFEndpointRetriever

    fake = SimpleNamespace(
        embedding_endpoint_url="http://embed.test/embed",
        reranker_endpoint_url="http://rerank.test/rerank",
        hf_token=None,
        hybrid_enabled=False,
        reranker_enabled=True,
        reranker_top_k=2,
        top_k=5,
        qdrant_mode="native",
        qdrant_collection="test",
        qdrant_url="http://qdrant.test",
    )
    for name in (
        "_aget_relevant_documents", "_asearch_qdrant", "_arerank",
        "_rerank_payload", "_documents_from_rerank",
        "_documents_without_rerank", "_log_hybrid_error_hint",
    ):
        setattr(fake, name, getattr(ChaBoHFEndpointRetriever, name).__get__(fake))

    async def query_points(**kwargs):
        if search_error:
            raise RuntimeError(search_error)
        return SimpleNamespace(points=points or [])

    async def get_client():
        return SimpleNamespace(query_points=query_points)

    async def no_dense_name(client):
        return None

    fake._aget_qdrant_client = get_client
    fake._aresolve_dense_vector_name = no_dense_name
    fake._query_kwargs = lambda *args: {}

    async def call_endpoint(url, token, payload):
        handler = embed if "embed" in url else rerank
        return handler(payload)

    return fake, call_endpoint


def _hit(i):
    return SimpleNamespace(id=i, payload={"text": f"doc {i}"}, score=1.0)


def _fail(message):
    def handler(payload):
        raise RuntimeError(message)
    return handler


@pytest.mark.parametrize(
    ("case", "expected", "warns"),
    [
        ("ok", "retrieval_outcome=ok documents=2", False),
        ("empty", "retrieval_outcome=empty documents=0", False),
        ("embed_fail", "retrieval_outcome=failed documents=0", True),
        ("search_fail", "retrieval_outcome=failed documents=0", True),
        ("rerank_fail", "retrieval_outcome=rerank_fallback documents=2", True),
    ],
)
def test_retrieval_outcome_in_request_summary(
    isolated_logging, monkeypatch, case, expected, warns
):
    from components.retriever import retriever_orchestrator

    embed = _fail("EMBED_DOWN") if case == "embed_fail" else (lambda p: [[0.1, 0.2]])
    rerank = (
        _fail("RERANK_DOWN") if case == "rerank_fail"
        else (lambda p: [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.5}])
    )
    retriever, call_endpoint = _fake_retriever(
        embed=embed,
        points=[] if case == "empty" else [_hit(i) for i in range(3)],
        search_error="QDRANT_DOWN" if case == "search_fail" else None,
        rerank=rerank,
    )
    monkeypatch.setattr(retriever_orchestrator, "_acall_hf_endpoint", call_endpoint)

    async def app(request_scope, request_receive, send):
        await retriever._aget_relevant_documents("query")
        await ok_app(request_scope, request_receive, send)

    run_request(app)

    content = read_single_log(isolated_logging)
    assert expected in summary_line(content)
    assert ("Request ended with degraded retrieval:" in content) is warns
    if case == "search_fail":
        assert "Qdrant search failed:" in content
        assert "No documents retrieved for query" in content


def test_a_failed_retrieval_is_not_overwritten_by_later_outcomes(isolated_logging):
    async def app(request_scope, request_receive, send):
        logs.mark_retrieval_outcome("failed")
        logs.mark_retrieval_outcome("empty")
        await ok_app(request_scope, request_receive, send)

    run_request(app)

    assert "retrieval_outcome=failed documents=0" in summary_line(
        read_single_log(isolated_logging)
    )


def test_the_first_guard_block_is_reported(isolated_logging):
    from components.guardrails.output_guard import (
        StreamingBlocklistFilter,
        compile_blocklist,
    )

    async def app(request_scope, request_receive, send):
        blocklist = StreamingBlocklistFilter(
            compile_blocklist({"en": ["forbidden"]}), "[blocked]"
        )

        async def stream():
            # Child task, like the output guard inside the LangGraph stream.
            blocklist.feed("a forbidden word ")

        await asyncio.create_task(stream())
        logs.mark_blocked("output_classifier")
        await ok_app(request_scope, request_receive, send)

    run_request(app)

    assert "blocked_by=output_blocklist" in summary_line(
        read_single_log(isolated_logging)
    )
