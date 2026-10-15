import asyncio
import logging

import pytest

from components import logging_setup as logs


@pytest.fixture
def isolated_logging(tmp_path, monkeypatch):
    """Use temporary logs without replacing pytest's logging configuration."""
    monkeypatch.setenv("LOG_DIR", str(tmp_path))

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


def scope(path="/test", request_id="test-request"):
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [(b"x-request-id", request_id.encode("ascii"))],
    }


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


@pytest.mark.parametrize(
    ("exception_type", "outcome"),
    [
        (ValueError, "failed"),
        (asyncio.CancelledError, "cancelled"),
    ],
)
def test_failure_and_cancellation_preserve_exception_and_summary(
    isolated_logging, exception_type, outcome
):
    async def app(request_scope, request_receive, send):
        raise exception_type("deliberate test")

    async def send(message):
        pass

    with pytest.raises(exception_type):
        asyncio.run(
            logs.ExecutionLoggingMiddleware(app)(scope(), receive, send)
        )

    files = list(isolated_logging.glob("run-*/execution.log"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert f"response_complete=False outcome={outcome}" in content
    assert (files[0].parent / ".finished").is_file()
    assert logs.run_id_context.get() == "-"
    assert logs.execution_log_context.get() is None


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


def test_health_executes_without_request_log(isolated_logging):
    called = []

    async def app(request_scope, request_receive, send):
        called.append(True)

    async def send(message):
        pass

    asyncio.run(
        logs.ExecutionLoggingMiddleware(app)(
            scope("/health"), receive, send
        )
    )

    assert called == [True]
    assert not list(isolated_logging.glob("run-*"))

@pytest.mark.parametrize("prefix", ["run", "process"])
def test_retention_keeps_recent_unfinished_and_unrelated_logs(
    isolated_logging, prefix
):
    import os
    import time

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
    old_time = time.time() - 8 * 24 * 60 * 60
    os.utime(marker, (old_time, old_time))
    (recent / ".finished").touch()

    # Even an old marker must not authorize deleting an unrelated folder.
    unrelated_marker = unrelated / ".finished"
    unrelated_marker.touch()
    os.utime(unrelated_marker, (old_time, old_time))

    logs.cleanup_completed_logs(base)

    assert not old.exists()
    assert recent.exists()
    assert unfinished.exists()
    assert unrelated.exists()


def test_request_continues_when_log_directory_is_invalid(
    isolated_logging, monkeypatch
):
    invalid = isolated_logging / "regular-file"
    invalid.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("LOG_DIR", str(invalid))
    responses = []

    async def app(request_scope, request_receive, send):
        assert logs.run_id_context.get() == "test-request"
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [],
        })
        await send({"type": "http.response.body", "body": b"OK"})

    async def send(message):
        responses.append(message)

    asyncio.run(
        logs.ExecutionLoggingMiddleware(app)(scope(), receive, send)
    )

    assert responses[0]["status"] == 200
    assert responses[1]["body"] == b"OK"
    assert invalid.is_file()
    assert logs.run_id_context.get() == "-"
    assert logs.execution_log_context.get() is None


@pytest.mark.parametrize("fail_at_startup", [False, True])
def test_process_exit_marks_logs_and_records_uncaught_exception(
    tmp_path, fail_at_startup
):
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = os.environ.copy()
    env["LOG_DIR"] = str(tmp_path)
    env["LOG_LEVEL"] = "INFO"
    env["PYTHONPATH"] = str(Path(logs.__file__).resolve().parents[1])

    code = (
        "from components.logging_setup import configure_logging; "
        "import logging; configure_logging(); "
        "logging.info('PROCESS_TEST_OK'); "
    )
    if fail_at_startup:
        code += "raise RuntimeError('STARTUP_TEST_FAILURE')"

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
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

@pytest.mark.parametrize(
    ("streaming", "scenario", "expected"),
    [
        (True, "success", "succeeded"),
        (True, "failure", "failed"),
        (True, "cancel", "cancelled"),
        (False, "success", "succeeded"),
        (False, "failure", "failed"),
    ],
)
def test_generation_outcome_in_request_summary(
    isolated_logging, streaming, scenario, expected
):
    from types import SimpleNamespace

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
    client.chat_model = FakeChatModel()

    generator = object.__new__(Generator)
    generator.llm_client = client
    output = []

    async def app(request_scope, request_receive, send):
        async def generate():
            if streaming:
                async for chunk in generator.astream([]):
                    output.append(chunk)
            else:
                try:
                    output.append(await generator.ainvoke([]))
                except RuntimeError:
                    # Simulate an application handling a generation failure.
                    output.append("HANDLED_FAILURE")

        # Verify that a child task updates the middleware's shared state.
        await asyncio.create_task(generate())

        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [],
        })
        await send({
            "type": "http.response.body",
            "body": b"complete",
            "more_body": False,
        })

    async def send(message):
        pass

    request = logs.ExecutionLoggingMiddleware(app)(scope(), receive, send)

    if scenario == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(request)
    else:
        asyncio.run(request)

    files = list(isolated_logging.glob("run-*/execution.log"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    summary = next(
        line for line in content.splitlines()
        if "Request summary:" in line
    )
    assert f"generation_outcome={expected}" in summary

    if scenario == "failure":
        assert "Request ended with generation failure:" in content
        assert "status=200" in summary
        assert "response_complete=True outcome=finished" in summary
        if streaming:
            assert output[0] == "FIRST_CHUNK"
            assert output[-1] == "Error: TEST_PROVIDER_FAILURE"
            assert "inference_provider=test-provider" in content
    elif scenario == "success":
        assert output == (
            ["FIRST_CHUNK", "LAST_CHUNK"]
            if streaming else ["TEST_ANSWER"]
        )
        assert "Request ended with generation failure:" not in content
    else:
        assert "response_complete=False outcome=cancelled" in summary

    assert logs.generation_log_context.get() is None

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
