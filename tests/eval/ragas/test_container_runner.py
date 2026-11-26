from pathlib import Path

import container_runner


def test_secret_values_never_appear_in_docker_command(monkeypatch):
    calls = []
    monkeypatch.setattr(container_runner, "_run", lambda cmd, env=None: calls.append((cmd, env)))

    container_runner.start_container("1.0", Path("/tmp/cfg"), 18860, {"HF_TOKEN": "s3cret-value"})

    run_cmd, run_env = calls[-1]
    assert "s3cret-value" not in " ".join(run_cmd)
    assert "HF_TOKEN" in run_cmd
    assert run_env["HF_TOKEN"] == "s3cret-value"


def test_each_container_gets_a_unique_name(monkeypatch):
    calls = []
    monkeypatch.setattr(container_runner, "_run", lambda cmd, env=None: calls.append(cmd))

    first = container_runner.start_container("1.0", Path("/tmp/cfg"), 18860, {})
    second = container_runner.start_container("1.0", Path("/tmp/cfg"), 18861, {})

    assert first != second
    assert first.startswith("chabo-eval-") and first in calls[1]


def test_temp_container_stops_the_container_it_started(monkeypatch):
    import asyncio

    stopped = []
    monkeypatch.setattr(container_runner, "start_container", lambda *a: "chabo-eval-abc")
    monkeypatch.setattr(container_runner, "stop_container", stopped.append)

    async def healthy(port):
        return None

    monkeypatch.setattr(container_runner, "wait_for_health", healthy)

    async def go():
        async with container_runner.temp_container("1.0", Path("/tmp/cfg"), 18860, {}):
            raise RuntimeError("run crashed")

    try:
        asyncio.run(go())
    except RuntimeError:
        pass
    assert stopped == ["chabo-eval-abc"]
