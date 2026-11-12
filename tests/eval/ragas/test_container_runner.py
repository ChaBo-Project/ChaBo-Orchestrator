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
