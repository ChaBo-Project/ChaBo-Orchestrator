import configparser
import subprocess
import sys

import config_render
from config_render import apply_override, render_config


def test_render_scripts_run_with_current_interpreter(tmp_path, monkeypatch):
    repo = tmp_path / "instance"
    (repo / "orchestrator" / "instance_config").mkdir(parents=True)
    (repo / "orchestrator" / "instance_config" / "instance.yaml").write_text("", encoding="utf-8")
    (repo / "deploy.env").write_text('CHABO_TAG="0.9.5"\n', encoding="utf-8")
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(config_render.subprocess, "run", fake_run)

    rendered_dir, chabo_tag = render_config(repo)

    assert chabo_tag == "0.9.5"
    assert [cmd[0] for cmd in commands] == [sys.executable, sys.executable]
    assert rendered_dir.exists()


def test_override_replaces_values_case_insensitively_and_keeps_percent(tmp_path):
    rendered = tmp_path / "params.override.cfg"
    rendered.write_text("[retriever]\nTOP_K = 5\nPATTERN = 100%\n", encoding="utf-8")
    override = tmp_path / "override.cfg"
    override.write_text("[retriever]\ntop_k = 20\n", encoding="utf-8")

    swept = apply_override(rendered, override)

    result = configparser.ConfigParser(interpolation=None)
    result.read(rendered)
    assert result.get("retriever", "top_k") == "20"
    assert result.get("retriever", "pattern") == "100%"
    assert swept == {"retriever": {"top_k": "20"}}