import configparser
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_missing_override_file_raises(tmp_path):
    rendered = tmp_path / "params.override.cfg"
    rendered.write_text("[retriever]\nTOP_K = 5\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        apply_override(rendered, tmp_path / "typo.cfg")


def test_render_config_accepts_relative_instance_repo(tmp_path, monkeypatch):
    repo = tmp_path / "inst"
    config_dir = repo / "orchestrator" / "instance_config"
    config_dir.mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "deploy.env").write_text('CHABO_TAG="v1"\n', encoding="utf-8")
    (config_dir / "instance.yaml").write_text("x: 1\n", encoding="utf-8")
    (config_dir / "params.override.cfg.template").write_text("[a]\nb = 1\n", encoding="utf-8")
    render = "import sys, shutil; shutil.copy(sys.argv[1], sys.argv[2])"
    (repo / "scripts" / "render-config.py").write_text(render, encoding="utf-8")
    (repo / "scripts" / "validate-config.py").write_text("", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    rendered_dir, tag = render_config(Path("inst"))

    assert tag == "v1"
    assert (rendered_dir / "params.override.cfg").exists()
    shutil.rmtree(rendered_dir)

def test_render_failure_removes_temp_dir(tmp_path, monkeypatch):
    repo = tmp_path / "instance"
    (repo / "orchestrator" / "instance_config").mkdir(parents=True)
    (repo / "orchestrator" / "instance_config" / "instance.yaml").write_text("", encoding="utf-8")
    (repo / "deploy.env").write_text('CHABO_TAG="0.9.5"\n', encoding="utf-8")
    created = []
    real_mkdtemp = config_render.tempfile.mkdtemp

    def tracking_mkdtemp(**kwargs):
        created.append(real_mkdtemp(dir=tmp_path, **kwargs))
        return created[-1]

    monkeypatch.setattr(config_render.tempfile, "mkdtemp", tracking_mkdtemp)
    monkeypatch.setattr(
        config_render.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "validation failed"),
    )

    with pytest.raises(config_render.ConfigRenderError):
        render_config(repo)
    assert created and not Path(created[0]).exists()


def test_parse_env_file_handles_export_quotes_and_comments(tmp_path):
    env_file = tmp_path / "deploy.env"
    env_file.write_text(
        "# comment line\n"
        'export CHABO_TAG="0.9.5"  # pinned release\n'
        "PLAIN=value  # trailing comment\n"
        "SINGLE='a # not a comment'\n"
        "URL=https://example.org/#anchor\n"
        "EMPTY=\n",
        encoding="utf-8",
    )

    values = config_render._parse_env_file(env_file)

    assert values == {
        "CHABO_TAG": "0.9.5",
        "PLAIN": "value",
        "SINGLE": "a # not a comment",
        "URL": "https://example.org/#anchor",
        "EMPTY": "",
    }
