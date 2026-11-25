"""Tests for eval_config - the eval CLI's own config helpers, independent of src/."""
from pathlib import Path

import pytest

from eval_config import get_api_key, get_config_value, load_config

EVAL_DIR = Path(__file__).parent


def _config(tmp_path, text):
    path = tmp_path / "eval.cfg"
    path.write_text(text, encoding="utf-8")
    return load_config(str(path))


def test_env_var_wins_over_file(tmp_path, monkeypatch):
    config = _config(tmp_path, "[ragas]\nJUDGE_MODEL = from-file\n")
    monkeypatch.setenv("RAGAS_JUDGE_MODEL", "from-env")
    assert get_config_value(config, "ragas", "JUDGE_MODEL", "RAGAS_JUDGE_MODEL") == "from-env"


def test_file_value_and_inline_comment_stripped(tmp_path, monkeypatch):
    config = _config(tmp_path, "[ragas]\nJUDGE_MODEL = from-file  # note\n")
    monkeypatch.delenv("RAGAS_JUDGE_MODEL", raising=False)
    assert get_config_value(config, "ragas", "JUDGE_MODEL", "RAGAS_JUDGE_MODEL") == "from-file"


def test_fallback_then_error_when_missing(tmp_path, monkeypatch):
    config = _config(tmp_path, "[ragas]\n")
    monkeypatch.delenv("RAGAS_X", raising=False)
    assert get_config_value(config, "ragas", "X", "RAGAS_X", fallback=5) == 5
    with pytest.raises(ValueError, match="RAGAS_X"):
        get_config_value(config, "ragas", "X", "RAGAS_X")


def test_missing_file_gives_empty_config(tmp_path):
    assert load_config(str(tmp_path / "nope.cfg")).sections() == []


def test_get_api_key(monkeypatch):
    monkeypatch.setenv("SOME_KEY", "secret")
    assert get_api_key("SOME_KEY") == "secret"
    monkeypatch.delenv("SOME_KEY")
    with pytest.raises(RuntimeError, match="SOME_KEY"):
        get_api_key("SOME_KEY")


def test_eval_tool_does_not_import_orchestrator_src():
    for path in EVAL_DIR.glob("*.py"):
        if path.name.startswith("test_"):
            continue
        text = path.read_text(encoding="utf-8")
        assert "from components" not in text and "import components" not in text, path.name
        assert "sys.path.insert" not in text, path.name
