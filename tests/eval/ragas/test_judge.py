"""
Tests for judge.build_judge() - no network required, no live Azure resource needed.

llm_factory only builds client/wrapper objects at construction time; it doesn't make a real
API call, so these tests can run against a fake API key and a fake endpoint URL.

Covers:
- a non-azure JUDGE_PROVIDER is rejected (v1 only supports azure)
- a well-formed azure config builds successfully
- RAGAS_* environment variables take precedence over eval.cfg, same as the rest of ChaBo
- a missing AZURE_API_KEY env var fails loudly rather than silently
"""
import configparser

import pytest

from judge import UnsupportedJudgeProviderError, build_judge, resolve_judge_model

AZURE_CONFIG = """
[ragas]
JUDGE_PROVIDER = azure
JUDGE_MODEL = my-judge-deployment
JUDGE_MAX_TOKENS = 512
JUDGE_TEMPERATURE = 0.2
AZURE_ENDPOINT = https://fake-resource.openai.azure.com/openai/v1/
"""


def _config(text=AZURE_CONFIG) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read_string(text)
    return config


def test_rejects_unsupported_provider(monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "fake-key")
    config = _config(
        """
        [ragas]
        JUDGE_PROVIDER = openai
        JUDGE_MODEL = gpt-4o
        """
    )

    with pytest.raises(UnsupportedJudgeProviderError):
        build_judge(config)


def test_builds_azure_judge_from_config_file(monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "fake-key")
    # No RAGAS_* overrides set - values should come from the config file.
    for var in ("RAGAS_JUDGE_PROVIDER", "RAGAS_JUDGE_MODEL", "RAGAS_JUDGE_MAX_TOKENS", "RAGAS_JUDGE_TEMPERATURE"):
        monkeypatch.delenv(var, raising=False)

    llm = build_judge(_config())

    assert llm.model == "my-judge-deployment"
    assert llm.model_args["max_tokens"] == 512
    assert llm.model_args["temperature"] == 0.2


def test_env_vars_take_precedence_over_config_file(monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "fake-key")
    monkeypatch.setenv("RAGAS_JUDGE_MODEL", "overridden-deployment")
    monkeypatch.setenv("RAGAS_JUDGE_TEMPERATURE", "0.7")

    llm = build_judge(_config())

    assert llm.model == "overridden-deployment"
    assert llm.model_args["temperature"] == 0.7


@pytest.mark.parametrize("endpoint", ["DEF", "fake-resource.openai.azure.com/openai/v1/"])
def test_non_url_azure_endpoint_raises(monkeypatch, endpoint):
    monkeypatch.setenv("AZURE_API_KEY", "fake-key")
    monkeypatch.delenv("RAGAS_AZURE_ENDPOINT", raising=False)

    with pytest.raises(ValueError, match="must be an http"):
        build_judge(_config(AZURE_CONFIG.replace(
            "https://fake-resource.openai.azure.com/openai/v1/", endpoint
        )))


def test_resolve_judge_model_prefers_env_var(monkeypatch):
    monkeypatch.delenv("RAGAS_JUDGE_MODEL", raising=False)
    assert resolve_judge_model(_config()) == "my-judge-deployment"

    monkeypatch.setenv("RAGAS_JUDGE_MODEL", "overridden-deployment")
    assert resolve_judge_model(_config()) == "overridden-deployment"


def test_missing_azure_api_key_raises(monkeypatch):
    monkeypatch.delenv("AZURE_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        build_judge(_config())


def test_eval_cfg_next_to_cli_has_judge_settings():
    from run_eval import EVAL_CONFIG_PATH

    config = configparser.ConfigParser(inline_comment_prefixes=("#",))
    assert config.read(EVAL_CONFIG_PATH) == [EVAL_CONFIG_PATH]
    for key in ("JUDGE_PROVIDER", "JUDGE_MODEL", "AZURE_ENDPOINT"):
        assert config.get("ragas", key)
