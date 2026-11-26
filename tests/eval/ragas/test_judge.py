"""
Tests for judge.build_judge() - no network required, no live Azure resource needed.

llm_factory only builds client/wrapper objects at construction time; it doesn't make a real
API call, so these tests can run against a fake API key and a fake endpoint URL.

Covers:
- an unknown JUDGE_PROVIDER is rejected
- openai, huggingface_router and openai_compatible judges get the right URL, key and model
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
    monkeypatch.delenv("RAGAS_JUDGE_PROVIDER", raising=False)
    config = _config(
        """
        [ragas]
        JUDGE_PROVIDER = cohere
        JUDGE_MODEL = command-r
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


def _client(llm):
    """The AsyncOpenAI client inside RAGAS's instructor wrapper."""
    return llm.client.client


@pytest.fixture
def no_ragas_env(monkeypatch):
    for var in ("RAGAS_JUDGE_PROVIDER", "RAGAS_JUDGE_MODEL", "RAGAS_INFERENCE_PROVIDER",
                "RAGAS_ORGANIZATION", "RAGAS_BASE_URL", "RAGAS_API_KEY_ENV"):
        monkeypatch.delenv(var, raising=False)


def test_openai_judge(monkeypatch, no_ragas_env):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    llm = build_judge(_config("[ragas]\nJUDGE_PROVIDER = openai\nJUDGE_MODEL = gpt-4.1\n"))
    assert llm.model == "gpt-4.1"
    assert str(_client(llm).base_url).startswith("https://api.openai.com/")


def test_huggingface_router_judge(monkeypatch, no_ragas_env):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    config = _config(
        "[ragas]\nJUDGE_PROVIDER = huggingface_router\nJUDGE_MODEL = Qwen/Qwen3-235B\n"
        "INFERENCE_PROVIDER = novita\nORGANIZATION = my-org\n"
    )
    llm = build_judge(config)

    assert llm.model == "Qwen/Qwen3-235B:novita"
    assert resolve_judge_model(config) == "Qwen/Qwen3-235B:novita"
    assert str(_client(llm).base_url) == "https://router.huggingface.co/v1/"
    assert _client(llm).api_key == "fake-token"
    assert _client(llm)._custom_headers.get("X-HF-Bill-To") == "my-org"


def test_huggingface_router_auto_provider_keeps_model(monkeypatch, no_ragas_env):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    config = _config(
        "[ragas]\nJUDGE_PROVIDER = huggingface_router\nJUDGE_MODEL = Qwen/Qwen3-235B\n"
        "INFERENCE_PROVIDER = auto\n"
    )
    assert build_judge(config).model == "Qwen/Qwen3-235B"


def test_huggingface_router_missing_token_raises(monkeypatch, no_ragas_env):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        build_judge(_config("[ragas]\nJUDGE_PROVIDER = huggingface_router\nJUDGE_MODEL = m\n"))


def test_openai_compatible_judge(monkeypatch, no_ragas_env):
    monkeypatch.setenv("MY_VLLM_KEY", "fake-key")
    llm = build_judge(_config(
        "[ragas]\nJUDGE_PROVIDER = openai_compatible\nJUDGE_MODEL = qwen3\n"
        "BASE_URL = https://vllm.example.org/v1\nAPI_KEY_ENV = MY_VLLM_KEY\n"
    ))
    assert str(_client(llm).base_url) == "https://vllm.example.org/v1/"
    assert _client(llm).api_key == "fake-key"


def test_openai_compatible_requires_base_url(monkeypatch, no_ragas_env):
    monkeypatch.setenv("RAGAS_JUDGE_API_KEY", "fake-key")
    with pytest.raises(ValueError, match="BASE_URL"):
        build_judge(_config("[ragas]\nJUDGE_PROVIDER = openai_compatible\nJUDGE_MODEL = m\n"))

@pytest.mark.parametrize(
    ("provider", "url_setting", "key_setting"),
    [
        ("azure", "RAGAS_AZURE_ENDPOINT", "AZURE_API_KEY"),
        ("openai_compatible", "RAGAS_BASE_URL", "RAGAS_JUDGE_API_KEY"),
    ],
)
def test_remote_judge_requires_https(
    monkeypatch, no_ragas_env, provider, url_setting, key_setting
):
    monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", provider)
    monkeypatch.setenv("RAGAS_JUDGE_MODEL", "test-model")
    monkeypatch.setenv(url_setting, "http://judge.example.invalid/v1")
    monkeypatch.setenv(key_setting, "dummy-test-key")

    with pytest.raises(ValueError, match="HTTPS"):
        build_judge(_config())


def test_openai_judge_uses_official_url(monkeypatch, no_ragas_env):
    monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("RAGAS_JUDGE_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://other.example.invalid/v1")

    client = _client(build_judge(_config()))

    assert str(client.base_url).rstrip("/") == "https://api.openai.com/v1"

@pytest.mark.parametrize(
    ("host", "api_key", "error"),
    [
        ("judge", None, None),
        ("other-judge", None, "requires HTTPS"),
        ("judge", "dummy-test-key", "Use HTTPS"),
    ],
)
def test_private_docker_http_judge(monkeypatch, no_ragas_env, host, api_key, error):
    monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "openai_compatible")
    monkeypatch.setenv("RAGAS_JUDGE_MODEL", "test-model")
    monkeypatch.setenv("RAGAS_BASE_URL", f"http://{host}:8000/v1")
    monkeypatch.setenv("RAGAS_ALLOW_HTTP_JUDGE_HOST", "judge")
    monkeypatch.delenv("RAGAS_JUDGE_API_KEY", raising=False)
    if api_key is not None:
        monkeypatch.setenv("RAGAS_JUDGE_API_KEY", api_key)

    if error is not None:
        with pytest.raises(ValueError, match=error):
            build_judge(_config())
    else:
        client = _client(build_judge(_config()))
        assert str(client.base_url).rstrip("/") == "http://judge:8000/v1"
        assert client.api_key == "no-auth"
