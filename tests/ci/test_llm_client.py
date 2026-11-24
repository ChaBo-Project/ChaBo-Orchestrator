"""
Tests for LLM client config validation.

*No network required (.env / services etc.)

Covers:
- azure provider fails fast without a valid http(s) endpoint (issue #60)
- optional per-task chat_template_kwargs reach the request body only when set (issue #71)
"""

import configparser
import pytest
import asyncio
from types import SimpleNamespace

from aiohttp import ClientResponseError
from multidict import CIMultiDict
from components.llm import LLMClient, build_llm_client


@pytest.mark.parametrize("endpoint", [None, "", "DEF", "https://"])
def test_azure_client_rejects_missing_or_invalid_endpoint(endpoint):
    with pytest.raises(ValueError, match="azure_endpoint"):
        LLMClient(provider="azure", model="m", max_tokens=8, temperature=0.0,
                  azure_endpoint=endpoint, auth_config={"api_key": "k"})


def test_azure_client_accepts_http_endpoint():
    client = LLMClient(provider="azure", model="m", max_tokens=8, temperature=0.0,
                       azure_endpoint="https://res.example/openai/v1/", auth_config={"api_key": "k"})
    assert client.chat_model.openai_api_base == "https://res.example/openai/v1/"


def test_build_llm_client_error_names_the_task_config_key():
    config = configparser.ConfigParser()
    config["query_rewriter"] = {"llm_provider": "azure", "llm_model": "m"}
    with pytest.raises(ValueError, match=r"\[query_rewriter\] llm_azure_endpoint .*QUERY_REWRITE_LLM_AZURE_ENDPOINT"):
        build_llm_client(config, "query_rewrite")


def _client(provider, **kwargs):
    return LLMClient(provider=provider, model="org/model", max_tokens=8, temperature=0.0,
                     azure_endpoint="https://res.example/openai/v1/",
                     auth_config={"api_key": "k"}, **kwargs)


THINKING_OFF = {"enable_thinking": False}


def test_chat_template_kwargs_unset_sends_nothing():
    assert _client("openai").chat_model.extra_body is None
    assert _client("azure").chat_model.extra_body == {"max_tokens": 8}
    assert "extra_body" not in _client("huggingface", inference_provider="novita").chat_model.kwargs


def test_chat_template_kwargs_reach_the_request_body():
    body = {"chat_template_kwargs": THINKING_OFF}
    assert _client("openai", chat_template_kwargs=THINKING_OFF).chat_model.extra_body == body
    assert _client("azure", chat_template_kwargs=THINKING_OFF).chat_model.extra_body == {
        "max_tokens": 8, **body}
    hf = _client("huggingface", inference_provider="novita", chat_template_kwargs=THINKING_OFF)
    assert hf.chat_model.kwargs == {"max_tokens": 8, "extra_body": body}


def test_chat_template_kwargs_rejected_for_unsupported_provider():
    with pytest.raises(ValueError, match="not supported for provider 'anthropic'"):
        _client("anthropic", chat_template_kwargs=THINKING_OFF)


def _rewrite_config(**extra):
    config = configparser.ConfigParser()
    config["query_rewriter"] = {"llm_provider": "openai", "llm_model": "m", **extra}
    return config


def test_build_llm_client_parses_chat_template_kwargs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    config = _rewrite_config(llm_chat_template_kwargs='{"enable_thinking": false}')
    assert build_llm_client(config, "query_rewrite").chat_template_kwargs == THINKING_OFF
    # Empty value = unset.
    assert build_llm_client(_rewrite_config(llm_chat_template_kwargs=""), "query_rewrite").chat_template_kwargs is None


def test_chat_template_kwargs_env_var_overrides_config(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("QUERY_REWRITE_LLM_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": true}')
    client = build_llm_client(_rewrite_config(llm_chat_template_kwargs='{"enable_thinking": false}'), "query_rewrite")
    assert client.chat_template_kwargs == {"enable_thinking": True}


@pytest.mark.parametrize("value", ["enable_thinking=false", "[1, 2]", "{bad json"])
def test_invalid_chat_template_kwargs_fail_fast_naming_the_key(monkeypatch, value):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with pytest.raises(ValueError, match=r"\[query_rewriter\] llm_chat_template_kwargs .*JSON object"):
        build_llm_client(_rewrite_config(llm_chat_template_kwargs=value), "query_rewrite")

def _stream_test_client(chat_model):
    client = object.__new__(LLMClient)
    client.chat_model = chat_model
    client.task = "generation"
    client.provider = "huggingface"
    client.inference_provider = "novita"
    client.model = "test-model"
    return client


def _rate_limit_error():
    return ClientResponseError(
        SimpleNamespace(real_url="https://example.invalid"),
        (),
        status=429,
        headers=CIMultiDict({"Retry-After": "0"}),
    )


def test_stream_retries_429_once_before_first_chunk(monkeypatch):
    delays = []

    async def no_wait(delay):
        delays.append(delay)

    monkeypatch.setattr("components.llm.llm_client.asyncio.sleep", no_wait)

    class FlakyChatModel:
        calls = 0

        async def astream(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise _rate_limit_error()
            yield SimpleNamespace(content="OK")

    chat_model = FlakyChatModel()
    client = _stream_test_client(chat_model)

    async def collect():
        return [chunk async for chunk in client.astream([])]

    assert asyncio.run(collect()) == ["OK"]
    assert chat_model.calls == 2
    assert delays == [0]


def test_stream_does_not_retry_429_after_first_chunk(monkeypatch):
    delays = []

    async def no_wait(delay):
        delays.append(delay)

    monkeypatch.setattr("components.llm.llm_client.asyncio.sleep", no_wait)

    class ChatModelFailsAfterText:
        calls = 0

        async def astream(self, messages):
            self.calls += 1
            yield SimpleNamespace(content="partial")
            raise _rate_limit_error()

    chat_model = ChatModelFailsAfterText()
    client = _stream_test_client(chat_model)

    async def check_stream():
        stream = client.astream([])
        assert await anext(stream) == "partial"
        with pytest.raises(ClientResponseError):
            await anext(stream)

    asyncio.run(check_stream())
    assert chat_model.calls == 1
    assert delays == []
