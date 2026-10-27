import configparser
import asyncio
import json

import httpx
import openai
import pytest

from components.llm import LLMClient, build_llm_client


@pytest.mark.parametrize("provider,model,expected", [
    ("deepinfra", "org/model", "org/model:deepinfra"),
    ("auto", "org/model", "org/model"),
    (None, "org/model", "org/model"),
    ("deepinfra", "org/model:fastest", "org/model:fastest"),
])
def test_router_selects_provider_without_sdk_registry(provider, model, expected):
    client = LLMClient(provider="huggingface_router", model=model, max_tokens=8,
                       temperature=0.0, inference_provider=provider,
                       organization="test-org", auth_config={"api_key": "test-key"})
    assert client.chat_model.model_name == expected
    assert client.chat_model.openai_api_base == "https://router.huggingface.co/v1"
    assert client.chat_model.default_headers == {"X-HF-Bill-To": "test-org"}


def test_router_factory_uses_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    monkeypatch.delenv("GENERATOR_PROVIDER", raising=False)
    monkeypatch.delenv("GENERATOR_MODEL", raising=False)
    monkeypatch.delenv("GENERATOR_ORGANIZATION", raising=False)
    config = configparser.ConfigParser()
    config["generator"] = {"provider": "huggingface_router", "model": "org/model"}
    client = build_llm_client(config, "generation")
    assert client.chat_model.openai_api_key.get_secret_value() == "test-hf-token"
    assert client.chat_model.default_headers == {}


@pytest.mark.parametrize("model", ["org/model", "https://tgi.example"])
def test_existing_huggingface_provider_keeps_its_original_path(monkeypatch, model):
    import components.llm.llm_client as module
    captured = {}

    def endpoint(**kwargs):
        captured.update(kwargs)
        return object()

    class WrappedEndpoint:
        def bind(self, **kwargs):
            return self

    monkeypatch.setattr(module, "HuggingFaceEndpoint", endpoint)
    monkeypatch.setattr(module, "ChatHuggingFace", lambda **kwargs: WrappedEndpoint())
    LLMClient(provider="huggingface", model=model, max_tokens=8, temperature=0.0,
              inference_provider="auto", auth_config={"api_key": "test-key"})
    assert captured.get("endpoint_url", captured.get("repo_id")) == model


@pytest.mark.parametrize("template", [None, {"enable_thinking": False}])
def test_router_preserves_optional_chat_template_kwargs(template):
    client = LLMClient(provider="huggingface_router", model="org/model", max_tokens=8,
                       temperature=0.0, chat_template_kwargs=template,
                       auth_config={"api_key": "test-key"})
    assert client.chat_model.extra_body == ({"chat_template_kwargs": template} if template else None)


@pytest.mark.parametrize("inference_provider,model,expected", [
    ("deepinfra", "org/model", "org/model:deepinfra"),
    ("auto", "org/model", "org/model"),
    (None, "org/model", "org/model"),
    ("deepinfra", "org/model:novita", "org/model:novita"),
])
@pytest.mark.parametrize("streaming", [False, True])
def test_router_http_request_and_answer(monkeypatch, inference_provider, model, expected, streaming):
    """Exercise the real OpenAI SDK's HTTP boundary, including SSE decoding."""
    import components.llm.llm_client as module
    original = module.ChatOpenAI
    requests = []

    def handler(request):
        requests.append(request)
        if streaming:
            chunks = [json.dumps({"id": "test", "object": "chat.completion.chunk",
                                  "choices": [{"index": 0, "delta": {"content": "OK"}, "finish_reason": None}]}),
                      json.dumps({"id": "test", "object": "chat.completion.chunk",
                                  "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}), "[DONE]"]
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
                                  content="".join("data: "+chunk+"\n\n" for chunk in chunks))
        return httpx.Response(200, json={"id": "test", "object": "chat.completion",
                                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}]})

    async def run():
        with httpx.Client(transport=httpx.MockTransport(handler)) as sync:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
                monkeypatch.setattr(module, "ChatOpenAI", lambda **kwargs: original(
                    http_client=sync, http_async_client=transport, max_retries=0, **kwargs))
                client = LLMClient(provider="huggingface_router", model=model, max_tokens=8,
                                   temperature=0.0, inference_provider=inference_provider,
                                   organization="test-org", chat_template_kwargs={"enable_thinking": False},
                                   streaming=streaming, auth_config={"api_key": "test-key"})
                if streaming:
                    return "".join([chunk async for chunk in client.astream([("user", "Reply OK")])])
                return await client.ainvoke([("user", "Reply OK")])

    assert asyncio.run(run()) == "OK"
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://router.huggingface.co/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["X-HF-Bill-To"] == "test-org"
    body = json.loads(request.content)
    assert body["model"] == expected
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["stream"] is streaming


@pytest.mark.parametrize("streaming", [False, True])
def test_router_controlled_failure_preserves_current_logging_and_errors(monkeypatch, caplog, streaming):
    import components.llm.llm_client as module
    original = module.ChatOpenAI

    def handler(request):
        return httpx.Response(429, json={"error": {"message": "Synthetic overload", "type": "rate_limit_error"}})

    async def run():
        with httpx.Client(transport=httpx.MockTransport(handler)) as sync:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
                monkeypatch.setattr(module, "ChatOpenAI", lambda **kwargs: original(
                    http_client=sync, http_async_client=transport, max_retries=0, **kwargs))
                client = LLMClient(provider="huggingface_router", model="org/model", max_tokens=8,
                                   temperature=0.0, streaming=streaming, task="generation",
                                   auth_config={"api_key": "test-key"})
                if streaming:
                    return [chunk async for chunk in client.astream([("user", "Reply OK")])]
                return await client.ainvoke([("user", "Reply OK")])

    with pytest.raises(openai.RateLimitError):
        asyncio.run(run())
    assert ("LLM streaming failed" if streaming else "LLM call failed") in caplog.text
    assert "task=generation" in caplog.text
