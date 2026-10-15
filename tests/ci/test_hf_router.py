import configparser

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
