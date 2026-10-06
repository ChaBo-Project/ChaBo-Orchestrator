"""
Tests for LLM client config validation.

*No network required (.env / services etc.)

Covers:
- azure provider fails fast without a valid http(s) endpoint (issue #60)
"""

import configparser

import pytest

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
