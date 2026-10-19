"""
Builds the RAGAS judge LLM used to score eval cases (Step 3 - Scoring - of the RAGAS eval
plan). v1 supports JUDGE_PROVIDER = azure only.

Built directly from the provider's own SDK client via RAGAS's llm_factory, not a LangChain
wrapper - this is the one substantial part of the RAGAS 0.4 migration (the old eval.py wraps
ChaBo's own LLMClient in ragas.llms.LangchainLLMWrapper; the new metrics need a
structured-output client instead). Deliberately independent of ChaBo's own LLMClient /
[generator]: judging with the same model that generated the answer risks self-preference
bias, so this never reuses build_llm_client().
"""
import os
import sys
from typing import Any

# Add src/ to path so imports match how main.py resolves them
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from components.utils import get_auth_for_generator, get_config_value


class UnsupportedJudgeProviderError(ValueError):
    """Raised when [ragas] JUDGE_PROVIDER isn't a provider v1 supports."""


def build_judge(config) -> Any:
    """
    Build the RAGAS judge LLM from the [ragas] section of params.cfg. Environment variables
    take precedence over the file (RAGAS_JUDGE_PROVIDER, RAGAS_JUDGE_MODEL, etc.) - same
    precedence convention as the rest of ChaBo's config, via get_config_value().

    v1 only supports JUDGE_PROVIDER = azure (huggingface is deferred - see the eval plan's
    Open items, it needs a structured-output spike first since smaller open models are less
    reliable at strict JSON output).
    """
    provider = get_config_value(config, "ragas", "JUDGE_PROVIDER", "RAGAS_JUDGE_PROVIDER").strip().lower()

    if provider != "azure":
        raise UnsupportedJudgeProviderError(
            f"[ragas] JUDGE_PROVIDER={provider!r} is not supported yet - v1 only supports "
            "'azure'."
        )

    model = get_config_value(config, "ragas", "JUDGE_MODEL", "RAGAS_JUDGE_MODEL").strip()
    max_tokens = int(
        get_config_value(config, "ragas", "JUDGE_MAX_TOKENS", "RAGAS_JUDGE_MAX_TOKENS", fallback=1024)
    )
    temperature = float(
        get_config_value(config, "ragas", "JUDGE_TEMPERATURE", "RAGAS_JUDGE_TEMPERATURE", fallback=0.1)
    )
    azure_endpoint = get_config_value(config, "ragas", "AZURE_ENDPOINT", "RAGAS_AZURE_ENDPOINT")
    api_key = get_auth_for_generator("azure")["api_key"]

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    # Async client, pointed at the Azure endpoint's base_url the same way ChaBo's own
    # LLMClient does for provider="azure" (src/components/llm/llm_client.py) - Azure OpenAI
    # deployments are OpenAI-API-compatible, so no separate Azure SDK class or api_version is
    # needed. Must be the async client specifically: scoring.py calls RAGAS metrics' .ascore()
    # (async), and RAGAS's async path raises TypeError if the underlying client is sync-only
    # (confirmed via a real test run - a plain sync OpenAI client here fails at the first
    # metric score with "Cannot use agenerate() with a synchronous client").
    client = AsyncOpenAI(api_key=api_key, base_url=azure_endpoint)

    # ragas==0.4.3's llm_factory has a real bug with provider="azure": azure isn't one of
    # its explicitly-handled providers, so it falls through to a generic fallback path that
    # hardcodes Anthropic's client.messages.create shape - which an OpenAI-compatible client
    # doesn't have, and crashes with AttributeError. Passing provider="openai" instead routes
    # through llm_factory's correctly-working OpenAI path; safe here because the client above
    # already only ever speaks the OpenAI API shape.
    return llm_factory(
        model,
        provider="openai",
        client=client,
        max_tokens=max_tokens,
        temperature=temperature,
    )
