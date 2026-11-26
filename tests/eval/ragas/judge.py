"""
Builds the RAGAS judge LLM used to score eval cases (Step 3 - Scoring - of the RAGAS eval
plan).

Supported JUDGE_PROVIDER values - all speak the OpenAI chat API, so one AsyncOpenAI client
covers them, pointed at the right base URL with the right key:
  azure               AZURE_ENDPOINT (…/openai/v1/), key AZURE_API_KEY
  openai              api.openai.com, key OPENAI_API_KEY
  huggingface_router  router.huggingface.co/v1, key HF_TOKEN; optional INFERENCE_PROVIDER and
                      ORGANIZATION, same meaning as in the generator's [generator] section
  openai_compatible   any OpenAI-compatible server (vLLM, TGI, ...): BASE_URL, key in the env
                      var named by API_KEY_ENV (default RAGAS_JUDGE_API_KEY)

Built directly from the provider's own SDK client via RAGAS's llm_factory, not a LangChain
wrapper - the RAGAS 0.4 metrics need a structured-output client. Deliberately independent of
ChaBo's own LLMClient / [generator]: judging with the same model that generated the answer
risks self-preference bias, so this never reuses build_llm_client().

RAGAS metrics ask the judge for strict structured (JSON) output. Large models handle this
well; small open models, or inference providers without tool-call / JSON support, may fail -
those cases show up as scoring_error, so try a new judge with --limit 2 first.
"""

import os
import httpx

from typing import Any, Dict, Optional
from eval_config import get_api_key, get_config_value

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
SUPPORTED_PROVIDERS = ("azure", "openai", "huggingface_router", "openai_compatible")
OPENAI_BASE_URL = "https://api.openai.com/v1"
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class UnsupportedJudgeProviderError(ValueError):
    """Raised when [ragas] JUDGE_PROVIDER isn't a supported provider."""


def _value(config, key: str, fallback: Any = None) -> Any:
    """[ragas] <key>, overridable by the env var RAGAS_<key>."""
    return get_config_value(config, "ragas", key, f"RAGAS_{key}", fallback=fallback)


def _optional(config, key: str) -> str:
    return str(_value(config, key, fallback="")).strip()


def _require_url(config, key: str, *, allow_compatible_http: bool = False) -> httpx.URL:
    try:
        url = httpx.URL(_optional(config, key))
    except httpx.InvalidURL:
        raise ValueError(f"[ragas] {key} must be an http(s) URL.") from None

    if url.scheme not in {"https", "http"} or not url.host:
        raise ValueError(f"[ragas] {key} must be an http(s) URL.")
    if url.username or url.password:
        raise ValueError(f"[ragas] {key} must not contain credentials.")

    if url.scheme == "https":
        return url

    allowed_host = _optional(config, "ALLOW_HTTP_JUDGE_HOST").strip().lower()
    if allow_compatible_http and (
        url.host in LOOPBACK_HOSTS
        or (url.host == allowed_host and "." not in url.host and ":" not in url.host)
    ):
        return url

    raise ValueError(f"[ragas] {key} requires HTTPS for this judge host.")


def resolve_judge_model(config) -> str:
    """The judge model actually used: RAGAS_JUDGE_MODEL env var, else [ragas] JUDGE_MODEL.
    For huggingface_router with INFERENCE_PROVIDER set, ':<provider>' is appended, exactly
    as the generator does."""
    model = str(_value(config, "JUDGE_MODEL")).strip()
    provider = _optional(config, "JUDGE_PROVIDER").lower()
    inference_provider = _optional(config, "INFERENCE_PROVIDER")
    if (
        provider == "huggingface_router"
        and inference_provider
        and inference_provider != "auto"
        and ":" not in model
    ):
        model = f"{model}:{inference_provider}"
    return model


def _client_settings(config, provider: str) -> Dict[str, Any]:
    """base_url, api_key and extra headers for the judge's OpenAI-compatible client."""
    headers: Dict[str, str] = {}
    base_url: Optional[str]
    if provider == "azure":
        base_url = str(_require_url(config, "AZURE_ENDPOINT"))
        api_key = get_api_key("AZURE_API_KEY")
    elif provider == "openai":
        base_url = OPENAI_BASE_URL
        api_key = get_api_key("OPENAI_API_KEY")
    elif provider == "huggingface_router":
        base_url = HF_ROUTER_BASE_URL
        api_key = get_api_key("HF_TOKEN")
        organization = _optional(config, "ORGANIZATION")
        if organization:
            headers["X-HF-Bill-To"] = organization
    else:  # openai_compatible
        url = _require_url(config, "BASE_URL", allow_compatible_http=True)
        base_url = str(url)
        key_env = _optional(config, "API_KEY_ENV") or "RAGAS_JUDGE_API_KEY"

        if url.scheme == "http" and url.host not in LOOPBACK_HOSTS:
            if os.getenv(key_env):
                raise ValueError("[ragas] Use HTTPS when a Docker judge API key is configured.")
            api_key = "no-auth"  # Nonsecret placeholder required by the SDK.
        else:
            api_key = get_api_key(key_env)
    return {"base_url": base_url, "api_key": api_key, "default_headers": headers or None}


def build_judge(config) -> Any:
    """
    Build the RAGAS judge LLM from the [ragas] section of eval.cfg. Environment variables
    take precedence over the file (RAGAS_JUDGE_PROVIDER, RAGAS_JUDGE_MODEL, etc.) - same
    precedence convention as the rest of ChaBo's config.
    """
    provider = _optional(config, "JUDGE_PROVIDER").lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise UnsupportedJudgeProviderError(
            f"[ragas] JUDGE_PROVIDER={provider!r} is not supported - use one of "
            f"{', '.join(SUPPORTED_PROVIDERS)}."
        )

    model = resolve_judge_model(config)
    max_tokens = int(_value(config, "JUDGE_MAX_TOKENS", fallback=1024))
    temperature = float(_value(config, "JUDGE_TEMPERATURE", fallback=0.1))
    settings = _client_settings(config, provider)

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    # Must be the async client specifically: scoring.py calls RAGAS metrics' .ascore()
    # (async), and RAGAS's async path raises TypeError if the underlying client is sync-only
    # ("Cannot use agenerate() with a synchronous client").
    client = AsyncOpenAI(**settings)

    # provider="openai" for every judge provider: they all speak the OpenAI API shape, and
    # ragas==0.4.3's llm_factory mishandles provider="azure" (falls through to a generic path
    # that calls Anthropic's client.messages.create and crashes with AttributeError).
    return llm_factory(
        model,
        provider="openai",
        client=client,
        max_tokens=max_tokens,
        temperature=temperature,
    )
