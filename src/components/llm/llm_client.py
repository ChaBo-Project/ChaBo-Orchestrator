"""
LLMClient: portable, provider-agnostic LLM inference layer,
plus the per-task client factory.

"""
import logging
import os
from typing import AsyncGenerator, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.language_models import BaseChatModel  # for typing

from ..utils import get_auth_for_generator

logger = logging.getLogger(__name__)


class LLMClient:
    """A single configured LLM endpoint (reusable across modules)"""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        max_tokens: int,
        temperature: float,
        inference_provider: Optional[str] = None,
        organization: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        auth_config: Optional[dict] = None,
        streaming: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.inference_provider = inference_provider
        self.organization = organization
        self.azure_endpoint = azure_endpoint
        self.streaming = streaming

        # Resolve auth from the provider's env var unless explicitly supplied.
        self.auth_config = auth_config if auth_config is not None else get_auth_for_generator(provider)

        self.chat_model: BaseChatModel = self._build_chat_model()
        logger.info(f"LLMClient initialized with provider: {self.provider}, model: {self.model}")

    def _build_chat_model(self) -> BaseChatModel:
        """Initialize the appropriate LangChain chat model based on provider."""
        common_params = {"temperature": self.temperature, "max_tokens": self.max_tokens, "streaming": self.streaming}

        providers = {
            "openai": lambda: ChatOpenAI(model=self.model, openai_api_key=self.auth_config["api_key"], **common_params),
            "anthropic": lambda: ChatAnthropic(model=self.model, anthropic_api_key=self.auth_config["api_key"], **common_params),
            "cohere": lambda: ChatCohere(model=self.model, cohere_api_key=self.auth_config["api_key"], **common_params),
            "huggingface": lambda: self._build_hf_chat_model(**common_params),
            "azure": lambda: ChatOpenAI(model=self.model, openai_api_key=self.auth_config["api_key"], base_url=self.azure_endpoint, extra_body={"max_tokens": self.max_tokens}, temperature=self.temperature, streaming=self.streaming),
        }

        if self.provider not in providers:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return providers[self.provider]()

    def _build_hf_chat_model(self, **common_params) -> BaseChatModel:
        """Helper to configure HuggingFace model for Serverless or TGI Endpoint."""
        # Check if the MODEL config parameter looks like a URL (custom TGI endpoint)
        if self.model.startswith("http"):
            logger.info(f"Using TGI Hosted Endpoint: {self.model}")
            # --- TGI Endpoint Configuration ---
            llm = HuggingFaceEndpoint(
                endpoint_url=self.model,  # The model parameter is the TGI URL
                huggingfacehub_api_token=self.auth_config["api_key"],
                # Note: Setting the task is often unnecessary for TGI endpoints
                task="text-generation",
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                streaming=self.streaming,
            )
        else:
            logger.info(f"Using Hugging Face Serverless API for model: {self.model}")
            # --- Serverless Configuration ---
            llm = HuggingFaceEndpoint(
                repo_id=self.model,  # The model parameter is the model ID
                huggingfacehub_api_token=self.auth_config["api_key"],
                task="text-generation",
                provider=self.inference_provider,
                server_kwargs={"bill_to": self.organization},
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                streaming=self.streaming,
            )

        # ChatHuggingFace wraps the endpoint and handles chat templating
        return ChatHuggingFace(llm=llm).bind(max_tokens=self.max_tokens)

    # --- Inference calls -----------------------------------------------------

    async def ainvoke(self, messages: list) -> str:
        """Provider-agnostic LLM call using LangChain (non-streaming)."""
        try:
            logger.debug("Calling LLM without streaming")
            response = await self.chat_model.ainvoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.exception(f"LLM generation failed: {e}")
            raise

    async def astream(self, messages: list) -> AsyncGenerator[str, None]:
        """Provider-agnostic streaming LLM call using LangChain."""
        try:
            logger.debug("Calling LLM with streaming")
            async for chunk in self.chat_model.astream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.exception(f"LLM streaming failed: {e}")
            # Stream the error message out to the consumer
            yield f"Error: {str(e)}"

    # Backward-compatible aliases (legacy call sites / test doubles use these names).
    async def _call_llm(self, messages: list) -> str:
        return await self.ainvoke(messages)

    async def _call_llm_streaming(self, messages: list) -> AsyncGenerator[str, None]:
        async for chunk in self.astream(messages):
            yield chunk


# --- Per-task LLMClient setup
# Each task uses an independent LLM config with no fallback.
# !! to do: add a default llm config !!

# Lookup table to pull the correct module config
TASK_CONFIG = {
    "generation":        ("generator",        "",     "GENERATOR_"),
    "filter_extraction": ("metadata_filters", "llm_", "FILTER_EXTRACTION_LLM_"),
    "query_rewrite":     ("query_rewriter",   "llm_", "QUERY_REWRITE_LLM_"),
    "input_guard":       ("input_guard",      "llm_", "INPUT_GUARD_LLM_"),
}

# LLMClient params
_FIELDS = {
    "provider":           ("provider",           None, True),
    "model":              ("model",              None, True),
    "max_tokens":         ("max_tokens",         512,  False),
    "temperature":        ("temperature",        0.1,  False),
    "inference_provider": ("inference_provider", None, False),
    "organization":       ("organization",       None, False),
    "azure_endpoint":     ("azure_endpoint",     None, False),
}


def _resolve(config, section, option, env_var, fallback, required):
    """env var > [section] option > fallback (or error when required and absent)."""
    env_value = os.getenv(env_var)
    if env_value is not None:
        return env_value
    if config.has_section(section) and config.has_option(section, option):
        return config.get(section, option)
    if required:
        raise ValueError(
            f"LLM config missing: [{section}] {option} (or env {env_var}) is required. "
            f"Each task declares its own provider/model — there is no fallback to [generator]."
        )
    return fallback


def build_llm_client(config, task: str) -> "LLMClient":
    """Build the self-contained LLMClient for a single task. Raises if provider/model are unset."""
    if task not in TASK_CONFIG:
        raise ValueError(f"build_llm_client: unknown task {task!r}; known tasks: {list(TASK_CONFIG)}")

    section, key_prefix, env_prefix = TASK_CONFIG[task]
    resolved = {}
    for field, (option, fallback, required) in _FIELDS.items():
        opt_name = key_prefix + option
        env_var = env_prefix + option.upper()
        resolved[field] = _resolve(config, section, opt_name, env_var, fallback, required)

    resolved["max_tokens"] = int(resolved["max_tokens"])
    resolved["temperature"] = float(resolved["temperature"])

    # Resolve auth eagerly so a misconfigured task provider fails fast at construction.
    auth_config = get_auth_for_generator(resolved["provider"])

    logger.info(f"LLM client for task '{task}': provider={resolved['provider']}, model={resolved['model']}")
    return LLMClient(auth_config=auth_config, **resolved)


def build_llm_clients(config, tasks=None) -> dict:
    """Build clients for the given tasks (default: all). Build only ACTIVE tasks to avoid
    requiring config for disabled features."""
    tasks = tasks if tasks is not None else list(TASK_CONFIG)
    return {task: build_llm_client(config, task) for task in tasks}
