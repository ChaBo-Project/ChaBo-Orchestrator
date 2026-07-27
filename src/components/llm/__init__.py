"""Portable LLM inference layer (provider-agnostic) and per-task client factory."""
from .llm_client import LLMClient, build_llm_client, build_llm_clients, TASK_CONFIG

__all__ = ["LLMClient", "build_llm_client", "build_llm_clients", "TASK_CONFIG"]
