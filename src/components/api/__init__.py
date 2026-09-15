"""
HTTP surfaces that are not the legacy LangServe routes.

  openai_compat.py   /v1/chat/completions and /v1/models
"""
from .openai_compat import build_openai_router

__all__ = ["build_openai_router"]
