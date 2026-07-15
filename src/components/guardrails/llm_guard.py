"""
Shared components used by input guard and the output guard classifiers:

  - guard-agnostic verdict model (`GuardVerdict`), block decision (`decide_safe`);
  - 2 verdict PARSERS — `parse_llm_guard_verdict` (portable JSON) and
    `parse_qwen_verdict` (Qwen3Guard-Gen `Safety:`/`Categories:` regex);

    NOTE: parse_qwen_verdict is model specific (tailored to qwenguard output structure)

  - 2 classification BACKENDS — `LLMGuardBackend` (in-context JSON classifier via
    any `LLMClient`) and `QwenGuardBackend` (a Qwen3Guard-Gen vLLM/SGLang endpoint) — plus
    `build_guard_backend()` to pick one from config.

Each backend exposes `async classify(text) > GuardVerdict`
The input guard wraps a single call in a timeout. The streaming output classifier sends backend calls 
on a best-effort basis and bounds only its end-of-stream await with a timeout. 
Prompts are task-specific and injected into `LLMGuardBackend` by each guard 
Qwen has its own template, so it needs no prompt.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

from components.utils import _acall_hf_endpoint

logger = logging.getLogger(__name__)

DEFAULT_CLASSIFIER_MODEL = "Qwen/Qwen3Guard-Gen-0.6B"

# Allowed classification severities: the three verdicts a classifier can return,
# plus "unknown" for the fail-open fallback. Enforced upstream by the parsers.
Severity = Literal["safe", "unsafe", "controversial", "unknown"]


@dataclass
class GuardVerdict:
    """Result of an LLM-guard classification."""
    safe: bool                 # False: block
    severity: Severity         # "safe" | "unsafe" | "controversial" | "unknown"
    category: str              # e.g. "Jailbreak", "Violent", "None" (logging only)
    fallback_used: bool = False  # True if the guard errored/timed out
    notes: Dict[str, Any] = field(default_factory=dict)


def decide_safe(severity: str, block_controversial: bool) -> bool:
    """Block on 'unsafe'; + block 'controversial' when configured."""
    sev = severity.lower()
    if sev == "unsafe":
        return False
    if sev == "controversial" and block_controversial:
        return False
    return True


# --- verdict parsers ---------------------------------------------------------

def parse_llm_guard_verdict(raw: str, block_controversial: bool) -> GuardVerdict:
    """
    Parse an `llm`-mode JSON verdict: {"verdict": "...", "category": "..."}.

    Tolerates ```json fences.
    Raises ValueError on unparseable / shapeless output (mapped to the fail-policy).

    NOTE: JSON is used because it is portable across different inference providers.
    """
    cleaned = (raw or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(cleaned)  # JSONDecodeError → caught by caller
    if not isinstance(parsed, dict):
        raise ValueError("guard verdict is not a JSON object")
    verdict = str(parsed.get("verdict", "")).lower()
    category = str(parsed.get("category") or "none")
    if verdict not in ("safe", "unsafe", "controversial"):
        raise ValueError(f"unexpected verdict value: {verdict!r}")
    return GuardVerdict(
        safe=decide_safe(verdict, block_controversial),
        severity=verdict,
        category=category,
        notes={"raw": (raw or "")[:500]},
    )


# Qwen3Guard-Gen does full decoding (no classification head), so its response is parsed
# with regex: `Safety: Safe|Unsafe|Controversial` + `Categories: ...`.
_QWEN_SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
_QWEN_CATEGORY_RE = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
    r"Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
    r"Copyright Violation|Jailbreak|None)",
    re.IGNORECASE,
)


def parse_qwen_verdict(raw: str, block_controversial: bool) -> GuardVerdict:
    """
    Parse the Qwen3Guard moderation response. Raises ValueError if the
    main classification is absent (maps to fail-policy).
    """
    m = _QWEN_SAFETY_RE.search(raw or "")
    if not m:
        raise ValueError("no classification in guard output")
    severity = m.group(1).lower()
    cat_m = _QWEN_CATEGORY_RE.search(raw or "")
    category = cat_m.group(1) if cat_m else "None"
    return GuardVerdict(
        safe=decide_safe(severity, block_controversial),
        severity=severity,
        category=category,
        notes={"raw": (raw or "")[:500]},
    )


# --- classification backends -------------------------------------------------
# Each exposes `async classify(text) > GuardVerdict` and raises on transport/parse error

class LLMGuardBackend:
    """In-context classifier: build a prompt for `text`, call an LLMClient, parse JSON."""

    def __init__(self, llm_client: Any, prompt_builder: Callable[[str], List], block_controversial: bool):
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.block_controversial = block_controversial

    async def classify(self, text: str) -> GuardVerdict:
        raw = await self.llm_client.ainvoke(self.prompt_builder(text))
        return parse_llm_guard_verdict(raw, self.block_controversial)


class QwenGuardBackend:
    """Qwen3Guard-Gen classifier over a vLLM/SGLang OpenAI-compatible endpoint.

    Classifies `text` as a single message; the input guard passes the user query, the
    output classifier passes the assistant response — the model flags the same content
    categories either way.
    """

    def __init__(self, endpoint: str, model: str, hf_token: Optional[str], block_controversial: bool):
        self.endpoint = (endpoint or "").rstrip("/")
        self.model = model or DEFAULT_CLASSIFIER_MODEL
        self.hf_token = hf_token
        self.block_controversial = block_controversial

    async def classify(self, text: str) -> GuardVerdict:
        url = f"{self.endpoint}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0.0,
            "max_tokens": 64,
        }
        resp = await _acall_hf_endpoint(url, self.hf_token or "", payload)
        raw = resp["choices"][0]["message"]["content"]
        return parse_qwen_verdict(raw, self.block_controversial)


def build_guard_backend(
    mode: str,
    *,
    block_controversial: bool,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    hf_token: Optional[str] = None,
    llm_client: Any = None,
    prompt_builder: Optional[Callable[[str], List]] = None,
):
    """Pick a backend by `mode`. Raises on a misconfigured active mode (fail fast)."""
    if mode == "classifier":
        if not (endpoint or "").strip():
            raise ValueError("guard classifier mode requires an endpoint_url")
        return QwenGuardBackend(endpoint, model or DEFAULT_CLASSIFIER_MODEL, hf_token, block_controversial)
    if mode == "llm":
        if llm_client is None or prompt_builder is None:
            raise ValueError("guard llm mode requires an llm_client and a prompt_builder")
        return LLMGuardBackend(llm_client, prompt_builder, block_controversial)
    raise ValueError(f"unknown guard mode {mode!r}; expected 'classifier' or 'llm'")
