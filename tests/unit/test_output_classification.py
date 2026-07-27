"""
Unit tests for the streaming LLM classifier output guard. No live model.

Covers:
  - window trigger: a classification fires only after `window_chars` new chars
  - async block: a completed "unsafe"/"controversial" verdict flips `blocked`,
    and the next `feed` reports the hit (non-blocking flag check)
  - block_controversial flag: controversial blocks only when enabled
  - fail-open: classifier error never blocks; a hung mid-stream call never blocks
  - flush_final: classifies the trailing tail, and its bounded wait fails open on timeout
  - aclose: cancels in-flight classifications
  - prompt builder shape + reuse of parse_llm_guard_verdict

Run from repo root:
    python tests/unit/test_output_classification.py
"""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from components.guardrails.output_classification import (
    StreamingClassifier,
    OutputClassificationConfig,
    build_output_classification_messages,
)
from components.guardrails.llm_guard import parse_llm_guard_verdict, LLMGuardBackend

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"[PASS] {name}")
    else:
        _failed += 1
        print(f"[FAIL] {name} {detail}")


# ── fake LLM clients (ainvoke is the only method the classifier uses) ───────────────

class _Client:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return self.response


class _RaisingClient:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        raise RuntimeError("boom")


class _HangingClient:
    async def ainvoke(self, messages):
        await asyncio.sleep(5)
        return '{"verdict": "unsafe", "category": "x"}'


SAFE = '{"verdict": "safe", "category": "none"}'
UNSAFE = '{"verdict": "unsafe", "category": "violent"}'
CONTRO = '{"verdict": "controversial", "category": "politics"}'


def _cfg(client, *, block_controversial=True, **kw):
    kw.setdefault("window_chars", 10)
    kw.setdefault("notice", "[withheld]")
    kw.setdefault("timeout_s", 0.5)
    # Wrap the fake LLM client in the shared in-context backend (exercises prompt + JSON parse).
    backend = LLMGuardBackend(client, build_output_classification_messages, block_controversial=block_controversial)
    return OutputClassificationConfig(backend=backend, **kw)


async def _settle():
    # let background create_task classifications run to completion
    await asyncio.sleep(0.05)


# ── window trigger ────────────────────────────────────────────────────────────

async def _window_trigger():
    client = _Client(SAFE)
    j = StreamingClassifier(_cfg(client, window_chars=10))
    j.feed("12345")                      # 5 chars < window → no launch
    check("no classification before window", len(j.pending) == 0 and client.calls == 0)
    j.feed("678901")                     # 11 chars ≥ window → launch
    check("classification launched at window", len(j.pending) == 1)
    await _settle()
    check("window classification ran once", client.calls == 1)


# ── async block on unsafe ─────────────────────────────────────────────────────

async def _unsafe_blocks():
    client = _Client(UNSAFE)
    j = StreamingClassifier(_cfg(client, window_chars=5))
    hit_immediately = j.feed("hello world")   # triggers, but verdict not back yet
    check("feed non-blocking (no hit before verdict)", hit_immediately is False)
    await _settle()
    check("blocked flag set by background task", j.blocked is True)
    check("next feed reports the hit", j.feed("x") is True)


# ── controversial flag ────────────────────────────────────────────────────────

async def _controversial_flag():
    on = StreamingClassifier(_cfg(_Client(CONTRO), window_chars=5, block_controversial=True))
    on.feed("hello world")
    await _settle()
    check("controversial blocks when flag on", on.blocked is True)

    off = StreamingClassifier(_cfg(_Client(CONTRO), window_chars=5, block_controversial=False))
    off.feed("hello world")
    await _settle()
    check("controversial allowed when flag off", off.blocked is False)


# ── fail-open ─────────────────────────────────────────────────────────────────

async def _fail_open_error():
    client = _RaisingClient()
    j = StreamingClassifier(_cfg(client, window_chars=5))
    j.feed("hello world")
    await _settle()
    check("error fails open (not blocked)", client.calls == 1 and j.blocked is False)


async def _hung_call_never_blocks_midstream():
    # A mid-stream classification that never returns is harmless — nothing awaits it.
    j = StreamingClassifier(_cfg(_HangingClient(), window_chars=5))
    j.feed("hello world")
    await asyncio.sleep(0.1)
    check("hung mid-stream call does not block", j.blocked is False)
    await j.aclose()


async def _flush_final_timeout_fails_open():
    # The one bounded wait: if the final verdict doesn't return within timeout_s,
    # flush_final gives up and ships the answer (fail-open).
    j = StreamingClassifier(_cfg(_HangingClient(), window_chars=10_000, timeout_s=0.05))
    j.feed("short answer")
    notice, hit = await j.flush_final()
    check("flush_final timeout fails open (not blocked)", hit is False and notice == "")
    await j.aclose()


# ── flush_final ───────────────────────────────────────────────────────────────

async def _flush_final_tail_unsafe():
    # window never triggers mid-stream; the tail must still be classified at the end
    client = _Client(UNSAFE)
    j = StreamingClassifier(_cfg(client, window_chars=10_000))
    j.feed("short answer")
    check("no mid-stream classification for short answer", client.calls == 0)
    notice, hit = await j.flush_final()
    check("flush_final classifies tail → block", hit is True and notice == "[withheld]")
    check("tail classification ran", client.calls == 1)


async def _flush_final_safe():
    client = _Client(SAFE)
    j = StreamingClassifier(_cfg(client, window_chars=10_000))
    j.feed("short answer")
    notice, hit = await j.flush_final()
    check("flush_final safe → no block", hit is False and notice == "")


async def _flush_final_awaits_pending():
    # a slow-but-not-timing-out unsafe verdict launched mid-stream is awaited at flush
    class _Slow:
        calls = 0
        async def ainvoke(self, messages):
            _Slow.calls += 1
            await asyncio.sleep(0.1)
            return UNSAFE
    j = StreamingClassifier(_cfg(_Slow(), window_chars=5, timeout_s=2.0))
    j.feed("hello world")               # launches slow classification
    notice, hit = await j.flush_final()  # must await the in-flight task
    check("flush_final awaits in-flight verdict", hit is True)


# ── aclose ────────────────────────────────────────────────────────────────────

async def _aclose_cancels():
    j = StreamingClassifier(_cfg(_HangingClient(), window_chars=5))
    j.feed("hello world")
    check("task pending before aclose", len(j.pending) == 1)
    await j.aclose()
    check("no pending tasks after aclose", len(j.pending) == 0)


# ── classifier over a classifier backend ───────────────────────────────────────────

async def _classifier_over_qwen_backend():
    # The classifier is backend-agnostic: prove it blocks via a Qwen classifier backend too.
    from components.guardrails import llm_guard
    from components.guardrails.llm_guard import QwenGuardBackend

    async def _fake_call(url, token, payload):
        return {"choices": [{"message": {"content": "Safety: Unsafe\nCategories: Violent"}}]}
    saved = llm_guard._acall_hf_endpoint
    llm_guard._acall_hf_endpoint = _fake_call
    try:
        backend = QwenGuardBackend("http://guard:8000", "Qwen/Qwen3Guard-Gen-0.6B", "tok", block_controversial=True)
        j = StreamingClassifier(OutputClassificationConfig(backend=backend, window_chars=5, notice="[withheld]", timeout_s=0.5))
        j.feed("hello world")
        await _settle()
        check("classifier blocks via Qwen classifier backend", j.blocked is True)
    finally:
        llm_guard._acall_hf_endpoint = saved


# ── prompt builder + parser reuse ─────────────────────────────────────────────

def _prompt_and_parse():
    msgs = build_output_classification_messages("some answer text")
    check("prompt builder returns 2 messages", len(msgs) == 2)
    check("answer text embedded in human message", "some answer text" in msgs[1].content)
    v = parse_llm_guard_verdict(CONTRO, block_controversial=True)
    check("classifier JSON parses via reused parser", (not v.safe) and v.severity == "controversial")


if __name__ == "__main__":
    for coro in [
        _window_trigger,
        _unsafe_blocks,
        _controversial_flag,
        _fail_open_error,
        _hung_call_never_blocks_midstream,
        _flush_final_timeout_fails_open,
        _flush_final_tail_unsafe,
        _flush_final_safe,
        _flush_final_awaits_pending,
        _aclose_cancels,
        _classifier_over_qwen_backend,
    ]:
        asyncio.run(coro())
    _prompt_and_parse()
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
