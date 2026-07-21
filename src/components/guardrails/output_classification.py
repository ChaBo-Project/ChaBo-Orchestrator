"""
Output classifier runs a safety classifier over the RAG output on a sliding window basis
to catch controversial or harmful CONTENT the keyword-based blocklist filter can't identify
(e.g. climate-change denial, partisan political stances, religion / spiritual advice, plus violence /
illegal acts / sexual content / self-harm / PII leakage).

Classification is implemented via `GuardBackend` (llm_guard.py) — either an in-context
LLM (`llm` mode) or a Qwen3Guard-Gen endpoint (`classifier` mode), same as the input guard.
The classifier only classifies the RESPONSE text (not the query).

Windowing strategy: each classification runs on the CUMULATIVE answer-so-far (all streamed
chunks of the current response concatenated)
Per request and RESPONSE-only (no query, no prior conversation turns).
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional, Set, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from .llm_guard import GuardVerdict
from ..generator.prompts import FRAMEWORK_VALUES

logger = logging.getLogger(__name__)


def build_output_classification_messages(answer_text: str, instance_guidelines: str = "") -> list:
    """
    Build [SystemMessage, HumanMessage] for classifying windowed RAG output.

    Always checks the answer against FRAMEWORK_VALUES (imported directly from
    generator/prompts.py — same source the generator itself is instructed with, so
    the two can never drift apart) plus general harmful-content risks. When
    `instance_guidelines` is non-empty, additionally asks the judge to report
    compliance with those as a SEPARATE, non-blocking-by-default signal
    (guideline_compliant/guideline_note) — never mixed into the primary verdict.
    """
    guideline_block = ""
    guideline_field = ""
    if instance_guidelines and instance_guidelines.strip():
        guideline_block = (
            "\n\nThis deployment also has its own INSTANCE GUIDELINES (separate from "
            "the Framework Values above, and not itself a safety concern):\n"
            f"{instance_guidelines.strip()}\n\n"
            "Additionally report whether the answer complies with these instance "
            "guidelines, as a separate field from the safety verdict above."
        )
        guideline_field = ', "guideline_compliant": true|false, "guideline_note": "<short note or empty>"'

    system = SystemMessage(content=(
        "You are a safety classifier reviewing the ASSISTANT ANSWER produced by a "
        "retrieval assistant. The assistant was instructed to follow these FRAMEWORK "
        "VALUES; check whether the answer breaks any of them:\n"
        f"{FRAMEWORK_VALUES}\n\n"
        "Also flag other HARMFUL content: violence, illegal acts, sexual content, "
        "self-harm, or leakage of private/personal data (PII).\n"
        "Classify ONLY the text shown; it may be a partial answer that is still "
        "streaming. Neutral, factual, on-topic answers grounded in source documents "
        "are \"safe\"."
        f"{guideline_block}\n"
        "Return ONLY a JSON object, no markdown fences, no explanation:\n"
        '{"verdict": "safe|unsafe|controversial", "category": "<short label or none>"' + guideline_field + '}\n'
        "Use \"controversial\" for a Framework Values violation, \"unsafe\" for other "
        "harmful content, and \"safe\" otherwise."
    ))
    human = HumanMessage(content=(
        f"### ASSISTANT ANSWER\n{answer_text}\n\nNow classify the answer text."
    ))
    return [system, human]


@dataclass
class OutputClassificationConfig:
    """Shared (compiled-once) config for the streaming classifier."""
    backend: Any                           # GuardBackend (llm or classifier)
    window_chars: int = 600                # new-answer chars between classifications
    notice: str = "[response withheld]"    # shown once on a block
    timeout_s: float = 5.0                  # end-of-stream: how long flush_final waits for the last verdict
    # Consequence for a guideline_compliant=False verdict — independent of `notice`
    # above, which is reserved for FRAMEWORK_VALUES/harmful-content blocks (`safe`).
    # "off" (default): ignore. "warn": log only, never blocks. "block": also stops
    # the stream (using `notice`), same as a framework violation. classifier mode
    # never sets guideline_compliant, so this has no effect there.
    guideline_enforcement: str = "off"


class StreamingClassifier:
    """
    Stateful per-request streaming classifier.

    Construct one per request from a `OutputClassificationConfig` instantiation. Drive it with
    `feed(chunk)` for each streamed answer chunk (observe-only — the chunk is never
    held back or altered), then `await flush_final()` once at the end. 
    
    On a block, `feed`/`flush_final` signals a hit so the adapter can stop the stream and emit message
    """

    def __init__(self, config: OutputClassificationConfig):
        self.cfg = config
        self.buf = ""
        self.since_last = 0
        self.pending: Set[asyncio.Task] = set()
        self.blocked = False
        self.block_category: Optional[str] = None

    # --- classification ------------------------------------------------------

    async def _classify(self, text: str) -> None:
        """
        Background task: classify `text`, set `blocked` on an unsafe verdict.

        Framework-values/harmful-content violations (`verdict.safe`) always block,
        unconditionally. Instance-guideline non-compliance is a separate, independent
        check — its consequence follows `guideline_enforcement` and never affects
        (or is affected by) the framework verdict above.
        """
        try:
            verdict: GuardVerdict = await self.cfg.backend.classify(text)
            if not verdict.safe:
                self.blocked = True
                self.block_category = verdict.category
                logger.info(
                    f"output_classification: BLOCK (severity={verdict.severity}, category={verdict.category})"
                )
            elif verdict.guideline_compliant is False and self.cfg.guideline_enforcement != "off":
                if self.cfg.guideline_enforcement == "block":
                    self.blocked = True
                    self.block_category = "guideline_noncompliance"
                    logger.info(f"output_classification: BLOCK (guideline non-compliance: {verdict.guideline_note})")
                else:  # "warn"
                    logger.warning(f"output_classification: guideline non-compliance (not blocking): {verdict.guideline_note}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"output_classification: classification failed ({type(e).__name__}: {e}) — failing open")

    def _launch(self) -> None:
        """Fire a background classification on the cumulative buffer."""
        snapshot = self.buf
        task = asyncio.create_task(self._classify(snapshot))
        self.pending.add(task)
        task.add_done_callback(self.pending.discard)

    # --- streaming driver ----------------------------------------------------

    def feed(self, chunk: str) -> bool:
        """
        Observe `chunk`; launch a window classification when enough new text has
        accumulated. 
        
        Returns the current block flag (True only if a background classification 
        has ALREADY completed with an unsafe verdict)
        """
        if self.blocked:
            return True
        if not chunk:
            return False
        self.buf += chunk
        self.since_last += len(chunk)
        if self.since_last >= self.cfg.window_chars:
            self.since_last = 0
            self._launch()
        return self.blocked

    async def flush_final(self) -> Tuple[str, bool]:
        """
        End of stream: classify any unclassified tail, then await incoming
        classifications and report the final verdict. 
        
        This is the ONLY place the classifier waits (bounded by `timeout_s`)
        So a hung call can't hang the request (an answer shorter than one window is classified only here). 
        """
        if self.blocked:
            return self.cfg.notice, True
        if self.since_last > 0 and self.buf:
            self.since_last = 0
            self._launch()
        if self.pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.pending, return_exceptions=True),
                    timeout=self.cfg.timeout_s,
                )
            except asyncio.TimeoutError:
                logger.warning("output_classification: final classification await timed out — failing open")
        if self.blocked:
            return self.cfg.notice, True
        return "", False

    async def aclose(self) -> None:
        """
        Cancel any incoming classifications (cleanup on mid-stream break).
        """
        for task in list(self.pending):
            if not task.done():
                task.cancel()
        if self.pending:
            await asyncio.gather(*self.pending, return_exceptions=True)
