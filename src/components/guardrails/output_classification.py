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

logger = logging.getLogger(__name__)


def build_output_classification_messages(answer_text: str) -> list:
    """
    Build [SystemMessage, HumanMessage] for classifying windowed RAG output

    """
    system = SystemMessage(content=(
        "You are a safety classifier reviewing the ASSISTANT ANSWER produced by a "
        "retrieval assistant. Classify the answer text for two risks:\n"
        "  (1) CONTROVERSIAL content — denial or disputing of well-established "
        "scientific consensus (e.g. climate change), partisan political stances or "
        "endorsements, religious or spiritual advice / theological claims, or other "
        "divisive opinionated positions;\n"
        "  (2) HARMFUL content — violence, illegal acts, sexual content, self-harm, "
        "or leakage of private/personal data (PII).\n"
        "Classify ONLY the text shown; it may be a partial answer that is still "
        "streaming. Neutral, factual, on-topic answers grounded in source documents "
        "are \"safe\".\n"
        "Return ONLY a JSON object, no markdown fences, no explanation:\n"
        '{"verdict": "safe|unsafe|controversial", "category": "<short label or none>"}\n'
        "Use \"controversial\" for risk (1), \"unsafe\" for risk (2), and \"safe\" "
        "otherwise."
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
        
        """
        try:
            verdict: GuardVerdict = await self.cfg.backend.classify(text)
            if not verdict.safe:
                self.blocked = True
                self.block_category = verdict.category
                logger.info(
                    f"output_classification: BLOCK (severity={verdict.severity}, category={verdict.category})"
                )
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
