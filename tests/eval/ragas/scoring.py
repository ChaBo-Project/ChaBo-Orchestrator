"""
Scores "ok"-status eval cases with RAGAS's v1 metrics (Step 3 - Scoring - of the RAGAS eval
plan): faithfulness, context recall, context precision, factual correctness.

Only "ok" cases are scored - a failed/blocked/no-context case is never averaged in as if it
were a real answer (see graph_runner.py's status taxonomy).
"""
import os
import sys
from typing import Any, Dict, List

# Add src/ to path so imports match how main.py resolves them
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

METRIC_NAMES = ("faithfulness", "context_recall", "context_precision", "factual_correctness")


def build_metrics(judge) -> Dict[str, Any]:
    """Build all four v1 metrics once, sharing one judge LLM - reused across every case in a
    run rather than rebuilt per case."""
    from ragas.metrics.collections import ContextPrecision, ContextRecall, FactualCorrectness, Faithfulness

    return {
        "faithfulness": Faithfulness(llm=judge),
        "context_recall": ContextRecall(llm=judge),
        "context_precision": ContextPrecision(llm=judge),
        "factual_correctness": FactualCorrectness(llm=judge),
    }


async def score_case(
    metrics: Dict[str, Any],
    question: str,
    answer: str,
    expected_answer: str,
    chunks: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Score one "ok" case against all four v1 metrics. `chunks` is the same shape
    graph_runner.run_case() returns (a list of {"text": ..., "metadata": ...} dicts) - only
    the text goes to RAGAS.
    """
    contexts = [chunk["text"] for chunk in chunks]

    faithfulness = await metrics["faithfulness"].ascore(
        user_input=question, response=answer, retrieved_contexts=contexts
    )
    context_recall = await metrics["context_recall"].ascore(
        user_input=question, retrieved_contexts=contexts, reference=expected_answer
    )
    context_precision = await metrics["context_precision"].ascore(
        user_input=question, reference=expected_answer, retrieved_contexts=contexts
    )
    factual_correctness = await metrics["factual_correctness"].ascore(
        response=answer, reference=expected_answer
    )

    return {
        "faithfulness": faithfulness.value,
        "context_recall": context_recall.value,
        "context_precision": context_precision.value,
        "factual_correctness": factual_correctness.value,
    }


async def score_testset(metrics: Dict[str, Any], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add RAGAS scores to every scorable case in a run_eval.py result list. Returns new dicts
    (does not mutate the input). A case passes through with scores=None, never averaged in,
    when either: status != "ok" (a failed/blocked/no-context case is never scored as if it
    were a real answer), or expected_answer is empty (RAGAS's reference-based metrics, e.g.
    context_recall, require a real reference to score against - cases deliberately left
    without one, like a should-be-declined question that got answered anyway, can't be
    scored even though they did get a real answer).
    """
    scored: List[Dict[str, Any]] = []
    for result in results:
        if result["status"] != "ok" or not result.get("expected_answer"):
            scored.append({**result, "scores": None})
            continue

        scores = await score_case(
            metrics,
            question=result["question"],
            answer=result["answer"],
            expected_answer=result["expected_answer"],
            chunks=result["chunks"],
        )
        scored.append({**result, "scores": scores})

    return scored
