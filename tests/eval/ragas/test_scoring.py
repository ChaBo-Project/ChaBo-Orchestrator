"""
Tests for scoring.score_case()/score_testset() against fake RAGAS metrics - no judge LLM or
network needed.

Covers:
- each metric is called with the argument names RAGAS's real classes expect
  (faithfulness: user_input/response/retrieved_contexts; context_recall:
  user_input/retrieved_contexts/reference; context_precision:
  user_input/reference/retrieved_contexts; factual_correctness: response/reference)
- only chunk text (not metadata) is sent to RAGAS as retrieved_contexts
- only "ok"-status cases get scored; error/no_context/blocked pass through with scores=None
- score_testset does not mutate its input
"""
import asyncio

from scoring import score_case, score_testset


class _StubResult:
    def __init__(self, value):
        self.value = value


class FakeMetric:
    def __init__(self, value):
        self._value = value
        self.calls = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResult(self._value)


def _fake_metrics():
    return {
        "faithfulness": FakeMetric(0.9),
        "context_recall": FakeMetric(0.8),
        "context_precision": FakeMetric(0.7),
        "factual_correctness": FakeMetric(0.6),
    }


def _chunks():
    return [
        {"text": "chunk one", "metadata": {"source": "a.pdf"}},
        {"text": "chunk two", "metadata": {"source": "b.pdf"}},
    ]


def test_score_case_returns_all_four_scores():
    metrics = _fake_metrics()

    scores = asyncio.run(
        score_case(metrics, question="Q?", answer="A.", expected_answer="Expected.", chunks=_chunks())
    )

    assert scores == {
        "faithfulness": 0.9,
        "context_recall": 0.8,
        "context_precision": 0.7,
        "factual_correctness": 0.6,
    }


def test_score_case_calls_each_metric_with_expected_arguments():
    metrics = _fake_metrics()

    asyncio.run(score_case(metrics, question="Q?", answer="A.", expected_answer="Expected.", chunks=_chunks()))

    contexts = ["chunk one", "chunk two"]
    assert metrics["faithfulness"].calls == [
        {"user_input": "Q?", "response": "A.", "retrieved_contexts": contexts}
    ]
    assert metrics["context_recall"].calls == [
        {"user_input": "Q?", "retrieved_contexts": contexts, "reference": "Expected."}
    ]
    assert metrics["context_precision"].calls == [
        {"user_input": "Q?", "reference": "Expected.", "retrieved_contexts": contexts}
    ]
    assert metrics["factual_correctness"].calls == [{"response": "A.", "reference": "Expected."}]


def test_score_testset_only_scores_ok_cases():
    metrics = _fake_metrics()
    results = [
        {"case_id": 1, "question": "q1", "answer": "a1", "expected_answer": "e1", "chunks": _chunks(), "status": "ok"},
        {"case_id": 2, "question": "q2", "answer": None, "expected_answer": "e2", "chunks": [], "status": "error"},
        {"case_id": 3, "question": "q3", "answer": "a3", "expected_answer": "e3", "chunks": [], "status": "no_context"},
        {"case_id": 4, "question": "q4", "answer": "blocked msg", "expected_answer": "e4", "chunks": [], "status": "blocked"},
    ]

    scored = asyncio.run(score_testset(metrics, results))

    assert [r["scores"] is not None for r in scored] == [True, False, False, False]
    # Only the one "ok" case actually invoked the metrics.
    assert len(metrics["faithfulness"].calls) == 1


def test_score_testset_does_not_mutate_input():
    metrics = _fake_metrics()
    results = [
        {"case_id": 1, "question": "q1", "answer": "a1", "expected_answer": "e1", "chunks": _chunks(), "status": "ok"},
    ]

    asyncio.run(score_testset(metrics, results))

    assert "scores" not in results[0]
