"""
Tests for hybrid retrieval under [qdrant] mode = gradio, with a mocked gradio client.

*No network required (.env / services etc.)

Covers:
- hybrid off: query_points receives exactly the pre-hybrid arguments (any wrapper version)
- hybrid on: dense vector name, sparse vector and fusion settings are sent; a query without
  BM25 terms runs dense-only
- startup validation via the wrapper's collection_info: recipe applied, missing sparse
  vector and a pre-hybrid wrapper (no collection_info) fail with a clear message
"""

import asyncio
import json

import pytest
from qdrant_client.http import models as rest

from components.retriever.retriever_orchestrator import ChaBoHFEndpointRetriever

HYBRID_INFO = {
    "collection_name": "demo",
    "points_count": 3,
    "dense_vectors": {"dense": {"size": 4, "distance": "Cosine"}},
    "dense_vector_name": "dense",
    "sparse_vectors": {"sparse": {"modifier": "idf"}},
    "metadata": {"chabo_retrieval": {"version": 1, "sparse": {
        "vector": "sparse", "model": "Qdrant/bm25", "language": "german", "fastembed": "0.8.0"}}},
}
HIT = {"answer": "doc", "answer_metadata": {"filename": "a.pdf"}, "score": 0.5}
SPARSE = rest.SparseVector(indices=[3, 9], values=[0.5, 1.5])


class FakeGradioClient:
    """Records predict() calls; serves query_points and (optionally) collection_info."""

    def __init__(self, info=HYBRID_INFO, results=None):
        self.info, self.results, self.calls = info, results, []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["api_name"] == "/collection_info":
            if self.info is None:
                raise ValueError("Cannot find a function with `api_name`: /collection_info.")
            return self.info
        return self.results.pop(0) if self.results else [HIT]


def _retriever(monkeypatch, client, **overrides):
    params = dict(
        hf_token="t", embedding_endpoint_url="http://embed", reranker_endpoint_url="http://rerank",
        qdrant_mode="gradio", qdrant_url="https://qdrant.example", qdrant_api_key=None,
        qdrant_port=443, qdrant_collection="demo", top_k=5, reranker_top_k=2,
        hybrid_enabled=True, prefetch_top_k=50, dense_weight=0.3, sparse_weight=0.7, rrf_k=40,
    )
    params.update(overrides)
    monkeypatch.setattr(ChaBoHFEndpointRetriever, "_get_qdrant_client", lambda self: client)
    return ChaBoHFEndpointRetriever.from_config(**params)


def _search_calls(client):
    return [c for c in client.calls if c["api_name"] == "/query_points"]


PRE_HYBRID_KEYS = {"query_vector_json", "collection_name", "top_k", "query_filter", "api_name"}


@pytest.mark.parametrize("use_async", [False, True])
def test_hybrid_off_sends_exactly_the_pre_hybrid_arguments(monkeypatch, use_async):
    client = FakeGradioClient()
    retriever = _retriever(monkeypatch, client, hybrid_enabled=False)

    if use_async:
        results, _, _ = asyncio.run(retriever._asearch_qdrant([0.1, 0.2], filters={"country": "AFR"}))
    else:
        results, _, _ = retriever._search_qdrant([0.1, 0.2], filters={"country": "AFR"})

    assert results == [HIT]
    (call,) = _search_calls(client)
    assert set(call) == PRE_HYBRID_KEYS
    assert call["query_filter"] == json.dumps({"country": "AFR"})


@pytest.mark.parametrize("use_async", [False, True])
def test_hybrid_on_sends_sparse_vector_and_fusion_settings(monkeypatch, use_async):
    client = FakeGradioClient()
    retriever = _retriever(monkeypatch, client)
    retriever.validate_hybrid_collection()

    if use_async:
        asyncio.run(retriever._asearch_qdrant([0.1, 0.2], sparse_vector=SPARSE))
    else:
        retriever._search_qdrant([0.1, 0.2], sparse_vector=SPARSE)

    (call,) = _search_calls(client)
    assert call["vector_name"] == "dense"
    assert json.loads(call["sparse_vector_json"]) == {"indices": [3, 9], "values": [0.5, 1.5]}
    assert (call["sparse_vector_name"], call["prefetch_top_k"]) == ("sparse", 50)
    assert (call["dense_weight"], call["sparse_weight"], call["rrf_k"]) == (0.3, 0.7, 40)


def test_hybrid_query_without_bm25_terms_runs_dense_only(monkeypatch):
    client = FakeGradioClient()
    retriever = _retriever(monkeypatch, client)
    retriever.validate_hybrid_collection()

    retriever._search_qdrant([0.1, 0.2], sparse_vector=None)

    (call,) = _search_calls(client)
    assert call["vector_name"] == "dense"
    assert "sparse_vector_json" not in call


def test_filter_retry_keeps_the_hybrid_arguments(monkeypatch):
    client = FakeGradioClient(results=[[], [HIT]])
    retriever = _retriever(monkeypatch, client)
    retriever.validate_hybrid_collection()

    results, applied, narrowed = retriever._search_qdrant(
        [0.1], filters={"country": "AFR", "year": 2020}, sparse_vector=SPARSE)

    first, retry = _search_calls(client)
    assert (results, applied, narrowed) == ([HIT], {"country": "AFR"}, True)
    assert json.loads(retry["query_filter"]) == {"country": "AFR"}
    assert retry["sparse_vector_json"] == first["sparse_vector_json"]


def test_validation_reads_names_and_recipe_from_collection_info(monkeypatch):
    retriever = _retriever(monkeypatch, FakeGradioClient())

    retriever.validate_hybrid_collection()

    assert retriever.dense_vector_name == "dense"
    assert retriever.sparse_language == "german"


def test_pre_hybrid_wrapper_fails_with_a_clear_message(monkeypatch):
    retriever = _retriever(monkeypatch, FakeGradioClient(info=None))

    with pytest.raises(ValueError, match="needs the ChaBo-Deploy Qdrant Space with hybrid support"):
        retriever.validate_hybrid_collection()


def test_dense_only_collection_fails_validation(monkeypatch):
    info = {**HYBRID_INFO, "dense_vectors": None, "dense_vector_name": None,
            "sparse_vectors": {}, "metadata": {}}
    retriever = _retriever(monkeypatch, FakeGradioClient(info=info))

    with pytest.raises(ValueError, match="has no sparse vector named 'sparse'"):
        retriever.validate_hybrid_collection()


def test_collection_info_error_is_reported(monkeypatch):
    info = {"error": "Qdrant Error", "message": "Collection demo not found"}
    retriever = _retriever(monkeypatch, FakeGradioClient(info=info))

    with pytest.raises(ValueError, match="Collection demo not found"):
        retriever.validate_hybrid_collection()
