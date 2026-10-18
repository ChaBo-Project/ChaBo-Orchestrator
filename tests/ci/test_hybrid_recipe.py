"""
Tests for the hybrid retrieval recipe read from Qdrant collection metadata.

*No network required (.env / services etc.)

Covers:
- a collection recipe overrides [retrieval] sparse_model / sparse_language (with a warning)
- collections without a recipe (or older clients without metadata) keep the config values
- a recipe naming a different sparse vector fails at startup
"""

import logging
from types import SimpleNamespace

import pytest
from qdrant_client.http import models as rest

from components.retriever.retriever_orchestrator import ChaBoHFEndpointRetriever

LOGGER = "components.retriever.retriever_orchestrator"


def _retriever(**overrides):
    params = dict(
        hf_token="t", embedding_endpoint_url="http://embed", reranker_endpoint_url="http://rerank",
        qdrant_mode="native", qdrant_url="http://qdrant", qdrant_api_key=None, qdrant_port=6333,
        qdrant_collection="demo", top_k=5, reranker_top_k=2, hybrid_enabled=True,
        sparse_model="Qdrant/bm25", sparse_language="english",
    )
    params.update(overrides)
    return ChaBoHFEndpointRetriever.from_config(**params)


def _collection(metadata="absent"):
    config = SimpleNamespace(params=SimpleNamespace(
        vectors={"dense": rest.VectorParams(size=4, distance=rest.Distance.COSINE)},
        sparse_vectors={"sparse": rest.SparseVectorParams(modifier=rest.Modifier.IDF)},
    ))
    if metadata != "absent":
        config.metadata = metadata
    return SimpleNamespace(config=config)


def _recipe(**sparse):
    return {"chabo_retrieval": {"version": 1, "dense": {"vector": "dense", "size": 4},
                                "sparse": {"vector": "sparse", "model": "Qdrant/bm25",
                                           "language": "english", **sparse}}}


@pytest.fixture
def with_collection(monkeypatch):
    def install(collection):
        client = SimpleNamespace(get_collection=lambda name: collection)
        monkeypatch.setattr(ChaBoHFEndpointRetriever, "_get_qdrant_client", lambda self: client)
    return install


def test_collection_recipe_overrides_config_with_a_warning(with_collection, caplog):
    with_collection(_collection(_recipe(language="german")))
    retriever = _retriever()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        retriever.validate_hybrid_collection()

    assert (retriever.sparse_model, retriever.sparse_language) == ("Qdrant/bm25", "german")
    assert "differ from collection 'demo' recipe" in caplog.text
    assert retriever.dense_vector_name == "dense"


def test_matching_recipe_logs_no_warning(with_collection, caplog):
    with_collection(_collection(_recipe()))
    retriever = _retriever()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        retriever.validate_hybrid_collection()

    assert "differ from collection" not in caplog.text
    assert "Sparse encoder settings from collection metadata" in caplog.text


@pytest.mark.parametrize("metadata", ["absent", None, {}, {"other": 1}, {"chabo_retrieval": {"sparse": {}}}])
def test_without_a_recipe_the_config_values_are_kept(with_collection, caplog, metadata):
    with_collection(_collection(metadata))
    retriever = _retriever(sparse_language="french")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        retriever.validate_hybrid_collection()

    assert (retriever.sparse_model, retriever.sparse_language) == ("Qdrant/bm25", "french")
    assert "has no retrieval recipe" in caplog.text


def test_recipe_with_a_different_sparse_vector_name_fails(with_collection):
    with_collection(_collection(_recipe(vector="bm25")))

    with pytest.raises(ValueError, match="recipe names sparse vector 'bm25'"):
        _retriever().validate_hybrid_collection()
