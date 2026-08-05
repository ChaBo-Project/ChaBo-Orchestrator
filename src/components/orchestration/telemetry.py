# Helper function to extract configuration and score data from documents
from typing import Dict, Any, List
from langchain_core.documents import Document


def _config_telemetry(retriever_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Configuration fields helper
    """
    hybrid_enabled = bool(retriever_config.get("hybrid_enabled", False))
    return {
        "top_k_config": retriever_config.get("top_k"),
        "reranker_top_k_config": retriever_config.get("reranker_top_k"),
        "reranker_enabled": retriever_config.get("reranker_enabled", True),
        "hybrid_enabled": hybrid_enabled,
        "dense_weight_config": retriever_config.get("dense_weight") if hybrid_enabled else None,
        "sparse_weight_config": retriever_config.get("sparse_weight") if hybrid_enabled else None,
        "rrf_k_config": retriever_config.get("rrf_k") if hybrid_enabled else None,
    }


def extract_retriever_telemetry(docs: List[Document], retriever_config:Dict[str,Any]) -> Dict[str, Any]:
    """Extracts min/max scores and configuration metadata from the retrieved documents."""
    config_fields = _config_telemetry(retriever_config)

    if not docs:
        return {
            "total_docs_retrieved": 0,
            "min_rerank_score": None,
            "max_rerank_score": None,
            "min_retriever_score": None,
            "max_retriever_score": None,
            **config_fields,
            }

    # Assuming 'rerank_score' and 'retriever_score' are added by your orchestrator
    rerank_scores = [doc.metadata.get('rerank_score') for doc in docs if doc.metadata.get('rerank_score') is not None]
    retriever_scores = [doc.metadata.get('retriever_score') for doc in docs if doc.metadata.get('retriever_score') is not None]

    telemetry = {
        "total_docs_retrieved": len(docs),
        "min_rerank_score": min(rerank_scores) if rerank_scores else None,
        "max_rerank_score": max(rerank_scores) if rerank_scores else None,
        "min_retriever_score": min(retriever_scores) if retriever_scores else None,
        "max_retriever_score": max(retriever_scores) if retriever_scores else None,
        **config_fields,
    }
    return telemetry