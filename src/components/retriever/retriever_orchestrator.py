import os
import json
import configparser
import logging
import asyncio
import threading
logger = logging.getLogger(__name__)

from typing import List, Dict, Any, Union, Optional, Tuple
from pydantic import Field
from qdrant_client import QdrantClient, AsyncQdrantClient
from gradio_client import Client as GradioClient
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ..utils import getconfig, get_config_value, _call_hf_endpoint, _acall_hf_endpoint
from qdrant_client.http import models as rest


# !!! Weighted RRF (rest.Rrf(weights=[...])) is only available from Qdrant server 1.17.0. !!!

# Default qdrant vector names for hybrid (we lookup the actual names from the collection)
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


# --- Sparse (BM25) encoder -------------------------------------------------
# Cache the loaded model so it isn't reloaded on every request (each query calls get_sparse_encoder via _embed_sparse)
_SPARSE_ENCODERS: Dict[Tuple[str, str], Any] = {}
_SPARSE_ENCODER_LOCK = threading.Lock()


def get_sparse_encoder(model_name: str, language: str):
    """
    Return the process-wide fastembed sparse encoder.

    Loaded eagerly at startup when `[retrieval] hybrid_enabled = true` (see `main.py`);
    otherwise loaded lazily on first call.
    """
    key = (model_name, language)
    encoder = _SPARSE_ENCODERS.get(key)
    if encoder is not None:
        return encoder

    with _SPARSE_ENCODER_LOCK:
        encoder = _SPARSE_ENCODERS.get(key)
        if encoder is None:
            try:
                from fastembed import SparseTextEmbedding
            except ImportError as e:
                raise ValueError(
                    "[retrieval] hybrid_enabled = true requires fastembed"
                    "Install with: pip install -r requirements.txt"
                ) from e
            logger.info(
                f"Loading sparse encoder '{model_name}' (language={language}). "
                f"First load downloads model assets into the fastembed cache."
            )
            encoder = SparseTextEmbedding(model_name=model_name, language=language)
            _SPARSE_ENCODERS[key] = encoder
            logger.info(f"Sparse encoder '{model_name}' ready.")
    return encoder


def _get_dense_vector_name(vectors_config: Any, collection: str) -> Optional[str]:
    """
    Get dense vector name from qdrant config

    A qdrant collection either has a single unnamed vector or a dict of named vectors. 
    A collection cannot mix an unnamed vector with named sparse ones.Returning None means 
    the unnamed default.

    Note: dense and sparse vector info is stored in different places in qdrant metadata:
    - dense: client.get_collection(self.qdrant_collection).config.params.vectors
    - sparse: client.get_collection(self.qdrant_collection).config.params.sparse_vectors
    """
    if not isinstance(vectors_config, dict):
        return None
    if not vectors_config:
        raise ValueError(f"Qdrant collection '{collection}' has no dense vectors configured.")
    if len(vectors_config) == 1:
        return next(iter(vectors_config))
    raise ValueError(
        f"Qdrant collection '{collection}' has several named dense vectors "
        f"{sorted(vectors_config)}, so ChaBo cannot tell which one to search. It expects a "
        f"collection with a single dense vector, as built by upload_parquet.py."
    )


def _build_qdrant_filter(filters: Dict[str, Any]) -> Optional[rest.Filter]:
    """
    Convert a plain {field: value} dict into a Qdrant rest.Filter.
    Payload fields are nested under 'metadata', so keys become 'metadata.<field>'.
    - list value   → MatchAny  (match any element in list)
    - scalar value → MatchValue (exact match)
    All conditions ANDed via must[]. Returns None if filters is empty or None.
    """
    if not filters:
        return None

    must_conditions = []
    for field, value in filters.items():
        qdrant_key = f"metadata.{field}"
        if isinstance(value, list):
            must_conditions.append(
                rest.FieldCondition(key=qdrant_key, match=rest.MatchAny(any=value))
            )
        else:
            must_conditions.append(
                rest.FieldCondition(key=qdrant_key, match=rest.MatchValue(value=value))
            )

    return rest.Filter(must=must_conditions)


def _format_hit(hit) -> Dict[str, Any]:
    """Format a Qdrant ScoredPoint into the standard retriever result dict."""
    return {
        "id": hit.id, # added for query rewrite eval
        "answer": hit.payload.get("text", hit.payload.get("page_content", "")),
        "answer_metadata": hit.payload.get("metadata", {}),
        "score": hit.score
    }


def _make_document(candidate: Dict[str, Any], rerank_score: Any) -> Document:
    """
    Build a LangChain Document from one search candidate

    Refactored to unify reranked and non-reranked paths (previously 2 different metadata fallbacks)
    """
    content = candidate.get("answer", candidate.get("page_content", ""))
    metadata = candidate.get("answer_metadata", candidate.get("metadata", {})).copy()
    metadata['retriever_score'] = candidate.get("score")
    metadata['rerank_score'] = rerank_score
    return Document(page_content=content, metadata=metadata)


# --- THE MAIN RETRIEVER ORCHESTRATOR CLASS  ---
class ChaBoHFEndpointRetriever(BaseRetriever):
    """
    LangChain Retriever that orchestrates three decoupled microservices:
    1. HF Endpoint for Embedding.
    2. Dynamic Qdrant search (Native or Gradio Client).
    3. HF Endpoint for Reranking.
    """
    # Configuration Fields (Used for pydantic validation and internal storage)
    hf_token: str
    embedding_endpoint_url: str
    reranker_endpoint_url: str
    
    qdrant_mode: str
    qdrant_url: str
    qdrant_api_key: Union[str, None]
    qdrant_port: int
    qdrant_collection: str
    qdrant_https: Optional[bool] = None

    top_k: int
    reranker_top_k: int
    reranker_enabled: bool = True

    # --- Hybrid retrieval (dense + BM25 sparse, fused by Qdrant's weighted RRF) ---
    hybrid_enabled: bool = False
    sparse_model: str = "Qdrant/bm25"
    sparse_language: str = "english"
    prefetch_top_k: int = 100
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    rrf_k: int = 60

    dense_vector_name: Optional[str] = Field(default=None, exclude=True)
    dense_vector_name_known: bool = Field(default=False, exclude=True)

    # We use separate caches for the sync and async clients
    sync_qdrant_client: QdrantClient = Field(default=None, exclude=True)
    async_qdrant_client: AsyncQdrantClient = Field(default=None, exclude=True)
    gradio_client: GradioClient = Field(default=None, exclude=True)


    # --- Client Lazy Initialization  ---
    @classmethod
    def from_config(cls, **kwargs) -> 'ChaBoHFEndpointRetriever':
        """Initializes the class and the appropriate Qdrant client based on qdrant_mode."""
        instance = cls(**kwargs)
        
        mode = instance.qdrant_mode.lower()
        if mode not in ['native', 'gradio']:
            logger.error(f"Unsupported qdrant_mode: {mode}. Must be 'native' or 'gradio'.")
            raise ValueError(f"Unsupported qdrant_mode: {mode}. Must be 'native' or 'gradio'.")

        if instance.hybrid_enabled:
            if mode == 'gradio':
                raise ValueError(
                    "[retrieval] hybrid_enabled = true is not supported with [qdrant] mode = gradio: "
                    "the Gradio gateway's /query_points API has no prefetch or named-vector "
                    "parameters. Use mode = native."
                )
            if instance.dense_weight < 0 or instance.sparse_weight < 0:
                raise ValueError("[retrieval] dense_weight and sparse_weight must be >= 0.")
            if instance.dense_weight <= 0 and instance.sparse_weight <= 0:
                raise ValueError("[retrieval] at least one of dense_weight / sparse_weight must be > 0.")
            if instance.rrf_k <= 0:
                raise ValueError("[retrieval] rrf_k must be > 0.")
            if instance.prefetch_top_k < instance.top_k:
                raise ValueError(
                    f"[retrieval] prefetch_top_k ({instance.prefetch_top_k}) must be >= top_k "
                    f"({instance.top_k}); a per-branch prefetch limit below the outer limit makes "
                    f"Qdrant return nothing."
                )

        # No client initialization here!
        logger.info(f"Retriever initialized for {mode} mode (Clients will be loaded lazily).")
        return instance

    # --- Shared connection settings for both Qdrant clients ---
    def _qdrant_client_kwargs(self) -> Dict[str, Any]:
        """
        Connection kwargs shared by sync and async native clients.
        """
        kwargs: Dict[str, Any] = {
            "url": self.qdrant_url,
            "port": self.qdrant_port,
            "api_key": self.qdrant_api_key or self.hf_token,
            "timeout": 60,
        }
        if self.qdrant_https is not None:
            kwargs["https"] = self.qdrant_https
        elif "://" not in self.qdrant_url:
            kwargs["https"] = (self.qdrant_port == 443)
        return kwargs

    # ---  Client Init (Synchronous) ---
    def _get_qdrant_client(self) -> Union[QdrantClient, GradioClient]:
        """Returns the appropriate synchronous client, initializing it if necessary."""
        if self.qdrant_mode.lower() == 'native':
            if not self.sync_qdrant_client:
                logger.info(f"Lazy Init: Creating Sync Native QdrantClient from  {self.qdrant_url}")
                self.sync_qdrant_client = QdrantClient(**self._qdrant_client_kwargs())
            return self.sync_qdrant_client
        
        elif self.qdrant_mode.lower() == 'gradio':
            if not self.gradio_client:
                logger.info(f"Lazy Init: Creating GradioClient from {self.qdrant_url}")
                self.gradio_client = GradioClient(
                    self.qdrant_url, 
                    hf_token=self.qdrant_api_key # Assuming Gradio uses HF_TOKEN
                )
            return self.gradio_client
        
        raise ValueError("Invalid qdrant_mode.")

    # --- Client Init (Asynchronous) ---
    async def _aget_qdrant_client(self)->Union[AsyncQdrantClient, GradioClient]:
        """Returns the appropriate asynchronous client, initializing it if necessary."""
        if self.qdrant_mode.lower() == 'native':
            if not self.async_qdrant_client:
                logger.info(f"Lazy Init: Creating Async Native QdrantClient from {self.qdrant_url}")
                self.async_qdrant_client = AsyncQdrantClient(**self._qdrant_client_kwargs())
            return self.async_qdrant_client
        
        # Gradio client object is synchronous but its predict method is inherently async
        return self._get_qdrant_client()
    
    # --- Which dense vector to search ---
    def _cache_dense_vector_name(self, vectors_config: Any) -> Optional[str]:
        """
        Get dense vector name and cache for later
        """
        self.dense_vector_name = _get_dense_vector_name(vectors_config, self.qdrant_collection)
        self.dense_vector_name_known = True
        return self.dense_vector_name

    def _resolve_dense_vector_name(self, client: QdrantClient) -> Optional[str]:
        """
        Check for existing dense vector name - if not, then get it from qdrant
        """
        if self.dense_vector_name_known:
            return self.dense_vector_name
        params = client.get_collection(self.qdrant_collection).config.params
        return self._cache_dense_vector_name(params.vectors)

    async def _aresolve_dense_vector_name(self, client: AsyncQdrantClient) -> Optional[str]:
        """
        Async version of _resolve_dense_vector_name
        """
        if self.dense_vector_name_known:
            return self.dense_vector_name
        collection = await client.get_collection(self.qdrant_collection)
        return self._cache_dense_vector_name(collection.config.params.vectors)

    # --- Sparse (BM25) query encoding ---
    def _embed_sparse(self, query: str) -> Optional[rest.SparseVector]:
        """
        BM25-encode the query for the sparse search

        Returns None when the query has no indexable terms (e.g. all stopwords).
        Request then runs dense-only.
        """
        encoder = get_sparse_encoder(self.sparse_model, self.sparse_language)
        # query_embed() returns a generator
        embedding = next(iter(encoder.query_embed(query)), None)
        if embedding is None or len(embedding.indices) == 0:
            logger.info(
                f"Sparse encoder produced no terms for query '{query[:50]}...' - "
                f"running dense-only for this request."
            )
            return None
        # convert fastembed SparseEmbedding object to qdrant friendly 
        return rest.SparseVector(
            indices=[int(i) for i in embedding.indices],
            values=[float(v) for v in embedding.values],
        )

    # --- Query construction (dense-only or hybrid fusion) ---
    def _query_kwargs(self, query_vector: List[float], sparse_vector, filters: Dict = None,
                      dense_vector_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Build the query_points() kwargs for a single native search.

        Dense-only keeps the original query shape. Hybrid issues both search branches as "prefetches"
        and lets qdrant fuse them with weighted RRF
        
        Note: metadata filters goes on each prefetch rather than on the outer request.
        """
        qdrant_filter = _build_qdrant_filter(filters)
        base: Dict[str, Any] = {
            "collection_name": self.qdrant_collection,
            "limit": self.top_k,
            "with_payload": True,
            "with_vectors": False,
        }

        if not (self.hybrid_enabled and sparse_vector is not None):
            return {
                **base,
                "query": query_vector,
                "using": dense_vector_name,
                "query_filter": qdrant_filter,
            }

        return {
            **base,
            # Prefetch limits (top k) must be at least the outer limit or Qdrant returns nothing.
            "prefetch": [
                rest.Prefetch(
                    query=query_vector,
                    using=dense_vector_name,
                    filter=qdrant_filter,
                    limit=self.prefetch_top_k,
                ),
                rest.Prefetch(
                    query=sparse_vector,
                    using=SPARSE_VECTOR_NAME,
                    filter=qdrant_filter,
                    limit=self.prefetch_top_k,
                ),
            ],
            "query": rest.RrfQuery(
                rrf=rest.Rrf(k=self.rrf_k, weights=[self.dense_weight, self.sparse_weight])
            ),
        }

    # --- Startup validation for hybrid retrieval ---
    def validate_hybrid_collection(self) -> None:
        """
        Check deployment
        """
        if not self.hybrid_enabled:
            return
        if self.qdrant_mode.lower() != 'native':
            raise ValueError("[retrieval] hybrid_enabled = true requires [qdrant] mode = native.")

        client = self._get_qdrant_client()

        params = client.get_collection(self.qdrant_collection).config.params
        sparse_config = params.sparse_vectors or {}

        if SPARSE_VECTOR_NAME not in sparse_config:
            raise ValueError(
                f"[retrieval] collection '{self.qdrant_collection}' has no sparse vector named "
                f"'{SPARSE_VECTOR_NAME}'. Available sparse vectors: {sorted(sparse_config)}. "
                f"Re-ingest with: python src/components/ingestor/upload_parquet.py --hybrid "
                f"--collection {self.qdrant_collection}"
            )

        # Raise an error if collection has several named dense vectors
        self._cache_dense_vector_name(params.vectors)

        # Check for IDF modifier: query works with TF only, but worse results
        if getattr(sparse_config[SPARSE_VECTOR_NAME], "modifier", None) != rest.Modifier.IDF:
            logger.warning(
                f"Sparse vector '{SPARSE_VECTOR_NAME}' in collection '{self.qdrant_collection}' "
                f"does not have modifier=IDF. BM25 scores will use TF only, without "
                f"IDF weighting - sparse ranking quality will be degraded. "
                f"Re-create the collection with SparseVectorParams(modifier=Modifier.IDF)."
            )

    # --- Hybrid failure hint ---
    def _log_hybrid_error_hint(self) -> None:
        """
        Annotate search failure with the most likely hybrid-specific cause: weighted-RRF query
        (rest.Rrf(weights=[...]) needs server >= 1.17.0). Only logged when hybrid is on.
        """
        if self.hybrid_enabled:
            logger.error(
                "Hybrid retrieval is enabled: check for qdrant >= 1.17.0 - required for weighted RRF "
                "(rest.Rrf(weights=[...])). Upgrade the server, or set [retrieval] hybrid_enabled = false."
            )

    # --- Qdrant Synchronous Search Helper (Handles Mode Switching) ---
    def _search_qdrant(self, query_vector: List[float], filters: Dict = None, sparse_vector=None) -> tuple:
        """Performs the synchronous Qdrant search. If mode is gradio expects
         the api_endpoint = 'query_points' similar to native mode.
         When hybrid retrieval enabled, sparse_vector triggers fused dense+sparse query.
         Returns (results, applied_filter, narrowed)."""

        try:

            client = self._get_qdrant_client()

            if self.qdrant_mode.lower() == 'native':
                logger.debug(
                    f"Sync Native Qdrant search: collection={self.qdrant_collection}, "
                    f"k={self.top_k}, hybrid={self.hybrid_enabled and sparse_vector is not None}"
                )
                applied_filter = filters
                narrowed = False
                dense_name = self._resolve_dense_vector_name(client)
                search_result = client.query_points(
                    **self._query_kwargs(query_vector, sparse_vector, filters, dense_name)
                )

                # Safeguard: if AND filter returns 0 results and multiple fields were applied,
                # retry with the priority field only (first key in filters dict).
                if not search_result.points and filters and len(filters) > 1:
                    priority_field = next(iter(filters))
                    priority_filter = {priority_field: filters[priority_field]}
                    logger.info(
                        f"AND filter returned 0 results, retrying with priority field only: {priority_filter}"
                    )
                    search_result = client.query_points(
                        **self._query_kwargs(query_vector, sparse_vector, priority_filter, dense_name)
                    )
                    applied_filter = priority_filter
                    narrowed = True

                return [_format_hit(hit) for hit in search_result.points], applied_filter, narrowed

            elif self.qdrant_mode.lower() == 'gradio':
                logger.debug(f"Sync Gradio Qdrant search: collection={self.qdrant_collection}, k={self.top_k}")
                applied_filter = filters
                narrowed = False
                result = client.predict(
                    query_vector_json=json.dumps(query_vector),
                    collection_name=self.qdrant_collection,
                    top_k=self.top_k,
                    query_filter=json.dumps(filters) if filters else None,
                    api_name="/query_points"
                )
                if isinstance(result, dict) and "error" in result:
                    logger.error(f"Gradio wrapper error: {result.get('message', result)}")
                    return [], None, False

                # Safeguard: if AND filter returns 0 results and multiple fields were applied,
                # retry with the priority field only (first key in filters dict).
                if not result and filters and len(filters) > 1:
                    priority_field = next(iter(filters))
                    priority_filter = {priority_field: filters[priority_field]}
                    logger.info(
                        f"AND filter returned 0 results (Gradio), retrying with priority field only: {priority_filter}"
                    )
                    result = client.predict(
                        query_vector_json=json.dumps(query_vector),
                        collection_name=self.qdrant_collection,
                        top_k=self.top_k,
                        query_filter=json.dumps(priority_filter),
                        api_name="/query_points"
                    )
                    if isinstance(result, dict) and "error" in result:
                        logger.error(f"Gradio wrapper error on priority retry: {result.get('message', result)}")
                        return [], None, False
                    applied_filter = priority_filter
                    narrowed = True

                return result, applied_filter, narrowed

        except Exception as e:
            logger.error(f"Search failed at {self.qdrant_url}. Error: {e}")
            self._log_hybrid_error_hint()
            return [], None, False

    # --- Qdrant Asynchronous search ---
    async def _asearch_qdrant(self, query_vector: List[float], filters: Dict = None, sparse_vector=None) -> tuple:
        """Performs the asynchronous Qdrant search. If mode is gradio expects
         the api_endpoint = 'query_points' similar to native mode.
         When hybrid retrieval is on, sparse_vector triggers the fused dense+sparse query.
         Returns (results, applied_filter, narrowed)."""
        try:
            client = await self._aget_qdrant_client()

            if self.qdrant_mode.lower() == 'native':
                logger.debug(
                    f"Async Native Qdrant search: collection={self.qdrant_collection}, "
                    f"k={self.top_k}, hybrid={self.hybrid_enabled and sparse_vector is not None}"
                )
                applied_filter = filters
                narrowed = False
                dense_name = await self._aresolve_dense_vector_name(client)

                search_result = await client.query_points(
                    **self._query_kwargs(query_vector, sparse_vector, filters, dense_name)
                )

                # Safeguard: if AND filter returns 0 results and multiple fields were applied,
                # retry with the priority field only (first key in filters dict).
                if not search_result.points and filters and len(filters) > 1:
                    priority_field = next(iter(filters))
                    priority_filter = {priority_field: filters[priority_field]}
                    logger.info(
                        f"AND filter returned 0 results, retrying with priority field only: {priority_filter}"
                    )
                    search_result = await client.query_points(
                        **self._query_kwargs(query_vector, sparse_vector, priority_filter, dense_name)
                    )
                    applied_filter = priority_filter
                    narrowed = True

                return [_format_hit(hit) for hit in search_result.points], applied_filter, narrowed

            elif self.qdrant_mode.lower() == 'gradio':
                logger.debug(f"Async Gradio Qdrant search: collection={self.qdrant_collection}, k={self.top_k}")
                applied_filter = filters
                narrowed = False
                loop = asyncio.get_running_loop()

                # Use run_in_executor to make the synchronous .predict() awaitable
                result = await loop.run_in_executor(
                    None,
                    lambda: client.predict(
                        query_vector_json=json.dumps(query_vector),
                        collection_name=self.qdrant_collection,
                        top_k=self.top_k,
                        query_filter=json.dumps(filters) if filters else None,
                        api_name="/query_points"
                    )
                )
                if isinstance(result, dict) and "error" in result:
                    logger.error(f"Gradio wrapper error: {result.get('message', result)}")
                    return [], None, False

                # Safeguard: if AND filter returns 0 results and multiple fields were applied,
                # retry with the priority field only (first key in filters dict).
                if not result and filters and len(filters) > 1:
                    priority_field = next(iter(filters))
                    priority_filter = {priority_field: filters[priority_field]}
                    logger.info(
                        f"AND filter returned 0 results (Gradio), retrying with priority field only: {priority_filter}"
                    )
                    result = await loop.run_in_executor(
                        None,
                        lambda: client.predict(
                            query_vector_json=json.dumps(query_vector),
                            collection_name=self.qdrant_collection,
                            top_k=self.top_k,
                            query_filter=json.dumps(priority_filter),
                            api_name="/query_points"
                        )
                    )
                    if isinstance(result, dict) and "error" in result:
                        logger.error(f"Gradio wrapper error on priority retry: {result.get('message', result)}")
                        return [], None, False
                    applied_filter = priority_filter
                    narrowed = True

                return result, applied_filter, narrowed

        except Exception as e:
            logger.error(f"Search failed at {self.qdrant_url}. Error: {e}")
            self._log_hybrid_error_hint()
            return [], None, False


    # --- Reranking --- (normalized refactor of previous sync/async functions)
    def _rerank_payload(self, query: str, candidate_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"query": query, "texts": [candidate["answer"] for candidate in candidate_results]}

    def _documents_from_rerank(self, candidate_results: List[Dict[str, Any]], reranked: List[Dict[str, Any]]) -> List[Document]:
        """
        Map reranker's scores back onto the candidates.

        The reranker returns positional indexes into the list of texts it was sent, so
        candidate_results must not be reordered between building the payload and this call.
        """
        return [
            _make_document(candidate_results[doc_data['index']], doc_data.get('score'))
            for doc_data in reranked[:self.reranker_top_k]
        ]

    def _documents_without_rerank(self, candidate_results: List[Dict[str, Any]], rerank_score: Any) -> List[Document]:
        """
        reranker_top_k candidates in retrieval order
        """
        return [_make_document(candidate, rerank_score) for candidate in candidate_results[:self.reranker_top_k]]

    def _rerank(self, query: str, candidate_results: List[Dict[str, Any]]) -> List[Document]:
        if not self.reranker_enabled:
            logger.info(f"Reranker disabled - returning top {self.reranker_top_k} candidates in retrieval order")
            return self._documents_without_rerank(candidate_results, None)
        try:
            logger.info(f"Performing Reranking for {len(candidate_results)}")
            reranked = _call_hf_endpoint(
                self.reranker_endpoint_url,
                self.hf_token,
                self._rerank_payload(query, candidate_results),
            )
            return self._documents_from_rerank(candidate_results, reranked)
        except Exception as e:
            # FALLBACK: If Reranker fails (503/Timeout), return top k from vector search
            logger.warning(f"NON-CRITICAL: Reranking failed ({e}). Falling back to vector search order.")
            return self._documents_without_rerank(candidate_results, "FALLBACK")

    async def _arerank(self, query: str, candidate_results: List[Dict[str, Any]]) -> List[Document]:
        if not self.reranker_enabled:
            logger.info(f"Reranker disabled - returning top {self.reranker_top_k} candidates in retrieval order")
            return self._documents_without_rerank(candidate_results, None)
        try:
            logger.info(f"Async Reranking for {len(candidate_results)} candidates")
            reranked = await _acall_hf_endpoint(
                self.reranker_endpoint_url,
                self.hf_token,
                self._rerank_payload(query, candidate_results),
            )
            return self._documents_from_rerank(candidate_results, reranked)
        except Exception as e:
            # FALLBACK: Return top k from initial vector search
            logger.warning(f"NON-CRITICAL: Async Reranking failed ({e}). Returning search results.")
            return self._documents_without_rerank(candidate_results, "FALLBACK")

    # --- Core Retrieval Orchestration (LangChain Required Method) ---
    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Executes the three-step pipeline: Embed -> Search -> Rerank.
        """
        # A. Embed Query (Call HF Endpoint 1)
        logger.info(f"Emebedding query: {query[:50]}....")
        try:

            embed_payload = {"inputs": query}
            embed_response = _call_hf_endpoint(
                self.embedding_endpoint_url,
                self.hf_token,
                embed_payload
            )
            query_vector = embed_response[0]
        except Exception as e:
            logger.error(f"CRITICAL: Embedding Failed. Details: {e}")
            return []

        sparse_vector = self._embed_sparse(query) if self.hybrid_enabled else None

        # B. Search Qdrant (Dynamic Call)
        candidate_results, _, _ = self._search_qdrant(
            query_vector, filters=kwargs.get("filters"), sparse_vector=sparse_vector
        )
        logger.debug(f"Candidate Results {candidate_results}")
        if not candidate_results:
            logger.info(f"No candidates found for query: {query[:50]}...")
            return []

        # C. Rerank Documents (Call HF Endpoint 2), with fallback in case of error
        return self._rerank(query, candidate_results)


    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        [ASYNC METHOD IMPLEMENTATION] Executes the three-step pipeline: Embed -> Search -> Rerank.
        """
        # A. Embed Query (Call HF Endpoint 1)
        logger.info(f"Emebedding query: {query[:50]}....")
        try:
            embed_payload = {"inputs": query}
            embed_response = await _acall_hf_endpoint(
                self.embedding_endpoint_url,
                self.hf_token,
                embed_payload
            )
            query_vector = embed_response[0]
        except Exception as e:
            logger.error(f"CRITICAL: Embedding Failed. Details: {e}")
            return []

        sparse_vector = await asyncio.to_thread(self._embed_sparse, query) if self.hybrid_enabled else None

        # B. Search Qdrant (Dynamic Async Call)
        candidate_results, applied_filter, narrowed = await self._asearch_qdrant(
            query_vector, filters=kwargs.get("filters"), sparse_vector=sparse_vector
        )
        logger.debug(f"Candidate Results {candidate_results}")

        if not candidate_results:
            logger.info(f"No candidates found for query: {query[:50]}...")
            return []

        # C. Rerank Documents (Call HF Endpoint 2)
        documents = await self._arerank(query, candidate_results)

        # Inject filter info into first doc so it travels with ainvoke result to retrieve_node.
        # retrieve_node pops these keys and writes them into graph state (per-request, not shared).
        if documents and applied_filter is not None:
            documents[0].metadata["_applied_filter"] = applied_filter
            documents[0].metadata["_narrowed"] = narrowed

        return documents


def create_retriever_from_config(config_file: str = "params.cfg"):
    """Loads configuration and instantiates the CustomHFRAGRetriever."""
    config = getconfig(config_file)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required but not set")

    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_api_key:
        raise ValueError("QDRANT_API_KEY environment variable is required but not set")

    config_map = {
        "embedding_endpoint_url": ("hf_endpoints", "embedding_endpoint_url", "EMBEDDING_ENDPOINT_URL"),
        "reranker_endpoint_url":  ("hf_endpoints", "reranker_endpoint_url", "RERANKER_ENDPOINT_URL"),

        "qdrant_mode":          ("qdrant", "mode", "QDRANT_MODE"),
        "qdrant_url":           ("qdrant", "url", "QDRANT_URL"),
        "qdrant_port":          ("qdrant", "port", "QDRANT_PORT"),
        "qdrant_collection":    ("qdrant", "collection", "QDRANT_COLLECTION"),
    }

    retriever_config_kwargs = {
        "hf_token": hf_token,
        "qdrant_api_key": qdrant_api_key
    }

    for key, (section, option, env_var) in config_map.items():
        value = get_config_value(config, section, option, env_var)
        if key == "qdrant_port":
            value = int(value)
        retriever_config_kwargs[key] = value

    retriever_config_kwargs.update(
        top_k=config.getint("retrieval", "top_k", fallback=20),
        reranker_top_k=config.getint("retrieval", "reranker_top_k", fallback=5),
        reranker_enabled=config.getboolean("retrieval", "reranker_enabled", fallback=True),
        hybrid_enabled=config.getboolean("retrieval", "hybrid_enabled", fallback=False),
        sparse_model=config.get("retrieval", "sparse_model", fallback="Qdrant/bm25").strip(),
        sparse_language=config.get("retrieval", "sparse_language", fallback="english").strip(),
        prefetch_top_k=config.getint("retrieval", "prefetch_top_k", fallback=100),
        dense_weight=config.getfloat("retrieval", "dense_weight", fallback=0.5),
        sparse_weight=config.getfloat("retrieval", "sparse_weight", fallback=0.5),
        rrf_k=config.getint("retrieval", "rrf_k", fallback=60),
        # None = derive TLS from the URL scheme / port (see _qdrant_client_kwargs).
        qdrant_https=config.getboolean("qdrant", "https", fallback=None),
    )

    logger.info(f"Configuration loaded. Qdrant Mode: {retriever_config_kwargs['qdrant_mode']}")
    return ChaBoHFEndpointRetriever.from_config(**retriever_config_kwargs)