"""
LangGraph orchestration nodes for retrieval and generation

NEEDS TO BE UPDATED
"""
import logging
logger = logging.getLogger(__name__)
from datetime import datetime
import json
from typing import TYPE_CHECKING, Dict, Any, Optional
from langchain_core.documents import Document
from .telemetry import extract_retriever_telemetry
from components.ingestor.ingestor import process_document
from components.generator.prompts import build_filter_extraction_messages, build_query_rewrite_messages

# Assuming these Type definitions are available from state.py and retriever_orchestrator.py
if TYPE_CHECKING:
    from components.retriever.retriever_orchestrator import ChaBoHFEndpointRetriever
    from components.generator.generator_orchestrator import Generator
    from components.llm import LLMClient
    from components.orchestration.state import GraphState
    from components.rewriter.db_context import DBContext



async def retrieve_node(
    state: 'GraphState',
    retriever: 'ChaBoHFEndpointRetriever', # Injected LangChain BaseRetriever instance
    *,
    writer
    ) -> 'GraphState':
    """
    Node to retrieve relevant context using the ChaBoHFEndpointRetriever.
    The retriever performs Embed -> Search -> Rerank in one async call.
    Emits a 'filters_applied' custom event so adapters can surface the footnote to ChatUI.
    """

    start_time = datetime.now()

    # 1. Extract Query and Filters
    filters = state.get("metadata_filters")
    metadata = state.get("metadata", {})
    # Prefer rewritten query when present; falls back to raw query when rewriter is disabled or pass-through.
    query = state.get("query_rewrite") or state["query"]
    logger.info(f"Retrieval: {query[:50]}...")

    raw_documents: list[Document] = []

    try:
        retriever_kwargs = {}
        if filters:
            retriever_kwargs['filters'] = filters

        raw_documents = await retriever.ainvoke(
            input=query,
            **retriever_kwargs
        )

        # Extract filter info injected by the retriever into the first doc's metadata.
        # Using doc metadata (not retriever instance state) avoids race conditions under concurrent requests.
        applied_filter = None
        narrowed = False
        if raw_documents:
            applied_filter = raw_documents[0].metadata.pop("_applied_filter", None)
            narrowed = raw_documents[0].metadata.pop("_narrowed", False)

        if applied_filter:
            writer({"event": "filters_applied", "data": {"filters": applied_filter, "narrowed": narrowed}})

        duration = (datetime.now() - start_time).total_seconds()
        retriever_config = {
            "top_k": retriever.top_k,
            "reranker_top_k": retriever.reranker_top_k,
            "qdrant_mode": retriever.qdrant_mode,
            "reranker_enabled": retriever.reranker_enabled,
            "hybrid_enabled": retriever.hybrid_enabled,
            "dense_weight": retriever.dense_weight,
            "sparse_weight": retriever.sparse_weight,
            "rrf_k": retriever.rrf_k,
        }

        retriever_telemetry = extract_retriever_telemetry(raw_documents, retriever_config)

        metadata.update({
            "retrieval_duration": duration,
            "filters_applied": json.dumps(filters) if filters else "None",
            "retriever_config": retriever_telemetry,
            "retrieval_success": True
        })
        return {
            "raw_documents": raw_documents,
            "metadata": metadata,
            "applied_filters": applied_filter,
            "filters_narrowed": narrowed,
        }

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Retrieval failed: {str(e)}", exc_info=True)

        metadata.update({
            "retrieval_duration": duration,
            "retrieval_success": False,
            "retrieval_error": str(e)
        })

        return {"raw_documents": [], "metadata": metadata}
    

async def generate_node_streaming(state: "GraphState", generator: "Generator", *, writer):
    """
    Node to generate the final response with StreamWriter for LangGraph custom streaming.
    Uses StreamWriter to emit events that LangGraph can capture with stream_mode="custom".
    """
    start_time = datetime.now()

    query = state.get("query")
    raw_docs = state.get("raw_documents", [])
    metadata = state.get("metadata", {})
    ingestor_context = state.get("ingestor_context")

    # If we have ingestor_context, prepend it to raw_docs as a Document
    if ingestor_context:
        ingestor_doc = Document(
            page_content=ingestor_context,
            metadata={"source": "uploaded_file", "filename": state.get("filename", "unknown")}
        )
        raw_docs = [ingestor_doc] + raw_docs
        logger.info(f"Including ingestor context ({len(ingestor_context)} chars) with retrieved docs")

    accumulated_text = ""
    logger.info(f"Generation: {query[:50]}... ({len(raw_docs)} docs)")
    conversation_context = state.get("conversation_context")

    try:
        async for event in generator.generate_streaming(
            query=query,
            context=raw_docs,
            chatui_format=True,
            conversation_context=conversation_context
        ):
            # Track content to calculate metadata (length) at the end
            if event.get("event") == "data":
                accumulated_text += event.get("data", "")

            # Use StreamWriter to emit custom events
            writer(event)

        # Final Telemetry Update
        duration = (datetime.now() - start_time).total_seconds()
        metadata.update({
            "generation_duration": duration,
            "generation_success": True,
            "response_length": len(accumulated_text)
        })

        logger.info(f"Streaming complete in {duration:.2f}s. Length: {len(accumulated_text)}")

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Generation node failed: {e}", exc_info=True)
        metadata.update({
            "generation_duration": duration,
            "generation_success": False,
            "generation_error": str(e)
        })
        writer({"event": "error", "data": {"error": str(e)}})


async def ingest_node(state: 'GraphState') -> 'GraphState':
    """
    Node to process uploaded documents (PDF, DOCX) and extract chunked context.
    Only runs if file_content and filename are present in state.
    """
    start_time = datetime.now()

    file_content = state.get("file_content")
    filename = state.get("filename")
    metadata = state.get("metadata", {})

    # Skip if no file uploaded
    if not file_content or not filename:
        logger.info("No file to ingest, skipping ingest_node")
        return {}

    logger.info(f"Ingesting document: {filename}")

    try:
        # Process document and get chunked context
        ingestor_context = process_document(file_content, filename)

        duration = (datetime.now() - start_time).total_seconds()

        metadata.update({
            "ingest_duration": duration,
            "ingest_success": True,
            "ingested_filename": filename,
            "ingestor_context_length": len(ingestor_context)
        })

        logger.info(f"Document ingested successfully in {duration:.2f}s")

        return {
            "ingestor_context": ingestor_context,
            "metadata": metadata
        }

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Document ingestion failed: {str(e)}", exc_info=True)

        metadata.update({
            "ingest_duration": duration,
            "ingest_success": False,
            "ingest_error": str(e)
        })

        return {"ingestor_context": "", "metadata": metadata}


def _parse_filter_response(
    raw_response: str,
    filterable_fields: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """
    Parse and validate an LLM filter extraction response.
    Returns a validated {field: cast_value} dict, or None if parsing fails or result is empty.
    """
    try:
        cleaned = raw_response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        extracted = json.loads(cleaned)
        if not isinstance(extracted, dict):
            return None
    except (json.JSONDecodeError, ValueError):
        return None

    filters = {}
    for key, value in extracted.items():
        if key not in filterable_fields:
            continue
        declared_type = filterable_fields[key]
        try:
            if declared_type == "list":
                filters[key] = [str(v) for v in value] if isinstance(value, list) else [str(value)]
            elif declared_type == "int":
                filters[key] = int(value)
            else:
                filters[key] = str(value)
        except (TypeError, ValueError):
            pass  # drop uncastable values silently

    return filters if filters else None


async def extract_filters_node(
    state: "GraphState",
    llm_client: "LLMClient",
    filterable_fields: Dict[str, str],
    filter_values: Dict[str, list],
) -> "GraphState":
    """
    Node to extract metadata filters from query + user conversation history before retrieval.
    No-op if filterable_fields is empty. Fails gracefully — retrieval proceeds unfiltered on any error.

    Single-pass strategy: one LLM call with both current query and user history.
    For each field independently: extracts from current query first; if not found,
    carries forward from user history (user turns only — no assistant responses or
    retrieved document content). This ensures all applicable fields are extracted
    regardless of which source contains them.
    """
    if not filterable_fields:
        logger.info("extract_filters_node: no filterable_fields configured, skipping")
        return {}

    # Prefer rewritten query when present; falls back to raw query when rewriter is disabled or pass-through.
    query = state.get("query_rewrite") or state.get("query", "")
    user_messages_history = state.get("user_messages_history") or "(none)"

    try:
        messages = build_filter_extraction_messages(filterable_fields, filter_values, query, user_messages_history)
        raw = await llm_client.ainvoke(messages)
        filters = _parse_filter_response(raw, filterable_fields)
    except Exception as e:
        logger.warning(f"extract_filters_node: LLM call failed ({e}). Proceeding without filters.")
        return {"metadata_filters": None}

    if filters:
        logger.info(f"extract_filters_node: extracted filters: {filters}")
    else:
        logger.info("extract_filters_node: no filters found in query or history")

    return {"metadata_filters": filters}


def _parse_rewrite_response(raw_response: str) -> Optional[Dict[str, Any]]:
    """
    Parse and validate LLM query-rewrite response.

    Expected schema:
        {"query_rewrite": "<string>", "notes": {...}}

    Returns the dict with a non-empty 'query_rewrite' string, or None on any
    parse / shape failure. Caller falls back to pass-through on None.
    """
    try:
        cleaned = raw_response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            return None
    except (json.JSONDecodeError, ValueError):
        return None

    rewritten = parsed.get("query_rewrite")
    if not isinstance(rewritten, str) or not rewritten.strip():
        return None

    notes = parsed.get("notes")
    if not isinstance(notes, dict):
        notes = {}

    return {"query_rewrite": rewritten.strip(), "notes": notes}


async def rewrite_query_node(
    state: "GraphState",
    llm_client: "LLMClient",
    db_context: "DBContext",
    *,
    writer,
) -> "GraphState":
    """
    Node to rewrite user query.

    Single LLM call for rewriting backed by db_context
    On failure falls back to pass-through:
    `query_rewrite == query` and `rewriter_fallback_used = True`. 

    Emits a 'query_rewritten' custom event for observability.
    """
    start_time = datetime.now()
    metadata = state.get("metadata", {})

    original_query = state.get("query", "") or ""
    conversation_context = state.get("conversation_context")

    # Empty/trivial query results in pass-through.
    if not original_query.strip():
        logger.info("rewrite_query_node: empty query, skipping")
        return {
            "query_rewrite": original_query,
            "rewriter_fallback_used": True,
            "rewriter_notes": {"reason": "empty_query"},
        }

    try:
        messages = build_query_rewrite_messages(db_context, original_query, conversation_context)
        raw = await llm_client.ainvoke(messages)
        parsed = _parse_rewrite_response(raw)
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.warning(f"rewrite_query_node: LLM call failed ({e}); falling back to original query")
        metadata.update({
            "rewrite_duration": duration,
            "rewrite_success": False,
            "rewrite_error": str(e),
        })
        return {
            "query_rewrite": original_query,
            "rewriter_fallback_used": True,
            "rewriter_notes": {"reason": "llm_error"},
            "metadata": metadata,
        }

    duration = (datetime.now() - start_time).total_seconds()

    if parsed is None:
        logger.warning("rewrite_query_node: parse failed; falling back to original query")
        metadata.update({
            "rewrite_duration": duration,
            "rewrite_success": False,
            "rewrite_error": "parse_failed",
        })
        return {
            "query_rewrite": original_query,
            "rewriter_fallback_used": True,
            "rewriter_notes": {"reason": "parse_failed"},
            "metadata": metadata,
        }

    rewritten = parsed["query_rewrite"]
    notes = parsed["notes"]

    logger.info(
        f"rewrite_query_node: rewrote in {duration:.2f}s | "
        f"orig: {original_query[:60]!r} → rewrite: {rewritten[:60]!r}"
    )

    metadata.update({
        "rewrite_duration": duration,
        "rewrite_success": True,
    })

    # Observability event
    try:
        writer({
            "event": "query_rewritten",
            "data": {
                "original": original_query,
                "rewrite": rewritten,
                "notes": notes,
            },
        })
    except Exception:
        # writer may not be available in some test contexts; non-fatal.
        pass

    return {
        "query_rewrite": rewritten,
        "rewriter_fallback_used": False,
        "rewriter_notes": notes,
        "metadata": metadata,
    }


async def input_guard_node(
    state: "GraphState",
    guard_client,  # InputGuardClient (injected)
    *,
    writer,
) -> "GraphState":
    """
    Classify raw user query for prompt-injection/jailbreak + harmful content.

    When 'unsafe' set guard_blocked=True which triggers blocked_response via conditional edge.
    """
    start_time = datetime.now()
    query = state.get("query", "") or ""

    verdict = await guard_client.classify(query)
    duration = (datetime.now() - start_time).total_seconds()

    notes = dict(verdict.notes)
    notes.update({"duration": duration, "mode": guard_client.mode})

    logger.info(
        f"input_guard_node: severity={verdict.severity} category={verdict.category} "
        f"blocked={not verdict.safe} fallback={verdict.fallback_used} ({duration:.2f}s)"
    )

    # Observability
    try:
        writer({
            "event": "guard_checked",
            "data": {
                "severity": verdict.severity,
                "category": verdict.category,
                "blocked": not verdict.safe,
                "fallback_used": verdict.fallback_used,
            },
        })
    except Exception:
        pass  # writer may be unavailable in some test contexts

    return {
        "guard_verdict": verdict.severity,
        "guard_blocked": not verdict.safe,
        "guard_category": verdict.category,
        "guard_mode_used": guard_client.mode,
        "guard_fallback_used": verdict.fallback_used,
        "guard_notes": notes,
    }


async def guard_gate_node(state: "GraphState") -> "GraphState":
    """
    Both the guard branch and the main branch (rewrite_query → extract_filters) flow into this node,
    so LangGraph runs it only after BOTH complete. 
    It's just a pass-through for the main branch - unless the conditional edge reads guard_blocked from state.
    """
    return {}


def _route_after_guard(state):
    """
    Conditional router: block if the input was flagged.
    """
    return "blocked" if state.get("guard_blocked") else "continue"


async def blocked_response_node(
    state: "GraphState",
    blocked_message: str,
    *,
    writer,
) -> "GraphState":
    """
    Return static warning for a blocked input and end the pipeline.

    NOTE: Chatui friendly - same streaming event shape as generate_node_streaming so the
    existing adapters render it with no special handling.
    """
    logger.info(
        f"blocked_response_node: input blocked "
        f"(category={state.get('guard_category')}, severity={state.get('guard_verdict')})"
    )
    writer({"event": "data", "data": blocked_message})
    writer({"event": "final_answer", "data": {"text": blocked_message, "webSources": []}})
    writer({"event": "end", "data": {}})
    return {}

