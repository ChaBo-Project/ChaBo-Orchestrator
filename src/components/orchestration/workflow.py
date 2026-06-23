"""
LangGraph Workflow Setup for ChaBo RAG Orchestrator
"""
import logging
from functools import partial
from typing import Dict, Optional
from langgraph.graph import StateGraph, START, END

from .state import GraphState
from .nodes import (
    retrieve_node, generate_node_streaming, ingest_node, extract_filters_node,
    rewrite_query_node, input_guard_node, guard_gate_node, blocked_response_node,
    _route_after_guard,
)
from components.rewriter.db_context import DBContext

logger = logging.getLogger(__name__)


def build_workflow(
    retriever_instance,
    generator_instance,
    filterable_fields: Dict[str, str] = None,
    filter_values: Dict[str, list] = None,
    db_context: Optional["DBContext"] = None,
    rewriter_enabled: bool = False,
    guard_client=None,
    input_guard_enabled: bool = False,
    blocked_message: str = "I'm sorry, but I can't help with that request.",
):
    """
    Build and compile the LangGraph workflow for RAG orchestration.

    Args:
        retriever_instance: Initialised ChaBoHFEndpointRetriever
        generator_instance: Initialised Generator
        filterable_fields: Dict of {field_name: type} for LLM-based metadata filter extraction.
                           Pass {} or None to disable (extract_filters node becomes a pass-through).
        filter_values: Dict of {field_name: [valid_values]} for constrained LLM extraction.
                       Every field in filterable_fields must have an entry here.
        db_context: Optional DBContext for query rewriter. Required if rewriter_enabled=True.
        rewriter_enabled: When True, inserts rewrite_query_node between ingest and extract_filters.
                          When False, the node is omitted and downstream nodes read state['query'] directly.
        guard_client: Optional InputGuardClient for classifier/llm call. Required if input_guard_enabled=True.
        input_guard_enabled: When True, runs input_guard_node as a parallel branch off ingest,
                             joins at guard_gate, and conditionally edged to
                             blocked_response node when classifier returns 'unsafe'. 
                             When False, path defaults to main.
        blocked_message: Fixed warning for blocked_response.
    """
    if filterable_fields is None:
        filterable_fields = {}
    if filter_values is None:
        filter_values = {}

    workflow = StateGraph(GraphState)

    # Inject services into nodes
    r_node = partial(retrieve_node, retriever=retriever_instance)
    g_node = partial(generate_node_streaming, generator=generator_instance)
    f_node = partial(extract_filters_node, generator=generator_instance, filterable_fields=filterable_fields, filter_values=filter_values)

    # Add nodes
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("extract_filters", f_node)
    workflow.add_node("retrieve", r_node)
    workflow.add_node("generate", g_node)

    # Define edges
    workflow.add_edge(START, "ingest")

    # Main branch: ingest → [rewrite_query →] extract_filters
    if rewriter_enabled:
        if db_context is None:
            raise ValueError("build_workflow: rewriter_enabled=True requires a db_context")
        rw_node = partial(rewrite_query_node, generator=generator_instance, db_context=db_context)
        workflow.add_node("rewrite_query", rw_node)
        workflow.add_edge("ingest", "rewrite_query")
        workflow.add_edge("rewrite_query", "extract_filters")
    else:
        workflow.add_edge("ingest", "extract_filters")

    if input_guard_enabled:
        if guard_client is None:
            raise ValueError("build_workflow: input_guard_enabled=True requires a guard_client")
        ig_node = partial(input_guard_node, guard_client=guard_client)
        br_node = partial(blocked_response_node, blocked_message=blocked_message)
        workflow.add_node("input_guard", ig_node)
        # defer=True holds guard_gate node until both incoming branches complete (otherwise x2 generations...)
        workflow.add_node("guard_gate", guard_gate_node, defer=True)
        workflow.add_node("blocked_response", br_node)

        # Fan-out: guard runs in parallel with the main branch (both off ingest)
        workflow.add_edge("ingest", "input_guard")
        # Join: both branches converge on the barrier gate
        workflow.add_edge("input_guard", "guard_gate")
        workflow.add_edge("extract_filters", "guard_gate")
        # Conditional branch (block or go to retriever if safe)
        workflow.add_conditional_edges(
            "guard_gate",
            _route_after_guard,
            {"blocked": "blocked_response", "continue": "retrieve"},
        )
        workflow.add_edge("blocked_response", END)
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
    else:
        workflow.add_edge("extract_filters", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

    compiled_graph = workflow.compile()

    logger.info(
        f"LangGraph workflow compiled "
        f"(filterable_fields={list(filterable_fields.keys())}, "
        f"filter_values_fields={list(filter_values.keys())}, "
        f"rewriter_enabled={rewriter_enabled}, "
        f"input_guard_enabled={input_guard_enabled})"
    )
    return compiled_graph