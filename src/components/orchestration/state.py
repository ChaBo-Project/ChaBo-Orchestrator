"""
Pydantic models for orchestration state and ChatUI input
"""
from typing import Optional, Dict, Any, List
from typing_extensions import TypedDict
from pydantic import BaseModel
from langchain_core.documents import Document

class GraphState(TypedDict, total=False):
    """State object passed through LangGraph workflow"""
    query: str
    context: str
    raw_documents: List[Document]  # Retrieved documents for generation
    raw_context: List[Document]    # Alias for backward compatibility
    ingestor_context: str
    result: str
    sources: List[Dict[str, str]]
    metadata: Dict[str, Any]
    conversation_context: Optional[str]  # Conversation history for multi-turn
    file_content: Optional[bytes]
    filename: Optional[str]
    file_type: Optional[str]
    workflow_type: Optional[str]  # 'standard' or 'geojson_direct'
    metadata_filters: Optional[Dict[str, Any]]
    user_messages_history: Optional[str]  # User-turn-only history for filter extraction (no assistant responses)
    applied_filters: Optional[Dict[str, Any]]  # Actual filters used after AND-safeguard fallback
    filters_narrowed: Optional[bool]  # True if AND safeguard fired and fell back to priority field
    query_rewrite: Optional[str]  # Rewritten query produced by rewrite_query_node; downstream nodes prefer it over `query`
    rewriter_fallback_used: Optional[bool]  # True if the rewriter degraded to pass-through (LLM error / empty / parse fail)
    rewriter_notes: Optional[Dict[str, Any]]  # Observability: scenarios applied, glossary terms used, detected language
    # --- input guard (parallel branch) ---
    # IMPORTANT: input_guard_node writes ONLY to these guard keys (never to any other metadata)
    # Otherwise, bad stuff happens here (duplicate writes will raise an error)
    guard_verdict: Optional[str]  # categories: "safe", "unsafe", "controversial", "unknown"
    guard_blocked: Optional[bool]  # When True, flagged as blocked_response and conditional edge will end the graph
    guard_category: Optional[str]  # e.g. "Jailbreak", "Violent", "None" (currently this is only logged - not acted upon)
    guard_mode_used: Optional[str]  # "classifier" vs. "llm"
    guard_fallback_used: Optional[bool]  # True if guard errored/timed out and fail-policy applied (we don't stop the app)
    guard_notes: Optional[Dict[str, Any]]  # Observability: raw output, latency, fallback reason

class Message(BaseModel):
    """Single message in conversation history"""
    role: str  # 'user', 'assistant', or 'system'
    content: str
    id: Optional[str] = None

class ChatUIInput(BaseModel):
    """Input model for text-only ChatUI requests"""
    messages: Optional[List[Message]] = None
    preprompt: Optional[str] = None

class ChatUIFileInput(BaseModel):
    """Input model for ChatUI requests with file attachments"""
    files: Optional[List[Dict[str, Any]]] = None
    messages: Optional[List[Message]] = None
    preprompt: Optional[str] = None
