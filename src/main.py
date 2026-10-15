"""
ChaBo RAG Orchestrator - Production Entry Point
"""
import os
import logging
from fastapi import FastAPI, Header, HTTPException
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
import uvicorn
from functools import partial

from components.retriever.retriever_orchestrator import (
    create_retriever_from_config, get_sparse_encoder, SPARSE_VECTOR_NAME,
)
from components.generator.generator_orchestrator import Generator
from components.llm import build_llm_client
from components.orchestration.workflow import build_workflow
from components.orchestration.ui_adapters import chatui_adapter
from components.orchestration.state import ChatUIInput, ChatUIFileInput
from components.api import build_openai_router
from components.utils import getconfig, instance_config_dir, load_prompt_overrides
from components.retriever.filters import FILTER_VALUES, validate_filterable_fields
from components.rewriter.db_context import DBContext, load_db_context_from_instance
from components.generator.prompts import load_instance_guidelines
from components.guardrails.input_guard import InputGuardClient
from components.guardrails.output_guard import load_compiled_blocklist
from components.guardrails.output_classification import OutputClassificationConfig, build_output_classification_messages
from components.guardrails.llm_guard import build_guard_backend, DEFAULT_CLASSIFIER_MODEL
from typing import Dict, List, Any, Optional

config = getconfig("params.cfg")
MAX_TURNS = config.getint("conversation_history", "MAX_TURNS", fallback=3)
MAX_CHARS = config.getint("conversation_history", "MAX_CHARS", fallback=8000)

# --- Frontend-agnostic API surface ([api], all optional) ---
# Model id advertised by GET /v1/models and echoed back in completions.
API_MODEL_NAME = config.get("api", "model_name", fallback="chabo").strip() or "chabo"
# Non-standard `citations` array on OpenAI-compatible responses
# (sources are always rendered as markdown in the message body regardless).
API_CITATIONS = config.getboolean("api", "openai_citations", fallback=True)

# Parse filterable_fields: "field:type,field:type" → {"field": "type"}
_filterable_fields_raw = config.get("metadata_filters", "filterable_fields", fallback="")
FILTERABLE_FIELDS: Dict[str, str] = {}
for _item in _filterable_fields_raw.split(","):
    _item = _item.strip()
    if ":" in _item:
        _name, _ftype = _item.split(":", 1)
        FILTERABLE_FIELDS[_name.strip()] = _ftype.strip()
    elif _item:
        FILTERABLE_FIELDS[_item] = "str"  # default to str if no type declared

if FILTERABLE_FIELDS:
    validate_filterable_fields(FILTERABLE_FIELDS)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_instance_dir = instance_config_dir()
logger.info(
    f"Instance config: {_instance_dir}" if _instance_dir
    else "Instance config: none (INSTANCE_CONFIG_DIR unset — running with generic defaults)"
)

# Query rewriter config (optional — defaults to disabled if section is absent)
REWRITER_ENABLED = config.getboolean("query_rewriter", "enabled", fallback=False)

db_context: DBContext = DBContext() # instantiating a typed object here for potential future use (e.g. with metadata filter node)
if REWRITER_ENABLED:
    db_context = load_db_context_from_instance()
    if not db_context.abstract.strip():
        logger.warning(
            "Query rewriter: db_context.abstract is empty - running in conservative mode "
            "(no glossary-based expansion). Populate INSTANCE_CONFIG_DIR/instance.yaml's "
            "db_context key to ground the rewriter.")

# Initialize services
logger.info("Initializing ChaBoHFEndpointRetriever and per-task LLM clients...")
retriever_instance = create_retriever_from_config(config_file="params.cfg")

# Hybrid retrieval
if retriever_instance.hybrid_enabled:
    retriever_instance.validate_hybrid_collection()  # raises ValueError with remediation text
    get_sparse_encoder(retriever_instance.sparse_model, retriever_instance.sparse_language)
    logger.info(
        "Hybrid retrieval ENABLED - dense='%s' sparse='%s' weights=%.2f/%.2f rrf_k=%d model=%s/%s",
        retriever_instance.dense_vector_name or "<unnamed default>",
        SPARSE_VECTOR_NAME,
        retriever_instance.dense_weight, retriever_instance.sparse_weight,
        retriever_instance.rrf_k,
        retriever_instance.sparse_model, retriever_instance.sparse_language,
    )
else:
    logger.info("Hybrid retrieval disabled - dense-only retrieval.")

if not retriever_instance.reranker_enabled:
    logger.warning("Reranker DISABLED - returning top reranker_top_k candidates in retrieval order.")

# Instance-specific system prompt guidelines (optional — empty/absent = none, framework
# defaults + base prompt only). See FRAMEWORK_VALUES in components/generator/prompts.py
# for the non-negotiable, code-only defaults this can never override.
INSTANCE_GUIDELINES = load_instance_guidelines()

# Tier-1 prompt overrides (optional — resolved once at startup, not per-request; see
# load_prompt_overrides() docstring). Each node gets its own section via partial() below.
PROMPT_OVERRIDES = load_prompt_overrides()

# Get module-specific LLM configs
generation_client = build_llm_client(config, "generation")
generator_instance = Generator(llm_client=generation_client, instance_guidelines=INSTANCE_GUIDELINES)

filter_extraction_client = build_llm_client(config, "filter_extraction") if FILTERABLE_FIELDS else None
query_rewrite_client = build_llm_client(config, "query_rewrite") if REWRITER_ENABLED else None

# Get input guard config 
INPUT_GUARD_ENABLED = config.getboolean("input_guard", "enabled", fallback=False)
INPUT_GUARD_MODE = config.get("input_guard", "mode", fallback="llm").strip()
INPUT_GUARD_ENDPOINT = config.get("input_guard", "endpoint_url", fallback="").strip()
INPUT_GUARD_MODEL = config.get("input_guard", "model", fallback="Qwen/Qwen3Guard-Gen-0.6B").strip()
INPUT_GUARD_BLOCK_CTRL = config.getboolean("input_guard", "block_controversial", fallback=False)
INPUT_GUARD_TIMEOUT = config.getfloat("input_guard", "timeout_s", fallback=2.0)
INPUT_GUARD_MSG = config.get(
    "input_guard", "blocked_message", fallback="I'm sorry, but I can't help with that request."
).strip()

guard_client = None
if INPUT_GUARD_ENABLED:
    if INPUT_GUARD_MODE == "classifier" and not INPUT_GUARD_ENDPOINT:
        raise ValueError("[input_guard] mode=classifier requires endpoint_url to be set.")
    # Get module-specific LLM config
    # Only for llm mode - classifier mode uses a dedicated model (ref. config)
    input_guard_client = build_llm_client(config, "input_guard") if INPUT_GUARD_MODE == "llm" else None
    guard_client = InputGuardClient(
        mode=INPUT_GUARD_MODE,
        classifier_endpoint=INPUT_GUARD_ENDPOINT,
        classifier_model=INPUT_GUARD_MODEL,
        hf_token=os.getenv("HF_TOKEN"),
        llm_client=input_guard_client,
        timeout_s=INPUT_GUARD_TIMEOUT,
        block_controversial=INPUT_GUARD_BLOCK_CTRL,
    )
    logger.info(f"Input guard enabled (mode={INPUT_GUARD_MODE})")

# Get output guard config — two independent elements: Classification (classifier or LLM) + keyword blocklist
OUTPUT_CLASSIFICATION_ENABLED = config.getboolean("output_guard", "classification_enabled", fallback=False)
OUTPUT_CLASSIFICATION_MODE = config.get("output_guard", "mode", fallback="llm").strip()
OUTPUT_CLASSIFICATION_BLOCK_CTRL = config.getboolean("output_guard", "block_controversial", fallback=True)
# Consequence for instance-guideline non-compliance (independent of framework-values
# blocking above, which is always enforced). Only meaningful under mode=llm — the
# classifier backend never evaluates instance guidelines at all.
GUIDELINE_ENFORCEMENT = config.get("output_guard", "guideline_enforcement", fallback="off").strip().lower()
if GUIDELINE_ENFORCEMENT not in ("off", "warn", "block"):
    raise ValueError(
        f"[output_guard] guideline_enforcement must be 'off', 'warn', or 'block', got {GUIDELINE_ENFORCEMENT!r}"
    )
output_classification_config = None
if OUTPUT_CLASSIFICATION_ENABLED:
    if OUTPUT_CLASSIFICATION_MODE == "classifier":
        classification_endpoint = config.get("output_guard", "endpoint_url", fallback="").strip()
        if not classification_endpoint:
            raise ValueError("[output_guard] mode=classifier requires endpoint_url to be set.")
        classification_backend = build_guard_backend(
            "classifier",
            block_controversial=OUTPUT_CLASSIFICATION_BLOCK_CTRL,
            endpoint=classification_endpoint,
            model=config.get("output_guard", "model", fallback=DEFAULT_CLASSIFIER_MODEL).strip(),
            hf_token=os.getenv("HF_TOKEN"),
        )
    else:  # llm
        classification_backend = build_guard_backend(
            "llm",
            block_controversial=OUTPUT_CLASSIFICATION_BLOCK_CTRL,
            llm_client=build_llm_client(config, "output_classification"),
            prompt_builder=partial(build_output_classification_messages, instance_guidelines=INSTANCE_GUIDELINES),
        )
    output_classification_config = OutputClassificationConfig(
        backend=classification_backend,
        window_chars=config.getint("output_guard", "window_chars", fallback=600),
        notice=config.get("output_guard", "classification_message", fallback="[response withheld]"),
        timeout_s=config.getfloat("output_guard", "timeout_s", fallback=5.0),
        guideline_enforcement=GUIDELINE_ENFORCEMENT,
    )
    logger.info(f"Output classifier enabled (mode={OUTPUT_CLASSIFICATION_MODE}, window_chars={output_classification_config.window_chars})")

# --- Keyword blocklist ---
# Base list ships in the image (generic, open-source-sourced — see blocklist.txt's own
# header). INSTANCE_CONFIG_DIR/instance.yaml's `blocklist` key adds instance-specific
# terms on top, per language — additions only, no removal.
BLOCKLIST_ENABLED = config.getboolean("output_guard", "blocklist_enabled", fallback=False)
BLOCKLIST_MESSAGE = config.get("output_guard", "blocklist_message", fallback="[response withheld]")
blocklist = None
if BLOCKLIST_ENABLED:
    blocklist_path = config.get(
        "output_guard", "blocklist_path", fallback="src/components/guardrails/blocklist.txt"
    )
    blocklist = load_compiled_blocklist(blocklist_path)

# Build the LangGraph workflow
compiled_graph = build_workflow(
    retriever_instance,
    generator_instance,
    config,
    filterable_fields=FILTERABLE_FIELDS,
    filter_values=FILTER_VALUES,
    db_context=db_context if REWRITER_ENABLED else None,
    rewriter_enabled=REWRITER_ENABLED,
    guard_client=guard_client,
    input_guard_enabled=INPUT_GUARD_ENABLED,
    blocked_message=INPUT_GUARD_MSG,
    filter_extraction_client=filter_extraction_client,
    query_rewrite_client=query_rewrite_client,
    filter_extraction_steps=PROMPT_OVERRIDES.get("filter_extraction_steps"),
    query_rewrite_steps=PROMPT_OVERRIDES.get("query_rewrite_steps"),
)


#----------------------------------------
# FASTAPI SETUP
#----------------------------------------

app = FastAPI(title="ChaBo RAG Orchestrator", version="1.0.0")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {
        "message": "ChaBo RAG Orchestrator API",
        "endpoints": {
            "health": "/health",
            "chat_completions": "/v1/chat/completions (OpenAI-compatible)",
            "models": "/v1/models (OpenAI-compatible)",
            "chatfed-ui-stream": "/chatfed-ui-stream (LangServe)",
            "chatfed-with-file-stream": "/chatfed-with-file-stream (LangServe)",
        }
    }


#----------------------------------------
# EVAL-ONLY ROUTE (internal, token-gated)
#----------------------------------------
# Lets the RAGAS eval tooling (tests/eval/ragas/, a separate client with its own
# dependencies - see that folder) ask the real compiled_graph a question over HTTP and get
# back the retrieved context + internal status that the public chat routes never expose.
# Gated by EVAL_API_KEY so this doesn't leak corpus content to anyone who can reach the
# instance - 404 (not 401) when unset, so the route's existence isn't revealed on instances
# that haven't opted in.
EVAL_API_KEY = os.getenv("EVAL_API_KEY")


class EvalQueryRequest(BaseModel):
    question: str


class EvalContext(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}


class EvalQueryResponse(BaseModel):
    answer: Optional[str] = None
    retrieved_contexts: List[EvalContext] = []
    status: str  # "ok" | "error" | "no_context" | "blocked"
    error: Optional[str] = None


@app.post("/eval-query", response_model=EvalQueryResponse, include_in_schema=False)
async def eval_query(request: EvalQueryRequest, x_eval_token: Optional[str] = Header(default=None)):
    """
    Run one question through the real compiled_graph and return the answer, retrieved
    context, and a status taxonomy - status is exactly one of:
    - "ok": an answer was produced from at least one retrieved chunk.
    - "error": the run raised, the graph emitted an error event, retrieval explicitly
      failed, or no answer came back at all.
    - "no_context": retrieval succeeded but returned zero chunks.
    - "blocked": the input guard blocked the question before retrieval ran.

    Listens on both the "custom" channel (carries the citation-cleaned final answer, same
    text a user sees) and the "values" channel (carries the final graph state - the only
    place raw_documents and internal flags like retrieval_success/guard_blocked live;
    production's process_query_streaming never needs them, so it only requests "custom").
    """
    if not EVAL_API_KEY or x_eval_token != EVAL_API_KEY:
        raise HTTPException(status_code=404)

    # Mirrors process_query_streaming's initial_state block (streaming.py) - keep both in
    # sync if that shape changes. No conversation history: eval questions are standalone.
    initial_state = {
        "query": request.question,
        "metadata": {"session_type": "eval"},
        "raw_documents": [],
        "conversation_context": None,
        "metadata_filters": None,
        "user_messages_history": None,
    }

    answer: Optional[str] = None
    final_state: Dict[str, Any] = {}
    error_message: Optional[str] = None
    saw_error_event = False

    try:
        async for mode, payload in compiled_graph.astream(initial_state, stream_mode=["custom", "values"]):
            if mode == "custom":
                event = payload.get("event")
                if event == "final_answer":
                    answer = payload["data"].get("text")
                elif event == "error":
                    saw_error_event = True
                    error_message = payload.get("data", {}).get("error")
            elif mode == "values":
                final_state = payload
    except Exception as exc:
        logger.error(f"eval_query pipeline error: {exc}", exc_info=True)
        return EvalQueryResponse(status="error", error=str(exc))

    chunks = [
        EvalContext(text=doc.page_content, metadata=doc.metadata)
        for doc in final_state.get("raw_documents", [])
    ]

    # Absent (not False) whenever retrieve_node never ran, e.g. a guard-blocked question -
    # only an explicit False means retrieval itself failed.
    metadata = final_state.get("metadata", {})
    retrieval_success = metadata.get("retrieval_success")

    if final_state.get("guard_blocked"):
        status = "blocked"
    elif saw_error_event:
        status = "error"
        error_message = error_message or "graph emitted an error event"
    elif retrieval_success is False:
        status = "error"
        error_message = error_message or metadata.get("retrieval_error") or "retrieval failed"
    elif not answer:
        status = "error"
        error_message = error_message or metadata.get("generation_error") or "no answer produced"
    elif not chunks:
        status = "no_context"
    else:
        status = "ok"

    return EvalQueryResponse(answer=answer, retrieved_contexts=chunks, status=status, error=error_message)


#----------------------------------------
# FRONTEND-AGNOSTIC ROUTES
#----------------------------------------
# /v1/chat/completions is standard for many frontend UIs (OpenWebUI, LibreChat).
# Attachments are provided as already-extracted text (`files`), because generic
# frontends do their own extraction and send raw text.
app.include_router(
    build_openai_router(
        compiled_graph,
        config=config,
        model_name=API_MODEL_NAME,
        max_turns=MAX_TURNS,
        max_chars=MAX_CHARS,
        blocklist=blocklist,
        blocklist_notice=BLOCKLIST_MESSAGE,
        classification_config=output_classification_config,
        include_citations=API_CITATIONS,
    )
)
logger.info(
    f"OpenAI-compatible API enabled: /v1/chat/completions, /v1/models (model id "
    f"'{API_MODEL_NAME}', citations={'on' if API_CITATIONS else 'off'})"
)


#----------------------------------------
# LANGSERVE ROUTES (ChatUI)
#----------------------------------------
# ChatUI-specific: Kept for the existing ChatUI deployments.
# New frontends should use /v1/chat/completions above.

# Inject compiled_graph and config into the adapter (blocklist=None when the blocklist is disabled).
# Same underlying adapter for both routes below — file handling is a no-op when `files` is absent.
text_adapter = partial(chatui_adapter, compiled_graph=compiled_graph, max_turns=MAX_TURNS, max_chars=MAX_CHARS,
                       blocklist=blocklist, blocklist_notice=BLOCKLIST_MESSAGE, classification_config=output_classification_config)
file_adapter = partial(chatui_adapter, compiled_graph=compiled_graph, max_turns=MAX_TURNS, max_chars=MAX_CHARS,
                       blocklist=blocklist, blocklist_notice=BLOCKLIST_MESSAGE, classification_config=output_classification_config)

# Text-only endpoint
add_routes(
    app,
    RunnableLambda(text_adapter),
    path="/chatfed-ui-stream",
    input_type=ChatUIInput,
    output_type=str,
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
)

# File upload endpoint
add_routes(
    app,
    RunnableLambda(file_adapter),
    path="/chatfed-with-file-stream",
    input_type=ChatUIFileInput,
    output_type=str,
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
)

logger.debug(f"ChatUIInput schema: {ChatUIInput.model_json_schema()}")
logger.debug(f"ChatUIFileInput schema: {ChatUIFileInput.model_json_schema()}")

#----------------------------------------

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))

    logger.info(f"Starting ChaBo RAG Orchestrator on {host}:{port}")
    logger.info(f"API Docs: http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port, log_level="info", access_log=True)