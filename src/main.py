"""
ChaBo RAG Orchestrator - Production Entry Point
"""
import os
import logging
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
import uvicorn
from functools import partial

from components.retriever.retriever_orchestrator import create_retriever_from_config
from components.generator.generator_orchestrator import Generator
from components.llm import build_llm_client
from components.orchestration.workflow import build_workflow
from components.orchestration.ui_adapters import chatui_adapter, chatui_file_adapter
from components.orchestration.state import ChatUIInput, ChatUIFileInput
from components.utils import getconfig
from components.retriever.filters import FILTER_VALUES
from components.rewriter.db_context import load_db_context, DBContext
from components.guardrails.input_guard import InputGuardClient
from components.guardrails.output_guard import load_compiled_blocklist
from components.guardrails.output_classification import OutputClassificationConfig, build_output_classification_messages
from components.guardrails.llm_guard import build_guard_backend, DEFAULT_CLASSIFIER_MODEL
from typing import Dict

config = getconfig("params.cfg")
MAX_TURNS = config.getint("conversation_history", "MAX_TURNS", fallback=3)
MAX_CHARS = config.getint("conversation_history", "MAX_CHARS", fallback=8000)

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

# Validate: every field declared in params.cfg must have valid values in filters.py
if FILTERABLE_FIELDS:
    _missing = [f for f in FILTERABLE_FIELDS if f not in FILTER_VALUES]
    if _missing:
        raise ValueError(
            f"Fields declared in params.cfg [metadata_filters] are missing from filters.py: {_missing}. "
            "Add valid values for these fields in src/components/retriever/filters.py "
            "or remove them from filterable_fields in params.cfg."
        )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Query rewriter config (optional — defaults to disabled if section is absent)
REWRITER_ENABLED = config.getboolean("query_rewriter", "enabled", fallback=False)
DB_CONTEXT_PATH = config.get("query_rewriter", "db_context_path", fallback="db_context.yaml")

db_context: DBContext = DBContext() # instantiating a typed object here for potential future use (e.g. with metadata filter node)
if REWRITER_ENABLED:
    db_context = load_db_context(DB_CONTEXT_PATH)
    if not db_context.abstract.strip():
        logger.warning(
            f"Query rewriter: db_context.abstract is empty - running in conservative mode \
             (no glossary-based expansion). Populate {DB_CONTEXT_PATH} to ground the rewriter.")

# Initialize services
logger.info("Initializing ChaBoHFEndpointRetriever and per-task LLM clients...")
retriever_instance = create_retriever_from_config(config_file="params.cfg")

# Get module-specific LLM configs
generation_client = build_llm_client(config, "generation")
generator_instance = Generator(llm_client=generation_client)

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
            prompt_builder=build_output_classification_messages,
        )
    output_classification_config = OutputClassificationConfig(
        backend=classification_backend,
        window_chars=config.getint("output_guard", "window_chars", fallback=600),
        notice=config.get("output_guard", "classification_message", fallback="[response withheld]"),
        timeout_s=config.getfloat("output_guard", "timeout_s", fallback=5.0),
    )
    logger.info(f"Output classifier enabled (mode={OUTPUT_CLASSIFICATION_MODE}, window_chars={output_classification_config.window_chars})")

# --- Keyword blocklist ---
BLOCKLIST_ENABLED = config.getboolean("output_guard", "blocklist_enabled", fallback=False)
BLOCKLIST_MESSAGE = config.get("output_guard", "blocklist_message", fallback="[response withheld]")
blocklist = None
if BLOCKLIST_ENABLED:
    blocklist_path = config.get(
        "output_guard", "blocklist_path", fallback="src/components/guardrails/blocklist.txt"
    )
    blocklist = load_compiled_blocklist(blocklist_path)
    logger.info("Output blocklist enabled")

# Build the LangGraph workflow
compiled_graph = build_workflow(
    retriever_instance,
    generator_instance,
    filterable_fields=FILTERABLE_FIELDS,
    filter_values=FILTER_VALUES,
    db_context=db_context if REWRITER_ENABLED else None,
    rewriter_enabled=REWRITER_ENABLED,
    guard_client=guard_client,
    input_guard_enabled=INPUT_GUARD_ENABLED,
    blocked_message=INPUT_GUARD_MSG,
    filter_extraction_client=filter_extraction_client,
    query_rewrite_client=query_rewrite_client,
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
            "chatfed-ui-stream": "/chatfed-ui-stream (LangServe)",
            "chatfed-with-file-stream": "/chatfed-with-file-stream (LangServe)",
        }
    }


#----------------------------------------
# LANGSERVE ROUTES
#----------------------------------------

# Inject compiled_graph and config into adapters (blocklist=None when the blocklist is disabled)
text_adapter = partial(chatui_adapter, compiled_graph=compiled_graph, max_turns=MAX_TURNS, max_chars=MAX_CHARS,
                       blocklist=blocklist, blocklist_notice=BLOCKLIST_MESSAGE, classification_config=output_classification_config)
file_adapter = partial(chatui_file_adapter, compiled_graph=compiled_graph, max_turns=MAX_TURNS, max_chars=MAX_CHARS,
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