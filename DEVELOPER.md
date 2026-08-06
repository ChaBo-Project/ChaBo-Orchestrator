# ChaBo — Developer Guide

This document is for contributors and developers working on the ChaBo codebase. For deployment and configuration, see [README.md](README.md).

---

## Local Setup

```bash
# Activate the virtual environment (always required before running Python)
source chabo_env/bin/activate

# Export required env vars
export HF_TOKEN=your_token
export QDRANT_API_KEY=your_key

# Run locally without Docker (from repo root)
cd src
python main.py
```

API will be available at `http://localhost:7860`. Interactive docs at `http://localhost:7860/docs`.

> Configuration precedence: **kwargs → env vars → `params.cfg` → hardcoded defaults**

---

## Codebase Layers

Not all files are equal. The codebase has four distinct layers — understanding which layer a file sits in tells you immediately whether to touch it.

```
┌──────────────────────────────────────────────────────────────────────┐
│  CONFIGURE — params.cfg only, zero code changes                      │
│                                                                      │
│  LLM provider · model · endpoints · Qdrant URL · collection         │
│  top_k · reranker_top_k · filterable_fields · MAX_TURNS · chunking  │
└──────────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────────┐
│  CUSTOMIZE — files expected to change per deployment / corpus        │
│                                                                      │
│  Must                              Optional                          │
│  ──────────────────────────────    ──────────────────────────────    │
│  retriever/filters.py              orchestration/ui_adapters.py      │
│  → valid filter values for           _build_filters_footnote()       │
│    your corpus; startup fails         controls the italic text        │
│    if any declared field missing      shown in ChatUI when filters    │
│                                       are applied                    │
│  generator/prompts.py                                                │
│  → system_prompt controls all                                        │
│    RAG behavior: tone, citation                                      │
│    rules, response format,                                           │
│    missing-info handling                                             │
│  → build_filter_extraction_messages()                                │
│    controls how the LLM extracts                                     │
│    metadata filters from queries                                     │
└──────────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────────┐
│  EXTEND — touch when adding new capabilities to the framework        │
│                                                                      │
│  orchestration/nodes.py       → define a new async node function     │
│  orchestration/workflow.py    → wire it into the graph               │
│  orchestration/state.py       → add new fields the node reads/writes │
│  generator/generator_orchestrator.py → add a new LLM provider        │
└──────────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────────┐
│  INFRASTRUCTURE — don't touch                                        │
│                                                                      │
│  retriever/retriever_orchestrator.py   ingestor/ingestor.py          │
│  ingestor/upload_parquet.py            generator/sources.py          │
│  generator/generator_orchestrator.py   orchestration/telemetry.py   │
│  orchestration/ui_adapters.py (core)   utils.py                      │
└──────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────────┐
│  TESTS — adapt to your corpus before use                             │
│                                                                      │
│  tests/health/test_components.py    → hardcoded filter tests;        │
│                                       replace values for your corpus │
│  tests/health/test_rag_pipeline.py  → domain spot-check scenarios;   │
│                                       replace example queries        │
│  tests/eval/test_questions.py       → eval question sets;            │
│                                       replace with your domain       │
│  tests/unit/test_extract_filters.py → unit tests; extend as needed   │
│                                                                      │
│  tests/health/run_all.py            → orchestrator; no changes needed│
│  tests/eval/eval.py                 → runner; no changes needed      │
│  tests/eval/report.py               → formatter; no changes needed   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Full File Reference

```
src/
├── main.py                                     # FastAPI app, LangServe route registration, startup validation of filterable_fields vs instance.yaml
└── components/
    ├── utils.py                                # Shared: getconfig, get_config_value, build_conversation_context, HTTP helpers (_call_hf_endpoint, _acall_hf_endpoint)
    ├── orchestration/
    │   ├── workflow.py                         # Builds and compiles the LangGraph state machine — EXTEND
    │   ├── nodes.py                            # The 4 async graph node functions — EXTEND
    │   ├── state.py                            # GraphState TypedDict + ChatUIInput / ChatUIFileInput Pydantic models — EXTEND
    │   ├── ui_adapters.py                      # Translates LangServe input → graph state and graph events → streamed text — INFRASTRUCTURE
    │   │                                       # Optional customize: _build_filters_footnote() controls filter display text in ChatUI
    │   └── telemetry.py                        # Extracts retriever telemetry from Document metadata for logging — INFRASTRUCTURE
    ├── retriever/
    │   ├── retriever_orchestrator.py           # ChaBoHFEndpointRetriever: Embed → Qdrant Search → Rerank — INFRASTRUCTURE
    │   └── filters.py                          # FILTER_VALUES, loaded from INSTANCE_CONFIG_DIR/instance.yaml — INFRASTRUCTURE (real values live in instance.yaml, not here)
    ├── generator/
    │   ├── generator_orchestrator.py           # LLM provider wiring, config resolution, generate() / generate_streaming() — INFRASTRUCTURE
    │   ├── prompts.py                          # All LLM prompt content — MUST / OPTIONAL CUSTOMIZE
    │   │                                       # system_prompt → RAG behavior, citations, tone (must)
    │   │                                       # build_filter_extraction_messages() → filter extraction behavior (optional)
    │   │                                       # build_messages() → assembles context + history into LLM input (infrastructure)
    │   └── sources.py                          # Pure utility: citation regex parsing, source list formatting — INFRASTRUCTURE
    └── ingestor/
        ├── ingestor.py                         # In-memory PDF/DOCX → chunked text (no disk writes) — INFRASTRUCTURE
        └── upload_parquet.py                   # One-off script: bulk-upload parquet into Qdrant (follow schema, no changes needed) — INFRASTRUCTURE
```

---

## Request Lifecycle

A complete trace of a text query from HTTP request to streamed response:

```
POST /chatfed-ui-stream
  → LangServe (main.py)
      RunnableLambda(text_adapter) registered via add_routes()

  → chatui_adapter(data, compiled_graph, ...) [ui_adapters.py]
      - Parses messages[], extracts latest user query
      - Builds conversation_context  — full N-turn history for LLM generation
      - Builds user_messages_history — user-only turns (no assistant text) for filter extraction
      - Calls process_query_streaming()

  → process_query_streaming() [ui_adapters.py]
      - Constructs initial GraphState dict
      - compiled_graph.astream(initial_state, stream_mode="custom")

      ┌─ ingest_node [nodes.py] ───────────────────────────────────────
      │  If file_content + filename in state:
      │    calls process_document() [ingestor.py] → chunked text string
      │  Writes: ingestor_context → state
      └────────────────────────────────────────────────────────────────

      ┌─ extract_filters_node [nodes.py] ──────────────────────────────
      │  If filterable_fields empty: no-op, returns {}
      │  Calls build_filter_extraction_messages() [prompts.py]
      │    → builds [SystemMessage, HumanMessage] from filterable_fields,
      │      FILTER_VALUES, current query, and user history
      │  Calls generator._call_llm(messages)
      │  Parses JSON response via _parse_filter_response()
      │    → validates field names, casts types
      │  Writes: metadata_filters → state (or None on failure)
      └────────────────────────────────────────────────────────────────

      ┌─ retrieve_node [nodes.py] ─────────────────────────────────────
      │  Calls retriever.ainvoke(query, filters=metadata_filters)
      │
      │    → _aget_relevant_documents() [retriever_orchestrator.py]
      │        A. _acall_hf_endpoint(embedding_url) → query_vector
      │        B. _asearch_qdrant(query_vector, filters)
      │             - Builds Qdrant rest.Filter from filters dict
      │             - AND-safeguard: if 0 results with multi-field AND,
      │               retries with first (priority) field only
      │             - Returns (results, applied_filter, narrowed)
      │        C. _acall_hf_endpoint(reranker_url) → reranked scores
      │             - Fallback: if reranker fails, returns vector-search order
      │        D. Injects _applied_filter + _narrowed into docs[0].metadata
      │           so filter info travels with the ainvoke result
      │
      │  Pops _applied_filter / _narrowed from docs[0].metadata
      │  Emits writer({"event": "filters_applied", ...}) custom event
      │  Writes: raw_documents, applied_filters, filters_narrowed → state
      └────────────────────────────────────────────────────────────────

      ┌─ generate_node_streaming [nodes.py] ───────────────────────────
      │  If ingestor_context present: prepends as Document to raw_documents
      │  Calls generator.generate_streaming(query, context=raw_docs, ...)
      │    → process_context() [sources.py] formats docs + metadata for LLM
      │    → build_messages() [prompts.py] assembles system_prompt + context
      │       + conversation history into [SystemMessage, HumanMessage]
      │    → streams tokens via _call_llm_streaming()
      │    → after stream: clean_citations(), parse_citations(),
      │      create_sources_list() [sources.py]
      │  writer({"event": "data", "data": chunk}) per token
      │  writer({"event": "final_answer", "data": {webSources: [...]}})
      └────────────────────────────────────────────────────────────────

  ← process_query_streaming yields typed dicts:
      "data"            → {"type": "data",            "content": chunk}
      "filters_applied" → {"type": "filters_applied", "content": {filters, narrowed}}
      "final_answer"    → {"type": "sources",          "content": [...]}
      "end"             → {"type": "end"}

  ← chatui_adapter assembles the final stream:
      - Yields token chunks to ChatUI as they arrive
      - Stores filters_footnote when filters_applied event received
      - At "end": yields _build_filters_footnote() text then sources markdown
```

For file uploads the flow is identical via `chatui_file_adapter` / `/chatfed-with-file-stream`. The only difference is `file_content` + `filename` are decoded from base64 and added to initial state, causing `ingest_node` to run instead of skip.

---

## GraphState Field Reference

`GraphState` is a `TypedDict` in `state.py`. All nodes read from and write to this shared dict.

| Field | Type | Written by | Read by | Purpose |
|-------|------|-----------|---------|---------|
| `query` | `str` | `ui_adapters` | all nodes | Current user query |
| `conversation_context` | `str` | `ui_adapters` | `generate_node` | Full N-turn history for LLM generation |
| `user_messages_history` | `str` | `ui_adapters` | `extract_filters_node` | User-only turns — no assistant text, no retrieved content |
| `file_content` | `bytes` | `ui_adapters` | `ingest_node` | Raw uploaded file bytes |
| `filename` | `str` | `ui_adapters` | `ingest_node`, `generate_node` | Uploaded filename (used as source label) |
| `ingestor_context` | `str` | `ingest_node` | `generate_node` | Chunked text extracted from uploaded document |
| `metadata_filters` | `dict` | `extract_filters_node` | `retrieve_node` | LLM-extracted `{field: value}` filters, or `None` |
| `raw_documents` | `List[Document]` | `retrieve_node` | `generate_node` | Reranked documents returned by the retriever |
| `applied_filters` | `dict` | `retrieve_node` | `ui_adapters` (via event) | Actual filter used — may differ if AND-safeguard fired |
| `filters_narrowed` | `bool` | `retrieve_node` | `ui_adapters` (via event) | `True` if AND-safeguard fired and fell back to priority field |
| `metadata` | `dict` | all nodes | — | Per-request telemetry (durations, success flags, counts) |

`applied_filters` and `filters_narrowed` reach `ui_adapters` via a LangGraph custom event (`writer({"event": "filters_applied", ...})`), not by reading state directly after graph completion.

---

## Qdrant Modes

Configured via `[qdrant] mode` in `params.cfg`. Two modes are supported:

| Mode | Client used | When to use |
|------|------------|-------------|
| `native` | `QdrantClient` / `AsyncQdrantClient` | ChaBo can reach Qdrant directly — self-hosted or Qdrant Cloud |
| `gradio` | `GradioClient` (sync, wrapped in executor for async) | Qdrant is behind a HuggingFace Space or access-controlled Gradio app |

Both modes implement the same AND-safeguard retry logic. Clients are initialised lazily on first use.

---

## Filter Values Contract

Allowed values for each filterable field are **instance content, not source** — they live in
`INSTANCE_CONFIG_DIR/instance.yaml`'s `filters` key, loaded via `load_instance_yaml()` and exposed
through `validate_filterable_fields()`. No real values ship in the repo, by design (the published
image is instance-blind — see README's "Instance Configuration").

- **Keys must exactly match** field names declared in `[metadata_filters] filterable_fields` in `params.cfg`
- **Every declared field must have an entry** — `main.py` calls `filters.validate_filterable_fields()` at
  startup and it raises `ValueError` if any field is missing
- **Values must match what is stored in Qdrant** — the LLM uses this list to snap user queries to the closest valid value
- Extra keys in `instance.yaml`'s `filters` not in `filterable_fields` are silently ignored

**To add a new filterable field:**

1. Add it to `params.cfg`: `filterable_fields = existing_field:str,new_field:str`
2. Add it to `INSTANCE_CONFIG_DIR/instance.yaml`:
   ```yaml
   filters:
     new_field: ["value_a", "value_b"]
   ```
3. Ensure your Qdrant payloads store the field under `metadata.new_field`

If you're running without `INSTANCE_CONFIG_DIR` set (e.g. local dev against a test corpus),
`FILTER_VALUES` resolves to `{}` and startup will fail as soon as `filterable_fields` is non-empty —
that's the intended signal to go configure `instance.yaml`, not a bug.

---

## How to Extend

### Add a new LLM provider

1. Add the LangChain provider import to `generator_orchestrator.py`
2. Add the provider name and initialisation lambda to the `providers` dict in `Generator._get_chat_model()`
3. Add any required API key to the env var table in `README.md`

### Add a new graph node

1. Define `async def my_node(state: "GraphState", ...) -> "GraphState"` in `nodes.py`
2. Import and register it in `workflow.py` via `workflow.add_node("my_node", my_node)`
3. Wire edges — insert between two existing nodes or append before `END`
4. Add any new state fields the node reads/writes to `GraphState` in `state.py`

### Change RAG behavior (prompts.py)

`prompts.py` is the primary file for tuning how the assistant responds — no orchestration code needs changing:

- **`system_prompt`** — citation format, response structure, language matching, missing-info handling, follow-up question logic
- **`build_filter_extraction_messages()`** — the instructions given to the LLM for extracting metadata filters; tune this if extraction accuracy is poor for your domain

### Change filter display in ChatUI

Edit `_build_filters_footnote()` in `ui_adapters.py` to change the wording, emoji, or format of the italic footnote shown when filters are applied.

### Change retrieval parameters at runtime

`top_k` (retriever candidate count) and `reranker_top_k` (final reranked count) are set in the `[retrieval]` section of `params.cfg`. Unlike the deployment-identity settings, these tuning knobs have no env-var override - read from `params.cfg` only. The same applies to the hybrid retrieval knobs (`hybrid_enabled`, `reranker_enabled`, `prefetch_top_k`, `dense_weight`, `sparse_weight`, `rrf_k`) - `params.cfg` only, no env-var override.

---

## Known Tech Debt

| Location | Issue |
|----------|-------|
| `nodes.py` top docstring | Says "NEEDS TO BE UPDATED" — safe to ignore, the code is current |
| `nodes.py` lines ~290–390 | Large block of commented-out old `retrieve_node` implementations — dead code, pending cleanup |
| `ui_adapters.py:39` | Comment "TO BE REPLACED WITH AGENTIC WORKFLOW" — `process_query_streaming` is functional but flagged for future rework |
| `state.py` `raw_context` field | Listed as "Alias for backward compatibility" — not written by any current node, candidate for removal |
