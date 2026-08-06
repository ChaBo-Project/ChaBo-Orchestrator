# ChaBo

A RAG (Retrieval-Augmented Generation) orchestrator API built with FastAPI, LangChain, and LangGraph. ChaBo orchestrates embedding, vector search, reranking, and LLM generation to answer queries using retrieved context. It also supports query rewriting (including cross-lingual), LLM-based metadata filtering, and optional input/output safety guardrails.

## Architecture

```
┌─────────────┐     ┌────────────────────────────────────────────────────────┐
│   ChatUI    │────▶│                        ChaBo                           │
│  (Frontend) │     │  ┌─────────┐   ┌──────────────┐   ┌───────────────┐    │
└─────────────┘     │  │ Embed   │──▶│ Smart Search │──▶│    Rerank     │    │
                    │  │ (HF)    │   │   (Qdrant)   │   │    (HF)       │    │
                    │  └─────────┘   └──────▲───────┘   └───────┬───────┘    │
                    │                       │                   │            │
                    │               ┌───────┴──────┐   ┌───────▼────────┐    │
                    │               │   Extract    │   │    Generate    │    │
                    │               │  Filters*    │   │  (Multi-LLM)   │    │
                    │               └──────────────┘   └────────────────┘    │
                    └────────────────────────────────────────────────────────┘
```

**Pipeline:** Query → [Input Guard*] → [Rewrite*] → Extract Filters* → Smart Search → Rerank → Generate → [Output Guard*] (with citations)

> Stages marked `*` are optional and configured in `params.cfg`; all are off unless enabled.

> **Smart Search** applies LLM-extracted metadata filters to narrow Qdrant results before reranking. Filters are pulled from the current query, with conversation history as fallback. When filters are applied, ChatUI displays a footnote at the end of each response (e.g. *🔍 Searched within: category: news · lang: en*) — including a note if the AND-safeguard fired and narrowed the filter to the priority field. Activated only when `filterable_fields` is configured under `[metadata_filters]` in `params.cfg` — omit or leave empty for standard unfiltered search.

> **Query rewriting** normalises the query (term/acronym expansion, pronoun resolution, query completion) and can translate it into the corpus language before retrieval. Grounded by `instance.yaml`'s `db_context` key (see Instance Configuration below). Configured under `[query_rewriter]`.

> **Guardrails** are two independent, opt-in defenses. The **input guard** classifies the query for prompt-injection/jailbreak and harmful content before retrieval (runs in parallel with rewrite/filter, ≈zero added latency); an unsafe verdict short-circuits to a fixed message. The **output guard** screens the streamed answer against a multilingual blocklist and replaces it with a notice on a hit. Configured under `[input_guard]` / `[output_guard]`.

**Supported LLM Providers:** HuggingFace, OpenAI, Anthropic, Cohere, Azure OpenAI

## Deployment

Deploy ChaBo as a standalone API using the root `Dockerfile`. This is the setup used on HuggingFace Spaces, where a frontend (e.g. ChatUI) runs separately. Full-stack deployment (ChatUI, local TEI, local Qdrant via Docker Compose) lives in a separate stack outside this repo.

**Prerequisites:** Remote HuggingFace Inference Endpoints for embedding and reranking, an existing Qdrant instance, and API keys.

#### Configuration

Edit `params.cfg` with your service endpoints:

```ini
[hf_endpoints]
embedding_endpoint_url = https://your-embedding-endpoint.huggingface.cloud
reranker_endpoint_url = https://your-reranker-endpoint.huggingface.cloud

[qdrant]
mode = native
url = https://your-qdrant-instance.cloud.qdrant.io:6333
collection = your_collection

[retrieval]
top_k = 20
reranker_top_k = 5
reranker_enabled = true    # false = skip the reranker, return top reranker_top_k in retrieval order
hybrid_enabled = false     # dense + BM25 sparse, fused by Qdrant's weighted RRF (see below)

[generator]
PROVIDER = huggingface
MODEL = meta-llama/Meta-Llama-3-8B-Instruct
MAX_TOKENS = 1024
TEMPERATURE = 0.1
INFERENCE_PROVIDER = ABC
ORGANIZATION = XYZ
CONTEXT_META_FIELDS = filename,project_id,document_source
TITLE_META_FIELDS = filename,page

[metadata_filters]
filterable_fields = project_id:str,year:int,tags:list

[conversation_history]
MAX_TURNS = 3
MAX_CHARS = 8000
```

#### Metadata Filters Setup

Enabling `filterable_fields` requires two additional steps beyond `params.cfg`:

**1. Set `filters` in `INSTANCE_CONFIG_DIR/instance.yaml`** with valid values for each declared field (see Instance Configuration below):

```yaml
filters:
  project_id: ["proj-001", "proj-002", "proj-003"]
  year: [2022, 2023, 2024]
  tags: ["report", "policy", "technical"]
```

Every field listed in `filterable_fields` must have an entry here — ChaBo validates this at startup and will refuse to start if any field is missing. Values must exactly match what is stored in your Qdrant collection (the LLM will snap user queries to the closest match from this list).

**2. Ensure your Qdrant payloads use the correct schema.** ChaBo expects filterable fields stored as a nested `metadata` object inside each point's payload:

```json
{
  "text": "document content...",
  "metadata": {
    "project_id": "proj-001",
    "year": 2023,
    "tags": ["report", "policy"]
  }
}
```

Fields stored as top-level keys or as JSON strings will not be found by the filter. If you use `upload_parquet.py` for ingestion (see Data Upload below), this schema is handled automatically.

> Omit or leave `filterable_fields` empty to run standard unfiltered search — no `instance.yaml` changes needed.

#### Query Rewriting & Guardrails Setup (optional)

Three optional pipeline stages, all configured in `params.cfg` and **off by default**. Omit each section (or set `enabled = false`) to skip it.

> **Upgrading from an earlier version:** each optional stage now uses its **own** independent LLM config (`llm_provider` / `llm_model`, etc.) under its own `params.cfg` section — there is no fallback to `[generator]` anymore. If you already have `[metadata_filters] filterable_fields` set and/or `[query_rewriter] enabled = true`, you must add `llm_provider` / `llm_model` (and, for HuggingFace, `llm_inference_provider` / `llm_organization`) under those same sections before upgrading, or ChaBo will fail to start. The startup error names the exact missing key, e.g. `LLM config missing: [metadata_filters] llm_provider (or env FILTER_EXTRACTION_LLM_PROVIDER) is required.` See the example blocks below for the full set of keys each stage expects.
>
> **No implicit reuse of `[generator]` across stages, ever — not even as a fallback.** This isn't just an upgrade-time gotcha: there is no mechanism, now or later, where an unconfigured guard/filter/rewriter stage silently falls back to the main generation model. If you want the input guard (or output classifier, filter extraction, query rewriter) to run on the *same* underlying model/endpoint you already use for generation, you must explicitly repeat those same `provider` / `model` values (and any provider-specific fields, e.g. `inference_provider` / `organization` for HuggingFace) under that stage's own section — copy-paste, not inheritance. An unconfigured stage that's turned on fails to start rather than quietly reusing `[generator]`.

**Query rewriter** — normalises the query and optionally translates it into the corpus language before retrieval:

```ini
[query_rewriter]
enabled = true
target_language =                   # ISO code (e.g. "ar") for cross-lingual rewriting; empty to disable
```

Grounding (corpus abstract + glossary) comes from `INSTANCE_CONFIG_DIR/instance.yaml`'s `db_context` key — see Instance Configuration below. With an empty/absent abstract and glossary it runs in conservative mode (pronoun/filler/language normalisation only) and logs a startup warning.

**Input guard** — classifies the query before retrieval; an unsafe verdict short-circuits to `blocked_message`:

```ini
[input_guard]
enabled = true
mode = llm                    # 'llm' = in-context classification via its own LLM (see llm_* below); 'classifier' calls a Qwen3Guard-Gen endpoint
endpoint_url =                 # required for mode = classifier, e.g. http://qwen3guard:8000
model = Qwen/Qwen3Guard-Gen-0.6B
llm_provider = huggingface     # required for mode = llm — its own independent LLM config, no fallback to [generator]
llm_model = ...
llm_max_tokens = 64
llm_temperature = 0.0
block_controversial = false
timeout_s = 2.0                # on timeout/error the guard fails open (allows the query)
blocked_message = I'm sorry, but I can't help with that request.
```

> `mode = llm` needs no extra infrastructure beyond its own `llm_*` config above, but its latency tracks that model — set `timeout_s` above its typical response time or the guard will time out and fail open (silently disabling protection). It does **not** reuse `[generator]`'s model — see the note above. `mode = classifier` needs a Qwen3Guard-Gen endpoint (see the `guard` Compose profile below).

**Output guard** — two independent sub-features, both off by default, applied to the streamed answer:

```ini
[output_guard]
# 1. Windowed classification — re-classifies a sliding window of the streamed
#    answer as it's generated; a hit stops the stream and classification_message
#    replaces it. Same mode/backend choice as the input guard.
classification_enabled = true
mode = llm                     # 'llm' or 'classifier', same semantics as input_guard
endpoint_url =                 # required for mode = classifier
model = Qwen/Qwen3Guard-Gen-0.6B
llm_provider = huggingface     # required for mode = llm — its own independent LLM config
llm_model = ...
window_chars = 200             # re-classify every N new chars of the answer
block_controversial = true
timeout_s = 5.0
classification_message = This application is not able to answer certain queries. Please try again.

# 2. Keyword blocklist — screens the stream against a multilingual term list
#    (Arabic-aware normalisation + CJK/Thai substring matching).
blocklist_enabled = true
blocklist_path = src/components/guardrails/blocklist.txt
blocklist_message = There was an error in the output. Please try again.
```

#### System Prompt: Framework Values vs Instance Guidelines

The generator's system prompt is composed from three layers, in this order:

1. **`FRAMEWORK_VALUES`** — the framework owner's non-negotiable content rules (currently: don't dispute scientific consensus, stay politically/religiously neutral, avoid divisive stances). This is a **code-only constant** in `src/components/generator/prompts.py` — it is never read from `params.cfg` or any instance file, and changing it requires a deliberate source edit, not a config flip.
2. **`BASE_PROMPT`** — the generic RAG instructions (citation format, context-only answers, formatting) — the same for every deployment.
3. **Instance guidelines** *(optional, off by default)* — your own deployment-specific guidance (tone, domain scope, formatting preferences), loaded from `INSTANCE_CONFIG_DIR/instance.yaml`'s `instance_guidelines` key (see Instance Configuration below) — empty/absent = none.

If an instance guideline ever conflicts with a Framework Value, the composed prompt tells the model the Framework Value takes precedence.

**Output-guard enforcement of these two layers is asymmetric, by design:** the windowed output classifier (above) always checks the answer against `FRAMEWORK_VALUES` (a violation always blocks, using `classification_message` — not configurable). It can *additionally* check compliance with your instance guidelines, but only under `mode = llm` (the `classifier`/Qwen3Guard-Gen backend is a fixed-taxonomy model and cannot evaluate custom instance text at all). The consequence for instance-guideline non-compliance is your own choice:

```ini
[output_guard]
guideline_enforcement = off   # off (default) | warn (log only) | block (stop the stream, same as a framework violation)
```

This exists so a deployment can check its own rules against itself without ever being able to weaken the framework-values check — `guideline_enforcement` has no effect on Framework Values enforcement, which is always on.

Pass API keys as environment variables at runtime:

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace API token |
| `QDRANT_API_KEY` | Yes | Qdrant API key |
| `OPENAI_API_KEY` | If using OpenAI | OpenAI API key |
| `ANTHROPIC_API_KEY` | If using Anthropic | Anthropic API key |
| `COHERE_API_KEY` | If using Cohere | Cohere API key |
| `AZURE_API_KEY` | If using Azure OpenAI | Azure OpenAI API key |

#### Build and Run

```bash
docker build -t chabo .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e QDRANT_API_KEY=your_key \
  chabo
```

## Instance Configuration

The image this repo publishes is generic and instance-blind — it ships no corpus content,
filter values, or per-deployment prompt text. Everything specific to one deployment is
supplied at runtime via the `INSTANCE_CONFIG_DIR` environment variable, pointing at a
directory with up to three files. All three are optional; if `INSTANCE_CONFIG_DIR` is unset,
or a given file is absent, that piece falls back to a safe generic default (empty filters, no
extra blocklist terms, no instance guidelines, default prompt wording) — no behavior change
from today.

| File | Tier | Format | Contains |
|---|---|---|---|
| `params.override.cfg` | 1 — pipeline knobs | INI, same `[section]key=value` shape as `params.cfg` | Retrieval tuning, endpoints, provider/model selection, guardrail toggles. Layered on top of `params.cfg`; later value wins per key. |
| `prompt_overrides.md` | 1 — prompt engineering | Markdown, `## section_name` headers | `query_rewrite_steps`, `filter_extraction_steps` — overrides the *decision-making* portion of those two prompts only. The output JSON schema/contract is always core-owned and never overridden, so a bad override can degrade quality but can't break parsing. |
| `instance.yaml` | 2 — safe content | YAML, four independent top-level keys | `filters` (valid values for `[metadata_filters] filterable_fields`), `db_context` (`abstract`/`glossary` for the query rewriter), `blocklist` (`{lang: [terms]}`, additions on top of the shipped list — never removes from it), `instance_guidelines` (plain text appended to the system prompt, subordinate to the framework's non-negotiable content rules). |

**Tier 1 is "change at your own risk."** These values affect pipeline behavior directly — a
bad edit can degrade retrieval or generation quality, not just wording. Tier 2 is safe for a
non-engineer to edit: worst case is a less helpful answer, never a broken pipeline.

`instance.yaml` example, showing all four keys:

```yaml
filters:
  crop_type: ["wheat", "maize", "cotton"]
  title: ["Cultivation and producing Wheat", "Cultivation and producing Maize"]

db_context:
  # Short natural-language description of what's in the document store
  # (domain, scope, time period, document types).
  abstract: "Agricultural extension guides published by the Ministry of Agriculture, covering crop cultivation practices."
  # Each entry: canonical term to rewrite into, plus acronyms/synonyms/variant
  # spellings that should resolve to it, and an optional short gloss for the LLM.
  glossary:
    - canonical: "Egyptian Ministry of Agriculture and Land Reclamation"
      aliases: ["MALR", "Ministry of Agriculture"]
      definition: "Egyptian government body responsible for agricultural policy."

blocklist:
  # Additions only, per language — layered on top of the shipped list, never
  # removes from it.
  en: ["some-term"]
  ar: ["مصطلح"]

instance_guidelines: |
  Keep answers focused on crop cultivation topics. Prefer metric units.
```

Note: `db_context`'s `target_language` for cross-lingual rewriting is **not** part of this
key — it's configured separately via `[query_rewriter] target_language` in `params.cfg` (or
`params.override.cfg`).

Example layout:

```
instance_config/
├── params.override.cfg
├── prompt_overrides.md
└── instance.yaml
```

```bash
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e QDRANT_API_KEY=your_key \
  -e INSTANCE_CONFIG_DIR=/app/instance_config \
  -v $(pwd)/instance_config:/app/instance_config \
  chabo
```

## Data Upload

To populate a Qdrant collection with your embedding data, place your data file at `data/data.parquet` and run (inside the activated `chabo_env` virtual environment, or via `docker exec` into a running `chabo` container):

```bash
python src/components/ingestor/upload_parquet.py \
    --file data/data.parquet \
    --collection YOUR_COLLECTION_NAME \
    --vector_size 1024
```

**For hybrid retrieval**, add `--hybrid`. This builds the collection with a *named* dense vector plus a BM25 sparse vector, and computes the sparse vectors from each chunk's `text`:

```bash
python src/components/ingestor/upload_parquet.py \
    --file data/data.parquet \
    --collection YOUR_COLLECTION_NAME \
    --vector_size 1024 \
    --hybrid \
    --sparse_language english        # use 'arabic' for an Arabic corpus
```

> **`--sparse_model` and `--sparse_language` must match `[retrieval] sparse_model` and `sparse_language` in `params.cfg`.** 

Hybrid needs a **fresh** collection — an existing dense-only collection stores one unnamed vector and cannot gain a sparse one in place, so re-ingest under a new `--collection` name. It also requires **Qdrant server 1.17 or newer** (weighted RRF) and `[qdrant] mode = native`; both are verified at startup.

**Expected parquet schema** — the file must have two columns:

| Column | Type | Description |
|--------|------|-------------|
| `vector` | list of float | Pre-computed embedding vector for this chunk |
| `payload` | dict | Must contain `text` (string) and `metadata` (dict) |

Each row becomes one Qdrant point with three components:

```
id      → auto-assigned as row index
vector  → df.vector  (your pre-computed embedding)
payload → { "text": "...", "metadata": { "field": "value", ... } }
```

The `metadata` dict inside `payload` is where filterable fields live (see Metadata Filters Setup above). The upload script handles collection creation automatically if it does not already exist.

> **Note:** Use the same collection name as in your `params.cfg` and the correct vector dimension for your embedding model (e.g. 1024 for BGE-large, 768 for BGE-base). The `vector_size` must match when the collection is first created — it cannot be changed afterwards.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |
| `/chatfed-ui-stream` | POST | Text query streaming (LangServe) |
| `/chatfed-with-file-stream` | POST | File upload + query streaming (LangServe) |


## Health Checks & Testing

The `tests/health/` directory contains scripts to verify your setup before or after deployment. These run **locally outside Docker**, against already-running services, using the project's virtual environment.

### Prerequisites

Before running any health checks ensure:
1. Services are up — Qdrant, embedding endpoint, reranker endpoint
2. `params.cfg` is configured with correct endpoint URLs and collection name
3. Required env vars are exported (`HF_TOKEN`, `QDRANT_API_KEY`, etc.)

### Setup

```bash
# Navigate to repo root — required for src/ imports to resolve
cd /path/to/chabo

```

### Running the full health suite

```bash
python tests/health/run_all.py
```

`run_all.py` checks each component in order and prints a pass/fail summary:

| Step | Check | What it verifies |
|------|-------|-----------------|
| 1 | Qdrant | Reachable and configured collection exists |
| 1 | Embedding endpoint | Returns a valid vector |
| 1 | Reranker endpoint | Returns scores |
| 2 | Retriever unit | Full retrieval + reranking returns ranked documents |
| 2 | RAG pipeline | End-to-end retrieval → streaming generation with a sample query |
| 2 | Metadata filters | Three sub-tests against live Qdrant: single-field filter returns docs; valid multi-field AND returns docs; impossible AND combination triggers the priority-field safeguard and retries with the first `filterable_fields` entry only |

Step 2 (component checks) only runs if all Step 1 connectivity checks pass. Logs are written to `tests/health/logs/` with a timestamp for each run.

> **Note — Metadata Filters check:** requires `filterable_fields` to be configured in `params.cfg` and the Qdrant collection to have the corresponding payload fields stored as nested dicts (not JSON strings).
>
> The three sub-tests are hardcoded to a sample corpus — **you must adapt them to your own collection** before running. Open `tests/health/test_components.py` and update the queries and filter values inside `test_metadata_filters()`:
>
> | Sub-test | What to change | Example (agriculture corpus) |
> |----------|---------------|------------------------------|
> | 1 — single field | Query + one valid field/value | `filters={"crop_type": ["wheat"]}` |
> | 2 — valid AND | Query + two fields that co-exist in your data | `filters={"crop_type": ["maize"], "title": "Maize cultivation in the old and new lands"}` |
> | 3 — safeguard | Query + an impossible combination (valid value for field 1, non-existent value for field 2) so AND returns 0 and the retry fires | `filters={"crop_type": ["wheat"], "title": "Cultivation and producing Maize"}` |
>
> The priority field (used in the safeguard retry) is always the **first key in `filterable_fields`** in `params.cfg`.

### Running individual scripts

```bash
# Retriever + pipeline tests only (skips connectivity pre-checks)
python tests/health/test_rag_pipeline.py
```

`test_rag_pipeline.py` is for manual, qualitative spot-checks during development — inspect
logs to verify retrieval scores and response quality for specific scenarios.

Edit the `test_cases` list to add your own scenarios. The examples below use an
**agriculture knowledge base** as a reference — imagine a RAG system built on crop guides,
irrigation manuals, and farming practices. The scenarios themselves apply to any domain:
swap in questions relevant to your own corpus.

```python
test_cases = [
    # In-domain factual — system should retrieve and answer well
    ("factual_question", "What fertilizer is recommended for wheat in sandy soil?"),

    # In-domain summary — requires synthesising multiple docs
    ("summary_question", "What are the irrigation methods used for sugarcane?"),

    # Out-of-domain / hallucination risk — completely outside your corpus,
    # system should return a graceful no-answer rather than hallucinate
    ("out_of_domain", "What is the transformer architecture used in LLMs?"),

    # Ambiguous — underspecified, tests behaviour under low retrieval confidence
    ("ambiguous_query", "How do I grow it?"),
]
```

The `case_name` (first element) is used as a label in logs and the final pass/fail summary.
The hallucination risk warning fires automatically when the LLM gives a long answer despite
very low retrieval scores — a useful signal for out-of-domain queries.

---

## RAG Evaluation

The `tests/eval/` directory contains scripts to evaluate retrieval, reranking, and answer quality. Like health checks, these run locally outside Docker with the venv active from the repo root.

### How it works

Evaluation runs in stages — run them independently or in sequence:

**Stage 1 — Retrieval** (`--mode retrieval`)
Runs each question through the full retriever pipeline, capturing both raw vector search candidates and final reranked results. Output saved to `tests/eval/results/retrieval_eval_results.json`.

**Stage 2 — LLM Judging** (`--mode batch`)
Loads the retrieval results and uses the configured LLM to judge each retrieved document for relevance. Saves a judged report to `tests/eval/results/judged_eval_report.json`. Supports **checkpointing** — if interrupted, it resumes from where it left off.

**Stage 3 — RAGAS** (`--mode ragas`)
Runs the full pipeline end-to-end (retrieval + generation) and scores answer quality using [RAGAS](https://docs.ragas.io) metrics. Each run saves a timestamped JSON report to `tests/eval/results/` and appends a summary row to `tests/eval/results/ragas_history.csv` for tracking quality over time.

### Setup

```bash
cd /path/to/chabo
source chabo_env/bin/activate

# For RAGAS mode only — install the extended dependencies
pip install -r tests/eval/requirements-eval.txt
```

### Define your test questions

Edit `tests/eval/test_questions.py` to add your evaluation questions. Questions are organised into three subsets:

- **`standalone_questions`** — each query explicitly contains a filterable value; evaluated with no history
- **`history_blocks`** — conversation sequences where later turns rely on filter carry-forward; only the last turn per block is evaluated
- **`safeguard_questions`** — contradictory or unlikely field combinations that should trigger the AND-safeguard fallback

Each entry supports the following fields:

| Field | Required for | Description |
|-------|-------------|-------------|
| `question` / `turns` | all modes | The query or conversation turns |
| `expected_filters` | retrieval, batch, sample | Ground truth metadata filters |
| `expected_answer` | ragas | Rough ground truth answer string |
| `expected_sources` | none (reserved) | List of `{filename, page}` dicts identifying the expected source document(s). Not currently read by any eval mode, including RAGAS — reserved for a future deterministic retrieval-hit check |

These should be realistic queries representative of what actual users ask — curated with knowledge of your corpus. The examples below assume an **agriculture knowledge base**; replace them with questions and filter values relevant to your own domain:

```python
standalone_questions = [
    {
        "question": "What irrigation method is recommended for sugarcane on new land?",
        "expected_filters": {"crop": "sugarcane"},
        "expected_answer": "Drip irrigation is recommended for sugarcane on new land...",
        "expected_sources": [{"filename": "sugarcane_guide.pdf", "page": 12}],
    },
]

history_blocks = [
    {
        "turns": [
            "I'm looking for information about wheat crop management.",
            "What are the recommended pesticide applications?",
        ],
        "expected_filters": {"crop": "wheat"},
        "expected_answer": "Recommended pesticides for wheat include...",
        "expected_sources": [{"filename": "wheat_manual.pdf", "page": 5}],
    },
]

safeguard_questions = [
    {
        "question": "What does the maize report on wheat fertilisation say?",
        "expected_filters": None,
        "expected_answer": "The knowledge base does not contain a document combining those topics.",
        "expected_sources": [],
    },
]
```

Cases with `expected_answer` still set to `TODO` are skipped automatically in RAGAS mode.

> **Note:** `test_questions.py` is for automated scoring via LLM-as-judge (`eval.py`).
> For manual qualitative spot-checks with categorised scenarios, use
> `tests/health/test_rag_pipeline.py` instead — the two complement each other.

### Run evaluation

Pass the `--mode` flag to select which stage to run, and `--filters` to enable metadata filter extraction:

```bash
# Stage 1: Run retrieval and save results
python tests/eval/eval.py --mode retrieval

# Stage 1 with metadata filter extraction and ground truth checking
python tests/eval/eval.py --mode retrieval --filters

# Stage 2: Judge retrieved results with LLM (resumes from checkpoint if interrupted)
python tests/eval/eval.py --mode batch
python tests/eval/eval.py --mode batch --filters

# Quick sample run (first 2 questions only, useful for testing)
python tests/eval/eval.py --mode sample

# Stage 3: Full pipeline eval with RAGAS metrics
python tests/eval/eval.py --mode ragas
python tests/eval/eval.py --mode ragas --filters
```

`--mode retrieval` is the default if no flag is provided.

Results are saved to `tests/eval/results/` (gitignored). The `--filters` flag appends `_filtered` to all output filenames — compare `judged_eval_report.json` vs `judged_eval_report_filtered.json` to measure the impact of filtering on retrieval quality.

### RAGAS configuration

RAGAS mode uses a separate judge LLM configured in `params.cfg`'s `[ragas]` section — independent of the chatbot's generator's `[generator]` section, to avoid self-preference bias in scoring:

```ini
[ragas]
JUDGE_PROVIDER = your-provider        # openai, anthropic, cohere, azure, huggingface
JUDGE_MODEL = your-model-name-or-url

METRICS = faithfulness,answer_relevancy,context_recall,context_precision
```

Set the corresponding API key as an environment variable (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). RAGAS metrics rely on the judge model's reasoning ability — use a capable model for reliable scores.

Each run produces a timestamped JSON with per-question scores (e.g. `ragas_report_20260721_143022.json`) and appends a summary row to `ragas_history.csv` for tracking quality trends across releases and feature updates.

### Filter Ground Truth Checking

When running with `--filters`, the extracted filters are automatically compared against `expected_filters` from `test_questions.py`. Two files are produced:

- **`retrieval_eval_results_filtered.json`** — full retrieval results, each entry includes a `filter_check` field with `expected`, `extracted`, and `result`
- **`filter_check_report_filtered.json`** — a dedicated report for at-a-glance inspection

Possible `result` values:

| Result | Meaning |
|--------|---------|
| `correct` | Extraction matches expected exactly, or no filter expected and none extracted |
| `partial_match` | At least one field matches, others wrong or missing |
| `mismatch` | Filter was extracted but no fields match expected |
| `no_filter` | A filter was expected but none was extracted |
| `spurious_filter` | No filter was expected but one was extracted |

A console summary is also printed at the end of each `--mode retrieval --filters` run.

---

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.
