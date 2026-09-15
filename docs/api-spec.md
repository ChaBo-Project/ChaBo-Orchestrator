# ChaBo API Specification

Version 1.0.0 · Base URL `http://<host>:7860`

Two HTTP surfaces sit in front of the **same** LangGraph pipeline, the same guardrails and
the same citation logic. They differ only in the renderer that turns ChaBo's internal event
stream into wire format (`src/components/orchestration/renderers.py`).

| Surface | Endpoints | Audience |
|---|---|---|
| OpenAI-compatible | `POST /v1/chat/completions`, `GET /v1/models` | Any generic frontend / SDK (OpenWebUI, LibreChat, curl) |
| LangServe | `/chatfed-ui-stream/*`, `/chatfed-with-file-stream/*` | Chabo-ChatUI only (its `langserve-streaming` connector) |

Authentication: none at the app layer. Deploy behind a proxy if you need it.

---

## 1. Service endpoints

### `GET /health`
```json
{"status": "healthy"}
```

### `GET /`
Endpoint directory (`message`, `endpoints`).

### `GET /docs`
Interactive OpenAPI docs (FastAPI). LangServe routes also expose `/…/playground`.

---

## 2. OpenAI-compatible surface

### `GET /v1/models`

```json
{"object":"list","data":[{"id":"chabo","object":"model","created":1756252800,"owned_by":"chabo"}]}
```

`id` comes from `[api] model_name`. It labels the **deployment**, not the LLM — which model
answers is fixed per task in `params.cfg` and cannot be selected by a client.

### `POST /v1/chat/completions`

**Request**

| Field | Type | Req | Notes |
|---|---|---|---|
| `messages` | `[{role, content}]` | ✅ | `content` may be a string or an OpenAI parts array; only `type: "text"` parts are kept. Non-empty; must contain ≥1 `user` turn. |
| `model` | `string` | — | Echoed back. Falls back to `[api] model_name`. |
| `stream` | `bool` | — | Default `false`. `true` → SSE. |
| `files` | `[{name, content}]` | — | **Non-standard.** This turn's attachments as already-extracted **text**. |

Unknown fields are accepted and ignored.

**Conversation handling** — the last `user` message is the query. History is truncated to
`[conversation_history] MAX_TURNS` / `MAX_CHARS`. A user-turn-only history is passed
separately to filter extraction. Client `system` messages are ignored (the system prompt is
code-owned, `generator/prompts.py`).

**Response — non-streaming** (`stream: false`), `200 application/json`:

```json
{
  "id": "chatcmpl-<hex>",
  "object": "chat.completion",
  "created": 1756252800,
  "model": "chabo",
  "choices": [{"index":0,"message":{"role":"assistant","content":"…markdown…"},"finish_reason":"stop"}],
  "citations": ["https://…", "Document title"]
}
```

`content` is the full markdown body: answer, then an optional filters footnote
(`---\n*🔍 Searched within: …*`), then a `**Sources:**` numbered list.
`citations` is present only when `[api] openai_citations = true` and sources exist.

**Response — streaming** (`stream: true`), `200 text/event-stream`. Frames are
`data: {chat.completion.chunk}\n\n`, terminated by `data: [DONE]\n\n`.

| Order | Frame delta |
|---|---|
| 1 | `{"role":"assistant","content":""}` (prelude) |
| 2..n | `{"content":"<token>"}` |
| — | `{"content":"\n\n---\n*🔍 Searched within: …*"}` (only if filters applied) |
| — | `{"content":"\n\n**Sources:**\n1. …"}` — this frame may carry a top-level `citations` array |
| last | `{}` with `"finish_reason":"stop"`, then `[DONE]` |

Errors after headers are sent arrive as a content frame (`\n\nError: …`) followed by the
finish frame — never a truncated stream.

**Errors** (before streaming starts):

| Code | Condition |
|---|---|
| `400` | empty `messages`; no user message; a `files` entry with no text |
| `422` | request body fails schema validation |
| `500` | attachment processing failure |

**Documented deviations from OpenAI**
- Sampling params (`temperature`, `top_p`, `max_tokens`, …) are accepted and **ignored**.
- `usage` is omitted (token accounting isn't uniform across providers).
- Client `system` messages are ignored.
- `files` and top-level `citations` are non-standard additions.
- No `/v1/completions`, no tools/function calling, no `n > 1`, no logprobs.

**Attachments.** Generic frontends extract file text themselves, so `content` must already be
plain text. It is cleaned, chunked and capped by `[ingestor]` exactly as a server-parsed PDF
would be. `name` is the citation label only. Attachments are **per request** — there is no
server-side store, so a follow-up turn must resend them.

```bash
curl -N http://localhost:7860/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"chabo","stream":true,
       "files":[{"name":"report.pdf","content":"Section 1. …"}],
       "messages":[{"role":"user","content":"Summarise the attached report."}]}'
```

---

## 3. LangServe surface (Chabo-ChatUI)

Mounted via `langserve.add_routes` over a `RunnableLambda`, output type `str`. Each path
exposes the standard set: `/invoke`, `/batch`, `/stream`, `/stream_log`, `/input_schema`,
`/output_schema`, `/config_schema`, `/playground`, plus `/feedback` and
`/public_trace_link` (both enabled).

Requests wrap the payload in `{"input": …}`; `/stream` returns SSE with `event: data` frames
whose `data` is a JSON-encoded string chunk, ending in `event: end`.

### `/chatfed-ui-stream` — text

```json
{"input": {"messages": [{"role":"user","content":"…","id":"optional"}],
           "preprompt": null}}
```

| Field | Type | Notes |
|---|---|---|
| `messages` | `[{role, content, id?}]` | `role` ∈ `user` \| `assistant` \| `system` |
| `preprompt` | `string?` | Accepted, unused |

### `/chatfed-with-file-stream` — text + upload

```json
{"input": {"messages": [...],
           "files": [{"name":"doc.pdf","type":"base64","content":"<base64>"}]}}
```

| Field | Type | Notes |
|---|---|---|
| `files` | `[{name, type, content}]` | Only the **first** entry is used; `type` must be `"base64"` |
| `messages`, `preprompt` | as above | |

Unlike the OpenAI surface, this route accepts the **raw file** and parses it server-side
(PDF via PyPDF2, DOCX via python-docx, `.txt`/`.md` as-is). A decode failure yields
`Error: …` as stream text, not an HTTP error.

**Output.** A plain markdown string (same body as the non-streaming OpenAI `content`), with
two ChatUI-specific quirks applied by the renderer: a `TRAILING_FLUSH_DELAY` sleep before
trailing frames, and `doc://#` substituted for sources with no URL (the ChatUI fork only
renders citations whose link carries an accepted scheme).

---

## 4. Cross-cutting behaviour

**Guardrails** (both surfaces, both off by default — see `params.cfg`):
- *Input guard* — classified pre-retrieval. On block the response is the fixed
  `[input_guard] blocked_message`, delivered through the normal success path (HTTP `200`).
- *Output blocklist / classifier* — applied to the answer as it streams. On a hit the stream
  stops, `blocklist_message` / `classification_message` is emitted as a notice, and the
  footnote and sources are suppressed. Text already streamed stays visible.
- Both guards **fail open**: an error or timeout allows the content and logs.

**Citations.** Sources are always rendered as markdown in the message body — the only form
that works in every frontend. The OpenAI `citations` array is strictly additive.

**Statelessness.** No sessions, no stored documents. Every request carries its own history
and attachments.

**Configuration** (`params.cfg`): `[api] model_name`, `[api] openai_citations`,
`[conversation_history] MAX_TURNS`/`MAX_CHARS`, `[ingestor] max_chunks`/`chunk_size`,
`[input_guard]`, `[output_guard]`. Env var `TRAILING_FLUSH_DELAY` (default `0.05`) affects
the LangServe path only.
