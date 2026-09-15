# Connecting a frontend to ChaBo

ChaBo exposes two HTTP surfaces:

| Surface | Endpoints | Who it's for |
|---|---|---|
| **OpenAI-compatible** | `/v1/chat/completions`, `/v1/models` | Any modern frontend (OpenWebUI, LibreChat, an SDK, curl) — no bespoke connector |
| **LangServe** | `/chatfed-ui-stream`, `/chatfed-with-file-stream` | The existing Chabo-ChatUI, whose "langserve-streaming" connector understands LangServe's own envelope and nothing else |

Both run the same graph, the same guardrails, and the same citation rendering. They differ
only in the *renderer* that turns ChaBo's internal event stream into wire format
(`src/components/orchestration/renderers.py`). New frontends should use the OpenAI surface;
the LangServe routes are kept so the current ChatUI deployments don't break.

## `/v1/chat/completions`

Standard OpenAI request and response shapes, streaming (`stream: true`, SSE of
`chat.completion.chunk`) and non-streaming.

```bash
curl -N http://localhost:7860/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"chabo","stream":true,
       "messages":[{"role":"user","content":"When is wheat sown?"}]}'
```

Deviations from OpenAI's API:
- Sampling parameters (`temperature`, `top_p`, `max_tokens`, …) are ignored.
- `usage` is omitted. Token accounting isn't available uniformly across inference providers
- Client-supplied `system` message is ignored. The system prompt is set in Chabo `prompts.py`
- 1 non-standard request field carries attachments, since no chat protocol does:
   - `files`: `[{"name": ..., "content": "<extracted text>"}]` — see below
- Responses may carry a top-level `citations` array (Perplexity-style). Sources are always *also*
  rendered as markdown in the message body.

`GET /v1/models` advertises a single model id (`[api] model_name`) — this is how OpenWebUI
and LibreChat discover the endpoint. It names the deployment, not the LLM.

## Attachments

Generic frontends do their own file extraction and don't send the original upload. So the only thing they can hand over is raw **text**:

```bash
curl -N http://localhost:7860/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"chabo","stream":true,
       "files":[{"name":"report.pdf","content":"Section 1. ..."}],
       "messages":[{"role":"user","content":"Summarise the attached report."}]}'
```

- The text is cleaned, chunked, and capped at `[ingestor] max_chunks` by `process_text()` —
  the same treatment a parsed PDF gets — so an attachment can't blow the context window just because the frontend did the extraction.
- `name` is the citation label. Nothing is parsed from it.
- An entry with no text is a **400**, not a quietly attachment-free answer.
- Attachments are per request. There is no server-side session, so a follow-up turn that
  should still see the document must include it again — which is what OpenWebUI does
  natively, since it treats an attachment as belonging to the conversation.

**Why there is no upload endpoint.** An earlier version of this branch had `POST
/v1/documents`: upload a file, get a `document_id`, pass the id on the chat call. The point
of that store was to hold the result of an expensive parse so it wasn't repeated — but no
generic frontend sends a file to parse, so it was caching a string to avoid re-sending a
string. It bought nothing for OpenWebUI, nothing for LibreChat (whose files never reach a
custom endpoint at all), and nothing for ChatUI (which uses the LangServe route), while
adding a TTL, a capacity cap, an id lifecycle, and a sticky-sessions requirement behind a
load balancer. It was removed.

If a client that genuinely holds raw bytes ever turns up, the cheap answer is to accept
base64 on this same `files` field and let `ingest_node` parse it — the ChatUI file route
already does exactly that. The expensive answer, a re-parse cache, belongs inside the
ingestor keyed on a content hash, not exposed as an endpoint with ids clients must manage.

## Per-UI notes

### OpenWebUI

1. Settings → Connections → add an OpenAI-compatible connection pointing at
   `http://<chabo-host>:7860/v1` (any non-empty API key). The model from `/v1/models`
   appears in the picker; chat and corpus retrieval work with no further setup.
2. For uploads, install the Pipe Function in
   [`docs/integrations/openwebui_pipe.py`](integrations/openwebui_pipe.py) (Workspace →
   Functions → paste → enable → set valves). It reads the text OpenWebUI already extracted
   and forwards it on the chat call as `files`.
3. In Admin Settings → Interface, point the **task model** (title / tag / search-query
   generation) at a plain LLM rather than ChaBo. Those background calls arrive as ordinary
   chat completions, so leaving them on ChaBo runs a full retrieval per generated title.

**The Pipe's file path is not yet verified against a live OpenWebUI.** What a Pipe can see of
an upload is unresolved upstream ([#19963](https://github.com/open-webui/open-webui/issues/19963),
[#17293](https://github.com/open-webui/open-webui/issues/17293)); the Pipe reads
`files[].data.content` and degrades to a corpus-only answer when that isn't there. Turning on
its `DEBUG_FILES` valve logs what actually arrives — that log is the spike.

### LibreChat

Add ChaBo as a [custom endpoint](https://www.librechat.ai/docs/quick_start/custom_endpoints)
with `baseURL: http://<chabo-host>:7860/v1`. Chat, streaming, and corpus retrieval work.
Files route through LibreChat's own [RAG API](https://www.librechat.ai/docs/features/rag_api)
service, which is a separate component from a custom endpoint — attachments never reach a
custom endpoint in any form, so nothing on this side can bridge them. Either point that
service at a ChaBo-backed implementation, or accept LibreChat's native file RAG for that
deployment (only ad hoc uploaded-doc handling is affected).

### Chabo-ChatUI

Works unchanged on the LangServe routes today but can be migrated to `/v1/chat/completions`. 
Three ChatUI-shaped workarounds can then be deprecated:

1. The `langserve-streaming` connector.
2. The `doc://#` placeholder link. The fork's `langserve-streaming` endpoint finds sources by
  running `/\[([^\]]+)\]\(((?:doc|https?):\/\/[^)]+)\)/g` over any chunk containing
  `**Sources:**`, and *also* truncates the displayed text at that marker — so a source whose
  link carries no accepted scheme is dropped from the sources panel **and** from the message
  body. Hence the placeholder. 
3. `TRAILING_FLUSH_DELAY` (the sleep that keeps ChatUI's parser from dropping coalesced
  trailing chunks). The OpenAI renderer's frames are self-delimiting, so it is not required.


## Adding another frontend

Subclass `BaseRenderer` in `src/components/orchestration/renderers.py` and pass an instance
to `consume_stream` (`src/components/orchestration/streaming.py`). Everything else — graph,
guards, footnote, citation logic — is shared: `consume_stream` is the only consumer of the
internal event stream, so a new frontend never forks the streaming logic.
