"""
Tests for the OpenAI-compatible surface (components/api/openai_compat.py).

*No network required (.env / services etc.) — the LangGraph app is replaced by a fake that
emits the same custom event stream the real graph emits, so these exercise the wire contract
and the request plumbing, not the RAG pipeline.

Covers:
- /v1/models discovery
- streaming SSE shape and non-streaming JSON shape
- conversation history, message content-parts, and rejected requests
- text attachments reaching the graph state, chunked and labelled
"""
import configparser
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from components.api import build_openai_router


class FakeGraph:
    """Stands in for the compiled LangGraph app; records the state it was invoked with."""

    def __init__(self, events=None):
        self.events = events if events is not None else [
            {"event": "data", "data": "Wheat is sown "},
            {"event": "data", "data": "in November [1]."},
            {"event": "filters_applied", "data": {"filters": {"crop_type": ["wheat"]}, "narrowed": False}},
            {"event": "final_answer", "data": {
                "text": "Wheat is sown in November [1].",
                "webSources": [{"title": "Wheat guide", "uri": "https://example.org/wheat"}],
            }},
        ]
        self.last_state = None

    async def astream(self, state, stream_mode=None):
        self.last_state = state
        for event in self.events:
            yield event


def build_client(graph=None, **kwargs):
    graph = graph or FakeGraph()
    kwargs.setdefault("config", configparser.ConfigParser())
    app = FastAPI()
    app.include_router(build_openai_router(graph, **kwargs))
    return TestClient(app), graph


def sse_frames(body: str):
    """Parse the JSON payload of every `data:` frame, skipping the [DONE] sentinel."""
    return [
        json.loads(line[len("data: "):])
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def sse_content(body: str) -> str:
    """Concatenate the content deltas of an SSE response, as a client would."""
    return "".join(f["choices"][0]["delta"].get("content", "") for f in sse_frames(body))


ONE_TURN = {"messages": [{"role": "user", "content": "When is wheat sown?"}]}


# --- discovery ---

def test_models_lists_the_configured_deployment_id():
    client, _ = build_client(model_name="chabo-egypt")
    data = client.get("/v1/models").json()
    assert data["object"] == "list"
    assert [m["id"] for m in data["data"]] == ["chabo-egypt"]


# --- non-streaming ---

def test_non_streaming_returns_one_chat_completion_with_the_whole_answer():
    client, _ = build_client()
    body = client.post("/v1/chat/completions", json=ONE_TURN).json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "Wheat is sown in November [1]." in body["choices"][0]["message"]["content"]


def test_non_streaming_answer_carries_the_footnote_and_sources_markdown():
    client, _ = build_client()
    content = client.post("/v1/chat/completions", json=ONE_TURN).json()["choices"][0]["message"]["content"]
    assert "Searched within" in content
    assert "**Sources:**" in content
    assert "[Wheat guide](https://example.org/wheat)" in content


def test_non_streaming_response_carries_the_citations_array_when_enabled():
    client, _ = build_client(include_citations=True)
    body = client.post("/v1/chat/completions", json=ONE_TURN).json()
    assert body["citations"] == ["https://example.org/wheat"]


def test_citations_array_is_absent_when_the_enhancement_is_off():
    client, _ = build_client(include_citations=False)
    body = client.post("/v1/chat/completions", json=ONE_TURN).json()
    assert "citations" not in body
    # The universal markdown rendering is unaffected by the toggle.
    assert "**Sources:**" in body["choices"][0]["message"]["content"]


def test_the_requested_model_id_is_echoed_back():
    client, _ = build_client(model_name="chabo")
    body = client.post("/v1/chat/completions", json={**ONE_TURN, "model": "whatever-the-ui-sent"}).json()
    assert body["model"] == "whatever-the-ui-sent"


# --- streaming ---

def test_streaming_response_is_sse():
    client, _ = build_client()
    response = client.post("/v1/chat/completions", json={**ONE_TURN, "stream": True})
    assert response.headers["content-type"].startswith("text/event-stream")


def test_streaming_opens_with_a_role_delta_and_ends_with_stop_then_done():
    client, _ = build_client()
    body = client.post("/v1/chat/completions", json={**ONE_TURN, "stream": True}).text
    frames = sse_frames(body)
    assert frames[0]["choices"][0]["delta"]["role"] == "assistant"
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert body.rstrip().endswith("data: [DONE]")


def test_streamed_deltas_reassemble_into_the_same_answer_as_non_streaming():
    client, _ = build_client()
    streamed = sse_content(client.post("/v1/chat/completions", json={**ONE_TURN, "stream": True}).text)
    single = client.post("/v1/chat/completions", json=ONE_TURN).json()["choices"][0]["message"]["content"]
    assert streamed == single


def test_a_pipeline_error_event_is_surfaced_to_the_client_as_content():
    graph = FakeGraph(events=[{"event": "error", "data": {"error": "retriever exploded"}}])
    client, _ = build_client(graph=graph)
    body = client.post("/v1/chat/completions", json=ONE_TURN).json()
    assert "retriever exploded" in body["choices"][0]["message"]["content"]


# --- request unpacking ---

def test_the_latest_user_message_becomes_the_query():
    client, graph = build_client()
    client.post("/v1/chat/completions", json={"messages": [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]})
    assert graph.last_state["query"] == "second"


def test_earlier_turns_become_conversation_context_and_user_only_history():
    client, graph = build_client()
    client.post("/v1/chat/completions", json={"messages": [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "an assistant answer"},
        {"role": "user", "content": "second"},
    ]})
    assert "first" in graph.last_state["conversation_context"]
    # Filter extraction must not see assistant text — it causes spurious filter matches.
    assert "an assistant answer" not in graph.last_state["user_messages_history"]
    assert "first" in graph.last_state["user_messages_history"]


def test_content_parts_arrays_are_flattened_to_text():
    client, graph = build_client()
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": [
        {"type": "text", "text": "When is wheat sown?"},
        {"type": "image_url", "image_url": {"url": "https://example.org/x.png"}},
    ]}]})
    assert graph.last_state["query"] == "When is wheat sown?"


def test_unknown_sampling_parameters_are_accepted_and_ignored():
    client, _ = build_client()
    response = client.post("/v1/chat/completions", json={**ONE_TURN, "temperature": 0.9, "top_p": 0.3})
    assert response.status_code == 200


def test_an_empty_messages_array_is_rejected():
    client, _ = build_client()
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400


def test_a_conversation_with_no_user_turn_is_rejected():
    client, _ = build_client()
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "system", "content": "hi"}]})
    assert response.status_code == 400


# --- attachments ---

def test_a_text_attachment_is_chunked_into_the_graph_state():
    client, graph = build_client()
    client.post("/v1/chat/completions", json={**ONE_TURN, "files": [
        {"name": "report.pdf", "content": "Wheat is sown in November."},
    ]})
    # process_text applies the same chunk markers an uploaded file would get.
    assert graph.last_state["ingestor_context"] == "[Chunk 1]: Wheat is sown in November."


def test_several_attachments_are_rejected():
    client, _ = build_client()
    response = client.post("/v1/chat/completions", json={**ONE_TURN, "files": [
        {"name": "a.pdf", "content": "first doc"},
        {"name": "b.pdf", "content": "second doc"},
    ]})
    assert response.status_code == 400


def test_attachment_names_become_the_citation_label():
    # generate_node_streaming builds the ingestor Document with state["filename"], falling
    # back to "unknown" — so without this the answer cites an attachment as "unknown".
    client, graph = build_client()
    client.post("/v1/chat/completions", json={**ONE_TURN, "files": [
        {"name": "memo.txt", "content": "uploaded text"},
    ]})
    assert graph.last_state["filename"] == "memo.txt"
    # Nothing to parse: ingest_node must still skip, which it does on file_content.
    assert graph.last_state.get("file_content") is None


def test_an_attachment_with_no_text_is_reported_rather_than_dropped():
    # Silently answering without the attachment the user made is the worse failure.
    client, _ = build_client()
    response = client.post("/v1/chat/completions", json={**ONE_TURN, "files": [
        {"name": "notes.pdf", "content": ""},
    ]})
    assert response.status_code == 400


def test_a_turn_with_no_attachment_seeds_no_ingestor_context():
    client, graph = build_client()
    client.post("/v1/chat/completions", json=ONE_TURN)
    assert "ingestor_context" not in graph.last_state
