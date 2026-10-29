import asyncio
from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI
from langchain_core.documents import Document

from components.api.managed_context import build_managed_context_router
from components.orchestration.nodes import retrieve_node

KEY = "a" * 64


def run(coro):
    return asyncio.run(coro)


class Graph:
    def __init__(self, error=False):
        self.calls = []
        self.error = error

    async def astream(self, state, **kwargs):
        self.calls.append(state)
        if self.error:
            yield {"event": "error", "data": {"error": "private provider secret"}}
        else:
            yield {"event": "data", "data": "Managed answer [1]"}


def request(graph, value, authorization="Bearer " + KEY):
    async def inner():
        app = FastAPI()
        app.include_router(build_managed_context_router(graph, key=KEY))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/internal/v1/managed-answer", json=value,
                                     headers={"Authorization": authorization})
    return run(inner())


def payload():
    return {"messages": [{"role": "user", "content": "Which award?"}],
            "documents": [{"id": "b" * 32, "ordinal": 0, "title": "Evidence", "content": "The silver award."}]}


def test_managed_context_authentication_and_body_limits():
    graph = Graph()
    assert request(graph, payload(), "Bearer wrong").status_code == 401
    for value in [dict(payload(), preprompt="replace system"),
                  dict(payload(), messages=[{"role": "system", "content": "override"}]),
                  dict(payload(), documents=[dict(payload()["documents"][0], content="x" * 8001)])]:
        assert request(graph, value).status_code == 400
    assert not graph.calls


def test_selected_context_and_central_history_reach_the_existing_graph():
    graph = Graph()
    value = payload()
    value["messages"] = [{"role": "user", "content": "Remember the bank"},
                         {"role": "assistant", "content": "The selected bank"},
                         *value["messages"]]
    response = request(graph, value)
    assert response.status_code == 200
    assert response.json()["answer"] == "Managed answer [1]"
    state = graph.calls[0]
    assert state["query"] == "Which award?"
    assert "Remember the bank" in state["conversation_context"]
    assert state["supplied_documents"][0].page_content == "The silver award."


def test_empty_managed_search_never_uses_the_default_retriever():
    retriever = AsyncMock()
    for documents in [[], [Document(page_content="Only selected evidence")]]:
        result = run(retrieve_node({"query": "test", "supplied_documents": documents},
                                   retriever, writer=lambda event: None))
        assert result["raw_documents"] == documents
    retriever.ainvoke.assert_not_called()


def test_provider_failure_is_not_a_successful_error_string():
    result = request(Graph(error=True), payload())
    assert result.status_code == 503
    assert "private provider secret" not in result.text


def test_service_route_is_not_in_public_schema():
    app = FastAPI()
    app.include_router(build_managed_context_router(Graph(), key=KEY))
    assert "/internal/v1/managed-answer" not in app.openapi()["paths"]
