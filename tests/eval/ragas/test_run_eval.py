
import asyncio

import httpx
import pytest

from run_eval import EvalAuthError, run_case


def _client(status_code: int, calls: list) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status_code, json={"status": "ok", "answer": "a", "retrieved_contexts": []})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("status_code", [401, 404])
def test_auth_rejection_raises_without_retry(status_code):
    calls = []

    async def go():
        async with _client(status_code, calls) as client:
            await run_case(client, "http://x", "bad-token", "q?")

    with pytest.raises(EvalAuthError):
        asyncio.run(go())
    assert len(calls) == 1  # failed fast, no retry


def test_success_passes_through():
    calls = []

    async def go():
        async with _client(200, calls) as client:
            return await run_case(client, "http://x", "tok", "q?")

    assert asyncio.run(go())["status"] == "ok"
