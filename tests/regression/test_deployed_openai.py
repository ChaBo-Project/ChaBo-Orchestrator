"""Regression checks for the deployed OpenAI-compatible API."""

import json
import os
import re
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from huggingface_hub import get_token


QUESTION = (
    "What is the training grant per participant in the "
    "Orion Training Programme, and when do applications close?"
)


class DeployedOpenAITests(unittest.TestCase):
    def setUp(self):
        self.base_url = os.environ.get(
            "REGRESSION_ORCHESTRATOR_URL", ""
        ).strip().rstrip("/")
        self.assertTrue(
            self.base_url, "REGRESSION_ORCHESTRATOR_URL is required"
        )
        self.token = os.environ.get("HF_TOKEN") or get_token()
        self.assertTrue(
            self.token, "HF_TOKEN or a saved Hugging Face login is required"
        )

    def request(self, path, body=None):
        headers = {"Authorization": f"Bearer {self.token}"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        return Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method="POST" if body is not None else "GET",
        )

    def read_json(self, path, body=None):
        with urlopen(self.request(path, body), timeout=120) as response:
            self.assertEqual(response.status, 200)
            self.assertTrue(
                response.headers.get("Content-Type", "").startswith(
                    "application/json"
                )
            )
            return json.load(response)

    def check_answer(self, answer, expected_source):
        self.assertIsInstance(answer, str)
        self.assertTrue(answer.strip(), "The answer was empty")
        self.assertIn("**Sources:**", answer)

        text, sources = answer.split("**Sources:**", 1)
        self.assertTrue(text.strip(), "Missing answer before sources")

        # Recognize both [1] and grouped references such as [1, 3].
        references = set()
        for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
            references.update(int(value) for value in group.split(","))

        source_numbers = {
            int(value)
            for value in re.findall(r"(?m)^(\d+)\.\s", sources)
        }
        self.assertTrue(references, "The answer contained no citations")
        self.assertTrue(
            references <= source_numbers,
            (
                f"Citation numbers: {sorted(references)}\n"
                f"Source numbers: {sorted(source_numbers)}\n"
                f"Full response:\n{answer}"
            ),
        )
        self.assertIn(expected_source, sources)

    def complete(self, messages, expected_source):
        result = self.read_json(
            "/v1/chat/completions",
            {"messages": messages, "stream": False},
        )
        self.assertEqual(result["object"], "chat.completion")
        self.assertIsInstance(result["id"], str)
        self.assertTrue(result["id"].startswith("chatcmpl-"))
        self.assertIsInstance(result["created"], int)
        self.assertIsInstance(result["model"], str)
        self.assertTrue(result["model"])
        self.assertIsInstance(result["choices"], list)
        self.assertEqual(len(result["choices"]), 1)

        choice = result["choices"][0]
        self.assertEqual(choice["index"], 0)
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["message"]["role"], "assistant")

        answer = choice["message"]["content"]
        self.check_answer(answer, expected_source)

        if "citations" in result:
            self.assertIsInstance(result["citations"], list)
            self.assertTrue(result["citations"])
            for citation in result["citations"]:
                self.assertIsInstance(citation, str)
                self.assertTrue(citation)

        return answer

    def test_models_returns_deployment_model(self):
        result = self.read_json("/v1/models")
        self.assertEqual(result["object"], "list")
        self.assertIsInstance(result["data"], list)
        self.assertTrue(result["data"])

        for model in result["data"]:
            self.assertEqual(model["object"], "model")
            self.assertIsInstance(model["id"], str)
            self.assertTrue(model["id"])
            self.assertIsInstance(model["created"], int)
            self.assertIsInstance(model["owned_by"], str)

    def test_non_streaming_returns_answer_and_sources(self):
        self.complete(
            [{"role": "user", "content": QUESTION}],
            "reg-orion-grant.txt",
        )

    def test_streaming_returns_chunks_and_completion(self):
        body = {
            "messages": [{"role": "user", "content": QUESTION}],
            "stream": True,
        }
        frames = []
        data_lines = []

        with urlopen(
            self.request("/v1/chat/completions", body), timeout=120
        ) as response:
            self.assertEqual(response.status, 200)
            self.assertTrue(
                response.headers.get("Content-Type", "").startswith(
                    "text/event-stream"
                )
            )
            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line:
                    if data_lines:
                        frames.append("\n".join(data_lines))
                    data_lines = []
                elif line.startswith("data:"):
                    data_lines.append(line[5:].removeprefix(" "))

        self.assertFalse(data_lines, "Incomplete final SSE frame")
        self.assertGreater(len(frames), 2)
        self.assertEqual(frames[-1], "[DONE]")
        self.assertEqual(frames.count("[DONE]"), 1)

        chunks = [json.loads(frame) for frame in frames[:-1]]
        first = chunks[0]
        self.assertIsInstance(first["id"], str)
        self.assertTrue(first["id"].startswith("chatcmpl-"))
        self.assertIsInstance(first["created"], int)
        self.assertIsInstance(first["model"], str)
        self.assertEqual(
            first["choices"][0]["delta"]["role"], "assistant"
        )

        text_chunks = []
        finished = False
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk["object"], "chat.completion.chunk")
            self.assertEqual(chunk["id"], first["id"])
            self.assertEqual(chunk["created"], first["created"])
            self.assertEqual(chunk["model"], first["model"])
            self.assertEqual(len(chunk["choices"]), 1)

            choice = chunk["choices"][0]
            self.assertEqual(choice["index"], 0)
            self.assertIsInstance(choice["delta"], dict)
            self.assertFalse(finished, "Received a chunk after completion")

            if "content" in choice["delta"]:
                content = choice["delta"]["content"]
                self.assertIsInstance(content, str)
                if content:
                    text_chunks.append(content)

            if choice["finish_reason"] is not None:
                self.assertEqual(choice["finish_reason"], "stop")
                self.assertEqual(index, len(chunks) - 1)
                finished = True

        self.assertTrue(finished, "Missing finish_reason: stop")
        self.assertGreater(len(text_chunks), 1)
        self.check_answer(
            "".join(text_chunks), "reg-orion-grant.txt"
        )

    def check_rejected(self, messages):
        body = {"messages": messages, "stream": False}
        with self.assertRaises(HTTPError) as caught:
            with urlopen(
                self.request("/v1/chat/completions", body), timeout=60
            ):
                pass

        with caught.exception as error:
            self.assertEqual(error.code, 400)
            self.assertTrue(
                error.headers.get("Content-Type", "").startswith(
                    "application/json"
                )
            )
            result = json.load(error)

        self.assertIsInstance(result.get("detail"), str)
        self.assertTrue(result["detail"])

    def test_empty_messages_are_rejected(self):
        self.check_rejected([])

    def test_messages_without_user_turn_are_rejected(self):
        self.check_rejected([
            {"role": "assistant", "content": "Previous answer"}
        ])

    def test_follow_up_uses_conversation_history(self):
        first_answer = self.complete(
            [{"role": "user", "content": QUESTION}],
            "reg-orion-grant.txt",
        )
        self.complete(
            [
                {"role": "user", "content": QUESTION},
                {"role": "assistant", "content": first_answer},
                {
                    "role": "user",
                    "content": (
                        "Where is its induction session held, "
                        "and what must participants bring?"
                    ),
                },
            ],
            "reg-orion-induction.txt",
        )


if __name__ == "__main__":
    unittest.main()
