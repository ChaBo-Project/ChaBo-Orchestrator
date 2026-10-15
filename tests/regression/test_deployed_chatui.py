"""Regression checks against a deployed Hugging Face Orchestrator."""

import json
import os
import re
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from huggingface_hub import get_token


class DeployedChatUITests(unittest.TestCase):
    def stream_answer(self, messages, expected_source):
        base_url = os.environ.get("REGRESSION_ORCHESTRATOR_URL", "").strip()
        self.assertTrue(base_url, "REGRESSION_ORCHESTRATOR_URL is required")
        base_url = base_url.rstrip("/")
        token = os.environ.get("HF_TOKEN") or get_token()
        self.assertTrue(token, "HF_TOKEN or a saved Hugging Face login is required")

        body = {"input": {"messages": messages}}
        request = Request(
            base_url + "/chatfed-ui-stream/stream",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )

        events = []
        event_name = "message"
        data_lines = []

        with urlopen(request, timeout=120) as response:
            self.assertEqual(response.status, 200)
            self.assertTrue(
                response.headers.get("Content-Type", "").startswith(
                    "text/event-stream"
                )
            )

            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line:
                    if data_lines or event_name != "message":
                        events.append((event_name, "\n".join(data_lines)))
                    event_name = "message"
                    data_lines = []
                elif line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].removeprefix(" "))

        self.assertTrue(events, "The stream contained no events")
        self.assertFalse(data_lines, "The final SSE frame was incomplete")
        self.assertEqual(event_name, "message", "An SSE frame was incomplete")
        self.assertNotIn("error", [name for name, _ in events])
        self.assertEqual(events[-1][0], "end", "Missing stream completion")
        self.assertEqual(sum(name == "end" for name, _ in events), 1)

        chunks = []
        for name, data in events:
            if name == "data":
                chunk = json.loads(data)
                self.assertIsInstance(chunk, str)
                chunks.append(chunk)

        self.assertGreater(
            len(chunks),
            1,
            f"Expected multiple text chunks; received {len(chunks)}. "
            f"Response: {''.join(chunks)!r}",
        )
        answer = "".join(chunks)
        self.assertIn("**Sources:**", answer)
        text, sources = answer.split("**Sources:**", 1)
        self.assertTrue(text.strip(), "The answer was empty")

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
        return answer

    def test_stream_returns_answer_citation_and_completion(self):
        self.stream_answer(
            [{
                "role": "user",
                "content": (
                    "What is the training grant per participant in the "
                    "Orion Training Programme, and when do applications close?"
                ),
            }],
            "reg-orion-grant.txt",
        )

    def test_follow_up_with_conversation_history(self):
        cases = [
            (
                "Orion Training Programme",
                "reg-orion-grant.txt",
                "420",
                "175",
            ),
            (
                "Vega Gardening Programme",
                "reg-vega-grant.txt",
                "175",
                "420",
            ),
        ]

        for programme, source, expected_amount, other_amount in cases:
            with self.subTest(programme=programme):
                question = (
                    f"When do applications close for the {programme}?"
                )
                first_answer = self.stream_answer(
                    [{"role": "user", "content": question}],
                    source,
                )

                answer = self.stream_answer(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": first_answer},
                        {
                            "role": "user",
                            "content": (
                                "How much is its grant per participant? "
                                "Give the amount in digits for only the "
                                "programme we were discussing, "
                                "and include a source citation."
                            ),
                        },
                    ],
                    source,
                )

                answer_text = answer.split("**Sources:**", 1)[0]
                self.assertRegex(
                    answer_text,
                    rf"\b{expected_amount}\b",
                    "The follow-up did not give the expected grant",
                )
                self.assertNotRegex(
                    answer_text,
                    rf"\b{other_amount}\b",
                    "The follow-up included the other programme's grant",
                )

    def test_invalid_request_is_rejected(self):
        base_url = os.environ.get("REGRESSION_ORCHESTRATOR_URL", "").strip()
        self.assertTrue(base_url, "REGRESSION_ORCHESTRATOR_URL is required")
        base_url = base_url.rstrip("/")
        token = os.environ.get("HF_TOKEN") or get_token()
        self.assertTrue(token, "Hugging Face authentication is required")

        # Each message requires a content field.
        body = {"input": {"messages": [{"role": "user"}]}}
        request = Request(
            base_url + "/chatfed-ui-stream/stream",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with self.assertRaises(HTTPError) as caught:
            with urlopen(request, timeout=60):
                pass

        with caught.exception as error:
            self.assertEqual(error.code, 422)
            self.assertTrue(
                error.headers.get("Content-Type", "").startswith(
                    "application/json"
                )
            )
            response = json.load(error)

        self.assertIsInstance(response.get("detail"), list)
        self.assertTrue(response["detail"], "Missing validation error details")
        self.assertTrue(
            any("content" in item.get("loc", []) for item in response["detail"]),
            f"Expected an error about missing content: {response}",
        )

    def test_private_space_rejects_missing_and_invalid_tokens(self):
        base_url = os.environ.get("REGRESSION_ORCHESTRATOR_URL", "").strip()
        self.assertTrue(base_url, "REGRESSION_ORCHESTRATOR_URL is required")
        base_url = base_url.rstrip("/")
        token = os.environ.get("HF_TOKEN") or get_token()
        self.assertTrue(token, "Hugging Face authentication is required")

        url = base_url + "/chatfed-ui-stream/input_schema"

        # Confirm the route exists and is accessible with valid authentication.
        request = Request(
            url,
            headers={"Authorization": f"Bearer {token}"},
        )
        with urlopen(request, timeout=60) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.load(response)["title"], "ChatUIInput")

        cases = {
            "missing_token": {},
            "invalid_token": {
                "Authorization": "Bearer invalid-regression-token"
            },
        }
        for name, headers in cases.items():
            with self.subTest(authentication=name):
                request = Request(url, headers=headers)
                with self.assertRaises(HTTPError) as caught:
                    with urlopen(request, timeout=60):
                        pass

                with caught.exception as error:
                    # Private resources may be concealed with HTTP 404.
                    self.assertIn(error.code, (401, 403, 404))

if __name__ == "__main__":
    unittest.main()
