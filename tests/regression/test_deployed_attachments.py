"""Regression checks for documents attached to deployed requests."""

import base64
import json
import os
import re
import unittest
from urllib.request import Request, urlopen

from huggingface_hub import get_token


class DeployedAttachmentTests(unittest.TestCase):
    DOCUMENT = (
        "The fictional Lyra Workshop uses the access code CEDAR-6842. "
        "Its check-in desk is at Amber Lantern Hall."
    )
    QUESTION = (
        "According to the attached Lyra Workshop document, "
        "what is its access code? Include a source citation."
    )

    def setUp(self):
        self.base_url = os.environ.get(
            "REGRESSION_ORCHESTRATOR_URL", ""
        ).strip().rstrip("/")
        self.assertTrue(
            self.base_url, "REGRESSION_ORCHESTRATOR_URL is required"
        )
        self.token = os.environ.get("HF_TOKEN") or get_token()
        self.assertTrue(self.token, "Hugging Face authentication is required")

    def request(self, path, body, accept):
        return Request(
            self.base_url + path,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": accept,
            },
            method="POST",
        )

    def check_attachment_answer(self, answer, filename):
        self.assertIsInstance(answer, str)
        self.assertIn("**Sources:**", answer)
        text, sources = answer.split("**Sources:**", 1)

        self.assertIn(
            "CEDAR-6842",
            text,
            "The answer did not contain the attachment's access code",
        )

        references = {
            int(number)
            for group in re.findall(
                r"\[(\d+(?:\s*,\s*\d+)*)\]", text
            )
            for number in re.findall(r"\d+", group)
        }
        source_entries = {
            int(number): entry
            for number, entry in re.findall(
                r"(?m)^(\d+)\.\s+(.+)$", sources
            )
        }

        self.assertTrue(references, "The answer contained no citations")
        self.assertTrue(
            references <= set(source_entries),
            f"Citations have no matching source entries:\n{answer}",
        )
        self.assertTrue(
            any(filename in source_entries[number] for number in references),
            f"The answer did not cite the attached document:\n{answer}",
        )

    def test_langserve_txt_attachment(self):
        filename = "reg-upload-lyra.txt"
        body = {
            "input": {
                "messages": [
                    {"role": "user", "content": self.QUESTION}
                ],
                "files": [{
                    "name": filename,
                    "type": "base64",
                    "content": base64.b64encode(
                        self.DOCUMENT.encode("utf-8")
                    ).decode("ascii"),
                }],
            }
        }

        events = []
        event_name = "message"
        data_lines = []

        request = self.request(
            "/chatfed-with-file-stream/stream",
            body,
            "text/event-stream",
        )
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
                        events.append(
                            (event_name, "\n".join(data_lines))
                        )
                    event_name = "message"
                    data_lines = []
                elif line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].removeprefix(" "))

        self.assertFalse(data_lines, "An SSE frame was incomplete")
        self.assertEqual(event_name, "message")
        self.assertTrue(events, "The stream contained no events")
        self.assertNotIn("error", [name for name, _ in events])
        self.assertEqual(events[-1][0], "end")
        self.assertEqual(sum(name == "end" for name, _ in events), 1)

        chunks = []
        for name, data in events:
            if name == "data":
                chunk = json.loads(data)
                self.assertIsInstance(chunk, str)
                chunks.append(chunk)

        self.assertTrue(chunks, "The stream contained no answer text")
        self.check_attachment_answer("".join(chunks), filename)

    def test_openai_extracted_text_attachment(self):
        filename = "reg-upload-lyra-openai.txt"
        request = self.request(
            "/v1/chat/completions",
            {
                "messages": [
                    {"role": "user", "content": self.QUESTION}
                ],
                "stream": False,
                "files": [{
                    "name": filename,
                    "content": self.DOCUMENT,
                }],
            },
            "application/json",
        )

        with urlopen(request, timeout=120) as response:
            self.assertEqual(response.status, 200)
            self.assertTrue(
                response.headers.get("Content-Type", "").startswith(
                    "application/json"
                )
            )
            result = json.load(response)

        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(len(result["choices"]), 1)
        choice = result["choices"][0]
        self.assertEqual(choice["index"], 0)
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["message"]["role"], "assistant")
        self.check_attachment_answer(
            choice["message"]["content"], filename
        )


if __name__ == "__main__":
    unittest.main()
