"""Verify the deployed Qdrant corpus using pinned synthetic vectors."""

import json
import os
import unittest
from urllib.request import Request, urlopen

from huggingface_hub import get_token, hf_hub_download


class DeployedRetrievalTests(unittest.TestCase):
    def test_each_synthetic_document_is_retrievable(self):
        token = os.environ.get("HF_TOKEN") or get_token()
        self.assertTrue(token, "Hugging Face authentication is required")

        dataset_path = hf_hub_download(
            repo_id="GIZ/chabo-regression-embeddings",
            repo_type="dataset",
            filename="train.json",
            revision="c05ec1ab390c029ebc100314a5359ed5fdf738bd",
            token=token,
        )
        with open(dataset_path, encoding="utf-8") as dataset_file:
            documents = json.load(dataset_file)

        self.assertEqual(len(documents), 3)
        base_url = os.environ.get(
            "REGRESSION_QDRANT_URL",
            "https://giz-chabo-regression-qdrant.hf.space",
        ).rstrip("/")
        collection = os.environ.get(
            "REGRESSION_COLLECTION", "chabo-regression-v1"
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        for document in documents:
            metadata = document["payload"]["metadata"]
            with self.subTest(document_id=metadata["document_id"]):
                # Searching with a document's own vector should retrieve itself.
                request = Request(
                    base_url + "/gradio_api/call/query_points",
                    data=json.dumps({
                        "data": [
                            json.dumps(document["vector"]),
                            collection,
                            1,
                            "",
                            "",
                            False,
                        ]
                    }).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urlopen(request, timeout=60) as response:
                    event_id = json.load(response)["event_id"]

                request = Request(
                    base_url + "/gradio_api/call/query_points/" + event_id,
                    headers={"Authorization": f"Bearer {token}"},
                )
                completed = None
                event_name = ""
                with urlopen(request, timeout=60) as response:
                    for raw_line in response:
                        line = raw_line.decode("utf-8").rstrip("\r\n")
                        if line.startswith("event:"):
                            event_name = line[6:].strip()
                            self.assertNotEqual(
                                event_name, "error", "Qdrant API call failed"
                            )
                        elif line.startswith("data:") and event_name == "complete":
                            completed = json.loads(line[5:].strip())

                self.assertIsNotNone(completed, "Missing API completion event")
                self.assertIsInstance(completed, list)
                self.assertEqual(len(completed), 1)
                results = completed[0]
                if isinstance(results, str):
                    results = json.loads(results)

                self.assertIsInstance(results, list, f"Retrieval failed: {results}")
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["answer_metadata"], metadata)
                self.assertEqual(
                    results[0]["answer"], document["payload"]["text"]
                )


if __name__ == "__main__":
    unittest.main()
