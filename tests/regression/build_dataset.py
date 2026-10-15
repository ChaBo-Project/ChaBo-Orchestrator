"""Build the synthetic regression dataset using the deployed embedding endpoint."""

import argparse
import getpass
import json
import math
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dimension", type=int, default=1024)
    args = parser.parse_args()

    if not args.endpoint.startswith("https://"):
        raise ValueError("The embedding endpoint must use HTTPS.")

    fixture = Path(__file__).parent / "fixtures" / "documents.json"
    documents = json.loads(fixture.read_text(encoding="utf-8-sig"))

    if not documents:
        raise ValueError("The fixture contains no documents.")

    ids = [document["id"] for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Document IDs must be unique.")

    token = getpass.getpass("Hugging Face token (hidden): ").strip()
    if not token:
        raise ValueError("A Hugging Face token is required.")

    rows = []
    for document in documents:
        request = Request(
            args.endpoint,
            data=json.dumps({
                "inputs": document["payload"]["text"]
            }).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
        except HTTPError as error:
            raise RuntimeError(
                f"Embedding request failed: HTTP {error.code}."
            ) from None
        except URLError:
            raise RuntimeError(
                "Cannot reach the embedding endpoint."
            ) from None

        if not isinstance(result, list) or len(result) != 1:
            raise ValueError("Expected exactly one embedding.")

        vector = result[0]
        if not isinstance(vector, list) or len(vector) != args.dimension:
            raise ValueError(
                f"Expected a vector with {args.dimension} dimensions."
            )

        if not all(
            type(value) in (int, float) and math.isfinite(value)
            for value in vector
        ):
            raise ValueError("Embedding contains invalid numeric values.")

        if not any(value != 0 for value in vector):
            raise ValueError("Embedding is a zero vector.")

        rows.append({
            "id": document["id"],
            "vector": vector,
            "payload": document["payload"],
        })
        print(f"PASS: Document {document['id']} embedded.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"PASS: Saved {len(rows)} documents to {output.resolve()}")


if __name__ == "__main__":
    main()
