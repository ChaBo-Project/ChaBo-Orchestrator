# Deployed Hugging Face regression tests

These tests check a deployed ChaBo-Orchestrator and its synthetic Qdrant
corpus. They use Hugging Face inference and have no Azure dependency.

This is the first part of the regression suite. It does not yet implement
the required check for pull requests targeting main.

## Current coverage

Eleven unittest tests cover:

- A basic question returning a non-empty streamed answer.
- SSE content type, JSON string chunks, complete frames and one final end event.
- Single-number citation markers matching source-list numbers and an expected filename.
- A follow-up request containing previous user and assistant messages.
- Rejection of a malformed message with HTTP 422 and validation details.
- Private Orchestrator Space access with valid, missing and invalid credentials.
- Retrieval of each synthetic document using its own stored embedding vector,
  checking returned document metadata and text.
- OpenAI-compatible model listing.
- Non-streaming chat completion structure and source references.
- OpenAI streaming frames, consistent response identifiers and completion markers.
- Rejection of empty messages and conversations without a user turn.
- Conversation follow-ups through the OpenAI-compatible API.

The streaming and conversation tests call the Orchestrator's LangServe
and OpenAI-compatible endpoints directly. They do not automate the
ChatUI browser interface.

The retrieval test calls the Qdrant Space's Gradio API. It checks indexed
document retrieval, rather than the Orchestrator's query embedding pipeline.

Tests check response structure and source identity instead of requiring
identical AI-generated wording.

## Test environment

The current environment uses:

- ChaBo-Deploy: `hf-spaces-v0.2.0`.
- ChaBo-ChatUI: `0.9.3`.
- A candidate Orchestrator identified by its full Git commit SHA.
- Qdrant collection: `chabo-regression-v1`.
- Embedding dimension: 1024.
- Dataset: configured through `REGRESSION_DATASET_ID`.
- Dataset revision: a full commit SHA supplied through
  `REGRESSION_DATASET_REVISION`.

The fixtures contain three fictional documents about the Orion and Vega
programmes. They contain no production document content.

The retrieval test requires an immutable dataset commit SHA through
REGRESSION_DATASET_REVISION.
Deploying Qdrant must use the matching corpus.

## Run locally

Run commands from the Orchestrator repository root.

Install the dependency:

```bash
python -m pip install "huggingface_hub==2.1.1"
```

Authenticate using a saved Hugging Face login:

```bash
hf auth login
```

Alternatively, supply an authorized token through `HF_TOKEN`.
Never commit tokens to the repository.

The token needs read access to the private Orchestrator Space, Qdrant Space
and synthetic dataset.

Set the target deployment:

```bash
export REGRESSION_ORCHESTRATOR_URL="https://YOUR-ORCHESTRATOR-SPACE.hf.space"
export REGRESSION_QDRANT_URL="https://YOUR-QDRANT-SPACE.hf.space"
export REGRESSION_COLLECTION="YOUR-REGRESSION-COLLECTION"
export REGRESSION_DATASET_ID="YOUR-NAMESPACE/YOUR-REGRESSION-DATASET"
export REGRESSION_DATASET_REVISION="YOUR-FULL-40-CHARACTER-DATASET-COMMIT-SHA"
```

Run all deployed tests:

```bash
python -m unittest discover \
  -s tests/regression \
  -p 'test_deployed_*.py' \
  -v
```

These tests contact live services and can incur inference charges.
They are separate from the normal local CI test suite.

## Build the synthetic dataset

This is a corpus preparation step, not part of each test run.

```bash
python tests/regression/build_dataset.py \
  --endpoint "https://YOUR-EMBEDDING-ENDPOINT" \
  --dimension 1024 \
  --output "../chabo-regression-data/train.json"
```

The script prompts for a token, embeds each fixture document and validates
the vector size and numeric values.

Upload the generated file to the dedicated dataset and record its commit
SHA. When changing the corpus, update the retrieval test's dataset revision
and the deployed Qdrant corpus together. Keep generated embeddings outside
this source repository.

## Private deployment workflow

A separate private instance repository hosts the manually triggered
regression workflow.

The workflow:

1. Checks out the selected full Orchestrator commit SHA.
2. Builds its Docker image and packages that source revision.
3. Deploys the packaged source with the instance configuration to the
   dedicated Orchestrator Space.
4. Waits for the uploaded Space commit to be running and healthy.
5. Runs the regression tests from the fixed test revision.
6. Checks that the Space commit remained unchanged during testing.

The private workflow pins its test checkout to a reviewed full commit SHA.
Update that pin when adopting a new version of the regression tests.

Deployment uses the private repository's `HF_TOKEN`. The test job uses
`HF_REGRESSION_READ_TOKEN`, exposed to the tests as `HF_TOKEN`.
No Hugging Face secrets are required in the public Orchestrator repository.

The private deployment and regression workflows share the concurrency group
`hf-regression` to avoid overlapping deployments within that repository.

This workflow currently reports results in the private repository. It does
not publish a required check on public Orchestrator pull requests.

## Known failure

Issue #63 tracks a source-numbering defect:
https://github.com/ChaBo-Project/ChaBo-Orchestrator/issues/63

An answer citing documents [1] and [3] can receive a source list numbered
1 and 2. The citation assertion correctly rejects that response.

The defect was reproduced with fixed input without an AI call.
Generated answers expose it intermittently. A passing run does not prove
the defect is fixed. Keep the citation assertion unchanged.

## Remaining coverage and integration

- A required regression status for pull requests targeting main.
- Automated end-to-end conversation through the supported ChatUI.
- Existing configuration and selected optional-feature settings.
- File uploads and additional API validation cases.
- Grouped citations on the LangServe route and exact citation-to-document mapping.
- Verification of immutable companion deployment revisions.

Add relevant regression coverage when introducing features or fixing bugs.