"""
Manual test for extract_filters_node — single-pass extraction covering:
  - Current query field extraction (all fields independently)
  - Dual-field extraction from a single query
  - History carry-forward when current query has no filter
  - No filter in either source

Run from repo root with venv active:
    python tests/unit/test_extract_filters.py
"""
import sys, os, asyncio, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

from components.generator.generator_orchestrator import Generator
from components.orchestration.nodes import extract_filters_node


# ── Config ────────────────────────────────────────────────────────────────────
# Local fixture, not components.retriever.filters.FILTER_VALUES — that now resolves
# to {} unless INSTANCE_CONFIG_DIR is set (real values are instance content, no longer
# committed to source). This test needs valid values for its own cases regardless of
# what INSTANCE_CONFIG_DIR happens to be set to when it's run. Tests aren't packaged
# into the published image, so this can be — and is — a real sample drawn from an
# actual instance's document collection (agricultural data), not synthetic content.

FILTERABLE_FIELDS = {
    "crop_type": "list",
    "title": "str",
}

FILTER_VALUES = {
    "crop_type": ["wheat", "maize"],
    "title": [
        "Cultivation and producing Maize",
        "Maize cultivation in the old and new lands",
    ],
}

# ── Test cases ────────────────────────────────────────────────────────────────

ALL_CASES = [
    # --- Current query extraction ---
    {
        "name": "crop_type from current query",
        "query": "What are the irrigation requirements for wheat?",
        "user_messages_history": None,
        "expect_filters": True,
        "expect_keys": ["crop_type"],
    },
    {
        "name": "dual-field extraction from same query",
        "query": "Tell me about growing maize in new reclaimed lands",
        "user_messages_history": None,
        "expect_filters": True,
        "expect_keys": ["crop_type", "title"],  # both must be present
    },
    {
        "name": "no filter in query",
        "query": "What is the best time to irrigate?",
        "user_messages_history": None,
        "expect_filters": False,
        "expect_keys": [],
    },
    # --- History carry-forward ---
    {
        "name": "crop_type carried forward from history",
        "query": "What is the recommended harvest time?",  # no crop mentioned
        "user_messages_history": (
            "User: Show me documents about wheat cultivation.\n"
            "User: What are the planting dates for wheat?"
        ),
        "expect_filters": True,
        "expect_keys": ["crop_type"],
    },
    {
        "name": "no filter in history or query",
        "query": "What is the recommended harvest time?",
        "user_messages_history": "User: Tell me about farming in general.",
        "expect_filters": False,
        "expect_keys": [],
    },
]


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_case(generator, case):
    state = {
        "query": case["query"],
        "user_messages_history": case.get("user_messages_history"),
    }
    result = await extract_filters_node(
        state=state,
        llm_client=generator.llm_client,
        filterable_fields=FILTERABLE_FIELDS,
        filter_values=FILTER_VALUES,
    )
    filters = result.get("metadata_filters")

    presence_ok = bool(filters) == case["expect_filters"]
    keys_ok = all(k in filters for k in case["expect_keys"]) if case["expect_keys"] and filters else True
    status = "PASS" if (presence_ok and keys_ok) else "FAIL"

    print(f"\n[{status}] {case['name']}")
    print(f"  query   : {case['query']}")
    if case.get("user_messages_history"):
        print(f"  history : {case['user_messages_history'][:80]}...")
    print(f"  filters : {filters}")
    if case["expect_keys"]:
        print(f"  expect  : keys={case['expect_keys']} present={'yes' if keys_ok else 'NO'}")
    return status == "PASS"


async def main():
    generator = Generator()

    print("\n" + "=" * 60)
    print("Single-pass filter extraction tests")
    print("=" * 60)

    results = [await run_case(generator, c) for c in ALL_CASES]

    total = len(results)
    passed = sum(results)
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
