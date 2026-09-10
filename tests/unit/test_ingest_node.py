"""
Unit tests for ingest_node (graph-node level).

Regression guard for a LangGraph gotcha: a node function parameter literally
named `config` is reserved — LangGraph auto-injects its own RunnableConfig
(a plain dict) there when the node is invoked inside a compiled graph, silently
overriding whatever functools.partial bound the name to. ingest_node used to
accept the app's ConfigParser as a parameter named `config`, which meant the
real config was replaced by LangGraph's dict at graph-run time and
`config.getint(...)` in clean_and_chunk_text() blew up with
`'dict' object has no attribute 'getint'`. The fix renames the parameter to
`app_config`. This test asserts that invariant holds and that the node still
works end-to-end with a real ConfigParser.

Run from repo root:
    python tests/unit/test_ingest_node.py
"""
import sys, os, asyncio, inspect, configparser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from components.orchestration.nodes import ingest_node

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"[PASS] {name}")
    else:
        _failed += 1
        print(f"[FAIL] {name} {detail}")


def _make_config():
    config = configparser.ConfigParser()
    config["ingestor"] = {
        "chunk_size": "700",
        "chunk_overlap": "50",
        "max_chunks": "20",
        "separators": r"\n\n,\n,. ,! ,? , ,",
    }
    return config


def test_param_not_named_config():
    # The regression guard: `config` is a reserved node-parameter name in LangGraph.
    params = list(inspect.signature(ingest_node).parameters)
    check("ingest_node's config-like param is not literally named 'config'",
          "config" not in params, f"(params={params})")


def test_ingest_node_processes_file_with_real_config():
    state = {
        "file_content": b"Project code LUMEN-PDF-862. 57 lamps delivered. 8 pending.",
        "filename": "test.txt",
        "metadata": {},
    }
    out = asyncio.run(ingest_node(state, app_config=_make_config()))
    check("ingest_node returns ingestor_context", "ingestor_context" in out, f"(keys={list(out.keys())})")
    check("ingestor_context contains extracted text",
          "LUMEN-PDF-862" in out.get("ingestor_context", ""))
    check("metadata reports ingest_success", out.get("metadata", {}).get("ingest_success") is True)


def test_ingest_node_skips_without_file():
    out = asyncio.run(ingest_node({"metadata": {}}, app_config=_make_config()))
    check("no file → empty dict (no-op)", out == {})


if __name__ == "__main__":
    for fn in [test_param_not_named_config, test_ingest_node_processes_file_with_real_config,
               test_ingest_node_skips_without_file]:
        fn()
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
