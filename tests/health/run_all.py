"""
run_all.py — ChaBo Health Check Suite

Runs connectivity checks for each service in order, then executes the
retriever unit test and full RAG pipeline test.

Usage (from repo root):
    source chabo_env/bin/activate
    python tests/health/run_all.py
"""

import asyncio
import logging
import datetime
import os
import sys

# Add src/ to path so imports match how main.py resolves them
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from test_components import (
    check_qdrant, check_hybrid_collection, check_embedding, check_reranker,
    test_retriever_unit, run_full_pipeline_test, test_metadata_filters
)


def setup_logging():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"health_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )


logger = logging.getLogger("HealthCheck")

SAMPLE_QUERY = "What is the purpose of this system?"


async def main():
    setup_logging()
    logger.info("=" * 50)
    logger.info("ChaBo Health Check Suite")
    logger.info("=" * 50)

    from components.retriever.retriever_orchestrator import create_retriever_from_config
    from components.generator.generator_orchestrator import Generator

    try:
        retriever = create_retriever_from_config("params.cfg")
    except Exception as e:
        logger.error(f"❌ Failed to load retriever config: {e}")
        return False

    results = {}

    # --- Connectivity checks (retriever instance carries all config/clients) ---
    logger.info("\n--- Step 1: Connectivity Checks ---")
    results["Qdrant"] = await check_qdrant(retriever)
    # Returns True (skipped) when hybrid retrieval is off.
    results["Hybrid Collection"] = await check_hybrid_collection(retriever)
    results["Embedding Endpoint"] = await check_embedding(retriever)
    results["Reranker Endpoint"] = await check_reranker(retriever)

    if not all(results.values()):
        logger.error("❌ One or more connectivity checks failed. Fix the above errors before proceeding.")
    else:
        # --- Component checks (only if connectivity passed) ---
        logger.info("\n--- Step 2: Component Checks ---")
        generator = Generator()

        retriever_result = await test_retriever_unit(SAMPLE_QUERY, retriever)
        results["Retriever Unit"] = retriever_result["success"]

        pipeline_result = await run_full_pipeline_test(SAMPLE_QUERY, retriever, generator)
        results["RAG Pipeline"] = pipeline_result["success"]

        filter_result = await test_metadata_filters(retriever)
        results["Metadata Filters"] = filter_result["success"]

    # --- Summary ---
    logger.info("\n--- HEALTH CHECK SUMMARY ---")
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {check}")

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n✅ All checks passed. ChaBo is ready.")
    else:
        logger.error("\n❌ Some checks failed. Review logs above.")

    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
