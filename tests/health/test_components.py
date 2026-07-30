import logging
import asyncio
import os

logger = logging.getLogger("RAG_Test_Suite")


class _LogCapture(logging.Handler):
    """Minimal log handler to capture messages during a test."""
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    def contains(self, substr):
        return any(substr in m for m in self.messages)


async def check_embedding(retriever):
    """Verify the embedding endpoint returns a valid vector using the retriever's own config."""
    try:
        from components.utils import _acall_hf_endpoint
        payload = {"inputs": "health check"}
        result = await _acall_hf_endpoint(retriever.embedding_endpoint_url, retriever.hf_token, payload)
        if not isinstance(result, list) or not result:
            raise ValueError("Unexpected response shape from embedding endpoint")
        logger.info(f"✅ Embedding endpoint: reachable, vector dim={len(result[0])}.")
        return True
    except Exception as e:
        logger.error(f"❌ Embedding endpoint check failed: {str(e)}", exc_info=True)
        return False


async def check_reranker(retriever):
    """Verify the reranker endpoint returns scores using the retriever's own config."""
    try:
        from components.utils import _acall_hf_endpoint
        payload = {"query": "health check", "texts": ["sample document"]}
        result = await _acall_hf_endpoint(retriever.reranker_endpoint_url, retriever.hf_token, payload)
        if not isinstance(result, list) or 'score' not in result[0]:
            raise ValueError("Unexpected response shape from reranker endpoint")
        logger.info(f"✅ Reranker endpoint: reachable, returned {len(result)} score(s).")
        return True
    except Exception as e:
        logger.error(f"❌ Reranker endpoint check failed: {str(e)}", exc_info=True)
        return False


async def check_qdrant(retriever):
    """Verify Qdrant is reachable and the configured collection exists using the retriever's own client."""
    try:
        if retriever.qdrant_mode.lower() == 'native':
            client = await retriever._aget_qdrant_client()
            collections = [c.name for c in (await client.get_collections()).collections]
            if retriever.qdrant_collection not in collections:
                logger.error(f"❌ Qdrant: collection '{retriever.qdrant_collection}' not found. Available: {collections}")
                return False
            logger.info(f"✅ Qdrant: reachable, collection '{retriever.qdrant_collection}' exists.")
        elif retriever.qdrant_mode.lower() == 'gradio':
            # Triggering _aget_qdrant_client forces GradioClient init — success means space is reachable
            await retriever._aget_qdrant_client()
            logger.info(f"✅ Qdrant (Gradio): space reachable at {retriever.qdrant_url}.")
        return True
    except Exception as e:
        logger.error(f"❌ Qdrant check failed: {str(e)}", exc_info=True)
        return False


async def check_hybrid_collection(retriever):
    """
    Verify hybrid retrieval works. Returns True (skipped) when hybrid disabled.
    """
    if not getattr(retriever, "hybrid_enabled", False):
        logger.info("Hybrid retrieval disabled - skipping hybrid collection check.")
        return True

    try:
        from components.retriever.retriever_orchestrator import SPARSE_VECTOR_NAME

        retriever.validate_hybrid_collection()
        client = await retriever._aget_qdrant_client()

        points, _ = await client.scroll(
            collection_name=retriever.qdrant_collection,
            limit=1, with_payload=True, with_vectors=False,
        )
        if not points:
            logger.error(f"Hybrid: collection '{retriever.qdrant_collection}' is empty - nothing to probe.")
            return False

        payload = points[0].payload or {}
        probe = payload.get("text") or payload.get("page_content") or ""
        sparse_vector = retriever._embed_sparse(probe)
        if sparse_vector is None:
            logger.error("Hybrid: sparse encoder produced no terms from the sampled corpus text.")
            return False

        result = await client.query_points(
            collection_name=retriever.qdrant_collection,
            query=sparse_vector, using=SPARSE_VECTOR_NAME, limit=5,
            with_payload=False, with_vectors=False,
        )
        if not result.points:
            logger.error(
                "Hybrid: sparse branch returned 0 hits for text sampled from the collection itself. "
                "[retrieval] sparse_model / sparse_language almost certainly differ from the "
                "--sparse_model / --sparse_language used at ingestion. Re-ingest, or fix params.cfg."
            )
            return False

        logger.info(
            f"Hybrid: collection '{retriever.qdrant_collection}' is hybrid-shaped "
            f"(dense='{retriever.dense_vector_name or '<unnamed default>'}', "
            f"sparse='{SPARSE_VECTOR_NAME}'); sparse branch returned {len(result.points)} hit(s)."
        )
        return True

    except Exception as e:
        logger.error(f"Hybrid collection check failed: {e}", exc_info=True)
        return False


async def test_retriever_unit(query, retriever_instance):
    """
    Unit test for the Retriever only.
    Verifies connectivity, scoring, and document structure.
    Returns a dict with success, docs, and optional error.
    """
    try:
        logger.info(f"⏳ [Retriever Unit] Calling ainvoke: '{query}'")
        docs = await retriever_instance.ainvoke(query)

        if not docs:
            logger.warning("⚠️ No documents returned. Check endpoint or query relevance.")
            return {"success": False, "docs": [], "error": "No documents returned"}

        logger.info(f"✅ Found {len(docs)} docs.")

        top_score = docs[0].metadata.get('rerank_score')
        if not getattr(retriever_instance, "reranker_enabled", True):
            logger.info("Reranker disabled in config - documents are in retrieval order.")
        elif top_score is not None:
            logger.info(f"🔝 Top Rerank Score: {top_score} (reranker active)")
        else:
            logger.warning("⚠️ No rerank_score in metadata — reranker may not have run")

        return {"success": True, "docs": docs}

    except Exception as e:
        logger.error(f"❌ Retriever Unit Crash: {str(e)}", exc_info=True)
        return {"success": False, "docs": [], "error": str(e)}


async def run_full_pipeline_test(query, retriever_instance, generator_instance):
    """
    Integration test for the full RAG pipeline.
    Steps: Retrieval -> Streaming Generation -> Source Attribution.
    """
    try:
        # 1. Step: Retrieval
        logger.info(f"🚀 [Pipeline Test] Step 1: Retrieving context for '{query}'...")
        docs = await retriever_instance.ainvoke(query)

        if not docs:
            logger.warning("⚠️ No context found. Generator will run in 'no-context' mode.")

        # 2. Step: Manual Streaming Generation
        logger.info("🚀 [Pipeline Test] Step 2: Starting streaming generation...")
        accumulated_response = ""
        captured_sources = []

        async for event in generator_instance.generate_streaming(
            query=query,
            context=docs,
            chatui_format=True
        ):
            event_type = event.get("event")
            data = event.get("data", "")

            if event_type == "data":
                accumulated_response += data
                if len(accumulated_response) % 100 < len(data):
                    logger.info(f"  ...Streaming: {len(accumulated_response)} chars")

            elif event_type == "final_answer":
                captured_sources = data.get("webSources", [])
                logger.info(f"📍 Sources received: {len(captured_sources)} citations.")

        # 3. Validation & Reporting
        logger.info("✅ Full Pipeline Test Complete!")
        logger.info(f"📊 Summary: {len(accumulated_response)} chars generated | {len(captured_sources)} sources.")

        return {
            "answer": accumulated_response,
            "sources": captured_sources,
            "docs": docs,
            "success": True
        }

    except Exception as e:
        logger.error(f"❌ Full Pipeline Test Failed: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


async def test_metadata_filters(retriever):
    """
    Tests metadata filtering against live Qdrant using the existing retriever instance.

    Three sub-tests:
    1. Single-field filter  — crop_type only, AND returns results as expected
    2. Multi-field filter   — valid combination (crop_type + matching title), AND returns results
    3. AND safeguard        — impossible combination (wheat crop_type + maize title),
                              AND returns 0, safeguard retries with priority field (crop_type) only

    Uses retriever.ainvoke(query, filters=...) — the same path as production.
    """
    if retriever.qdrant_mode.lower() not in ('native', 'gradio'):
        logger.warning("⚠️  Metadata filter test skipped — unsupported qdrant_mode.")
        return {"success": True, "skipped": True}

    sub_results = {}

    # ── Sub-test 1: single-field filter ──────────────────────────────────────
    logger.info("⏳ [Filter Test 1/3] Single-field filter: crop_type=['wheat']")
    try:
        docs = await retriever.ainvoke(
            "irrigation and planting requirements for wheat",
            filters={"crop_type": ["wheat"]}
        )
        if not docs:
            logger.error("❌ [Filter Test 1/3] No documents returned with crop_type filter.")
            sub_results["single_field"] = False
        else:
            logger.info(f"✅ [Filter Test 1/3] Returned {len(docs)} docs with crop_type filter.")
            sub_results["single_field"] = True
    except Exception as e:
        logger.error(f"❌ [Filter Test 1/3] Crashed: {e}", exc_info=True)
        sub_results["single_field"] = False

    # ── Sub-test 2: multi-field AND — valid combination ───────────────────────
    logger.info("⏳ [Filter Test 2/3] Multi-field AND: crop_type=['maize'] + matching title")
    try:
        docs = await retriever.ainvoke(
            "cultivation techniques for maize",
            filters={
                "crop_type": ["maize"],
                "title": "Maize cultivation in the old and new lands"
            }
        )
        if not docs:
            logger.warning("⚠️  [Filter Test 2/3] AND returned 0 docs — combination may not exist in collection.")
            sub_results["multi_field_and"] = False
        else:
            logger.info(f"✅ [Filter Test 2/3] AND returned {len(docs)} docs for valid crop+title combo.")
            sub_results["multi_field_and"] = True
    except Exception as e:
        logger.error(f"❌ [Filter Test 2/3] Crashed: {e}", exc_info=True)
        sub_results["multi_field_and"] = False

    # ── Sub-test 3: AND safeguard — impossible combination ────────────────────
    logger.info("⏳ [Filter Test 3/3] AND safeguard: crop_type=['wheat'] + maize title (impossible combo)")
    cap = _LogCapture()
    retriever_logger = logging.getLogger("components.retriever.retriever_orchestrator")
    retriever_logger.addHandler(cap)
    try:
        docs = await retriever.ainvoke(
            "cultivation techniques for wheat",
            filters={
                "crop_type": ["wheat"],
                "title": "Cultivation and producing Maize"   # wheat + maize title = 0 AND results
            }
        )
        safeguard_fired = cap.contains("AND filter returned 0 results")

        if not safeguard_fired:
            logger.warning(
                "⚠️  [Filter Test 3/3] Safeguard did NOT fire — AND may have returned results "
                "for this combo (collection data may differ). Check manually."
            )
        else:
            logger.info("✅ [Filter Test 3/3] Safeguard fired as expected — AND returned 0, retried with priority field.")

        if docs:
            logger.info(f"✅ [Filter Test 3/3] Priority-field retry returned {len(docs)} docs.")
        else:
            logger.warning("⚠️  [Filter Test 3/3] Priority-field retry also returned 0 docs.")

        sub_results["safeguard"] = safeguard_fired and bool(docs)
    except Exception as e:
        logger.error(f"❌ [Filter Test 3/3] Crashed: {e}", exc_info=True)
        sub_results["safeguard"] = False
    finally:
        retriever_logger.removeHandler(cap)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(sub_results.values())
    total = len(sub_results)
    logger.info(f"📊 Metadata Filter Tests: {passed}/{total} sub-tests passed — {sub_results}")
    return {"success": all(sub_results.values()), "sub_results": sub_results}
