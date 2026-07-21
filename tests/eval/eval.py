import asyncio
import configparser
import csv
import re
import sys
import os
import json
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
from ragas.run_config import RunConfig

# Add src/ to path so imports match how main.py resolves them
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from components.retriever.retriever_orchestrator import create_retriever_from_config
from components.generator.generator_orchestrator import Generator
from components.orchestration.nodes import extract_filters_node
from components.utils import _acall_hf_endpoint, getconfig
from components.retriever.filters import FILTER_VALUES

from test_questions import standalone_questions, history_blocks, safeguard_questions

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Parse config (same logic as main.py)
_config = getconfig("params.cfg")
MAX_TURNS = _config.getint("conversation_history", "MAX_TURNS", fallback=3)
_filterable_fields_raw = _config.get("metadata_filters", "filterable_fields", fallback="")
FILTERABLE_FIELDS: Dict[str, str] = {}
for _item in _filterable_fields_raw.split(","):
    _item = _item.strip()
    if ":" in _item:
        _name, _ftype = _item.split(":", 1)
        FILTERABLE_FIELDS[_name.strip()] = _ftype.strip()
    elif _item:
        FILTERABLE_FIELDS[_item] = "str"


@dataclass
class EvalCase:
    question: str
    subset: str                              # "standalone" | "history" | "safeguard"
    user_messages_history: Optional[str]     # pre-built history string, None if not applicable
    expected_filters: Optional[Dict]         # ground truth filters, None if no filter expected
    expected_answer: Optional[str] = None    # rough ground truth answer for RAGAS scoring
    expected_sources: Optional[List[Dict]] = field(default=None)  # expected {filename, page} dicts for RAGAS


def build_eval_suite() -> List[EvalCase]:
    """Flatten the three test_questions subsets into a unified list of EvalCase objects.

    History blocks: only the last turn is evaluated; prior turns (sliced to MAX_TURNS)
    are passed as user_messages_history — mirroring what ui_adapters does in production.
    """
    cases = []

    for item in standalone_questions:
        cases.append(EvalCase(
            question=item["question"],
            subset="standalone",
            user_messages_history=None,
            expected_filters=item.get("expected_filters"),
            expected_answer=item.get("expected_answer"),
            expected_sources=item.get("expected_sources"),
        ))

    for block in history_blocks:
        turns = block["turns"]
        if len(turns) < 2:
            continue
        current_query = turns[-1]
        prior_turns = turns[:-1][-MAX_TURNS:]
        history_str = "\n".join(f"USER: {t}" for t in prior_turns)
        cases.append(EvalCase(
            question=current_query,
            subset="history",
            user_messages_history=history_str,
            expected_filters=block.get("expected_filters"),
            expected_answer=block.get("expected_answer"),
            expected_sources=block.get("expected_sources"),
        ))

    for item in safeguard_questions:
        cases.append(EvalCase(
            question=item["question"],
            subset="safeguard",
            user_messages_history=None,
            expected_filters=item.get("expected_filters"),
            expected_answer=item.get("expected_answer"),
            expected_sources=item.get("expected_sources"),
        ))

    return cases


def _check_filters(expected: Optional[Dict], extracted: Optional[Dict]) -> str:
    """Compare extracted filters against ground truth and return a result category."""
    if expected is None and extracted is None:
        return "correct"
    if expected is None and extracted is not None:
        return "spurious_filter"
    if expected is not None and extracted is None:
        return "no_filter"
    if expected == extracted:
        return "correct"
    matching_fields = [k for k in expected if extracted.get(k) == expected[k]]
    if matching_fields:
        return "partial_match"
    return "mismatch"


def _result_path(base: str, filtered: bool) -> str:
    suffix = "_filtered" if filtered else ""
    return os.path.join(RESULTS_DIR, f"{base}{suffix}.json")


async def evaluate_questions(
    cases: List[EvalCase],
    retriever,
    generator=None,
    filterable_fields: Dict[str, str] = None,
    filter_values: Dict[str, list] = None,
) -> List[Dict]:
    """Run retrieval for each case. If generator is provided, extract filters first."""
    eval_data = []

    for case in cases:
        print(f"🧐 [{case.subset}] Processing: {case.question[:50]}...")

        # --- Filter extraction (filters mode only) ---
        filters = None
        if generator is not None:
            state = {"query": case.question, "user_messages_history": case.user_messages_history}
            result_state = await extract_filters_node(
                state,
                generator=generator,
                filterable_fields=filterable_fields,
                filter_values=filter_values,
            )
            filters = result_state.get("metadata_filters")
            filter_check = _check_filters(case.expected_filters, filters)
            print(f"   Filters extracted: {filters} | check: {filter_check}")

        # --- Embed ---
        embed_res = await _acall_hf_endpoint(
            retriever.embedding_endpoint_url,
            retriever.hf_token,
            {"inputs": case.question},
        )
        query_vector = embed_res[0]

        # --- Stage 1: vector search ---
        raw_candidates = await retriever._asearch_qdrant(query_vector, filters=filters)

        # --- Stage 2: full pipeline (rerank) ---
        retriever_kwargs = {"filters": filters} if filters else {}
        final_docs = await retriever.ainvoke(case.question, **retriever_kwargs)

        entry = {
            "question": case.question,
            "subset": case.subset,
            "stage1_raw_vector_results": [
                {"content": c.get("answer"), "score": c.get("score"), "metadata": c.get("answer_metadata")}
                for c in raw_candidates
            ],
            "stage2_reranked_results": [
                {"content": d.page_content, "rerank_score": d.metadata.get("rerank_score"), "metadata": d.metadata}
                for d in final_docs
            ],
        }
        if generator is not None:
            entry["filters_applied"] = filters
            entry["filter_check"] = {
                "expected": case.expected_filters,
                "extracted": filters,
                "result": filter_check,
            }

        eval_data.append(entry)

    return eval_data


def _subset_score(subset_details: List[Dict]) -> Dict:
    """Compute counts and score for a list of detail entries."""
    counts: Dict[str, int] = {}
    for d in subset_details:
        counts[d["result"]] = counts.get(d["result"], 0) + 1
    total = len(subset_details)
    correct = counts.get("correct", 0)
    partial = counts.get("partial_match", 0)
    return {
        "total": total,
        "score": round((correct + 0.5 * partial) / total, 3) if total > 0 else 0.0,
        **counts,
    }


def _export_filter_report(results: List[Dict], filters_enabled: bool) -> None:
    """Write a standalone filter check report and print a console summary."""
    details = []
    counts: Dict[str, int] = {}

    for entry in results:
        fc = entry.get("filter_check", {})
        result = fc.get("result", "not_annotated")
        counts[result] = counts.get(result, 0) + 1
        details.append({
            "question": entry["question"],
            "subset": entry["subset"],
            "expected": fc.get("expected"),
            "extracted": fc.get("extracted"),
            "result": result,
        })

    total = len(details)
    correct = counts.get("correct", 0)
    partial = counts.get("partial_match", 0)
    score = round((correct + 0.5 * partial) / total, 3) if total > 0 else 0.0

    subsets = ["standalone", "history", "safeguard"]
    by_subset = {
        s: _subset_score([d for d in details if d["subset"] == s])
        for s in subsets
        if any(d["subset"] == s for d in details)
    }

    report = {
        "summary": {
            "total": total,
            "score": score,
            "score_note": "correct=1pt, partial_match=0.5pt, all others=0pt",
            **counts,
        },
        "by_subset": by_subset,
        "details": details,
    }

    report_path = _result_path("filter_check_report", filters_enabled)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    print("\n📊 Filter extraction summary:")
    for outcome, count in sorted(counts.items()):
        print(f"   {outcome}: {count}")
    print(f"   score: {score} ({correct} correct + {partial} partial out of {total})")
    print(f"💾 Filter report saved to {report_path}")


async def run_retrieval_only(filters_enabled: bool):
    all_questions = (
        [item["question"] for item in standalone_questions]
        + [block["turns"][-1] for block in history_blocks if len(block["turns"]) >= 2]
        + [item["question"] for item in safeguard_questions]
    )
    if not all_questions or all(len(q.split()) <= 1 for q in all_questions):
        print("💥 test_questions.py still has placeholder content.")
        print("   Edit tests/eval/test_questions.py and add real questions before running eval.")
        sys.exit(1)

    print("🚀 Initializing Retriever...")
    try:
        retriever = create_retriever_from_config("params.cfg")
    except Exception as e:
        print(f"💥 Failed to load config/retriever: {e}")
        sys.exit(1)

    generator = None
    if filters_enabled:
        if not FILTERABLE_FIELDS:
            print("💥 --filters passed but filterable_fields is empty in params.cfg. Aborting.")
            sys.exit(1)
        print("🔍 Filter extraction enabled. Initializing Generator for filter extraction...")
        generator = Generator()

    cases = build_eval_suite()
    results = await evaluate_questions(
        cases,
        retriever,
        generator=generator,
        filterable_fields=FILTERABLE_FIELDS if filters_enabled else None,
        filter_values=FILTER_VALUES if filters_enabled else None,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = _result_path("retrieval_eval_results", filters_enabled)

    print("💾 Exporting results...")
    df = pd.DataFrame(results)
    df.to_json(output_path, orient="records", indent=4, force_ascii=False)

    if filters_enabled:
        _export_filter_report(results, filters_enabled)

    print(f"\n✅ Retrieval complete! Results saved to {output_path}")
    sys.exit(0)


async def get_judge_verdict(generator: Generator, query: str, content: str, metadata: dict):
    from langchain_core.documents import Document
    context_as_docs = [Document(page_content=content, metadata=metadata)]

    judge_query = (
        f"[SYSTEM EVALUATION]\n"
        f"Task: Judge if the provided Context & Source answer the Question.\n"
        f"Question: \"{query}\"\n\n"
        f"Response Format:\nREASON: [Short reasoning in English]\nVERDICT: YES or VERDICT: NO (exactly as written, no other characters)"
    )

    response = await generator.generate(
        query=judge_query,
        context=context_as_docs,
        chatui_format=False
    )
    return response


def _verdict_is_yes(judge_output: str) -> bool:
    """Extract YES/NO verdict robustly, stripping markdown formatting and collapsing whitespace."""
    clean = re.sub(r"[*_`#]", "", judge_output.upper())
    clean = re.sub(r"\s+", " ", clean)
    match = re.search(r"VERDICT\s*:\s*(YES|NO)", clean)
    return bool(match and match.group(1) == "YES")


async def run_evaluation_batch(filters_enabled: bool, input_file=None):
    if input_file is None:
        input_file = _result_path("retrieval_eval_results", filters_enabled)

    generator_instance = Generator()
    output_filename = _result_path("judged_eval_report", filters_enabled)

    with open(input_file, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    final_report = []
    if os.path.exists(output_filename):
        with open(output_filename, "r", encoding="utf-8") as f:
            final_report = json.load(f)
        print(f"🔄 Resuming from checkpoint. {len(final_report)} questions already judged.")

    processed_questions = {entry['question'] for entry in final_report}

    for entry in source_data:
        user_q = entry['question']
        if user_q in processed_questions:
            continue

        print(f"⚖️ Judging results for: {user_q[:40]}...")

        for doc in entry['stage2_reranked_results']:
            judge_output = await get_judge_verdict(
                generator_instance,
                user_q,
                doc['content'],
                doc['metadata']
            )
            doc['judge_raw_output'] = judge_output
            doc['is_relevant'] = _verdict_is_yes(judge_output)

        final_report.append(entry)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=4)

        print(f"💾 Checkpoint saved. Progress: {len(final_report)}/{len(source_data)}")

    print(f"✅ Success! Full report finalized at {output_filename}")
    sys.exit(0)


async def run_sample_eval(filters_enabled: bool, input_file=None):
    if input_file is None:
        input_file = _result_path("retrieval_eval_results", filters_enabled)

    generator_instance = Generator()

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample_data = data[:2]
    final_report = []

    print(f"🚀 Starting Sample Run: Evaluating {len(sample_data)} queries...")

    for entry in sample_data:
        user_q = entry['question']
        print(f"\n🔍 Query: {user_q}")

        for i, doc in enumerate(entry['stage2_reranked_results']):
            judge_output = await get_judge_verdict(
                generator_instance,
                user_q,
                doc['content'],
                doc['metadata']
            )

            is_relevant = _verdict_is_yes(judge_output)
            doc['judge_raw_output'] = judge_output
            doc['is_relevant'] = is_relevant

            status = "✅ RELEVANT" if is_relevant else "❌ IRRELEVANT"
            print(f"  - Doc {i+1}: {status}")
            print(f"    Reasoning: {judge_output.split('REASON:')[1].split('VERDICT:')[0].strip() if 'REASON:' in judge_output else 'N/A'}")

        final_report.append(entry)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_filename = _result_path("sample_judged_report", filters_enabled)
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4)

    print(f"\n✨ Sample complete. Check '{output_filename}' for details.")
    sys.exit(0)


_RAGAS_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ragas_config.cfg")


def _load_ragas_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser(inline_comment_prefixes=("#",))
    if not config.read(_RAGAS_CONFIG_PATH):
        print(f"💥 ragas_config.cfg not found at {_RAGAS_CONFIG_PATH}")
        sys.exit(1)
    return config


def _build_ragas_llm(config: configparser.ConfigParser):
    from ragas.llms import LangchainLLMWrapper

    provider = config.get("ragas", "JUDGE_PROVIDER").strip().lower()
    model = config.get("ragas", "JUDGE_MODEL").strip()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model)
    elif provider == "cohere":
        from langchain_cohere import ChatCohere
        llm = ChatCohere(model=model)
    elif provider == "azure":                                                   
        from langchain_openai import ChatOpenAI     
        llm = ChatOpenAI(
            model=model,
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=config.get("ragas", "AZURE_ENDPOINT").strip(),)
    elif provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        endpoint = HuggingFaceEndpoint(
            endpoint_url=model,
            huggingfacehub_api_token=os.environ.get("HF_TOKEN"),
        )
        llm = ChatHuggingFace(llm=endpoint)
    else:
        print(f"💥 Unknown JUDGE_PROVIDER '{provider}'. Supported: openai, anthropic, cohere, azure, huggingface")
        sys.exit(1)

    return LangchainLLMWrapper(llm)

def _build_ragas_embeddings(retriever):
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    return HuggingFaceEndpointEmbeddings(
        model=retriever.embedding_endpoint_url,
        huggingfacehub_api_token=retriever.hf_token,
    )

def _build_ragas_metrics(config: configparser.ConfigParser, ragas_llm):
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision

    METRIC_MAP = {
        "faithfulness": Faithfulness,
        "answer_relevancy": AnswerRelevancy,
        "context_recall": ContextRecall,
        "context_precision": ContextPrecision,
    }

    metrics_raw = config.get("ragas", "METRICS", fallback="faithfulness,answer_relevancy,context_recall,context_precision")
    metric_names = [m.strip() for m in metrics_raw.split(",") if m.strip()]

    metrics = []
    for name in metric_names:
        if name not in METRIC_MAP:
            print(f"⚠️  Unknown metric '{name}' in ragas_config.cfg, skipping.")
            continue
        metrics.append(METRIC_MAP[name](llm=ragas_llm))

    if not metrics:
        print("💥 No valid metrics configured in ragas_config.cfg.")
        sys.exit(1)

    return metrics


async def run_ragas_eval(filters_enabled: bool):
    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

    ragas_config = _load_ragas_config()
    ragas_llm = _build_ragas_llm(ragas_config)
    metrics = _build_ragas_metrics(ragas_config, ragas_llm)

    print("🚀 Initializing Retriever and Generator...")
    try:
        retriever = create_retriever_from_config("params.cfg")
    except Exception as e:
        print(f"💥 Failed to load retriever: {e}")
        sys.exit(1)

    ragas_embeddings = _build_ragas_embeddings(retriever)
    generator = Generator()

    filter_generator = None
    if filters_enabled:
        if not FILTERABLE_FIELDS:
            print("💥 --filters passed but filterable_fields is empty in params.cfg. Aborting.")
            sys.exit(1)
        filter_generator = Generator()

    cases = build_eval_suite()
    ragas_cases = [
        c for c in cases
        if c.expected_answer and not c.expected_answer.startswith("TODO")
    ]
    skipped = len(cases) - len(ragas_cases)
    if not ragas_cases:
        print("💥 No test cases with expected_answer found. Fill in expected_answer fields in test_questions.py first.")
        sys.exit(1)
    if skipped:
        print(f"⚠️  Skipping {skipped} case(s) with TODO placeholders in expected_answer.")

    print(f"📋 Running RAGAS eval on {len(ragas_cases)} case(s)...")

    samples = []
    for case in ragas_cases:
        print(f"🧐 [{case.subset}] Processing: {case.question[:50]}...")

        filters = None
        if filter_generator is not None:
            state = {"query": case.question, "user_messages_history": case.user_messages_history}
            result_state = await extract_filters_node(
                state,
                generator=filter_generator,
                filterable_fields=FILTERABLE_FIELDS,
                filter_values=FILTER_VALUES,
            )
            filters = result_state.get("metadata_filters")

        retriever_kwargs = {"filters": filters} if filters else {}
        docs = await retriever.ainvoke(case.question, **retriever_kwargs)
        contexts = [doc.page_content for doc in docs]

        answer = await generator.generate(
            query=case.question,
            context=docs,
            chatui_format=False,
        )

        samples.append(SingleTurnSample(
            user_input=case.question,
            response=answer,
            retrieved_contexts=contexts,
            reference=case.expected_answer,
        ))

    print(f"\n⚖️  Scoring {len(samples)} sample(s) with RAGAS...")
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(dataset=dataset, 
                      metrics=metrics, 
                      llm=ragas_llm,
                      embeddings=ragas_embeddings,
                      run_config=RunConfig(max_workers=2, timeout=900),)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_filtered" if filters_enabled else ""
    output_path = os.path.join(RESULTS_DIR, f"ragas_report_{timestamp}{suffix}.json")

    scores_df = result.to_pandas()
    scores_df.to_json(output_path, orient="records", indent=4, force_ascii=False)

    score_cols = [c for c in scores_df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
    avg_scores = {col: round(float(scores_df[col].mean()), 4) for col in score_cols}

    print("\n📊 RAGAS Summary (averages):")
    for col, val in avg_scores.items():
        print(f"   {col}: {val:.3f}")

    history_path = os.path.join(RESULTS_DIR, "ragas_history.csv")
    history_exists = os.path.exists(history_path)
    history_row = {
        "timestamp": timestamp,
        "filters_enabled": filters_enabled,
        "judge_provider": ragas_config.get("ragas", "JUDGE_PROVIDER").strip(),
        "judge_model": ragas_config.get("ragas", "JUDGE_MODEL").strip(),
        "num_cases": len(samples),
        **avg_scores,
    }
    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history_row.keys())
        if not history_exists:
            writer.writeheader()
        writer.writerow(history_row)

    print(f"\n✅ RAGAS report saved to {output_path}")
    print(f"📈 History updated at {history_path}")
    sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChaBo RAG Evaluation")
    parser.add_argument(
        "--mode",
        choices=["retrieval", "batch", "sample", "ragas"],
        default="retrieval",
        help="retrieval: run Stage 1 and save results | batch: judge with LLM (resumes from checkpoint) | sample: judge first 2 questions only | ragas: full pipeline eval with RAGAS metrics"
    )
    parser.add_argument(
        "--filters",
        action="store_true",
        default=False,
        help="Enable metadata filter extraction before retrieval. Results saved with '_filtered' suffix for comparison."
    )
    args = parser.parse_args()

    modes = {
        "retrieval": lambda: run_retrieval_only(args.filters),
        "batch": lambda: run_evaluation_batch(args.filters),
        "sample": lambda: run_sample_eval(args.filters),
        "ragas": lambda: run_ragas_eval(args.filters),
    }
    asyncio.run(modes[args.mode]())
