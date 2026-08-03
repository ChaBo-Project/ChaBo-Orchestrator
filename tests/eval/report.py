"""
Report generator for ChaBo eval results.

Loads judged eval JSON(s), prints hit-rate summaries for both retriever and reranker
(overall + per subset), and saves per-question CSV reports.

NOTE: Requires top_k == reranker_top_k in params.cfg so that stage2_reranked_results
contains the full retriever candidate set. Retriever hit rates are computed by sorting
stage2 results by retriever_score; reranker hit rates by rerank_score.

Usage:
    # Baseline only
    python tests/eval/report.py --baseline tests/eval/results/judged_eval_report.json

    # Baseline + filtered, with delta comparison
    python tests/eval/report.py \
        --baseline tests/eval/results/judged_eval_report.json \
        --filtered tests/eval/results/judged_eval_report_filtered.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

METADATA_KEYS = ["filename", "page", "retriever_score", "rerank_score", "crop_type", "title"]
DEFAULT_OUT_DIR = Path("tests/eval/results")


def load_report(path: str) -> pd.DataFrame:
    """Load a judged eval report JSON and flatten to one row per result item."""
    with open(path) as f:
        data = json.load(f)

    rows = []
    for entry in data:
        question = entry["question"]
        subset = entry["subset"]
        filters_applied = entry.get("filters_applied")

        for result in entry["stage2_reranked_results"]:
            meta = result.get("metadata", {})
            row = {
                "question": question,
                "subset": subset,
                "filters_applied": filters_applied,
                "is_relevant": bool(result.get("is_relevant", False)),
                "rerank_score": result.get("rerank_score"),
                "judge_raw_output": result.get("judge_raw_output"),
            }
            for k in METADATA_KEYS:
                row[k] = meta.get(k)
            rows.append(row)

    return pd.DataFrame(rows)


def calculate_hit_rates(df: pd.DataFrame, label: str = "", sort_by: str = "rerank_score") -> dict:
    """
    Calculate HR@5, HR@10, HR@20 for a DataFrame slice. Prints and returns results.

    sort_by: 'rerank_score' for reranker hit rates, 'retriever_score' for retriever hit rates.
    Requires top_k == reranker_top_k so that all retriever candidates appear in stage2 results.
    """
    df = df.sort_values(["question", sort_by], ascending=[True, False])
    grouped = df.groupby("question")

    hits = {5: [], 10: [], 20: []}
    for _, group in grouped:
        for n in hits:
            hits[n].append(group.head(n)["is_relevant"].any())

    rates = {f"HR@{n}": (sum(v) / len(v)) * 100 for n, v in hits.items()}

    print(f"--- {label} ---" if label else "--- Hit Rates ---")
    print(f"  Unique queries : {len(grouped)}")
    for k, v in rates.items():
        print(f"  {k}            : {v:.1f}%")
    print()

    return rates


def generate_question_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-question summary table with Relevant@N and Hit@N for both
    retriever (sorted by retriever_score) and reranker (sorted by rerank_score).
    """
    has_filters = df["filters_applied"].notna().any()
    rows = []

    for question, group in df.groupby("question"):
        row = {
            "question": question,
            "subset": group["subset"].iloc[0],
        }
        if has_filters:
            row["filters_applied"] = str(group["filters_applied"].iloc[0])

        for stage, score_col in [("retriever", "retriever_score"), ("reranker", "rerank_score")]:
            sorted_group = group.sort_values(score_col, ascending=False)
            for n in [5, 10, 20]:
                top = sorted_group.head(n)["is_relevant"]
                row[f"{stage}_Relevant_in_Top{n}"] = int(top.sum())
                row[f"{stage}_Hit@{n}"] = 1 if top.any() else 0
            row[f"{stage}_Total_Relevant"] = int(group["is_relevant"].sum())

        rows.append(row)

    return pd.DataFrame(rows)


def print_hit_rates_both_stages(df: pd.DataFrame, label: str = ""):
    """Print retriever and reranker hit rates for a DataFrame slice."""
    calculate_hit_rates(df, label=f"{label} [retriever]", sort_by="retriever_score")
    calculate_hit_rates(df, label=f"{label} [reranker]", sort_by="rerank_score")


def print_subset_breakdown(df: pd.DataFrame, label: str = ""):
    for subset in sorted(df["subset"].unique()):
        print_hit_rates_both_stages(df[df["subset"] == subset], label=f"{label} | {subset}")


def run_report(df: pd.DataFrame, title: str, out_path: Path):
    print("=" * 55)
    print(title)
    print("=" * 55)
    print_hit_rates_both_stages(df, label="Overall")
    print_subset_breakdown(df, label=title)

    report = generate_question_report(df)
    report.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate ChaBo eval performance report")
    parser.add_argument("--baseline", required=True, help="Path to judged_eval_report.json")
    parser.add_argument("--filtered", help="Path to judged_eval_report_filtered.json (optional)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for CSVs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_base = load_report(args.baseline)
    run_report(df_base, "BASELINE (no filters)", out_dir / "query_performance_report.csv")

    if not args.filtered:
        return

    df_filt = load_report(args.filtered)
    run_report(df_filt, "WITH METADATA FILTERING", out_dir / "query_performance_report_filtered.csv")

    # ── Comparison ────────────────────────────────────────────────────────────
    print("=" * 55)
    print("COMPARISON: Baseline vs Filtered (delta)")
    print("=" * 55)
    for subset in sorted(df_base["subset"].unique()):
        print(f"  Subset: {subset}")
        for stage, score_col in [("retriever", "retriever_score"), ("reranker", "rerank_score")]:
            base_rates = calculate_hit_rates(
                df_base[df_base["subset"] == subset],
                label=f"Baseline [{stage}] | {subset}",
                sort_by=score_col,
            )
            filt_rates = calculate_hit_rates(
                df_filt[df_filt["subset"] == subset],
                label=f"Filtered [{stage}] | {subset}",
                sort_by=score_col,
            )
            deltas = ", ".join(f"{k}: {filt_rates[k] - base_rates[k]:+.1f}%" for k in base_rates)
            print(f"    Delta [{stage}]: {deltas}")
        print()


if __name__ == "__main__":
    main()
