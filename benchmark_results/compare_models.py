"""
LiveBench Benchmark Comparison Tool

Compares multiple models across tasks, categories, and generates reports.
Supports CSV, LaTeX, and Markdown output.

Usage:
    python benchmark_results/compare_models.py
    python benchmark_results/compare_models.py --models phi4-mini llama3.2
    python benchmark_results/compare_models.py --generate-latex
"""

import argparse
import os
import sys
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livebench import LIVE_BENCH_RELEASES, CATEGORIES, TASKS
from livebench.model import get_model_config, MODELS


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_all_results(bench_names: list = None) -> pd.DataFrame:
    """Load all model judgment results."""
    if bench_names is None:
        bench_names = ["live_bench"]

    files = []
    for bench_name in bench_names:
        # Try multiple path patterns
        patterns = [
            f"data/{bench_name}/**/model_judgment/*.jsonl",
            f"data/{bench_name}/model_judgment/*.jsonl",
        ]
        for pattern in patterns:
            files.extend(glob.glob(pattern, recursive=True))

    if not files:
        print(f"No judgment files found")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_json(f, lines=True)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def load_all_answers(bench_names: list = None) -> pd.DataFrame:
    """Load all model answer files."""
    if bench_names is None:
        bench_names = ["live_bench"]

    files = []
    for bench_name in bench_names:
        patterns = [
            f"data/{bench_name}/**/model_answer/*.jsonl",
            f"data/{bench_name}/model_answer/*.jsonl",
        ]
        for pattern in patterns:
            files.extend(glob.glob(pattern, recursive=True))

    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_json(f, lines=True)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def compute_task_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average scores by model and task."""
    df = df.copy()
    df["score"] = df["score"] * 100  # Convert to 0-100 scale

    task_scores = df.groupby(["model", "task"])["score"].mean().reset_index()
    pivot = pd.pivot_table(
        task_scores, index="model", values="score", columns="task", aggfunc="sum"
    )
    return pivot


def compute_category_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average scores by model and category."""
    df = df.copy()
    df["score"] = df["score"] * 100

    cat_scores = (
        df.groupby(["model", "task", "category"])["score"]
        .mean()
        .groupby(["model", "category"])
        .mean()
        .reset_index()
    )
    pivot = pd.pivot_table(
        cat_scores, index="model", values="score", columns="category", aggfunc="sum"
    )

    # Add average column
    if len(pivot.columns) > 1:
        pivot["average"] = pivot.mean(axis=1)
        first_col = pivot.pop("average")
        pivot.insert(0, "average", first_col)

    return pivot.sort_values(by="average" if "average" in pivot.columns else pivot.columns[0], ascending=False)


def compute_token_usage(answers_df: pd.DataFrame, judgments_df: pd.DataFrame) -> pd.DataFrame:
    """Compute token usage statistics."""
    if "total_output_tokens" not in answers_df.columns:
        return pd.DataFrame()

    # Merge answers with judgments
    merged = judgments_df.merge(
        answers_df[["model", "question_id", "total_output_tokens"]],
        on=["model", "question_id"],
        how="left",
    )

    merged = merged.dropna(subset=["total_output_tokens"])
    merged = merged[merged["total_output_tokens"] != -1]

    # Compute by model and task
    usage = merged.groupby(["model", "task"])["total_output_tokens"].mean().reset_index()
    pivot = pd.pivot_table(
        usage, index="model", values="total_output_tokens", columns="task"
    )
    return pivot


def compute_category_usage(answers_df: pd.DataFrame, judgments_df: pd.DataFrame) -> pd.DataFrame:
    """Compute token usage by category."""
    if "total_output_tokens" not in answers_df.columns:
        return pd.DataFrame()

    merged = judgments_df.merge(
        answers_df[["model", "question_id", "total_output_tokens"]],
        on=["model", "question_id"],
        how="left",
    )

    merged = merged.dropna(subset=["total_output_tokens"])
    merged = merged[merged["total_output_tokens"] != -1]

    usage = (
        merged.groupby(["model", "category"])["total_output_tokens"]
        .mean()
        .reset_index()
    )
    pivot = pd.pivot_table(
        usage, index="model", values="total_output_tokens", columns="category"
    )

    if len(pivot.columns) > 1:
        pivot["average"] = pivot.mean(axis=1)
        first_col = pivot.pop("average")
        pivot.insert(0, "average", first_col)

    return pivot


def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Generate LaTeX table from DataFrame."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\small")

    # Column alignment
    cols = "l" + "c" * len(df.columns)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append(r"\toprule")

    # Header
    header = " & ".join([r"\textbf{" + str(c) + "}" for c in df.columns])
    lines.append(f"Model & {header} \\\\")
    lines.append(r"\midrule")

    # Find max per column for bold
    max_vals = df.max()

    # Rows
    for model, row in df.iterrows():
        cells = []
        for col, val in row.items():
            if val == max_vals[col]:
                cells.append(f"\\textbf{{{val:.1f}}}")
            else:
                cells.append(f"{val:.1f}")
        lines.append(f"{model} & {' & '.join(cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_markdown_report(
    task_scores: pd.DataFrame,
    category_scores: pd.DataFrame,
    token_usage: pd.DataFrame,
    category_usage: pd.DataFrame,
) -> str:
    """Generate Markdown comparison report."""
    lines = []
    lines.append("# LiveBench Model Comparison Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Task Scores
    lines.append("## Task Scores (0-100)\n")
    lines.append("| Model | " + " | ".join(task_scores.columns) + " |")
    lines.append("|" + "---|" * (len(task_scores.columns) + 1))
    for model, row in task_scores.iterrows():
        vals = " | ".join([f"{v:.1f}" for v in row])
        lines.append(f"| {model} | {vals} |")

    # Category Scores
    lines.append("\n## Category Scores (0-100)\n")
    lines.append("| Model | " + " | ".join(category_scores.columns) + " |")
    lines.append("|" + "---|" * (len(category_scores.columns) + 1))
    for model, row in category_scores.iterrows():
        vals = " | ".join([f"{v:.1f}" for v in row])
        lines.append(f"| {model} | {vals} |")

    # Token Usage
    if not token_usage.empty:
        lines.append("\n## Token Usage (Avg Output Tokens)\n")
        lines.append("| Model | " + " | ".join(token_usage.columns) + " |")
        lines.append("|" + "---|" * (len(token_usage.columns) + 1))
        for model, row in token_usage.iterrows():
            vals = " | ".join([f"{v:.0f}" for v in row])
            lines.append(f"| {model} | {vals} |")

    # Rankings
    lines.append("\n## Rankings\n")
    if "average" in category_scores.columns:
        ranked = category_scores.sort_values("average", ascending=False)
        for i, (model, row) in enumerate(ranked.iterrows(), 1):
            medal = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else f" {i}."
            lines.append(f"{medal} **{model}** — {row['average']:.1f}")

    return "\n".join(lines)


def generate_json_report(
    task_scores: pd.DataFrame,
    category_scores: pd.DataFrame,
    token_usage: pd.DataFrame,
) -> dict:
    """Generate JSON report."""
    return {
        "generated_at": datetime.now().isoformat(),
        "models": list(task_scores.index),
        "task_scores": task_scores.to_dict(),
        "category_scores": category_scores.to_dict(),
        "token_usage": token_usage.to_dict() if not token_usage.empty else {},
        "summary": {
            "best_model": category_scores["average"].idxmax()
            if "average" in category_scores.columns
            else None,
            "best_score": float(category_scores["average"].max())
            if "average" in category_scores.columns
            else None,
        },
    }


def print_comparison_table(df: pd.DataFrame, title: str):
    """Pretty print a comparison table."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(df.round(1).to_string())
    print(f"{'='*70}")


def run_comparison(args):
    """Run the full comparison pipeline."""
    print("\n" + "=" * 70)
    print("  LiveBench Model Comparison Tool")
    print("=" * 70)

    # Load data
    judgments_df = load_all_results(args.bench_name)
    answers_df = load_all_answers(args.bench_name)

    if judgments_df.empty:
        print("\nNo judgment data found. Run benchmarks first:")
        print("  python show_livebench_result.py --model-list phi4-mini --run-benchmark")
        return

    print(f"\nLoaded {len(judgments_df)} judgments")
    print(f"Models: {judgments_df['model'].unique().tolist()}")

    # Filter models if specified
    if args.models:
        judgments_df = judgments_df[judgments_df["model"].isin(args.models)]
        if not answers_df.empty:
            answers_df = answers_df[answers_df["model_id"].isin(args.models)]

    # Compute scores
    task_scores = compute_task_scores(judgments_df)
    category_scores = compute_category_scores(judgments_df)

    # Compute token usage
    token_usage = pd.DataFrame()
    category_usage = pd.DataFrame()
    if not answers_df.empty:
        token_usage = compute_token_usage(answers_df, judgments_df)
        category_usage = compute_category_usage(answers_df, judgments_df)

    # Print tables
    print_comparison_table(task_scores, "Task Scores (0-100)")
    print_comparison_table(category_scores, "Category Scores (0-100)")

    if not token_usage.empty:
        print_comparison_table(token_usage, "Token Usage (Avg Output Tokens)")
    if not category_usage.empty:
        print_comparison_table(category_usage, "Category Token Usage")

    # Save CSV files
    task_scores.round(1).to_csv(os.path.join(OUTPUT_DIR, "task_scores.csv"))
    category_scores.round(1).to_csv(os.path.join(OUTPUT_DIR, "category_scores.csv"))
    if not token_usage.empty:
        token_usage.round(0).to_csv(os.path.join(OUTPUT_DIR, "token_usage.csv"))
    if not category_usage.empty:
        category_usage.round(0).to_csv(os.path.join(OUTPUT_DIR, "category_usage.csv"))

    # Generate Markdown report
    md_report = generate_markdown_report(task_scores, category_scores, token_usage, category_usage)
    with open(os.path.join(OUTPUT_DIR, "comparison_report.md"), "w", encoding="utf-8") as f:
        f.write(md_report)

    # Generate JSON report
    json_report = generate_json_report(task_scores, category_scores, token_usage)
    with open(os.path.join(OUTPUT_DIR, "comparison_report.json"), "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    # Generate LaTeX tables
    if args.generate_latex:
        latex_task = generate_latex_table(task_scores, "Task Scores by Model", "tab:task_scores")
        latex_cat = generate_latex_table(category_scores, "Category Scores by Model", "tab:category_scores")

        with open(os.path.join(OUTPUT_DIR, "task_scores.tex"), "w", encoding="utf-8") as f:
            f.write(latex_task)
        with open(os.path.join(OUTPUT_DIR, "category_scores.tex"), "w", encoding="utf-8") as f:
            f.write(latex_cat)

        print(f"\nLaTeX tables saved to {OUTPUT_DIR}/")

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if "average" in category_scores.columns:
        best = category_scores["average"].idxmax()
        best_score = category_scores["average"].max()
        print(f"  Best Model: {best}")
        print(f"  Best Score: {best_score:.1f}")

    print(f"\n  Files saved:")
    print(f"    - task_scores.csv")
    print(f"    - category_scores.csv")
    print(f"    - comparison_report.md")
    print(f"    - comparison_report.json")
    if not token_usage.empty:
        print(f"    - token_usage.csv")
        print(f"    - category_usage.csv")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare LiveBench benchmark results")
    parser.add_argument(
        "--bench-name",
        type=str,
        default=["live_bench"],
        nargs="+",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Specific models to compare",
    )
    parser.add_argument(
        "--generate-latex",
        action="store_true",
        help="Generate LaTeX tables",
    )
    args = parser.parse_args()

    run_comparison(args)
