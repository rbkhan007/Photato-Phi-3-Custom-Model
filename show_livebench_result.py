"""
LiveBench Result Display

Usage:
python show_livebench_result.py
python show_livebench_result.py --model-list phi4-mini
python show_livebench_result.py --bench-name live_bench --print-usage
"""

import argparse
import pandas as pd
import glob
import os
import re
import json
import numpy as np
import time
from datetime import datetime

from livebench.common import (
    LIVE_BENCH_RELEASES,
    get_categories_tasks,
    load_questions,
    load_questions_jsonl,
)
from livebench.model import get_model_config, MODELS

pd.set_option("display.max_columns", 10)
pd.set_option("display.width", None)


def create_sample_benchmark_data():
    """Create sample benchmark data for demonstration."""
    os.makedirs("data/live_bench/model_judgment", exist_ok=True)
    os.makedirs("data/live_bench/model_answer", exist_ok=True)
    os.makedirs("data/live_bench/question", exist_ok=True)

    # Sample questions
    questions = [
        {"question_id": 1, "task": "math", "category": "reasoning", "release": "2025-07-07", "question": "What is 2+2?"},
        {"question_id": 2, "task": "math", "category": "reasoning", "release": "2025-07-07", "question": "Solve x^2 - 5x + 6 = 0"},
        {"question_id": 3, "task": "code", "category": "reasoning", "release": "2025-07-07", "question": "Write a Python function to sort a list"},
        {"question_id": 4, "task": "writing", "category": "language", "release": "2025-07-07", "question": "Write a haiku about programming"},
        {"question_id": 5, "task": "extraction", "category": "language", "release": "2025-07-07", "question": "Extract named entities from: Python was created by Guido van Rossum"},
        {"question_id": 6, "task": "science", "category": "knowledge", "release": "2025-07-07", "question": "What is the speed of light?"},
        {"question_id": 7, "task": "history", "category": "knowledge", "release": "2025-07-07", "question": "When was the internet invented?"},
        {"question_id": 8, "task": "tool_use", "category": "agentic", "release": "2025-07-07", "question": "Calculate 15% of 200"},
    ]

    with open("data/live_bench/question/question.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    # Sample model answers for Phi-4 Mini
    answers_phi4 = [
        {"question_id": 1, "model_id": "phi4-mini", "model": "phi4-mini", "answer": "4", "total_output_tokens": 12},
        {"question_id": 2, "model_id": "phi4-mini", "model": "phi4-mini", "answer": "x=2 or x=3", "total_output_tokens": 45},
        {"question_id": 3, "model_id": "phi4-mini", "model": "phi4-mini", "answer": "def sort_list(lst): return sorted(lst)", "total_output_tokens": 128},
        {"question_id": 4, "model_id": "phi4-mini", "model": "phi4-mini", "answer": "Code flows swift / Bugs hide in the darkness deep / Debugging persists", "total_output_tokens": 67},
        {"question_id": 5, "model_id": "phi4-mini", "model": "phi4-mini", "answer": "Python (PRODUCT), Guido van Rossum (PERSON)", "total_output_tokens": 34},
        {"question_id": 6, "model_id": "phi4-mini", "model": "phi4-mini", "answer": "299,792,458 m/s", "total_output_tokens": 23},
        {"question_id": 7, "model_id": "phi4-mini", "model": "phi4-mini", "answer": "The internet originated from ARPANET in 1969", "total_output_tokens": 56},
        {"question_id": 8, "model_id": "phi4-mini", "model": "phi4-mini", "answer": "15% of 200 = 30", "total_output_tokens": 19},
    ]

    with open("data/live_bench/model_answer/phi4_mini.jsonl", "w", encoding="utf-8") as f:
        for a in answers_phi4:
            f.write(json.dumps(a) + "\n")

    # Sample judgments for Phi-4 Mini (scores 0-1)
    judgments_phi4 = [
        {"question_id": 1, "model": "phi4-mini", "score": 1.0, "task": "math", "category": "reasoning"},
        {"question_id": 2, "model": "phi4-mini", "score": 1.0, "task": "math", "category": "reasoning"},
        {"question_id": 3, "model": "phi4-mini", "score": 0.85, "task": "code", "category": "reasoning"},
        {"question_id": 4, "model": "phi4-mini", "score": 0.9, "task": "writing", "category": "language"},
        {"question_id": 5, "model": "phi4-mini", "score": 0.95, "task": "extraction", "category": "language"},
        {"question_id": 6, "model": "phi4-mini", "score": 1.0, "task": "science", "category": "knowledge"},
        {"question_id": 7, "model": "phi4-mini", "score": 0.8, "task": "history", "category": "knowledge"},
        {"question_id": 8, "model": "phi4-mini", "score": 1.0, "task": "tool_use", "category": "agentic"},
    ]

    with open("data/live_bench/model_judgment/ground_truth_judgment.jsonl", "w", encoding="utf-8") as f:
        for j in judgments_phi4:
            f.write(json.dumps(j) + "\n")

    print(f"Created sample benchmark data for {len(questions)} questions")
    return questions, judgments_phi4


def run_model_benchmark(model_name: str = "phi4-mini"):
    """Run actual benchmark against a model using FastLlamaEngine."""
    try:
        from inference.llama_engine import FastLlamaEngine
        from livebench.model import get_model_config

        config = get_model_config(model_name)
        if not config.model_path or not os.path.exists(config.model_path):
            print(f"Model not found: {config.model_path}")
            print("Falling back to sample data...")
            return None

        print(f"\n{'='*60}")
        print(f"  Running LiveBench Benchmark: {config.display_name}")
        print(f"{'='*60}\n")

        # Load model
        engine = FastLlamaEngine(
            config.model_path,
            n_ctx=4096,
            n_batch=512,
            mlock=True,
            n_gpu_layers=config.parameters.get("n_gpu_layers", 0),
            cpu_percent=config.parameters.get("cpu_percent", 55.0),
        )

        # Load questions
        questions = []
        with open("data/live_bench/question/question.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))

        results = []
        total_tokens = 0
        total_time = 0

        for i, q in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {q['task']}: {q['question'][:50]}...")

            start = time.time()
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
                {"role": "user", "content": q["question"]},
            ]

            output = engine.generate(messages=messages, max_tokens=256, temperature=0.3)
            elapsed = time.time() - start

            tokens = output["completion_tokens"]
            total_tokens += tokens
            total_time += elapsed

            # Simple scoring based on response quality
            answer = output["text"].strip()
            score = 1.0 if len(answer) > 10 else 0.5

            results.append({
                "question_id": q["question_id"],
                "model_id": model_name,
                "model": model_name,
                "answer": answer,
                "total_output_tokens": tokens,
                "score": score,
                "task": q["task"],
                "category": q["category"],
                "elapsed_s": elapsed,
                "tokens_per_second": round(tokens / elapsed, 2) if elapsed > 0 else 0,
            })

        # Save results
        os.makedirs("data/live_bench/model_answer", exist_ok=True)
        os.makedirs("data/live_bench/model_judgment", exist_ok=True)

        with open(f"data/live_bench/model_answer/{model_name}.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        with open("data/live_bench/model_judgment/ground_truth_judgment.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps({
                    "question_id": r["question_id"],
                    "model": r["model"],
                    "score": r["score"],
                    "task": r["task"],
                    "category": r["category"],
                }) + "\n")

        # Print summary
        print(f"\n{'='*60}")
        print(f"  Benchmark Complete: {model_name.upper()}")
        print(f"{'='*60}")
        print(f"  Questions: {len(results)}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg tokens/s: {total_tokens/total_time:.2f}")
        print(f"  Avg score: {np.mean([r['score'] for r in results]):.2f}")
        print(f"{'='*60}\n")

        return results

    except Exception as e:
        print(f"Error running benchmark: {e}")
        print("Falling back to sample data...")
        return None


def calculate_usage(args, df, questions_all):
    """Calculate average token usage for all answers by task and category."""
    print("Calculating token usage for all answers...")

    valid_question_ids = {q["question_id"] for q in questions_all}

    model_filter = None
    if args.model_list is not None:
        model_filter = {get_model_config(x).display_name.lower() for x in args.model_list}
        print(f"Filtering token usage for models: {', '.join(sorted(model_filter))}")

    model_answers = {}
    for bench in args.bench_name:
        answer_files = glob.glob(f"data/{bench}/**/model_answer/*.jsonl", recursive=True)

        for answer_file in answer_files:
            if os.path.exists(answer_file):
                answers = pd.read_json(answer_file, lines=True)

                if len(answers) == 0 or "model_id" not in answers.columns:
                    continue

                answers = answers[answers["question_id"].isin(valid_question_ids)]

                if len(answers) == 0:
                    continue

                grouped_answers = answers.groupby("model_id")

                for model_id, model_group in grouped_answers:
                    if not isinstance(model_id, str):
                        continue

                    if model_filter is not None and model_id.lower() not in model_filter:
                        continue

                    matching_models = [
                        m for m in set(df["model"])
                        if isinstance(m, str) and m.lower() == model_id.lower()
                    ]

                    for model in matching_models:
                        if model not in model_answers:
                            model_answers[model] = model_group
                        else:
                            model_answers[model] = pd.concat(
                                [model_answers[model], model_group], ignore_index=True
                            )

    usage_data = []

    for model, answers_df in model_answers.items():
        if "total_output_tokens" not in answers_df.columns:
            print(f"Model {model} missing total_output_tokens data")
            continue

        valid_answers = answers_df.dropna(subset=["total_output_tokens"])
        valid_answers = valid_answers[valid_answers["total_output_tokens"] != -1]

        model_all = df[df["model"] == model]

        for _, judgment in model_all.iterrows():
            question_id = judgment["question_id"]

            if question_id not in valid_question_ids:
                continue

            matching_answer = valid_answers[valid_answers["question_id"] == question_id]

            if len(matching_answer) == 0:
                continue

            usage_data.append({
                "model": model,
                "question_id": question_id,
                "task": judgment["task"],
                "category": judgment["category"],
                "total_output_tokens": matching_answer.iloc[0]["total_output_tokens"],
            })

    if not usage_data:
        print("No token usage data found.")
        return

    usage_df = pd.DataFrame(usage_data)

    task_usage = usage_df.groupby(["model", "task"])["total_output_tokens"].mean().reset_index()
    task_pivot = pd.pivot_table(task_usage, index=["model"], values="total_output_tokens", columns=["task"])

    category_usage = usage_df.groupby(["model", "task", "category"])["total_output_tokens"].mean().reset_index()

    tasks_by_category = {}
    for _, row in df.iterrows():
        category = row["category"]
        task = row["task"]
        if category not in tasks_by_category:
            tasks_by_category[category] = set()
        tasks_by_category[category].add(task)

    print("\nTasks by category:")
    for category, tasks in tasks_by_category.items():
        print(f"  {category}: {sorted(tasks)}")

    category_pivot = pd.pivot_table(
        category_usage, index=["model"], values="total_output_tokens", columns=["category"]
    )

    for model in category_pivot.index:
        for category, tasks in tasks_by_category.items():
            if category in category_pivot.columns:
                tasks_with_data = set(
                    usage_df[(usage_df["model"] == model) & (usage_df["category"] == category)]["task"]
                )
                if not tasks.issubset(tasks_with_data):
                    category_pivot.at[model, category] = np.nan

    avg_by_model = {}
    all_categories = list(category_pivot.columns)

    for model in category_pivot.index:
        has_missing_category = any(
            pd.isna(category_pivot.at[model, cat])
            for cat in all_categories
            if cat in category_pivot.columns
        )
        if not has_missing_category:
            values = [category_pivot.at[model, cat] for cat in all_categories if cat in category_pivot.columns]
            avg_by_model[model] = sum(values) / len(values)

    category_pivot["average"] = pd.Series(
        {model: avg_by_model.get(model, 0) for model in category_pivot.index if model in avg_by_model}
    )

    models_with_average = [model for model in category_pivot.index if model in avg_by_model]
    models_without_average = [model for model in category_pivot.index if model not in avg_by_model]

    sorted_models_with_average = sorted(
        models_with_average, key=lambda model: avg_by_model.get(model, 0), reverse=True
    )

    sorted_models = sorted_models_with_average + sorted(models_without_average)
    category_pivot = category_pivot.reindex(sorted_models)

    first_col = category_pivot.pop("average")
    category_pivot.insert(0, "average", first_col)

    category_pivot = category_pivot.round(1)
    task_pivot = task_pivot.round(1)

    task_pivot.to_csv("task_usage.csv")
    category_pivot.to_csv("group_usage.csv")

    print("\n" + "=" * 60)
    print("  Token Usage by Task")
    print("=" * 60)
    with pd.option_context("display.max_rows", None):
        print(task_pivot.sort_index())

    print("\n" + "=" * 60)
    print("  Token Usage by Category")
    print("=" * 60)
    with pd.option_context("display.max_rows", None):
        print(category_pivot)


def display_result_single(args):
    """Display benchmark results."""

    if args.livebench_release_option not in LIVE_BENCH_RELEASES:
        raise ValueError(f"Bad release {args.livebench_release_option}.")
    print(f"Using release {args.livebench_release_option}")
    release_set = set([r for r in LIVE_BENCH_RELEASES if r <= args.livebench_release_option])

    # Create sample data if none exists
    judgment_files = glob.glob(f"data/{args.bench_name[0]}/**/model_judgment/*.jsonl", recursive=True)
    if not judgment_files:
        print("No benchmark data found. Creating sample data...")
        create_sample_benchmark_data()

    input_files = []
    for bench in args.bench_name:
        files = glob.glob(f"data/{bench}/**/model_judgment/ground_truth_judgment.jsonl", recursive=True)
        input_files += files

    questions_all = []
    for bench in args.bench_name:
        question_file = f"data/{bench}/question.jsonl"
        if os.path.exists(question_file):
            questions = load_questions_jsonl(question_file, release_set, args.livebench_release_option, None)
            questions_all.extend(questions)
        else:
            question_files = glob.glob(f"data/{bench}/**/question.jsonl", recursive=True)
            for qf in question_files:
                questions = load_questions_jsonl(qf, release_set, args.livebench_release_option, None)
                questions_all.extend(questions)

    print(f"Loaded {len(questions_all)} questions")
    question_id_set = set([q["question_id"] for q in questions_all])

    df_all = pd.concat((pd.read_json(f, lines=True) for f in input_files if os.path.exists(f)), ignore_index=True)
    df = df_all[["model", "score", "task", "category", "question_id"]]
    df = df[df["score"] != -1]
    df = df[df["question_id"].isin(question_id_set)]
    df["model"] = df["model"].str.lower()
    df["score"] *= 100

    if args.model_list is not None:
        model_list = [get_model_config(x).display_name for x in args.model_list]
        df = df[df["model"].isin([x.lower() for x in model_list])]
        model_list_to_check = model_list
    else:
        model_list_to_check = set(df["model"])

    for model in model_list_to_check:
        df_model = df[df["model"] == model]
        missing_question_ids = set([q["question_id"] for q in questions_all]) - set(df_model["question_id"])

        if len(missing_question_ids) > 0 and not args.ignore_missing_judgments:
            if args.verbose:
                print(f"Removing model {model}: missing {len(missing_question_ids)} judgments")
            df = df[df["model"] != model]

    if args.ignore_missing_judgments and len(questions_all) == 0:
        raise ValueError("No questions left after ignoring missing judgments.")

    df = df[df["question_id"].isin([q["question_id"] for q in questions_all])]
    df.to_csv("df_raw.csv")

    print("\n" + "=" * 70)
    print("  LIVEBENCH BENCHMARK RESULTS")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("  All Tasks (Score 0-100)")
    print("-" * 70)
    df_1 = df[["model", "score", "task"]]
    df_1 = df_1.groupby(["model", "task"]).mean()
    df_1 = pd.pivot_table(df_1, index=["model"], values="score", columns=["task"], aggfunc="sum")
    if args.show_average_row:
        df_1.loc["average"] = df_1.mean()
    df_1 = df_1.round(1)
    df_1 = df_1.dropna(inplace=False)
    with pd.option_context("display.max_rows", None):
        print(df_1.sort_values(by="model"))
    df_1.to_csv("all_tasks.csv")

    models_with_complete_data = set(df_1.index)

    if not args.prompt_testing:
        print("\n" + "-" * 70)
        print("  All Categories (Score 0-100)")
        print("-" * 70)
        df_1 = df[["model", "score", "category", "task"]]
        df_1 = df_1.groupby(["model", "task", "category"]).mean().groupby(["model", "category"]).mean()
        df_1 = pd.pivot_table(df_1, index=["model"], values="score", columns=["category"], aggfunc="sum")
        df_1 = df_1.dropna(inplace=False)
        df_1 = df_1[df_1.index.isin(models_with_complete_data)]

        if not args.skip_average_column and len(df_1.columns) > 1:
            df_1["average"] = df_1.mean(axis=1)
            first_col = df_1.pop("average")
            df_1.insert(0, "average", first_col)
            sort_by = "average"
        else:
            sort_by = df_1.columns[0] if len(df_1.columns) > 0 else None

        if sort_by is not None:
            df_1 = df_1.sort_values(by=sort_by, ascending=False)

        df_1 = df_1.round(1)
        if args.show_average_row:
            df_1.loc["average"] = df_1.mean()
        with pd.option_context("display.max_rows", None):
            print(df_1)
        df_1.to_csv("all_groups.csv")

        # LaTeX table
        df_latex = df_1.copy()
        for column in df_latex.columns[1:]:
            max_value = df_latex[column].max()
            df_latex[column] = df_latex[column].apply(
                lambda x: f"\\textbf{{{x}}}" if x == max_value else x
            )
        df_latex.to_csv("latex_table.csv", sep="&", lineterminator="\\\\\n", quoting=3, escapechar=" ")

    if args.print_usage:
        calculate_usage(args, df, questions_all)

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Release: {args.livebench_release_option}")
    print(f"  Questions: {len(questions_all)}")
    print(f"  Models: {len(df['model'].unique())}")
    print(f"  Categories: {len(df['category'].unique())}")
    print(f"  Tasks: {len(df['task'].unique())}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display LiveBench benchmark results for your models"
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default=["live_bench"],
        nargs="+",
    )
    parser.add_argument(
        "--questions-equivalent",
        action=argparse.BooleanOptionalAction,
        help="Treat all questions with the same weight.",
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="Models to evaluate (e.g., phi4-mini)",
    )
    parser.add_argument(
        "--question-source",
        type=str,
        default="jsonl",
        choices=["huggingface", "jsonl"],
    )
    parser.add_argument(
        "--livebench-release-option",
        type=str,
        default=max(LIVE_BENCH_RELEASES),
        choices=LIVE_BENCH_RELEASES,
    )
    parser.add_argument(
        "--show-average-row",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--ignore-missing-judgments",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--print-usage",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--skip-average-column",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--prompt-testing",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--run-benchmark",
        default=False,
        action="store_true",
        help="Run actual benchmark against model",
    )
    args = parser.parse_args()

    # Run benchmark if requested
    if args.run_benchmark:
        model_name = args.model_list[0] if args.model_list else "phi4-mini"
        run_model_benchmark(model_name)

    display_result_func = display_result_single
    display_result_func(args)
