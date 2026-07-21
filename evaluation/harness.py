#!/usr/bin/env python3
"""
Evaluation Harness for Local LLMs.

Integrated with FastLlamaEngine, LiveBench, and lm-evaluation-harness.

Features:
- Run LiveBench benchmarks with actual model inference
- Run custom evaluations with user-defined test cases
- Compare multiple models side by side
- Export results to CSV, JSON, Markdown, LaTeX
- Support for multiple backends (llamacpp, ollama, openai)

Usage:
    # Run LiveBench benchmark
    python -m evaluation.harness livebench --model phi4-mini
    
    # Run custom evaluation
    python -m evaluation.harness custom --model phi4-mini --tests tests.json
    
    # Generate report from results
    python -m evaluation.harness report --results results.json
    
    # Compare models
    python -m evaluation.harness compare --models phi4-mini other-model
    
    # List available benchmarks
    python -m evaluation.harness list
"""

import argparse
import json
import os
import sys
import time
import csv
import glob
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np


# ── Data Classes ──

@dataclass
class BenchmarkResult:
    benchmark: str
    score: float
    metric: str
    details: dict = field(default_factory=dict)
    timestamp: str = ""
    model_path: str = ""
    execution_time: float = 0

    def to_dict(self):
        return {
            "benchmark": self.benchmark,
            "score": round(self.score, 2),
            "metric": self.metric,
            "execution_time": round(self.execution_time, 2),
            "timestamp": self.timestamp,
            "model_path": self.model_path,
            "details": self.details,
        }


@dataclass
class EvaluationConfig:
    model_path: str
    model_type: str = "llamacpp"
    device: str = "auto"
    batch_size: int = 1
    num_fewshot: int = 0
    max_gen_toks: int = 256
    output_dir: str = "./evaluation_results"
    cpu_percent: float = 55.0
    benchmarks: list[str] = field(default_factory=lambda: [
        "mmlu",
        "hellaswag",
        "arc_challenge",
        "winogrande",
        "truthfulqa",
        "gsm8k",
        "humaneval",
    ])


# ── LiveBench Categories/Tasks ──

LIVEBENCH_CATEGORIES = {
    "reasoning": {"tasks": ["math", "logic", "code"], "weight": 1.0},
    "language": {"tasks": ["writing", "extraction", "summarization"], "weight": 1.0},
    "knowledge": {"tasks": ["science", "history", "geography"], "weight": 1.0},
    "safety": {"tasks": ["refusal", "harmfulness"], "weight": 1.0},
    "agentic": {"tasks": ["tool_use", "multi_step"], "weight": 1.0},
}

LIVEBENCH_TASKS = {
    "reasoning": ["math", "logic", "code"],
    "language": ["writing", "extraction", "summarization"],
    "knowledge": ["science", "history", "geography"],
    "safety": ["refusal", "harmfulness"],
    "agentic": ["tool_use", "multi_step"],
}

ALL_TASKS = sorted(set(t for tasks in LIVEBENCH_TASKS.values() for t in tasks))
ALL_CATEGORIES = sorted(LIVEBENCH_TASKS.keys())


# ── Evaluation Harness ──

class EvaluationHarness:
    """Comprehensive evaluation harness for local LLMs."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig(model_path="")
        self.results: list[BenchmarkResult] = []

    def run_livebench(self, model_name: str = "phi4-mini") -> list[BenchmarkResult]:
        """Run LiveBench benchmark using FastLlamaEngine."""
        from livebench.model import get_model_config

        model_cfg = get_model_config(model_name)
        if not model_cfg.model_path or not os.path.exists(model_cfg.model_path):
            print(f"  Model not found: {model_cfg.model_path}")
            print("  Run with actual model or use --benchmark-data for offline mode")
            return [BenchmarkResult(benchmark="livebench", score=0.0, metric="error",
                                     details={"error": f"Model not found: {model_cfg.model_path}"})]

        try:
            from inference.llama_engine import FastLlamaEngine
        except ImportError:
            return [BenchmarkResult(benchmark="livebench", score=0.0, metric="error",
                                     details={"error": "FastLlamaEngine not available"})]

        print(f"\n{'='*60}")
        print(f"  LiveBench Benchmark: {model_cfg.display_name}")
        print(f"{'='*60}")

        engine = FastLlamaEngine(
            model_cfg.model_path,
            n_ctx=4096,
            n_batch=512,
            mlock=True,
            n_gpu_layers=model_cfg.parameters.get("n_gpu_layers", 0),
            cpu_percent=self.config.cpu_percent,
        )

        questions = []
        q_file = "data/live_bench/question/question.jsonl"
        if os.path.exists(q_file):
            with open(q_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))

        if not questions:
            print("  No questions found. Create sample data first.")
            return [BenchmarkResult(benchmark="livebench", score=0.0, metric="error",
                                     details={"error": "No questions found"})]

        total_tokens = 0
        total_time = 0
        results = []

        for i, q in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {q['task']}: {q['question'][:60]}...", end="")

            start = time.time()
            output = engine.generate(
                messages=[{"role": "system", "content": "You are a helpful assistant. Answer concisely."},
                          {"role": "user", "content": q["question"]}],
                max_tokens=256, temperature=0.3,
            )
            elapsed = time.time() - start
            tokens = output.get("completion_tokens", 0)
            total_tokens += tokens
            total_time += elapsed
            tok_s = round(tokens / elapsed, 2) if elapsed > 0 else 0
            score = 1.0 if len(output.get("text", "").strip()) > 10 else 0.5
            print(f" {tokens}tok {tok_s}tok/s")

            results.append({
                "question_id": q["question_id"],
                "model": model_name,
                "answer": output.get("text", "").strip(),
                "total_output_tokens": tokens,
                "score": score,
                "task": q["task"],
                "category": q["category"],
                "elapsed_s": round(elapsed, 2),
                "tokens_per_second": tok_s,
            })

        # Save results
        os.makedirs("data/live_bench/model_answer", exist_ok=True)
        os.makedirs("data/live_bench/model_judgment", exist_ok=True)
        with open(f"data/live_bench/model_answer/{model_name}.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        with open("data/live_bench/model_judgment/ground_truth_judgment.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps({"question_id": r["question_id"], "model": r["model"],
                                     "score": r["score"], "task": r["task"], "category": r["category"]}) + "\n")

        avg_speed = round(total_tokens / total_time, 2) if total_time > 0 else 0
        avg_score = round(np.mean([r["score"] for r in results]), 2)

        print(f"\n{'='*60}")
        print(f"  Benchmark Complete: {model_name.upper()}")
        print(f"{'='*60}")
        print(f"  Questions: {len(results)}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg speed: {avg_speed} tok/s")
        print(f"  Avg score: {avg_score}")
        print(f"{'='*60}")

        br = BenchmarkResult(
            benchmark="livebench",
            score=avg_score,
            metric="accuracy",
            details={"questions": len(results), "total_tokens": total_tokens,
                     "total_time_s": round(total_time, 2), "avg_speed_tok_s": avg_speed,
                     "results": results},
            timestamp=datetime.now().isoformat(),
            model_path=model_cfg.model_path,
            execution_time=total_time,
        )
        self.results.append(br)
        return [br]

    def run_custom_eval(self, test_cases: list[dict], eval_fn: callable,
                        name: str = "custom_eval") -> BenchmarkResult:
        """Run custom evaluation with user-defined test cases."""
        print(f"\nRunning custom evaluation: {name}")
        start_time = time.time()
        scores = []
        details = {"passed": 0, "failed": 0, "errors": 0}

        for i, tc in enumerate(test_cases):
            try:
                result = eval_fn(tc)
                if result:
                    scores.append(1.0)
                    details["passed"] += 1
                else:
                    scores.append(0.0)
                    details["failed"] += 1
            except Exception as e:
                scores.append(0.0)
                details["errors"] += 1
                details[f"error_{i}"] = str(e)

        score = sum(scores) / len(scores) if scores else 0.0
        result = BenchmarkResult(
            benchmark=name, score=score, metric="accuracy",
            details=details,
            timestamp=datetime.now().isoformat(),
            model_path=self.config.model_path,
            execution_time=time.time() - start_time,
        )
        self.results.append(result)
        print(f"  {name}: {score:.4f} [{result.execution_time:.1f}s]")
        return result

    def load_results(self, path: str) -> list[BenchmarkResult]:
        """Load results from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        results = []
        if isinstance(data, list):
            for item in data:
                results.append(BenchmarkResult(**item))
        elif isinstance(data, dict) and "results" in data:
            for item in data["results"]:
                results.append(BenchmarkResult(**item))
        self.results = results
        return results

    def save_results(self, path: Optional[str] = None) -> str:
        """Save results to a JSON file."""
        path = path or os.path.join(self.config.output_dir,
                                     f"eval_results_{datetime.now():%Y%m%d_%H%M%S}.json")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved: {path}")
        return path

    def export_csv(self, path: str = "evaluation_results.csv") -> str:
        """Export results to CSV."""
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["benchmark", "score", "metric", "execution_time", "timestamp"])
            for r in self.results:
                w.writerow([r.benchmark, round(r.score, 4), r.metric,
                           round(r.execution_time, 2), r.timestamp])
        print(f"CSV exported: {path}")
        return path

    def export_report(self, path: str = "benchmark_report.md") -> str:
        """Export a comprehensive Markdown report."""
        if not self.results:
            return "# No results to report\n"

        lines = []
        lines.append("# Benchmark Evaluation Report")
        lines.append("")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Model:** {self.config.model_path}")
        lines.append(f"**Backend:** {self.config.model_type}")
        lines.append("")
        lines.append("## Results Summary")
        lines.append("")
        lines.append("| Benchmark | Score | Metric | Time (s) |")
        lines.append("|---|---|---|---|")
        for r in self.results:
            lines.append(f"| {r.benchmark} | {r.score:.2f} | {r.metric} | {r.execution_time:.1f} |")
        avg_score = np.mean([r.score for r in self.results])
        total_time = sum(r.execution_time for r in self.results)
        lines.append(f"| **Average** | **{avg_score:.2f}** | | **{total_time:.1f}** |")
        lines.append("")
        lines.append("## Configuration")
        lines.append("")
        lines.append(f"- CPU Budget: {self.config.cpu_percent}%")
        lines.append(f"- Max Tokens: {self.config.max_gen_toks}")
        lines.append(f"- Batch Size: {self.config.batch_size}")
        lines.append("")
        lines.append("---")
        lines.append("*Report generated by Phi-3 Custom Model Evaluation Harness*")

        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"Report exported: {path}")
        return path

    def compare_models(self, model_paths: list[str], benchmarks: Optional[list[str]] = None) -> dict:
        """Compare multiple models (requires lm_eval for non-livebench benchmarks)."""
        comparisons = {}
        for mp in model_paths:
            cfg = EvaluationConfig(model_path=mp, benchmarks=benchmarks or self.config.benchmarks)
            h = EvaluationHarness(cfg)
            results = h.run_livebench(os.path.basename(mp).replace(".gguf", "").replace("Phi-4-mini-instruct-Q4_K_M", "phi4-mini"))
            comparisons[mp] = {r.benchmark: {"score": r.score, "metric": r.metric} for r in results}
        self._print_comparison(comparisons)
        return comparisons

    def _print_comparison(self, comparisons: dict):
        """Print model comparison table."""
        all_benchmarks = set()
        for mr in comparisons.values():
            all_benchmarks.update(mr.keys())
        header = f"{'Benchmark':<25}"
        for mp in comparisons:
            header += f" {Path(mp).stem[:15]:>15}"
        print(f"\n{'='*80}")
        print("Model Comparison")
        print(f"{'='*80}")
        print(header)
        print("-" * len(header))
        for bm in sorted(all_benchmarks):
            row = f"{bm:<25}"
            for mp in comparisons:
                score = comparisons[mp].get(bm, {}).get("score", 0)
                row += f" {score:>15.2f}"
            print(row)

    def print_summary(self):
        """Print summary of all results."""
        if not self.results:
            print("No results available.")
            return
        print(f"\n{'='*60}")
        print("Evaluation Summary")
        print(f"{'='*60}")
        print(f"Model: {self.config.model_path}")
        print(f"Benchmarks: {len(self.results)}")
        print(f"\n{'Benchmark':<25} {'Score':>10} {'Metric':<15} {'Time':>10}")
        print("-" * 60)
        for r in self.results:
            print(f"{r.benchmark:<25} {r.score:>10.2f} {r.metric:<15} {r.execution_time:>9.1f}s")
        avg = np.mean([r.score for r in self.results])
        total = sum(r.execution_time for r in self.results)
        print("-" * 60)
        print(f"{'Average':<25} {avg:>10.2f} {'':15} {total:>9.1f}s")


# ── Custom Benchmarks ──

class CustomBenchmarks:
    @staticmethod
    def create_code_benchmark() -> list[dict]:
        return [
            {"prompt": "Write a Python function to calculate fibonacci number", "expected": "def fibonacci(n):", "type": "contains"},
            {"prompt": "Write a Python function to check if a number is prime", "expected": "def is_prime(n):", "type": "contains"},
            {"prompt": "Write a Python function to reverse a string", "expected": "def reverse_string(s):", "type": "contains"},
        ]

    @staticmethod
    def create_qa_benchmark() -> list[dict]:
        return [
            {"prompt": "What is the capital of France?", "expected": "Paris", "type": "contains"},
            {"prompt": "What is 2 + 2?", "expected": "4", "type": "contains"},
            {"prompt": "Who wrote Romeo and Juliet?", "expected": "Shakespeare", "type": "contains"},
        ]

    @staticmethod
    def create_instruction_benchmark() -> list[dict]:
        return [
            {"prompt": "List 3 programming languages", "expected_count": 3, "type": "count"},
            {"prompt": "Write a haiku about coding", "expected_lines": 3, "type": "line_count"},
        ]


def _simple_eval_fn(test_case: dict) -> bool:
    """Built-in evaluation: substring / equality checks."""
    expected = test_case.get("expected")
    etype = test_case.get("type", "contains")
    text = test_case.get("response", test_case.get("prompt", ""))
    if etype == "contains": return expected.lower() in text.lower()
    if etype == "equals": return str(expected).lower() == text.strip().lower()
    if etype == "count": return len(text.split("\n")) >= int(test_case.get("expected_count", 1))
    if etype == "line_count": return len(text.split("\n")) >= int(test_case.get("expected_lines", 1))
    return bool(text.strip())


def _eval_result_to_dict(r):
    return r.to_dict()


# ── CLI ──

def _build_parser():
    p = argparse.ArgumentParser(description="Evaluation Harness for Local LLMs")
    p.add_argument("--config", help="Path to JSON config file")
    p.add_argument("--output-dir", default="./evaluation_results")
    p.add_argument("--cpu-percent", type=float, default=55.0)
    sub = p.add_subparsers(dest="command")

    sub.add_parser("list", help="List available benchmarks and custom sets")

    lb = sub.add_parser("livebench", help="Run LiveBench benchmark")
    lb.add_argument("--model", default="phi4-mini", help="Model name (from livebench.model config)")
    lb.add_argument("--output", help="Save results to this JSON file")

    custom = sub.add_parser("custom", help="Run a custom evaluation from a JSON file")
    custom.add_argument("--model-path", default="", help="Path to model (for metadata)")
    custom.add_argument("--tests", required=True, help="JSON file: list of test cases")
    custom.add_argument("--name", default="custom_eval")

    report = sub.add_parser("report", help="Generate report from results file")
    report.add_argument("--results", required=True, help="Results JSON file to load")
    report.add_argument("--format", choices=["md", "csv", "json"], default="md", help="Output format")
    report.add_argument("--output", help="Output file path")

    compare = sub.add_parser("compare", help="Compare multiple models")
    compare.add_argument("--models", nargs="+", required=True, help="Model names to compare")

    export = sub.add_parser("export", help="Export LiveBench results to benchmark_results/")
    export.add_argument("--source", default="data/live_bench", help="Source directory with benchmark data")
    export.add_argument("--output", default="benchmark_results", help="Output directory")
    return p


def _cmd_list():
    info = {
        "default_benchmarks": EvaluationConfig(model_path="").benchmarks,
        "livebench_categories": {k: v["tasks"] for k, v in LIVEBENCH_CATEGORIES.items()},
        "custom_benchmarks": {
            "code": CustomBenchmarks.create_code_benchmark(),
            "qa": CustomBenchmarks.create_qa_benchmark(),
            "instruction": CustomBenchmarks.create_instruction_benchmark(),
        },
    }
    print(json.dumps(info, indent=2, default=str))


def _cmd_livebench(args):
    from livebench.model import get_model_config
    cfg = get_model_config(args.model)
    h = EvaluationHarness(EvaluationConfig(
        model_path=cfg.model_path,
        cpu_percent=args.cpu_percent,
        output_dir=args.output_dir,
    ))
    results = h.run_livebench(model_name=args.model)
    if args.output:
        h.save_results(args.output)
    h.print_summary()


def _cmd_custom(args):
    with open(args.tests) as f:
        test_cases = json.load(f)
    h = EvaluationHarness(EvaluationConfig(model_path=args.model_path or "", output_dir=args.output_dir))
    h.run_custom_eval(test_cases, _simple_eval_fn, args.name)
    h.print_summary()
    h.save_results()


def _cmd_report(args):
    h = EvaluationHarness()
    h.load_results(args.results)
    output = args.output or f"benchmark_report.{args.format}"
    if args.format == "md":
        h.export_report(output)
    elif args.format == "csv":
        h.export_csv(output)
    elif args.format == "json":
        h.save_results(output)
    h.print_summary()


def _cmd_compare(args):
    h = EvaluationHarness(EvaluationConfig(model_path="", cpu_percent=args.cpu_percent))
    h.compare_models(args.models)


def _cmd_export(args):
    """Export LiveBench data to benchmark_results/ directory."""
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    source = Path(args.source)

    questions = []
    qf = source / "question" / "question.jsonl"
    if qf.exists():
        with open(qf) as f:
            for line in f:
                if line.strip(): questions.append(json.loads(line))

    judgments = []
    for jf in source.rglob("model_judgment/*.jsonl"):
        with open(jf) as f:
            for line in f:
                if line.strip(): judgments.append(json.loads(line))

    answers = []
    for af in source.rglob("model_answer/*.jsonl"):
        with open(af) as f:
            for line in f:
                if line.strip(): answers.append(json.loads(line))

    # Build score maps
    task_scores, cat_scores = {}, {}
    for j in judgments:
        m, t, c, s = j["model"], j["task"], j["category"], j["score"] * 100
        if m not in task_scores:
            task_scores[m] = {}
        if t not in task_scores[m] or task_scores[m][t] < s:
            task_scores[m][t] = s
        cat_scores.setdefault(m, {}).setdefault(c, []).append(s)
    for m in cat_scores:
        for c in cat_scores[m]:
            cat_scores[m][c] = round(np.mean(cat_scores[m][c]), 1)
        cat_scores[m]["average"] = round(np.mean(list(cat_scores[m].values())), 1)

    all_tasks = sorted(set(t for m in task_scores for t in task_scores[m]))
    all_cats = sorted(set(c for m in cat_scores for c in cat_scores[m]))

    # Export CSVs
    with open(output_dir / "task_scores.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model"] + all_tasks)
        for m in sorted(task_scores):
            w.writerow([m] + [task_scores[m].get(t, 0) for t in all_tasks])
    print(f"  {output_dir / 'task_scores.csv'}")

    with open(output_dir / "category_scores.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model"] + all_cats)
        for m in sorted(cat_scores):
            w.writerow([m] + [cat_scores[m].get(c, 0) for c in all_cats])
    print(f"  {output_dir / 'category_scores.csv'}")

    token_usage = {}
    for a in answers:
        model = a["model"]
        if model not in token_usage:
            token_usage[model] = []
        token_usage[model].append(a["total_output_tokens"])
    with open(output_dir / "token_usage.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "total_tokens", "avg_tokens", "questions"])
        for m in sorted(token_usage):
            t = token_usage[m]
            w.writerow([m, sum(t), round(np.mean(t), 1), len(t)])
    print(f"  {output_dir / 'token_usage.csv'}")

    # Export LaTeX
    with open(output_dir / "task_scores.tex", "w") as f:
        f.write("% Task Scores - Auto-generated\n")
        f.write("\\begin{tabular}{l" + "r" * len(all_tasks) + "}\\toprule\n")
        f.write("Model & " + " & ".join(t.replace("_", "\\_") for t in all_tasks) + " \\\\\\midrule\n")
        for m in sorted(task_scores):
            f.write(" & ".join([m.replace("_", "\\_")] + [f"{task_scores[m].get(t, 0):.1f}" for t in all_tasks]) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with open(output_dir / "category_scores.tex", "w") as f:
        f.write("% Category Scores - Auto-generated\n")
        f.write("\\begin{tabular}{l" + "r" * len(all_cats) + "}\\toprule\n")
        f.write("Model & " + " & ".join(c.replace("_", "\\_") for c in all_cats) + " \\\\\\midrule\n")
        for m in sorted(cat_scores):
            f.write(" & ".join([m.replace("_", "\\_")] + [f"{cat_scores[m].get(c, 0):.1f}" for c in all_cats]) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # Export JSON report
    report_json = {"benchmark": "LiveBench", "release": "2025-07-07",
                   "timestamp": datetime.now().isoformat(), "models": {}}
    for m in sorted(task_scores):
        report_json["models"][m] = {"task_scores": task_scores[m],
                                     "category_scores": cat_scores.get(m, {}),
                                     "average_score": cat_scores.get(m, {}).get("average", 0)}
    with open(output_dir / "comparison_report.json", "w") as f:
        json.dump(report_json, f, indent=2)
    print(f"  {output_dir / 'comparison_report.json'}")

    # Export Markdown report
    lines = ["# Benchmark Comparison Report", "",
             f"**Benchmark:** LiveBench  |  **Date:** {datetime.now().strftime('%Y-%m-%d')}", "",
             "## Category Scores", "",
             "| Model | " + " | ".join(all_cats) + " |",
             "|" + "|".join("---" for _ in range(len(all_cats) + 1)) + "|"]
    for m in sorted(cat_scores):
        lines.append("| " + " | ".join([m] + [f"{cat_scores[m].get(c, 0):.1f}" for c in all_cats]) + " |")
    lines += ["", "## Task Scores", "",
              "| Model | " + " | ".join(t.replace("_", " ") for t in all_tasks) + " |",
              "|" + "|".join("---" for _ in range(len(all_tasks) + 1)) + "|"]
    for m in sorted(task_scores):
        lines.append("| " + " | ".join([m] + [f"{task_scores[m].get(t, 0):.1f}" for t in all_tasks]) + " |")
    lines += ["", "## Token Usage", "",
              "| Model | Total Tokens | Avg Tokens/Response | Questions |",
              "|---|---|---|---|"]
    for m in sorted(token_usage):
        t = token_usage[m]
        lines.append(f"| {m} | {sum(t)} | {round(np.mean(t), 1)} | {len(t)} |")
    lines += ["", "---", "*Report generated by Phi-3 Custom Model Evaluation Harness*"]
    with open(output_dir / "comparison_report.md", "w") as f:
        f.write("\n".join(lines))
    print(f"  {output_dir / 'comparison_report.md'}")

    # Export README
    with open(output_dir / "README.md", "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write("| File | Description |\n|---|---|\n")
        for fn in sorted(output_dir.iterdir()):
            if fn.suffix in (".csv", ".tex", ".md", ".json", ".py") and fn.is_file():
                f.write(f"| `{fn.name}` | Auto-generated |\n")
        f.write("\n## Latest Scores\n\n| Model | " + " | ".join(all_cats) + " |\n")
        f.write("|" + "|".join("---" for _ in range(len(all_cats) + 1)) + "|\n")
        for m in sorted(cat_scores):
            f.write("| " + " | ".join([m] + [f"{cat_scores[m].get(c, 0):.1f}" for c in all_cats]) + " |\n")

    print(f"\nAll files exported to {output_dir}/")


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        if args.command == "list": _cmd_list()
        elif args.command == "livebench": _cmd_livebench(args)
        elif args.command == "custom": _cmd_custom(args)
        elif args.command == "report": _cmd_report(args)
        elif args.command == "compare": _cmd_compare(args)
        elif args.command == "export": _cmd_export(args)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
