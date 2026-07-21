#!/usr/bin/env python3
"""
Evaluation Harness for Local LLMs.

Integrated with lm-evaluation-harness for comprehensive model evaluation.

Usage:
    from evaluation.harness import EvaluationHarness

    harness = EvaluationHarness(model_path="./phi3-mini-q4_k_m.gguf")
    results = harness.run_all_benchmarks()
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class BenchmarkResult:
    benchmark: str
    score: float
    metric: str
    details: dict = field(default_factory=dict)
    timestamp: str = ""
    model_path: str = ""
    execution_time: float = 0


@dataclass
class EvaluationConfig:
    model_path: str
    model_type: str = "llama"  # llama, gpt2, custom
    device: str = "auto"
    batch_size: int = 1
    num_fewshot: int = 0
    max_gen_toks: int = 256
    output_dir: str = "./evaluation_results"
    benchmarks: list[str] = field(default_factory=lambda: [
        "mmlu",
        "hellaswag",
        "arc_challenge",
        "winogrande",
        "truthfulqa",
        "gsm8k",
        "humaneval",
    ])


class EvaluationHarness:
    """
    Comprehensive evaluation harness for local LLMs.

    Features:
    - Integration with lm-evaluation-harness
    - Multiple benchmark support
    - Custom benchmark creation
    - Result comparison and reporting
    - Automated evaluation pipelines
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize evaluation harness.

        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig(model_path="")
        self.results: list[BenchmarkResult] = []
        self._check_dependencies()

    def _check_dependencies(self):
        """Check for required dependencies."""
        try:
            import lm_eval
            print("lm-evaluation-harness available")
        except ImportError:
            print("Installing lm-evaluation-harness...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "lm_eval"],
                check=True,
            )

    def run_benchmark(
        self,
        benchmark: str,
        num_fewshot: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Run a single benchmark.

        Args:
            benchmark: Benchmark name
            num_fewshot: Number of few-shot examples
            limit: Limit number of examples

        Returns:
            BenchmarkResult
        """
        print(f"\nRunning benchmark: {benchmark}")
        start_time = time.time()

        try:
            import lm_eval
            from lm_eval import evaluator, tasks

            # Get task
            task_dict = tasks.get_task_dict([benchmark])

            # Run evaluation
            results = evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={self.config.model_path}",
                tasks=[benchmark],
                num_fewshot=num_fewshot or self.config.num_fewshot,
                batch_size=self.config.batch_size,
                limit=limit,
            )

            # Extract score
            score = 0.0
            metric = "accuracy"

            if "results" in results:
                for task_name, task_results in results["results"].items():
                    for metric_name, metric_value in task_results.items():
                        if isinstance(metric_value, (int, float)):
                            score = metric_value
                            metric = metric_name
                            break

            execution_time = time.time() - start_time

            result = BenchmarkResult(
                benchmark=benchmark,
                score=score,
                metric=metric,
                details=results.get("results", {}),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                model_path=self.config.model_path,
                execution_time=execution_time,
            )

            self.results.append(result)
            print(f"  {benchmark}: {score:.4f} ({metric}) [{execution_time:.1f}s]")

            return result

        except Exception as e:
            print(f"Error running {benchmark}: {e}")
            return BenchmarkResult(
                benchmark=benchmark,
                score=0.0,
                metric="error",
                details={"error": str(e)},
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                model_path=self.config.model_path,
            )

    def run_all_benchmarks(
        self,
        benchmarks: Optional[list[str]] = None,
        num_fewshot: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[BenchmarkResult]:
        """
        Run all specified benchmarks.

        Args:
            benchmarks: List of benchmark names
            num_fewshot: Number of few-shot examples
            limit: Limit number of examples

        Returns:
            List of BenchmarkResult
        """
        benchmarks = benchmarks or self.config.benchmarks
        results = []

        print(f"\n{'='*60}")
        print(f"Running {len(benchmarks)} benchmarks")
        print(f"{'='*60}")

        for benchmark in benchmarks:
            result = self.run_benchmark(benchmark, num_fewshot, limit)
            results.append(result)

        self._save_results(results)
        self._print_summary(results)

        return results

    def run_custom_eval(
        self,
        test_cases: list[dict],
        eval_fn: callable,
        name: str = "custom_eval",
    ) -> BenchmarkResult:
        """
        Run custom evaluation.

        Args:
            test_cases: List of test cases
            eval_fn: Evaluation function
            name: Evaluation name

        Returns:
            BenchmarkResult
        """
        print(f"\nRunning custom evaluation: {name}")
        start_time = time.time()

        scores = []
        details = {"passed": 0, "failed": 0, "errors": 0}

        for i, test_case in enumerate(test_cases):
            try:
                result = eval_fn(test_case)
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
        execution_time = time.time() - start_time

        result = BenchmarkResult(
            benchmark=name,
            score=score,
            metric="accuracy",
            details=details,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            model_path=self.config.model_path,
            execution_time=execution_time,
        )

        self.results.append(result)
        print(f"  {name}: {score:.4f} [{execution_time:.1f}s]")

        return result

    def compare_models(
        self,
        model_paths: list[str],
        benchmarks: Optional[list[str]] = None,
    ) -> dict:
        """
        Compare multiple models.

        Args:
            model_paths: List of model paths
            benchmarks: Benchmarks to run

        Returns:
            Comparison results
        """
        comparisons = {}

        for model_path in model_paths:
            config = EvaluationConfig(
                model_path=model_path,
                benchmarks=benchmarks or self.config.benchmarks,
            )
            harness = EvaluationHarness(config)
            results = harness.run_all_benchmarks()

            comparisons[model_path] = {
                r.benchmark: {"score": r.score, "metric": r.metric}
                for r in results
            }

        self._print_comparison(comparisons)
        return comparisons

    def _save_results(self, results: list[BenchmarkResult]):
        """Save results to file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"eval_results_{timestamp}.json"

        data = {
            "model_path": self.config.model_path,
            "timestamp": timestamp,
            "results": [
                {
                    "benchmark": r.benchmark,
                    "score": r.score,
                    "metric": r.metric,
                    "execution_time": r.execution_time,
                    "details": r.details,
                }
                for r in results
            ],
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def _print_summary(self, results: list[BenchmarkResult]):
        """Print summary of results."""
        print(f"\n{'='*60}")
        print(f"Evaluation Summary")
        print(f"{'='*60}")
        print(f"Model: {self.config.model_path}")
        print(f"Benchmarks: {len(results)}")
        print(f"\n{'Benchmark':<25} {'Score':>10} {'Metric':<15} {'Time':>10}")
        print("-" * 60)

        for r in results:
            print(f"{r.benchmark:<25} {r.score:>10.4f} {r.metric:<15} {r.execution_time:>9.1f}s")

        avg_score = sum(r.score for r in results) / len(results) if results else 0
        total_time = sum(r.execution_time for r in results)
        print("-" * 60)
        print(f"{'Average':<25} {avg_score:>10.4f} {'':15} {total_time:>9.1f}s")

    def _print_comparison(self, comparisons: dict):
        """Print model comparison."""
        print(f"\n{'='*80}")
        print(f"Model Comparison")
        print(f"{'='*80}")

        # Get all benchmarks
        all_benchmarks = set()
        for model_results in comparisons.values():
            all_benchmarks.update(model_results.keys())

        # Print header
        header = f"{'Benchmark':<25}"
        for model_path in comparisons:
            model_name = Path(model_path).stem[:15]
            header += f" {model_name:>15}"
        print(header)
        print("-" * len(header))

        # Print results
        for benchmark in sorted(all_benchmarks):
            row = f"{benchmark:<25}"
            for model_path in comparisons:
                score = comparisons[model_path].get(benchmark, {}).get("score", 0)
                row += f" {score:>15.4f}"
            print(row)


class CustomBenchmarks:
    """Custom benchmark creation utilities."""

    @staticmethod
    def create_code_benchmark() -> list[dict]:
        """Create code generation benchmark."""
        return [
            {
                "prompt": "Write a Python function to calculate fibonacci number",
                "expected": "def fibonacci(n):",
                "type": "contains",
            },
            {
                "prompt": "Write a Python function to check if a number is prime",
                "expected": "def is_prime(n):",
                "type": "contains",
            },
            {
                "prompt": "Write a Python function to reverse a string",
                "expected": "def reverse_string(s):",
                "type": "contains",
            },
        ]

    @staticmethod
    def create_qa_benchmark() -> list[dict]:
        """Create Q&A benchmark."""
        return [
            {
                "prompt": "What is the capital of France?",
                "expected": "Paris",
                "type": "contains",
            },
            {
                "prompt": "What is 2 + 2?",
                "expected": "4",
                "type": "contains",
            },
            {
                "prompt": "Who wrote Romeo and Juliet?",
                "expected": "Shakespeare",
                "type": "contains",
            },
        ]

    @staticmethod
    def create_instruction_benchmark() -> list[dict]:
        """Create instruction following benchmark."""
        return [
            {
                "prompt": "List 3 programming languages",
                "expected_count": 3,
                "type": "count",
            },
            {
                "prompt": "Write a haiku about coding",
                "expected_lines": 3,
                "type": "line_count",
            },
        ]


def _eval_result_to_dict(result):
    """Serialize a BenchmarkResult to a JSON-friendly dict."""
    return {
        "benchmark": result.benchmark,
        "score": result.score,
        "metric": result.metric,
        "execution_time": result.execution_time,
        "timestamp": result.timestamp,
        "model_path": result.model_path,
        "details": result.details,
    }


def _simple_eval_fn(test_case):
    """Built-in evaluation: substring / equality checks from the test case."""
    expected = test_case.get("expected")
    expected_type = test_case.get("type", "contains")
    text = test_case.get("response", test_case.get("prompt", ""))

    if expected_type == "contains":
        return expected.lower() in text.lower()
    if expected_type == "equals":
        return str(expected).lower() == text.strip().lower()
    if expected_type == "count":
        return len(text.split("\n")) >= int(test_case.get("expected_count", 1))
    if expected_type == "line_count":
        return len(text.split("\n")) >= int(test_case.get("expected_lines", 1))
    return bool(text.strip())


def main(argv=None):
    """Run evaluation harness operations from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluation harness for local LLMs"
    )
    parser.add_argument(
        "--config",
        help="Path to a JSON file with EvaluationConfig fields",
    )
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List default benchmarks and custom sets")
    p_run = sub.add_parser("run", help="Run benchmarks for a model")
    p_run.add_argument("--model-path", required=True, help="Path to model")
    p_run.add_argument(
        "--benchmarks",
        nargs="*",
        help="Benchmarks to run (defaults to configured set)",
    )
    p_run.add_argument("--num-fewshot", type=int, default=0)
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--output-dir", default="./evaluation_results")

    p_custom = sub.add_parser("custom", help="Run a custom evaluation from a JSON file")
    p_custom.add_argument("--model-path", required=True, help="Path to model")
    p_custom.add_argument(
        "--tests",
        required=True,
        help="JSON file: list of {prompt, response, expected, type, ...}",
    )
    p_custom.add_argument("--name", default="custom_eval")
    p_custom.add_argument("--output-dir", default="./evaluation_results")

    args = parser.parse_args(argv)

    try:
        if args.command == "list" or args.command is None:
            info = {
                "default_benchmarks": EvaluationConfig(model_path="").benchmarks,
                "custom_benchmarks": {
                    "code": CustomBenchmarks.create_code_benchmark(),
                    "qa": CustomBenchmarks.create_qa_benchmark(),
                    "instruction": CustomBenchmarks.create_instruction_benchmark(),
                },
            }
            print(json.dumps(info, indent=2, default=str))
            return 0

        if args.command == "run":
            config_fields = {}
            if args.config:
                with open(args.config) as f:
                    config_fields = json.load(f)
            config = EvaluationConfig(
                model_path=args.model_path,
                output_dir=args.output_dir,
                num_fewshot=args.num_fewshot,
                **{k: v for k, v in config_fields.items() if k not in ("model_path", "output_dir")},
            )
            harness = EvaluationHarness(config)
            results = harness.run_all_benchmarks(
                benchmarks=args.benchmarks or None,
                num_fewshot=args.num_fewshot,
                limit=args.limit,
            )
            print(json.dumps([_eval_result_to_dict(r) for r in results], indent=2, default=str))
            return 0

        if args.command == "custom":
            with open(args.tests) as f:
                test_cases = json.load(f)
            config = EvaluationConfig(model_path=args.model_path, output_dir=args.output_dir)
            harness = EvaluationHarness(config)
            result = harness.run_custom_eval(test_cases, _simple_eval_fn, args.name)
            print(json.dumps(_eval_result_to_dict(result), indent=2, default=str))
            return 0

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
