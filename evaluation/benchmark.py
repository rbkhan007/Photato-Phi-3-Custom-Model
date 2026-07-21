#!/usr/bin/env python3
"""
Benchmarking Suite for Local LLMs.

Comprehensive performance benchmarking for inference speed, memory, and quality.

Usage:
    from evaluation.benchmark import BenchmarkSuite

    suite = BenchmarkSuite(model_path="./phi3-mini-q4_k_m.gguf")
    results = suite.run_full_benchmark()
"""

import argparse
import gc
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BenchmarkMetrics:
    """Benchmark performance metrics."""
    tokens_per_second: float = 0.0
    latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    first_token_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    batch_efficiency: float = 0.0


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    prompt: str
    response: str
    metrics: BenchmarkMetrics
    tokens_generated: int = 0
    prompt_tokens: int = 0
    total_time: float = 0.0
    success: bool = True
    error: Optional[str] = None


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for local LLMs.

    Features:
    - Inference speed benchmark
    - Memory usage profiling
    - Latency measurement
    - Throughput testing
    - Batch processing efficiency
    - Multi-sequence benchmark
    - Quality metrics
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str = "./benchmark_results",
        host: str = "http://localhost:11434",
        model: Optional[str] = None,
    ):
        """
        Initialize benchmark suite.

        Args:
            model_path: Path to model (used for reporting / model-name resolution)
            output_dir: Output directory for results
            host: Ollama server base URL (default http://localhost:11434)
            model: Explicit Ollama model name; if omitted, resolved from /api/tags
        """
        self.model_path = model_path
        self.host = host.rstrip("/")
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[BenchmarkResult] = []

    def benchmark_inference_speed(
        self,
        prompts: Optional[list[str]] = None,
        max_tokens: int = 100,
        num_runs: int = 3,
    ) -> list[BenchmarkResult]:
        """
        Benchmark inference speed.

        Args:
            prompts: List of prompts to test
            max_tokens: Maximum tokens to generate
            num_runs: Number of runs per prompt

        Returns:
            List of BenchmarkResult
        """
        if prompts is None:
            prompts = [
                "The quick brown fox jumps over the lazy dog.",
                "Write a short story about a robot.",
                "Explain quantum computing in simple terms.",
                "What are the benefits of exercise?",
                "Write a Python function to sort a list.",
            ]

        print(f"\nBenchmarking inference speed ({len(prompts)} prompts, {num_runs} runs each)")
        results = []

        for i, prompt in enumerate(prompts):
            print(f"  Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")

            run_results = []
            for run in range(num_runs):
                result = self._run_single_benchmark(prompt, max_tokens)
                run_results.append(result)

            # Average results
            avg_result = self._average_results(run_results)
            results.append(avg_result)

            print(f"    Speed: {avg_result.metrics.tokens_per_second:.1f} tokens/s")
            print(f"    Latency: {avg_result.metrics.latency_ms:.1f}ms")

        self.results.extend(results)
        return results

    def benchmark_memory_usage(
        self,
        prompts: Optional[list[str]] = None,
        max_tokens: int = 50,
    ) -> dict:
        """
        Benchmark memory usage.

        Args:
            prompts: List of prompts to test
            max_tokens: Maximum tokens to generate

        Returns:
            Memory usage metrics
        """
        print("\nBenchmarking memory usage...")

        memory_stats = {
            "initial_mb": 0,
            "peak_mb": 0,
            "final_mb": 0,
            "growth_mb": 0,
        }

        try:
            import psutil
            process = psutil.Process(os.getpid())

            # Initial memory
            gc.collect()
            if hasattr(psutil, "VIRTUAL_MEMORY"):
                memory_stats["initial_mb"] = process.memory_info().rss / 1024 / 1024

            # Run inference
            if prompts is None:
                prompts = ["Hello, how are you?"] * 5

            for prompt in prompts:
                self._run_single_benchmark(prompt, max_tokens)

                # Check memory
                current_mb = process.memory_info().rss / 1024 / 1024
                memory_stats["peak_mb"] = max(memory_stats["peak_mb"], current_mb)

            # Final memory
            gc.collect()
            memory_stats["final_mb"] = process.memory_info().rss / 1024 / 1024
            memory_stats["growth_mb"] = memory_stats["final_mb"] - memory_stats["initial_mb"]

        except ImportError:
            print("  psutil not available, skipping memory benchmark")

        # GPU memory if available
        try:
            import torch
            if torch.cuda.is_available():
                memory_stats["gpu_initial_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                memory_stats["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
                memory_stats["gpu_final_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        except ImportError:
            pass

        print(f"  Initial: {memory_stats['initial_mb']:.1f} MB")
        print(f"  Peak: {memory_stats['peak_mb']:.1f} MB")
        print(f"  Growth: {memory_stats['growth_mb']:.1f} MB")

        return memory_stats

    def benchmark_latency(
        self,
        prompts: Optional[list[str]] = None,
        max_tokens: int = 10,
        num_runs: int = 10,
    ) -> dict:
        """
        Benchmark latency (time to first token).

        Args:
            prompts: List of prompts to test
            max_tokens: Maximum tokens to generate
            num_runs: Number of runs

        Returns:
            Latency metrics
        """
        print("\nBenchmarking latency...")

        if prompts is None:
            prompts = ["Hello"] * num_runs

        latencies = []

        for prompt in prompts:
            for _ in range(num_runs):
                start_time = time.time()
                self._run_single_benchmark(prompt, max_tokens)
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)

        latency_stats = {
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        }

        print(f"  Mean: {latency_stats['mean_ms']:.1f}ms")
        print(f"  Median: {latency_stats['median_ms']:.1f}ms")
        print(f"  P95: {latency_stats['p95_ms']:.1f}ms")
        print(f"  P99: {latency_stats['p99_ms']:.1f}ms")

        return latency_stats

    def benchmark_throughput(
        self,
        prompts: Optional[list[str]] = None,
        max_tokens: int = 100,
        batch_sizes: Optional[list[int]] = None,
    ) -> dict:
        """
        Benchmark throughput with different batch sizes.

        Args:
            prompts: List of prompts to test
            max_tokens: Maximum tokens to generate
            batch_sizes: List of batch sizes to test

        Returns:
            Throughput metrics
        """
        print("\nBenchmarking throughput...")

        if prompts is None:
            prompts = ["Hello, how are you?"] * 10

        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]

        throughput_results = {}

        for batch_size in batch_sizes:
            start_time = time.time()
            total_tokens = 0

            # Process in batches
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                for prompt in batch:
                    result = self._run_single_benchmark(prompt, max_tokens)
                    total_tokens += result.tokens_generated

            total_time = time.time() - start_time
            tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

            throughput_results[batch_size] = {
                "tokens_per_second": tokens_per_sec,
                "total_time": total_time,
                "total_tokens": total_tokens,
            }

            print(f"  Batch size {batch_size}: {tokens_per_sec:.1f} tokens/s")

        return throughput_results

    def benchmark_context_length(
        self,
        context_lengths: Optional[list[int]] = None,
        max_tokens: int = 50,
    ) -> dict:
        """
        Benchmark performance at different context lengths.

        Args:
            context_lengths: List of context lengths to test
            max_tokens: Maximum tokens to generate

        Returns:
            Context length performance metrics
        """
        print("\nBenchmarking context length performance...")

        if context_lengths is None:
            context_lengths = [128, 256, 512, 1024, 2048]

        context_results = {}

        for ctx_len in context_lengths:
            # Create prompt of specified length
            prompt = "Hello " * (ctx_len // 6)

            start_time = time.time()
            result = self._run_single_benchmark(prompt, max_tokens)
            total_time = time.time() - start_time

            context_results[ctx_len] = {
                "tokens_per_second": result.metrics.tokens_per_second,
                "latency_ms": total_time * 1000,
                "success": result.success,
            }

            print(f"  Context {ctx_len}: {result.metrics.tokens_per_second:.1f} tokens/s")

        return context_results

    def run_full_benchmark(self) -> dict:
        """
        Run complete benchmark suite.

        Returns:
            Complete benchmark results
        """
        print(f"\n{'='*60}")
        print(f"Full Benchmark Suite")
        print(f"Model: {self.model_path}")
        print(f"{'='*60}")

        results = {
            "model_path": self.model_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Run all benchmarks
        results["inference_speed"] = self._format_speed_results(
            self.benchmark_inference_speed()
        )
        results["memory"] = self.benchmark_memory_usage()
        results["latency"] = self.benchmark_latency()
        results["throughput"] = self.benchmark_throughput()
        results["context_length"] = self.benchmark_context_length()

        # Save results
        self._save_results(results)

        # Print summary
        self._print_summary(results)

        return results

    def _resolve_model(self) -> str:
        """Resolve the Ollama model name to benchmark against."""
        if self.model:
            return self.model
        import json
        import urllib.request

        names: list[str] = []
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as resp:
                names = [m.get("name", "") for m in json.loads(resp.read()).get("models", [])]
        except Exception:
            names = []
        if not names:
            raise RuntimeError(
                "No Ollama models available at "
                f"{self.host}/api/tags. Start Ollama and create a model first."
            )
        stem = Path(self.model_path).stem.lower()
        for n in names:
            nl = n.lower()
            if stem in nl or nl in stem:
                return n
        print(f"  [warn] model '{stem}' not found in Ollama; using '{names[0]}'")
        return names[0]

    def _get_engine(self):
        """Lazily build (and cache) the fast in-process llama.cpp engine."""
        if getattr(self, "_engine", None) is None:
            from inference.llama_engine import FastLlamaEngine

            self._engine = FastLlamaEngine(self.model_path)
        return self._engine

    def _run_single_benchmark(self, prompt: str, max_tokens: int) -> BenchmarkResult:
        """Run a single benchmark against the real model.

        Uses the in-process FastLlamaEngine (llama.cpp) for true local
        timing — no HTTP/JSON overhead — with CPU optimizations
        (mmap, mlock, physical-core threads) and task-aware sampling.
        """
        engine = self._get_engine()
        out = engine.generate(prompt, max_tokens=max_tokens)

        return BenchmarkResult(
            prompt=prompt,
            response=out["text"].strip(),
            metrics=BenchmarkMetrics(
                tokens_per_second=out["tokens_per_second"],
                latency_ms=out["elapsed"] * 1000.0,
                first_token_latency_ms=out["first_token_ms"],
                throughput_tokens_per_sec=out["tokens_per_second"],
            ),
            tokens_generated=out["completion_tokens"],
            prompt_tokens=out["prompt_tokens"],
            total_time=out["elapsed"],
        )

    def _average_results(self, results: list[BenchmarkResult]) -> BenchmarkResult:
        """Average multiple benchmark results."""
        if not results:
            return BenchmarkResult(prompt="", response="", metrics=BenchmarkMetrics())

        avg_metrics = BenchmarkMetrics(
            tokens_per_second=statistics.mean(r.metrics.tokens_per_second for r in results),
            latency_ms=statistics.mean(r.metrics.latency_ms for r in results),
            memory_usage_mb=max(r.metrics.memory_usage_mb for r in results),
            first_token_latency_ms=statistics.mean(r.metrics.first_token_latency_ms for r in results),
        )

        return BenchmarkResult(
            prompt=results[0].prompt,
            response=results[0].response,
            metrics=avg_metrics,
            tokens_generated=results[0].tokens_generated,
            total_time=statistics.mean(r.total_time for r in results),
        )

    def _format_speed_results(self, results: list[BenchmarkResult]) -> dict:
        """Format speed results."""
        return {
            "prompts_tested": len(results),
            "avg_tokens_per_second": statistics.mean(r.metrics.tokens_per_second for r in results),
            "avg_latency_ms": statistics.mean(r.metrics.latency_ms for r in results),
            "min_tokens_per_second": min(r.metrics.tokens_per_second for r in results),
            "max_tokens_per_second": max(r.metrics.tokens_per_second for r in results),
        }

    def _save_results(self, results: dict):
        """Save benchmark results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def _print_summary(self, results: dict):
        """Print benchmark summary."""
        print(f"\n{'='*60}")
        print(f"Benchmark Summary")
        print(f"{'='*60}")

        if "inference_speed" in results:
            speed = results["inference_speed"]
            print(f"\nInference Speed:")
            print(f"  Average: {speed['avg_tokens_per_second']:.1f} tokens/s")
            print(f"  Range: {speed['min_tokens_per_second']:.1f} - {speed['max_tokens_per_second']:.1f} tokens/s")

        if "latency" in results:
            latency = results["latency"]
            print(f"\nLatency:")
            print(f"  Mean: {latency['mean_ms']:.1f}ms")
            print(f"  P95: {latency['p95_ms']:.1f}ms")

        if "memory" in results:
            memory = results["memory"]
            print(f"\nMemory:")
            print(f"  Peak: {memory['peak_mb']:.1f} MB")


def main(argv=None):
    """Run the benchmark suite against a real model path."""
    parser = argparse.ArgumentParser(
        description="Benchmark suite for local LLMs"
    )
    parser.add_argument("--model", required=True, help="Path to model (for reporting / name resolution)")
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Ollama model name to benchmark (defaults to resolved name)",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama server base URL",
    )
    parser.add_argument(
        "--output-dir",
        default="./benchmark_results",
        help="Directory to write results",
    )
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--num-runs", type=int, default=3, help="Runs per prompt")
    parser.add_argument(
        "--prompts",
        help="Path to a JSON file containing a list of prompts",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output aggregated results as JSON",
    )
    args = parser.parse_args(argv)

    try:
        suite = BenchmarkSuite(
            model_path=args.model,
            output_dir=args.output_dir,
            host=args.host,
            model=args.ollama_model,
        )

        prompts = None
        if args.prompts:
            with open(args.prompts) as f:
                prompts = json.load(f)

        suite.benchmark_inference_speed(
            prompts=prompts, max_tokens=args.max_tokens, num_runs=args.num_runs
        )
        memory = suite.benchmark_memory_usage(prompts=prompts, max_tokens=args.max_tokens)
        latency = suite.benchmark_latency(prompts=prompts, max_tokens=args.max_tokens)
        throughput = suite.benchmark_throughput(prompts=prompts, max_tokens=args.max_tokens)
        context_length = suite.benchmark_context_length(max_tokens=args.max_tokens)

        results = {
            "model_path": args.model,
            "memory": memory,
            "latency": latency,
            "throughput": throughput,
            "context_length": context_length,
        }

        if args.json:
            print(json.dumps(results, indent=2, default=str))

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
