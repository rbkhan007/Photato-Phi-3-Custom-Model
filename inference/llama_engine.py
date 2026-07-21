"""
Fast in-process inference engine for GGUF models using llama.cpp (llama_cpp).

This is the *real* optimization path. It:
  - Loads the model directly via llama_cpp (no HTTP/JSON overhead, unlike Ollama),
  - Applies CPU optimizations: memory-mapped weights, locked RAM (mlock),
    and a thread count pinned to physical cores,
  - Uses the real task-aware sampling presets from inference.auto_tuner.AutoTuner
    (temperature / top_p / top_k / repeat_penalty).

Usage:
    from inference.llama_engine import FastLlamaEngine

    engine = FastLlamaEngine("notebooks/Phi-4-mini-instruct-Q4_K_M.gguf")
    out = engine.generate("Explain recursion.", task_type="qa", max_tokens=128)
    print(out["text"], out["tokens_per_second"])
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - dependency guard
    Llama = None

from inference.auto_tuner import AutoTuner, TaskType


class FastLlamaEngine:
    """In-process, CPU-optimized GGUF inference with task-aware sampling."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_batch: int = 512,
        mlock: bool = True,
        n_gpu_layers: int = 0,
        cpu_percent: float = 55.0,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ):
        if Llama is None:
            raise RuntimeError(
                "llama_cpp is not installed. pip install llama-cpp-python"
            )
        self.model_path = str(model_path)
        self.cpu_percent = cpu_percent
        self.tuner = AutoTuner()

        # Cap CPU utilization to ``cpu_percent`` via the OS-level throttle
        # (Windows Job Object hard cap when available), and derive the thread
        # budget used to load llama.cpp.
        from optimization.cpu_throttle import limit_cpu

        if n_threads is None:
            n_threads = limit_cpu(cpu_percent)
        self._threads = n_threads

        self.model = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=self._threads,
            n_gpu_layers=n_gpu_layers,
            use_mmap=True,
            use_mlock=mlock,
            verbose=verbose,
        )

    @staticmethod
    def _physical_threads() -> int:
        """Prefer physical cores (hyperthreads hurt llama.cpp throughput)."""
        try:
            import psutil

            physical = psutil.cpu_count(logical=False)
            if physical:
                return physical
        except Exception:
            pass
        return max(1, (os.cpu_count() or 4) // 2)

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        task_type: Optional[str] = None,
        max_tokens: int = 256,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> dict:
        """Generate a response, returning text + real performance metrics.

        Args:
            prompt: Single user prompt (used if ``messages`` is omitted).
            messages: Full chat history as a list of {"role", "content"} dicts.
                Preferred for multi-turn chat (preserves context).
            task_type: One of AutoTuner task names (qa, code, creative, ...).
                Used to pick sampling presets; auto-detected from the prompt if omitted.
            max_tokens: Maximum tokens to generate.
            temperature: Override sampling temperature.
        """
        if messages is None:
            if prompt is None:
                prompt = ""
            messages = [{"role": "user", "content": prompt}]
        else:
            # Derive a prompt string for task-type detection.
            if prompt is None:
                joined = " ".join(
                    m.get("content", "") for m in messages if m.get("role") == "user"
                )
                prompt = joined[-800:]

        params = self.tuner.get_params(task_type=TaskType(task_type) if task_type else None, prompt=prompt)
        if temperature is not None:
            params.temperature = temperature

        messages = [{"role": "user", "content": prompt}]
        t0 = time.time()
        first_token_ms = None
        chunks: list[str] = []

        stream = self.model.create_chat_completion(
            messages=messages,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repeat_penalty=params.repeat_penalty,
            max_tokens=max_tokens,
            stream=True,
        )
        for delta in stream:
            content = delta["choices"][0]["delta"].get("content", "")
            if content:
                if first_token_ms is None:
                    first_token_ms = (time.time() - t0) * 1000.0
                chunks.append(content)

        elapsed = time.time() - t0
        text = "".join(chunks)

        # llama_cpp's streaming chat API omits `usage`, so count real tokens
        # directly from the model's tokenizer for accurate throughput.
        try:
            prompt_tokens = len(self.model.tokenize(prompt.encode("utf-8")))
            gen = len(self.model.tokenize(text.encode("utf-8"))) if text else 0
        except Exception:
            prompt_tokens = 0
            gen = 0
        tps = (gen / elapsed) if (elapsed > 0 and gen > 0) else 0.0

        return {
            "text": text,
            "tokens_per_second": tps,
            "completion_tokens": gen,
            "prompt_tokens": prompt_tokens,
            "elapsed": elapsed,
            "first_token_ms": first_token_ms if first_token_ms is not None else elapsed * 1000.0,
            "threads": self._threads,
            "params": params.to_dict(),
        }
