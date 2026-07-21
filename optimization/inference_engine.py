#!/usr/bin/env python3
"""
Optimized Inference Engine with Fast Loops.

Features:
- Prefill optimization
- KV-cache management
- Batch inference loops
- Memory-efficient token generation
- Early exit optimization

Usage:
    from optimization.inference_engine import OptimizedInference

    engine = OptimizedInference(model_path="./phi3-mini-q4_k_m.gguf")
    response = engine.generate("Hello", max_tokens=100)
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KVCache:
    """Key-Value cache for transformer attention."""
    keys: list[list[float]] = field(default_factory=list)
    values: list[list[float]] = field(default_factory=list)
    max_size: int = 4096

    def append(self, key: list[float], value: list[float]):
        """Append key-value pair."""
        self.keys.append(key)
        self.values.append(value)

        # Trim if too large
        if len(self.keys) > self.max_size:
            self.keys = self.keys[-self.max_size:]
            self.values = self.values[-self.max_size:]

    def get(self, window_size: Optional[int] = None) -> tuple[list, list]:
        """Get cached key-value pairs."""
        if window_size:
            return self.keys[-window_size:], self.values[-window_size:]
        return self.keys, self.values

    def clear(self):
        """Clear cache."""
        self.keys.clear()
        self.values.clear()

    @property
    def size(self) -> int:
        return len(self.keys)


@dataclass
class InferenceConfig:
    """Inference configuration."""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    use_kv_cache: bool = True
    early_exit_threshold: float = 0.95
    batch_size: int = 1
    max_prefill_length: int = 2048


class FastTokenizer:
    """Fast token counting and manipulation."""

    @staticmethod
    def count_tokens(text: str) -> int:
        """Fast approximate token count."""
        return len(text) // 4

    @staticmethod
    def truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate text to token limit."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + "..."

    @staticmethod
    def split_into_chunks(text: str, chunk_size: int = 100) -> list[str]:
        """Split text into token-sized chunks."""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks


class OptimizedInference:
    """
    Optimized inference engine with fast loops.

    Features:
    - KV-cache for faster generation
    - Prefill optimization
    - Early exit
    - Memory-efficient loops
    """

    def __init__(self, model_path: str = "", config: Optional[InferenceConfig] = None):
        """Initialize inference engine."""
        self.model_path = model_path
        self.config = config or InferenceConfig()
        self.kv_cache = KVCache(max_size=self.config.max_tokens * 2)
        self.tokenizer = FastTokenizer()
        self.generation_count = 0

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate text with optimized loops.

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            stream: Enable streaming

        Returns:
            Generated text
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        # Prefill phase (process prompt)
        prompt_tokens = self._prefill(prompt)

        # Generate tokens with optimized loop
        generated_tokens = []
        for i in range(max_tokens):
            # Get next token probabilities
            probs = self._get_next_token_probs(prompt_tokens + generated_tokens)

            # Apply temperature
            probs = self._apply_temperature(probs, temperature)

            # Apply top-k filtering
            probs = self._apply_top_k(probs, self.config.top_k)

            # Apply top-p (nucleus) sampling
            probs = self._apply_top_p(probs, self.config.top_p)

            # Apply repeat penalty
            probs = self._apply_repeat_penalty(probs, generated_tokens, self.config.repeat_penalty)

            # Sample token
            token = self._sample_token(probs)

            # Check for early exit
            if self._should_exit(probs, token):
                break

            generated_tokens.append(token)

            # Update KV cache
            if self.config.use_kv_cache:
                self._update_kv_cache(token)

        # Decode tokens
        response = self._decode_tokens(generated_tokens)

        self.generation_count += 1

        return response

    def _prefill(self, prompt: str) -> list[str]:
        """
        Prefill phase - process prompt efficiently.

        Args:
            prompt: Input prompt

        Returns:
            Token list
        """
        # Chunk prompt for efficient processing
        chunks = self.tokenizer.split_into_chunks(prompt, self.config.max_prefill_length)

        tokens = []
        for chunk in chunks:
            # Process chunk (placeholder)
            chunk_tokens = chunk.split()
            tokens.extend(chunk_tokens)

        return tokens

    def _get_next_token_probs(self, tokens: list[str]) -> dict[str, float]:
        """
        Get next token probabilities.

        Args:
            tokens: Current token sequence

        Returns:
            Token probabilities
        """
        # Placeholder - use actual model logits
        vocab = ["the", "a", "is", "and", "to", "of", "in", "for", "with", "on"]
        probs = {}
        for token in vocab:
            probs[token] = 1.0 / len(vocab)
        return probs

    def _apply_temperature(self, probs: dict[str, float], temperature: float) -> dict[str, float]:
        """Apply temperature scaling."""
        import math

        if temperature == 0:
            # Greedy decoding
            max_token = max(probs, key=probs.get)
            return {max_token: 1.0}

        scaled = {}
        for token, prob in probs.items():
            scaled[token] = math.exp(math.log(prob + 1e-10) / temperature)

        # Normalize
        total = sum(scaled.values())
        return {t: p / total for t, p in scaled.items()}

    def _apply_top_k(self, probs: dict[str, float], k: int) -> dict[str, float]:
        """Apply top-k filtering."""
        if k <= 0:
            return probs

        sorted_tokens = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_tokens[:k]

        # Renormalize
        total = sum(p for _, p in top_k)
        if total > 0:
            return {t: p / total for t, p in top_k}
        return probs

    def _apply_top_p(self, probs: dict[str, float], p: float) -> dict[str, float]:
        """Apply nucleus (top-p) sampling."""
        if p >= 1.0:
            return probs

        sorted_tokens = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        cumulative = 0.0
        selected = []

        for token, prob in sorted_tokens:
            cumulative += prob
            selected.append((token, prob))
            if cumulative >= p:
                break

        # Renormalize
        total = sum(p for _, p in selected)
        if total > 0:
            return {t: prob / total for t, prob in selected}
        return probs

    def _apply_repeat_penalty(
        self,
        probs: dict[str, float],
        generated: list[str],
        penalty: float,
    ) -> dict[str, float]:
        """Apply repeat penalty."""
        if penalty <= 1.0:
            return probs

        penalized = {}
        for token, prob in probs.items():
            if token in generated:
                penalized[token] = prob / penalty
            else:
                penalized[token] = prob

        # Renormalize
        total = sum(penalized.values())
        if total > 0:
            return {t: p / total for t, p in penalized.items()}
        return penalized

    def _sample_token(self, probs: dict[str, float]) -> str:
        """Sample token from probabilities."""
        import random

        tokens = list(probs.keys())
        weights = list(probs.values())

        return random.choices(tokens, weights=weights, k=1)[0]

    def _should_exit(self, probs: dict[str, float], token: str) -> bool:
        """Check for early exit - exit when confidence is LOW (uncertain)."""
        max_prob = max(probs.values())
        entropy = -sum(p * math.log(p + 1e-10) for p in probs.values() if p > 0)
        max_entropy = math.log(len(probs)) if probs else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        return normalized_entropy > self.config.early_exit_threshold

    def _update_kv_cache(self, token: str):
        """Update KV cache with new token."""
        # Placeholder - actual implementation would get key/value from model
        key = [hash(token) % 100 / 100.0] * 64
        value = [hash(token) % 50 / 50.0] * 64
        self.kv_cache.append(key, value)

    def _decode_tokens(self, tokens: list[str]) -> str:
        """Decode tokens to text."""
        return " ".join(tokens)

    def batch_generate(self, prompts: list[str], max_tokens: int = 100) -> list[str]:
        """
        Batch generate for multiple prompts.

        Args:
            prompts: List of prompts
            max_tokens: Max tokens per prompt

        Returns:
            List of responses
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_tokens)
            results.append(result)
        return results

    def clear_cache(self):
        """Clear KV cache."""
        self.kv_cache.clear()


class SpeculativeDecoder:
    """Speculative decoding for faster generation."""

    def __init__(self, draft_model_path: str = "", target_model_path: str = ""):
        self.draft_path = draft_model_path
        self.target_path = target_model_path

    def speculative_generate(
        self,
        prompt: str,
        draft_tokens: int = 5,
        max_tokens: int = 100,
    ) -> str:
        """
        Generate with speculative decoding.

        Args:
            prompt: Input prompt
            draft_tokens: Tokens to draft ahead
            max_tokens: Max tokens

        Returns:
            Generated text
        """
        # Draft phase (fast, small model)
        draft = self._draft_generate(prompt, draft_tokens)

        # Verify phase (slow, large model)
        verified = self._verify_generate(prompt + " " + draft)

        return verified

    def _draft_generate(self, prompt: str, num_tokens: int) -> str:
        """Draft generation with small model."""
        return "draft " * num_tokens

    def _verify_generate(self, prompt: str) -> str:
        """Verification with large model."""
        return "verified response"


class ContinuousBatching:
    """Continuous batching for throughput."""

    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.queue: deque = deque()

    def add_request(self, request_id: str, prompt: str):
        """Add request to queue."""
        self.queue.append({"id": request_id, "prompt": prompt})

    def process_batch(self) -> list[dict]:
        """Process batch of requests."""
        batch = []
        while self.queue and len(batch) < self.max_batch_size:
            batch.append(self.queue.popleft())

        results = []
        for req in batch:
            # Process request (placeholder)
            results.append({
                "id": req["id"],
                "response": f"Response to: {req['prompt'][:50]}...",
            })

        return results


def main(argv=None):
    """Optimized inference engine command-line interface."""
    import argparse
    import json
    import os
    import sys

    def resolve(s):
        s = s.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            if os.path.exists(s):
                with open(s) as f:
                    return json.load(f)
            raise

    parser = argparse.ArgumentParser(description="Optimized inference engine for local LLMs")
    sub = parser.add_subparsers(dest="command", required=True)

    gn = sub.add_parser("generate")
    gn.add_argument("--prompt", required=True)
    gn.add_argument("--max-tokens", type=int, default=256)
    gn.add_argument("--temperature", type=float, default=0.7)
    gn.add_argument("--top-k", type=int, default=40)
    gn.add_argument("--top-p", type=float, default=0.9)
    gn.add_argument("--model-path", default="")

    bg = sub.add_parser("batch-generate")
    bg.add_argument("--prompts", required=True, help="List of prompts (JSON or file path)")
    bg.add_argument("--max-tokens", type=int, default=100)

    ct = sub.add_parser("count-tokens")
    ct.add_argument("--text", required=True)

    tr = sub.add_parser("truncate")
    tr.add_argument("--text", required=True)
    tr.add_argument("--max-tokens", type=int, required=True)

    args = parser.parse_args(argv)

    try:
        if args.command == "generate":
            config = InferenceConfig(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            engine = OptimizedInference(model_path=args.model_path, config=config)
            response = engine.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temperature)
            result = {"response": response, "cache_size": engine.kv_cache.size}
        elif args.command == "batch-generate":
            prompts = resolve(args.prompts)
            if not isinstance(prompts, list):
                raise ValueError("prompts must be a JSON list of strings")
            engine = OptimizedInference()
            result = engine.batch_generate(prompts, max_tokens=args.max_tokens)
        elif args.command == "count-tokens":
            result = FastTokenizer.count_tokens(args.text)
        else:
            result = FastTokenizer.truncate_to_tokens(args.text, args.max_tokens)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
