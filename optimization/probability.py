#!/usr/bin/env python3
"""
Probability Sampling Optimizer for Local LLMs.

Features:
- Top-k sampling
- Top-p (nucleus) sampling
- Temperature scaling
- Repetition penalty
- Frequency/presence penalty
- Min-p sampling
- Typical sampling

Usage:
    from optimization.probability import ProbabilitySampler

    sampler = ProbabilitySampler()
    token = sampler.sample(logits, temperature=0.7)
"""

import math
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingConfig:
    """Sampling configuration."""
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None


class ProbabilitySampler:
    """
    Optimized probability sampling for token generation.

    Features:
    - Multiple sampling strategies
    - Fast loop implementations
    - Batch sampling
    - Deterministic sampling with seeds
    """

    def __init__(self, config: Optional[SamplingConfig] = None):
        """Initialize sampler."""
        self.config = config or SamplingConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)

    def sample(
        self,
        logits: dict[str, float],
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Sample token from logits.

        Args:
            logits: Token logits
            temperature: Temperature override
            top_k: Top-k override
            top_p: Top-p override

        Returns:
            Sampled token
        """
        probs = self.logits_to_probs(logits)

        # Apply temperature
        temp = temperature if temperature is not None else self.config.temperature
        probs = self.apply_temperature(probs, temp)

        # Apply top-k
        k = top_k if top_k is not None else self.config.top_k
        probs = self.apply_top_k(probs, k)

        # Apply top-p
        p = top_p if top_p is not None else self.config.top_p
        probs = self.apply_top_p(probs, p)

        # Sample
        return self.sample_from_probs(probs)

    def logits_to_probs(self, logits: dict[str, float]) -> dict[str, float]:
        """Convert logits to probabilities via softmax."""
        if not logits:
            return {}

        max_logit = max(logits.values())
        exp_logits = {}

        for token, logit in logits.items():
            exp_logits[token] = math.exp(logit - max_logit)

        total = sum(exp_logits.values())
        return {t: e / total for t, e in exp_logits.items()}

    def apply_temperature(self, probs: dict[str, float], temperature: float) -> dict[str, float]:
        """Apply temperature scaling."""
        if temperature <= 0:
            # Greedy
            max_token = max(probs, key=probs.get)
            return {max_token: 1.0}

        if temperature == 1.0:
            return probs

        scaled = {}
        for token, prob in probs.items():
            if prob > 0:
                scaled[token] = math.exp(math.log(prob) / temperature)
            else:
                scaled[token] = 0.0

        total = sum(scaled.values())
        if total > 0:
            return {t: p / total for t, p in scaled.items()}
        return probs

    def apply_top_k(self, probs: dict[str, float], k: int) -> dict[str, float]:
        """Apply top-k filtering."""
        if k <= 0 or k >= len(probs):
            return probs

        sorted_tokens = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_k_tokens = sorted_tokens[:k]

        # Zero out non-top-k
        result = {}
        for token, prob in top_k_tokens:
            result[token] = prob

        # Renormalize
        total = sum(result.values())
        if total > 0:
            return {t: p / total for t, p in result.items()}
        return result

    def apply_top_p(self, probs: dict[str, float], p: float) -> dict[str, float]:
        """Apply nucleus (top-p) sampling."""
        if p >= 1.0:
            return probs

        sorted_tokens = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        cumulative = 0.0
        nucleus = []

        for token, prob in sorted_tokens:
            cumulative += prob
            nucleus.append((token, prob))
            if cumulative >= p:
                break

        # Renormalize
        total = sum(prob for _, prob in nucleus)
        if total > 0:
            return {t: prob / total for t, prob in nucleus}
        return probs

    def apply_min_p(self, probs: dict[str, float], min_p: float) -> dict[str, float]:
        """Apply min-p filtering."""
        if min_p <= 0:
            return probs

        max_prob = max(probs.values()) if probs else 0
        threshold = max_prob * min_p

        filtered = {t: p for t, p in probs.items() if p >= threshold}

        total = sum(filtered.values())
        if total > 0:
            return {t: p / total for t, p in filtered.items()}
        return probs

    def apply_repeat_penalty(
        self,
        probs: dict[str, float],
        generated_tokens: list[str],
        penalty: float,
    ) -> dict[str, float]:
        """Apply repetition penalty."""
        if penalty <= 1.0 or not generated_tokens:
            return probs

        penalized = {}
        for token, prob in probs.items():
            if token in generated_tokens:
                penalized[token] = prob / penalty
            else:
                penalized[token] = prob

        total = sum(penalized.values())
        if total > 0:
            return {t: p / total for t, p in penalized.items()}
        return probs

    def apply_frequency_penalty(
        self,
        probs: dict[str, float],
        token_counts: dict[str, int],
        penalty: float,
    ) -> dict[str, float]:
        """Apply frequency penalty."""
        if penalty <= 0:
            return probs

        penalized = {}
        for token, prob in probs.items():
            count = token_counts.get(token, 0)
            penalized[token] = prob - (penalty * count)

        # Clamp negative probabilities
        penalized = {t: max(0, p) for t, p in penalized.items()}

        total = sum(penalized.values())
        if total > 0:
            return {t: p / total for t, p in penalized.items()}
        return probs

    def apply_presence_penalty(
        self,
        probs: dict[str, float],
        generated_tokens: list[str],
        penalty: float,
    ) -> dict[str, float]:
        """Apply presence penalty."""
        if penalty <= 0:
            return probs

        penalized = {}
        for token, prob in probs.items():
            if token in generated_tokens:
                penalized[token] = prob - penalty
            else:
                penalized[token] = prob

        penalized = {t: max(0, p) for t, p in penalized.items()}

        total = sum(penalized.values())
        if total > 0:
            return {t: p / total for t, p in penalized.items()}
        return probs

    def sample_from_probs(self, probs: dict[str, float]) -> str:
        """Sample token from probabilities."""
        if not probs:
            return ""

        tokens = list(probs.keys())
        weights = list(probs.values())

        return random.choices(tokens, weights=weights, k=1)[0]

    def batch_sample(
        self,
        batch_logits: list[dict[str, float]],
        temperature: float = 1.0,
    ) -> list[str]:
        """Batch sample from multiple logit distributions."""
        return [
            self.sample(logits, temperature=temperature)
            for logits in batch_logits
        ]

    def get_top_tokens(
        self,
        probs: dict[str, float],
        n: int = 5,
    ) -> list[tuple[str, float]]:
        """Get top-n tokens with probabilities."""
        sorted_tokens = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_tokens[:n]


class TypicalSampler:
    """Typical sampling implementation."""

    @staticmethod
    def sample(
        probs: dict[str, float],
        typical_p: float = 0.9,
    ) -> str:
        """
        Typical sampling.

        Args:
            probs: Token probabilities
            typical_p: Typical probability threshold

        Returns:
            Sampled token
        """
        if not probs:
            return ""

        # Calculate entropy
        entropy = -sum(p * math.log(p + 1e-10) for p in probs.values() if p > 0)

        # Calculate information content
        info_contents = {}
        for token, prob in probs.items():
            if prob > 0:
                info = -math.log(prob)
                info_contents[token] = abs(info - entropy)

        # Sort by information content (closest to entropy first)
        sorted_tokens = sorted(info_contents.items(), key=lambda x: x[1])

        # Select typical tokens
        cumulative = 0.0
        typical_tokens = []
        for token, _ in sorted_tokens:
            cumulative += probs[token]
            typical_tokens.append((token, probs[token]))
            if cumulative >= typical_p:
                break

        # Renormalize
        total = sum(p for _, p in typical_tokens)
        if total > 0:
            probs = {t: p / total for t, p in typical_tokens}

        # Sample
        tokens = list(probs.keys())
        weights = list(probs.values())
        return random.choices(tokens, weights=weights, k=1)[0]


class MirostatSampler:
    """Mirostat sampling for perplexity control."""

    def __init__(self, tau: float = 5.0, eta: float = 0.1):
        self.tau = tau
        self.eta = eta
        self.mu = tau * 2  # Initial mu

    def sample(self, probs: dict[str, float]) -> str:
        """Sample with perplexity control."""
        if not probs:
            return ""

        # Sort by probability
        sorted_tokens = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        # Find cutoff based on current mu
        cumulative = 0.0
        for i, (token, prob) in enumerate(sorted_tokens):
            cumulative += prob
            if cumulative >= self._cdf(self.mu, i + 1):
                # Sample from top tokens
                top_tokens = sorted_tokens[:i + 1]
                tokens = [t for t, _ in top_tokens]
                weights = [p for _, p in top_tokens]
                total = sum(weights)
                weights = [w / total for w in weights]

                selected = random.choices(tokens, weights=weights, k=1)[0]

                # Update mu
                self._update_mu(selected, probs)

                return selected

        # Fallback
        return sorted_tokens[0][0]

    def _cdf(self, mu: float, k: int) -> float:
        """CDF for mirostat."""
        return 1 - math.exp(-mu / k)

    def _update_mu(self, selected: str, probs: dict[str, float]):
        """Update mu based on selected token."""
        if selected in probs and probs[selected] > 0:
            surprisal = -math.log(probs[selected])
            self.mu += self.eta * (self.tau - surprisal)
            self.mu = max(0, self.mu)


def main(argv=None):
    """Probability sampler command-line interface."""
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

    parser = argparse.ArgumentParser(description="Probability sampling optimizer")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("sample")
    sp.add_argument("--logits", required=True, help="Token logits as JSON dict or file path")
    sp.add_argument("--temperature", type=float, default=None)
    sp.add_argument("--top-k", type=int, default=None)
    sp.add_argument("--top-p", type=float, default=None)
    sp.add_argument("--seed", type=int, default=None)

    tp = sub.add_parser("top")
    tp.add_argument("--logits", required=True, help="Token logits as JSON dict or file path")
    tp.add_argument("--n", type=int, default=5)

    mp = sub.add_parser("minp")
    mp.add_argument("--logits", required=True, help="Token logits as JSON dict or file path")
    mp.add_argument("--min-p", type=float, default=0.05)

    ts = sub.add_parser("typical")
    ts.add_argument("--logits", required=True, help="Token logits as JSON dict or file path")
    ts.add_argument("--typical-p", type=float, default=0.9)

    args = parser.parse_args(argv)

    try:
        if args.command == "sample":
            logits = resolve(args.logits)
            if not isinstance(logits, dict):
                raise ValueError("logits must be a JSON object mapping token->logit")
            cfg = SamplingConfig(
                temperature=args.temperature if args.temperature is not None else 0.7,
                top_k=args.top_k if args.top_k is not None else 40,
                top_p=args.top_p if args.top_p is not None else 0.9,
                seed=args.seed,
            )
            sampler = ProbabilitySampler(config=cfg)
            result = sampler.sample(logits, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        elif args.command == "top":
            logits = resolve(args.logits)
            sampler = ProbabilitySampler()
            probs = sampler.logits_to_probs(logits)
            result = sampler.get_top_tokens(probs, n=args.n)
        elif args.command == "minp":
            logits = resolve(args.logits)
            sampler = ProbabilitySampler()
            probs = sampler.logits_to_probs(logits)
            result = sampler.apply_min_p(probs, args.min_p)
        else:
            logits = resolve(args.logits)
            sampler = ProbabilitySampler()
            probs = sampler.logits_to_probs(logits)
            result = TypicalSampler.sample(probs, typical_p=args.typical_p)

        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
