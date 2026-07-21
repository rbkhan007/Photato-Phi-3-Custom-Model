#!/usr/bin/env python3
"""
Attention Optimization for Local LLMs.

Features:
- Flash Attention simulation
- Multi-head attention optimization
- Sparse attention patterns
- Attention caching
- Grouped Query Attention (GQA)

Usage:
    from optimization.attention import AttentionOptimizer

    optimizer = AttentionOptimizer(num_heads=8)
    output = optimizer.attend(query, key, value)
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class AttentionConfig:
    """Attention configuration."""
    num_heads: int = 8
    head_dim: int = 64
    max_sequence_length: int = 4096
    use_flash_attention: bool = True
    use_gqa: bool = False  # Grouped Query Attention
    num_kv_heads: int = 2  # For GQA
    dropout: float = 0.0


class AttentionOptimizer:
    """
    Optimized attention mechanisms.

    Features:
    - Flash Attention simulation
    - Multi-head attention
    - KV-cache support
    - Memory-efficient attention
    """

    def __init__(self, config: Optional[AttentionConfig] = None):
        """Initialize attention optimizer."""
        self.config = config or AttentionConfig()
        self.kv_cache: dict[str, list] = {}

    def multi_head_attention(
        self,
        query: list[list[float]],
        key: list[list[float]],
        value: list[list[float]],
        mask: Optional[list[list[bool]]] = None,
    ) -> list[list[float]]:
        """
        Multi-head attention with optimization.

        Args:
            query: Query tensor [seq_len, dim]
            key: Key tensor [seq_len, dim]
            value: Value tensor [seq_len, dim]
            mask: Attention mask

        Returns:
            Attention output
        """
        seq_len = len(query)
        head_dim = self.config.head_dim
        num_heads = self.config.num_heads

        # Reshape to heads
        query_heads = self._reshape_to_heads(query, num_heads, head_dim)
        key_heads = self._reshape_to_heads(key, num_heads, head_dim)
        value_heads = self._reshape_to_heads(value, num_heads, head_dim)

        # Compute attention for each head
        output_heads = []
        for i in range(num_heads):
            if self.config.use_flash_attention:
                head_output = self._flash_attention(
                    query_heads[i], key_heads[i], value_heads[i], mask
                )
            else:
                head_output = self._standard_attention(
                    query_heads[i], key_heads[i], value_heads[i], mask
                )
            output_heads.append(head_output)

        # Concatenate heads
        output = self._concatenate_heads(output_heads)

        return output

    def _standard_attention(
        self,
        query: list[list[float]],
        key: list[list[float]],
        value: list[list[float]],
        mask: Optional[list[list[bool]]] = None,
    ) -> list[list[float]]:
        """Standard scaled dot-product attention."""
        seq_len = len(query)
        head_dim = len(query[0]) if query else 0

        # Compute attention scores
        scores = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                # Dot product
                dot = sum(query[i][k] * key[j][k] for k in range(head_dim))
                # Scale
                score = dot / math.sqrt(head_dim)
                # Apply mask
                if mask and not mask[i][j]:
                    score = float('-inf')
                row.append(score)
            scores.append(row)

        # Softmax
        probs = self._softmax_2d(scores)

        # Apply to values
        output = []
        for i in range(seq_len):
            row = [0.0] * head_dim
            for j in range(seq_len):
                for k in range(head_dim):
                    row[k] += probs[i][j] * value[j][k]
            output.append(row)

        return output

    def _flash_attention(
        self,
        query: list[list[float]],
        key: list[list[float]],
        value: list[list[float]],
        mask: Optional[list[list[bool]]] = None,
    ) -> list[list[float]]:
        """
        Flash Attention simulation (memory-efficient).

        Processes attention in blocks to reduce memory usage.
        """
        seq_len = len(query)
        head_dim = len(query[0]) if query else 0
        block_size = 64  # Flash attention block size

        # Initialize output
        output = [[0.0] * head_dim for _ in range(seq_len)]
        max_scores = [float('-inf')] * seq_len
        sum_exp = [0.0] * seq_len

        # Process in blocks
        for j_start in range(0, seq_len, block_size):
            j_end = min(j_start + block_size, seq_len)

            # Compute attention scores for block
            for i in range(seq_len):
                for j in range(j_start, j_end):
                    # Dot product
                    dot = sum(query[i][k] * key[j][k] for k in range(head_dim))
                    score = dot / math.sqrt(head_dim)

                    # Apply mask
                    if mask and not mask[i][j]:
                        score = float('-inf')

                    # Online softmax update
                    if score > max_scores[i]:
                        # Rescale existing sum
                        scale = math.exp(max_scores[i] - score)
                        sum_exp[i] *= scale
                        for k in range(head_dim):
                            output[i][k] *= scale
                        max_scores[i] = score

                    # Add new contribution
                    exp_score = math.exp(score - max_scores[i])
                    sum_exp[i] += exp_score
                    for k in range(head_dim):
                        output[i][k] += exp_score * value[j][k]

        # Normalize output
        for i in range(seq_len):
            if sum_exp[i] > 0:
                for k in range(head_dim):
                    output[i][k] /= sum_exp[i]

        return output

    def _reshape_to_heads(
        self,
        tensor: list[list[float]],
        num_heads: int,
        head_dim: int,
    ) -> list[list[list[float]]]:
        """Reshape tensor to multiple heads."""
        heads = [[] for _ in range(num_heads)]

        for vec in tensor:
            for h in range(num_heads):
                start = h * head_dim
                end = start + head_dim
                heads[h].append(vec[start:end])

        return heads

    def _concatenate_heads(self, heads: list[list[list[float]]]) -> list[list[float]]:
        """Concatenate heads back to single tensor."""
        if not heads:
            return []

        num_heads = len(heads)
        seq_len = len(heads[0])
        head_dim = len(heads[0][0]) if heads[0] else 0

        output = []
        for i in range(seq_len):
            vec = []
            for h in range(num_heads):
                vec.extend(heads[h][i])
            output.append(vec)

        return output

    def _softmax_2d(self, scores: list[list[float]]) -> list[list[float]]:
        """Apply softmax to 2D tensor."""
        result = []
        for row in scores:
            max_val = max(row)
            exp_row = [math.exp(s - max_val) for s in row]
            total = sum(exp_row)
            result.append([e / total for e in exp_row])
        return result

    def sparse_attention(
        self,
        query: list[list[float]],
        key: list[list[float]],
        value: list[list[float]],
        pattern: str = "local",
        window_size: int = 256,
    ) -> list[list[float]]:
        """
        Sparse attention with different patterns.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            pattern: Attention pattern (local, stride, random)
            window_size: Window size for local attention

        Returns:
            Attention output
        """
        seq_len = len(query)

        if pattern == "local":
            mask = self._create_local_mask(seq_len, window_size)
        elif pattern == "stride":
            mask = self._create_stride_mask(seq_len, window_size)
        else:
            mask = None

        return self._standard_attention(query, key, value, mask)

    def _create_local_mask(self, seq_len: int, window_size: int) -> list[list[bool]]:
        """Create local attention mask."""
        mask = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                row.append(abs(i - j) <= window_size)
            mask.append(row)
        return mask

    def _create_stride_mask(self, seq_len: int, stride: int) -> list[list[bool]]:
        """Create stride attention mask."""
        mask = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                row.append((i - j) % stride == 0)
            mask.append(row)
        return mask

    def update_kv_cache(
        self,
        layer: int,
        key: list[list[float]],
        value: list[list[float]],
    ):
        """Update KV cache for layer."""
        cache_key = f"layer_{layer}"
        if cache_key not in self.kv_cache:
            self.kv_cache[cache_key] = {"keys": [], "values": []}

        self.kv_cache[cache_key]["keys"].extend(key)
        self.kv_cache[cache_key]["values"].extend(value)

        # Trim to max length
        max_len = self.config.max_sequence_length
        if len(self.kv_cache[cache_key]["keys"]) > max_len:
            self.kv_cache[cache_key]["keys"] = self.kv_cache[cache_key]["keys"][-max_len:]
            self.kv_cache[cache_key]["values"] = self.kv_cache[cache_key]["values"][-max_len:]

    def get_kv_cache(self, layer: int) -> dict:
        """Get KV cache for layer."""
        cache_key = f"layer_{layer}"
        return self.kv_cache.get(cache_key, {"keys": [], "values": []})

    def clear_kv_cache(self):
        """Clear all KV caches."""
        self.kv_cache.clear()


class GroupedQueryAttention:
    """Grouped Query Attention (GQA) implementation."""

    def __init__(self, num_heads: int = 8, num_kv_heads: int = 2, head_dim: int = 64):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = num_heads // num_kv_heads

    def forward(
        self,
        query: list[list[float]],
        key: list[list[float]],
        value: list[list[float]],
    ) -> list[list[float]]:
        """GQA forward pass."""
        seq_len = len(query)

        # Expand KV heads to match Q heads
        expanded_key = self._expand_kv(key)
        expanded_value = self._expand_kv(value)

        # Compute attention scores
        scores = []
        for i in range(seq_len):
            row_scores = []
            for j in range(seq_len):
                dot = sum(query[i][k] * expanded_key[j][k] for k in range(self.head_dim))
                row_scores.append(dot / math.sqrt(self.head_dim))
            scores.append(row_scores)

        # Apply softmax to each row
        attn_weights = []
        for row_scores in scores:
            max_score = max(row_scores)
            exp_scores = [math.exp(s - max_score) for s in row_scores]
            sum_exp = sum(exp_scores)
            attn_weights.append([s / sum_exp for s in exp_scores])

        # Weighted sum of values
        output = []
        for i in range(seq_len):
            row = [0.0] * self.head_dim
            for j in range(seq_len):
                for k in range(self.head_dim):
                    row[k] += attn_weights[i][j] * expanded_value[j][k]
            output.append(row)

        return output

    def _expand_kv(self, kv: list[list[float]]) -> list[list[float]]:
        """Expand KV to match number of query heads."""
        expanded = []
        for vec in kv:
            expanded_vec = []
            for _ in range(self.num_queries_per_kv):
                expanded_vec.extend(vec)
            expanded.append(expanded_vec)
        return expanded


def main(argv=None):
    """Attention optimizer command-line interface."""
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

    parser = argparse.ArgumentParser(description="Attention optimization for local LLMs")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_qkv(p):
        p.add_argument("--query", required=True, help="Query tensor (JSON matrix or file path)")
        p.add_argument("--key", required=True, help="Key tensor (JSON matrix or file path)")
        p.add_argument("--value", required=True, help="Value tensor (JSON matrix or file path)")

    mha = sub.add_parser("mha")
    add_qkv(mha)
    mha.add_argument("--num-heads", type=int, default=8)
    mha.add_argument("--head-dim", type=int, default=64)
    mha.add_argument("--flash", action="store_true", help="Use flash attention")

    sp = sub.add_parser("sparse")
    add_qkv(sp)
    sp.add_argument("--pattern", default="local", choices=["local", "stride"])
    sp.add_argument("--window-size", type=int, default=256)

    gqa = sub.add_parser("gqa")
    add_qkv(gqa)
    gqa.add_argument("--num-heads", type=int, default=8)
    gqa.add_argument("--num-kv-heads", type=int, default=2)
    gqa.add_argument("--head-dim", type=int, default=64)

    args = parser.parse_args(argv)

    try:
        q = resolve(args.query)
        k = resolve(args.key)
        v = resolve(args.value)
        if args.command == "mha":
            config = AttentionConfig(num_heads=args.num_heads, head_dim=args.head_dim, use_flash_attention=args.flash)
            result = AttentionOptimizer(config).multi_head_attention(q, k, v)
        elif args.command == "sparse":
            result = AttentionOptimizer().sparse_attention(q, k, v, pattern=args.pattern, window_size=args.window_size)
        else:
            result = GroupedQueryAttention(
                num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, head_dim=args.head_dim
            ).forward(q, k, v)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
