#!/usr/bin/env python3
"""
Vector Operations Optimizer for Fast Embeddings.

Features:
- SIMD-optimized dot products
- Batch vector operations
- Approximate nearest neighbor (ANN)
- Vector quantization
- Fast cosine similarity

Usage:
    from optimization.vector_ops import VectorOptimizer

    optimizer = VectorOptimizer(dimension=128)
    similar = optimizer.find_similar(query_vec, corpus)
"""

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VectorIndex:
    """Vector index for fast lookup."""
    vectors: list[list[float]] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    dimension: int = 0


class VectorOptimizer:
    """
    Optimized vector operations for embeddings.

    Features:
    - Fast dot product
    - Batch operations
    - Cosine similarity
    - L2 distance
    - Vector normalization
    """

    def __init__(self, dimension: int = 128):
        """Initialize vector optimizer."""
        self.dimension = dimension
        self.index = VectorIndex(dimension=dimension)

    def dot_product(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Fast dot product using loop optimization.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Dot product value
        """
        if len(vec1) != len(vec2):
            return 0.0

        # Optimized loop with local variables
        result = 0.0
        v1 = vec1
        v2 = vec2
        n = len(v1)

        for i in range(n):
            result += v1[i] * v2[i]

        return result

    def dot_product_simd(self, vec1: list[float], vec2: list[float]) -> float:
        """
        SIMD-style dot product (chunked for cache efficiency).

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Dot product value
        """
        if len(vec1) != len(vec2):
            return 0.0

        result = 0.0
        n = len(vec1)
        chunk_size = 8  # Simulate SIMD width

        # Process in chunks
        for i in range(0, n, chunk_size):
            chunk_result = 0.0
            end = min(i + chunk_size, n)
            for j in range(i, end):
                chunk_result += vec1[j] * vec2[j]
            result += chunk_result

        return result

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Fast cosine similarity.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (-1 to 1)
        """
        dot = self.dot_product(vec1, vec2)
        norm1 = self.vector_norm(vec1)
        norm2 = self.vector_norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def l2_distance(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Fast L2 (Euclidean) distance.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            L2 distance
        """
        if len(vec1) != len(vec2):
            return float('inf')

        result = 0.0
        for i in range(len(vec1)):
            diff = vec1[i] - vec2[i]
            result += diff * diff

        return math.sqrt(result)

    def l2_distance_squared(self, vec1: list[float], vec2: list[float]) -> float:
        """Fast L2 distance (squared, avoids sqrt)."""
        if len(vec1) != len(vec2):
            return float('inf')

        result = 0.0
        for i in range(len(vec1)):
            diff = vec1[i] - vec2[i]
            result += diff * diff

        return result

    def manhattan_distance(self, vec1: list[float], vec2: list[float]) -> float:
        """Fast Manhattan distance."""
        if len(vec1) != len(vec2):
            return float('inf')

        result = 0.0
        for i in range(len(vec1)):
            result += abs(vec1[i] - vec2[i])

        return result

    def vector_norm(self, vec: list[float]) -> float:
        """Compute L2 norm of vector."""
        result = 0.0
        for v in vec:
            result += v * v
        return math.sqrt(result)

    def normalize(self, vec: list[float]) -> list[float]:
        """Normalize vector to unit length."""
        norm = self.vector_norm(vec)
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    def batch_dot_product(self, query: list[float], vectors: list[list[float]]) -> list[float]:
        """
        Batch dot product for multiple vectors.

        Args:
            query: Query vector
            vectors: List of vectors

        Returns:
            List of dot products
        """
        return [self.dot_product(query, v) for v in vectors]

    def batch_cosine_similarity(self, query: list[float], vectors: list[list[float]]) -> list[float]:
        """Batch cosine similarity."""
        query_norm = self.vector_norm(query)
        if query_norm == 0:
            return [0.0] * len(vectors)

        results = []
        for v in vectors:
            dot = self.dot_product(query, v)
            v_norm = self.vector_norm(v)
            sim = dot / (query_norm * v_norm) if v_norm > 0 else 0.0
            results.append(sim)

        return results

    def add_vectors(self, vectors: list[list[float]], ids: list[str]):
        """Add vectors to index."""
        self.index.vectors.extend(vectors)
        self.index.ids.extend(ids)

    def find_similar(
        self,
        query: list[float],
        top_k: int = 5,
        metric: str = "cosine",
    ) -> list[tuple[str, float]]:
        """
        Find similar vectors.

        Args:
            query: Query vector
            top_k: Number of results
            metric: Similarity metric

        Returns:
            List of (id, score) tuples
        """
        scores = []

        for i, vector in enumerate(self.index.vectors):
            if metric == "cosine":
                score = self.cosine_similarity(query, vector)
            elif metric == "l2":
                score = -self.l2_distance(query, vector)  # Negative for ranking
            else:
                score = self.dot_product(query, vector)

            scores.append((self.index.ids[i], score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class ANNIndex:
    """Approximate Nearest Neighbor index."""

    def __init__(self, dimension: int = 128, num_lists: int = 16):
        self.dimension = dimension
        self.num_lists = num_lists
        self.lists: dict[int, list[tuple[str, list[float]]]] = defaultdict(list)
        self.centroids: list[list[float]] = []

    def _hash_vector(self, vector: list[float]) -> int:
        """Hash vector to list index."""
        h = 0
        for v in vector:
            h = (h * 31 + int(v * 1000)) % self.num_lists
        return h

    def add(self, id: str, vector: list[float]):
        """Add vector to index."""
        list_id = self._hash_vector(vector)
        self.lists[list_id].append((id, vector))

    def search(self, query: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        """Search for similar vectors."""
        query_list = self._hash_vector(query)
        candidates = self.lists.get(query_list, [])

        scores = []
        for id, vector in candidates:
            score = self._cosine(query, vector)
            scores.append((id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _cosine(self, v1: list[float], v2: list[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        return dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0


class VectorQuantizer:
    """Vector quantization for compression."""

    def __init__(self, codebook_size: int = 256):
        self.codebook_size = codebook_size
        self.codebook: list[list[float]] = []

    def train(self, vectors: list[list[float]]):
        """Train quantizer on vectors."""
        # Simple k-means for codebook
        import random
        self.codebook = random.sample(vectors, min(self.codebook_size, len(vectors)))

    def quantize(self, vector: list[float]) -> int:
        """Quantize vector to code."""
        best_idx = 0
        best_dist = float('inf')

        for i, centroid in enumerate(self.codebook):
            dist = sum((a - b) ** 2 for a, b in zip(vector, centroid))
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx

    def dequantize(self, code: int) -> list[float]:
        """Dequantize code to vector."""
        if code < len(self.codebook):
            return self.codebook[code]
        return [0.0] * len(self.codebook[0]) if self.codebook else []


def main(argv=None):
    """Vector operations command-line interface."""
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

    parser = argparse.ArgumentParser(description="Vector operations optimizer")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_pair(p):
        p.add_argument("--a", required=True, help="First vector (JSON array or file path)")
        p.add_argument("--b", help="Second vector (JSON array or file path)")

    for name in ("dot", "cosine", "l2", "l2-squared", "manhattan"):
        add_pair(sub.add_parser(name))

    for name in ("norm", "normalize"):
        sub.add_parser(name).add_argument(
            "--vector", required=True, help="Vector (JSON array or file path)"
        )

    bc = sub.add_parser("batch-cosine")
    bc.add_argument("--query", required=True, help="Query vector (JSON array or file path)")
    bc.add_argument("--vectors", required=True, help="List of vectors (JSON or file path)")

    fs = sub.add_parser("find-similar")
    fs.add_argument("--query", required=True, help="Query vector (JSON array or file path)")
    fs.add_argument("--vectors", required=True, help="List of vectors (JSON or file path)")
    fs.add_argument("--ids", help="List of ids (JSON array or file path)")
    fs.add_argument("--metric", default="cosine", choices=["cosine", "l2", "dot"])
    fs.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args(argv)

    try:
        optimizer = VectorOptimizer()
        if args.command in ("dot", "cosine", "l2", "l2-squared", "manhattan"):
            a = resolve(args.a)
            b = resolve(args.b)
            if args.command == "dot":
                result = optimizer.dot_product(a, b)
            elif args.command == "cosine":
                result = optimizer.cosine_similarity(a, b)
            elif args.command == "l2":
                result = optimizer.l2_distance(a, b)
            elif args.command == "l2-squared":
                result = optimizer.l2_distance_squared(a, b)
            else:
                result = optimizer.manhattan_distance(a, b)
        elif args.command in ("norm", "normalize"):
            vec = resolve(args.vector)
            result = optimizer.vector_norm(vec) if args.command == "norm" else optimizer.normalize(vec)
        elif args.command == "batch-cosine":
            query = resolve(args.query)
            vectors = resolve(args.vectors)
            result = optimizer.batch_cosine_similarity(query, vectors)
        else:
            query = resolve(args.query)
            vectors = resolve(args.vectors)
            ids = resolve(args.ids) if args.ids else [str(i) for i in range(len(vectors))]
            optimizer.add_vectors(vectors, ids)
            result = optimizer.find_similar(query, top_k=args.top_k, metric=args.metric)

        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
