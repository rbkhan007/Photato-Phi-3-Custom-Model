#!/usr/bin/env python3
"""
Vector Operations Optimizer for Fast Embeddings.

Features:
- Optimized dot product, cosine similarity, L2/Manhattan distances
- Batch operations
- LSH-based Approximate Nearest Neighbor (ANN)
- IVF (Inverted File Index) with k-means clustering
- Vector quantization
- Result caching

Usage:
    from optimization.vector_ops import VectorOptimizer, ANNIndex, IVFIndex
"""

import math
import random
import time
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VectorIndex:
    vectors: list[list[float]] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    dimension: int = 0


class VectorOptimizer:
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.index = VectorIndex(dimension=dimension)

    def dot_product(self, vec1: list[float], vec2: list[float]) -> float:
        if len(vec1) != len(vec2):
            return 0.0
        result = 0.0
        v1, v2 = vec1, vec2
        n = len(v1)
        for i in range(n):
            result += v1[i] * v2[i]
        return result

    def dot_product_simd(self, vec1: list[float], vec2: list[float]) -> float:
        if len(vec1) != len(vec2):
            return 0.0
        result = 0.0
        n = len(vec1)
        chunk_size = 8
        for i in range(0, n, chunk_size):
            chunk_result = 0.0
            end = min(i + chunk_size, n)
            for j in range(i, end):
                chunk_result += vec1[j] * vec2[j]
            result += chunk_result
        return result

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        dot = 0.0
        n1 = 0.0
        n2 = 0.0
        for a, b in zip(vec1, vec2):
            dot += a * b
            n1 += a * a
            n2 += b * b
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (math.sqrt(n1) * math.sqrt(n2))

    def l2_distance(self, vec1: list[float], vec2: list[float]) -> float:
        if len(vec1) != len(vec2):
            return float('inf')
        result = 0.0
        for i in range(len(vec1)):
            diff = vec1[i] - vec2[i]
            result += diff * diff
        return math.sqrt(result)

    def l2_distance_squared(self, vec1: list[float], vec2: list[float]) -> float:
        if len(vec1) != len(vec2):
            return float('inf')
        result = 0.0
        for i in range(len(vec1)):
            diff = vec1[i] - vec2[i]
            result += diff * diff
        return result

    def manhattan_distance(self, vec1: list[float], vec2: list[float]) -> float:
        if len(vec1) != len(vec2):
            return float('inf')
        result = 0.0
        for i in range(len(vec1)):
            result += abs(vec1[i] - vec2[i])
        return result

    def vector_norm(self, vec: list[float]) -> float:
        result = 0.0
        for v in vec:
            result += v * v
        return math.sqrt(result)

    def normalize(self, vec: list[float]) -> list[float]:
        norm = self.vector_norm(vec)
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    def batch_dot_product(self, query: list[float], vectors: list[list[float]]) -> list[float]:
        return [self.dot_product(query, v) for v in vectors]

    def batch_cosine_similarity(self, query: list[float], vectors: list[list[float]]) -> list[float]:
        qn = self.vector_norm(query)
        if qn == 0:
            return [0.0] * len(vectors)
        results = []
        for v in vectors:
            dot = self.dot_product(query, v)
            vn = self.vector_norm(v)
            sim = dot / (qn * vn) if vn > 0 else 0.0
            results.append(sim)
        return results

    def add_vectors(self, vectors: list[list[float]], ids: list[str]):
        self.index.vectors.extend(vectors)
        self.index.ids.extend(ids)

    def find_similar(
        self,
        query: list[float],
        top_k: int = 5,
        metric: str = "cosine",
    ) -> list[tuple[str, float]]:
        scores = []
        if metric == "cosine":
            for i, vector in enumerate(self.index.vectors):
                scores.append((self.index.ids[i], self.cosine_similarity(query, vector)))
        elif metric == "l2":
            for i, vector in enumerate(self.index.vectors):
                scores.append((self.index.ids[i], -self.l2_distance(query, vector)))
        else:
            for i, vector in enumerate(self.index.vectors):
                scores.append((self.index.ids[i], self.dot_product(query, vector)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def find_similar_batch(
        self,
        queries: list[list[float]],
        top_k: int = 5,
        metric: str = "cosine",
    ) -> list[list[tuple[str, float]]]:
        return [self.find_similar(q, top_k, metric) for q in queries]


class LSHIndex:
    def __init__(self, dimension: int = 128, num_tables: int = 4, num_planes: int = 8):
        self.dimension = dimension
        self.num_tables = num_tables
        self.num_planes = num_planes
        self.tables: list[dict[int, list[tuple[str, list[float]]]]] = [defaultdict(list) for _ in range(num_tables)]
        self._planes = [
            [[random.gauss(0, 1) for _ in range(dimension)] for _ in range(num_planes)]
            for _ in range(num_tables)
        ]
        self._ids_set: set[str] = set()

    def _hash(self, vector: list[float], planes: list[list[float]]) -> int:
        bits = 0
        for i, plane in enumerate(planes):
            dot = sum(v * p for v, p in zip(vector, plane))
            if dot >= 0:
                bits |= (1 << i)
        return bits

    def add(self, id: str, vector: list[float]):
        self._ids_set.add(id)
        for t in range(self.num_tables):
            h = self._hash(vector, self._planes[t])
            self.tables[t][h].append((id, vector))

    def add_batch(self, items: list[tuple[str, list[float]]]):
        for id, vec in items:
            self.add(id, vec)

    def search(
        self,
        query: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        candidates: dict[str, list[float]] = {}
        for t in range(self.num_tables):
            h = self._hash(query, self._planes[t])
            for id, vec in self.tables[t].get(h, []):
                if id not in candidates:
                    candidates[id] = vec
        scores = []
        for id, vec in candidates.items():
            dot = 0.0
            n1 = 0.0
            n2 = 0.0
            for a, b in zip(query, vec):
                dot += a * b
                n1 += a * a
                n2 += b * b
            sim = dot / (math.sqrt(n1) * math.sqrt(n2)) if n1 > 0 and n2 > 0 else 0.0
            if sim >= threshold:
                scores.append((id, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    @property
    def count(self) -> int:
        return len(self._ids_set)


class IVFIndex:
    def __init__(self, dimension: int = 128, num_centroids: int = 16):
        self.dimension = dimension
        self.num_centroids = num_centroids
        self.centroids: list[list[float]] = []
        self.lists: dict[int, list[tuple[str, list[float]]]] = defaultdict(list)
        self._trained = False

    def train(self, vectors: list[list[float]]):
        k = min(self.num_centroids, len(vectors))
        self.centroids = random.sample(vectors, k)
        for _ in range(5):
            assignments = defaultdict(list)
            for vec in vectors:
                best = min(range(len(self.centroids)), key=lambda i: sum((a - b) ** 2 for a, b in zip(vec, self.centroids[i])))
                assignments[best].append(vec)
            for i in range(len(self.centroids)):
                if assignments[i]:
                    self.centroids[i] = [sum(dim) / len(assignments[i]) for dim in zip(*assignments[i])]
        self._trained = True

    def _nearest_centroid(self, vector: list[float]) -> int:
        return min(range(len(self.centroids)), key=lambda i: sum((a - b) ** 2 for a, b in zip(vector, self.centroids[i])))

    def add(self, id: str, vector: list[float]):
        if not self._trained and len(self.centroids) < self.num_centroids:
            self.centroids.append(vector)
        cid = self._nearest_centroid(vector)
        self.lists[cid].append((id, vector))

    def add_batch(self, items: list[tuple[str, list[float]]]):
        if not self._trained and len(items) > self.num_centroids:
            self.train([v for _, v in items])
        for id, vec in items:
            self.add(id, vec)

    def search(
        self,
        query: list[float],
        top_k: int = 5,
        nprobe: int = 2,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        if not self.centroids:
            return []
        dists = [(i, sum((a - b) ** 2 for a, b in zip(query, self.centroids[i]))) for i in range(len(self.centroids))]
        dists.sort(key=lambda x: x[1])
        probe_lists = [self.lists[dists[i][0]] for i in range(min(nprobe, len(dists)))]
        scores = []
        for lst in probe_lists:
            for id, vec in lst:
                dot = 0.0
                n1 = 0.0
                n2 = 0.0
                for a, b in zip(query, vec):
                    dot += a * b
                    n1 += a * a
                    n2 += b * b
                sim = dot / (math.sqrt(n1) * math.sqrt(n2)) if n1 > 0 and n2 > 0 else 0.0
                if sim >= threshold:
                    scores.append((id, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class ANNIndex:
    def __init__(self, dimension: int = 128, num_lists: int = 16):
        self.dimension = dimension
        self.num_lists = num_lists
        self.lists: dict[int, list[tuple[str, list[float]]]] = defaultdict(list)
        self.centroids: list[list[float]] = []

    def _hash_vector(self, vector: list[float]) -> int:
        h = 0
        for v in vector:
            h = (h * 31 + int(v * 1000)) % self.num_lists
        return h

    def add(self, id: str, vector: list[float]):
        list_id = self._hash_vector(vector)
        self.lists[list_id].append((id, vector))

    def search(self, query: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        query_list = self._hash_vector(query)
        candidates = self.lists.get(query_list, [])
        scores = []
        for id, vector in candidates:
            dot = sum(a * b for a, b in zip(query, vector))
            n1 = math.sqrt(sum(a * a for a in query))
            n2 = math.sqrt(sum(b * b for b in vector))
            score = dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0
            scores.append((id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class VectorQuantizer:
    def __init__(self, codebook_size: int = 256):
        self.codebook_size = codebook_size
        self.codebook: list[list[float]] = []

    def train(self, vectors: list[list[float]]):
        self.codebook = random.sample(vectors, min(self.codebook_size, len(vectors)))
        for _ in range(3):
            assignments = defaultdict(list)
            for vec in vectors:
                best = min(range(len(self.codebook)), key=lambda i: sum((a - b) ** 2 for a, b in zip(vec, self.codebook[i])))
                assignments[best].append(vec)
            for i in range(len(self.codebook)):
                if assignments[i]:
                    self.codebook[i] = [sum(dim) / len(assignments[i]) for dim in zip(*assignments[i])]

    def quantize(self, vector: list[float]) -> int:
        return min(range(len(self.codebook)), key=lambda i: sum((a - b) ** 2 for a, b in zip(vector, self.codebook[i])))

    def dequantize(self, code: int) -> list[float]:
        if code < len(self.codebook):
            return self.codebook[code]
        return [0.0] * len(self.codebook[0]) if self.codebook else []


def main(argv=None):
    import argparse

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
        p.add_argument("--a", required=True, help="First vector (JSON array or file)")
        p.add_argument("--b", help="Second vector (JSON array or file)")

    for name in ("dot", "cosine", "l2", "l2-squared", "manhattan"):
        add_pair(sub.add_parser(name))

    for name in ("norm", "normalize"):
        sub.add_parser(name).add_argument("--vector", required=True, help="Vector (JSON array or file)")

    bc = sub.add_parser("batch-cosine")
    bc.add_argument("--query", required=True)
    bc.add_argument("--vectors", required=True)

    fs = sub.add_parser("find-similar")
    fs.add_argument("--query", required=True)
    fs.add_argument("--vectors", required=True)
    fs.add_argument("--ids")
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
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
