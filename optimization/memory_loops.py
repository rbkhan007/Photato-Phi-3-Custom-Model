#!/usr/bin/env python3
"""
Memory-Efficient Loops for Local LLMs.

Features:
- Chunked processing
- Generator-based iteration
- Memory-mapped operations
- Lazy evaluation
- Buffer recycling

Usage:
    from optimization.memory_loops import MemoryEfficientLoop

    loop = MemoryEfficientLoop(chunk_size=1000)
    for chunk in loop.process(large_dataset):
        process(chunk)
"""

import sys
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional


@dataclass
class LoopConfig:
    """Loop configuration."""
    chunk_size: int = 1000
    buffer_size: int = 10
    max_memory_mb: float = 4000
    use_generators: bool = True
    recycle_buffers: bool = True


class BufferPool:
    """Pool for recycling buffers."""

    def __init__(self, buffer_size: int = 10):
        self.buffer_size = buffer_size
        self.available: deque = deque(maxlen=buffer_size)
        self.in_use: set = set()

    def acquire(self, size: int) -> list:
        """Acquire a buffer."""
        if self.available:
            buf = self.available.popleft()
            if len(buf) >= size:
                self.in_use.add(id(buf))
                return buf[:size]

        # Create new buffer
        buf = [None] * size
        self.in_use.add(id(buf))
        return buf

    def release(self, buffer: list):
        """Release a buffer back to pool."""
        buf_id = id(buffer)
        if buf_id in self.in_use:
            self.in_use.remove(buf_id)
            if len(self.available) < self.buffer_size:
                self.available.append(buffer)


class MemoryEfficientLoop:
    """
    Memory-efficient loop implementations.

    Features:
    - Chunked processing
    - Generator-based iteration
    - Buffer recycling
    - Lazy evaluation
    """

    def __init__(self, config: Optional[LoopConfig] = None):
        """Initialize memory-efficient loop."""
        self.config = config or LoopConfig()
        self.buffer_pool = BufferPool(self.config.buffer_size)

    def chunked_iter(self, data: list, chunk_size: Optional[int] = None) -> Generator:
        """
        Iterate over data in chunks.

        Args:
            data: Input data
            chunk_size: Chunk size

        Yields:
            Data chunks
        """
        chunk_size = chunk_size or self.config.chunk_size
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

    def map_reduce(
        self,
        data: list,
        map_fn: Callable,
        reduce_fn: Callable,
        chunk_size: Optional[int] = None,
    ) -> Any:
        """
        Map-reduce with memory efficiency.

        Args:
            data: Input data
            map_fn: Map function
            reduce_fn: Reduce function
            chunk_size: Chunk size

        Returns:
            Reduced result
        """
        chunk_size = chunk_size or self.config.chunk_size
        results = []

        for chunk in self.chunked_iter(data, chunk_size):
            mapped = [map_fn(item) for item in chunk]
            results.extend(mapped)

        return reduce_fn(results)

    def parallel_map(
        self,
        data: list,
        map_fn: Callable,
        num_workers: int = 4,
    ) -> list:
        """
        Parallel map with chunking.

        Args:
            data: Input data
            map_fn: Map function
            num_workers: Number of workers

        Returns:
            Mapped results
        """
        # Split data into chunks
        chunks = list(self.chunked_iter(data, len(data) // num_workers + 1))

        # Process chunks (sequential simulation)
        results = []
        for chunk in chunks:
            chunk_results = [map_fn(item) for item in chunk]
            results.extend(chunk_results)

        return results

    def sliding_window(
        self,
        data: list,
        window_size: int,
        step: int = 1,
    ) -> Generator:
        """
        Sliding window iteration.

        Args:
            data: Input data
            window_size: Window size
            step: Step size

        Yields:
            Window slices
        """
        for i in range(0, len(data) - window_size + 1, step):
            yield data[i:i + window_size]

    def batch_process(
        self,
        data: list,
        process_fn: Callable,
        batch_size: int = 32,
    ) -> list:
        """
        Process data in batches.

        Args:
            data: Input data
            process_fn: Processing function
            batch_size: Batch size

        Returns:
            Processed results
        """
        results = []
        for chunk in self.chunked_iter(data, batch_size):
            batch_result = process_fn(chunk)
            results.append(batch_result)
        return results

    def lazy_filter(
        self,
        data: list,
        filter_fn: Callable,
    ) -> Generator:
        """
        Lazy filter evaluation.

        Args:
            data: Input data
            filter_fn: Filter function

        Yields:
            Filtered items
        """
        for item in data:
            if filter_fn(item):
                yield item

    def lazy_transform(
        self,
        data: list,
        transform_fn: Callable,
    ) -> Generator:
        """
        Lazy transform evaluation.

        Args:
            data: Input data
            transform_fn: Transform function

        Yields:
            Transformed items
        """
        for item in data:
            yield transform_fn(item)

    def chain_generators(self, *generators) -> Generator:
        """
        Chain multiple generators.

        Args:
            *generators: Generators to chain

        Yields:
            All items from all generators
        """
        for gen in generators:
            yield from gen

    def zip_generators(self, *generators) -> Generator:
        """
        Zip multiple generators.

        Args:
            *generators: Generators to zip

        Yields:
            Tuples from zipped generators
        """
        yield from zip(*generators)

    def take(self, generator: Generator, n: int) -> list:
        """
        Take first n items from generator.

        Args:
            generator: Input generator
            n: Number of items

        Returns:
            List of items
        """
        result = []
        for i, item in enumerate(generator):
            if i >= n:
                break
            result.append(item)
        return result

    def consume(self, generator: Generator) -> int:
        """
        Consume entire generator.

        Args:
            generator: Input generator

        Returns:
            Number of items consumed
        """
        count = 0
        for _ in generator:
            count += 1
        return count


class MemoryMappedLoop:
    """Memory-mapped file processing."""

    @staticmethod
    def process_large_file(
        file_path: str,
        process_fn: Callable,
        chunk_size: int = 8192,
    ) -> Generator:
        """
        Process large file in chunks.

        Args:
            file_path: Path to file
            process_fn: Processing function
            chunk_size: Chunk size in bytes

        Yields:
            Processed chunks
        """
        with open(file_path, 'r', buffering=chunk_size) as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield process_fn(chunk)


class RecursiveLoop:
    """Recursive data structure iteration."""

    @staticmethod
    def flatten(data: Any) -> Generator:
        """
        Flatten nested data structures.

        Args:
            data: Nested data

        Yields:
            Flat items
        """
        if isinstance(data, (list, tuple)):
            for item in data:
                yield from RecursiveLoop.flatten(item)
        else:
            yield data

    @staticmethod
    def tree_traverse(root: Any, children_fn: Callable) -> Generator:
        """
        Traverse tree structure.

        Args:
            root: Root node
            children_fn: Function to get children

        Yields:
            Tree nodes
        """
        yield root
        for child in children_fn(root):
            yield from RecursiveLoop.tree_traverse(child, children_fn)


def main(argv=None):
    """Memory-efficient loops command-line interface."""
    import argparse
    import json
    import math
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

    MAP_OPS = {
        "double": lambda x: x * 2,
        "square": lambda x: x * x,
        "negate": lambda x: -x,
        "increment": lambda x: x + 1,
        "strlen": lambda x: len(x) if isinstance(x, str) else x,
    }
    REDUCE_OPS = {
        "sum": sum,
        "product": lambda xs: math.prod(xs) if xs else 0,
        "min": min,
        "max": max,
        "count": len,
    }

    parser = argparse.ArgumentParser(description="Memory-efficient loops for local LLMs")
    sub = parser.add_subparsers(dest="command", required=True)

    ck = sub.add_parser("chunk")
    ck.add_argument("--data", required=True, help="Input data (JSON or file path)")
    ck.add_argument("--chunk-size", type=int, default=1000)

    sw = sub.add_parser("sliding")
    sw.add_argument("--data", required=True, help="Input data (JSON or file path)")
    sw.add_argument("--window-size", type=int, required=True)
    sw.add_argument("--step", type=int, default=1)

    fl = sub.add_parser("flatten")
    fl.add_argument("--data", required=True, help="Nested data (JSON or file path)")

    ft = sub.add_parser("filter")
    ft.add_argument("--data", required=True, help="Input data (JSON or file path)")
    ft.add_argument("--pred", default="even", choices=["even", "odd", "positive"])

    mr = sub.add_parser("map-reduce")
    mr.add_argument("--data", required=True, help="Input data (JSON or file path)")
    mr.add_argument("--map-op", default="double", choices=sorted(MAP_OPS))
    mr.add_argument("--reduce-op", default="sum", choices=sorted(REDUCE_OPS))
    mr.add_argument("--chunk-size", type=int, default=1000)

    tk = sub.add_parser("take")
    tk.add_argument("--data", required=True, help="Input data (JSON or file path)")
    tk.add_argument("--n", type=int, required=True)
    tk.add_argument("--chunk-size", type=int, default=1000)

    cs = sub.add_parser("consume")
    cs.add_argument("--data", required=True, help="Input data (JSON or file path)")
    cs.add_argument("--chunk-size", type=int, default=1000)

    args = parser.parse_args(argv)

    try:
        loop = MemoryEfficientLoop()
        if args.command == "chunk":
            result = list(loop.chunked_iter(resolve(args.data), args.chunk_size))
        elif args.command == "sliding":
            result = list(loop.sliding_window(resolve(args.data), args.window_size, args.step))
        elif args.command == "flatten":
            result = list(RecursiveLoop.flatten(resolve(args.data)))
        elif args.command == "filter":
            data = resolve(args.data)
            pred = {"even": lambda x: x % 2 == 0, "odd": lambda x: x % 2 == 1, "positive": lambda x: x > 0}[args.pred]
            result = list(loop.lazy_filter(data, pred))
        elif args.command == "map-reduce":
            data = resolve(args.data)
            result = loop.map_reduce(
                data, MAP_OPS[args.map_op], REDUCE_OPS[args.reduce_op], args.chunk_size
            )
        elif args.command == "take":
            data = resolve(args.data)
            result = loop.take(loop.chunked_iter(data, args.chunk_size), args.n)
        else:
            data = resolve(args.data)
            result = loop.consume(loop.chunked_iter(data, args.chunk_size))
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
