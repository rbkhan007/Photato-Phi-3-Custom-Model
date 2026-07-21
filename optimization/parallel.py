#!/usr/bin/env python3
"""
Parallel Processing Module for Local LLMs.

Features:
- Multi-threaded inference
- Pipeline parallelism
- Data parallelism
- Async processing
- Work stealing scheduler

Usage:
    from optimization.parallel import ParallelProcessor

    processor = ParallelProcessor(num_workers=4)
    results = processor.parallel_map(process_fn, data)
"""

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Callable, Optional


@dataclass
class ParallelConfig:
    """Parallel processing configuration."""
    num_workers: int = 4
    max_queue_size: int = 100
    chunk_size: int = 10
    use_threads: bool = True
    timeout: float = 30.0


@dataclass
class WorkItem:
    """Work item for parallel processing."""
    id: str
    data: Any
    priority: int = 0
    metadata: dict = field(default_factory=dict)


class WorkStealingQueue:
    """Work-stealing queue for load balancing."""

    def __init__(self):
        self.queues: dict[int, deque] = {}
        self.lock = threading.Lock()

    def add(self, worker_id: int, item: WorkItem):
        """Add item to worker's queue."""
        with self.lock:
            if worker_id not in self.queues:
                self.queues[worker_id] = deque()
            self.queues[worker_id].append(item)

    def steal(self, thief_id: int) -> Optional[WorkItem]:
        """Steal work from another worker."""
        with self.lock:
            for victim_id, queue in self.queues.items():
                if victim_id != thief_id and queue:
                    return queue.pop()
        return None

    def get(self, worker_id: int) -> Optional[WorkItem]:
        """Get item from worker's queue."""
        with self.lock:
            if worker_id in self.queues and self.queues[worker_id]:
                return self.queues[worker_id].popleft()
        return None

    def is_empty(self) -> bool:
        """Check if all queues are empty."""
        with self.lock:
            return all(not q for q in self.queues.values())


class ParallelProcessor:
    """
    Parallel processor for LLM inference.

    Features:
    - Multi-threaded processing
    - Work stealing
    - Pipeline parallelism
    - Async execution
    """

    def __init__(self, config: Optional[ParallelConfig] = None):
        """Initialize parallel processor."""
        self.config = config or ParallelConfig()
        self.work_queue = WorkStealingQueue()
        self.results: dict[str, Any] = {}
        self.lock = threading.Lock()

    def parallel_map(
        self,
        func: Callable,
        data: list,
        chunk_size: Optional[int] = None,
    ) -> list:
        """
        Parallel map operation.

        Args:
            func: Function to apply
            data: Input data
            chunk_size: Chunk size

        Returns:
            List of results
        """
        chunk_size = chunk_size or self.config.chunk_size
        results = [None] * len(data)

        def process_chunk(start_idx: int, chunk: list):
            for i, item in enumerate(chunk):
                results[start_idx + i] = func(item)

        # Split into chunks
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunks.append((i, data[i:i + chunk_size]))

        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = [
                executor.submit(process_chunk, idx, chunk)
                for idx, chunk in chunks
            ]
            for future in as_completed(futures):
                future.result()

        return results

    def parallel_for(
        self,
        func: Callable,
        start: int,
        end: int,
        chunk_size: Optional[int] = None,
    ):
        """
        Parallel for loop.

        Args:
            func: Function to apply (takes index)
            start: Start index
            end: End index
            chunk_size: Chunk size
        """
        chunk_size = chunk_size or self.config.chunk_size

        def process_range(s: int, e: int):
            for i in range(s, e):
                func(i)

        # Split into chunks
        ranges = []
        for i in range(start, end, chunk_size):
            ranges.append((i, min(i + chunk_size, end)))

        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = [
                executor.submit(process_range, s, e)
                for s, e in ranges
            ]
            for future in as_completed(futures):
                future.result()

    def pipeline_process(
        self,
        stages: list[Callable],
        data: list,
    ) -> list:
        """
        Process data through pipeline stages.

        Args:
            stages: List of processing functions
            data: Input data

        Returns:
            Processed data
        """
        current_data = data

        for stage in stages:
            # Process stage in parallel
            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = [executor.submit(stage, item) for item in current_data]
                current_data = [f.result() for f in as_completed(futures)]

        return current_data

    def async_process(
        self,
        func: Callable,
        data: list,
    ) -> list:
        """
        Asynchronous processing.

        Args:
            func: Function to apply
            data: Input data

        Returns:
            List of futures
        """
        executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        futures = [executor.submit(func, item) for item in data]
        return futures

    def map_unordered(
        self,
        func: Callable,
        data: list,
    ) -> list:
        """
        Map with unordered results (faster).

        Args:
            func: Function to apply
            data: Input data

        Returns:
            List of results
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = {executor.submit(func, item): i for i, item in enumerate(data)}
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def parallel_reduce(
        self,
        data: list,
        map_fn: Callable,
        reduce_fn: Callable,
        chunk_size: Optional[int] = None,
    ) -> Any:
        """
        Parallel map-reduce.

        Args:
            data: Input data
            map_fn: Map function
            reduce_fn: Reduce function
            chunk_size: Chunk size

        Returns:
            Reduced result
        """
        chunk_size = chunk_size or self.config.chunk_size

        # Map phase
        mapped = self.parallel_map(map_fn, data, chunk_size)

        # Reduce phase
        return reduce_fn(mapped)

    def work_stealing_process(
        self,
        func: Callable,
        data: list,
    ) -> list:
        """
        Process with work stealing.

        Args:
            func: Function to apply
            data: Input data

        Returns:
            List of results
        """
        results = [None] * len(data)
        completed = [False] * len(data)
        lock = threading.Lock()
        counter = [0]

        def worker(worker_id: int):
            while True:
                with lock:
                    if counter[0] >= len(data):
                        break
                    idx = counter[0]
                    counter[0] += 1

                # Process item
                results[idx] = func(data[idx])
                with lock:
                    completed[idx] = True

        # Start workers
        threads = []
        for i in range(self.config.num_workers):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        return results


class Pipeline:
    """Pipeline parallelism implementation."""

    def __init__(self, stages: list[Callable], num_workers: int = 2):
        self.stages = stages
        self.num_workers = num_workers
        self.queues = [Queue() for _ in stages]

    def process(self, data: list) -> list:
        """Process data through pipeline."""
        # Simple sequential pipeline
        current = data
        for stage in self.stages:
            current = [stage(item) for item in current]
        return current


class AsyncProcessor:
    """Async processing with callbacks."""

    def __init__(self):
        self.callbacks: dict[str, Callable] = {}

    def process_async(
        self,
        task_id: str,
        func: Callable,
        data: Any,
        callback: Optional[Callable] = None,
    ):
        """Process async with callback."""
        def worker():
            result = func(data)
            if callback:
                callback(task_id, result)
            self.callbacks[task_id] = result

        thread = threading.Thread(target=worker)
        thread.start()
        return thread

    def get_result(self, task_id: str) -> Optional[Any]:
        """Get async result."""
        return self.callbacks.get(task_id)


def main(argv=None):
    """Parallel processor command-line interface."""
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

    parser = argparse.ArgumentParser(description="Parallel processing for local LLMs")
    sub = parser.add_subparsers(dest="command", required=True)

    mp = sub.add_parser("map")
    mp.add_argument("--data", required=True, help="Input data (JSON or file path)")
    mp.add_argument("--op", default="double", choices=sorted(MAP_OPS))
    mp.add_argument("--workers", type=int, default=4)
    mp.add_argument("--chunk-size", type=int, default=10)

    rd = sub.add_parser("reduce")
    rd.add_argument("--data", required=True, help="Input data (JSON or file path)")
    rd.add_argument("--map-op", default="double", choices=sorted(MAP_OPS))
    rd.add_argument("--reduce-op", default="sum", choices=sorted(REDUCE_OPS))
    rd.add_argument("--chunk-size", type=int, default=10)

    pl = sub.add_parser("pipeline")
    pl.add_argument("--data", required=True, help="Input data (JSON or file path)")
    pl.add_argument("--stages", required=True, help="Comma-separated stage op names")
    pl.add_argument("--workers", type=int, default=4)

    mu = sub.add_parser("map-unordered")
    mu.add_argument("--data", required=True, help="Input data (JSON or file path)")
    mu.add_argument("--op", default="double", choices=sorted(MAP_OPS))
    mu.add_argument("--workers", type=int, default=4)

    args = parser.parse_args(argv)

    try:
        processor = ParallelProcessor()
        if args.command == "map":
            config = ParallelConfig(num_workers=args.workers, chunk_size=args.chunk_size)
            result = ParallelProcessor(config).parallel_map(MAP_OPS[args.op], resolve(args.data), args.chunk_size)
        elif args.command == "reduce":
            config = ParallelConfig(chunk_size=args.chunk_size)
            p = ParallelProcessor(config)
            result = p.parallel_reduce(
                resolve(args.data), MAP_OPS[args.map_op], REDUCE_OPS[args.reduce_op], args.chunk_size
            )
        elif args.command == "pipeline":
            config = ParallelConfig(num_workers=args.workers)
            stages = [MAP_OPS[s.strip()] for s in args.stages.split(",") if s.strip()]
            result = ParallelProcessor(config).pipeline_process(stages, resolve(args.data))
        else:
            config = ParallelConfig(num_workers=args.workers)
            result = ParallelProcessor(config).map_unordered(MAP_OPS[args.op], resolve(args.data))
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
