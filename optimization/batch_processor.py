#!/usr/bin/env python3
"""
Batch Processing Optimizer for Local LLMs.

Features:
- Dynamic batch sizing
- Sequence padding and packing
- Parallel batch processing
- Memory-efficient batching
- Adaptive batch tuning

Usage:
    from optimization.batch_processor import BatchProcessor

    processor = BatchProcessor(max_batch_size=32)
    results = processor.process_batch(prompts)
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class BatchConfig:
    """Batch processing configuration."""
    max_batch_size: int = 32
    min_batch_size: int = 1
    max_sequence_length: int = 2048
    padding_token: str = "<pad>"
    dynamic_sizing: bool = True
    memory_limit_mb: float = 8000


@dataclass
class BatchItem:
    """Item in a batch."""
    id: str
    data: Any
    length: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of batch processing."""
    items: list[BatchItem]
    results: list[Any]
    batch_size: int
    processing_time: float
    tokens_per_second: float = 0.0


class BatchProcessor:
    """
    Optimized batch processor for LLM inference.

    Features:
    - Dynamic batch sizing
    - Sequence packing
    - Memory-efficient processing
    - Parallel processing simulation
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize batch processor."""
        self.config = config or BatchConfig()
        self.queue: deque[BatchItem] = deque()
        self.processed_count = 0

    def add_item(self, id: str, data: Any, length: int = 0):
        """Add item to batch queue."""
        self.queue.append(BatchItem(id=id, data=data, length=length))

    def get_optimal_batch_size(self, items: list[BatchItem]) -> int:
        """
        Calculate optimal batch size based on sequence lengths.

        Args:
            items: Batch items

        Returns:
            Optimal batch size
        """
        if not self.config.dynamic_sizing:
            return self.config.max_batch_size

        # Calculate average sequence length
        avg_length = sum(item.length for item in items) / len(items) if items else 1

        # Estimate memory per item
        memory_per_item = avg_length * 4 * 2  # 4 bytes per param, 2 for key+value

        # Calculate max items based on memory
        max_by_memory = int(self.config.memory_limit_mb * 1024 * 1024 / memory_per_item)

        # Take minimum of constraints
        optimal = min(
            self.config.max_batch_size,
            max_by_memory,
            len(items),
        )

        return max(self.config.min_batch_size, optimal)

    def pad_batch(self, items: list[BatchItem]) -> list[list[str]]:
        """
        Pad batch items to same length.

        Args:
            items: Batch items

        Returns:
            Padded batch
        """
        max_length = max(item.length for item in items) if items else 0
        max_length = min(max_length, self.config.max_sequence_length)

        padded = []
        for item in items:
            tokens = item.data.split() if isinstance(item.data, str) else item.data
            # Truncate if too long
            tokens = tokens[:max_length]
            # Pad if too short
            padding_needed = max_length - len(tokens)
            tokens.extend([self.config.padding_token] * padding_needed)
            padded.append(tokens)

        return padded

    def pack_batch(self, items: list[BatchItem]) -> list[list[str]]:
        """
        Pack multiple sequences into single sequence.

        Args:
            items: Batch items

        Returns:
            Packed batch
        """
        packed = []
        current_pack = []
        current_length = 0

        for item in items:
            tokens = item.data.split() if isinstance(item.data, str) else item.data

            if current_length + len(tokens) <= self.config.max_sequence_length:
                current_pack.extend(tokens)
                current_length += len(tokens)
            else:
                if current_pack:
                    packed.append(current_pack)
                current_pack = tokens
                current_length = len(tokens)

        if current_pack:
            packed.append(current_pack)

        return packed

    def process_batch(
        self,
        process_fn: Callable,
        items: Optional[list[BatchItem]] = None,
    ) -> BatchResult:
        """
        Process batch of items.

        Args:
            process_fn: Processing function
            items: Items to process (uses queue if None)

        Returns:
            BatchResult
        """
        start_time = time.time()

        # Get items from queue if not provided
        if items is None:
            items = []
            while self.queue and len(items) < self.config.max_batch_size:
                items.append(self.queue.popleft())

        if not items:
            return BatchResult(items=[], results=[], batch_size=0, processing_time=0)

        # Determine batch size
        batch_size = self.get_optimal_batch_size(items)

        # Process in batches
        all_results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = process_fn(batch)
            all_results.extend(batch_results)

        processing_time = time.time() - start_time

        # Calculate throughput
        total_tokens = sum(item.length for item in items)
        tokens_per_second = total_tokens / processing_time if processing_time > 0 else 0

        self.processed_count += len(items)

        return BatchResult(
            items=items,
            results=all_results,
            batch_size=batch_size,
            processing_time=processing_time,
            tokens_per_second=tokens_per_second,
        )

    def process_streaming(
        self,
        process_fn: Callable,
        chunk_size: int = 10,
    ):
        """
        Process items in streaming fashion.

        Args:
            process_fn: Processing function
            chunk_size: Chunk size

        Yields:
            BatchResult for each chunk
        """
        while self.queue:
            chunk = []
            for _ in range(min(chunk_size, len(self.queue))):
                if self.queue:
                    chunk.append(self.queue.popleft())

            if chunk:
                result = self.process_batch(process_fn, chunk)
                yield result


class DynamicBatcher:
    """Dynamic batching with adaptive sizing."""

    def __init__(self, target_batch_time: float = 0.1):
        self.target_batch_time = target_batch_time
        self.current_batch_size = 1
        self.batch_times: deque = deque(maxlen=100)

    def update_batch_size(self, processing_time: float, batch_size: int) -> int:
        """Adjust batch size based on processing time."""
        self.batch_times.append(processing_time)

        if len(self.batch_times) < 5:
            return self.current_batch_size

        avg_time = sum(self.batch_times) / len(self.batch_times)

        if avg_time < self.target_batch_time * 0.8:
            # Too fast, increase batch size
            self.current_batch_size = min(self.current_batch_size * 2, 64)
        elif avg_time > self.target_batch_time * 1.2:
            # Too slow, decrease batch size
            self.current_batch_size = max(self.current_batch_size // 2, 1)

        return self.current_batch_size


class SequencePacker:
    """Pack sequences for efficient processing."""

    @staticmethod
    def pack(
        sequences: list[list[str]],
        max_length: int = 2048,
        pad_token: str = "<pad>",
    ) -> list[list[str]]:
        """
        Pack sequences into fixed-length batches.

        Args:
            sequences: Input sequences
            max_length: Maximum sequence length
            pad_token: Padding token

        Returns:
            Packed sequences
        """
        # Sort by length for efficient packing
        sorted_seqs = sorted(sequences, key=len)

        packed = []
        current_batch = []
        current_length = 0

        for seq in sorted_seqs:
            if current_length + len(seq) <= max_length:
                current_batch.append(seq)
                current_length += len(seq)
            else:
                if current_batch:
                    # Pad batch
                    max_len = max(len(s) for s in current_batch)
                    padded = [s + [pad_token] * (max_len - len(s)) for s in current_batch]
                    packed.append(padded)
                current_batch = [seq]
                current_length = len(seq)

        if current_batch:
            max_len = max(len(s) for s in current_batch)
            padded = [s + [pad_token] * (max_len - len(s)) for s in current_batch]
            packed.append(padded)

        return packed


class MemoryEfficientBatcher:
    """Memory-efficient batching with gradient accumulation."""

    def __init__(self, max_memory_mb: float = 4000):
        self.max_memory_mb = max_memory_mb
        self.accumulation_steps = 1

    def calculate_accumulation_steps(
        self,
        batch_size: int,
        model_size_mb: float,
    ) -> int:
        """Calculate gradient accumulation steps."""
        memory_per_batch = model_size_mb * batch_size * 0.5
        steps = max(1, int(memory_per_batch / self.max_memory_mb))
        self.accumulation_steps = steps
        return steps

    def get_effective_batch_size(self, batch_size: int) -> int:
        """Get effective batch size with accumulation."""
        return batch_size * self.accumulation_steps


def main(argv=None):
    """Batch processor command-line interface."""
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

    def build_items(raw):
        items = []
        for entry in raw:
            if isinstance(entry, str):
                data = entry
                length = len(data.split())
                items.append(BatchItem(id=f"item_{len(items)}", data=data, length=length))
            elif isinstance(entry, dict):
                items.append(BatchItem(
                    id=entry.get("id", f"item_{len(items)}"),
                    data=entry.get("data", ""),
                    length=entry.get("length", len(str(entry.get("data", "")).split())),
                ))
            else:
                raise ValueError("each item must be a string or object")
        return items

    parser = argparse.ArgumentParser(description="Batch processing optimizer for local LLMs")
    sub = parser.add_subparsers(dest="command", required=True)

    pad = sub.add_parser("pad")
    pad.add_argument("--items", required=True, help="Items (JSON list of strings/objects or file path)")
    pad.add_argument("--max-sequence-length", type=int, default=2048)
    pad.add_argument("--padding-token", default="<pad>")

    pk = sub.add_parser("pack")
    pk.add_argument("--items", required=True, help="Items (JSON list of strings/objects or file path)")
    pk.add_argument("--max-sequence-length", type=int, default=2048)

    ob = sub.add_parser("optimal-size")
    ob.add_argument("--items", required=True, help="Items (JSON list of strings/objects or file path)")
    ob.add_argument("--max-batch-size", type=int, default=32)
    ob.add_argument("--min-batch-size", type=int, default=1)
    ob.add_argument("--memory-limit-mb", type=float, default=8000.0)

    ps = sub.add_parser("pack-sequences")
    ps.add_argument("--sequences", required=True, help="List of token sequences (JSON or file path)")
    ps.add_argument("--max-length", type=int, default=2048)
    ps.add_argument("--pad-token", default="<pad>")

    args = parser.parse_args(argv)

    try:
        if args.command in ("pad", "pack", "optimal-size"):
            items = build_items(resolve(args.items))
        if args.command == "pad":
            config = BatchConfig(
                max_sequence_length=args.max_sequence_length, padding_token=args.padding_token
            )
            result = BatchProcessor(config).pad_batch(items)
        elif args.command == "pack":
            config = BatchConfig(max_sequence_length=args.max_sequence_length)
            result = BatchProcessor(config).pack_batch(items)
        elif args.command == "optimal-size":
            config = BatchConfig(
                max_batch_size=args.max_batch_size,
                min_batch_size=args.min_batch_size,
                memory_limit_mb=args.memory_limit_mb,
            )
            result = BatchProcessor(config).get_optimal_batch_size(items)
        else:
            sequences = resolve(args.sequences)
            result = SequencePacker.pack(sequences, max_length=args.max_length, pad_token=args.pad_token)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
