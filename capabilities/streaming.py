#!/usr/bin/env python3
"""
Streaming Responses for Local LLMs.

Features:
- Token-by-token streaming
- Server-Sent Events (SSE)
- Streaming callbacks
- Buffer management

Usage:
    from capabilities.streaming import StreamingHandler

    handler = StreamingHandler()
    for token in handler.stream("Hello, how are you?"):
        print(token, end="", flush=True)
"""

import json
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class StreamChunk:
    """Single stream chunk."""
    token: str
    index: int
    finish_reason: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class StreamConfig:
    """Streaming configuration."""
    buffer_size: int = 10
    flush_interval: float = 0.01
    timeout: float = 30.0
    max_retries: int = 3


class StreamingHandler:
    """
    Streaming response handler for local LLMs.

    Features:
    - Token-by-token streaming
    - Callback support
    - Buffer management
    - SSE formatting
    """

    def __init__(self, config: Optional[StreamConfig] = None):
        """
        Initialize streaming handler.

        Args:
            config: Stream configuration
        """
        self.config = config or StreamConfig()
        self.buffer: deque[str] = deque(maxlen=self.config.buffer_size)
        self.callbacks: list[Callable] = []
        self.is_streaming = False

    def add_callback(self, callback: Callable):
        """Add streaming callback."""
        self.callbacks.append(callback)

    def stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ):
        """
        Stream response token by token.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature

        Yields:
            StreamChunk objects
        """
        self.is_streaming = True
        index = 0

        try:
            # Generate tokens (placeholder - use actual model)
            tokens = self._generate_tokens(prompt, max_tokens, temperature)

            for token in tokens:
                chunk = StreamChunk(
                    token=token,
                    index=index,
                    metadata={"temperature": temperature},
                )

                # Update buffer
                self.buffer.append(token)

                # Call callbacks
                for callback in self.callbacks:
                    try:
                        callback(chunk)
                    except Exception as e:
                        print(f"Callback error: {e}")

                yield chunk
                index += 1

                # Simulate streaming delay
                time.sleep(self.config.flush_interval)

            # Final chunk
            yield StreamChunk(
                token="",
                index=index,
                finish_reason="stop",
            )

        finally:
            self.is_streaming = False

    def _generate_tokens(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> list[str]:
        """
        Generate tokens for streaming.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Temperature

        Returns:
            List of tokens
        """
        # Placeholder - in production use actual model
        response = f"This is a streaming response to: {prompt[:50]}..."
        return response.split()

    def stream_to_string(
        self,
        prompt: str,
        max_tokens: int = 256,
    ) -> str:
        """
        Stream response and collect as string.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens

        Returns:
            Complete response string
        """
        tokens = []
        for chunk in self.stream(prompt, max_tokens):
            if chunk.finish_reason is None:
                tokens.append(chunk.token)
        return " ".join(tokens)


class SSEFormatter:
    """Server-Sent Events formatter."""

    @staticmethod
    def format_chunk(chunk: StreamChunk) -> str:
        """
        Format chunk as SSE.

        Args:
            chunk: Stream chunk

        Returns:
            SSE formatted string
        """
        data = {
            "token": chunk.token,
            "index": chunk.index,
            "finish_reason": chunk.finish_reason,
        }
        return f"data: {json.dumps(data)}\n\n"

    @staticmethod
    def format_done() -> str:
        """Format done event."""
        return "data: [DONE]\n\n"

    @staticmethod
    def format_error(error: str) -> str:
        """Format error event."""
        return f"data: {json.dumps({'error': error})}\n\n"


class AsyncStreamingHandler:
    """Async streaming handler."""

    def __init__(self):
        self.queue: deque[StreamChunk] = deque()
        self.is_done = False
        self.lock = threading.Lock()

    def put(self, chunk: StreamChunk):
        """Add chunk to queue."""
        with self.lock:
            self.queue.append(chunk)
            if chunk.finish_reason:
                self.is_done = True

    def get(self, timeout: float = 1.0) -> Optional[StreamChunk]:
        """Get chunk from queue."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if self.queue:
                    return self.queue.popleft()
            time.sleep(0.01)
        return None

    def __iter__(self):
        """Iterate over chunks."""
        while not self.is_done:
            chunk = self.get(timeout=0.1)
            if chunk:
                yield chunk

    def collect(self) -> str:
        """Collect all chunks as string."""
        tokens = []
        for chunk in self:
            if chunk.finish_reason is None:
                tokens.append(chunk.token)
        return " ".join(tokens)


class BufferedStreamer:
    """Buffered streaming for efficiency."""

    def __init__(self, buffer_size: int = 10):
        self.buffer_size = buffer_size
        self.buffer: list[str] = []
        self.output: list[str] = []

    def add_token(self, token: str):
        """Add token to buffer."""
        self.buffer.append(token)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Flush buffer to output."""
        if self.buffer:
            self.output.extend(self.buffer)
            self.buffer.clear()

    def get_output(self) -> str:
        """Get complete output."""
        self.flush()
        return " ".join(self.output)

    def clear(self):
        """Clear buffer and output."""
        self.buffer.clear()
        self.output.clear()


class StreamingResponse:
    """Complete streaming response handler."""

    def __init__(self):
        self.handler = StreamingHandler()
        self.sse = SSEFormatter()
        self.buffered = BufferedStreamer()

    def generate_streaming_response(
        self,
        prompt: str,
        callback: Optional[Callable] = None,
    ) -> str:
        """
        Generate streaming response.

        Args:
            prompt: Input prompt
            callback: Optional callback

        Returns:
            Complete response
        """
        if callback:
            self.handler.add_callback(callback)

        return self.handler.stream_to_string(prompt)

    def generate_sse_response(self, prompt: str) -> list[str]:
        """
        Generate SSE response.

        Args:
            prompt: Input prompt

        Returns:
            List of SSE events
        """
        events = []
        for chunk in self.handler.stream(prompt):
            events.append(self.sse.format_chunk(chunk))
            if chunk.finish_reason:
                events.append(self.sse.format_done())
        return events

    def generate_with_buffer(self, prompt: str) -> str:
        """
        Generate response with buffering.

        Args:
            prompt: Input prompt

        Returns:
            Complete response
        """
        for chunk in self.handler.stream(prompt):
            if chunk.finish_reason is None:
                self.buffered.add_token(chunk.token)
        return self.buffered.get_output()


def main(argv=None):
    """CLI for streaming responses."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="capabilities.streaming",
        description="Streaming responses for local LLMs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_stream = sub.add_parser("stream", help="Stream a response")
    src = p_stream.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt")
    src.add_argument("--file")
    p_stream.add_argument("--max-tokens", type=int, default=256)
    p_stream.add_argument("--temperature", type=float, default=0.7)
    p_stream.add_argument("--format", choices=["json", "tokens", "sse"], default="json")

    args = parser.parse_args(argv)

    def read_source(value, file_path):
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return value

    try:
        handler = StreamingHandler()
        prompt = read_source(args.prompt, args.file)
        if args.format == "tokens":
            for chunk in handler.stream(prompt, max_tokens=args.max_tokens, temperature=args.temperature):
                if chunk.finish_reason is None:
                    sys.stdout.write(chunk.token + " ")
                    sys.stdout.flush()
            sys.stdout.write("\n")
            return 0
        elif args.format == "sse":
            for chunk in handler.stream(prompt, max_tokens=args.max_tokens, temperature=args.temperature):
                sys.stdout.write(SSEFormatter.format_chunk(chunk))
                if chunk.finish_reason:
                    sys.stdout.write(SSEFormatter.format_done())
            return 0
        else:
            tokens = []
            for chunk in handler.stream(prompt, max_tokens=args.max_tokens, temperature=args.temperature):
                if chunk.finish_reason is None:
                    tokens.append(chunk.token)
            print(json.dumps({
                "prompt": prompt,
                "tokens": tokens,
                "response": " ".join(tokens),
                "num_chunks": len(tokens),
            }, indent=2, default=str))
            return 0
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
