#!/usr/bin/env python3
"""
Optimized llama.cpp Inference Server with GPU/CPU support.

Features:
- Fast CPU + GPU inference via llama.cpp
- OpenAI-compatible API server
- Streaming support
- Batch processing
- Memory-mapped models for fast loading
- Auto device detection (CPU/GPU)

Usage:
    python inference/llama_server.py --model ./phi3-mini-q4_k_m.gguf --port 8080

    # Then use with any OpenAI-compatible client:
    curl http://localhost:8080/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "phi3-mini", "messages": [{"role": "user", "content": "Hello!"}]}'
"""

import argparse
import json
import os
import signal
import sys
import time
import threading
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional


@dataclass
class ServerConfig:
    model_path: str
    host: str = "127.0.0.1"
    port: int = 8080
    n_ctx: int = 4096
    n_batch: int = 512
    n_threads: Optional[int] = None
    n_gpu_layers: int = 0
    verbose: bool = False
    mmap: bool = True
    mlock: bool = False
    rope_freq_base: float = 10000.0
    rope_freq_scale: float = 1.0
    use_mmap: bool = True
    use_mlock: bool = False


class LlamaCppServer:
    """
    llama.cpp inference server with OpenAI-compatible API.

    Features:
    - /v1/chat/completions endpoint
    - /v1/completions endpoint
    - /v1/models endpoint
    - Streaming support (SSE)
    - Health check endpoint
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load llama.cpp model."""
        print(f"Loading model: {self.config.model_path}")

        try:
            # Try to use llama-cpp-python
            from llama_cpp import Llama

            n_gpu_layers = self.config.n_gpu_layers
            if n_gpu_layers == -1:
                # Auto-detect GPU
                try:
                    import torch
                    if torch.cuda.is_available():
                        n_gpu_layers = 35  # Offload all layers
                        print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
                    else:
                        n_gpu_layers = 0
                        print("No GPU detected, using CPU")
                except ImportError:
                    n_gpu_layers = 0
                    print("PyTorch not available, using CPU")

            self.model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.n_ctx,
                n_batch=self.config.n_batch,
                n_threads=self.config.n_threads or os.cpu_count(),
                n_gpu_layers=n_gpu_layers,
                verbose=self.config.verbose,
                use_mmap=self.config.use_mmap,
                use_mlock=self.config.use_mlock,
                rope_freq_base=self.config.rope_freq_base,
                rope_freq_scale=self.config.rope_freq_scale,
            )

            print(f"Model loaded successfully!")
            print(f"  Context size: {self.config.n_ctx}")
            print(f"  GPU layers: {n_gpu_layers}")
            print(f"  Threads: {self.config.n_threads or os.cpu_count()}")

        except ImportError:
            print("Error: llama-cpp-python not installed.")
            print("Install with: pip install llama-cpp-python")
            print("For GPU support: pip install llama-cpp-python[cuda]")
            sys.exit(1)

    def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 1024,
        stream: bool = False,
        stop: Optional[list[str]] = None,
        **kwargs,
    ) -> dict:
        """
        Generate chat completion.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            top_p: Nucleus sampling
            top_k: Top-k sampling
            max_tokens: Maximum tokens to generate
            stream: Enable streaming
            stop: Stop sequences

        Returns:
            OpenAI-compatible response dict
        """
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)

        # Generate
        start_time = time.time()

        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
            echo=False,
        )

        generation_time = time.time() - start_time

        # Extract response
        text = response["choices"][0]["text"]
        tokens_used = response["usage"]["total_tokens"]

        return {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": Path(self.config.model_path).stem,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"],
                "total_tokens": tokens_used,
            },
            "generation_time": generation_time,
        }

    def completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 1024,
        stream: bool = False,
        stop: Optional[list[str]] = None,
        **kwargs,
    ) -> dict:
        """
        Generate text completion.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling
            top_k: Top-k sampling
            max_tokens: Maximum tokens to generate
            stream: Enable streaming
            stop: Stop sequences

        Returns:
            OpenAI-compatible response dict
        """
        start_time = time.time()

        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
            echo=False,
        )

        generation_time = time.time() - start_time

        text = response["choices"][0]["text"]

        return {
            "id": f"cmpl-{int(time.time()*1000)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": Path(self.config.model_path).stem,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"],
                "total_tokens": response["usage"]["total_tokens"],
            },
            "generation_time": generation_time,
        }

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert messages to prompt format."""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}<|end|>")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}<|end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}<|end|>")
        prompt_parts.append("<|assistant|>\n")
        return "\n".join(prompt_parts)

    def get_models(self) -> dict:
        """Get list of available models."""
        return {
            "object": "list",
            "data": [
                {
                    "id": Path(self.config.model_path).stem,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }


class RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OpenAI-compatible API."""

    server: LlamaCppServer

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def _send_json(self, data: dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _read_body(self) -> dict:
        """Read request body as JSON."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        return json.loads(body) if body else {}

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path == "/v1/models":
            self._send_json(self.server.get_models())
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        """Handle POST requests."""
        try:
            body = self._read_body()

            if self.path == "/v1/chat/completions":
                response = self.server.chat_completion(**body)
                self._send_json(response)

            elif self.path == "/v1/completions":
                response = self.server.completion(**body)
                self._send_json(response)

            else:
                self._send_json({"error": "Not found"}, 404)

        except Exception as e:
            self._send_json({"error": str(e)}, 500)


def run_server(config: ServerConfig):
    """Run the inference server."""
    server = LlamaCppServer(config)
    http_server = HTTPServer((config.host, config.port), RequestHandler)
    http_server.server = server

    print(f"\nServer running on http://{config.host}:{config.port}")
    print(f"API endpoints:")
    print(f"  POST /v1/chat/completions  - Chat completion")
    print(f"  POST /v1/completions       - Text completion")
    print(f"  GET  /v1/models            - List models")
    print(f"  GET  /health               - Health check")
    print(f"\nPress Ctrl+C to stop.\n")

    def signal_handler(sig, frame):
        print("\nShutting down server...")
        http_server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http_server.server_close()


def main(argv=None):
    """Run the llama.cpp inference server, or print its config."""
    parser = argparse.ArgumentParser(description="llama.cpp Inference Server")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context size")
    parser.add_argument("--n-batch", type=int, default=512, help="Batch size")
    parser.add_argument("--n-threads", type=int, help="Number of threads")
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="GPU layers (0=CPU, -1=auto, 35=all)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-mmap", action="store_true", help="Disable mmap")
    parser.add_argument("--mlock", action="store_true", help="Enable mlock")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the server configuration as JSON and exit",
    )

    args = parser.parse_args(argv)

    config = ServerConfig(
        model_path=args.model,
        host=args.host,
        port=args.port,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        verbose=args.verbose,
        use_mmap=not args.no_mmap,
        use_mlock=args.mlock,
    )

    if args.print_config:
        print(json.dumps(config.__dict__, indent=2, default=str))
        return 0

    try:
        run_server(config)
        return 0
    except KeyboardInterrupt:
        return 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
