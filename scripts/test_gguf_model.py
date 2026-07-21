"""
Quick GGUF model test — verify the local model works end-to-end.

Usage:
    python scripts/test_gguf_model.py
    python scripts/test_gguf_model.py --prompt "What is Python?"
    python scripts/test_gguf_model.py --model notebooks/Phi-4-mini-instruct-Q4_K_M.gguf
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Test GGUF model inference")
    parser.add_argument("--model", "-m", default="",
                        help="Path to GGUF model (default: notebooks/Phi-4-mini-instruct-Q4_K_M.gguf)")
    parser.add_argument("--prompt", "-p", default="Say hello in one word.",
                        help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Max tokens to generate")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")
    args = parser.parse_args()

    model_path = args.model or str(Path("notebooks/Phi-4-mini-instruct-Q4_K_M.gguf"))

    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}")
        print("Expected GGUF file in notebooks/ directory.")
        print("Available models:")
        for f in sorted(Path("notebooks").glob("*.gguf")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}  ({size_mb:.0f} MB)")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    t0 = time.time()

    from inference.llama_engine import FastLlamaEngine

    engine = FastLlamaEngine(
        model_path,
        n_ctx=2048,
        n_batch=256,
        verbose=False,
    )

    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Threads: {engine._threads}")
    print(f"  Model file size: {Path(model_path).stat().st_size / (1024*1024):.0f} MB")
    print()

    t0 = time.time()
    out = engine.generate(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
    elapsed = time.time() - t0

    print(f"Prompt: {args.prompt}")
    print(f"Response: {out['text']}")
    print()
    print(f"  Tokens: {out['completion_tokens']}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {out['tokens_per_second']:.1f} tok/s")
    print(f"  First token: {out['first_token_ms']:.0f}ms")

    if args.verbose:
        print(f"\n  Full output: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
