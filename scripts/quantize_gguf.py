#!/usr/bin/env python3
"""
Quantize a fine-tuned Phi-3 Mini model to GGUF format for llama.cpp/Ollama.

Usage:
    python quantize_gguf.py --adapter ./phi3-mini-lora-adapter --output ./phi3-mini-q4_k_m.gguf --quant Q4_K_M

Requirements:
    pip install transformers peft sentencepiece
    git clone https://github.com/ggerganov/llama.cpp
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def clone_llama_cpp(llama_cpp_dir: str) -> Path:
    """Clone llama.cpp if not already present."""
    llama_path = Path(llama_cpp_dir)
    if not llama_path.exists():
        print("Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp", str(llama_path)],
            check=True,
        )
    return llama_path


def build_llama_cpp(llama_path: Path) -> None:
    """Build llama.cpp."""
    print("Building llama.cpp...")
    subprocess.run(["make", "-j", str(os.cpu_count() or 4)], cwd=llama_path, check=True)
    print("Build complete.")


def merge_adapter_to_hf(adapter_path: str, output_path: str) -> Path:
    """Merge LoRA adapter with base model and save as HuggingFace format."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading base model...")
    adapter_dir = Path(adapter_path)

    # Read adapter config to get base model name
    import json
    with open(adapter_dir / "adapter_config.json") as f:
        config = json.load(f)
    base_model_name = config["base_model_name_or_path"]

    print(f"Base model: {base_model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("Merging adapter weights...")
    model = model.merge_and_unload()

    merged_dir = Path(output_path) / "merged_hf"
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    print("Merged model saved.")
    return merged_dir


def convert_to_gguf(llama_path: Path, hf_model_path: Path, gguf_output: Path) -> None:
    """Convert HuggingFace model to GGUF format."""
    convert_script = llama_path / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        print(f"Error: {convert_script} not found. Build llama.cpp first.")
        sys.exit(1)

    print("Converting to GGUF...")
    subprocess.run(
        [
            sys.executable,
            str(convert_script),
            str(hf_model_path),
            "--outfile",
            str(gguf_output),
            "--outtype",
            "f16",
        ],
        check=True,
    )
    print(f"GGUF file created: {gguf_output}")


def quantize_gguf(llama_path: Path, input_gguf: Path, output_gguf: Path, quant_type: str) -> None:
    """Quantize GGUF file to specified quantization type."""
    quantize_bin = llama_path / "llama-quantize"

    if not quantize_bin.exists():
        print(f"Error: {quantize_bin} not found. Build llama.cpp first.")
        sys.exit(1)

    print(f"Quantizing to {quant_type}...")
    subprocess.run(
        [str(quantize_bin), str(input_gguf), str(output_gguf), quant_type],
        check=True,
    )

    # Get file sizes
    input_size = input_gguf.stat().st_size / (1024 * 1024)
    output_size = output_gguf.stat().st_size / (1024 * 1024)
    print(f"Quantization complete: {output_gguf}")
    print(f"  Original: {input_size:.1f} MB")
    print(f"  Quantized: {output_size:.1f} MB ({output_size/input_size*100:.1f}%)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Quantize Phi-3 Mini to GGUF")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output", required=True, help="Output path for quantized GGUF file")
    parser.add_argument(
        "--quant",
        default="Q4_K_M",
        choices=["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"],
        help="Quantization type (default: Q4_K_M)",
    )
    parser.add_argument("--llama-cpp-dir", default="./llama.cpp", help="Path to llama.cpp directory")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge step (use existing merged model)")
    parser.add_argument("--merged-model", default=None, help="Path to already-merged HF model (use with --skip-merge)")
    args = parser.parse_args(argv)

    try:
        # Clone and build llama.cpp
        llama_path = clone_llama_cpp(args.llama_cpp_dir)
        build_llama_cpp(llama_path)

        # Merge adapter or use existing
        if args.skip_merge and args.merged_model:
            hf_model_path = Path(args.merged_model)
        else:
            output_dir = Path(args.output).parent
            hf_model_path = merge_adapter_to_hf(args.adapter, str(output_dir))

        # Convert to GGUF
        output_path = Path(args.output)
        f16_gguf = output_path.with_suffix(".f16.gguf")
        convert_to_gguf(llama_path, hf_model_path, f16_gguf)

        # Quantize
        quantize_gguf(llama_path, f16_gguf, output_path, args.quant)

        # Clean up intermediate F16 file
        f16_gguf.unlink(missing_ok=True)

        print(f"\nDone! Your quantized model is at: {output_path}")
        print(f"\nTo use with Ollama:")
        print(f"  echo 'FROM {output_path}' > Modelfile")
        print(f"  ollama create my-model -f Modelfile")
        print(f"  ollama run my-model")
        return 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
