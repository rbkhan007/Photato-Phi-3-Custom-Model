#!/usr/bin/env python3
"""
Quantize a fine-tuned Phi-3 Mini model using GPTQ (4-bit).

Usage:
    python quantize_gptq.py --adapter ./phi3-mini-lora-adapter --output ./phi3-mini-gptq

Requirements:
    pip install transformers peft auto-gptq datasets
"""

import argparse
import json
import sys
from pathlib import Path


def load_model_and_tokenizer(adapter_path: str):
    """Load base model with LoRA adapter, merge, and return."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_dir = Path(adapter_path)
    with open(adapter_dir / "adapter_config.json") as f:
        config = json.load(f)

    base_model_name = config["base_model_name_or_path"]
    print(f"Loading base model: {base_model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    print("Loading and merging LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    return model, tokenizer


def quantize_gptq(model, tokenizer, output_path: str, bits: int = 4, group_size: int = 128):
    """Quantize model using GPTQ."""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=True,
        sym=False,
        true_sequential=True,
    )

    print(f"Quantizing to {bits}-bit with group_size={group_size}...")

    # Prepare calibration data
    # Use a simple text for calibration (can be customized)
    calibration_text = "This is a sample calibration text for quantization. " * 100
    calibration_data = tokenizer(calibration_text, return_tensors="pt")

    # Create a simple dataset for calibration
    from datasets import Dataset

    cal_dataset = Dataset.from_dict({
        "input_ids": [calibration_data["input_ids"][0]],
        "attention_mask": [calibration_data["attention_mask"][0]],
    })

    quantizer = AutoGPTQForCausalLM.from_pretrained(
        model,
        quantize_config,
        trust_remote_code=True,
    )

    quantizer.quantize(
        cal_dataset,
        batch_size=1,
        use_triton=False,
    )

    print(f"Saving quantized model to {output_path}...")
    quantizer.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Quantized model saved to {output_path}")

    # Report size
    total_size = sum(
        f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file()
    ) / (1024 * 1024)
    print(f"Total model size: {total_size:.1f} MB")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Quantize Phi-3 Mini using GPTQ")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output", required=True, help="Output directory for quantized model")
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4], help="Quantization bits (default: 4)")
    parser.add_argument("--group-size", type=int, default=128, help="Group size (default: 128)")
    args = parser.parse_args(argv)

    try:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load and merge
        model, tokenizer = load_model_and_tokenizer(args.adapter)

        # Quantize
        quantize_gptq(model, tokenizer, str(output_path), args.bits, args.group_size)

        print(f"\nDone! Quantized model saved to: {output_path}")
        print(f"\nTo use with transformers:")
        print(f"  from auto_gptq import AutoGPTQForCausalLM")
        print(f"  model = AutoGPTQForCausalLM.from_quantized('{output_path}')")
        return 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
