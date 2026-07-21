#!/usr/bin/env python3
"""
Train the project model on your own dataset using QLoRA/LoRA.

End-to-end pipeline:
    1. Load CSV conversation data
    2. Apply LoRA/QLoRA to base model
    3. Train with SFTTrainer
    4. Save LoRA adapter + config + metrics

Usage:
    python scripts/train_model.py datas/Claudecode.csv --preset local_gpu --output ./my-custom-model
    python scripts/train_model.py datas/Claudecode.csv --preset phi4_mini --keep-think --epochs 5
    python scripts/train_model.py datas/Claudecode.csv --model microsoft/Phi-4-mini-instruct --lora-rank 32
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.memory_efficient import (
    MemoryEfficientTrainer,
    TrainingConfig,
    MemoryEfficientConfigPresets,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train model on CSV conversation data")

    parser.add_argument("data", help="Path to CSV or JSONL training data")

    parser.add_argument("--preset", "-p", default="phi4_mini",
                        choices=["phi4_mini", "qwen3_embedding", "colab_free_tier",
                                 "colab_pro", "local_gpu", "high_end_gpu", "cpu_only"],
                        help="Hardware preset (default: phi4_mini)")

    parser.add_argument("--model", "-m", default="",
                        help="Base model path (overrides preset)")

    parser.add_argument("--output", "-o", default="./output/my-custom-model",
                        help="Output directory for LoRA adapter")

    parser.add_argument("--epochs", type=int, default=0,
                        help="Number of epochs (overrides preset)")

    parser.add_argument("--batch-size", type=int, default=0,
                        help="Batch size (overrides preset)")

    parser.add_argument("--lora-rank", type=int, default=0,
                        help="LoRA rank (overrides preset)")

    parser.add_argument("--max-seq-length", type=int, default=0,
                        help="Max sequence length (overrides preset)")

    parser.add_argument("--learning-rate", type=float, default=0,
                        help="Learning rate (overrides preset)")

    parser.add_argument("--keep-think", action="store_true", default=True,
                        help="Keep <think> tags in assistant responses")
    parser.add_argument("--no-think", action="store_false", dest="keep_think",
                        help="Strip <think> tags")

    parser.add_argument("--template", default="phi4",
                        choices=["phi4", "phi3", "llama3", "chatml"],
                        help="Chat template format (default: phi4)")

    parser.add_argument("--eval-split", type=float, default=0.1,
                        help="Fraction of data for evaluation (default: 0.1)")

    parser.add_argument("--max-rows", type=int, default=0,
                        help="Max training rows (0 = all)")

    parser.add_argument("--dry-run", action="store_true",
                        help="Load dataset and print stats, don't train")

    return parser.parse_args()


def get_config(args) -> TrainingConfig:
    if args.preset == "phi4_mini":
        cfg = MemoryEfficientConfigPresets.phi4_mini()
    elif args.preset == "qwen3_embedding":
        cfg = MemoryEfficientConfigPresets.qwen3_embedding()
    elif args.preset == "colab_free_tier":
        cfg = MemoryEfficientConfigPresets.colab_free_tier()
    elif args.preset == "colab_pro":
        cfg = MemoryEfficientConfigPresets.colab_pro()
    elif args.preset == "local_gpu":
        cfg = MemoryEfficientConfigPresets.local_gpu()
    elif args.preset == "high_end_gpu":
        cfg = MemoryEfficientConfigPresets.high_end_gpu()
    elif args.preset == "cpu_only":
        cfg = MemoryEfficientConfigPresets.cpu_only()
    else:
        cfg = TrainingConfig()

    if args.model:
        cfg.model_path = args.model
    if args.epochs:
        cfg.num_train_epochs = args.epochs
    if args.batch_size:
        cfg.per_device_train_batch_size = args.batch_size
    if args.lora_rank:
        cfg.lora_r = args.lora_rank
    if args.max_seq_length:
        cfg.max_seq_length = args.max_seq_length
    if args.learning_rate:
        cfg.learning_rate = args.learning_rate

    cfg.output_dir = args.output
    cfg.dataset_path = args.data
    cfg.csv_keep_think = args.keep_think
    cfg.csv_template = args.template
    cfg.csv_train_split = 1.0 - args.eval_split

    return cfg


def main():
    args = parse_args()
    cfg = get_config(args)

    if not Path(args.data).exists():
        print(f"Error: Data not found: {args.data}")
        sys.exit(1)

    fmt = "CSV" if open(args.data, encoding="utf-8").readline().startswith("system,") else "JSONL"

    print("=" * 60)
    print("Training Pipeline")
    print("=" * 60)
    print(f"  Model:     {cfg.model_path}")
    print(f"  Data:      {args.data} ({fmt})")
    print(f"  Preset:    {args.preset}")
    print(f"  Output:    {cfg.output_dir}")
    print(f"  LoRA rank: {cfg.lora_r}")
    print(f"  Epochs:    {cfg.num_train_epochs}")
    print(f"  Keep CoT:  {args.keep_think}")
    print()

    # Count rows (stdlib only)
    if fmt == "CSV":
        import csv
        with open(args.data, encoding="utf-8") as f:
            total_rows = sum(1 for _ in csv.DictReader(f))
    else:
        total_rows = sum(1 for line in open(args.data, encoding="utf-8") if line.strip())
    print(f"  Rows:      {total_rows}")
    if args.max_rows:
        total_rows = min(total_rows, args.max_rows)
    train_count = int(total_rows * cfg.csv_train_split)
    eval_count = total_rows - train_count
    print(f"  Train/eval: {train_count}/{eval_count}")
    print()

    if args.dry_run:
        print("Dry run complete. Remove --dry-run to start training.")
        return

    train_ds, eval_ds = MemoryEfficientTrainer.load_dataset(
        path=args.data,
        template=args.template,
        keep_think=args.keep_think,
        train_split=cfg.csv_train_split,
        max_rows=args.max_rows,
    )

    trainer = MemoryEfficientTrainer(config=cfg)
    trainer.load_model()
    trainer.train(dataset=train_ds, eval_dataset=eval_ds, output_dir=cfg.output_dir)

    print(f"\nDone! Model saved to: {cfg.output_dir}")
    print(f"Next step: python scripts/quantize_gguf.py --adapter {cfg.output_dir} --output ./model-q4_k_m.gguf")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
