#!/usr/bin/env python3
"""
Convert any data source to HuggingFace dataset format for QLoRA training.

Supported inputs (auto-detected):
  CSV        – columns: system, user, assistant
  Chat JSONL – {"messages": [{"role": "...", "content": "..."}]}
  Rollout    – Codex CLI agent event stream (rollout-*.jsonl)
  Trace      – {"task":"...", "messages": [...]} (traces.jsonl)

Usage:
    python scripts/prepare_dataset.py datas/Claudecode.csv --output ./data/training
    python scripts/prepare_dataset.py datas/traces.jsonl --output ./data/training
    python scripts/prepare_dataset.py datas/rollout-*.jsonl --output ./data/training
    python scripts/prepare_dataset.py datas/ --output ./data/training --recursive
"""

import argparse
import csv
import json
import os
import random
import re
import glob
from collections import defaultdict
from pathlib import Path


CHAT_TEMPLATES = {
    "phi3": {
        "system": "<|system|>\n{text}<|end|>\n",
        "user": "<|user|>\n{text}<|end|>\n",
        "assistant": "<|assistant|>\n{text}<|end|>\n",
    },
    "phi4": {
        "system": "<|system|>\n{text}<|end|>\n",
        "user": "<|user|>\n{text}<|end|>\n",
        "assistant": "<|assistant|>\n{text}<|end|>\n",
    },
    "llama3": {
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{text}<|eot_id|>\n",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>\n",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{text}<|eot_id|>\n",
    },
    "chatml": {
        "system": "<|im_start|>system\n{text}<|im_end|>\n",
        "user": "<|im_start|>user\n{text}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{text}<|im_end|>\n",
    },
}


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def format_messages(messages: list, template: str, keep_think: bool) -> str:
    fmt = CHAT_TEMPLATES[template]
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if role == "system":
            parts.append(fmt["system"].format(text=content))
        elif role == "user":
            parts.append(fmt["user"].format(text=content))
        elif role == "assistant":
            if not keep_think:
                content = strip_think(content)
            parts.append(fmt["assistant"].format(text=content))
    return "".join(parts)


def detect_format(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        return "empty"
    if first_line.startswith("system,"):
        return "csv"
    try:
        obj = json.loads(first_line)
    except json.JSONDecodeError:
        return "unknown"
    if isinstance(obj, dict):
        if "messages" in obj:
            if any(k in obj for k in ["task", "category", "lang", "trace_format"]):
                return "trace"
            return "chat_jsonl"
        if "type" in obj and "payload" in obj:
            return "rollout"
    return "jsonl_other"


def extract_csv(path: str, template: str, keep_think: bool, max_rows: int) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            system = row.get("system", "").strip()
            user = row.get("user", "").strip()
            assistant = row.get("assistant", "").strip()
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            if user:
                msgs.append({"role": "user", "content": user})
            if assistant:
                if not keep_think:
                    assistant = strip_think(assistant)
                msgs.append({"role": "assistant", "content": assistant})
            text = format_messages(msgs, template, keep_think=True)
            if text.strip():
                rows.append({"text": text})
    return rows


def extract_chat_jsonl(path: str, template: str, keep_think: bool, max_rows: int) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            text = format_messages(msgs, template, keep_think)
            if text.strip():
                row = {"text": text}
                for extra in ["task", "lang", "category"]:
                    if extra in obj:
                        row[extra] = obj[extra]
                rows.append(row)
    return rows


def extract_trace(path: str, template: str, keep_think: bool, max_rows: int) -> list:
    return extract_chat_jsonl(path, template, keep_think, max_rows)


def extract_rollout(path: str, template: str, keep_think: bool, max_rows: int) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    pending_user = None
    pending_user_turn = None
    conversations = []

    for r in records:
        if max_rows and len(conversations) >= max_rows:
            break

        if r["type"] == "event_msg":
            p = r["payload"]
            if p["type"] == "user_message":
                pending_user = p.get("message", "").strip()
                pending_user_turn = p.get("turn_id", "direct")
            elif p["type"] == "agent_message":
                user_msg = pending_user if pending_user else ""
                assistant_msg = p.get("message", "").strip()
                if assistant_msg:
                    conversations.append({
                        "system": "",
                        "user": user_msg,
                        "assistant": assistant_msg,
                        "source": os.path.basename(path),
                    })
                pending_user = None

        elif r["type"] == "response_item":
            p = r["payload"]
            if p.get("role") == "assistant":
                content_blocks = p.get("content", [])
                text_parts = []
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") in ("output_text", "text"):
                        text_parts.append(block.get("text", ""))
                assistant_text = "".join(text_parts).strip()
                if assistant_text and pending_user:
                    conversations.append({
                        "system": "",
                        "user": pending_user,
                        "assistant": assistant_text,
                        "source": os.path.basename(path),
                    })
                    pending_user = None

    rows = []
    for conv in conversations:
        msgs = []
        if conv["system"]:
            msgs.append({"role": "system", "content": conv["system"]})
        if conv["user"]:
            msgs.append({"role": "user", "content": conv["user"]})
        if conv["assistant"]:
            content = conv["assistant"]
            if not keep_think:
                content = strip_think(content)
            msgs.append({"role": "assistant", "content": content})
        text = format_messages(msgs, template, keep_think=True)
        if text.strip():
            row = {"text": text}
            if "source" in conv:
                row["source"] = conv["source"]
            rows.append(row)

    return rows


EXTRACTORS = {
    "csv": extract_csv,
    "chat_jsonl": extract_chat_jsonl,
    "trace": extract_trace,
    "rollout": extract_rollout,
}

FORMAT_NAMES = {
    "csv": "CSV",
    "chat_jsonl": "Chat JSONL",
    "trace": "Trace JSONL",
    "rollout": "Rollout event log",
}


def process_path(
    path: str,
    template: str,
    keep_think: bool,
    max_rows: int,
    force_format: str = "",
) -> list:
    path = Path(path)
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return []

    fmt = force_format or detect_format(str(path))
    if fmt == "empty":
        print(f"  SKIP: {path} is empty")
        return []
    if fmt == "unknown":
        print(f"  SKIP: {path} unknown format")
        return []

    extractor = EXTRACTORS.get(fmt)
    if not extractor:
        print(f"  SKIP: {path} no extractor for format '{fmt}'")
        return []

    try:
        rows = extractor(str(path), template, keep_think, max_rows)
        print(f"  {FORMAT_NAMES.get(fmt, fmt):14s} {path.name}: {len(rows)} conversations")
        return rows
    except Exception as e:
        print(f"  ERROR: {path.name}: {e}")
        return []


def process_inputs(
    inputs: list,
    template: str,
    keep_think: bool,
    max_rows: int,
    recursive: bool,
    force_format: str = "",
) -> list:
    all_rows = []
    seen = set()

    for inp in inputs:
        inp = str(inp).strip()
        if os.path.isfile(inp):
            rows = process_path(inp, template, keep_think, max_rows, force_format)
            all_rows.extend(rows)
        elif os.path.isdir(inp):
            pattern = "**/*" if recursive else "*"
            for f in sorted(glob.glob(os.path.join(inp, pattern), recursive=recursive)):
                if os.path.isfile(f) and not f.endswith(".gguf"):
                    rows = process_path(f, template, keep_think, max_rows, force_format)
                    all_rows.extend(rows)
        else:
            for f in sorted(glob.glob(inp, recursive=recursive)):
                if os.path.isfile(f):
                    rows = process_path(f, template, keep_think, max_rows, force_format)
                    all_rows.extend(rows)

    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Prepare data for QLoRA training")
    parser.add_argument("inputs", nargs="+", help="Paths: files, directories, or globs")
    parser.add_argument("--output", "-o", default="./data/training", help="Output directory")
    parser.add_argument("--format", "-f", default="phi4", choices=list(CHAT_TEMPLATES.keys()))
    parser.add_argument("--keep-think", action="store_true", default=True)
    parser.add_argument("--no-think", action="store_false", dest="keep_think")
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan directories recursively")
    parser.add_argument("--force-format", default="", choices=[""] + list(EXTRACTORS.keys()))
    args = parser.parse_args()

    print("=" * 60)
    print("Dataset Preparation")
    print("=" * 60)

    all_rows = process_inputs(
        args.inputs, args.format, args.keep_think,
        args.max_rows, args.recursive, args.force_format,
    )

    if not all_rows:
        print("\nNo data extracted!")
        return

    random.shuffle(all_rows)
    split_idx = int(len(all_rows) * args.train_split)
    train = all_rows[:split_idx]
    eval_data = all_rows[split_idx:]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    sample_path = output_dir / "sample.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for r in eval_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(all_rows[0]["text"])

    print(f"\nDone: {len(train)} train + {len(eval_data)} eval samples")
    print(f"  Train: {train_path}")
    print(f"  Eval:  {eval_path}")
    print(f"  Sample: {sample_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
