#!/usr/bin/env python3
"""Adapt OpenVul rejection-SFT rows to Qwen3.5-4B chat messages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

THINK_OPEN = "<think>"
THINK_CLOSE = "</" + "think>"


def _as_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or content).strip()
    return str(content).strip()


def _norm_messages(items) -> list[dict]:
    if items is None:
        return []
    if isinstance(items, dict):
        items = [items]
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        text = _as_text(item.get("content"))
        if role in {"system", "user", "assistant"} and text:
            out.append({"role": role, "content": text})
    return out


def _normalize_assistant(text: str) -> str:
    text = text.strip()
    if THINK_OPEN not in text:
        text = f"{THINK_OPEN}\n{text}\n{THINK_CLOSE}"
    return text.strip()


def row_to_messages(row: dict) -> list[dict] | None:
    prompt = _norm_messages(row.get("prompt"))
    completion = _norm_messages(row.get("completion"))
    if not completion and row.get("completion"):
        raw = row["completion"]
        if isinstance(raw, str):
            completion = [{"role": "assistant", "content": raw}]
    system = next((m for m in prompt if m["role"] == "system"), None)
    users = [m for m in prompt if m["role"] == "user"]
    assistant = next((m for m in completion if m["role"] == "assistant"), None)
    if assistant is None and completion:
        assistant = {"role": "assistant", "content": completion[-1]["content"]}
    if not users or assistant is None:
        return None
    assistant = {"role": "assistant", "content": _normalize_assistant(assistant["content"])}
    messages = []
    if system:
        messages.append(system)
    messages.extend(users)
    messages.append(assistant)
    if not any(m["role"] == "user" for m in messages):
        return None
    if not messages[-1]["content"]:
        return None
    return messages


def count_chars(messages: list[dict]) -> int:
    return sum(len(m["content"]) for m in messages)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--min-assistant-chars", type=int, default=32)
    parser.add_argument("--max-chars", type=int, default=200000)
    args = parser.parse_args()

    src = Path(args.src)
    raw = load_from_disk(str(src))
    if isinstance(raw, DatasetDict):
        split_name = "train" if "train" in raw else list(raw.keys())[0]
        ds = raw[split_name]
    else:
        ds = raw
        split_name = "train"

    kept = []
    dropped = {"format": 0, "short": 0, "long": 0, "tokens": 0}
    for row in ds:
        messages = row_to_messages(row)
        if messages is None:
            dropped["format"] += 1
            continue
        assistant = messages[-1]["content"]
        if len(assistant) < args.min_assistant_chars:
            dropped["short"] += 1
            continue
        if count_chars(messages) > args.max_chars:
            dropped["long"] += 1
            continue
        kept.append({"key": row.get("key", ""), "messages": messages})

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    except Exception as exc:
        print(f"WARN: tokenizer load failed ({exc}); skip token-length filter")

    if tok is not None:
        filtered = []
        for item in kept:
            try:
                text = tok.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
                n = len(tok(text, add_special_tokens=False).input_ids)
            except Exception:
                dropped["tokens"] += 1
                continue
            if n > args.max_length:
                dropped["tokens"] += 1
                continue
            item["n_tokens"] = n
            filtered.append(item)
        kept = filtered

    out = Path(args.dst)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "train.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    dset = Dataset.from_list(kept)
    DatasetDict({"train": dset}).save_to_disk(str(out / "hf"))
    stats = {
        "src_rows": len(ds),
        "kept": len(kept),
        "dropped": dropped,
        "src_split": split_name,
        "model_chat_template": args.tokenizer,
        "max_length": args.max_length,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    if kept:
        print("example roles:", [m["role"] for m in kept[0]["messages"]])
        print("assistant head:", kept[0]["messages"][-1]["content"][:180].replace("\n", " "))


if __name__ == "__main__":
    main()
