#!/usr/bin/env python3
"""Read tokenizer ids and a few weight values from the local HF cache only."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open

CACHE_ROOT = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"


def find_snapshot(cache_root: Path) -> Path:
    snaps = sorted((cache_root / "snapshots").glob("*"))
    snaps = [p for p in snaps if (p / "model.safetensors").exists() and (p / "vocab.json").exists()]
    if not snaps:
        raise FileNotFoundError(f"No snapshot with model.safetensors + vocab.json under {cache_root}")
    return snaps[-1]


def load_vocab(vocab_path: Path) -> dict[str, int]:
    return json.loads(vocab_path.read_text())


def print_token_mapping(vocab: dict[str, int], vocab_path: Path, limit: int) -> None:
    real = vocab_path.resolve()
    items = sorted(vocab.items(), key=lambda kv: kv[1])
    print(f"\n=== token id mapping ===")
    print(f"file (snapshot) : {vocab_path}")
    print(f"file (blob)     : {real}")
    print(f"entries         : {len(vocab)} (showing {min(limit, len(items))})")
    print("token_id | token")
    for token, tid in items[:limit]:
        print(f"{tid:8d} | {token!r}")


def print_weights(safetensors_path: Path, max_tensors: int, n_values: int) -> None:
    real = safetensors_path.resolve()
    print("\n=== model weights ===")
    print(f"file (snapshot) : {safetensors_path}")
    print(f"file (blob)     : {real}")
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"tensor count    : {len(keys)}")
        for name in keys[:max_tensors]:
            t = f.get_tensor(name)
            vals = t.flatten()[:n_values].tolist()
            print(f"\n{name}")
            print(f"  file          : {real}")
            print(f"  shape={tuple(t.shape)} dtype={t.dtype}")
            print(f"  values[:{n_values}]={vals}")
        if len(keys) > max_tensors:
            print(f"\n... {len(keys) - max_tensors} more tensors not printed")


def lookup(vocab: dict[str, int], vocab_path: Path, ids: list[int]) -> None:
    inv = {i: t for t, i in vocab.items()}
    print("\n=== lookup ===")
    print(f"file (snapshot) : {vocab_path}")
    print(f"file (blob)     : {vocab_path.resolve()}")
    for i in ids:
        print(f"{i} -> {inv.get(i, '<missing>')!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--vocab-limit", type=int, default=40)
    p.add_argument("--max-tensors", type=int, default=8)
    p.add_argument("--n-values", type=int, default=6)
    p.add_argument("--ids", type=int, nargs="*", default=[0, 11, 81917, 24231, 70403])
    args = p.parse_args()

    snap = find_snapshot(args.cache_root)
    print(f"cache snapshot (local only): {snap}")

    vocab_path = snap / "vocab.json"
    weights_path = snap / "model.safetensors"
    vocab = load_vocab(vocab_path)
    print_token_mapping(vocab, vocab_path, args.vocab_limit)
    lookup(vocab, vocab_path, args.ids)
    print_weights(weights_path, args.max_tensors, args.n_values)


if __name__ == "__main__":
    main()
