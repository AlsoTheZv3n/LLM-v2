#!/usr/bin/env python3
"""
tokenize_data.py — Tokenization Pipeline for LLM Pre-Training
==============================================================
Streams datasets from HuggingFace, tokenizes with tiktoken GPT-2 BPE,
writes binary .bin shards for efficient training data loading.

Usage:
    python tokenize_data.py --dataset fineweb-edu --output /data/tokenized/
    python tokenize_data.py --dataset wikipedia --output /data/tokenized/
    python tokenize_data.py --all --output /data/tokenized/
    python tokenize_data.py --validate /data/tokenized/
    python tokenize_data.py --status /data/tokenized/
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# ─────────────────────────────────────────────────────────────
#  Dataset Registry
# ─────────────────────────────────────────────────────────────
DATASETS = {
    # General text
    "fineweb-edu": {
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "est_tokens": 250_000_000_000,
        "description": "FineWeb-Edu (~500GB, high-quality educational web text)",
    },
    "wikipedia": {
        "hf_path": "wikipedia",
        "hf_config": "20220301.en",
        "split": "train",
        "text_field": "text",
        "streaming": False,
        "est_tokens": 3_500_000_000,
        "description": "English Wikipedia (~20GB)",
    },
    "cosmopedia": {
        "hf_path": "HuggingFaceTB/cosmopedia",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "est_tokens": 25_000_000_000,
        "description": "Cosmopedia (~100GB, synthetic textbooks & stories)",
    },
    "openwebtext": {
        "hf_path": "Skylion007/openwebtext",
        "split": "train",
        "text_field": "text",
        "streaming": False,
        "est_tokens": 9_000_000_000,
        "description": "OpenWebText (~38GB)",
    },
    # Math & Reasoning
    "automathtext": {
        "hf_path": "OpenSQZ/AutoMathText-V2",
        "hf_config": "automathtext-v2-ultra",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "est_tokens": 5_000_000_000,
        "description": "AutoMathText-V2 (high-quality math & reasoning text)",
    },
    # Tech & Discussions
    "hackernews": {
        "hf_path": "open-index/hacker-news",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "est_tokens": 1_000_000_000,
        "description": "Hacker News (tech discussions, startups, programming)",
    },
    # Code
    "the-stack": {
        "hf_path": "bigcode/the-stack-v2",
        "split": "train",
        "text_field": "content",
        "streaming": True,
        "est_tokens": 900_000_000_000,
        "description": "The Stack v2 (900B tokens, 600+ programming languages, gated)",
    },
}

# Shard config
SHARD_SIZE = 100_000_000  # 100M tokens per shard (~200MB as uint16)
HEADER_SIZE = 256  # bytes
HEADER_MAGIC = b"TKN1"
HEADER_VERSION = 1

# ─────────────────────────────────────────────────────────────
#  Tokenizer
# ─────────────────────────────────────────────────────────────
def get_tokenizer():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # <|endoftext|> = 50256
    return enc, eot

# ─────────────────────────────────────────────────────────────
#  Shard Writer
# ─────────────────────────────────────────────────────────────
class ShardWriter:
    """Buffers tokens and writes .bin shards."""

    def __init__(self, output_dir: Path, prefix: str):
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_idx = 0
        self.buffer = np.empty(SHARD_SIZE, dtype=np.uint16)
        self.buf_pos = 0
        self.total_tokens = 0
        self.total_docs = 0
        output_dir.mkdir(parents=True, exist_ok=True)

    def add_tokens(self, tokens: list[int]):
        """Add tokens to buffer, flush shard when full."""
        i = 0
        while i < len(tokens):
            space = SHARD_SIZE - self.buf_pos
            chunk = min(space, len(tokens) - i)
            self.buffer[self.buf_pos:self.buf_pos + chunk] = tokens[i:i + chunk]
            self.buf_pos += chunk
            i += chunk
            if self.buf_pos >= SHARD_SIZE:
                self._flush()

    def _flush(self):
        """Write current buffer to a .bin shard file."""
        if self.buf_pos == 0:
            return
        fname = self.output_dir / f"{self.prefix}_{self.shard_idx:05d}.bin"
        n_tokens = self.buf_pos

        # Write header + data
        with open(fname, "wb") as f:
            # Header: magic(4) + version(4) + n_tokens(8) + dtype(4) + vocab_size(4) + padding
            header = bytearray(HEADER_SIZE)
            header[:4] = HEADER_MAGIC
            struct.pack_into("<I", header, 4, HEADER_VERSION)
            struct.pack_into("<Q", header, 8, n_tokens)
            struct.pack_into("<I", header, 16, 2)  # dtype: 2 = uint16
            struct.pack_into("<I", header, 20, 50257)  # vocab size
            f.write(header)
            f.write(self.buffer[:n_tokens].tobytes())

        size_mb = fname.stat().st_size / 1024**2
        print(f"  [shard] {fname.name}: {n_tokens:,} tokens ({size_mb:.1f} MB)")

        self.total_tokens += n_tokens
        self.shard_idx += 1
        self.buf_pos = 0

    def finalize(self):
        """Flush remaining buffer."""
        self._flush()
        return self.total_tokens


# ─────────────────────────────────────────────────────────────
#  Progress Tracker
# ─────────────────────────────────────────────────────────────
class ProgressTracker:
    def __init__(self, est_tokens: int):
        self.est_tokens = est_tokens
        self.tokens = 0
        self.docs = 0
        self.t_start = time.time()
        self.t_last_print = 0
        self.print_interval = 5.0  # seconds

    def update(self, n_tokens: int, n_docs: int = 1):
        self.tokens += n_tokens
        self.docs += n_docs
        now = time.time()
        if now - self.t_last_print >= self.print_interval:
            self._print_status()
            self.t_last_print = now

    def _print_status(self):
        elapsed = time.time() - self.t_start
        tps = self.tokens / max(elapsed, 1)
        pct = self.tokens / max(self.est_tokens, 1) * 100
        eta_s = (self.est_tokens - self.tokens) / max(tps, 1)
        if eta_s < 3600:
            eta = f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s"
        else:
            eta = f"{eta_s / 3600:.1f}h"
        print(
            f"\r  {self.tokens / 1e9:.2f}B / ~{self.est_tokens / 1e9:.0f}B tokens  "
            f"({pct:.1f}%)  |  {self.docs:,} docs  |  "
            f"{tps / 1e6:.2f}M tok/s  |  ETA {eta}   ",
            end="", flush=True,
        )

    def finish(self):
        elapsed = time.time() - self.t_start
        tps = self.tokens / max(elapsed, 1)
        print(f"\n  Done: {self.tokens:,} tokens from {self.docs:,} docs in {elapsed:.0f}s ({tps/1e6:.2f}M tok/s)")


# ─────────────────────────────────────────────────────────────
#  Manifest (resume support)
# ─────────────────────────────────────────────────────────────
def load_manifest(output_dir: Path, prefix: str) -> dict:
    mf = output_dir / f"{prefix}_manifest.json"
    if mf.exists():
        return json.loads(mf.read_text())
    return {"next_shard": 0, "total_tokens": 0, "total_docs": 0, "completed": False}

def save_manifest(output_dir: Path, prefix: str, data: dict):
    mf = output_dir / f"{prefix}_manifest.json"
    mf.write_text(json.dumps(data, indent=2))


# ─────────────────────────────────────────────────────────────
#  Tokenize a dataset
# ─────────────────────────────────────────────────────────────
def tokenize_dataset(name: str, output_dir: Path, max_tokens: int = 0, no_resume: bool = False):
    if name not in DATASETS:
        print(f"  ERROR: Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")
        return

    ds_cfg = DATASETS[name]
    print(f"\n{'='*60}")
    print(f"  Tokenizing: {name}")
    print(f"  {ds_cfg['description']}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # Check resume
    manifest = load_manifest(output_dir, name)
    if manifest.get("completed") and not no_resume:
        print(f"  Already completed ({manifest['total_tokens']:,} tokens). Use --no-resume to redo.")
        return

    # Init tokenizer
    enc, eot = get_tokenizer()
    print(f"  Tokenizer: tiktoken gpt2 (vocab={enc.n_vocab})")

    # Init writer
    writer = ShardWriter(output_dir, name)
    if not no_resume and manifest["next_shard"] > 0:
        writer.shard_idx = manifest["next_shard"]
        writer.total_tokens = manifest["total_tokens"]
        writer.total_docs = manifest["total_docs"]
        print(f"  Resuming from shard {writer.shard_idx} ({writer.total_tokens:,} tokens)")

    tracker = ProgressTracker(ds_cfg["est_tokens"])
    if writer.total_tokens > 0:
        tracker.tokens = writer.total_tokens
        tracker.docs = writer.total_docs

    # Load dataset
    from datasets import load_dataset
    hf_token = os.getenv("HF_TOKEN")
    load_kwargs = {
        "path": ds_cfg["hf_path"],
        "split": ds_cfg["split"],
        "streaming": ds_cfg.get("streaming", True),
    }
    if "hf_config" in ds_cfg:
        load_kwargs["name"] = ds_cfg["hf_config"]
    if hf_token:
        load_kwargs["token"] = hf_token

    print(f"  Loading {ds_cfg['hf_path']}...")
    ds = load_dataset(**load_kwargs)

    # Skip already-processed docs if resuming
    skip_docs = manifest["total_docs"] if not no_resume else 0
    skipped = 0

    try:
        for item in ds:
            if skipped < skip_docs:
                skipped += 1
                continue

            text = item.get(ds_cfg["text_field"], "")
            if not text or len(text) < 10:
                continue

            # Tokenize: eot + document tokens
            tokens = [eot] + enc.encode(text)
            writer.add_tokens(tokens)
            tracker.update(len(tokens))

            # Check max tokens limit
            if max_tokens > 0 and tracker.tokens >= max_tokens:
                print(f"\n  Reached max_tokens limit ({max_tokens:,})")
                break

    except KeyboardInterrupt:
        print("\n  Interrupted — saving progress...")

    # Finalize
    total = writer.finalize()
    tracker.finish()

    # Save manifest
    save_manifest(output_dir, name, {
        "next_shard": writer.shard_idx,
        "total_tokens": writer.total_tokens,
        "total_docs": tracker.docs,
        "completed": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    print(f"  Manifest saved: {name}_manifest.json")
    return writer.total_tokens


# ─────────────────────────────────────────────────────────────
#  Validate shards
# ─────────────────────────────────────────────────────────────
def validate_shards(data_dir: Path):
    print(f"\n{'='*60}")
    print(f"  Validating shards in {data_dir}")
    print(f"{'='*60}\n")

    bin_files = sorted(data_dir.glob("*.bin"))
    if not bin_files:
        print("  No .bin files found.")
        return

    enc, eot = get_tokenizer()
    total_tokens = 0
    errors = 0

    for f in bin_files:
        fsize = f.stat().st_size
        with open(f, "rb") as fp:
            header = fp.read(HEADER_SIZE)

            # Check magic
            magic = header[:4]
            if magic != HEADER_MAGIC:
                print(f"  {RED}[FAIL]{NC} {f.name}: bad magic {magic}")
                errors += 1
                continue

            version = struct.unpack_from("<I", header, 4)[0]
            n_tokens = struct.unpack_from("<Q", header, 8)[0]
            dtype_size = struct.unpack_from("<I", header, 16)[0]
            vocab = struct.unpack_from("<I", header, 20)[0]

            # Check file size
            expected_size = HEADER_SIZE + n_tokens * dtype_size
            if fsize != expected_size:
                print(f"  [FAIL] {f.name}: size mismatch (expected {expected_size}, got {fsize})")
                errors += 1
                continue

            # Spot check: read first 100 tokens
            data = np.frombuffer(fp.read(min(200, n_tokens * 2)), dtype=np.uint16)
            if np.any(data >= vocab):
                print(f"  [FAIL] {f.name}: token values exceed vocab size {vocab}")
                errors += 1
                continue

            # Decode test
            try:
                sample = enc.decode(data[:50].tolist())
                sample_preview = sample[:60].replace("\n", "\\n")
            except Exception:
                sample_preview = "<decode error>"

            total_tokens += n_tokens
            print(f"  [OK]   {f.name}: {n_tokens:,} tokens (v{version}, vocab={vocab}) | \"{sample_preview}...\"")

    print(f"\n  Total: {total_tokens:,} tokens ({total_tokens/1e9:.2f}B) across {len(bin_files)} shards")
    if errors:
        print(f"  {errors} errors found!")
    else:
        print(f"  All shards valid.")


# ─────────────────────────────────────────────────────────────
#  Status
# ─────────────────────────────────────────────────────────────
def show_status(data_dir: Path):
    print(f"\n{'='*60}")
    print(f"  Tokenization Status: {data_dir}")
    print(f"{'='*60}\n")

    total_tokens = 0
    total_size = 0

    for name in DATASETS:
        manifest = load_manifest(data_dir, name)
        shards = sorted(data_dir.glob(f"{name}_*.bin"))
        n_shards = len(shards)
        size = sum(f.stat().st_size for f in shards)
        tokens = manifest.get("total_tokens", 0)
        status = "DONE" if manifest.get("completed") else "INCOMPLETE"

        total_tokens += tokens
        total_size += size

        print(f"  {name:20s}  {status:12s}  {tokens/1e9:8.2f}B tokens  {n_shards:4d} shards  {size/1e9:.1f} GB")

    print(f"\n  {'TOTAL':20s}  {'':12s}  {total_tokens/1e9:8.2f}B tokens  {'':4s}        {total_size/1e9:.1f} GB")
    print(f"  Target: ~300B tokens\n")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Tokenization Pipeline for LLM Pre-Training")
    p.add_argument("--dataset", choices=DATASETS.keys(), help="Dataset to tokenize")
    p.add_argument("--all", action="store_true", help="Tokenize all datasets")
    p.add_argument("--output", default="./data/tokenized", help="Output directory")
    p.add_argument("--validate", metavar="DIR", help="Validate .bin shards in directory")
    p.add_argument("--status", metavar="DIR", help="Show tokenization status")
    p.add_argument("--max-tokens", type=int, default=0, help="Max tokens per dataset (0=unlimited)")
    p.add_argument("--no-resume", action="store_true", help="Start from scratch, ignore manifests")
    args = p.parse_args()

    if args.validate:
        validate_shards(Path(args.validate))
        return

    if args.status:
        show_status(Path(args.status))
        return

    output_dir = Path(args.output)

    if args.all:
        for name in DATASETS:
            tokenize_dataset(name, output_dir, args.max_tokens, args.no_resume)
    elif args.dataset:
        tokenize_dataset(args.dataset, output_dir, args.max_tokens, args.no_resume)
    else:
        p.print_help()
        print("\n  Available datasets:")
        for name, cfg in DATASETS.items():
            print(f"    {name:20s} — {cfg['description']}")


if __name__ == "__main__":
    main()
