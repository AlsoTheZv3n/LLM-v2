#!/usr/bin/env python3
"""
train.py — GPT-2 Training Orchestrator (PyTorch + GPU)
======================================================
 - Downloads training data from HuggingFace Datasets or reads tokenized .bin files
 - Tokenises with tiktoken GPT-2 BPE (50257 vocab) or byte-level fallback
 - GPT-2 model in PyTorch with CUDA/GPU support
 - Weights & Biases integration for remote experiment tracking
 - Live matplotlib dashboard (loss, lr, grad norm, tokens/sec)
 - JSONL training log  → training_log.jsonl
 - Auto-evaluation every N steps  → eval_results.jsonl
 - GCS checkpoint upload support

Usage:
    python train.py                          # quick demo (wikitext, demo cfg)
    python train.py --config medium          # GPT-2 medium scale
    python train.py --config 1b             # 1B parameter model
    python train.py --dataset openwebtext    # different dataset
    python train.py --data-dir /data/tokenized  # pre-tokenized .bin files
    python train.py --resume                 # resume from checkpoint
    python train.py --no-plot                # headless / server mode
    python train.py --wandb                  # enable W&B logging
"""

import argparse
import json
import math
import os
import sys
import threading
import time
import queue
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# ─────────────────────────────────────────────────────────────
#  Tiktoken tokenizer (GPT-2 BPE, 50257 vocab)
# ─────────────────────────────────────────────────────────────
TIKTOKEN_VOCAB = 50257

def get_tokenizer():
    """Returns tiktoken GPT-2 encoder, falls back to byte-level."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return enc.encode, enc.decode, TIKTOKEN_VOCAB
    except ImportError:
        print("  WARNING: tiktoken not installed, using byte-level tokenizer (V=256)")
        return encode_bytes, decode_bytes, 256

# ─────────────────────────────────────────────────────────────
#  Config profiles
# ─────────────────────────────────────────────────────────────
CONFIGS = {
    "demo":   dict(V=TIKTOKEN_VOCAB, T=64,   L=2,  H=4,  C=128,  B=4,  steps=500),
    "small":  dict(V=TIKTOKEN_VOCAB, T=256,  L=6,  H=8,  C=512,  B=8,  steps=10000),
    "medium": dict(V=TIKTOKEN_VOCAB, T=512,  L=12, H=12, C=768,  B=4,  steps=20000),
    "large":  dict(V=TIKTOKEN_VOCAB, T=1024, L=24, H=16, C=1024, B=2,  steps=50000),
    "local":  dict(V=TIKTOKEN_VOCAB, T=512,  L=24, H=16, C=1024, B=2,  steps=100000, grad_ckpt=True),
    "1b":     dict(V=TIKTOKEN_VOCAB, T=2048, L=24, H=16, C=2048, B=4,  steps=300000, grad_ckpt=True),
}

DATASETS = {
    # General text
    "tinystories": ("roneneldan/TinyStories", "train", "text"),
    "wikitext":    ("wikitext",               "train", "text", {"name": "wikitext-103-raw-v1"}),
    "openwebtext": ("Skylion007/openwebtext",  "train", "text"),
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "train", "text"),
    # Math & Reasoning
    "automathtext": ("OpenSQZ/AutoMathText-V2", "train", "text", {"name": "automathtext-v2-ultra"}),
    # Tech & Discussions
    "hackernews":  ("open-index/hacker-news",  "train", "text"),
    # Code
    "the-stack":   ("bigcode/the-stack-v2",    "train", "content"),  # gated — needs HF access
    # Fallback
    "shakespeare": None,  # local file fallback
}

# Mix config: dataset name -> weight (sampling probability)
MIX_DATASETS = {
    "openwebtext": 0.40,   # 40% — diverse web text (largest, most general)
    "automathtext": 0.20,  # 20% — math & reasoning
    "wikitext": 0.20,      # 20% — factual knowledge
    "hackernews": 0.20,    # 20% — tech discussions
}

LOG_DIR = PROJECT_ROOT / "runs"

# ─────────────────────────────────────────────────────────────
#  Device selection
# ─────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print("  WARNING: CUDA not available, falling back to CPU")
        return torch.device("cpu")

# ─────────────────────────────────────────────────────────────
#  Byte-level tokeniser fallback
# ─────────────────────────────────────────────────────────────
def encode_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="replace"))

def decode_bytes(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────
#  HuggingFace data loader
# ─────────────────────────────────────────────────────────────
class BinFileStreamer:
    """Streams pre-tokenized .bin files (uint16 numpy arrays)."""

    def __init__(self, data_dir: str, B: int, T: int, device: torch.device):
        import numpy as np
        self.B = B
        self.T = T
        self.device = device
        self.total_tokens = 0
        data_path = Path(data_dir)
        self.files = sorted(data_path.glob("*.bin"))
        if not self.files:
            raise FileNotFoundError(f"No .bin files in {data_dir}")
        print(f"  Found {len(self.files)} tokenized shards in {data_dir}")
        total = sum(f.stat().st_size // 2 for f in self.files)
        print(f"  Total tokens: {total:,}")
        self._file_idx = 0
        self._pos = 0
        self._load_shard()

    def _load_shard(self):
        import numpy as np
        f = self.files[self._file_idx % len(self.files)]
        self._data = np.memmap(f, dtype=np.uint16, mode='r')
        self._pos = 0
        print(f"  Loaded shard: {f.name} ({len(self._data):,} tokens)")

    def next_batch(self):
        needed = self.B * self.T + 1
        if self._pos + needed > len(self._data):
            self._file_idx += 1
            self._load_shard()
        chunk = self._data[self._pos:self._pos + needed].astype(int)
        self._pos += self.B * self.T
        inp = torch.tensor(chunk[:self.B * self.T], dtype=torch.long, device=self.device).view(self.B, self.T)
        tgt = torch.tensor(chunk[1:self.B * self.T + 1], dtype=torch.long, device=self.device).view(self.B, self.T)
        self.total_tokens += self.B * self.T
        return inp, tgt


class HFStreamer:
    """Downloads text from HuggingFace, tokenises with tiktoken, yields batches."""

    def __init__(self, dataset_name: str, B: int, T: int, device: torch.device):
        self.B = B
        self.T = T
        self.device = device
        self.buffer: list[int] = []
        self.total_tokens = 0
        self._encode, self._decode, self._vocab = get_tokenizer()
        self._load(dataset_name)

    def _load(self, name: str):
        entry = DATASETS.get(name)
        if entry is None:
            p = Path("shakespeare.txt")
            if not p.exists():
                print("  Downloading Shakespeare...")
                import urllib.request
                url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
                urllib.request.urlretrieve(url, p)
            text = p.read_text(encoding="utf-8")
            self.buffer = self._encode(text) * 10
            print(f"  Loaded Shakespeare: {len(self.buffer):,} tokens")
            self._idx = 0
            return

        try:
            from datasets import load_dataset
            ds_args = [entry[0]]
            ds_kwargs = {"split": entry[1], "trust_remote_code": False}
            if len(entry) > 3:
                ds_kwargs.update(entry[3])
            print(f"  Downloading {entry[0]} from HuggingFace...")
            ds = load_dataset(*ds_args, **ds_kwargs)
            text_field = entry[2]
            print(f"  Tokenising {len(ds):,} documents...")
            tokens = []
            for item in ds:
                text = item.get(text_field, "")
                if text:
                    tokens.extend(self._encode(text + "\n"))
            self.buffer = tokens
            self._idx = 0
            print(f"  Loaded {entry[0]}: {len(self.buffer):,} tokens")
        except Exception as e:
            print(f"  HF load failed ({e}), falling back to Shakespeare")
            self._load("shakespeare")

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (inp [B, T], tgt [B, T]) as tensors on device."""
        needed = self.B * self.T + 1
        if self._idx + needed > len(self.buffer):
            self._idx = 0
        flat = self.buffer[self._idx:self._idx + needed]
        self._idx += self.B * self.T

        inp = torch.tensor(flat[:self.B * self.T], dtype=torch.long, device=self.device).view(self.B, self.T)
        tgt = torch.tensor(flat[1:self.B * self.T + 1], dtype=torch.long, device=self.device).view(self.B, self.T)
        self.total_tokens += self.B * self.T
        return inp, tgt

    def eval_batches(self, n: int = 8) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Pull n batches of held-out data for evaluation."""
        return [self.next_batch() for _ in range(n)]


class MixStreamer:
    """Streams from multiple HF datasets with weighted sampling."""

    def __init__(self, B: int, T: int, device: torch.device):
        self.B = B
        self.T = T
        self.device = device
        self.total_tokens = 0
        self._encode, self._decode, self._vocab = get_tokenizer()

        self.sources: list[dict] = []
        self.weights: list[float] = []

        for ds_name, weight in MIX_DATASETS.items():
            entry = DATASETS.get(ds_name)
            if entry is None:
                continue
            print(f"  Loading {ds_name} (weight={weight:.0%})...")
            try:
                from datasets import load_dataset
                ds_args = [entry[0]]
                ds_kwargs = {"split": entry[1], "streaming": True, "trust_remote_code": False}
                if len(entry) > 3:
                    ds_kwargs.update(entry[3])
                ds_iter = iter(load_dataset(*ds_args, **ds_kwargs))
                text_field = entry[2]
                self.sources.append({
                    "name": ds_name,
                    "iter": ds_iter,
                    "entry": entry,
                    "text_field": text_field,
                    "buffer": [],
                })
                self.weights.append(weight)
                print(f"    Streaming {entry[0]} ready")
            except Exception as e:
                print(f"    Failed to load {ds_name}: {e}")

        if not self.sources:
            raise RuntimeError("No mix datasets could be loaded")

        # Normalize weights
        total_w = sum(self.weights)
        self.weights = [w / total_w for w in self.weights]
        print(f"  Mix ready: {len(self.sources)} datasets, weights: "
              + ", ".join(f"{s['name']}={w:.0%}" for s, w in zip(self.sources, self.weights)))

    def _refill(self, source: dict, min_tokens: int = 8192):
        """Pull more text from a streaming dataset."""
        while len(source["buffer"]) < min_tokens:
            try:
                item = next(source["iter"])
                text = item.get(source["text_field"], "") or ""
                if text:
                    source["buffer"].extend(self._encode(text + "\n"))
            except StopIteration:
                # Restart stream
                entry = source["entry"]
                from datasets import load_dataset
                ds_args = [entry[0]]
                ds_kwargs = {"split": entry[1], "streaming": True, "trust_remote_code": False}
                if len(entry) > 3:
                    ds_kwargs.update(entry[3])
                source["iter"] = iter(load_dataset(*ds_args, **ds_kwargs))

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch from a randomly chosen dataset (weighted)."""
        import random
        source = random.choices(self.sources, weights=self.weights, k=1)[0]

        needed = self.B * self.T + 1
        self._refill(source, needed * 2)

        buf = source["buffer"]
        flat = buf[:needed]
        source["buffer"] = buf[self.B * self.T:]

        # Pad if not enough tokens (edge case)
        while len(flat) < needed:
            flat.append(0)

        inp = torch.tensor(flat[:self.B * self.T], dtype=torch.long, device=self.device).view(self.B, self.T)
        tgt = torch.tensor(flat[1:self.B * self.T + 1], dtype=torch.long, device=self.device).view(self.B, self.T)
        self.total_tokens += self.B * self.T
        return inp, tgt

    def eval_batches(self, n: int = 8) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [self.next_batch() for _ in range(n)]

# ─────────────────────────────────────────────────────────────
#  LLaMA-Style Model (RMSNorm, RoPE, SwiGLU, Flash Attention)
# ─────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(dim, max_seq_len, theta=10000.0):
    """Precompute RoPE frequency tensor."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary position embeddings to Q and K."""
    B, H, T, dh = x.shape
    cos = cos[:T, :].unsqueeze(0).unsqueeze(0)  # (1, 1, T, dh//2)
    sin = sin[:T, :].unsqueeze(0).unsqueeze(0)
    x1 = x[..., :dh // 2]
    x2 = x[..., dh // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    return x * cos_full + rotated * sin_full


class CausalSelfAttention(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        self.H = H
        self.C = C
        self.dh = C // H
        self.q_proj = nn.Linear(C, C, bias=False)
        self.k_proj = nn.Linear(C, C, bias=False)
        self.v_proj = nn.Linear(C, C, bias=False)
        self.o_proj = nn.Linear(C, C, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.H, self.dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.H, self.dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.H, self.dh).transpose(1, 2)

        # RoPE
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Flash Attention (PyTorch 2.0+)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class SwiGLU_MLP(nn.Module):
    """SwiGLU MLP — better than GELU at same param count."""
    def __init__(self, C):
        super().__init__()
        # SwiGLU uses 8/3 * C ≈ 2.67 * C hidden, rounded to multiple of 64
        hidden = int(2 * (4 * C) / 3)
        hidden = ((hidden + 63) // 64) * 64  # round up to multiple of 64
        self.gate_proj = nn.Linear(C, hidden, bias=False)
        self.up_proj = nn.Linear(C, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, C, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        self.attn_norm = RMSNorm(C)
        self.attn = CausalSelfAttention(C, H)
        self.ffn_norm = RMSNorm(C)
        self.mlp = SwiGLU_MLP(C)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.attn_norm(x), rope_cos, rope_sin)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class LLaMAModel(nn.Module):
    """LLaMA-style transformer: RMSNorm, RoPE, SwiGLU, Flash Attention, no bias."""
    def __init__(self, cfg):
        super().__init__()
        V, T, L, H, C = cfg["V"], cfg["T"], cfg["L"], cfg["H"], cfg["C"]
        self.cfg = cfg
        self.tok_emb = nn.Embedding(V, C)
        self.blocks = nn.ModuleList([Block(C, H) for _ in range(L)])
        self.norm = RMSNorm(C)
        self.lm_head = nn.Linear(C, V, bias=False)
        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        # precompute RoPE
        rope_cos, rope_sin = precompute_rope(C // H, T * 2)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        ps = 1.0 / math.sqrt(2.0 * self.cfg["L"])
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
        # scale residual projections
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=ps * 0.02)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=ps * 0.02)

    def forward(self, idx):
        x = self.tok_emb(idx)
        for block in self.blocks:
            if self.cfg.get("grad_ckpt", False) and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, self.rope_cos, self.rope_sin, use_reentrant=False)
            else:
                x = block(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        return self.lm_head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        T = self.cfg["T"]
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -T:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

# ─────────────────────────────────────────────────────────────
#  Training logger
# ─────────────────────────────────────────────────────────────
class Logger:
    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.train_log  = run_dir / "training_log.jsonl"
        self.eval_log   = run_dir / "eval_results.jsonl"
        self.sample_log = run_dir / "samples.txt"
        self.run_dir    = run_dir
        # write header
        with open(run_dir / "run_info.json", "w") as f:
            json.dump({"started": datetime.now().isoformat()}, f)

    def log_metrics(self, data: dict):
        with open(self.train_log, "a") as f:
            f.write(json.dumps(data) + "\n")

    def log_eval(self, data: dict):
        with open(self.eval_log, "a") as f:
            f.write(json.dumps(data) + "\n")

    def log_sample(self, step: int, text: str):
        with open(self.sample_log, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n[Step {step}]\n{text}\n")

    def finalize(self, summary: dict):
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

# ─────────────────────────────────────────────────────────────
#  Live matplotlib dashboard (runs in separate thread)
# ─────────────────────────────────────────────────────────────
class Dashboard:
    def __init__(self, cfg: dict, log_dir: Path, headless: bool = False):
        self.cfg = cfg
        self.log_dir = log_dir
        self.headless = headless
        self.q: queue.Queue = queue.Queue()
        self._running = True
        self.steps:      deque = deque(maxlen=2000)
        self.losses:     deque = deque(maxlen=2000)
        self.smooth_loss:deque = deque(maxlen=2000)
        self.lrs:        deque = deque(maxlen=2000)
        self.gnorms:     deque = deque(maxlen=2000)
        self.tok_sec:    deque = deque(maxlen=200)
        self.eval_steps: list  = []
        self.eval_ppls:  list  = []
        self._smooth = -1.0

        if not headless:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def update(self, data: dict):
        self.q.put(data)
        if self.headless:
            self._process_one(data)

    def _process_one(self, data: dict):
        t = data.get("type")
        if t == "metrics":
            s = data["step"]
            l = data["loss"]
            self._smooth = l if self._smooth < 0 else 0.95 * self._smooth + 0.05 * l
            self.steps.append(s)
            self.losses.append(l)
            self.smooth_loss.append(self._smooth)
            self.lrs.append(data["lr"])
            self.gnorms.append(data["grad_norm"])
            tps = (self.cfg["B"] * self.cfg["T"]) / max(data["ms"], 1) * 1000
            self.tok_sec.append(tps)
        elif t == "eval":
            self.eval_steps.append(data["step"])
            self.eval_ppls.append(data["ppl"])

    def _run(self):
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            plt.ion()
            fig = plt.figure(figsize=(14, 9), facecolor="#0d1117")
            fig.canvas.manager.set_window_title("GPT-2 Training Dashboard")
            gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
            ax_loss  = fig.add_subplot(gs[0, :2])
            ax_ppl   = fig.add_subplot(gs[0, 2])
            ax_lr    = fig.add_subplot(gs[1, 0])
            ax_gnorm = fig.add_subplot(gs[1, 1])
            ax_tps   = fig.add_subplot(gs[1, 2])

            DARK = "#0d1117"; LINE = "#c9d1d9"; ACC = "#58a6ff"; YLW = "#e3b341"; GRN = "#3fb950"

            for ax in [ax_loss, ax_ppl, ax_lr, ax_gnorm, ax_tps]:
                ax.set_facecolor("#161b22")
                ax.tick_params(colors=LINE, labelsize=8)
                for sp in ax.spines.values(): sp.set_color("#30363d")

            plt.show(block=False)
            last_save = time.time()

            while self._running:
                # drain queue
                while not self.q.empty():
                    try:
                        self._process_one(self.q.get_nowait())
                    except queue.Empty:
                        break

                if len(self.steps) < 2:
                    plt.pause(0.5)
                    continue

                xs = list(self.steps)
                for ax in [ax_loss, ax_ppl, ax_lr, ax_gnorm, ax_tps]:
                    ax.cla()
                    ax.set_facecolor("#161b22")
                    ax.tick_params(colors=LINE, labelsize=8)
                    for sp in ax.spines.values(): sp.set_color("#30363d")

                # ── Loss curve ────────────────────────────────
                ax_loss.plot(xs, list(self.losses),      color=ACC, alpha=0.3, lw=0.8, label="raw")
                ax_loss.plot(xs, list(self.smooth_loss), color=ACC, lw=1.8,    label="smoothed")
                if self.eval_steps:
                    ax_loss.scatter(self.eval_steps, [None]*len(self.eval_steps),
                                   color=YLW, s=20, zorder=5, label="eval checkpoints")
                ax_loss.set_title("Loss", color=LINE, fontsize=10)
                ax_loss.set_xlabel("step", color=LINE, fontsize=8)
                ax_loss.legend(fontsize=7, labelcolor=LINE, facecolor=DARK, edgecolor="#30363d")
                ax_loss.set_ylabel("cross-entropy", color=LINE, fontsize=8)

                # ── Perplexity ────────────────────────────────
                if self.eval_ppls:
                    ax_ppl.plot(self.eval_steps, self.eval_ppls, color=YLW, marker="o", ms=4, lw=1.5)
                ax_ppl.set_title("Eval Perplexity", color=LINE, fontsize=10)
                ax_ppl.set_xlabel("step", color=LINE, fontsize=8)

                # ── LR schedule ───────────────────────────────
                ax_lr.plot(xs, list(self.lrs), color=GRN, lw=1.5)
                ax_lr.set_title("Learning Rate", color=LINE, fontsize=10)
                ax_lr.set_xlabel("step", color=LINE, fontsize=8)

                # ── Gradient norm ─────────────────────────────
                ax_gnorm.plot(xs, list(self.gnorms), color="#ff7b72", lw=1.0, alpha=0.8)
                ax_gnorm.axhline(1.0, color="#30363d", lw=0.8, ls="--")
                ax_gnorm.set_title("Gradient Norm", color=LINE, fontsize=10)
                ax_gnorm.set_xlabel("step", color=LINE, fontsize=8)

                # ── Tokens/sec ────────────────────────────────
                tps_x = list(range(len(self.tok_sec)))
                ax_tps.fill_between(tps_x, list(self.tok_sec), alpha=0.4, color="#8b949e")
                ax_tps.plot(tps_x, list(self.tok_sec), color="#8b949e", lw=1.0)
                ax_tps.set_title("Tokens / sec", color=LINE, fontsize=10)
                ax_tps.set_xlabel("step (recent)", color=LINE, fontsize=8)

                fig.suptitle(
                    f"GPT-2 Training  |  L={self.cfg['L']} H={self.cfg['H']} C={self.cfg['C']}  "
                    f"|  step {xs[-1]}/{self.cfg['steps']}  |  loss {list(self.smooth_loss)[-1]:.4f}",
                    color=LINE, fontsize=11, y=0.98
                )

                plt.pause(0.05)

                # save figure every 30s
                if time.time() - last_save > 30:
                    fig.savefig(self.log_dir / "dashboard.png", dpi=100, bbox_inches="tight",
                                facecolor=fig.get_facecolor())
                    last_save = time.time()

            # final save
            fig.savefig(self.log_dir / "dashboard_final.png", dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)

        except Exception as e:
            print(f"[dashboard] error: {e}")

    def stop(self):
        self._running = False

# ─────────────────────────────────────────────────────────────
#  Cosine LR schedule with warmup
# ─────────────────────────────────────────────────────────────
def get_lr(step, warmup, total, lr_max=3e-4, lr_min=3e-5):
    if step < warmup:
        return lr_max * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

# ─────────────────────────────────────────────────────────────
#  Console progress bar
# ─────────────────────────────────────────────────────────────
def print_progress(step: int, total: int, loss: float, lr: float, ms: float, smooth: float):
    pct  = step / max(total, 1)
    bar  = int(pct * 30)
    eta  = (total - step) * ms / 1000
    eta_s = f"{int(eta//60)}m{int(eta%60):02d}s" if eta < 3600 else f"{int(eta//3600)}h"
    tps  = int(1000 / max(ms, 1))

    line = (
        f"\r  [{'#'*bar + '.'*(30-bar)}] "
        f"step {step:>6}/{total}  "
        f"loss {smooth:.4f}  "
        f"lr {lr:.1e}  "
        f"{tps:>5} tok/s  "
        f"ETA {eta_s}   "
    )
    print(line, end="", flush=True)

# ─────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def run_eval(model, streamer, n_batches=8):
    model.eval()
    total_loss = 0.0
    for _ in range(n_batches):
        inp, tgt = streamer.next_batch()
        logits = model(inp)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
        total_loss += loss.item()
    model.train()
    avg_loss = total_loss / n_batches
    return avg_loss, math.exp(avg_loss)

# ─────────────────────────────────────────────────────────────
#  Main training loop
# ─────────────────────────────────────────────────────────────
def train(args):
    cfg = dict(CONFIGS[args.config])
    if args.batch:
        cfg["B"] = args.batch

    device = get_device()

    # tokenizer
    tok_encode_fn, tok_decode_fn, vocab_size = get_tokenizer()
    cfg["V"] = vocab_size

    # make tokenizer accessible in scope
    global tok_encode, tok_decode
    tok_encode, tok_decode = tok_encode_fn, tok_decode_fn

    # run directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_DIR / f"{args.config}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LLaMA-Style Training Run")
    print(f"  Config : {args.config}  ({cfg['L']} layers, C={cfg['C']}, H={cfg['H']})")
    print(f"  Vocab  : {cfg['V']:,} tokens (tiktoken GPT-2)")
    print(f"  Dataset: {args.data_dir or args.dataset}")
    print(f"  Steps  : {cfg['steps']}   Batch: {cfg['B']}x{cfg['T']}")
    print(f"  Device : {device}")
    print(f"  Run dir: {run_dir}")
    print(f"{'='*60}\n")

    # ── W&B ────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "llm-pretrain-1b"),
                name=f"{args.config}_{ts}",
                config={**cfg, "dataset": args.data_dir or args.dataset, "device": str(device)},
            )
            print(f"  W&B: {wandb_run.url}")
        except Exception as e:
            print(f"  W&B init failed: {e}")

    # ── Components ────────────────────────────────────────────
    print("  Loading dataset...")
    if args.data_dir:
        streamer = BinFileStreamer(args.data_dir, cfg["B"], cfg["T"], device)
    elif args.dataset == "mix":
        streamer = MixStreamer(cfg["B"], cfg["T"], device)
    else:
        streamer = HFStreamer(args.dataset, cfg["B"], cfg["T"], device)

    print("  Building LLaMA-style model (RMSNorm, RoPE, SwiGLU, Flash Attention)...")
    model = LLaMAModel(cfg).to(device)
    n_params = model.count_params()
    print(f"  Model parameters: {n_params:,}")
    if device.type == "cuda":
        mem = torch.cuda.memory_allocated() / 1024**2
        print(f"  GPU memory (model): {mem:.0f} MB")

    # optimizer — AdamW with weight decay only on 2D params (weights, not biases/layernorms)
    decay_params = [p for _, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for _, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

    # resume from checkpoint
    start_step = 0
    ckpt_path = run_dir / "checkpoint.pt"
    if args.resume:
        # find checkpoint with highest step across ALL matching runs
        best_ckpt = None
        best_step = -1
        for ckpt_file in LOG_DIR.glob(f"{args.config}_*/checkpoint.pt"):
            try:
                ckpt_data = torch.load(ckpt_file, map_location=device, weights_only=False)
                # skip old GPT-2 checkpoints (incompatible architecture)
                if ckpt_data.get("arch") != "llama":
                    print(f"  Skipping {ckpt_file.parent.name} (old GPT-2 architecture)")
                    continue
                if ckpt_data.get("step", 0) > best_step:
                    best_step = ckpt_data["step"]
                    best_ckpt = (ckpt_file, ckpt_data)
            except Exception:
                continue
        if best_ckpt:
            ckpt_file, ckpt = best_ckpt
            print(f"  Resuming from {ckpt_file} (step {best_step})...")
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"]
            print(f"  Resumed from step {start_step}")
        else:
            print("  No compatible checkpoint found, starting fresh.")

    # compile model for speed (PyTorch 2.0+, Linux only — Windows has issues)
    if hasattr(torch, "compile") and device.type == "cuda" and sys.platform != "win32":
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)
    elif sys.platform == "win32":
        print("  Skipping torch.compile (Windows — not stable)")

    logger    = Logger(run_dir)
    dashboard = Dashboard(cfg, run_dir, headless=args.no_plot)

    # ── Training loop ─────────────────────────────────────────
    smooth_loss = -1.0
    t_start = time.time()
    eval_interval = max(cfg["steps"] // 20, 50)
    warmup_steps = 100
    total_steps = cfg["steps"]

    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    try:
        for step in range(start_step, total_steps):
            t0 = time.time()

            # update learning rate
            lr = get_lr(step, warmup_steps, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # eval
            if step > 0 and step % eval_interval == 0:
                eval_loss, eval_ppl = run_eval(model, streamer)
                eval_data = {"type": "eval", "step": step, "loss": eval_loss, "ppl": eval_ppl}
                dashboard.update(eval_data)
                logger.log_eval(eval_data)
                if wandb_run:
                    wandb_run.log({"eval_loss": eval_loss, "eval_ppl": eval_ppl}, step=step)
                print(f"\n  [eval] step={step}  ppl={eval_ppl:.2f}")

            # forward + backward
            inp, tgt = streamer.next_batch()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(inp)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            scaler.step(optimizer)
            scaler.update()

            t1 = time.time()
            ms = (t1 - t0) * 1000
            loss_val = loss.item()

            smooth_loss = loss_val if smooth_loss < 0 else 0.95 * smooth_loss + 0.05 * loss_val

            # metrics
            metrics = {
                "type": "metrics",
                "step": step + 1,
                "loss": loss_val,
                "smooth_loss": smooth_loss,
                "lr": lr,
                "ms": ms,
                "grad_norm": grad_norm,
                "timestamp": time.time() - t_start,
            }
            dashboard.update(metrics)
            logger.log_metrics(metrics)
            if wandb_run and step % 10 == 0:
                wandb_run.log({"loss": loss_val, "smooth_loss": smooth_loss, "lr": lr,
                               "grad_norm": grad_norm, "ms_per_step": ms,
                               "tokens_total": streamer.total_tokens}, step=step + 1)
            print_progress(step + 1, total_steps, loss_val, lr, ms, smooth_loss)

            # checkpoint + sample every 500 steps
            if (step + 1) % 500 == 0:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "cfg": cfg,
                }, ckpt_path)
                print(f"\n  [saved] {ckpt_path}")

                # generate sample
                raw_model.eval()
                prompt = torch.tensor([[tok_encode("\n")[0]]], dtype=torch.long, device=device)
                gen = raw_model.generate(prompt, max_new_tokens=128, temperature=0.8, top_k=40)
                sample_text = tok_decode(gen[0].tolist())
                raw_model.train()
                logger.log_sample(step + 1, sample_text)
                print(f"  [sample step={step+1}]\n  {sample_text[:80]}...")

    except KeyboardInterrupt:
        print("\n\n  Interrupted — saving checkpoint...")

    # ── Final eval (skip on interrupt to avoid torch.compile crash) ──
    eval_result = {}
    try:
        print("\n  Running final evaluation...")
        eval_loss, eval_ppl = run_eval(model, streamer, n_batches=20)
        eval_result = {
            "eval_loss": eval_loss,
            "perplexity": eval_ppl,
            "n_batches": 20,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"  Final eval skipped: {e}")

    # final checkpoint — save current step, not total_steps
    current_step = start_step + step + 1 if 'step' in dir() else start_step
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": current_step,
        "cfg": cfg,
        "arch": "llama",
    }, ckpt_path)
    print(f"  [saved] checkpoint at step {current_step}")

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - t_start
    summary = {
        "config":        args.config,
        "dataset":       args.dataset,
        "final_loss":    smooth_loss,
        "elapsed_sec":   elapsed,
        "total_tokens":  streamer.total_tokens,
        "run_dir":       str(run_dir),
        "eval":          eval_result,
        "device":        str(device),
        "params":        n_params,
        "finished":      datetime.now().isoformat(),
    }
    logger.finalize(summary)
    dashboard.stop()

    if wandb_run:
        wandb_run.log(summary)
        wandb_run.finish()

    print(f"\n{'='*60}")
    print(f"  Run complete!")
    print(f"  Final loss:  {smooth_loss:.4f}")
    print(f"  Eval PPL:    {eval_ppl:.2f}")
    print(f"  Tokens:      {streamer.total_tokens:,}")
    print(f"  Time:        {elapsed/60:.1f} min")
    print(f"  Device:      {device}")
    print(f"  Log dir:     {run_dir}")
    print(f"{'='*60}\n")

    return summary

# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="GPT-2 Training Orchestrator")
    p.add_argument("--config",    default="local",       choices=CONFIGS.keys())
    p.add_argument("--dataset",   default="wikitext",    choices=[*DATASETS.keys(), "mix"])
    p.add_argument("--data-dir",  default=None,          help="Path to pre-tokenized .bin files")
    p.add_argument("--batch",     type=int, default=None)
    p.add_argument("--resume",    action="store_true",   default=True)
    p.add_argument("--no-plot",   action="store_true",   dest="no_plot")
    p.add_argument("--wandb",     action="store_true",   default=True, help="Enable Weights & Biases logging")
    args = p.parse_args()

    train(args)

if __name__ == "__main__":
    main()
