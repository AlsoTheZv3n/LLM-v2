#!/usr/bin/env python3
"""
finetune.py — Fine-Tune LLaMA Base Model on Chat/Instruction Data
==================================================================
Loads the pre-trained checkpoint and fine-tunes on instruction datasets
using the chat format: <|system|>, <|user|>, <|assistant|>.

Usage:
    python training/finetune.py                           # default: Claude-Reasoning dataset
    python training/finetune.py --dataset openhermes      # OpenHermes-2.5
    python training/finetune.py --epochs 3 --lr 1e-5      # custom hyperparams
"""

import argparse
import json
import os
import sys
import time
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Import model architecture from train.py
sys.path.insert(0, str(PROJECT_ROOT / "training"))
from train import LLaMAModel, get_tokenizer, get_device

# ─────────────────────────────────────────────────────────────
#  Fine-Tune Datasets
# ─────────────────────────────────────────────────────────────
FT_DATASETS = {
    "claude-reasoning": {
        "hf_path": "TeichAI/Claude-Opus-4.6-Reasoning-887x",
        "split": "train",
        "format": "messages",  # list of {role, content}
        "description": "887 Claude Opus reasoning examples with <think> blocks",
    },
    "openhermes": {
        "hf_path": "teknium/OpenHermes-2.5",
        "split": "train",
        "format": "conversations",  # list of {from, value}
        "description": "1M+ instruction-response pairs",
    },
}

# ─────────────────────────────────────────────────────────────
#  Chat Format
# ─────────────────────────────────────────────────────────────
SPECIAL_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "end": "<|end|>",
}

def format_chat(messages: list[dict], encode_fn) -> list[int]:
    """Convert chat messages to token IDs."""
    text = ""
    for msg in messages:
        role = msg.get("role", msg.get("from", "user"))
        content = msg.get("content", msg.get("value", ""))
        if role in ("system", "human", "user"):
            text += f"{SPECIAL_TOKENS['system'] if role == 'system' else SPECIAL_TOKENS['user']}\n{content}\n{SPECIAL_TOKENS['end']}\n"
        elif role in ("assistant", "gpt"):
            text += f"{SPECIAL_TOKENS['assistant']}\n{content}\n{SPECIAL_TOKENS['end']}\n"
    return encode_fn(text)


def load_ft_dataset(name: str, encode_fn, max_samples: int = 0) -> list[list[int]]:
    """Load and tokenize a fine-tuning dataset."""
    from datasets import load_dataset

    cfg = FT_DATASETS[name]
    print(f"  Loading {cfg['hf_path']}...")
    ds = load_dataset(cfg["hf_path"], split=cfg["split"])
    print(f"  Found {len(ds):,} samples")

    tokenized = []
    for i, item in enumerate(ds):
        if max_samples > 0 and i >= max_samples:
            break

        if cfg["format"] == "messages":
            messages = item.get("messages", [])
        elif cfg["format"] == "conversations":
            convs = item.get("conversations", [])
            messages = [{"role": c.get("from", "user"), "content": c.get("value", "")} for c in convs]
        else:
            continue

        if not messages:
            continue

        tokens = format_chat(messages, encode_fn)
        if len(tokens) > 4:  # skip empty
            tokenized.append(tokens)

    print(f"  Tokenized {len(tokenized):,} samples")
    return tokenized


# ─────────────────────────────────────────────────────────────
#  Fine-Tune Training Loop
# ─────────────────────────────────────────────────────────────
def finetune(args):
    device = get_device()
    encode_fn, decode_fn, vocab_size = get_tokenizer()

    # Find best checkpoint
    best_ckpt = None
    best_step = -1
    runs_dir = PROJECT_ROOT / "runs"
    for ckpt_file in runs_dir.glob("*/checkpoint.pt"):
        try:
            ckpt_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            if ckpt_data.get("arch") == "llama" and ckpt_data.get("step", 0) > best_step:
                best_step = ckpt_data["step"]
                best_ckpt = (ckpt_file, ckpt_data)
        except Exception:
            continue

    if not best_ckpt:
        print("ERROR: No LLaMA checkpoint found in runs/")
        sys.exit(1)

    ckpt_file, ckpt = best_ckpt
    cfg = ckpt["cfg"]
    cfg["V"] = vocab_size
    print(f"  Base model: {ckpt_file.parent.name} (step {best_step})")
    print(f"  Config: L={cfg['L']} H={cfg['H']} C={cfg['C']} T={cfg['T']}")

    # Build model
    print(f"  Loading model...")
    model = LLaMAModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  GPU memory: {torch.cuda.memory_allocated() // 1024**2} MB" if device.type == "cuda" else "")

    # Load dataset
    print(f"\n  Dataset: {args.dataset}")
    samples = load_ft_dataset(args.dataset, encode_fn, max_samples=args.max_samples)
    if not samples:
        print("ERROR: No samples loaded")
        sys.exit(1)

    # Split train/eval
    random.shuffle(samples)
    n_eval = max(1, len(samples) // 10)
    eval_samples = samples[:n_eval]
    train_samples = samples[n_eval:]
    print(f"  Train: {len(train_samples):,} | Eval: {len(eval_samples):,}")

    # Optimizer — lower LR for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    # Output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "runs" / f"ft_{args.dataset}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    T = cfg["T"]
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print(f"\n  Fine-tuning for {args.epochs} epochs, lr={args.lr}")
    print(f"  Output: {out_dir}\n")

    model.train()
    global_step = 0
    best_eval_loss = float("inf")

    for epoch in range(args.epochs):
        random.shuffle(train_samples)
        epoch_loss = 0
        n_batches = 0

        for i, tokens in enumerate(train_samples):
            # Truncate or pad to T+1
            if len(tokens) > T + 1:
                tokens = tokens[:T + 1]
            elif len(tokens) < T + 1:
                tokens = tokens + [0] * (T + 1 - len(tokens))

            inp = torch.tensor(tokens[:T], dtype=torch.long, device=device).unsqueeze(0)
            tgt = torch.tensor(tokens[1:T + 1], dtype=torch.long, device=device).unsqueeze(0)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(inp)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))

            scaler.scale(loss).backward()

            # Gradient accumulation (effective batch size = accum_steps)
            if (i + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item()
            n_batches += 1

            if (i + 1) % 50 == 0:
                avg = epoch_loss / n_batches
                pct = (i + 1) / len(train_samples) * 100
                print(f"  Epoch {epoch+1}/{args.epochs} | {pct:5.1f}% | step {global_step} | loss {avg:.4f}")

        # Epoch eval
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for tokens in eval_samples:
                if len(tokens) > T + 1:
                    tokens = tokens[:T + 1]
                elif len(tokens) < T + 1:
                    tokens = tokens + [0] * (T + 1 - len(tokens))
                inp = torch.tensor(tokens[:T], dtype=torch.long, device=device).unsqueeze(0)
                tgt = torch.tensor(tokens[1:T + 1], dtype=torch.long, device=device).unsqueeze(0)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(inp)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
                eval_loss += loss.item()
        eval_loss /= len(eval_samples)
        model.train()

        print(f"\n  Epoch {epoch+1} done | train_loss={epoch_loss/n_batches:.4f} | eval_loss={eval_loss:.4f} | ppl={2**eval_loss:.1f}")

        # Save best
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save({
                "model": model.state_dict(),
                "step": global_step,
                "cfg": cfg,
                "arch": "llama",
                "finetune": args.dataset,
                "eval_loss": eval_loss,
            }, out_dir / "checkpoint_best.pt")
            print(f"  Saved best checkpoint (eval_loss={eval_loss:.4f})")

    # Save final
    torch.save({
        "model": model.state_dict(),
        "step": global_step,
        "cfg": cfg,
        "arch": "llama",
        "finetune": args.dataset,
        "eval_loss": eval_loss,
    }, out_dir / "checkpoint_final.pt")

    # Save config
    with open(out_dir / "finetune_config.json", "w") as f:
        json.dump({
            "dataset": args.dataset,
            "base_checkpoint": str(ckpt_file),
            "base_step": best_step,
            "epochs": args.epochs,
            "lr": args.lr,
            "final_eval_loss": eval_loss,
            "best_eval_loss": best_eval_loss,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Fine-tuning complete!")
    print(f"  Best eval loss: {best_eval_loss:.4f}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    return out_dir


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune LLaMA base model")
    p.add_argument("--dataset", default="claude-reasoning", choices=FT_DATASETS.keys())
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--accum-steps", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--max-samples", type=int, default=0, help="Max samples (0=all)")
    args = p.parse_args()

    finetune(args)
