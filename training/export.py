#!/usr/bin/env python3
"""
export.py — Export LLaMA model to HuggingFace format + GGUF
=============================================================
Converts our custom checkpoint to HuggingFace Transformers format,
then optionally converts to GGUF for llama.cpp.

Usage:
    python training/export.py                              # export latest checkpoint
    python training/export.py --ckpt runs/ft_.../best.pt   # specific checkpoint
    python training/export.py --gguf                        # also convert to GGUF
    python training/export.py --push                        # upload to HuggingFace
"""

import argparse
import json
import os
import sys
import shutil
from pathlib import Path

import torch

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

sys.path.insert(0, str(PROJECT_ROOT / "training"))
from train import LLaMAModel, get_tokenizer


def find_best_checkpoint() -> Path:
    """Find checkpoint with highest step across all runs."""
    runs_dir = PROJECT_ROOT / "runs"
    best_path = None
    best_step = -1

    for ckpt_file in runs_dir.glob("*/checkpoint*.pt"):
        try:
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            if ckpt.get("arch") == "llama" and ckpt.get("step", 0) > best_step:
                best_step = ckpt["step"]
                best_path = ckpt_file
        except Exception:
            continue

    if not best_path:
        print("ERROR: No LLaMA checkpoint found")
        sys.exit(1)

    print(f"  Best checkpoint: {best_path} (step {best_step})")
    return best_path


def export_to_hf(ckpt_path: Path, output_dir: Path):
    """Convert custom checkpoint to HuggingFace Transformers format."""
    print(f"\n  Exporting to HuggingFace format...")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    state = ckpt["model"]

    # Map our weight names to HF LLaMA format
    hf_state = {}

    # Token embeddings
    hf_state["model.embed_tokens.weight"] = state["tok_emb.weight"]

    # Blocks
    for i in range(cfg["L"]):
        prefix_src = f"blocks.{i}"
        prefix_dst = f"model.layers.{i}"

        # Attention norm
        hf_state[f"{prefix_dst}.input_layernorm.weight"] = state[f"{prefix_src}.attn_norm.weight"]

        # Attention projections
        hf_state[f"{prefix_dst}.self_attn.q_proj.weight"] = state[f"{prefix_src}.attn.q_proj.weight"]
        hf_state[f"{prefix_dst}.self_attn.k_proj.weight"] = state[f"{prefix_src}.attn.k_proj.weight"]
        hf_state[f"{prefix_dst}.self_attn.v_proj.weight"] = state[f"{prefix_src}.attn.v_proj.weight"]
        hf_state[f"{prefix_dst}.self_attn.o_proj.weight"] = state[f"{prefix_src}.attn.o_proj.weight"]

        # FFN norm
        hf_state[f"{prefix_dst}.post_attention_layernorm.weight"] = state[f"{prefix_src}.ffn_norm.weight"]

        # MLP (SwiGLU)
        hf_state[f"{prefix_dst}.mlp.gate_proj.weight"] = state[f"{prefix_src}.mlp.gate_proj.weight"]
        hf_state[f"{prefix_dst}.mlp.up_proj.weight"] = state[f"{prefix_src}.mlp.up_proj.weight"]
        hf_state[f"{prefix_dst}.mlp.down_proj.weight"] = state[f"{prefix_src}.mlp.down_proj.weight"]

    # Final norm
    hf_state["model.norm.weight"] = state["norm.weight"]

    # LM head (weight-tied with embeddings)
    hf_state["lm_head.weight"] = state["tok_emb.weight"]

    # Compute hidden size for SwiGLU
    hidden = hf_state[f"model.layers.0.mlp.gate_proj.weight"].shape[0]

    # Save weights
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(hf_state, output_dir / "pytorch_model.bin")

    # Config
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": cfg["C"],
        "intermediate_size": hidden,
        "num_attention_heads": cfg["H"],
        "num_hidden_layers": cfg["L"],
        "num_key_value_heads": cfg["H"],
        "max_position_embeddings": cfg["T"],
        "vocab_size": cfg["V"],
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "hidden_act": "silu",
        "bos_token_id": 0,
        "eos_token_id": 0,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # Tokenizer config (tiktoken GPT-2)
    tokenizer_config = {
        "model_type": "gpt2",
        "tokenizer_class": "GPT2Tokenizer",
    }
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Generation config
    gen_config = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "do_sample": True,
    }
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)

    # Model card
    finetune_info = ckpt.get("finetune", "base")
    model_card = f"""---
license: mit
tags:
  - llama
  - pre-trained
  - from-scratch
  - pytorch
---

# LLaMA-355M Base Model

A 355M parameter LLaMA-style language model trained from scratch.

## Architecture
- **Type**: LLaMA-style Transformer
- **Parameters**: 355M
- **Layers**: {cfg['L']}
- **Heads**: {cfg['H']}
- **Hidden dim**: {cfg['C']}
- **Context**: {cfg['T']} tokens
- **Vocab**: {cfg['V']} (tiktoken GPT-2 BPE)

## Features
- RMSNorm (instead of LayerNorm)
- Rotary Position Embeddings (RoPE)
- SwiGLU activation
- Flash Attention
- No bias terms

## Training
- Pre-trained on a mix of OpenWebText, AutoMathText, WikiText, HackerNews
- Fine-tuned on: {finetune_info}
- Trained on RTX 3080 Ti

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("YOUR_USER/llama-355m")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(model_card)

    total_size = sum(f.stat().st_size for f in output_dir.iterdir()) / 1024**2
    print(f"  Exported to {output_dir} ({total_size:.0f} MB)")
    print(f"  Files: {[f.name for f in output_dir.iterdir()]}")
    return output_dir


def convert_to_gguf(hf_dir: Path, output_path: Path = None):
    """Convert HF model to GGUF format for llama.cpp."""
    print(f"\n  Converting to GGUF...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # Try using llama.cpp's convert script
        convert_script = shutil.which("convert-hf-to-gguf") or shutil.which("python3")
        if not convert_script:
            print("  llama.cpp convert script not found. Install with:")
            print("    pip install llama-cpp-python")
            print("  Or clone llama.cpp and use convert_hf_to_gguf.py")
            return None
    except ImportError:
        pass

    if output_path is None:
        output_path = hf_dir.parent / f"{hf_dir.name}.gguf"

    # Try using the llama-cpp-python package
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "llama_cpp.convert",
            str(hf_dir), "--outfile", str(output_path), "--outtype", "q4_0"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            size = output_path.stat().st_size / 1024**2
            print(f"  GGUF saved: {output_path} ({size:.0f} MB)")
            return output_path
        else:
            print(f"  GGUF conversion failed: {result.stderr[:200]}")
    except Exception as e:
        print(f"  GGUF conversion failed: {e}")

    print("  To convert manually:")
    print(f"    python llama.cpp/convert_hf_to_gguf.py {hf_dir} --outfile model.gguf --outtype q4_0")
    return None


def push_to_hub(hf_dir: Path, repo_name: str = None):
    """Upload model to HuggingFace Hub."""
    print(f"\n  Pushing to HuggingFace Hub...")

    hf_user = os.getenv("HF_USERNAME", "")
    if not repo_name:
        repo_name = f"{hf_user}/llama-355m" if hf_user else "llama-355m"

    try:
        from huggingface_hub import HfApi
        api = HfApi()

        # Create repo
        api.create_repo(repo_name, exist_ok=True, private=False)
        print(f"  Repo: https://huggingface.co/{repo_name}")

        # Upload all files
        api.upload_folder(folder_path=str(hf_dir), repo_id=repo_name)
        print(f"  Uploaded to https://huggingface.co/{repo_name}")
        return repo_name
    except Exception as e:
        print(f"  Push failed: {e}")
        print(f"  You can manually upload from: {hf_dir}")
        return None


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export LLaMA model")
    p.add_argument("--ckpt", default=None, help="Path to checkpoint (default: best)")
    p.add_argument("--output", default=None, help="Output directory")
    p.add_argument("--gguf", action="store_true", help="Also convert to GGUF")
    p.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    p.add_argument("--repo", default=None, help="HuggingFace repo name")
    args = p.parse_args()

    # Find checkpoint
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = find_best_checkpoint()

    # Output dir
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "exports" / "hf-llama-355m"

    # Export
    hf_dir = export_to_hf(ckpt_path, output_dir)

    # GGUF
    if args.gguf:
        convert_to_gguf(hf_dir)

    # Push
    if args.push:
        push_to_hub(hf_dir, args.repo)

    print("\nDone!")
