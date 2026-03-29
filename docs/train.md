# Training Guide

## Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Start training (defaults: local config, wikitext, wandb, auto-resume)
python training/train.py
```

That's it. Training starts on your local GPU with sensible defaults.

## Configuration

### Config Profiles

| Config | For | GPU Requirement | Training Time |
|--------|-----|----------------|---------------|
| `demo` | Quick test | Any GPU | ~1 min |
| `small` | Fast experiment | 4GB+ VRAM | ~30 min |
| `medium` | Serious experiment | 8GB+ VRAM | ~2 hours |
| `large` | Full training | 12GB+ VRAM | ~5 hours |
| `local` | **Default** — full local | 12GB+ VRAM | ~3-5 hours |
| `1b` | Cloud training | 24GB+ VRAM (A100) | ~60 hours |

### Override defaults

```bash
python training/train.py --config medium          # smaller model
python training/train.py --dataset openwebtext    # different dataset
python training/train.py --batch 2                # reduce batch size (saves VRAM)
python training/train.py --no-plot                # no matplotlib dashboard
```

## Datasets

| Dataset | Tokens | Source | Best For |
|---------|--------|--------|----------|
| `wikitext` | ~118M | WikiText-103 | Default, good quality |
| `openwebtext` | ~9B | OpenWebText | More diverse text |
| `fineweb-edu` | ~250B | FineWeb-Edu | Large-scale pre-training |
| `tinystories` | ~500M | TinyStories | Simple language testing |
| `shakespeare` | ~1M | Tiny Shakespeare | Quick debugging |

Pre-tokenized `.bin` files (from `tokenize_data.py`) can be used with:
```bash
python training/train.py --data-dir /path/to/tokenized/
```

## Resume Training

Resume is **enabled by default**. When you start training, it automatically finds the best checkpoint (highest step) and continues from there.

```bash
# These are equivalent:
python training/train.py
python training/train.py --resume
```

Checkpoints are saved every 500 steps to `runs/<config>_<timestamp>/checkpoint.pt`.

On interrupt (Ctrl+C), the current state is saved before exit.

### Fresh start (ignore old checkpoints)

```bash
python training/train.py --no-resume
```

## Monitoring

### W&B Dashboard

Enabled by default. View at: https://wandb.ai/

Tracks: loss, learning rate, gradient norm, tokens/sec, perplexity, samples.

### Matplotlib Dashboard

Opens automatically (unless `--no-plot`). Shows 6 real-time plots:
- Loss (raw + smoothed)
- Eval Perplexity
- Learning Rate schedule
- Gradient Norm
- Tokens/sec throughput

### Telegram Bot

```bash
python agents/telegram_bot.py
```

Commands:
- `/status` — Training progress, loss, ETA
- `/training` — Detailed training metrics with progress bar
- `/logs` — Last N log entries
- `/budget` — GCP spend tracking
- `/help` — All commands

## Cloud Training

### Step 1: Choose a provider

Check `.env` for provider credentials:
- **GCP**: `GCP_PROJECT_ID`, `GCP_SERVICE_ACCOUNT_KEY_PATH`
- **Lambda**: `LAMBDA_CLOUD_API_KEY`
- **AWS**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

### Step 2: Launch via orchestrator

```bash
python cloud/orchestrator.py start-pipeline --config 1b
```

This automatically:
1. Creates a GPU VM
2. Uploads training code
3. Tokenizes datasets
4. Starts training in tmux
5. Monitors budget + progress
6. Downloads weights when done
7. Destroys VM

### Step 3: Monitor remotely

```bash
# Telegram bot
python agents/telegram_bot.py

# Or check W&B dashboard
# https://wandb.ai/
```

## Tokenization (for large-scale training)

For pre-training on 300B+ tokens, pre-tokenize datasets first:

```bash
# Tokenize individual datasets
python training/tokenize_data.py --dataset wikipedia --output /data/tokenized/
python training/tokenize_data.py --dataset fineweb-edu --output /data/tokenized/

# Tokenize all datasets
python training/tokenize_data.py --all --output /data/tokenized/

# Check progress
python training/tokenize_data.py --status /data/tokenized/

# Validate shards
python training/tokenize_data.py --validate /data/tokenized/
```

Then train with pre-tokenized data:
```bash
python training/train.py --config 1b --data-dir /data/tokenized/
```

## Model Output

After training completes:

| File | Description |
|------|-------------|
| `runs/<name>/checkpoint.pt` | Model weights + optimizer state |
| `runs/<name>/training_log.jsonl` | Per-step metrics |
| `runs/<name>/eval_results.jsonl` | Evaluation checkpoints |
| `runs/<name>/samples.txt` | Generated text samples |
| `runs/<name>/summary.json` | Final metrics summary |
| `runs/<name>/dashboard_final.png` | Training curves |

### Evaluate a run

```bash
python training/eval.py                      # analyze latest run
python training/eval.py --run runs/<name>    # specific run
python training/eval.py --all                # compare all runs
```

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | beta1=0.9, beta2=0.95 |
| Learning Rate | 3e-4 | Peak, after warmup |
| LR Min | 3e-5 | Cosine decay target |
| Warmup Steps | 100 | Linear warmup |
| Weight Decay | 0.1 | Applied to 2D params only |
| Gradient Clipping | 1.0 | Global norm |
| Precision | fp16 | Via torch.amp |

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size: `--batch 1`
- Use gradient checkpointing (enabled in 1b config)
- Use a smaller config: `--config medium`

### Slow training on Windows
- `torch.compile` is disabled on Windows (known issues)
- Expected speed: ~80-120 tok/s on RTX 3080 Ti without compile

### Resume loads wrong checkpoint
- Resume finds the checkpoint with the highest step count
- Old GPT-2 checkpoints (without `arch: llama`) are automatically skipped
