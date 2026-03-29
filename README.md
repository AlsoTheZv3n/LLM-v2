# LLM Pre-Training Project — 1B Parameter Model

Train a **1 billion parameter LLaMA-style language model** from scratch, then fine-tune specialized variants locally.

Built entirely with Claude Code as an AI-assisted development experiment.

## What This Is

A complete LLM training infrastructure that includes:
- **LLaMA-style transformer** (RMSNorm, RoPE, SwiGLU, Flash Attention) implemented from scratch in PyTorch
- **Multi-dataset training** with weighted sampling across 7+ sources
- **Multi-cloud deployment** (GCP, AWS, Lambda Cloud) with auto-provisioning
- **Agent system** for monitoring, documentation, security, and orchestration
- **Telegram bot** for remote training control
- **W&B integration** for experiment tracking

## Architecture

```
                    Telegram Bot
                         |
                    Orchestrator
                    /    |    \
            Monitor   DocAgent  Security
                         |
                Cloud Provider (GCP / AWS / Lambda)
                         |
                   GPU Instance
                    /         \
            Tokenizer      train.py (LLaMA-style)
```

## Project Structure

```
LLM-v2/
├── training/                     # Core ML (runs locally or in cloud)
│   ├── train.py                  # LLaMA-style model + training loop
│   ├── eval.py                   # Evaluation, benchmarks, analysis
│   └── tokenize_data.py          # Dataset tokenization (tiktoken GPT-2 BPE)
│
├── cloud/                        # Cloud deployment & orchestration
│   ├── providers/                # Multi-cloud support
│   │   ├── gcp.py                # Google Cloud (A100/L4 Spot VMs)
│   │   ├── aws.py                # Amazon Web Services (placeholder)
│   │   └── lambda_cloud.py       # Lambda Cloud (A100/H100 on-demand)
│   ├── orchestrator.py           # Pipeline state machine + preemption recovery
│   └── check_gpu.sh              # GPU healthcheck script
│
├── agents/                       # Monitoring & automation
│   ├── monitoring_agent.py       # Cost tracking, training alerts, budget gate
│   ├── doc_agent.py              # Notion training journal updates
│   ├── security_agent.py         # Credential scanning, VM security
│   └── telegram_bot.py           # Telegram command interface
│
├── docs/                         # Documentation
│   ├── architecture.md           # System architecture deep-dive
│   └── train.md                  # Training guide & troubleshooting
│
├── env.example                   # Template for .env configuration
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Docker compose config
└── claude_desktop_config.json    # MCP server registration for Claude Desktop
```

## Model Architecture

**LLaMA-style transformer** with modern improvements over GPT-2:

| Component | Implementation |
|-----------|---------------|
| Normalization | RMSNorm (faster than LayerNorm) |
| Position Encoding | RoPE (Rotary Position Embeddings) |
| Activation | SwiGLU (gate * up projection) |
| Attention | Flash Attention via `scaled_dot_product_attention` |
| Bias | No bias terms (fewer params, better efficiency) |
| Weight Tying | lm_head shares weights with token embedding |

### Configs

| Config | Params | Layers | Heads | Dim  | Context | Steps  |
|--------|--------|--------|-------|------|---------|--------|
| demo   | ~7M    | 2      | 4     | 128  | 64      | 500    |
| small  | ~19M   | 6      | 8     | 512  | 256     | 10K    |
| medium | ~117M  | 12     | 12    | 768  | 512     | 20K    |
| large  | ~345M  | 24     | 16    | 1024 | 1024    | 50K    |
| local  | ~355M  | 24     | 16    | 1024 | 512     | 100K   |
| 1b     | ~1B    | 24     | 16    | 2048 | 2048    | 300K   |

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USER/LLM-v2.git
cd LLM-v2
cp env.example .env
# Edit .env with your API keys (HF_TOKEN, WANDB_API_KEY, etc.)
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 3. Train

```bash
# Quick demo (~1 min)
python training/train.py --config demo --no-plot

# Full local training on WikiText (~3-5h on RTX 3080 Ti)
python training/train.py

# Mixed dataset training (OpenWebText + AutoMathText + WikiText + HackerNews)
python training/train.py --dataset mix --no-plot
```

### 4. Evaluate

```bash
python training/eval.py              # analyze latest run
python training/eval.py --all        # compare all runs
```

## Datasets

### Pre-Training

| Dataset | Tokens | Type | Source |
|---------|--------|------|--------|
| OpenWebText | ~9B | Web text | `Skylion007/openwebtext` |
| AutoMathText-V2 | ~5B | Math/Reasoning | `OpenSQZ/AutoMathText-V2` |
| WikiText-103 | ~118M | Wikipedia | `wikitext` |
| Hacker News | ~1B | Tech discussions | `open-index/hacker-news` |
| FineWeb-Edu | ~250B | Educational web | `HuggingFaceFW/fineweb-edu` |
| The Stack v2 | ~900B | Source code | `bigcode/the-stack-v2` (gated) |
| Cosmopedia | ~25B | Synthetic textbooks | `HuggingFaceTB/cosmopedia` |

Use `--dataset mix` for weighted multi-dataset training (40% OpenWebText, 20% AutoMathText, 20% WikiText, 20% HackerNews).

### Fine-Tuning (Phase 4)

| Variant | Dataset | Purpose |
|---------|---------|---------|
| General Chat | OpenHermes-2.5 + UltraChat + Claude-Opus-Reasoning | Chat assistant with reasoning |
| Code Assistant | The Stack v2 + CodeAlpaca | Python, TypeScript, C++ |
| Deutsch / Swiss | German Wikipedia + mC4 DE | DE/CH language |
| Business / IT | Dolly-15k + synthetic | IT consulting |
| RAG-optimized | Synthetic Q&A from docs | Document Q&A |

## Cloud Training

Supports multiple cloud providers for large-scale training:

```bash
# Full automated pipeline (provisions VM, trains, downloads weights, cleans up)
python cloud/orchestrator.py start-pipeline --config 1b

# Check status
python cloud/orchestrator.py status
```

### Supported Providers

| Provider | GPU | Status |
|----------|-----|--------|
| GCP | A100/L4 Spot VMs | Implemented |
| Lambda Cloud | A100/H100 | Implemented |
| AWS | p4d/g5 instances | Placeholder |

## Remote Control (Telegram)

```bash
python agents/telegram_bot.py
```

Commands: `/status`, `/training`, `/budget`, `/logs`, `/start`, `/stop`, `/kill`, `/security`, `/help`

## Agent System

| Agent | Purpose | Integrations |
|-------|---------|-------------|
| Monitoring | Budget tracking, loss alerts, stall detection | W&B, Telegram, GCP Billing |
| Documentation | Auto-update training journal | Notion API |
| Security | Credential scanning, .gitignore enforcement | Local filesystem |
| Orchestrator | Pipeline state machine, preemption recovery | All providers + agents |

## W&B Dashboard

Training metrics are logged to Weights & Biases for remote monitoring:
- Loss curves (raw + smoothed)
- Learning rate schedule
- Gradient norms
- Evaluation perplexity
- Generated text samples

## Training Results (WikiText-103, 355M params)

| Metric | Value |
|--------|-------|
| Final Loss | 4.97 |
| Best Eval PPL | 71.32 |
| Total Tokens | 45.5M |
| Training Time | ~3.4h (RTX 3080 Ti) |
| Architecture | LLaMA-style (RMSNorm, RoPE, SwiGLU) |

## Cost Estimate

| Phase | Cost |
|-------|------|
| Local training (RTX 3080 Ti) | $0 |
| Cloud pre-training ~60h A100 | ~$87 |
| Cloud disk 600GB | ~$15 |
| Fine-tuning (local) | $0 |
| **Total Budget** | **$200 CHF** |

## Built With

- **PyTorch** — Model & training
- **tiktoken** — GPT-2 BPE tokenizer
- **HuggingFace Datasets** — Data streaming
- **Weights & Biases** — Experiment tracking
- **Notion API** — Training journal
- **Telegram Bot API** — Remote control
- **Google Cloud Compute** — Cloud GPU VMs
- **Claude Code** — AI-assisted development

## License

MIT
