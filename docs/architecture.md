# Architecture

## Overview

```
                    +------------------+
                    |   Telegram Bot   |  Remote control via commands
                    +--------+---------+
                             |
                    +--------v---------+
                    |   Orchestrator   |  Pipeline state machine
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v-----+  +-----v------+
     |  Monitoring |  |  Doc Agent |  |  Security  |
     |   Agent     |  |  (Notion)  |  |   Agent    |
     +--------+----+  +------+-----+  +-----+------+
              |              |              |
              +--------------+--------------+
                             |
                    +--------v---------+
                    |  Cloud Provider  |  GCP / AWS / Lambda
                    +--------+---------+
                             |
                    +--------v---------+
                    |   GPU Instance   |  A100 / L4 / T4
                    +--------+---------+
                             |
                    +--------v---------+
                    |    train.py      |  LLaMA-style model
                    +------------------+
```

## Project Structure

```
LLM-v2/
├── training/                 # Core ML code (runs anywhere)
│   ├── train.py              # LLaMA-style model + training loop
│   ├── eval.py               # Evaluation, benchmarks, analysis
│   └── tokenize_data.py      # Dataset tokenization (tiktoken GPT-2 BPE)
│
├── cloud/                    # Cloud deployment & orchestration
│   ├── providers/            # Cloud provider implementations
│   │   ├── gcp.py            # Google Cloud (A100/L4 Spot VMs)
│   │   ├── aws.py            # Amazon Web Services (placeholder)
│   │   └── lambda_cloud.py   # Lambda Cloud (A100/H100 on-demand)
│   ├── orchestrator.py       # Pipeline state machine + preemption recovery
│   └── check_gpu.sh          # GPU healthcheck script for cloud VMs
│
├── agents/                   # Monitoring & automation (local + cloud)
│   ├── monitoring_agent.py   # Cost tracking, training alerts, budget gate
│   ├── doc_agent.py          # Notion training journal updates
│   ├── security_agent.py     # Credential scanning, VM security
│   └── telegram_bot.py       # Telegram command interface
│
├── docs/                     # Documentation
│   ├── architecture.md       # This file
│   └── train.md              # Training guide
│
├── runs/                     # Training outputs (gitignored)
├── .env                      # API keys & config (gitignored)
├── gcp-agent-key.json        # GCP service account (gitignored)
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container definition
├── docker-compose.yml        # Docker compose config
├── CLAUDE.md                 # Project context for Claude Code
└── README.md                 # Project overview
```

## Model Architecture — LLaMA-Style Transformer

Replaces the original GPT-2 architecture with modern improvements:

| Component | GPT-2 (old) | LLaMA-Style (current) |
|-----------|-------------|----------------------|
| Normalization | LayerNorm | **RMSNorm** (10% faster) |
| Position Encoding | Learned embeddings | **RoPE** (better extrapolation) |
| Activation | GELU | **SwiGLU** (5% better loss) |
| Attention | Standard MHA | **Flash Attention** via `scaled_dot_product_attention` |
| Bias terms | Yes (all layers) | **No bias** (fewer params) |
| Weight tying | lm_head = wte | Same |

### Model Configs

| Config | Params | L | H | C | T | B | Steps |
|--------|--------|---|---|------|------|---|-------|
| demo | ~7M | 2 | 4 | 128 | 64 | 4 | 500 |
| small | ~19M | 6 | 8 | 512 | 256 | 8 | 10K |
| medium | ~117M | 12 | 12 | 768 | 512 | 4 | 20K |
| large | ~345M | 24 | 16 | 1024 | 1024 | 2 | 50K |
| local | ~355M | 24 | 16 | 1024 | 1024 | 4 | 100K |
| 1b | ~1B | 24 | 16 | 2048 | 2048 | 4 | 300K |

### Forward Pass

```
Input tokens [B, T]
    │
    ├─ Token Embedding (V, C)
    │
    ├─ RoPE precomputed (cos, sin)
    │
    ╞═ Block ×L ═══════════════════╡
    │  ├─ RMSNorm                  │
    │  ├─ Causal Self-Attention    │
    │  │  ├─ Q, K, V projections   │
    │  │  ├─ Apply RoPE to Q, K    │
    │  │  └─ Flash Attention        │
    │  ├─ Residual connection      │
    │  ├─ RMSNorm                  │
    │  ├─ SwiGLU MLP               │
    │  │  ├─ gate_proj (C → H)     │
    │  │  ├─ up_proj (C → H)       │
    │  │  ├─ SiLU(gate) * up       │
    │  │  └─ down_proj (H → C)     │
    │  └─ Residual connection      │
    ╞══════════════════════════════╡
    │
    ├─ RMSNorm
    └─ LM Head (C → V, weight-tied)
         │
    Output logits [B, T, V]
```

## Training Pipeline

### Local Training (on-prem)

```
python training/train.py
```

Runs directly on the local GPU (RTX 3080 Ti). Config, dataset, W&B, and resume are all defaulted.

### Cloud Training

```
python cloud/orchestrator.py start-pipeline
```

Full automated pipeline:
1. Create GPU VM (via selected provider)
2. GPU healthcheck
3. Upload training code
4. Tokenize datasets on VM
5. Start training (tmux session)
6. Monitor progress + budget
7. Download weights on completion
8. Destroy VM

### Cloud Providers

| Provider | GPU | Spot Price | Status |
|----------|-----|-----------|--------|
| **GCP** | A100 40GB | ~$1.20/h | Implemented (quota pending) |
| **Lambda** | A100 40GB | ~$1.48/h | Implemented (availability varies) |
| **AWS** | A10G 24GB | ~$0.30/h | Placeholder |

## Agent System

All agents work with both local and cloud training:

| Agent | Role | Integrations |
|-------|------|-------------|
| **Monitoring** | Budget tracking, loss alerts, stall detection | W&B, Telegram, GCP Billing |
| **Documentation** | Auto-update Notion training journal | Notion API |
| **Security** | Credential scanning, .gitignore checks | Local filesystem, GCP |
| **Telegram Bot** | Remote control via chat commands | All agents + orchestrator |

## Integrations

```
W&B ◄──── train.py (metrics, loss curves)
Notion ◄── doc_agent.py (training journal)
Telegram ◄── telegram_bot.py (commands & alerts)
HuggingFace ◄── train.py (datasets), deploy (model upload)
```
