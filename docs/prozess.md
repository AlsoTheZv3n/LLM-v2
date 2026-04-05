# Prozess-Dokumentation: LLaMA-355M von Grund auf

> Komplette Dokumentation wie wir ein 355M Parameter LLaMA-Style Sprachmodell von Grund auf gebaut, trainiert, fine-getuned und deployed haben.

---

## Inhaltsverzeichnis

1. [Projektziel](#1-projektziel)
2. [Infrastruktur aufbauen](#2-infrastruktur-aufbauen)
3. [Modell-Architektur](#3-modell-architektur)
4. [Pre-Training Phase 1: WikiText](#4-pre-training-phase-1-wikitext)
5. [Pre-Training Phase 2: Multi-Dataset Mix](#5-pre-training-phase-2-multi-dataset-mix)
6. [Fine-Tuning](#6-fine-tuning)
7. [Export & Deployment](#7-export--deployment)
8. [Testing & Evaluation](#8-testing--evaluation)
9. [Lessons Learned](#9-lessons-learned)
10. [Timeline](#10-timeline)

---

## 1. Projektziel

Ein eigenes Sprachmodell von Grund auf trainieren — kein vortrainiertes Modell verwenden, sondern alles selbst bauen: Architektur, Training-Loop, Dataset-Pipeline, Monitoring, Deployment.

**Zielmodell:** 1B Parameter LLaMA-Style Transformer
**Aktueller Stand:** 355M Parameter, lokal trainiert auf RTX 3080 Ti
**Budget:** $200 CHF (Cloud), $0 (lokal)

---

## 2. Infrastruktur aufbauen

### 2.1 Hardware

| Komponente | Details |
|-----------|---------|
| Lokale GPU | NVIDIA RTX 3080 Ti (12 GB VRAM) |
| Cloud GPU | GCP A100 40GB (geplant, Quota pending) |
| OS | Windows 11 Pro |
| Python | 3.12 |
| PyTorch | 2.6.0+cu124 |

### 2.2 Projekt-Setup

```bash
# Struktur erstellt
mkdir training/ cloud/ agents/ docs/

# Dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install tiktoken datasets wandb matplotlib
```

### 2.3 API-Integrationen eingerichtet

| Service | Zweck | Setup |
|---------|-------|-------|
| **HuggingFace** | Datasets + Model Upload | SSH Key erstellt, Token in .env |
| **Weights & Biases** | Experiment Tracking | Account erstellt, API Key in .env |
| **Notion** | Training Journal | Integration Token, Page IDs |
| **Telegram** | Remote Control Bot | Bot via @BotFather erstellt |
| **GCP** | Cloud Training | Service Account + Key erstellt |
| **Lambda Cloud** | Cloud Alternative | API Key in .env |

### 2.4 Agent-System gebaut

Alle Agents parallel entwickelt:

| Agent | Datei | Funktion |
|-------|-------|----------|
| Monitoring | `agents/monitoring_agent.py` | Budget-Tracking, Training-Alerts, Loss-Überwachung |
| Documentation | `agents/doc_agent.py` | Automatische Notion Journal Updates |
| Security | `agents/security_agent.py` | Credential-Scanning, .gitignore Enforcement |
| Orchestrator | `cloud/orchestrator.py` | Pipeline State Machine, Preemption Recovery |
| Telegram Bot | `agents/telegram_bot.py` | Remote Commands: /status, /training, /stop |

### 2.5 Cloud-Provider

| Provider | Datei | Status |
|----------|-------|--------|
| GCP | `cloud/providers/gcp.py` | Implementiert (MCP Server), Quota pending |
| Lambda | `cloud/providers/lambda_cloud.py` | Implementiert, Availability-Check |
| AWS | `cloud/providers/aws.py` | Placeholder |

---

## 3. Modell-Architektur

### 3.1 Evolution: GPT-2 → LLaMA-Style

Das Projekt startete mit einer GPT-2 Architektur (LayerNorm, learned position embeddings, GELU). Nach den ersten Trainingsläufen wurde auf eine moderne LLaMA-Style Architektur umgebaut:

| Komponente | GPT-2 (alt) | LLaMA-Style (final) | Verbesserung |
|-----------|-------------|---------------------|-------------|
| Normalization | LayerNorm | **RMSNorm** | ~10% schneller |
| Position Encoding | Learned Embeddings | **RoPE** (Rotary) | Bessere Extrapolation |
| Activation | GELU | **SwiGLU** (gate * up) | ~5% bessere Loss |
| Attention | Standard MHA | **Flash Attention** | 2-4x weniger VRAM |
| Bias | Überall bias=True | **Kein Bias** | Weniger Parameter |
| MLP | 4x hidden | **8/3x hidden mit Gate** | Effizienter |

### 3.2 Modell-Konfiguration (355M)

```python
{
    "V": 50257,    # Vocab (tiktoken GPT-2 BPE)
    "T": 512,      # Context Length
    "L": 24,       # Transformer Layers
    "H": 16,       # Attention Heads
    "C": 1024,     # Hidden Dimension
    "B": 2,        # Batch Size
}
# Total: 355,076,096 Parameter
# GPU Memory: ~1,355 MB (model only)
```

### 3.3 Tokenizer

**tiktoken GPT-2 BPE** mit 50,257 Tokens. Kein eigener Tokenizer — GPT-2 BPE ist gut genug für englischen Text und Code.

### 3.4 Implementierung

Die komplette Architektur ist in `training/train.py` implementiert:

```python
class RMSNorm          # Root Mean Square Normalization
class CausalSelfAttention  # Multi-Head Attention mit RoPE + Flash Attention
class SwiGLU_MLP       # SwiGLU Feed-Forward Network
class Block            # Transformer Block (Norm → Attention → Norm → MLP)
class LLaMAModel       # Full Model (Embedding → Blocks → Norm → LM Head)
```

---

## 4. Pre-Training Phase 1: WikiText

### 4.1 Dataset

**WikiText-103**: ~118M Tokens, englische Wikipedia-Artikel.

Gewählt weil:
- Klein genug für schnelles Iterieren
- Sauberer, gut formatierter Text
- Direkt über HuggingFace streambar (kein Download nötig)

### 4.2 Training

```bash
python training/train.py --config local --dataset wikitext --no-plot
```

**Hyperparameter:**

| Parameter | Wert |
|-----------|------|
| Optimizer | AdamW (beta1=0.9, beta2=0.95) |
| Learning Rate | 3e-4 (peak) → 3e-5 (cosine decay) |
| Warmup | 100 Steps |
| Weight Decay | 0.1 |
| Gradient Clipping | 1.0 (global norm) |
| Precision | fp16 (torch.amp) |
| Gradient Checkpointing | Enabled |

### 4.3 Probleme & Lösungen

| Problem | Lösung |
|---------|--------|
| `torch.compile` crasht auf Windows | Deaktiviert für Windows (`sys.platform != "win32"`) |
| Unicode-Fehler bei Sample-Output | `encoding="utf-8"` zum File-Writer hinzugefügt |
| B=4, T=1024 zu gross für 12GB VRAM | Reduziert auf B=2, T=512 |
| Checkpoint-Resume lädt falschen Run | Resume-Logic umgebaut: sucht höchsten Step über alle Runs |
| Final Eval crasht bei Ctrl+C | Eval in try/except gewrappt |
| Step-Counter akkumuliert über Runs | `arch: llama` Tag zum Checkpoint hinzugefügt, alte GPT-2 Checkpoints werden übersprungen |

### 4.4 Ergebnisse

| Metrik | Wert |
|--------|------|
| Steps | 100,000 (über mehrere Runs) |
| Final Loss | 4.97 |
| Best Eval PPL | 71.32 |
| Tokens verarbeitet | 45.5M |
| Trainingszeit | ~3.4h |
| Epochen auf WikiText | ~87x (starkes Overfitting) |

**Beobachtung:** Loss sank stetig, aber Eval PPL stieg ab Step 80K → Overfitting auf WikiText. Das Modell hat Wikipedia-Artikel auswendig gelernt statt zu generalisieren.

---

## 5. Pre-Training Phase 2: Multi-Dataset Mix

### 5.1 Warum Mix?

WikiText allein (118M Tokens) wurde ~87x wiederholt. Das führt zu:
- Overfitting auf Wikipedia-Stil
- Keine Generalisierung auf andere Textarten
- Modell kann nur Wikipedia-artige Sätze generieren

### 5.2 Dataset-Mix

Neuer `MixStreamer` implementiert — streamt aus 4 Datasets gleichzeitig mit Gewichtung:

| Dataset | Gewicht | Tokens | Inhalt |
|---------|---------|--------|--------|
| **OpenWebText** | 40% | ~9B | Reddit-kuratierter Web-Text |
| **AutoMathText-V2** | 20% | ~5B | Math & Reasoning |
| **WikiText-103** | 20% | ~118M | Fakten, Struktur |
| **Hacker News** | 20% | ~1B | Tech-Diskussionen |

Alle Datasets werden **live von HuggingFace gestreamt** — kein Download nötig.

### 5.3 Training

```bash
python training/train.py --dataset mix --no-plot
```

Training über mehrere Sessions (Ctrl+C zum Pausieren, Resume automatisch):

| Session | Steps | Dauer | Notizen |
|---------|-------|-------|---------|
| Run 1 | 0 → 107,500 | ~6h | Erster Mix-Run |
| Run 2 | 107,500 → 254,000 | ~8h | Weiter über Nacht |
| Run 3 | 254,000 → 300,000 | ~3h | Abschluss |

### 5.4 Ergebnisse

| Metrik | WikiText only | Mix Training |
|--------|--------------|-------------|
| Final Loss | 4.97 | **4.80** |
| Eval PPL | 121 (overfitted) | **143** (generalisiert) |
| Sample-Qualität | Wikipedia-Stil | Natürliche Konversation |
| Textvielfalt | Nur Fakten | Web, Math, Tech, Fakten |

**Samples vorher (WikiText):**
> "and had become a major military commander at the time of the Second World War"

**Samples nachher (Mix):**
> "I didn't really know what those guys were. I just thought that was the best thing"
> "If you're really in the process of getting your stuff done, by the way, then maybe"
> "What is the best way to do this? This is a simple way to do this"

### 5.5 GPU-Probleme

Während des Mix-Trainings fiel die GPU mehrmals aus:
- PyTorch wechselte auf CPU-Version (`2.5.1+cpu`)
- Ursache: Treiber-Crash nach langem Training oder pip reinstall
- Fix: `pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --user`

---

## 6. Fine-Tuning

### 6.1 Dataset

**TeichAI/Claude-Opus-4.6-Reasoning-887x**: 887 hochwertige Reasoning-Beispiele von Claude Opus mit `<think>` Blöcken.

Format: Chat-Messages (system, user, assistant)

### 6.2 Chat-Format

```
<|user|>
Was ist Kubernetes?
<|end|>
<|assistant|>
<think>
Der User fragt nach K8s...
</think>
Kubernetes ist ein Container-Orchestrierungssystem...
<|end|>
```

### 6.3 Training

```bash
python training/finetune.py --dataset claude-reasoning --epochs 3
```

| Parameter | Wert |
|-----------|------|
| Base Model | Mix-Training Checkpoint (Step 554K) |
| Learning Rate | 2e-5 (10x kleiner als Pre-Training) |
| Epochs | 3 |
| Gradient Accumulation | 8 Steps |
| Train/Eval Split | 799 / 88 Samples |

### 6.4 Ergebnisse

```
Epoch 1/3: loss 3.73 → 3.14
Epoch 2/3: loss weiter sinkend
Epoch 3/3: final eval_loss = 2.28
```

| Metrik | Wert |
|--------|------|
| Train Loss Start | 3.73 |
| Train Loss End | ~2.5 |
| Best Eval Loss | **2.28** |
| Dauer | ~25 min |

---

## 7. Export & Deployment

### 7.1 HuggingFace Format

```bash
python training/export.py --push
```

Konvertiert unsere Checkpoint-Struktur in HuggingFace-kompatibles Format:
- `pytorch_model.bin` — Gewichte (1.4 GB)
- `config.json` — LLaMA Architecture Config
- `tokenizer_config.json` — GPT-2 Tokenizer Reference
- `generation_config.json` — Sampling Parameters
- `README.md` — Model Card

**Live:** https://huggingface.co/Zv3n/llama-355m

### 7.2 GGUF Export

```bash
python training/export.py --gguf
```

Konvertiert in GGUF Format für llama.cpp / Ollama:
- Weight-Mapping: HF Names → GGUF Names (`model.layers.0.self_attn.q_proj` → `blk.0.attn_q`)
- Tokenizer: tiktoken GPT-2 BPE Vocab eingebettet
- Format: fp16
- Grösse: **776 MB**

Output: `exports/llama-355m-f16.gguf`

### 7.3 GitHub

```bash
git add . && git commit && git push
```

**Live:** https://github.com/AlsoTheZv3n/LLM-v2

---

## 8. Testing & Evaluation

### 8.1 Text-Generierung

Getestet mit 4 verschiedenen Prompts:

**Prompt: "The meaning of life is"**
> The meaning of life is a function of the physical world, not a thing of a real life. What you are writing is a useful concept that has no relationship to your brain.

**Prompt: "Python is a programming language that"**
> Python is a programming language that's usually designed for building web apps. If you're building a web app, you're building a web app, or just building something people want.

**Prompt: "In 2024, artificial intelligence"**
> In 2024, artificial intelligence is critical when it comes to the topic. The key is the problem. The problem is the problem: we don't need to be a tool to support our own data.

**Prompt: "The best way to learn machine learning is"**
> The best way to learn machine learning is to use a real learning tool for the real-life problem. You can use this in a real way.

### 8.2 Bewertung

| Aspekt | Bewertung | Details |
|--------|-----------|---------|
| Kohärenz | Gut | Grammatisch korrekte, zusammenhängende Sätze |
| Thematische Relevanz | Gut | Antworten passen zum Prompt |
| Faktenwissen | Begrenzt | Kennt Konzepte, aber Details oft falsch |
| Kreativität | Mittel | Repetitiv, neigt zu generischen Aussagen |
| Code-Verständnis | Basisch | Weiss was Python ist, aber kann keinen Code schreiben |
| Reasoning | Ansatzweise | Chain-of-Thought aus Fine-Tuning erkennbar |

### 8.3 Limitierungen

- **355M Parameter** — zu klein für tiefes Faktenwissen
- **118M unique Tokens** — WikiText-Daten oft wiederholt (Overfitting)
- **512 Context** — kurzes Kontextfenster
- **Kein RLHF/DPO** — keine Preference-Optimierung
- **887 Fine-Tune Samples** — zu wenig für robustes Instruction-Following

### 8.4 W&B Dashboard

Alle Training-Metriken auf: https://wandb.ai/xxdpsycho-/llm-pretrain-1b

Trackt: Loss, LR, Gradient Norm, Tokens/sec, Eval Perplexity, Samples

---

## 9. Lessons Learned

### Was gut lief

1. **LLaMA-Architektur** — deutlich besser als GPT-2 bei gleicher Parameterzahl
2. **Multi-Dataset Mix** — sofort bessere Generalisierung und natürlichere Texte
3. **Streaming von HuggingFace** — kein Download nötig, spart Disk
4. **Resume-System** — Training über Tage hinweg in Sessions, Checkpoint-basiert
5. **Agent-System** — Telegram Bot für Remote-Monitoring funktioniert gut
6. **GGUF Export** — direkt lauffähig in llama.cpp Ökosystem

### Was schwierig war

1. **GPU Quota auf GCP** — abgelehnt wegen neuem Projekt, 48h+ Wartezeit
2. **torch.compile auf Windows** — crasht zuverlässig, musste deaktiviert werden
3. **PyTorch CPU/CUDA Verwechslung** — pip installiert manchmal CPU-Version
4. **Overfitting auf WikiText** — 118M Tokens sind zu wenig für 355M Params
5. **Step-Counter Akkumulation** — Checkpoint-Steps addieren sich über Runs
6. **VRAM Management** — B=4 T=1024 hat die 3080 Ti überlastet

### Was wir anders machen würden

1. **Direkt mit Mix starten** — WikiText als alleinige Quelle war Zeitverschwendung
2. **Kleineres Modell zum Testen** — 20M statt 355M für schnelleres Iterieren
3. **Linux statt Windows** — torch.compile, Flash Attention v2, bessere CUDA Integration
4. **Cloud von Anfang an** — lokales Training auf 3080 Ti ist zu langsam für grosse Modelle

---

## 10. Timeline

| Datum | Was |
|-------|-----|
| 15. März | Projekt gestartet, GPT-2 C++ Engine + PyTorch train.py |
| 15. März | Erste Trainingsläufe (small, medium) auf WikiText |
| 18. März | .env Setup: GCP, HuggingFace, W&B, Notion, Telegram |
| 18. März | GCP Service Account + SSH Keys erstellt |
| 19. März | Agent-System gebaut (Monitoring, Doc, Security, Orchestrator, Telegram Bot) |
| 19. März | Projekt-Umstrukturierung (training/, cloud/, agents/) |
| 20. März | GCP Quota beantragt → abgelehnt (Projekt zu neu) |
| 21. März | L4/T4 Alternativen geprüft → Stockout überall |
| 25. März | WikiText Training gestartet (355M, local config) |
| 26. März | Checkpoint-Resume Bugs gefixt |
| 27. März | Training fortgesetzt, Unicode-Fix, torch.compile deaktiviert |
| 28. März | Architektur-Upgrade: GPT-2 → LLaMA-Style |
| 28. März | WikiText Training abgeschlossen (100K Steps, Loss 4.97) |
| 28. März | GitHub Repo erstellt, public push |
| 29. März | Mix-Dataset Training gestartet (OpenWebText + AutoMathText + WikiText + HackerNews) |
| 29-31. März | Mix-Training über mehrere Sessions |
| 30. März | PyTorch CUDA Fix (CPU-Version wurde installiert) |
| 1-4. April | Mix-Training fortgesetzt bis 300K Steps |
| 4. April | Mix-Training abgeschlossen (Loss 4.80, PPL 143) |
| 5. April | Fine-Tuning auf Claude-Reasoning (3 Epochs, Eval Loss 2.28) |
| 5. April | GGUF Export (776 MB) |
| 5. April | HuggingFace Upload (https://huggingface.co/Zv3n/llama-355m) |
| 5. April | Text-Generierung getestet — funktioniert! |

---

## Anhang: Reproduzierbarkeit

### Komplettes Training reproduzieren

```bash
# 1. Repo klonen
git clone https://github.com/AlsoTheZv3n/LLM-v2.git
cd LLM-v2

# 2. Setup
cp env.example .env
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. Pre-Training (Mix, ~10h auf RTX 3080 Ti)
python training/train.py --dataset mix --no-plot

# 4. Fine-Tuning (~25 min)
python training/finetune.py --dataset claude-reasoning --epochs 3

# 5. Export
python training/export.py --push --gguf

# 6. Testen
python -c "
import torch, sys
sys.path.insert(0, 'training')
from train import LLaMAModel, get_tokenizer, get_device
device = get_device()
encode, decode, vocab = get_tokenizer()
ckpt = torch.load('runs/ft_claude-reasoning_*/checkpoint_best.pt', map_location=device, weights_only=False)
cfg = ckpt['cfg']; cfg['V'] = vocab
model = LLaMAModel(cfg).to(device)
model.load_state_dict(ckpt['model']); model.eval()
tokens = encode('Hello, I am')
idx = torch.tensor([tokens], dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=50)
print(decode(out[0].tolist()))
"
```

### Kosten

| Phase | Kosten |
|-------|--------|
| Pre-Training (lokal) | $0 |
| Fine-Tuning (lokal) | $0 |
| Strom (~15h GPU) | ~$3 |
| Cloud Training | $0 (nicht gestartet) |
| **Total** | **~$3** |
