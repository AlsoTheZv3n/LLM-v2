#!/usr/bin/env python3
"""
eval.py — Training Progression Evaluator
=========================================
Analyzes completed training runs:
 - Loads training_log.jsonl
 - Plots final loss + perplexity curves
 - Computes convergence metrics
 - Generates text samples at different checkpoints
 - Prints ASCII summary table

Usage:
    python eval.py                            # analyze latest run
    python eval.py --run runs/demo_20240101   # specific run
    python eval.py --all                      # compare all runs
    python eval.py --sample checkpoint.bin    # generate samples
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────
#  Load training log
# ─────────────────────────────────────────────────────────────
def load_log(run_dir: Path) -> tuple[list, list, list]:
    """Returns (metrics, evals, samples)."""
    metrics, evals, samples = [], [], []
    log_file = run_dir / "training_log.jsonl"
    eval_file = run_dir / "eval_results.jsonl"

    if log_file.exists():
        for line in log_file.read_text().splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                    if d.get("type") == "metrics":
                        metrics.append(d)
                    elif d.get("type") == "eval":
                        evals.append(d)
                except:
                    pass

    if eval_file.exists():
        for line in eval_file.read_text().splitlines():
            if line.strip():
                try:
                    evals.append(json.loads(line))
                except:
                    pass

    sample_file = run_dir / "samples.txt"
    if sample_file.exists():
        samples = sample_file.read_text()

    return metrics, evals, samples

# ─────────────────────────────────────────────────────────────
#  Compute stats
# ─────────────────────────────────────────────────────────────
def compute_stats(metrics: list) -> dict:
    if not metrics:
        return {}
    losses = [m["loss"] for m in metrics]
    steps  = [m["step"] for m in metrics]
    ms_per = [m["ms"]   for m in metrics if "ms" in m]

    # smoothed losses (EMA)
    smooth, ema = [], -1.0
    for l in losses:
        ema = l if ema < 0 else 0.95 * ema + 0.05 * l
        smooth.append(ema)

    # convergence: step where loss first dropped below each threshold
    thresholds = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0]
    convergence = {}
    for thr in thresholds:
        for i, l in enumerate(smooth):
            if l < thr:
                convergence[f"<{thr}"] = steps[i]
                break

    # find best loss window
    window = 50
    best_avg = float("inf")
    best_start = 0
    for i in range(len(smooth) - window):
        avg = sum(smooth[i:i+window]) / window
        if avg < best_avg:
            best_avg = avg
            best_start = steps[i]

    avg_ms   = sum(ms_per) / len(ms_per) if ms_per else 0
    tps      = (metrics[0].get("step", 0) * (metrics[0].get("tokens", 1) // max(metrics[0].get("step", 1), 1))) / len(metrics) if metrics else 0

    total_tokens = max((m.get("tokens", 0) for m in metrics), default=0)

    return {
        "steps":         len(steps),
        "max_step":      max(steps),
        "initial_loss":  losses[0],
        "final_loss":    losses[-1],
        "best_smooth":   min(smooth),
        "final_smooth":  smooth[-1],
        "best_window":   best_avg,
        "best_step":     best_start,
        "convergence":   convergence,
        "avg_ms_step":   avg_ms,
        "total_tokens":  total_tokens,
        "smooth_losses": smooth,
        "raw_losses":    losses,
        "steps_list":    steps,
    }

# ─────────────────────────────────────────────────────────────
#  ASCII loss chart
# ─────────────────────────────────────────────────────────────
def ascii_chart(values: list, width: int = 60, height: int = 15, title: str = "") -> str:
    if not values:
        return "(no data)"
    mn, mx = min(values), max(values)
    if mx == mn:
        mx = mn + 0.01
    rows = []
    for row in range(height):
        thr = mx - (row / (height - 1)) * (mx - mn)
        line = ""
        for v in values[::max(len(values)//width, 1)][:width]:
            line += "█" if v >= thr else " "
        y_label = f"{thr:6.3f} │"
        rows.append(y_label + line)
    x_axis = "       └" + "─" * width
    header = f"  {title}" if title else ""
    return "\n".join([header] + rows + [x_axis])

# ─────────────────────────────────────────────────────────────
#  Print summary
# ─────────────────────────────────────────────────────────────
def print_summary(run_dir: Path, metrics: list, evals: list, samples: str):
    stats = compute_stats(metrics)
    if not stats:
        print(f"  No training data found in {run_dir}")
        return

    # load run info
    info = {}
    info_file = run_dir / "run_info.json"
    if info_file.exists():
        info = json.loads(info_file.read_text())

    summary_file = run_dir / "summary.json"
    summary = {}
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())

    # ── Header ────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  Training Run Analysis")
    print(f"  {run_dir.name}")
    if "started" in info:
        print(f"  Started: {info['started'][:19]}")
    print(f"{'═'*62}")

    # ── Config ────────────────────────────────────────────────
    cfg_name = summary.get("config", "unknown")
    ds_name  = summary.get("dataset", "unknown")
    print(f"\n  Config:     {cfg_name}  |  Dataset: {ds_name}")

    # ── Loss chart ────────────────────────────────────────────
    print(f"\n  Loss Curve (smoothed)")
    print(ascii_chart(stats["smooth_losses"][:500], width=55, height=12))

    # ── Metrics table ─────────────────────────────────────────
    print(f"\n  Training Metrics")
    print(f"  {'─'*40}")
    rows = [
        ("Steps completed",  f"{stats['max_step']:,}"),
        ("Initial loss",     f"{stats['initial_loss']:.4f}"),
        ("Final loss",       f"{stats['final_loss']:.4f}  (raw)"),
        ("Best smooth loss", f"{stats['best_smooth']:.4f}"),
        ("Final smooth",     f"{stats['final_smooth']:.4f}"),
        ("Total tokens",     f"{stats['total_tokens']:,}"),
        ("Avg ms/step",      f"{stats['avg_ms_step']:.1f} ms"),
        ("Throughput",       f"~{int(1000/max(stats['avg_ms_step'],1)):,} tok/s"),
    ]
    for k, v in rows:
        print(f"  {k:<25} {v}")

    # ── Convergence ───────────────────────────────────────────
    if stats["convergence"]:
        print(f"\n  Convergence Milestones")
        print(f"  {'─'*40}")
        for thr, step in sorted(stats["convergence"].items(), key=lambda x: float(x[0][1:])):
            bar = "▓" * min(int(step / stats["max_step"] * 30), 30)
            print(f"  Loss {thr:<6}  at step {step:>6}  [{bar:<30}]")

    # ── Eval perplexity ───────────────────────────────────────
    if evals:
        print(f"\n  Evaluation Perplexity  ({len(evals)} checkpoints)")
        print(f"  {'─'*40}")
        for e in evals[-5:]:  # last 5
            ppl = e.get("ppl", math.exp(e.get("loss", 0)))
            bar = "░" * min(int(ppl / 100 * 30), 30)
            print(f"  Step {e.get('step', '?'):>6}  PPL {ppl:>8.2f}  [{bar}]")

    if summary.get("eval"):
        ev = summary["eval"]
        print(f"\n  Final Eval  PPL={ev.get('perplexity',0):.2f}  "
              f"loss={ev.get('eval_loss',0):.4f}")

    # ── Samples ───────────────────────────────────────────────
    if samples:
        print(f"\n  Generated Samples")
        print(f"  {'─'*40}")
        # print last sample
        parts = samples.strip().split("="*60)
        if len(parts) >= 2:
            last = parts[-1].strip()
            lines = last.splitlines()[:6]
            for line in lines:
                print(f"  │ {line}")

    # ── Grade ─────────────────────────────────────────────────
    loss = stats["final_smooth"]
    if   loss < 1.5: grade, note = "A+", "near-perfect compression"
    elif loss < 2.0: grade, note = "A",  "very good language model"
    elif loss < 2.5: grade, note = "B",  "good structure learned"
    elif loss < 3.0: grade, note = "C",  "basic patterns emerging"
    elif loss < 4.0: grade, note = "D",  "training in early stages"
    else:            grade, note = "F",  "needs more training"

    print(f"\n  {'═'*40}")
    print(f"  Grade: {grade}   ({note})")
    print(f"  {'═'*40}\n")

# ─────────────────────────────────────────────────────────────
#  Compare runs
# ─────────────────────────────────────────────────────────────
def compare_runs(run_dirs: list[Path]):
    print(f"\n{'═'*70}")
    print(f"  Run Comparison")
    print(f"{'═'*70}")
    print(f"  {'Run':<30} {'Steps':>7} {'Init':>8} {'Final':>8} {'Best':>8} {'PPL':>8}")
    print(f"  {'─'*66}")

    for rd in sorted(run_dirs):
        metrics, evals, _ = load_log(rd)
        stats = compute_stats(metrics)
        if not stats:
            continue
        ppl = "?"
        if evals:
            ppl = f"{evals[-1].get('ppl', 0):.1f}"
        print(f"  {rd.name:<30} {stats['max_step']:>7,} "
              f"{stats['initial_loss']:>8.4f} {stats['final_loss']:>8.4f} "
              f"{stats['best_smooth']:>8.4f} {ppl:>8}")
    print()

# ─────────────────────────────────────────────────────────────
#  Generate samples from checkpoint
# ─────────────────────────────────────────────────────────────
def generate_samples(ckpt_path: str, n: int = 3, length: int = 200):
    """Use the C++ engine to generate samples from a checkpoint."""
    import subprocess
    import struct

    # we need a config to launch the engine - read from nearby run_info
    ckpt = Path(ckpt_path)
    # peek into checkpoint to get config
    with open(ckpt, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != 0x47455054:
            print("Invalid checkpoint!")
            return
        step = struct.unpack("<i", f.read(4))[0]
        cfg_bytes = f.read(8 * 4)
        V, T, L, H, C, B, steps, mode = struct.unpack("<8i", cfg_bytes)

    print(f"\n  Checkpoint: step={step}  L={L} H={H} C={C} V={V}")
    print(f"  Generating {n} samples ({length} chars each)...\n")

    for i in range(n):
        print(f"  ── Sample {i+1} {'─'*40}")
        # TODO: implement proper sampling via engine
        # For now, print a placeholder
        print(f"  (Run './gpt2_engine {ckpt.parent}' with mode=sample)")
        print()

# ─────────────────────────────────────────────────────────────
#  Plot (if matplotlib available)
# ─────────────────────────────────────────────────────────────
def plot_run(run_dir: Path, metrics: list, evals: list):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        stats = compute_stats(metrics)
        if not stats:
            return

        fig = plt.figure(figsize=(14, 8), facecolor="#0d1117")
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
        axes = [fig.add_subplot(gs[i//3, i%3]) for i in range(6)]

        DARK = "#0d1117"; LINE = "#c9d1d9"; ACC = "#58a6ff"; YLW = "#e3b341"

        for ax in axes:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors=LINE, labelsize=8)
            for sp in ax.spines.values(): sp.set_color("#30363d")

        xs = stats["steps_list"]

        # Loss
        axes[0].plot(xs, stats["raw_losses"],    color=ACC, alpha=0.3, lw=0.7)
        axes[0].plot(xs, stats["smooth_losses"], color=ACC, lw=2)
        axes[0].set_title("Training Loss", color=LINE, fontsize=10)
        axes[0].set_xlabel("step", color=LINE, fontsize=8)

        # Eval PPL
        if evals:
            es = [e.get("step", 0) for e in evals]
            pp = [e.get("ppl", math.exp(e.get("loss", 0))) for e in evals]
            axes[1].plot(es, pp, color=YLW, marker="o", ms=4, lw=1.5)
        axes[1].set_title("Eval Perplexity", color=LINE, fontsize=10)
        axes[1].set_xlabel("step", color=LINE, fontsize=8)

        # LR
        lrs = [m.get("lr", 0) for m in metrics]
        axes[2].plot(xs, lrs, color="#3fb950", lw=1.5)
        axes[2].set_title("Learning Rate", color=LINE, fontsize=10)
        axes[2].set_xlabel("step", color=LINE, fontsize=8)

        # Grad norm
        gnorms = [m.get("grad_norm", 0) for m in metrics]
        axes[3].plot(xs, gnorms, color="#ff7b72", lw=0.8, alpha=0.7)
        axes[3].axhline(1.0, color="#30363d", ls="--", lw=0.8)
        axes[3].set_title("Gradient Norm", color=LINE, fontsize=10)
        axes[3].set_xlabel("step", color=LINE, fontsize=8)

        # Loss reduction
        if len(xs) > 10:
            reductions = [stats["raw_losses"][0] - l for l in stats["smooth_losses"]]
            axes[4].fill_between(xs, 0, reductions, alpha=0.4, color=ACC)
            axes[4].plot(xs, reductions, color=ACC, lw=1)
        axes[4].set_title("Loss Reduction", color=LINE, fontsize=10)
        axes[4].set_xlabel("step", color=LINE, fontsize=8)

        # ms/step
        ms_list = [m.get("ms", 0) for m in metrics]
        axes[5].fill_between(xs, ms_list, alpha=0.3, color="#8b949e")
        axes[5].plot(xs, ms_list, color="#8b949e", lw=0.8)
        axes[5].set_title("ms / step", color=LINE, fontsize=10)
        axes[5].set_xlabel("step", color=LINE, fontsize=8)

        fig.suptitle(
            f"Training Analysis  |  {run_dir.name}  |  "
            f"final loss {stats['final_smooth']:.4f}",
            color=LINE, fontsize=11, y=0.98
        )

        out = run_dir / "eval_report.png"
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  Saved plot: {out}")
    except ImportError:
        print("  (matplotlib not available — skipping plot)")
    except Exception as e:
        print(f"  (plot error: {e})")

# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Training Progression Evaluator")
    p.add_argument("--run",    default=None, help="Specific run directory")
    p.add_argument("--all",    action="store_true", help="Compare all runs")
    p.add_argument("--sample", default=None, help="Generate samples from checkpoint.bin")
    p.add_argument("--plot",   action="store_true", help="Save evaluation plots")
    args = p.parse_args()

    if args.sample:
        generate_samples(args.sample)
        return

    log_root = Path(__file__).resolve().parent.parent / "runs"

    if args.all and log_root.exists():
        all_runs = [d for d in log_root.iterdir() if d.is_dir()]
        if all_runs:
            compare_runs(all_runs)
            for rd in all_runs:
                metrics, evals, samples = load_log(rd)
                if metrics:
                    print_summary(rd, metrics, evals, samples)
                    if args.plot:
                        plot_run(rd, metrics, evals)
        else:
            print("No runs found in ./runs/")
        return

    # single run
    if args.run:
        run_dir = Path(args.run)
    elif log_root.exists():
        all_runs = sorted([d for d in log_root.iterdir() if d.is_dir()])
        if not all_runs:
            print("No training runs found in ./runs/")
            print("Run train.py first!")
            sys.exit(1)
        run_dir = all_runs[-1]  # latest
    else:
        print("No runs directory found. Run train.py first!")
        sys.exit(1)

    metrics, evals, samples = load_log(run_dir)
    print_summary(run_dir, metrics, evals, samples)

    if args.plot:
        plot_run(run_dir, metrics, evals)

if __name__ == "__main__":
    main()
