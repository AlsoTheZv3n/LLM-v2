#!/usr/bin/env python3
"""
doc_agent.py — Phase 1.7: Documentation Agent
==============================================
Automatically updates the Notion Training Journal with training progress.

Usage:
    python doc_agent.py update --step 5000 --loss 1.9 --ppl 6.3
    python doc_agent.py snapshot
    python doc_agent.py milestone "Training started on A100"
    python doc_agent.py incident "Loss spike at step 3000"
"""

import argparse
import json
import os
import time
import threading
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────
NOTION_KEY       = os.getenv("NOTION_API_KEY", "")
JOURNAL_PAGE_ID  = os.getenv("NOTION_TRAINING_JOURNAL_ID", "")
ROADMAP_PAGE_ID  = os.getenv("NOTION_ROADMAP_PAGE_ID", "")
NOTION_VERSION   = "2022-06-28"
NOTION_BASE      = "https://api.notion.com/v1"

WANDB_KEY        = os.getenv("WANDB_API_KEY", "")
WANDB_PROJECT    = os.getenv("WANDB_PROJECT", "llm-pretrain-1b")


# ─────────────────────────────────────────────────────────────
#  Notion API Helpers
# ─────────────────────────────────────────────────────────────
def notion_headers():
    return {
        "Authorization": f"Bearer {NOTION_KEY}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def notion_append_blocks(page_id: str, blocks: list[dict]) -> bool:
    """Append blocks to a Notion page."""
    url = f"{NOTION_BASE}/blocks/{page_id}/children"
    resp = requests.patch(url, headers=notion_headers(), json={"children": blocks})
    if resp.status_code != 200:
        print(f"  [notion] Error {resp.status_code}: {resp.text[:200]}")
        return False
    return True


def notion_get_children(page_id: str) -> list[dict]:
    """Get all child blocks of a page."""
    url = f"{NOTION_BASE}/blocks/{page_id}/children?page_size=100"
    resp = requests.get(url, headers=notion_headers())
    if resp.status_code != 200:
        return []
    return resp.json().get("results", [])


def make_heading(text: str, level: int = 2) -> dict:
    key = f"heading_{level}"
    return {
        "object": "block",
        "type": key,
        key: {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def make_paragraph(text: str) -> dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def make_callout(text: str, emoji: str = "📊") -> dict:
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [{"type": "text", "text": {"content": text}}],
            "icon": {"type": "emoji", "emoji": emoji},
        },
    }


def make_divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


# ─────────────────────────────────────────────────────────────
#  Documentation Agent
# ─────────────────────────────────────────────────────────────
class DocumentationAgent:
    """Phase 1.7: Automatically documents training progress to Notion."""

    def __init__(self):
        self._running = False
        self._thread = None
        self._snapshot_interval = 300  # 5 minutes

    def start(self, interval: int = 300):
        """Start periodic snapshot thread."""
        self._snapshot_interval = interval
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"  [doc_agent] Started (snapshot every {interval}s)")

    def stop(self):
        self._running = False
        print("  [doc_agent] Stopped")

    @property
    def is_running(self):
        return self._running

    def _loop(self):
        while self._running:
            try:
                self.post_snapshot()
            except Exception as e:
                print(f"  [doc_agent] snapshot error: {e}")
            time.sleep(self._snapshot_interval)

    # ── Journal Entries ───────────────────────────────────────

    def post_update(self, step: int, loss: float, ppl: float = None,
                    tokens: int = None, eta: str = None, spend: float = None):
        """Post a training update to the journal."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [f"Step {step:,} | Loss {loss:.4f}"]
        if ppl is not None:
            lines.append(f"PPL {ppl:.2f}")
        if tokens is not None:
            lines.append(f"Tokens {tokens/1e9:.2f}B")
        if eta:
            lines.append(f"ETA {eta}")
        if spend is not None:
            lines.append(f"Spend ${spend:.2f}")

        text = " | ".join(lines)
        blocks = [make_callout(f"[{now}] {text}", "📊")]

        return notion_append_blocks(JOURNAL_PAGE_ID, blocks)

    def post_snapshot(self):
        """Auto-detect latest training state and post to journal."""
        # Try wandb first
        if WANDB_KEY:
            try:
                import wandb
                api = wandb.Api()
                runs = api.runs(f"{api.viewer.username}/{WANDB_PROJECT}", order="-created_at")
                if runs:
                    run = runs[0]
                    history = run.history(samples=1)
                    if not history.empty:
                        row = history.iloc[-1]
                        self.post_update(
                            step=int(row.get("_step", 0)),
                            loss=float(row.get("loss", 0)),
                            ppl=float(row.get("eval_ppl", 0)) if "eval_ppl" in row else None,
                            tokens=int(row.get("tokens_total", 0)) if "tokens_total" in row else None,
                        )
                        return
            except Exception:
                pass

        # Fallback: local logs
        runs_dir = Path("./runs")
        logs = sorted(runs_dir.glob("*/training_log.jsonl")) if runs_dir.exists() else []
        if logs:
            with open(logs[-1]) as f:
                lines = f.readlines()
            if lines:
                last = json.loads(lines[-1])
                self.post_update(
                    step=last.get("step", 0),
                    loss=last.get("loss", 0),
                )

    def post_milestone(self, message: str):
        """Post a milestone entry."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        blocks = [make_callout(f"[{now}] MILESTONE: {message}", "🏆")]
        return notion_append_blocks(JOURNAL_PAGE_ID, blocks)

    def post_incident(self, message: str):
        """Post an incident entry."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        blocks = [make_callout(f"[{now}] INCIDENT: {message}", "🚨")]
        return notion_append_blocks(JOURNAL_PAGE_ID, blocks)

    def post_hyperparameter_change(self, param: str, old_val: str, new_val: str, reason: str = ""):
        """Post a hyperparameter change."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        text = f"[{now}] HYPERPARAM: {param}: {old_val} -> {new_val}"
        if reason:
            text += f" (Reason: {reason})"
        blocks = [make_callout(text, "🔧")]
        return notion_append_blocks(JOURNAL_PAGE_ID, blocks)

    # ── Roadmap Updates ───────────────────────────────────────

    def update_roadmap(self, phase: str, status: str):
        """Update a phase status on the roadmap page."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        blocks = [make_paragraph(f"[{now}] {phase}: {status}")]
        return notion_append_blocks(ROADMAP_PAGE_ID, blocks)


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Phase 1.7: Documentation Agent")
    sub = p.add_subparsers(dest="command")

    # update
    up = sub.add_parser("update", help="Post training update")
    up.add_argument("--step", type=int, required=True)
    up.add_argument("--loss", type=float, required=True)
    up.add_argument("--ppl", type=float)
    up.add_argument("--tokens", type=int)
    up.add_argument("--eta", type=str)
    up.add_argument("--spend", type=float)

    # snapshot
    sub.add_parser("snapshot", help="Auto-detect and post snapshot")

    # milestone
    ms = sub.add_parser("milestone", help="Post milestone")
    ms.add_argument("message", type=str)

    # incident
    inc = sub.add_parser("incident", help="Post incident")
    inc.add_argument("message", type=str)

    # hyperparam
    hp = sub.add_parser("hyperparam", help="Post hyperparameter change")
    hp.add_argument("--param", required=True)
    hp.add_argument("--old", required=True)
    hp.add_argument("--new", required=True)
    hp.add_argument("--reason", default="")

    # roadmap
    rm = sub.add_parser("roadmap", help="Update roadmap phase status")
    rm.add_argument("--phase", required=True)
    rm.add_argument("--status", required=True)

    args = p.parse_args()
    agent = DocumentationAgent()

    if args.command == "update":
        ok = agent.post_update(args.step, args.loss, args.ppl, args.tokens, args.eta, args.spend)
        print(f"  {'OK' if ok else 'FAILED'}")

    elif args.command == "snapshot":
        agent.post_snapshot()
        print("  Snapshot posted.")

    elif args.command == "milestone":
        ok = agent.post_milestone(args.message)
        print(f"  {'OK' if ok else 'FAILED'}")

    elif args.command == "incident":
        ok = agent.post_incident(args.message)
        print(f"  {'OK' if ok else 'FAILED'}")

    elif args.command == "hyperparam":
        ok = agent.post_hyperparameter_change(args.param, args.old, args.new, args.reason)
        print(f"  {'OK' if ok else 'FAILED'}")

    elif args.command == "roadmap":
        ok = agent.update_roadmap(args.phase, args.status)
        print(f"  {'OK' if ok else 'FAILED'}")

    else:
        p.print_help()


if __name__ == "__main__":
    main()
