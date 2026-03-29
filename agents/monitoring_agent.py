#!/usr/bin/env python3
"""
monitoring_agent.py — Phase 1.6: Cost & Training Monitor
========================================================
Monitors GCP costs and training progress. Sends alerts via Telegram.

Usage:
    python monitoring_agent.py start    # run monitoring loop
    python monitoring_agent.py status   # one-shot status report
    python monitoring_agent.py budget   # show budget status
"""

import argparse
import json
import os
import time
import threading
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────
BUDGET_TOTAL      = float(os.getenv("BUDGET_TOTAL_USD", "200"))
BUDGET_WARN_PCT   = float(os.getenv("BUDGET_WARN_PERCENT", "80"))
BUDGET_PAUSE_PCT  = float(os.getenv("BUDGET_PAUSE_PERCENT", "95"))
BUDGET_STOP_PCT   = float(os.getenv("BUDGET_STOP_PERCENT", "100"))
WANDB_KEY         = os.getenv("WANDB_API_KEY", "")
WANDB_PROJECT     = os.getenv("WANDB_PROJECT", "llm-pretrain-1b")
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT     = os.getenv("TELEGRAM_CHAT_ID", "")
SA_KEY_PATH       = str(PROJECT_ROOT / os.getenv("GCP_SERVICE_ACCOUNT_KEY_PATH", "gcp-agent-key.json"))
PROJECT_ID        = os.getenv("GCP_PROJECT_ID", "")
ZONE              = os.getenv("GCP_ZONE", "europe-west6-a")
VM_NAME           = os.getenv("VM_INSTANCE_NAME", "llm-pretrain-1b")

MONITOR_INTERVAL  = 60  # seconds between checks
LOG_FILE          = PROJECT_ROOT / "monitoring_log.jsonl"


# ─────────────────────────────────────────────────────────────
#  Telegram Notifications
# ─────────────────────────────────────────────────────────────
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        import urllib.request
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = json.dumps({"chat_id": TELEGRAM_CHAT, "text": message, "parse_mode": "Markdown"}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"  [telegram] send failed: {e}")


# ─────────────────────────────────────────────────────────────
#  Budget Monitoring
# ─────────────────────────────────────────────────────────────
def estimate_spend() -> dict:
    """Estimate current spend based on VM uptime. Real billing has ~24h delay."""
    # A100 Spot VM cost: ~$1.60/hr
    COST_PER_HOUR = 1.60
    DISK_COST_PER_GB_MONTH = 0.04

    result = {"vm_status": "unknown", "estimated_spend": 0, "budget_total": BUDGET_TOTAL}

    try:
        from google.cloud import compute_v1
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(SA_KEY_PATH)
        client = compute_v1.InstancesClient(credentials=creds)

        try:
            instance = client.get(project=PROJECT_ID, zone=ZONE, instance=VM_NAME)
            result["vm_status"] = instance.status

            # Estimate: creation time to now
            # Note: This is a rough estimate. Real billing API has delay.
            if instance.status == "RUNNING":
                result["accruing"] = True
                result["cost_per_hour"] = COST_PER_HOUR
        except Exception:
            result["vm_status"] = "NOT_FOUND"

    except Exception as e:
        result["error"] = str(e)

    return result


def check_budget() -> dict:
    """Check budget thresholds and return status."""
    spend = estimate_spend()
    pct = (spend.get("estimated_spend", 0) / BUDGET_TOTAL) * 100

    status = "OK"
    if pct >= BUDGET_STOP_PCT:
        status = "STOP"
    elif pct >= BUDGET_PAUSE_PCT:
        status = "PAUSE"
    elif pct >= BUDGET_WARN_PCT:
        status = "WARNING"

    return {
        "status": status,
        "spend": spend.get("estimated_spend", 0),
        "budget": BUDGET_TOTAL,
        "percent": pct,
        "vm_status": spend.get("vm_status", "unknown"),
    }


# ─────────────────────────────────────────────────────────────
#  Training Monitoring
# ─────────────────────────────────────────────────────────────
def get_wandb_status() -> dict:
    """Get latest training metrics from W&B."""
    if not WANDB_KEY:
        return {"error": "WANDB_API_KEY not set"}
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(f"{api.viewer.username}/{WANDB_PROJECT}", order="-created_at")
        if not runs:
            return {"error": "No runs found"}
        run = runs[0]
        history = run.history(samples=1)
        if history.empty:
            return {"run_name": run.name, "status": run.state}
        latest = history.iloc[-1].to_dict()
        return {
            "run_name": run.name,
            "status": run.state,
            "step": latest.get("_step", 0),
            "loss": latest.get("loss"),
            "smooth_loss": latest.get("smooth_loss"),
            "lr": latest.get("lr"),
            "eval_ppl": latest.get("eval_ppl"),
        }
    except Exception as e:
        return {"error": str(e)}


def get_local_status() -> dict:
    """Get latest training metrics from local log files."""
    runs_dir = Path("./runs")
    if not runs_dir.exists():
        return {"error": "No runs directory"}

    logs = sorted(runs_dir.glob("*/training_log.jsonl"))
    if not logs:
        return {"error": "No training logs found"}

    latest_log = logs[-1]
    try:
        with open(latest_log) as f:
            lines = f.readlines()
        if not lines:
            return {"error": "Empty log"}
        last = json.loads(lines[-1])
        return {
            "run_dir": str(latest_log.parent.name),
            "step": last.get("step", 0),
            "loss": last.get("loss"),
            "smooth_loss": last.get("smooth_loss"),
            "lr": last.get("lr"),
            "total_entries": len(lines),
        }
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────
#  Monitoring Agent
# ─────────────────────────────────────────────────────────────
class MonitoringAgent:
    """Phase 1.6: Monitors cost and training, sends alerts."""

    def __init__(self):
        self._running = False
        self._thread = None
        self._last_loss = None
        self._smooth_loss = None
        self._stall_time = None
        self._alerts_sent = set()

    def start(self):
        """Start monitoring loop in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"  [monitor] Started (interval={MONITOR_INTERVAL}s)")

    def stop(self):
        self._running = False
        print("  [monitor] Stopped")

    @property
    def is_running(self):
        return self._running

    def _loop(self):
        while self._running:
            try:
                self._check_cycle()
            except Exception as e:
                self._log({"error": str(e)})
            time.sleep(MONITOR_INTERVAL)

    def _check_cycle(self):
        now = datetime.now().isoformat()

        # Budget check
        budget = check_budget()
        if budget["status"] == "WARNING" and "budget_warn" not in self._alerts_sent:
            self._alert(f"Budget WARNING: {budget['percent']:.0f}% used (${budget['spend']:.0f}/${budget['budget']:.0f})")
            self._alerts_sent.add("budget_warn")
        elif budget["status"] == "PAUSE" and "budget_pause" not in self._alerts_sent:
            self._alert(f"Budget PAUSE: {budget['percent']:.0f}% used! Pausing training.")
            self._alerts_sent.add("budget_pause")
        elif budget["status"] == "STOP" and "budget_stop" not in self._alerts_sent:
            self._alert(f"Budget STOP: {budget['percent']:.0f}% used! Stopping VM.")
            self._alerts_sent.add("budget_stop")

        # Training check
        training = get_wandb_status() if WANDB_KEY else get_local_status()
        loss = training.get("loss")

        if loss is not None:
            # Loss spike detection
            if self._smooth_loss is not None and loss > self._smooth_loss * 2:
                self._alert(f"Loss SPIKE: {loss:.4f} (smooth: {self._smooth_loss:.4f})")

            # Update smooth loss
            if self._smooth_loss is None:
                self._smooth_loss = loss
            else:
                self._smooth_loss = 0.95 * self._smooth_loss + 0.05 * loss

            self._last_loss = loss
            self._stall_time = None
        else:
            # Stall detection
            if self._stall_time is None:
                self._stall_time = time.time()
            elif time.time() - self._stall_time > 600:  # 10 minutes
                if "stall" not in self._alerts_sent:
                    self._alert("Training STALL: No progress for 10 minutes")
                    self._alerts_sent.add("stall")

        # Log
        self._log({
            "timestamp": now,
            "budget": budget,
            "training": training,
        })

    def _alert(self, message: str):
        print(f"  [ALERT] {message}")
        send_telegram(f"LLM Training Alert:\n{message}")

    def _log(self, data: dict):
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")

    def get_status(self) -> dict:
        """One-shot status report."""
        budget = check_budget()
        training = get_wandb_status() if WANDB_KEY else get_local_status()
        return {
            "monitoring": "running" if self._running else "stopped",
            "budget": budget,
            "training": training,
        }


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def print_status(status: dict):
    print(f"\n{'='*60}")
    print(f"  Monitoring Status")
    print(f"{'='*60}")
    print(f"  Agent: {status['monitoring']}")

    b = status["budget"]
    print(f"\n  Budget:")
    print(f"    Status:  {b['status']}")
    print(f"    Spend:   ${b.get('spend', 0):.2f} / ${b['budget']:.0f} ({b.get('percent', 0):.1f}%)")
    print(f"    VM:      {b.get('vm_status', 'unknown')}")

    t = status["training"]
    print(f"\n  Training:")
    if "error" in t:
        print(f"    {t['error']}")
    else:
        print(f"    Step:    {t.get('step', '?')}")
        print(f"    Loss:    {t.get('loss', '?')}")
        print(f"    LR:      {t.get('lr', '?')}")
        if "eval_ppl" in t:
            print(f"    PPL:     {t['eval_ppl']}")
    print(f"{'='*60}\n")


def main():
    p = argparse.ArgumentParser(description="Phase 1.6: Monitoring Agent")
    p.add_argument("command", choices=["start", "status", "budget"], help="Command to run")
    args = p.parse_args()

    agent = MonitoringAgent()

    if args.command == "start":
        print(f"  Starting monitoring agent (Ctrl+C to stop)...")
        agent.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            agent.stop()
            print("\n  Monitoring stopped.")

    elif args.command == "status":
        status = agent.get_status()
        print_status(status)

    elif args.command == "budget":
        budget = check_budget()
        print(f"\n  Budget: ${budget.get('spend', 0):.2f} / ${budget['budget']:.0f} ({budget.get('percent', 0):.1f}%)")
        print(f"  Status: {budget['status']}")
        print(f"  VM: {budget.get('vm_status', 'unknown')}\n")


if __name__ == "__main__":
    main()
