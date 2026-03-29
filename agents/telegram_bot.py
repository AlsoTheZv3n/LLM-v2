#!/usr/bin/env python3
"""
telegram_bot.py — Telegram Command Interface for LLM Training
==============================================================
Control the training pipeline via Telegram commands.

Commands:
    /status          — Pipeline status, VM, budget overview
    /training        — Current training progress (step, loss, ETA)
    /budget          — Budget details
    /vm              — VM status
    /start           — Start the full training pipeline
    /stop            — Stop training (graceful)
    /kill            — Destroy VM immediately
    /logs            — Last 20 lines of training log
    /checkpoint      — List available checkpoints
    /security        — Run security scan
    /help            — Show all commands

Usage:
    python infra/telegram_bot.py              # start bot (polling)
    python infra/telegram_bot.py --once       # process pending commands once, then exit
"""

import json
import os
import sys
import time
import threading
import traceback
from datetime import datetime
from pathlib import Path
from urllib import request, parse, error

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────
BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID", "")
POLL_INTERVAL = 2  # seconds between update checks

if not BOT_TOKEN:
    print("ERROR: TELEGRAM_BOT_TOKEN not set in .env")
    sys.exit(1)

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"


# ─────────────────────────────────────────────────────────────
#  Telegram API helpers
# ─────────────────────────────────────────────────────────────
def api_call(method: str, data: dict = None) -> dict:
    url = f"{API_BASE}/{method}"
    if data:
        encoded = parse.urlencode(data).encode()
        req = request.Request(url, data=encoded)
    else:
        req = request.Request(url)
    try:
        resp = request.urlopen(req, timeout=30)
        return json.loads(resp.read().decode())
    except error.HTTPError as e:
        return {"ok": False, "error": str(e)}


def send_message(chat_id: str, text: str, parse_mode: str = "Markdown"):
    # Split long messages (Telegram limit: 4096 chars)
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        api_call("sendMessage", {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": parse_mode,
        })


def send_typing(chat_id: str):
    api_call("sendChatAction", {"chat_id": chat_id, "action": "typing"})


# ─────────────────────────────────────────────────────────────
#  Lazy imports (avoid loading everything at startup)
# ─────────────────────────────────────────────────────────────
_orchestrator = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        sys.path.insert(0, str(PROJECT_ROOT))
        from cloud.orchestrator import Orchestrator
        _orchestrator = Orchestrator()
    return _orchestrator


# ─────────────────────────────────────────────────────────────
#  Command Handlers
# ─────────────────────────────────────────────────────────────
def cmd_help(chat_id, args):
    send_message(chat_id, """*LLM Training Bot Commands*

*Info:*
/status - Pipeline & VM & Budget overview
/training - Training progress (step, loss, ETA)
/budget - Detailed budget info
/vm - VM status

*Control:*
/start - Start full training pipeline
/stop - Stop training (graceful SIGINT)
/kill - Destroy VM immediately (!!!)

*Tools:*
/logs - Last 20 lines of training log
/checkpoint - List checkpoints on VM
/security - Run security scan
/ssh `command` - Run command on VM

/help - This message""")


def get_latest_run():
    """Find the most recent training run directory."""
    runs_dir = PROJECT_ROOT / "runs"
    if not runs_dir.exists():
        return None, None
    runs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        log = r / "training_log.jsonl"
        if log.exists():
            return r, log
    return None, None


def get_local_training_status():
    """Read local training logs for current progress."""
    run_dir, log_file = get_latest_run()
    if not log_file or not log_file.exists():
        return None

    # Read last few lines
    lines = log_file.read_text().strip().split("\n")
    if not lines:
        return None

    last = json.loads(lines[-1])
    first = json.loads(lines[0])

    # Check if training is recent (within last 5 min)
    import os
    mtime = log_file.stat().st_mtime
    age_sec = time.time() - mtime
    is_active = age_sec < 300  # 5 minutes

    # Read summary if exists
    summary_file = run_dir / "summary.json"
    summary = None
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())

    return {
        "run_dir": run_dir.name,
        "active": is_active,
        "age_sec": age_sec,
        "step": last.get("step", 0),
        "total_steps": last.get("total_steps", 0),
        "loss": last.get("loss", 0),
        "smooth_loss": last.get("smooth_loss", 0),
        "lr": last.get("lr", 0),
        "ms": last.get("ms", 0),
        "tokens": last.get("tokens", 0),
        "grad_norm": last.get("grad_norm", 0),
        "total_entries": len(lines),
        "summary": summary,
    }


def cmd_status(chat_id, args):
    send_typing(chat_id)

    # Check local training
    local = get_local_training_status()

    if local and local["active"]:
        pct = local["step"] / max(local["total_steps"], 1) * 100 if local["total_steps"] else 0
        tps = int(1000 / max(local["ms"], 1)) if local["ms"] else 0
        eta_sec = (local["total_steps"] - local["step"]) * local["ms"] / 1000 if local["ms"] else 0
        if eta_sec > 3600:
            eta = f"{eta_sec/3600:.1f}h"
        else:
            eta = f"{eta_sec/60:.0f}m"

        lines = [
            "*Training ACTIVE (local GPU)*",
            f"Run: `{local['run_dir']}`",
            f"Step: {local['step']:,} / {local['total_steps']:,} ({pct:.1f}%)",
            f"Loss: {local['smooth_loss']:.4f}",
            f"LR: {local['lr']:.2e}",
            f"Speed: {tps} tok/s ({local['ms']:.0f} ms/step)",
            f"Tokens: {local['tokens']:,}",
            f"Grad norm: {local['grad_norm']:.3f}",
            f"ETA: ~{eta}",
        ]
    elif local and not local["active"]:
        lines = [
            "*Training COMPLETED*",
            f"Run: `{local['run_dir']}`",
            f"Final step: {local['step']:,}",
            f"Final loss: {local['smooth_loss']:.4f}",
            f"Finished {local['age_sec']/60:.0f} min ago",
        ]
        if local.get("summary"):
            s = local["summary"]
            lines.append(f"Eval PPL: {s.get('eval', {}).get('perplexity', 'N/A')}")
    else:
        lines = ["*No active training run*", "Start with: /start"]

    # Budget
    try:
        orch = get_orchestrator()
        budget = orch.budget.check()
        lines.extend([
            "",
            f"Budget: ${budget.get('spend', 0):.2f} / ${budget['budget']:.0f}",
        ])
    except Exception:
        pass

    send_message(chat_id, "\n".join(lines))


def cmd_training(chat_id, args):
    send_typing(chat_id)
    local = get_local_training_status()
    if local:
        pct = local["step"] / max(local["total_steps"], 1) * 100 if local["total_steps"] else 0
        bar_len = 20
        filled = int(pct / 100 * bar_len)
        bar = "#" * filled + "." * (bar_len - filled)

        lines = [
            f"*Training {'ACTIVE' if local['active'] else 'DONE'}*",
            f"`[{bar}] {pct:.1f}%`",
            f"Step: {local['step']:,} / {local['total_steps']:,}",
            f"Loss: {local['smooth_loss']:.4f}",
            f"LR: {local['lr']:.2e}",
            f"Grad norm: {local['grad_norm']:.3f}",
            f"Tokens: {local['tokens']:,}",
        ]
        if local["active"] and local["ms"]:
            eta_sec = (local["total_steps"] - local["step"]) * local["ms"] / 1000
            if eta_sec > 3600:
                eta = f"{eta_sec/3600:.1f}h"
            else:
                eta = f"{eta_sec/60:.0f}m"
            lines.append(f"ETA: ~{eta}")
        send_message(chat_id, "\n".join(lines))
    else:
        # Fall back to remote
        try:
            orch = get_orchestrator()
            status = orch.gcp.get_status()
            send_message(chat_id, f"*Training Status*\n```\n{status}\n```")
        except Exception as e:
            send_message(chat_id, f"No active training found locally or remotely.")


def cmd_budget(chat_id, args):
    send_typing(chat_id)
    orch = get_orchestrator()
    b = orch.budget.check()
    lines = [
        "*Budget Status*",
        f"Spend: ${b.get('spend', 0):.2f} / ${b['budget']:.0f}",
        f"Used: {b.get('percent', 0):.1f}%",
        f"Status: `{b.get('status', 'OK')}`",
        "",
        f"Warn at: {b.get('warn_pct', 80)}%",
        f"Pause at: {b.get('pause_pct', 95)}%",
        f"Stop at: {b.get('stop_pct', 100)}%",
    ]
    send_message(chat_id, "\n".join(lines))


def cmd_vm(chat_id, args):
    send_typing(chat_id)
    try:
        orch = get_orchestrator()
        vm = orch.gcp.vm_status()
        send_message(chat_id, f"*VM Status:* `{vm or 'not created'}`")
    except Exception as e:
        send_message(chat_id, f"*VM Status:* `not created / error: {e}`")


def cmd_start(chat_id, args):
    config = args.strip() if args else "1b"
    send_message(chat_id, f"Starting pipeline with config `{config}`...\nThis will create a VM and start training.")

    def _run():
        try:
            orch = get_orchestrator()
            orch.run_pipeline(config=config)
            send_message(chat_id, "Pipeline completed successfully!")
        except Exception as e:
            send_message(chat_id, f"Pipeline error: `{e}`")

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def cmd_stop(chat_id, args):
    send_typing(chat_id)
    try:
        orch = get_orchestrator()
        result = orch.gcp.ssh("tmux send-keys -t training C-c")
        send_message(chat_id, f"Sent stop signal (SIGINT) to training.\n```\n{result}\n```")
    except Exception as e:
        send_message(chat_id, f"Could not stop training: `{e}`")


def cmd_kill(chat_id, args):
    send_message(chat_id, "Destroying VM...")
    try:
        orch = get_orchestrator()
        result = orch.gcp.destroy_vm()
        send_message(chat_id, f"VM destroyed.\n```\n{result}\n```")
    except Exception as e:
        send_message(chat_id, f"Error destroying VM: `{e}`")


def cmd_logs(chat_id, args):
    send_typing(chat_id)
    n = int(args.strip()) if args and args.strip().isdigit() else 10

    # Try local first
    run_dir, log_file = get_latest_run()
    if log_file and log_file.exists():
        lines = log_file.read_text().strip().split("\n")
        recent = lines[-n:]
        formatted = []
        for line in recent:
            try:
                d = json.loads(line)
                formatted.append(f"s{d.get('step',0):>6} | loss {d.get('loss',0):.4f} | lr {d.get('lr',0):.1e} | {d.get('ms',0):.0f}ms")
            except:
                formatted.append(line[:80])
        send_message(chat_id, f"*Last {n} entries ({run_dir.name}):*\n```\n" + "\n".join(formatted) + "\n```")
        return

    # Fall back to remote
    try:
        orch = get_orchestrator()
        result = orch.gcp.ssh(f"tail -n {n} /root/llm-training/training.log 2>/dev/null || echo 'No log found'")
        send_message(chat_id, f"*Last {n} lines:*\n```\n{result}\n```")
    except Exception as e:
        send_message(chat_id, f"No logs found locally or remotely.")


def cmd_checkpoint(chat_id, args):
    send_typing(chat_id)
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from cloud.providers.gcp import tool_list_checkpoints
        result = tool_list_checkpoints()
        send_message(chat_id, f"*Checkpoints:*\n```\n{result}\n```")
    except Exception as e:
        send_message(chat_id, f"Could not list checkpoints: `{e}`")


def cmd_security(chat_id, args):
    send_typing(chat_id)
    try:
        orch = get_orchestrator()
        result = orch.security.run_scan()
        send_message(chat_id, f"*Security Scan:*\n```\n{result}\n```")
    except Exception as e:
        send_message(chat_id, f"Security scan error: `{e}`")


def cmd_ssh(chat_id, args):
    if not args or not args.strip():
        send_message(chat_id, "Usage: /ssh `command`\nExample: /ssh nvidia-smi")
        return
    send_typing(chat_id)
    try:
        orch = get_orchestrator()
        result = orch.gcp.ssh(args.strip())
        send_message(chat_id, f"```\n{result[:3500]}\n```")
    except Exception as e:
        send_message(chat_id, f"SSH error: `{e}`")


# ─────────────────────────────────────────────────────────────
#  Command Router
# ─────────────────────────────────────────────────────────────
COMMANDS = {
    "/help":       cmd_help,
    "/start":      cmd_start,
    "/status":     cmd_status,
    "/training":   cmd_training,
    "/budget":     cmd_budget,
    "/vm":         cmd_vm,
    "/stop":       cmd_stop,
    "/kill":       cmd_kill,
    "/logs":       cmd_logs,
    "/checkpoint": cmd_checkpoint,
    "/security":   cmd_security,
    "/ssh":        cmd_ssh,
}


def handle_message(message: dict):
    chat_id = str(message.get("chat", {}).get("id", ""))
    text = message.get("text", "").strip()

    # Only respond to authorized chat
    if CHAT_ID and chat_id != CHAT_ID:
        send_message(chat_id, "Unauthorized. This bot only responds to its owner.")
        return

    if not text.startswith("/"):
        send_message(chat_id, "Send /help to see available commands.")
        return

    # Parse command and args
    parts = text.split(None, 1)
    cmd = parts[0].lower().split("@")[0]  # strip @botname suffix
    args = parts[1] if len(parts) > 1 else ""

    handler = COMMANDS.get(cmd)
    if handler:
        try:
            handler(chat_id, args)
        except Exception as e:
            send_message(chat_id, f"Error: `{e}`\n```\n{traceback.format_exc()[-500:]}\n```")
    else:
        send_message(chat_id, f"Unknown command: `{cmd}`\nSend /help for available commands.")


# ─────────────────────────────────────────────────────────────
#  Polling Loop
# ─────────────────────────────────────────────────────────────
def run_bot(once: bool = False):
    print(f"LLM Training Bot started — listening for commands...")
    print(f"  Bot: @LLM_helper_sven_bot")
    print(f"  Chat ID: {CHAT_ID}")
    print(f"  Press Ctrl+C to stop\n")

    offset = 0

    while True:
        try:
            result = api_call("getUpdates", {
                "offset": offset,
                "timeout": 30 if not once else 0,
            })

            if result.get("ok"):
                for update in result.get("result", []):
                    offset = update["update_id"] + 1
                    msg = update.get("message")
                    if msg:
                        handle_message(msg)

            if once:
                break

        except KeyboardInterrupt:
            print("\nBot stopped.")
            break
        except Exception as e:
            print(f"Polling error: {e}")
            time.sleep(5)

        if not once:
            time.sleep(POLL_INTERVAL)


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Telegram Bot for LLM Training")
    p.add_argument("--once", action="store_true", help="Process pending commands once, then exit")
    args = p.parse_args()

    run_bot(once=args.once)
