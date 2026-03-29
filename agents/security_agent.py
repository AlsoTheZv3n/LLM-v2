#!/usr/bin/env python3
"""
security_agent.py — Phase 1.8: Security Agent
==============================================
Monitors security of VM and credentials.

Usage:
    python security_agent.py scan           # full security scan
    python security_agent.py check-secrets  # check for leaked secrets
    python security_agent.py audit          # show audit log
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────
SA_KEY_PATH    = str(PROJECT_ROOT / os.getenv("GCP_SERVICE_ACCOUNT_KEY_PATH", "gcp-agent-key.json"))
PROJECT_ID     = os.getenv("GCP_PROJECT_ID", "")
ZONE           = os.getenv("GCP_ZONE", "europe-west6-a")
VM_NAME        = os.getenv("VM_INSTANCE_NAME", "llm-pretrain-1b")
AUDIT_LOG      = PROJECT_ROOT / "security_audit.jsonl"

SECRET_FILES   = [".env", "gcp-agent-key.json"]
GITIGNORE_FILE = PROJECT_ROOT / ".gitignore"


# ─────────────────────────────────────────────────────────────
#  Audit Logger
# ─────────────────────────────────────────────────────────────
def log_audit(event: str, severity: str, details: str):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        "severity": severity,
        "details": details,
    }
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    icon = {"INFO": "i", "WARN": "!", "CRITICAL": "X"}.get(severity, "?")
    print(f"  [{icon}] {severity:8s} {event}: {details}")


# ─────────────────────────────────────────────────────────────
#  Security Checks
# ─────────────────────────────────────────────────────────────
def check_gitignore() -> list[dict]:
    """Check that secret files are in .gitignore."""
    results = []
    if not GITIGNORE_FILE.exists():
        results.append({
            "check": ".gitignore exists",
            "status": "FAIL",
            "detail": ".gitignore not found — secrets may be committed!"
        })
        log_audit("gitignore_missing", "CRITICAL", ".gitignore not found")
        return results

    content = GITIGNORE_FILE.read_text()
    for sf in SECRET_FILES:
        if sf in content:
            results.append({"check": f"{sf} in .gitignore", "status": "PASS", "detail": "Listed in .gitignore"})
            log_audit("gitignore_check", "INFO", f"{sf} is in .gitignore")
        else:
            results.append({"check": f"{sf} in .gitignore", "status": "FAIL", "detail": f"{sf} NOT in .gitignore!"})
            log_audit("gitignore_missing_entry", "CRITICAL", f"{sf} not in .gitignore")

    return results


def check_git_tracking() -> list[dict]:
    """Check that secret files are not tracked by git."""
    results = []
    try:
        tracked = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, cwd="."
        ).stdout.strip().split("\n")

        for sf in SECRET_FILES:
            if sf in tracked:
                results.append({"check": f"{sf} not tracked", "status": "FAIL", "detail": f"{sf} IS tracked by git!"})
                log_audit("secret_tracked", "CRITICAL", f"{sf} is tracked in git")
            else:
                results.append({"check": f"{sf} not tracked", "status": "PASS", "detail": "Not tracked by git"})
                log_audit("secret_not_tracked", "INFO", f"{sf} not in git tracking")
    except FileNotFoundError:
        results.append({"check": "git available", "status": "SKIP", "detail": "git not found"})

    return results


def check_key_age() -> list[dict]:
    """Check if service account key is older than 90 days."""
    results = []
    key_path = Path(SA_KEY_PATH)

    if not key_path.exists():
        results.append({"check": "SA key exists", "status": "WARN", "detail": f"{SA_KEY_PATH} not found"})
        return results

    try:
        key_data = json.loads(key_path.read_text())
        # Check file modification time as proxy for creation
        mtime = os.path.getmtime(key_path)
        age_days = (time.time() - mtime) / 86400

        if age_days > 90:
            results.append({
                "check": "SA key age",
                "status": "WARN",
                "detail": f"Key is {age_days:.0f} days old — consider rotating"
            })
            log_audit("key_age_warning", "WARN", f"SA key is {age_days:.0f} days old")
        else:
            results.append({
                "check": "SA key age",
                "status": "PASS",
                "detail": f"Key is {age_days:.0f} days old"
            })
            log_audit("key_age_ok", "INFO", f"SA key is {age_days:.0f} days old")

    except Exception as e:
        results.append({"check": "SA key age", "status": "FAIL", "detail": str(e)})

    return results


def check_env_permissions() -> list[dict]:
    """Check that .env file has restrictive permissions."""
    results = []
    env_path = Path(".env")

    if not env_path.exists():
        results.append({"check": ".env exists", "status": "FAIL", "detail": ".env not found"})
        return results

    # On Windows, file permissions are different
    if os.name == "nt":
        results.append({"check": ".env permissions", "status": "SKIP", "detail": "Windows — manual check needed"})
    else:
        import stat
        mode = os.stat(env_path).st_mode
        if mode & stat.S_IROTH or mode & stat.S_IWOTH:
            results.append({
                "check": ".env permissions",
                "status": "WARN",
                "detail": f"World-readable! Run: chmod 600 .env"
            })
            log_audit("env_permissions", "WARN", ".env is world-readable")
        else:
            results.append({"check": ".env permissions", "status": "PASS", "detail": "Restricted permissions"})

    return results


def check_env_content() -> list[dict]:
    """Scan .env for common security issues."""
    results = []
    env_path = Path(".env")

    if not env_path.exists():
        return results

    content = env_path.read_text()

    # Check for obviously weak/default values
    if "changeme" in content.lower() or "password123" in content.lower():
        results.append({"check": "Weak credentials", "status": "FAIL", "detail": "Default/weak credentials found in .env"})
        log_audit("weak_credentials", "CRITICAL", "Weak credentials in .env")
    else:
        results.append({"check": "Weak credentials", "status": "PASS", "detail": "No obvious weak credentials"})

    # Check that required keys are set
    required = ["GCP_PROJECT_ID", "GCP_SERVICE_ACCOUNT_KEY_PATH"]
    for key in required:
        val = os.getenv(key, "")
        if not val:
            results.append({"check": f"{key} set", "status": "WARN", "detail": f"{key} is empty"})
        else:
            results.append({"check": f"{key} set", "status": "PASS", "detail": "Set"})

    return results


def check_budget_safety() -> list[dict]:
    """Independent budget verification (defense-in-depth)."""
    results = []
    budget = float(os.getenv("BUDGET_TOTAL_USD", "200"))
    warn = float(os.getenv("BUDGET_WARN_PERCENT", "80"))
    pause = float(os.getenv("BUDGET_PAUSE_PERCENT", "95"))
    stop = float(os.getenv("BUDGET_STOP_PERCENT", "100"))

    if budget > 500:
        results.append({"check": "Budget cap", "status": "WARN", "detail": f"Budget is ${budget} — is this intentional?"})
        log_audit("high_budget", "WARN", f"Budget set to ${budget}")
    else:
        results.append({"check": "Budget cap", "status": "PASS", "detail": f"${budget}"})

    if warn < 50 or pause < 80 or stop > 120:
        results.append({"check": "Budget thresholds", "status": "WARN", "detail": f"Unusual thresholds: warn={warn}% pause={pause}% stop={stop}%"})
    else:
        results.append({"check": "Budget thresholds", "status": "PASS", "detail": f"warn={warn}% pause={pause}% stop={stop}%"})

    return results


# ─────────────────────────────────────────────────────────────
#  Security Agent
# ─────────────────────────────────────────────────────────────
class SecurityAgent:
    """Phase 1.8: Security monitoring and scanning."""

    def __init__(self):
        pass

    def full_scan(self) -> list[dict]:
        """Run all security checks."""
        print(f"\n{'='*60}")
        print(f"  Security Scan — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}\n")

        all_results = []
        checks = [
            ("Gitignore", check_gitignore),
            ("Git Tracking", check_git_tracking),
            ("Key Age", check_key_age),
            ("Env Permissions", check_env_permissions),
            ("Env Content", check_env_content),
            ("Budget Safety", check_budget_safety),
        ]

        for name, check_fn in checks:
            print(f"\n  --- {name} ---")
            try:
                results = check_fn()
                all_results.extend(results)
            except Exception as e:
                all_results.append({"check": name, "status": "ERROR", "detail": str(e)})
                print(f"  [!] Error: {e}")

        # Summary
        passes = sum(1 for r in all_results if r["status"] == "PASS")
        warns = sum(1 for r in all_results if r["status"] == "WARN")
        fails = sum(1 for r in all_results if r["status"] == "FAIL")

        print(f"\n{'='*60}")
        print(f"  Results: {passes} PASS | {warns} WARN | {fails} FAIL")
        if fails > 0:
            print(f"  ACTION REQUIRED: {fails} critical issues found!")
        print(f"{'='*60}\n")

        log_audit("full_scan", "INFO", f"Completed: {passes} pass, {warns} warn, {fails} fail")
        return all_results

    def check_secrets(self) -> list[dict]:
        """Quick secret leak check."""
        print(f"\n  --- Secret Leak Check ---")
        results = check_gitignore() + check_git_tracking()
        fails = [r for r in results if r["status"] == "FAIL"]
        if fails:
            print(f"\n  SECRETS AT RISK!")
            for f in fails:
                print(f"    - {f['check']}: {f['detail']}")
        else:
            print(f"\n  All secrets are protected.")
        return results

    def show_audit(self, n: int = 20):
        """Show last N audit log entries."""
        if not AUDIT_LOG.exists():
            print("  No audit log found.")
            return

        lines = AUDIT_LOG.read_text().strip().split("\n")
        print(f"\n  Last {min(n, len(lines))} audit entries:")
        print(f"  {'─'*70}")
        for line in lines[-n:]:
            entry = json.loads(line)
            ts = entry["timestamp"][:19]
            sev = entry["severity"]
            evt = entry["event"]
            det = entry["details"][:50]
            print(f"  {ts}  {sev:8s}  {evt:30s}  {det}")
        print()


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Phase 1.8: Security Agent")
    p.add_argument("command", choices=["scan", "check-secrets", "audit"])
    p.add_argument("--last", type=int, default=20, help="Number of audit entries to show")
    args = p.parse_args()

    agent = SecurityAgent()

    if args.command == "scan":
        agent.full_scan()
    elif args.command == "check-secrets":
        agent.check_secrets()
    elif args.command == "audit":
        agent.show_audit(args.last)


if __name__ == "__main__":
    main()
