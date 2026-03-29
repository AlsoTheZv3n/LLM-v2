#!/usr/bin/env python3
"""
orchestrator.py — Phase 1.9: Agent Coordinator
===============================================
Coordinates all agents and manages the full training pipeline.

Usage:
    python orchestrator.py status          # show all agent status
    python orchestrator.py start-pipeline  # run full training pipeline
    python orchestrator.py start-agents    # start monitoring/doc/security agents
    python orchestrator.py stop-agents     # stop all agents
"""

import argparse
import enum
import json
import os
import time
import threading
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

import sys
sys.path.insert(0, str(PROJECT_ROOT))
from agents.monitoring_agent import MonitoringAgent, check_budget
from agents.doc_agent import DocumentationAgent
from agents.security_agent import SecurityAgent

# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────
PROJECT_ID   = os.getenv("GCP_PROJECT_ID", "")
ZONE         = os.getenv("GCP_ZONE", "europe-west6-a")
VM_NAME      = os.getenv("VM_INSTANCE_NAME", "llm-pretrain-1b")
BUDGET_TOTAL = float(os.getenv("BUDGET_TOTAL_USD", "200"))
SA_KEY_PATH  = str(PROJECT_ROOT / os.getenv("GCP_SERVICE_ACCOUNT_KEY_PATH", "gcp-agent-key.json"))

MAX_PREEMPT_RETRIES = 3

# ─────────────────────────────────────────────────────────────
#  Pipeline State Machine
# ─────────────────────────────────────────────────────────────
class PipelineState(enum.Enum):
    IDLE           = "idle"
    PROVISIONING   = "provisioning"
    GPU_CHECK      = "gpu_check"
    UPLOADING      = "uploading_code"
    TOKENIZING     = "tokenizing"
    TRAINING       = "training"
    DOWNLOADING    = "downloading_weights"
    CLEANUP        = "cleanup"
    COMPLETED      = "completed"
    ERROR          = "error"

VALID_TRANSITIONS = {
    PipelineState.IDLE:         [PipelineState.PROVISIONING],
    PipelineState.PROVISIONING: [PipelineState.GPU_CHECK, PipelineState.ERROR],
    PipelineState.GPU_CHECK:    [PipelineState.UPLOADING, PipelineState.ERROR],
    PipelineState.UPLOADING:    [PipelineState.TOKENIZING, PipelineState.TRAINING, PipelineState.ERROR],
    PipelineState.TOKENIZING:   [PipelineState.TRAINING, PipelineState.ERROR],
    PipelineState.TRAINING:     [PipelineState.DOWNLOADING, PipelineState.PROVISIONING, PipelineState.ERROR],
    PipelineState.DOWNLOADING:  [PipelineState.CLEANUP, PipelineState.ERROR],
    PipelineState.CLEANUP:      [PipelineState.COMPLETED, PipelineState.ERROR],
    PipelineState.COMPLETED:    [PipelineState.IDLE],
    PipelineState.ERROR:        [PipelineState.IDLE, PipelineState.PROVISIONING],
}


# ─────────────────────────────────────────────────────────────
#  GCP Operations (wraps gcp_mcp_server tools)
# ─────────────────────────────────────────────────────────────
class GCPOps:
    """Wrapper around GCP operations."""

    @staticmethod
    def create_vm() -> str:
        from cloud.providers.gcp import tool_create_vm
        return tool_create_vm()

    @staticmethod
    def destroy_vm() -> str:
        from cloud.providers.gcp import tool_destroy_vm
        return tool_destroy_vm(confirm=True)

    @staticmethod
    def ssh(command: str) -> str:
        from cloud.providers.gcp import tool_ssh_command
        return tool_ssh_command(command)

    @staticmethod
    def start_training(config: str = "1b") -> str:
        from cloud.providers.gcp import tool_start_training
        return tool_start_training(config)

    @staticmethod
    def get_status() -> str:
        from cloud.providers.gcp import tool_get_training_status
        return tool_get_training_status()

    @staticmethod
    def download_weights(local_dir: str = "./weights") -> str:
        from cloud.providers.gcp import tool_download_weights
        return tool_download_weights(local_dir)

    @staticmethod
    def vm_status() -> str:
        from cloud.providers.gcp import get_vm_status
        return get_vm_status()


# ─────────────────────────────────────────────────────────────
#  Budget Gate
# ─────────────────────────────────────────────────────────────
class BudgetGate:
    """Blocks actions if budget exceeded."""

    def __init__(self):
        self.budget = BUDGET_TOTAL
        self.warn_pct = float(os.getenv("BUDGET_WARN_PERCENT", "80"))
        self.pause_pct = float(os.getenv("BUDGET_PAUSE_PERCENT", "95"))
        self.stop_pct = float(os.getenv("BUDGET_STOP_PERCENT", "100"))

    def check(self) -> dict:
        return check_budget()

    def can_proceed(self) -> bool:
        status = self.check()
        return status["status"] not in ("PAUSE", "STOP")

    def should_stop(self) -> bool:
        status = self.check()
        return status["status"] == "STOP"


# ─────────────────────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────────────────────
class Orchestrator:
    """Phase 1.9: Coordinates all agents and the training pipeline."""

    def __init__(self):
        self.state = PipelineState.IDLE
        self.state_history: list[dict] = []
        self.monitor = MonitoringAgent()
        self.doc = DocumentationAgent()
        self.security = SecurityAgent()
        self.budget = BudgetGate()
        self.gcp = GCPOps()
        self.preempt_count = 0

    def _set_state(self, new_state: PipelineState, reason: str = ""):
        old = self.state
        if new_state not in VALID_TRANSITIONS.get(old, []):
            print(f"  [orchestrator] Invalid transition: {old.value} -> {new_state.value}")
            return False
        self.state = new_state
        self.state_history.append({
            "timestamp": datetime.now().isoformat(),
            "from": old.value,
            "to": new_state.value,
            "reason": reason,
        })
        print(f"  [orchestrator] {old.value} -> {new_state.value} ({reason})")
        return True

    # ── Agent Management ──────────────────────────────────────

    def start_agents(self):
        """Start all background agents."""
        print(f"\n  Starting agents...")
        self.monitor.start()
        self.doc.start()
        # Security agent runs on-demand, not continuously
        print(f"  All agents started.\n")

    def stop_agents(self):
        """Stop all background agents."""
        self.monitor.stop()
        self.doc.stop()
        print(f"  All agents stopped.")

    # ── Pipeline ──────────────────────────────────────────────

    def run_pipeline(self, config: str = "1b", skip_tokenize: bool = False):
        """Run the full training pipeline."""
        print(f"\n{'='*60}")
        print(f"  LLM Training Pipeline")
        print(f"  Config: {config} | Budget: ${BUDGET_TOTAL}")
        print(f"{'='*60}\n")

        # Pre-flight security scan
        print("  [1/7] Security scan...")
        self.security.full_scan()

        # Budget check
        if not self.budget.can_proceed():
            print("  BLOCKED: Budget exceeded. Cannot proceed.")
            return

        self.start_agents()

        try:
            # Step 1: Provision VM
            print("\n  [2/7] Provisioning VM...")
            self._set_state(PipelineState.PROVISIONING, "creating VM")
            result = self.gcp.create_vm()
            print(f"  {result}")
            self.doc.post_milestone(f"VM created: {VM_NAME}")

            # Step 2: GPU check
            print("\n  [3/7] GPU healthcheck...")
            self._set_state(PipelineState.GPU_CHECK, "checking GPU")
            gpu_result = self.gcp.ssh("bash /root/llm-training/check_gpu.sh")
            print(gpu_result)

            # Step 3: Upload code
            print("\n  [4/7] Uploading training code...")
            self._set_state(PipelineState.UPLOADING, "uploading code")
            # Upload all .py files and check_gpu.sh
            files_to_upload = [
                "training/train.py", "training/tokenize_data.py",
                "cloud/check_gpu.sh", "requirements.txt", ".env",
            ]
            for f in files_to_upload:
                if Path(f).exists():
                    self.gcp.ssh(f"cat > /root/llm-training/{f} << 'HEREDOC'\n{Path(f).read_text()}\nHEREDOC")
            self.gcp.ssh("cd /root/llm-training && pip install -r requirements.txt")

            # Step 4: Tokenize (optional)
            if not skip_tokenize:
                print("\n  [5/7] Tokenizing datasets...")
                self._set_state(PipelineState.TOKENIZING, "tokenizing")
                self.gcp.ssh("cd /root/llm-training && python tokenize_data.py --all --output /data/tokenized/")
                self.doc.post_milestone("Tokenization complete")
            else:
                print("\n  [5/7] Skipping tokenization (--skip-tokenize)")

            # Step 5: Train
            print("\n  [6/7] Starting training...")
            self._set_state(PipelineState.TRAINING, f"training config={config}")
            self.gcp.start_training(config)
            self.doc.post_milestone(f"Training started (config={config})")

            # Monitor training
            self._monitor_training()

            # Step 6: Download weights
            print("\n  [7/7] Downloading weights...")
            self._set_state(PipelineState.DOWNLOADING, "downloading weights")
            dl_result = self.gcp.download_weights()
            print(f"  {dl_result}")
            self.doc.post_milestone("Weights downloaded")

            # Cleanup
            self._set_state(PipelineState.CLEANUP, "destroying VM")
            self.gcp.destroy_vm()
            self.doc.post_milestone("VM destroyed, training complete")

            self._set_state(PipelineState.COMPLETED, "pipeline done")
            print(f"\n  Pipeline completed successfully!")

        except Exception as e:
            self._set_state(PipelineState.ERROR, str(e))
            self.doc.post_incident(f"Pipeline error: {e}")
            print(f"\n  Pipeline ERROR: {e}")

        finally:
            self.stop_agents()

    def _monitor_training(self):
        """Poll training status until completion or preemption."""
        print("  Monitoring training progress...")
        while True:
            time.sleep(60)

            # Budget check
            if self.budget.should_stop():
                print("  Budget limit reached — stopping training!")
                self.doc.post_incident("Budget limit reached, stopping training")
                break

            # Check VM status
            vm_status = self.gcp.vm_status()
            if vm_status == "TERMINATED":
                # Spot VM preempted
                self.preempt_count += 1
                if self.preempt_count <= MAX_PREEMPT_RETRIES:
                    print(f"  VM preempted! Retry {self.preempt_count}/{MAX_PREEMPT_RETRIES}")
                    self.doc.post_incident(f"Spot VM preempted (retry {self.preempt_count})")
                    self._set_state(PipelineState.PROVISIONING, "preemption recovery")
                    self.gcp.create_vm()
                    self._set_state(PipelineState.TRAINING, "resuming training")
                    self.gcp.start_training("1b")
                    continue
                else:
                    print(f"  Max preemption retries reached!")
                    self.doc.post_incident("Max preemption retries reached")
                    break

            # Check if training is done (look for "Run complete" in log)
            try:
                status = self.gcp.get_status()
                if "Run complete" in status:
                    print("  Training completed!")
                    break
            except Exception:
                pass

    # ── Status ────────────────────────────────────────────────

    def print_status(self):
        """Print formatted status dashboard."""
        budget = self.budget.check()
        vm = self.gcp.vm_status()

        print(f"\n{'='*60}")
        print(f"  Orchestrator Status Dashboard")
        print(f"{'='*60}")
        print(f"\n  Pipeline State: {self.state.value}")
        print(f"  VM Status:      {vm or 'not created'}")
        print(f"  Budget:         ${budget.get('spend', 0):.2f} / ${budget['budget']:.0f} ({budget.get('percent', 0):.1f}%)")
        print(f"  Preemptions:    {self.preempt_count}")

        print(f"\n  Agents:")
        print(f"    Monitor:      {'running' if self.monitor.is_running else 'stopped'}")
        print(f"    Documentation:{'running' if self.doc.is_running else 'stopped'}")
        print(f"    Security:     on-demand")

        if self.state_history:
            print(f"\n  State History (last 5):")
            for entry in self.state_history[-5:]:
                ts = entry["timestamp"][:19]
                print(f"    {ts}  {entry['from']:20s} -> {entry['to']:20s}  ({entry['reason']})")

        print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Phase 1.9: Orchestrator")
    p.add_argument("command", choices=["status", "start-pipeline", "start-agents", "stop-agents"])
    p.add_argument("--config", default="1b", help="Training config (default: 1b)")
    p.add_argument("--skip-tokenize", action="store_true", help="Skip tokenization step")
    args = p.parse_args()

    orch = Orchestrator()

    if args.command == "status":
        orch.print_status()

    elif args.command == "start-pipeline":
        orch.run_pipeline(config=args.config, skip_tokenize=args.skip_tokenize)

    elif args.command == "start-agents":
        orch.start_agents()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            orch.stop_agents()

    elif args.command == "stop-agents":
        orch.stop_agents()


if __name__ == "__main__":
    main()
