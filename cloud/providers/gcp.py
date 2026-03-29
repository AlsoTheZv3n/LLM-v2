#!/usr/bin/env python3
"""
gcp_mcp_server.py — Phase 1.5: GCP Deployment MCP Server
=========================================================
MCP Server that controls GCP VMs for LLM training.
Steuerbar via Claude Desktop or any MCP client.

Tools:
    create_vm       — Create A100 Spot VM, wait until ready
    destroy_vm      — Delete VM + disk
    ssh_command     — Run bash command on VM via SSH
    start_training  — Start train.py in tmux on VM
    get_training_status — Get loss, step, ETA
    download_weights — Download checkpoint from VM
    check_budget    — Query current GCP spend
    list_checkpoints — List available checkpoints on VM

Usage:
    python gcp_mcp_server.py          # start MCP server (stdio)
    python gcp_mcp_server.py --test   # test config without creating VM
"""

import json
import os
import io
import time
import subprocess
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# ─────────────────────────────────────────────────────────────
#  Config from .env
# ─────────────────────────────────────────────────────────────
PROJECT_ID     = os.getenv("GCP_PROJECT_ID", "")
ZONE           = os.getenv("GCP_ZONE", "europe-west6-a")
VM_NAME        = os.getenv("VM_INSTANCE_NAME", "llm-pretrain-1b")
MACHINE_TYPE   = os.getenv("VM_MACHINE_TYPE", "a2-highgpu-1g")
DISK_SIZE_GB   = int(os.getenv("VM_DISK_SIZE_GB", "600"))
BUDGET_TOTAL   = float(os.getenv("BUDGET_TOTAL_USD", "200"))
SA_KEY_PATH    = str(PROJECT_ROOT / os.getenv("GCP_SERVICE_ACCOUNT_KEY_PATH", "gcp-agent-key.json"))
WANDB_KEY      = os.getenv("WANDB_API_KEY", "")
HF_TOKEN       = os.getenv("HF_TOKEN", "")

# Ensure zone has a suffix (e.g., europe-west6 -> europe-west6-a)
if ZONE and ZONE.count("-") == 1:
    ZONE = ZONE + "-a"

# ─────────────────────────────────────────────────────────────
#  VM Startup Script
# ─────────────────────────────────────────────────────────────
STARTUP_SCRIPT = f"""#!/bin/bash
set -e

# Install NVIDIA drivers + CUDA
apt-get update -y
apt-get install -y nvidia-driver-550 nvidia-utils-550
apt-get install -y python3-pip python3-venv tmux git

# Create venv
python3 -m venv /opt/llm-env
source /opt/llm-env/bin/activate

# Install PyTorch + training deps
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install tiktoken wandb datasets numpy paramiko python-dotenv matplotlib

# Set wandb key
echo 'export WANDB_API_KEY="{WANDB_KEY}"' >> /etc/profile.d/llm-env.sh
echo 'source /opt/llm-env/bin/activate' >> /etc/profile.d/llm-env.sh

# Create data directory
mkdir -p /data/tokenized /root/llm-training

echo "STARTUP COMPLETE" > /tmp/startup_done
"""

# ─────────────────────────────────────────────────────────────
#  GCP Compute Helpers
# ─────────────────────────────────────────────────────────────
def get_compute_client():
    from google.cloud import compute_v1
    from google.oauth2 import service_account
    creds = service_account.Credentials.from_service_account_file(SA_KEY_PATH)
    return compute_v1.InstancesClient(credentials=creds), creds

def get_vm_status():
    """Get current VM status (RUNNING, TERMINATED, None if not exists)."""
    try:
        client, _ = get_compute_client()
        instance = client.get(project=PROJECT_ID, zone=ZONE, instance=VM_NAME)
        return instance.status
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            return None
        raise

def wait_for_operation(operation, client, project, zone):
    """Wait for a GCP zone operation to complete."""
    from google.cloud import compute_v1
    op_client = compute_v1.ZoneOperationsClient(credentials=client._transport._credentials)
    while True:
        result = op_client.get(project=project, zone=zone, operation=operation.name)
        if result.status == compute_v1.Operation.Status.DONE:
            if result.error:
                raise Exception(f"Operation error: {result.error}")
            return result
        time.sleep(5)

# ─────────────────────────────────────────────────────────────
#  SSH Helper
# ─────────────────────────────────────────────────────────────
def ssh_exec(command: str, timeout: int = 120) -> str:
    """Execute command on VM via SSH."""
    import paramiko
    client, _ = get_compute_client()
    instance = client.get(project=PROJECT_ID, zone=ZONE, instance=VM_NAME)

    # Get external IP
    ip = None
    for iface in instance.network_interfaces:
        for access in iface.access_configs:
            if access.nat_i_p:
                ip = access.nat_i_p
                break
    if not ip:
        raise Exception("VM has no external IP")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Try SSH key from default location
    key_path = Path.home() / ".ssh" / "id_rsa"
    if not key_path.exists():
        key_path = Path.home() / ".ssh" / "id_ed25519"

    ssh.connect(ip, username="root", key_filename=str(key_path), timeout=30)

    # Activate venv before running command
    full_cmd = f"source /opt/llm-env/bin/activate && {command}"
    stdin, stdout, stderr = ssh.exec_command(full_cmd, timeout=timeout)
    output = stdout.read().decode()
    err = stderr.read().decode()
    ssh.close()

    if err and not output:
        return f"STDERR: {err}"
    return output

# ─────────────────────────────────────────────────────────────
#  MCP Tool Implementations
# ─────────────────────────────────────────────────────────────
def tool_create_vm() -> str:
    """Create A100 Spot VM, wait until ready."""
    from google.cloud import compute_v1

    status = get_vm_status()
    if status == "RUNNING":
        return f"VM '{VM_NAME}' is already running."
    if status == "TERMINATED":
        # Restart existing VM
        client, _ = get_compute_client()
        op = client.start(project=PROJECT_ID, zone=ZONE, instance=VM_NAME)
        wait_for_operation(op, client, PROJECT_ID, ZONE)
        return f"VM '{VM_NAME}' restarted (was TERMINATED)."

    client, creds = get_compute_client()

    # Build instance config
    instance = compute_v1.Instance()
    instance.name = VM_NAME
    instance.machine_type = f"zones/{ZONE}/machineTypes/{MACHINE_TYPE}"

    # Boot disk
    disk = compute_v1.AttachedDisk()
    disk.auto_delete = True
    disk.boot = True
    init = compute_v1.AttachedDiskInitializeParams()
    init.disk_size_gb = DISK_SIZE_GB
    init.source_image = "projects/ml-images/global/images/c0-deeplearning-common-gpu-v20231209-debian-11-py310"
    disk.initialize_params = init
    instance.disks = [disk]

    # Network
    net = compute_v1.NetworkInterface()
    net.name = "global/networks/default"
    access = compute_v1.AccessConfig()
    access.name = "External NAT"
    access.type_ = "ONE_TO_ONE_NAT"
    net.access_configs = [access]
    instance.network_interfaces = [net]

    # GPU: auto-detect from machine type
    # g2-standard-* has L4 built-in (no separate accelerator needed)
    # a2-highgpu-* has A100 built-in (no separate accelerator needed)
    if not MACHINE_TYPE.startswith("g2-") and not MACHINE_TYPE.startswith("a2-"):
        gpu = compute_v1.AcceleratorConfig()
        gpu.accelerator_type = f"zones/{ZONE}/acceleratorTypes/nvidia-tesla-a100"
        gpu.accelerator_count = 1
        instance.guest_accelerators = [gpu]

    # Spot VM (preemptible)
    scheduling = compute_v1.Scheduling()
    scheduling.provisioning_model = "SPOT"
    scheduling.instance_termination_action = "STOP"
    scheduling.on_host_maintenance = "TERMINATE"
    instance.scheduling = scheduling

    # Startup script
    meta = compute_v1.Metadata()
    item = compute_v1.Items()
    item.key = "startup-script"
    item.value = STARTUP_SCRIPT
    meta.items = [item]
    instance.metadata = meta

    # Service account
    sa = compute_v1.ServiceAccount()
    sa.email = "default"
    sa.scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    instance.service_accounts = [sa]

    # Create
    op = client.insert(project=PROJECT_ID, zone=ZONE, instance_resource=instance)
    wait_for_operation(op, client, PROJECT_ID, ZONE)

    # Wait for SSH readiness
    for i in range(30):
        try:
            result = ssh_exec("cat /tmp/startup_done 2>/dev/null || echo 'NOT READY'", timeout=10)
            if "STARTUP COMPLETE" in result:
                return f"VM '{VM_NAME}' created and ready! (A100 Spot in {ZONE})"
        except Exception:
            pass
        time.sleep(10)

    return f"VM '{VM_NAME}' created but startup may still be in progress."


def tool_destroy_vm(confirm: bool = False) -> str:
    """Delete VM and disk."""
    if not confirm:
        return "Safety check: pass confirm=True to destroy the VM. This deletes the VM and all its data."

    status = get_vm_status()
    if status is None:
        return f"VM '{VM_NAME}' does not exist."

    client, _ = get_compute_client()
    op = client.delete(project=PROJECT_ID, zone=ZONE, instance=VM_NAME)
    wait_for_operation(op, client, PROJECT_ID, ZONE)
    return f"VM '{VM_NAME}' destroyed (disk auto-deleted)."


def tool_ssh_command(command: str) -> str:
    """Run a bash command on the VM."""
    return ssh_exec(command)


def tool_start_training(config: str = "1b", extra_args: str = "") -> str:
    """Start training in a tmux session on the VM."""
    cmd = f"tmux kill-session -t training 2>/dev/null; "
    cmd += f"tmux new-session -d -s training "
    cmd += f"'source /opt/llm-env/bin/activate && cd /root/llm-training && "
    cmd += f"python train.py --config {config} --wandb --no-plot {extra_args} 2>&1 | tee training.log'"
    ssh_exec(cmd)
    return f"Training started in tmux session 'training' (config={config}). Use ssh_command('tmux attach -t training') to view."


def tool_get_training_status() -> str:
    """Get current training status from logs."""
    try:
        # Try to get latest from training log
        result = ssh_exec("tail -5 /root/llm-training/training.log 2>/dev/null || echo 'No training log found'")
        # Also get GPU status
        gpu = ssh_exec("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo 'N/A'")
        return f"Training Log (last 5 lines):\n{result}\n\nGPU Status: {gpu}"
    except Exception as e:
        return f"Error getting status: {e}"


def tool_download_weights(local_dir: str = "./weights") -> str:
    """Download latest checkpoint from VM."""
    import paramiko
    client, _ = get_compute_client()
    instance = client.get(project=PROJECT_ID, zone=ZONE, instance=VM_NAME)

    ip = None
    for iface in instance.network_interfaces:
        for access in iface.access_configs:
            if access.nat_i_p:
                ip = access.nat_i_p
    if not ip:
        return "VM has no external IP"

    # Find latest checkpoint
    latest = ssh_exec("ls -t /root/llm-training/runs/*/checkpoint.pt 2>/dev/null | head -1").strip()
    if not latest:
        return "No checkpoints found on VM."

    # Download via SCP
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    local_file = local_path / "checkpoint.pt"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key_path = Path.home() / ".ssh" / "id_rsa"
    if not key_path.exists():
        key_path = Path.home() / ".ssh" / "id_ed25519"
    ssh.connect(ip, username="root", key_filename=str(key_path))

    sftp = ssh.open_sftp()
    sftp.get(latest, str(local_file))
    sftp.close()
    ssh.close()

    size_mb = local_file.stat().st_size / 1024**2
    return f"Downloaded {latest} → {local_file} ({size_mb:.0f} MB)"


def tool_check_budget() -> str:
    """Check current GCP spend estimate."""
    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request
        import urllib.request

        creds = service_account.Credentials.from_service_account_file(
            SA_KEY_PATH,
            scopes=["https://www.googleapis.com/auth/cloud-billing"]
        )
        creds.refresh(Request())

        # Use billing API to get cost estimate
        # Note: Real-time billing data has ~24h delay, so we estimate
        status = get_vm_status()
        if status == "RUNNING":
            msg = f"VM Status: RUNNING (costs accruing ~$1.60/hr for {MACHINE_TYPE})\n"
        elif status == "TERMINATED":
            msg = f"VM Status: TERMINATED (disk storage cost only)\n"
        else:
            msg = f"VM Status: Not created (no costs)\n"

        msg += f"Budget: ${BUDGET_TOTAL:.0f} USD\n"
        msg += f"Note: Real-time billing data has ~24h delay. Check GCP Console for exact spend."
        return msg

    except Exception as e:
        return f"Budget check error: {e}"


def tool_list_checkpoints() -> str:
    """List available checkpoints on VM."""
    try:
        result = ssh_exec(
            "for f in /root/llm-training/runs/*/checkpoint.pt; do "
            "  [ -f \"$f\" ] && ls -lh \"$f\"; "
            "done 2>/dev/null"
        )
        if not result.strip():
            return "No checkpoints found."
        return f"Checkpoints on VM:\n{result}"
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────────────────────
#  MCP Server
# ─────────────────────────────────────────────────────────────
def run_mcp_server():
    """Start MCP server with stdio transport."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print("ERROR: mcp package not installed. Run: pip install mcp")
        return

    mcp = FastMCP("GCP LLM Training Server")

    @mcp.tool()
    def create_vm() -> str:
        """Create A100 Spot VM on GCP, wait until ready."""
        return tool_create_vm()

    @mcp.tool()
    def destroy_vm(confirm: bool = False) -> str:
        """Delete VM and all its data. Pass confirm=True to proceed."""
        return tool_destroy_vm(confirm)

    @mcp.tool()
    def ssh_command(command: str) -> str:
        """Run a bash command on the GCP VM via SSH."""
        return tool_ssh_command(command)

    @mcp.tool()
    def start_training(config: str = "1b", extra_args: str = "") -> str:
        """Start LLM training in a tmux session on the VM."""
        return tool_start_training(config, extra_args)

    @mcp.tool()
    def get_training_status() -> str:
        """Get current training step, loss, GPU status."""
        return tool_get_training_status()

    @mcp.tool()
    def download_weights(local_dir: str = "./weights") -> str:
        """Download latest checkpoint from VM to local PC."""
        return tool_download_weights(local_dir)

    @mcp.tool()
    def check_budget() -> str:
        """Check current GCP spend and budget status."""
        return tool_check_budget()

    @mcp.tool()
    def list_checkpoints() -> str:
        """List available training checkpoints on the VM."""
        return tool_list_checkpoints()

    print(f"Starting GCP MCP Server (project={PROJECT_ID}, zone={ZONE}, vm={VM_NAME})")
    mcp.run(transport="stdio")


# ─────────────────────────────────────────────────────────────
#  Test mode
# ─────────────────────────────────────────────────────────────
def test_config():
    print(f"\n{'='*60}")
    print(f"  GCP MCP Server — Config Test")
    print(f"{'='*60}")
    print(f"  Project:      {PROJECT_ID}")
    print(f"  Zone:         {ZONE}")
    print(f"  VM Name:      {VM_NAME}")
    print(f"  Machine Type: {MACHINE_TYPE}")
    print(f"  Disk Size:    {DISK_SIZE_GB} GB")
    print(f"  Budget:       ${BUDGET_TOTAL}")
    print(f"  SA Key:       {SA_KEY_PATH} ({'exists' if Path(SA_KEY_PATH).exists() else 'MISSING'})")
    print(f"  W&B Key:      {'set' if WANDB_KEY else 'NOT SET'}")
    print(f"  HF Token:     {'set' if HF_TOKEN else 'NOT SET'}")

    # Test GCP auth
    try:
        status = get_vm_status()
        print(f"  GCP Auth:     OK")
        print(f"  VM Status:    {status or 'not created'}")
    except Exception as e:
        print(f"  GCP Auth:     FAILED ({e})")

    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="GCP Deployment MCP Server")
    p.add_argument("--test", action="store_true", help="Test config without starting server")
    args = p.parse_args()

    if args.test:
        test_config()
    else:
        run_mcp_server()
