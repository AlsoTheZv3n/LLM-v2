#!/usr/bin/env python3
"""
lambda_cloud.py — Lambda Cloud Provider for LLM Training
==========================================================
Manages Lambda Cloud GPU instances via their REST API.

Required .env keys:
    LAMBDA_CLOUD_API_KEY=secret_...

API Docs: https://cloud.lambda.ai/api/v1
"""

import json
import os
from pathlib import Path
from urllib import request, error

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

API_BASE = "https://cloud.lambda.ai/api/v1"


class LambdaProvider:
    """Lambda Cloud GPU instance manager."""

    def __init__(self):
        self.api_key = os.getenv("LAMBDA_CLOUD_API_KEY", "")
        if not self.api_key:
            raise ValueError("LAMBDA_CLOUD_API_KEY not set in .env")

    def _api(self, endpoint: str, method: str = "GET", data: dict = None) -> dict:
        url = f"{API_BASE}/{endpoint}"
        headers = {"Authorization": f"Basic {self.api_key}:"}
        if data:
            body = json.dumps(data).encode()
            headers["Content-Type"] = "application/json"
            req = request.Request(url, data=body, headers=headers, method=method)
        else:
            req = request.Request(url, headers=headers, method=method)
        resp = request.urlopen(req, timeout=30)
        return json.loads(resp.read().decode())

    def list_available(self) -> list[dict]:
        """List available GPU instance types with pricing."""
        data = self._api("instance-types")
        available = []
        for key, val in data.get("data", {}).items():
            info = val.get("instance_type", {})
            regions = val.get("regions_with_capacity_available", [])
            available.append({
                "id": key,
                "description": info.get("description", key),
                "price_per_hour": info.get("price_cents_per_hour", 0) / 100,
                "available": len(regions) > 0,
                "regions": [r.get("name", "") for r in regions],
            })
        return available

    def list_instances(self) -> list[dict]:
        """List running instances."""
        data = self._api("instances")
        return data.get("data", [])

    def create_instance(self, instance_type: str = "gpu_1x_a100_sxm4",
                        region: str = "us-east-1", name: str = "llm-training") -> dict:
        """Launch a new GPU instance."""
        payload = {
            "region_name": region,
            "instance_type_name": instance_type,
            "ssh_key_names": [],
            "name": name,
        }
        return self._api("instance-operations/launch", method="POST", data=payload)

    def destroy_instance(self, instance_id: str) -> dict:
        """Terminate an instance."""
        payload = {"instance_ids": [instance_id]}
        return self._api("instance-operations/terminate", method="POST", data=payload)

    def get_status(self) -> str:
        """Get status of all instances."""
        instances = self.list_instances()
        if not instances:
            return "No Lambda Cloud instances running"
        lines = []
        for inst in instances:
            lines.append(f"  {inst.get('name', '?')}: {inst.get('status', '?')} "
                        f"({inst.get('instance_type', {}).get('description', '?')})")
        return "\n".join(lines)

    def check_availability(self) -> str:
        """Check which GPU types are available right now."""
        types = self.list_available()
        lines = ["Lambda Cloud GPU Availability:"]
        for t in sorted(types, key=lambda x: x["price_per_hour"]):
            status = "AVAILABLE" if t["available"] else "sold out"
            regions = ", ".join(t["regions"][:3]) if t["available"] else ""
            lines.append(f"  {t['description']:<35} ${t['price_per_hour']:.2f}/h  {status} {regions}")
        return "\n".join(lines)
