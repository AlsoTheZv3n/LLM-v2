#!/usr/bin/env python3
"""
aws.py — AWS Cloud Provider for LLM Training
==============================================
Manages EC2 GPU instances (p3/p4/g5) for training.

Status: PLACEHOLDER — not yet implemented.

Required .env keys:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_REGION=us-east-1
    AWS_INSTANCE_TYPE=p4d.24xlarge  (A100) or g5.xlarge (A10G)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)


class AWSProvider:
    """AWS EC2 GPU instance manager."""

    def __init__(self):
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.instance_type = os.getenv("AWS_INSTANCE_TYPE", "g5.xlarge")
        self.key_name = os.getenv("AWS_KEY_NAME", "")

    def create_instance(self) -> str:
        raise NotImplementedError("AWS provider not yet implemented")

    def destroy_instance(self) -> str:
        raise NotImplementedError("AWS provider not yet implemented")

    def ssh_command(self, command: str) -> str:
        raise NotImplementedError("AWS provider not yet implemented")

    def get_status(self) -> str:
        return "AWS provider not yet implemented"

    def start_training(self, config: str = "1b") -> str:
        raise NotImplementedError("AWS provider not yet implemented")

    def download_weights(self, local_dir: str = "./weights") -> str:
        raise NotImplementedError("AWS provider not yet implemented")
