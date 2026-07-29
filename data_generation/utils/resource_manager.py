"""Resource manager — model/data download and upload utilities.

Uses huggingface_hub for model downloads and boto3 for S3 operations.
Falls back to local paths when available.
"""

import os
import subprocess
from typing import Optional, Dict, Any


def ensure_model_available(model_cfg: Dict[str, Any]) -> str:
    """Ensure model weights are available locally. Returns local path or HF ID.

    Resolution order:
    1. local_cache exists → return it
    2. s3_cache specified → download to local_cache
    3. hf_id → download via huggingface_hub to local_cache
    4. hf_id as-is → let vLLM handle the download
    """
    local_cache = model_cfg.get("local_cache", "")
    s3_cache = model_cfg.get("s3_cache", "")
    hf_id = model_cfg.get("hf_id", "")

    # Already available locally
    if local_cache and os.path.isdir(local_cache) and os.listdir(local_cache):
        print(f"Model found at local cache: {local_cache}")
        return local_cache

    # Try S3
    if s3_cache and local_cache:
        print(f"Downloading model from S3: {s3_cache} -> {local_cache}")
        os.makedirs(local_cache, exist_ok=True)
        try:
            _s3_sync(s3_cache, local_cache)
            return local_cache
        except Exception as e:
            print(f"  S3 download failed: {e}, trying HuggingFace...")

    # Try HuggingFace
    if hf_id and local_cache:
        print(f"Downloading model from HuggingFace: {hf_id} -> {local_cache}")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(hf_id, local_dir=local_cache)
            return local_cache
        except Exception as e:
            print(f"  HF download failed: {e}")

    # Fallback: return hf_id and let vLLM download
    if hf_id:
        return hf_id

    raise ValueError("No valid model path: set hf_id, local_cache, or s3_cache in configs/paths.yaml")


def upload_file(local_path: str, s3_path: str, s3_cfg: Optional[Dict] = None):
    """Upload a local file to S3."""
    try:
        import boto3
        endpoint_url = (s3_cfg or {}).get("endpoint_url")
        region = (s3_cfg or {}).get("region", "us-east-1")
        s3 = boto3.client("s3", endpoint_url=endpoint_url, region_name=region)

        bucket, key = _parse_s3_path(s3_path)
        s3.upload_file(local_path, bucket, key)
    except ImportError:
        # Fallback to AWS CLI
        subprocess.run(["aws", "s3", "cp", local_path, s3_path], check=True)


def download_file(s3_path: str, local_path: str, s3_cfg: Optional[Dict] = None):
    """Download a file from S3."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        import boto3
        endpoint_url = (s3_cfg or {}).get("endpoint_url")
        region = (s3_cfg or {}).get("region", "us-east-1")
        s3 = boto3.client("s3", endpoint_url=endpoint_url, region_name=region)

        bucket, key = _parse_s3_path(s3_path)
        s3.download_file(bucket, key, local_path)
    except ImportError:
        subprocess.run(["aws", "s3", "cp", s3_path, local_path], check=True)


def _s3_sync(s3_path: str, local_path: str):
    """Sync an S3 directory to local using aws CLI."""
    os.makedirs(local_path, exist_ok=True)
    subprocess.run(
        ["aws", "s3", "sync", s3_path, local_path, "--quiet"],
        check=True,
    )


def _parse_s3_path(s3_path: str):
    """Parse s3://bucket/key into (bucket, key)."""
    path = s3_path.replace("s3://", "")
    bucket = path.split("/")[0]
    key = "/".join(path.split("/")[1:])
    return bucket, key
