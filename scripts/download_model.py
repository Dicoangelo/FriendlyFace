#!/usr/bin/env python3
"""Download ONNX face recognition models for FriendlyFace.

Usage:
    python scripts/download_model.py --model mobilefacenet
    python scripts/download_model.py --model buffalo_l --dir /opt/models
    python scripts/download_model.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def _load_registry() -> dict:
    registry_path = Path(__file__).resolve().parent.parent / "models" / "MODEL_REGISTRY.json"
    if not registry_path.exists():
        print(f"Registry not found: {registry_path}", file=sys.stderr)
        sys.exit(1)
    with open(registry_path) as f:
        return json.load(f)


def _compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(model_name: str, output_dir: str) -> Path:
    registry = _load_registry()
    if model_name not in registry:
        print(f"Unknown model: {model_name}", file=sys.stderr)
        print(f"Available: {', '.join(registry.keys())}", file=sys.stderr)
        sys.exit(1)

    info = registry[model_name]
    url = info["url"]
    expected_hash = info.get("sha256")
    filename_in_zip = info.get("filename_in_zip")

    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / f"{model_name}.onnx"

    if dest_file.exists():
        print(f"Model already exists: {dest_file}")
        if expected_hash:
            actual = _compute_sha256(str(dest_file))
            if actual == expected_hash:
                print("Hash verified OK")
                return dest_file
            print(f"Hash mismatch! Re-downloading...")

    print(f"Downloading {model_name} from {url} ...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_download = os.path.join(tmpdir, "download")
        urllib.request.urlretrieve(url, tmp_download)  # noqa: S310

        if url.endswith(".zip") and filename_in_zip:
            print(f"Extracting {filename_in_zip} from zip ...")
            with zipfile.ZipFile(tmp_download) as zf:
                zf.extract(filename_in_zip, tmpdir)
            extracted = os.path.join(tmpdir, filename_in_zip)
        else:
            extracted = tmp_download

        # Verify hash
        if expected_hash:
            actual = _compute_sha256(extracted)
            if actual != expected_hash:
                print(f"HASH MISMATCH!", file=sys.stderr)
                print(f"  Expected: {expected_hash}", file=sys.stderr)
                print(f"  Got:      {actual}", file=sys.stderr)
                sys.exit(1)
            print(f"Hash verified: {actual[:16]}...")

        shutil.copy2(extracted, str(dest_file))

    size_mb = dest_file.stat().st_size / (1024 * 1024)
    print(f"Saved to {dest_file} ({size_mb:.1f} MB)")
    return dest_file


def list_models() -> None:
    registry = _load_registry()
    print(f"{'Name':<20} {'Size':>8} {'Description'}")
    print("-" * 80)
    for name, info in registry.items():
        size = f"{info.get('size_mb', '?')} MB"
        desc = info.get("description", "")
        print(f"{name:<20} {size:>8} {desc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ONNX face recognition models")
    parser.add_argument("--model", type=str, help="Model name to download")
    parser.add_argument("--dir", type=str, default="models", help="Output directory")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if not args.model:
        parser.print_help()
        sys.exit(1)

    download_model(args.model, args.dir)


if __name__ == "__main__":
    main()
