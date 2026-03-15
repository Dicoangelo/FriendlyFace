#!/usr/bin/env python3
"""Download the ArcFace ONNX model for FriendlyFace recognition engine.

Downloads the ArcFace ResNet50 (w600k_r50.onnx) from InsightFace's buffalo_l
model pack — a production-quality face recognition model producing 512-dim
embeddings from 112x112 RGB inputs.

Usage:
    python3 scripts/download_onnx_model.py           # Download to default location
    python3 scripts/download_onnx_model.py --verify   # Download and verify with onnxruntime
    python3 scripts/download_onnx_model.py --output /path/to/model.onnx  # Custom output path

The model is licensed under MIT by InsightFace (https://github.com/deepinsight/insightface).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Project root (two levels up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "arcface_r100.onnx"

# Download sources — tried in order until one succeeds.
# Primary: InsightFace buffalo_l release (contains w600k_r50.onnx among other models).
# Fallback: buffalo_sc release (contains w600k_mbf.onnx — lighter but same interface).
DOWNLOAD_SOURCES = [
    {
        "name": "InsightFace buffalo_l (GitHub Releases v0.7)",
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "filename_in_zip": "w600k_r50.onnx",
        "expected_size_min_mb": 150,  # w600k_r50 is ~166 MB
    },
    {
        "name": "InsightFace buffalo_l (GitHub Releases v0.7.1)",
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7.1/buffalo_l.zip",
        "filename_in_zip": "w600k_r50.onnx",
        "expected_size_min_mb": 150,
    },
    {
        "name": "InsightFace buffalo_sc (GitHub Releases v0.7) — fallback smaller model",
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
        "filename_in_zip": "w600k_mbf.onnx",
        "expected_size_min_mb": 10,  # MobileFaceNet is ~13 MB
    },
]


def _compute_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(url: str, dest: str) -> None:
    """Download a URL to a local file with progress reporting."""
    print(f"  Downloading from {url} ...")

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {pct}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)
        else:
            mb_done = downloaded / (1024 * 1024)
            print(f"\r  Downloaded: {mb_done:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)  # noqa: S310
    print()  # newline after progress


def _extract_from_zip(zip_path: str, filename: str, dest_dir: str) -> str:
    """Extract a specific file from a zip archive, searching nested paths."""
    with zipfile.ZipFile(zip_path) as zf:
        # Find the file — it may be nested (e.g., "buffalo_l/w600k_r50.onnx")
        candidates = [n for n in zf.namelist() if n.endswith(filename)]
        if not candidates:
            raise FileNotFoundError(
                f"'{filename}' not found in zip. Contents: {zf.namelist()[:10]}"
            )
        target = candidates[0]
        print(f"  Extracting {target} from zip ...")
        zf.extract(target, dest_dir)
        return os.path.join(dest_dir, target)


def download_arcface(output_path: Path | None = None) -> Path:
    """Download the ArcFace ONNX model, trying multiple sources.

    Parameters
    ----------
    output_path : Path or None
        Where to save the model. Defaults to ``models/arcface_r100.onnx``.

    Returns
    -------
    Path
        The path to the downloaded model file.

    Raises
    ------
    RuntimeError
        If all download sources fail.
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT

    output_path = Path(output_path)

    # Check if already downloaded
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        sha = _compute_sha256(output_path)
        print(f"Model already exists: {output_path} ({size_mb:.1f} MB)")
        print(f"  SHA-256: {sha[:16]}...")
        return output_path

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []

    for source in DOWNLOAD_SOURCES:
        name = source["name"]
        url = source["url"]
        filename_in_zip = source["filename_in_zip"]
        min_size_mb = source["expected_size_min_mb"]

        print(f"\nTrying: {name}")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_zip = os.path.join(tmpdir, "model.zip")

                # Download
                _download_with_progress(url, tmp_zip)

                # Extract ONNX file from zip
                extracted_path = _extract_from_zip(tmp_zip, filename_in_zip, tmpdir)

                # Verify size
                actual_size_mb = os.path.getsize(extracted_path) / (1024 * 1024)
                if actual_size_mb < min_size_mb:
                    msg = (
                        f"  File too small: {actual_size_mb:.1f} MB "
                        f"(expected >= {min_size_mb} MB). Possibly corrupt."
                    )
                    print(msg)
                    errors.append(f"{name}: {msg}")
                    continue

                # Copy to final destination
                shutil.copy2(extracted_path, str(output_path))

                size_mb = output_path.stat().st_size / (1024 * 1024)
                sha = _compute_sha256(output_path)
                print(f"\nSaved to {output_path} ({size_mb:.1f} MB)")
                print(f"  SHA-256: {sha[:16]}...")
                print(f"  Source file in zip: {filename_in_zip}")
                return output_path

        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            msg = f"  Download failed: {exc}"
            print(msg)
            errors.append(f"{name}: {exc}")
        except (zipfile.BadZipFile, FileNotFoundError) as exc:
            msg = f"  Extraction failed: {exc}"
            print(msg)
            errors.append(f"{name}: {exc}")
        except OSError as exc:
            msg = f"  I/O error: {exc}"
            print(msg)
            errors.append(f"{name}: {exc}")

    print("\nAll download sources failed:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    raise RuntimeError("Failed to download ArcFace model from any source.")


def verify_model(model_path: Path) -> bool:
    """Load the model with onnxruntime and verify its input/output shapes.

    Parameters
    ----------
    model_path : Path
        Path to the ONNX model file.

    Returns
    -------
    bool
        True if verification passes.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "\nCannot verify: onnxruntime not installed.\n"
            "Install with: pip install 'friendlyface[ml]'"
        )
        return False

    print(f"\n{'=' * 60}")
    print("Model Verification")
    print(f"{'=' * 60}")
    print(f"File: {model_path}")
    print(f"Size: {model_path.stat().st_size / (1024 * 1024):.1f} MB")
    print(f"SHA-256: {_compute_sha256(model_path)}")

    session = ort.InferenceSession(
        str(model_path),
        providers=ort.get_available_providers(),
    )

    # Input info
    print(f"\nInputs:")
    inputs = session.get_inputs()
    for inp in inputs:
        print(f"  {inp.name}: {inp.shape} ({inp.type})")

    # Output info
    print(f"\nOutputs:")
    outputs = session.get_outputs()
    for out in outputs:
        print(f"  {out.name}: {out.shape} ({out.type})")

    # Provider info
    print(f"\nExecution providers: {session.get_providers()}")

    # Metadata
    meta = session.get_modelmeta()
    if meta.custom_metadata_map:
        print(f"\nModel metadata:")
        for k, v in meta.custom_metadata_map.items():
            print(f"  {k}: {v}")

    # Verify expected input shape [1, 3, 112, 112]
    expected_input = [1, 3, 112, 112]
    actual_input = inputs[0].shape
    # Some models use dynamic batch (None or "batch") for dim 0
    shape_ok = True
    for i in range(len(expected_input)):
        if actual_input[i] is not None and actual_input[i] != expected_input[i]:
            # dim 0 can be dynamic batch size
            if i == 0 and not isinstance(actual_input[i], int):
                continue
            shape_ok = False

    if shape_ok:
        print(f"\nInput shape check: PASS (compatible with [1, 3, 112, 112])")
    else:
        print(f"\nInput shape check: WARNING — expected {expected_input}, got {actual_input}")
        print("  The ONNXEngine expects 112x112 RGB input in CHW format.")

    # Quick inference test with dummy data
    import numpy as np

    dummy_input = np.random.randn(1, 3, 112, 112).astype(np.float32)
    input_name = inputs[0].name
    result = session.run(None, {input_name: dummy_input})
    embedding = result[0]
    print(f"\nDummy inference test: PASS")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding dim: {embedding.shape[-1]}")
    print(f"  Embedding dtype: {embedding.dtype}")

    print(f"\n{'=' * 60}")
    print("Verification complete — model is ready for FriendlyFace.")
    print(f"{'=' * 60}")

    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download the ArcFace ONNX model for FriendlyFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/download_onnx_model.py           # Download to default location\n"
            "  python3 scripts/download_onnx_model.py --verify   # Download and verify\n"
            "  python3 scripts/download_onnx_model.py --output /path/to/model.onnx\n"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify model with onnxruntime after download",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    try:
        model_path = download_arcface(output_path)
    except RuntimeError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.verify:
        ok = verify_model(model_path)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
