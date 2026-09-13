"""Back up AskMyDocs state to a Hugging Face Storage Bucket.

Creates a compressed archive of PostgreSQL data, Qdrant storage, and uploaded
documents, then uploads it to the specified HF Storage Bucket.  The Space's
``start.sh`` restores this archive on boot when ``RESTORE_ON_BOOT=true``.

Usage:
    python scripts/hf_state_backup.py --bucket username/askmydocs-data
    python scripts/hf_state_backup.py --bucket username/askmydocs-data --data-root /home/user
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def create_archive(data_root: Path, output: Path) -> bool:
    """Create a zstd-compressed tar of the state directories."""
    dirs_to_include = ["pgdata", "qdrant_storage"]
    uploads_dir = data_root / "app" / "data" / "uploads"
    if uploads_dir.exists():
        dirs_to_include.append("app/data/uploads")

    existing = [d for d in dirs_to_include if (data_root / d).exists()]
    if not existing:
        print("No state directories found to back up.")
        return False

    cmd = [
        "tar",
        "--zstd",
        "-cf",
        str(output),
        "-C",
        str(data_root),
        *existing,
    ]
    print(f"Creating archive: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"tar failed: {result.stderr}")
        return False

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Archive created: {output} ({size_mb:.1f} MB)")
    return True


def upload_to_bucket(archive: Path, bucket: str, token: str | None = None) -> bool:
    """Upload the archive to an HF Storage Bucket.

    Storage Buckets are not repos: ``upload_file(repo_type="space_storage")``
    does not exist (supported repo types are model/dataset/space). The bucket
    API is ``batch_bucket_files(bucket_id=..., add=[(local, remote)])``.
    """
    try:
        from huggingface_hub import batch_bucket_files
    except ImportError:
        print("Error: huggingface_hub is required. Install with: pip install huggingface_hub")
        return False

    print(f"Uploading {archive.name} to bucket {bucket}...")
    batch_bucket_files(
        bucket_id=bucket,
        add=[(archive, "state.tar.zst")],
        token=token,
    )
    print(f"Uploaded to hf://buckets/{bucket}/state.tar.zst")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Back up AskMyDocs state to HF Storage Bucket.")
    parser.add_argument(
        "--bucket",
        required=True,
        help="HF Storage Bucket name (e.g. 'username/askmydocs-data')",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/user"),
        help="Data root directory (default: /home/user for HF Space)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token for private bucket access (default: HF_TOKEN env var)",
    )
    args = parser.parse_args()

    import os
    token = args.token or os.environ.get("HF_TOKEN")

    with tempfile.NamedTemporaryFile(suffix=".tar.zst", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        if not create_archive(args.data_root, tmp_path):
            return 1
        if not upload_to_bucket(tmp_path, args.bucket, token):
            return 1
    finally:
        tmp_path.unlink(missing_ok=True)

    print("State backup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
