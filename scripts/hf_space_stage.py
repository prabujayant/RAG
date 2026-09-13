"""Assemble a staging directory for the Hugging Face Docker Space.

Copies all files the Dockerfile expects into a clean staging dir with the
layout the Space repo needs.  Optionally pushes to a Space via
``huggingface_hub``.

Usage:
    python scripts/hf_space_stage.py --output ./hf_staging
    python scripts/hf_space_stage.py --output ./hf_staging --push username/space-name
    python scripts/hf_space_stage.py --push username/space-name   # temp dir
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Files/dirs that the HF Dockerfile COPY expects (relative to repo root).
COPY_LIST = [
    "pyproject.toml",
    "app",
    "scripts",
    "data/corpus",
    "data/uploads_seed",
]

DEPLOY_SRC = ROOT / "deploy" / "huggingface"
DEPLOY_FILES = ["Dockerfile", "start.sh", "warmup_models.py"]

SPACE_README_FRONTMATTER = """\
---
title: AskMyDocs
emoji: "\U0001F4DA"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

"""

SPACE_README_BODY = """\
# AskMyDocs

Grounded RAG question-answering for technical documentation.

**Set these secrets:** `OPENROUTER_API_KEY` (required), `HF_TOKEN`, `HF_BUCKET`, `RESTORE_ON_BOOT`.

See [deployment guide](deploy/huggingface/README.md) for full docs.
"""


def stage(output: Path) -> None:
    """Assemble the Space repo into *output*."""
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    # 1. Core project files
    for rel in COPY_LIST:
        src = ROOT / rel
        dst = output / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # 2. Deploy helpers (Dockerfile, start.sh, warmup_models.py) → repo root
    for name in DEPLOY_FILES:
        src = DEPLOY_SRC / name
        if src.exists():
            shutil.copy2(src, output / name)

    # 3. README_HF.md → README.md (pyproject.toml readme = "README.md")
    readme_hf = ROOT / "README_HF.md"
    if readme_hf.exists():
        shutil.copy2(readme_hf, output / "README.md")

    # 4. Space README.md (with HF frontmatter) → separate file
    space_readme = output / "README_SPACE.md"
    space_readme.write_text(SPACE_README_FRONTMATTER + SPACE_README_BODY, encoding="utf-8")

    print(f"Staging complete: {output}")
    print(f"  Files: {sum(1 for _ in output.rglob('*') if _.is_file())}")
    print(f"  Next: cd {output} && docker build -t askmydocs-hf .")


def push_to_space(staging: Path, space_id: str) -> None:
    """Push the staging directory to a Hugging Face Space."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub is required. Install with: pip install huggingface_hub")
        raise SystemExit(1) from None

    api = HfApi()

    # The Space needs its own README.md carrying the Docker-SDK frontmatter.
    # Keep the project README as README_HF.md; replace README.md atomically so
    # this works on Windows too (Path.rename fails when the target exists).
    space_readme = staging / "README_SPACE.md"
    if space_readme.exists():
        project_readme = staging / "README.md"
        if project_readme.exists():
            shutil.move(str(project_readme), str(staging / "README_HF.md"))
        shutil.move(str(space_readme), str(staging / "README.md"))

    print(f"Uploading to Space: {space_id}")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=space_id,
        repo_type="space",
    )
    print(f"Pushed to https://huggingface.co/spaces/{space_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage and deploy AskMyDocs to HF Spaces.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for staged files (default: temp dir)",
    )
    parser.add_argument(
        "--push",
        metavar="USER/SPACE",
        default=None,
        help="Push to this HF Space (requires huggingface_hub)",
    )
    args = parser.parse_args()

    if args.push and args.output is None:
        args.output = Path(tempfile.mkdtemp(prefix="hf_stage_"))
    elif args.output is None:
        args.output = ROOT / "hf_staging"

    stage(args.output)

    if args.push:
        push_to_space(args.output, args.push)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
