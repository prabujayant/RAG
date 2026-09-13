"""Build-time model download for the AskMyDocs HF Space image.

Bakes ``BAAI/bge-m3`` and the MiniLM cross-encoder into ``HF_HOME`` so the
Space does not pay a multi-GB Hub download on first startup. The BAAI repo
is gated, so the Space must have accepted its license; ``HF_TOKEN`` is
optional (public access still works once the license is accepted on the Hub).

The MiniLM reranker (``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``, ~118M)
is the default and runs well on CPU. The heavier ``BAAI/bge-reranker-v2-m3``
(568M) is selectable via ``RERANKER_MODEL`` at runtime.

This script never fails the image build: if a model cannot be fetched, the
app simply downloads it on first use at runtime.
"""

from __future__ import annotations

import os


def main() -> int:
    hf_home = os.environ.get("HF_HOME", "/home/user/hf_home")
    os.environ.setdefault("HF_HOME", hf_home)
    print(f"[warmup] HF_HOME={hf_home}")

    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer("BAAI/bge-m3")
        print("[warmup] embedding model cached")
    except Exception as exc:  # noqa: BLE001
        print(f"[warmup] embedding warm-up failed: {exc}")

    try:
        from sentence_transformers import CrossEncoder

        CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=512)
        print("[warmup] reranker cached")
    except Exception as exc:  # noqa: BLE001
        print(f"[warmup] reranker warm-up failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
