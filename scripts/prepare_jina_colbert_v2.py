#!/usr/bin/env python3
"""Prepare a PyLate-compatible Jina ColBERT v2 model directory."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download jinaai/jina-colbert-v2 via PyLate and export a local "
            "model directory compatible with pylate-rs."
        )
    )
    parser.add_argument(
        "--model",
        default="jinaai/jina-colbert-v2",
        help="HuggingFace model ID to export (default: jinaai/jina-colbert-v2)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the exported model",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for export (cpu or cuda, default: cpu)",
    )
    args = parser.parse_args()

    try:
        from pylate import models  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(
            "pylate is required. Install with: pip install pylate",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = models.ColBERT(
        model_name_or_path=args.model,
        device=args.device,
        trust_remote_code=True,
    )
    model.save_pretrained(str(output_dir))

    print(f"Exported PyLate model to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
