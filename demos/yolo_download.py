#!/usr/bin/env python3
"""Pre-download detection model weights at runtime (before first detect)."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from hackstorm.vision.detector import resolve_detection_backend
from hackstorm.vision.grounding_dino_detector import DEFAULT_DINO_MODEL, prepare_grounding_dino
from hackstorm.vision.yolo_detector import DEFAULT_YOLO_MODEL, prepare_yolo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download detection model weights (run once before demos)."
    )
    parser.add_argument(
        "--backend",
        choices=("grounding_dino", "yolo"),
        default=os.environ.get("HACKSTORM_DETECT_BACKEND", "grounding_dino"),
        help="detection backend to prepare (default: grounding_dino)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model id / checkpoint path",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("HACKSTORM_DINO_DEVICE")
        or os.environ.get("HACKSTORM_YOLO_DEVICE")
        or os.environ.get("HACKSTORM_VLM_DEVICE"),
        help="cuda, mps, or cpu (default: auto)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="yolo only — skip dummy predict",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    backend = resolve_detection_backend(args.backend)
    try:
        if backend == "yolo":
            model_id = args.model or os.environ.get("HACKSTORM_YOLO_MODEL", DEFAULT_YOLO_MODEL)
            ready = prepare_yolo(model_id, args.device, warmup=not args.no_warmup)
        else:
            model_id = args.model or os.environ.get("HACKSTORM_DINO_MODEL", DEFAULT_DINO_MODEL)
            ready = prepare_grounding_dino(model_id, args.device)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"ready ({backend}): {ready}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
