"""Quick test: save one frame from V4L2 webcam."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hackstorm.perception.webcam_camera import save_webcam_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture one webcam frame to PNG")
    parser.add_argument(
        "--device",
        default="/dev/video4",
        help="V4L2 device (default /dev/video4)",
    )
    parser.add_argument("output", type=Path, help="Output PNG path")
    args = parser.parse_args(argv)
    save_webcam_frame(args.device, args.output)
    print(f"ok: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
