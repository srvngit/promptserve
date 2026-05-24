#!/usr/bin/env python3
"""Snap laptop + Orbbec frames and run Grounding DINO — see what each camera sees."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _annotate(path: Path, color_rgb: np.ndarray, label: str, queries: list[str]) -> None:
    from hackstorm.vision.detector import detect_with_raw, draw_boxes, pick_best_in_frame
    from hackstorm.vision.grounding_dino_detector import prepare_grounding_dino

    prepare_grounding_dino(device="cuda")
    h, w = color_rgb.shape[:2]
    out_dir = path.parent
    raw_path = out_dir / f"{path.stem}_raw.jpg"
    cv2.imwrite(str(raw_path), cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))

    for q in queries:
        boxes, _raw = detect_with_raw(color_rgb, q, backend="grounding_dino", device="cuda")
        target = pick_best_in_frame(boxes, q, width=w, height=h)
        ann = draw_boxes(
            __import__("PIL").Image.fromarray(color_rgb),
            boxes,
            highlight=target,
        )
        tag = q.replace(" ", "_")[:24]
        ann_path = out_dir / f"{path.stem}_{tag}.jpg"
        cv2.imwrite(str(ann_path), ann)
        n = len(boxes)
        hit = "YES" if target else "NO"
        print(f"  {label} + {q!r}: {n} box(es) target={hit} -> {ann_path.name}")


def _snap_webcam(device: str, out: Path, queries: list[str]) -> bool:
    from hackstorm.perception.webcam_camera import WebcamCamera

    print(f"[webcam] {device} -> {out}")
    try:
        cam = WebcamCamera(device)
        cam.open()
        for _ in range(10):
            cam.capture()
            time.sleep(0.05)
        frame = cam.capture()
        cam.close()
    except Exception as exc:
        print(f"  SKIP: {exc}")
        return False
    cv2.imwrite(str(out), cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR))
    _annotate(out, frame.color_rgb, device, queries)
    return True


def _snap_orbbec(out: Path, queries: list[str]) -> bool:
    from hackstorm.perception.orbbec_camera import OrbbecCamera

    print(f"[orbbec] wrist camera -> {out}")
    try:
        cam = OrbbecCamera()
        cam.open()
        for _ in range(8):
            cam.capture()
            time.sleep(0.08)
        frame = cam.capture()
        cam.close()
    except Exception as exc:
        print(f"  SKIP: {exc}")
        return False
    cv2.imwrite(str(out), cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR))
    nz = int((frame.depth_mm > 0).sum())
    print(f"  depth valid pixels: {nz}/{frame.depth_mm.size}")
    _annotate(out, frame.color_rgb, "orbbec", queries)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Debug snap: laptop + Orbbec + DINO")
    parser.add_argument("--webcam", default="/dev/video0", help="Laptop V4L2 device")
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["ketchup bottle", "pink tray"],
        help="Detection queries",
    )
    args = parser.parse_args(argv)

    out_dir = Path("captures/snap_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_dir}/\n")

    _snap_webcam(args.webcam, out_dir / "laptop.jpg", args.queries)
    print()
    _snap_orbbec(out_dir / "orbbec.jpg", args.queries)
    print(f"\nok: inspect {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
