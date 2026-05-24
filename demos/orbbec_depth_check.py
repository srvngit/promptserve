#!/usr/bin/env python3
"""Sanity check: open Orbbec DaBai DC1, grab one frame, prove depth works.

Runs on the Linux robot host AFTER ./scripts/install_pyorbbecsdk.sh succeeds.

Saves three files under captures/orbbec_check/:
  color.jpg      — RGB
  depth_raw.png  — 16-bit depth (mm)
  depth_vis.jpg  — colorized depth for eyeballing
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from hackstorm.perception.orbbec_camera import OrbbecCamera


def _colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    import cv2

    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        return np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    near, far = int(np.percentile(valid, 5)), int(np.percentile(valid, 95))
    if far <= near:
        far = near + 1
    norm = np.clip((depth_mm.astype(np.int32) - near) / (far - near), 0, 1)
    norm[depth_mm == 0] = 1.0  # paint invalid as far/red
    norm_u8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)


def main() -> int:
    out_dir = Path("captures/orbbec_check")
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = OrbbecCamera()
    cam.open()
    if cam.stub_mode:
        print("ERROR: pyorbbecsdk not installed — got stub frames.", file=sys.stderr)
        print("Run: ./scripts/install_pyorbbecsdk.sh", file=sys.stderr)
        return 1

    # Discard first few frames; DaBai needs a moment to settle exposure + depth.
    for _ in range(5):
        cam.capture()
        time.sleep(0.05)
    frame = cam.capture()
    cam.close()

    import cv2

    depth = frame.depth_mm
    valid = depth[depth > 0]
    coverage = 100.0 * valid.size / depth.size
    center = depth[depth.shape[0] // 2, depth.shape[1] // 2]
    print(f"color   : {frame.color_rgb.shape[1]}x{frame.color_rgb.shape[0]}")
    print(f"depth   : {depth.shape[1]}x{depth.shape[0]}  (uint16 mm)")
    print(f"intrins : fx={frame.intrinsics.fx:.1f} fy={frame.intrinsics.fy:.1f} "
          f"cx={frame.intrinsics.cx:.1f} cy={frame.intrinsics.cy:.1f}")
    print(f"coverage: {coverage:.1f}% of pixels have valid depth")
    if valid.size:
        print(f"range   : min={valid.min()} mm  median={int(np.median(valid))} mm  max={valid.max()} mm")
    print(f"center  : {int(center)} mm  ({'valid' if center > 0 else 'INVALID — point camera at >25 cm target'})")

    cv2.imwrite(str(out_dir / "color.jpg"), cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "depth_raw.png"), depth)  # 16-bit
    cv2.imwrite(str(out_dir / "depth_vis.jpg"), _colorize_depth(depth))
    print(f"saved   : {out_dir}/")

    if coverage < 10.0:
        print("WARN: <10% depth coverage. Check camera distance (DC1 min ~25cm), lighting, USB 3 port.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
