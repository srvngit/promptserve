"""Pure math tests for RGB-D frame → arm coordinate helpers."""

from __future__ import annotations

import numpy as np
import pytest

from hackstorm.perception.frames import (
    ColorIntrinsics,
    camera_point_to_arm,
    depth_median_patch,
    pixel_depth_to_camera_point,
)


def test_depth_median_patch_5x5() -> None:
    depth = np.zeros((20, 20), dtype=np.uint16)
    cx, cy = 10, 10
    values = list(range(1, 26))
    idx = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            depth[cy + dy, cx + dx] = values[idx]
            idx += 1
    assert depth_median_patch(depth, cx, cy, radius=2) == 13


def test_depth_median_patch_ignores_zero_depth() -> None:
    depth = np.zeros((10, 10), dtype=np.uint16)
    depth[5, 5] = 100
    depth[5, 6] = 200
    assert depth_median_patch(depth, 5, 5, radius=2) == 150


def test_pixel_depth_to_camera_point() -> None:
    intrinsics = ColorIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0)
    x, y, z = pixel_depth_to_camera_point(150, 100, 1000.0, intrinsics)
    assert x == pytest.approx(1000.0)
    assert y == pytest.approx(500.0)
    assert z == pytest.approx(1000.0)


def test_camera_point_to_arm_4x4_transform() -> None:
    p_cam = (1.0, 2.0, 3.0)
    t = np.array(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    x, y, z = camera_point_to_arm(p_cam, t)
    assert (x, y, z) == pytest.approx((11.0, 22.0, 33.0))
