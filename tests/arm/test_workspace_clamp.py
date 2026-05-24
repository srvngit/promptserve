"""Pure math tests for workspace pose clamping."""

from __future__ import annotations

import pytest

from hackstorm.arm.piper_controller import Pose, WorkspaceBounds, clamp_pose


@pytest.fixture
def bounds() -> WorkspaceBounds:
    return WorkspaceBounds(
        x_min=0.1,
        x_max=0.45,
        y_min=-0.25,
        y_max=0.25,
        z_min=0.05,
        z_max=0.40,
    )


def test_clamp_pose_inside_bounds_unchanged(bounds: WorkspaceBounds) -> None:
    pose = Pose(x=0.25, y=0.0, z=0.35, rx=180.0, ry=0.0, rz=0.0)
    assert clamp_pose(pose, bounds) == pose


def test_clamp_pose_clamps_high_xyz(bounds: WorkspaceBounds) -> None:
    pose = Pose(x=1.0, y=0.5, z=0.9, rx=180.0, ry=0.0, rz=0.0)
    clamped = clamp_pose(pose, bounds)
    assert clamped.x == bounds.x_max
    assert clamped.y == bounds.y_max
    assert clamped.z == bounds.z_max
    assert clamped.rx == pose.rx


def test_clamp_pose_clamps_low_xyz(bounds: WorkspaceBounds) -> None:
    pose = Pose(x=-1.0, y=-0.5, z=0.01, rx=0.0, ry=0.0, rz=0.0)
    clamped = clamp_pose(pose, bounds)
    assert clamped.x == bounds.x_min
    assert clamped.y == bounds.y_min
    assert clamped.z == bounds.z_min


def test_clamp_pose_preserves_orientation(bounds: WorkspaceBounds) -> None:
    pose = Pose(x=2.0, y=0.0, z=0.2, rx=45.0, ry=-30.0, rz=90.0)
    clamped = clamp_pose(pose, bounds)
    assert clamped.rx == 45.0
    assert clamped.ry == -30.0
    assert clamped.rz == 90.0
