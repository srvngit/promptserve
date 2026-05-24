"""Tests for Piper numerical IK."""

from __future__ import annotations

import math

import pytest

from hackstorm.arm.piper_ik import GRASP_RPY_DEG, fk_end_pose_mm_deg, solve_ik


def test_fk_zero_joints_near_init_pos() -> None:
    joints = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    x, y, z, rx, ry, rz = fk_end_pose_mm_deg(joints)
    assert x == pytest.approx(56.128, abs=1.0)
    assert z == pytest.approx(213.266, abs=1.0)
    assert ry == pytest.approx(85.0, abs=0.5)


def test_ik_roundtrip_init_pose() -> None:
    seed = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    x, y, z, *_ = fk_end_pose_mm_deg(seed)
    solved = solve_ik((x / 1000, y / 1000, z / 1000), seed, target_rpy_deg=GRASP_RPY_DEG)
    sx, sy, sz, srx, sry, srz = fk_end_pose_mm_deg(solved)
    assert sx == pytest.approx(x, abs=2.0)
    assert sy == pytest.approx(y, abs=2.0)
    assert sz == pytest.approx(z, abs=2.0)
    assert sry == pytest.approx(85.0, abs=1.0)


def test_ik_home_workspace_target() -> None:
    seed = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    target_m = (0.25, 0.0, 0.35)
    solved = solve_ik(target_m, seed, target_rpy_deg=GRASP_RPY_DEG)
    x, y, z, _, ry, _ = fk_end_pose_mm_deg(solved)
    assert x == pytest.approx(target_m[0] * 1000, abs=2.0)
    assert y == pytest.approx(target_m[1] * 1000, abs=2.0)
    assert z == pytest.approx(target_m[2] * 1000, abs=2.0)
    assert ry == pytest.approx(85.0, abs=1.0)
    assert all(math.isfinite(j) for j in solved)
