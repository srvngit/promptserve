"""Unit tests for YAML config loaders."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hackstorm.core.config import (
    CameraArmConfig,
    WorkspaceConfig,
    load_camera_arm,
    load_workspace,
)


def test_load_workspace_defaults(config_dir: Path) -> None:
    cfg = load_workspace(config_dir / "workspace.yaml")
    assert isinstance(cfg, WorkspaceConfig)
    assert cfg.home_pose.x == pytest.approx(0.25)
    assert cfg.home_pose.z == pytest.approx(0.35)
    assert cfg.bounds.x_max == pytest.approx(0.45)
    assert cfg.approach_offset_mm == pytest.approx(80)
    assert cfg.speed_percent == 30


def test_load_camera_arm_defaults(config_dir: Path) -> None:
    cfg = load_camera_arm(config_dir / "camera_arm.yaml")
    assert isinstance(cfg, CameraArmConfig)
    assert len(cfg.T_cam_to_arm) == 4
    assert cfg.T_cam_to_arm[0][0] == pytest.approx(1.0)
    assert cfg.T_cam_to_arm[3][3] == pytest.approx(1.0)
    assert cfg.color_intrinsics.fx == pytest.approx(600)
    assert cfg.color_intrinsics.cy == pytest.approx(240)


def test_load_workspace_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "workspace.yaml"
    path.write_text(
        yaml.dump(
            {
                "home_pose": {"x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0},
                "bounds": {
                    "x_min": 0,
                    "x_max": 1,
                    "y_min": 0,
                    "y_max": 1,
                    "z_min": 0,
                    "z_max": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="approach_offset_mm"):
        load_workspace(path)


def test_load_camera_arm_invalid_matrix(tmp_path: Path) -> None:
    path = tmp_path / "camera_arm.yaml"
    path.write_text(
        yaml.dump(
            {
                "T_cam_to_arm": [[1, 0], [0, 1]],
                "color_intrinsics": {"fx": 1, "fy": 1, "cx": 0, "cy": 0},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="4x4"):
        load_camera_arm(path)
