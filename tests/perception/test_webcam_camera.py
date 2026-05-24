"""Tests for V4L2 webcam helpers."""

from hackstorm.perception.webcam_camera import _parse_device


def test_parse_dev_video_path() -> None:
    assert _parse_device("/dev/video4") == "/dev/video4"


def test_parse_numeric_index() -> None:
    assert _parse_device("4") == 4
