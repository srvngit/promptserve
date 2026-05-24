"""YOLO detector unit tests (no model download)."""

from __future__ import annotations

from hackstorm.vision.yolo_detector import _pixel_xyxy_to_box_2d


class TestPixelBoxConversion:
    def test_full_frame_box(self) -> None:
        box = _pixel_xyxy_to_box_2d(0, 0, 640, 480, 640, 480)
        assert box == (0, 0, 1000, 1000)

    def test_center_quarter(self) -> None:
        # 160,120 → 480,360 on 640×480
        box = _pixel_xyxy_to_box_2d(160, 120, 480, 360, 640, 480)
        assert box == (250, 250, 750, 750)
