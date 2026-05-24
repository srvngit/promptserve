"""YOLO prepare/download helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hackstorm.vision import yolo_detector


def test_prepare_yolo_is_idempotent() -> None:
    yolo_detector._PREPARED.clear()
    yolo_detector._YOLO_CACHE.clear()

    fake_model = MagicMock()
    with patch.object(yolo_detector, "_get_yolo_model", return_value=fake_model):
        mid = yolo_detector.prepare_yolo("yolov8s-worldv2.pt", "cpu", warmup=False)
        again = yolo_detector.prepare_yolo("yolov8s-worldv2.pt", "cpu", warmup=False)

    assert mid == "yolov8s-worldv2.pt"
    assert again == mid
    fake_model.set_classes.assert_called_once()
