"""Vision detector unit tests (no GPU) and optional GPU integration."""

from __future__ import annotations

import pytest
import numpy as np

from hackstorm.vision.detector import (
    BoxDetection,
    _image_width_height,
    parse_boxes_json,
    pick_best,
)

# Fixture: model-style fenced JSON (0–1000 grid), 640×480 image
PARSE_FIXTURE_TEXT = """\
Here are the detections:

```json
[
  {"box_2d": [100, 200, 300, 400], "label": "gray box"},
  {"box_2d": [50, 50, 150, 150], "label": "small thing"}
]
```
"""

IMAGE_W = 640
IMAGE_H = 480


class TestImageWidthHeight:
    def test_numpy_rgb_array(self) -> None:
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        assert _image_width_height(rgb) == (640, 480)

    def test_pil_like_tuple_size(self) -> None:
        class FakePil:
            size = (640, 480)

        assert _image_width_height(FakePil()) == (640, 480)


class TestParseBoxesJson:
    def test_parses_fenced_json_array(self) -> None:
        boxes = parse_boxes_json(PARSE_FIXTURE_TEXT, IMAGE_W, IMAGE_H)
        assert len(boxes) == 2
        assert boxes[0].label == "gray box"
        assert boxes[0].box_2d == (100, 200, 300, 400)

    def test_descales_center_and_size(self) -> None:
        boxes = parse_boxes_json(PARSE_FIXTURE_TEXT, IMAGE_W, IMAGE_H)
        det = boxes[0]
        # y: 100–300 → 48–144, x: 200–400 → 128–256 on 640×480
        assert det.center_xy[0] == pytest.approx(192.0)
        assert det.center_xy[1] == pytest.approx(96.0)
        assert det.size_wh[0] == pytest.approx(128.0)
        assert det.size_wh[1] == pytest.approx(96.0)

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="valid JSON"):
            parse_boxes_json("not json at all", IMAGE_W, IMAGE_H)


class TestPickBest:
    def _box(self, label: str, w: float, h: float) -> BoxDetection:
        return BoxDetection(
            label=label,
            box_2d=(0, 0, 100, 100),
            center_xy=(w / 2, h / 2),
            size_wh=(w, h),
        )

    def test_returns_largest_when_multiple(self) -> None:
        small = self._box("gray box", 10.0, 10.0)
        large = self._box("gray box", 200.0, 150.0)
        medium = self._box("gray container", 50.0, 50.0)
        result = pick_best([small, large, medium], "gray box")
        assert result is large

    def test_empty_returns_none(self) -> None:
        assert pick_best([], "anything") is None

    def test_prefers_label_matching_description(self) -> None:
        red = self._box("red cup", 500.0, 500.0)
        gray = self._box("small gray box", 20.0, 20.0)
        result = pick_best([red, gray], "gray box")
        assert result is gray


@pytest.mark.gpu
def test_detect_integration_on_image() -> None:
    """Optional GPU test: run Gemma detect on a real frame if available."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from pathlib import Path

    from PIL import Image

    from hackstorm.vision.detector import detect

    frame = Path(__file__).resolve().parents[2] / "captures" / "frame.jpg"
    if not frame.is_file():
        pytest.skip(f"no test image at {frame}")

    image = Image.open(frame).convert("RGB")
    boxes = detect(image, "gray box")
    assert isinstance(boxes, list)
