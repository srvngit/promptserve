"""Grounding DINO helpers."""

from hackstorm.vision.grounding_dino_detector import _format_text_prompt


def test_text_prompt_adds_period() -> None:
    assert _format_text_prompt("strawberry") == "strawberry."
    assert _format_text_prompt("Gray Box") == "gray box."
