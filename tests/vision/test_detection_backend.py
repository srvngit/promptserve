"""Detection backend resolution."""

from __future__ import annotations

import os

from hackstorm.vision.detector import resolve_detection_backend


def test_default_is_grounding_dino(monkeypatch) -> None:
    monkeypatch.delenv("HACKSTORM_DETECT_BACKEND", raising=False)
    monkeypatch.delenv("HACKSTORM_VLM_BACKEND", raising=False)
    assert resolve_detection_backend(None) == "grounding_dino"


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("HACKSTORM_DETECT_BACKEND", "ollama")
    assert resolve_detection_backend(None) == "ollama"


def test_explicit_request(monkeypatch) -> None:
    monkeypatch.setenv("HACKSTORM_DETECT_BACKEND", "yolo")
    assert resolve_detection_backend("transformers") == "transformers"
