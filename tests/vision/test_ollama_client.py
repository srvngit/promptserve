"""Ollama URL normalization."""

from hackstorm.vision.detector import _normalize_ollama_host


def test_strip_trailing_slash() -> None:
    assert _normalize_ollama_host("http://127.0.0.1:11434/") == "http://127.0.0.1:11434"


def test_strip_api_suffix() -> None:
    assert _normalize_ollama_host("http://127.0.0.1:11434/api") == "http://127.0.0.1:11434"
