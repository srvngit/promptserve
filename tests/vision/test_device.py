"""Tests for VLM pipeline device selection."""

import pytest

from hackstorm.vision.detector import _resolve_pipeline_device


def test_explicit_cpu() -> None:
    assert _resolve_pipeline_device("cpu") == -1


def test_explicit_cuda_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    assert _resolve_pipeline_device("cuda") == 0
    assert _resolve_pipeline_device(None) == 0
