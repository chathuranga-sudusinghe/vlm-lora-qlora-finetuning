"""Check the unavailable-CUDA boundary without requiring a PyTorch installation."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vlm_finetuning.environment import check_device

pytestmark = pytest.mark.unit


def test_unavailable_cuda_never_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    allocate = Mock()
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), ones=allocate)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        check_device("cuda")

    allocate.assert_not_called()
