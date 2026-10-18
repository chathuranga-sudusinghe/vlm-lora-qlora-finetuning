"""Exercise real PyTorch operations and the Python module command boundary."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from vlm_finetuning.environment import check_device

pytestmark = pytest.mark.integration


def test_cpu_forward_backward_and_report() -> None:
    import torch

    report = check_device("cpu")

    assert report.device == "cpu"
    assert report.gradient_check_passed
    assert report.torch_version == str(torch.__version__)
    assert report.gpu_name is None
    assert report.total_vram_gib is None
    assert report.compute_capability is None
    assert report.bf16_supported is None


def test_module_command_on_cpu() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vlm_finetuning", "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["device"] == "cpu"
    assert report["gradient_check_passed"] is True


def test_installed_console_command_on_cpu() -> None:
    command = Path(sys.executable).parent / "vlm-check-device"
    assert command.is_file(), (
        "Install the project to register vlm-check-device in this environment."
    )

    result = subprocess.run(
        [str(command), "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["device"] == "cpu"
    assert report["gradient_check_passed"] is True


def test_nonfinite_loss_is_rejected_with_finite_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    values = torch.full((4, 4), float("inf"), requires_grad=True)
    values.register_hook(lambda gradient: torch.zeros_like(gradient))
    monkeypatch.setattr(torch, "ones", Mock(return_value=values))

    with pytest.raises(RuntimeError, match="non-finite results"):
        check_device("cpu")

    assert values.grad is not None
    assert bool(torch.isfinite(values.grad).all().item())


def test_nonfinite_backward_result_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    values = torch.ones((4, 4), requires_grad=True)
    values.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    monkeypatch.setattr(torch, "ones", Mock(return_value=values))

    with pytest.raises(RuntimeError, match="non-finite results"):
        check_device("cpu")


@pytest.mark.cuda
def test_cuda_forward_backward_and_hardware_report() -> None:
    report = check_device("cuda")

    assert report.device == "cuda"
    assert report.gradient_check_passed
    assert report.cuda_runtime is not None
    assert report.gpu_name
    assert report.total_vram_gib is not None and report.total_vram_gib > 0
    assert report.compute_capability is not None
    assert isinstance(report.bf16_supported, bool)
