"""Verify command dispatch and errors without importing PyTorch."""

import importlib
import json
import sys
from unittest.mock import Mock

import pytest

from vlm_finetuning import cli
from vlm_finetuning.environment import DeviceReport

pytestmark = pytest.mark.unit


def test_default_requests_cuda_and_prints_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = DeviceReport("cuda", "test", "12.8", "test GPU", 8.0, (12, 0), True, True)
    check = Mock(return_value=report)
    monkeypatch.setattr(cli, "check_device", check)

    assert cli.main([]) == 0

    check.assert_called_once_with("cuda")
    assert json.loads(capsys.readouterr().out) == {
        "device": "cuda",
        "torch_version": "test",
        "cuda_runtime": "12.8",
        "gpu_name": "test GPU",
        "total_vram_gib": 8.0,
        "compute_capability": [12, 0],
        "bf16_supported": True,
        "gradient_check_passed": True,
    }


def test_explicit_cpu_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    check = Mock(return_value=DeviceReport("cpu", "test", None, None, None, None, None, True))
    monkeypatch.setattr(cli, "check_device", check)

    assert cli.main(["--device", "cpu"]) == 0

    check.assert_called_once_with("cpu")


@pytest.mark.parametrize("error", [RuntimeError("CUDA unavailable"), ModuleNotFoundError("torch")])
def test_runtime_failure_returns_nonzero_without_success_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    monkeypatch.setattr(cli, "check_device", Mock(side_effect=error))

    assert cli.main([]) == 1

    assert capsys.readouterr().out == ""
    assert "Device check failed" in caplog.text
    assert str(error) in caplog.text


def test_invalid_device_is_rejected_before_runtime_load(monkeypatch: pytest.MonkeyPatch) -> None:
    check = Mock()
    monkeypatch.setattr(cli, "check_device", check)

    with pytest.raises(SystemExit) as error:
        cli.main(["--device", "tpu"])

    assert error.value.code == 2
    check.assert_not_called()


def test_importing_module_entry_point_does_not_run_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = Mock()
    monkeypatch.setattr(cli, "main", main)
    monkeypatch.delitem(sys.modules, "vlm_finetuning.__main__", raising=False)

    importlib.import_module("vlm_finetuning.__main__")

    main.assert_not_called()
