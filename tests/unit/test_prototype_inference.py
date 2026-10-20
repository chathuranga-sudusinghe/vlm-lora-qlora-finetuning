"""Exercise the prototype with fake image, tensor, processor, and model backends."""

import os
import runpy
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from vlm_finetuning import prototype_inference as prototype
from vlm_finetuning.environment import Device

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def backends(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Install only mocks; this fixture imports no optional image or ML library."""
    source, rgb = MagicMock(), MagicMock()
    source.__enter__.return_value = source
    rgb.__enter__.return_value = rgb
    source.convert.return_value = rgb
    image = MagicMock()
    image.open.return_value = source

    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.cuda.is_bf16_supported.return_value = True
    torch.cuda.get_device_name.return_value = "mock GPU"
    torch.cuda.max_memory_allocated.return_value = 3 * 1024**3
    torch.no_grad.return_value = nullcontext()
    input_ids, pixels, generated = MagicMock(), MagicMock(), MagicMock()
    input_ids.shape = (1, 3)
    input_ids.to.return_value = input_ids
    pixels.to.return_value = pixels
    processor = MagicMock()
    processor.apply_chat_template.return_value = "synthetic prompt"
    processor.return_value = {"input_ids": input_ids, "pixel_values": pixels}
    processor.batch_decode.return_value = ["synthetic answer"]
    model = MagicMock()
    model.to.return_value = model
    model.generate.return_value = generated
    load_processor = MagicMock(return_value=processor)
    load_model = MagicMock(return_value=model)

    pil_module, torch_module, transformers_module = (
        ModuleType("PIL"),
        ModuleType("torch"),
        ModuleType("transformers"),
    )
    monkeypatch.setattr(pil_module, "Image", image, raising=False)
    for name in ("cuda", "no_grad", "float32", "bfloat16"):
        monkeypatch.setattr(torch_module, name, getattr(torch, name), raising=False)
    monkeypatch.setattr(
        transformers_module,
        "AutoProcessor",
        SimpleNamespace(from_pretrained=load_processor),
        raising=False,
    )
    monkeypatch.setattr(
        transformers_module,
        "AutoModelForImageTextToText",
        SimpleNamespace(from_pretrained=load_model),
        raising=False,
    )
    for module in (pil_module, torch_module, transformers_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return {
        "source": source,
        "rgb": rgb,
        "image": image,
        "torch": torch,
        "input_ids": input_ids,
        "pixels": pixels,
        "generated": generated,
        "processor": processor,
        "model": model,
        "load_processor": load_processor,
        "load_model": load_model,
    }


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    """The mocked decoder consumes this synthetic placeholder, not project data."""
    path = tmp_path / "synthetic.png"
    path.write_bytes(b"synthetic placeholder for the mocked image backend")
    return path


def test_package_and_wrapper_import_without_optional_libraries() -> None:
    script = """
import builtins
import importlib
import runpy

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'PIL', 'transformers'}:
        raise AssertionError('Optional backend imported: ' + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
importlib.import_module('vlm_finetuning.prototype_inference')
runpy.run_path('src/baseline_inference.py', run_name='import_probe')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("device", ["cpu", "cuda"], ids=["cpu-mock", "gpu-mock"])
def test_explicit_device_generation_and_answer_only_decoding(
    backends: dict[str, MagicMock], image_path: Path, device: Device
) -> None:
    result = prototype.run_prototype(image_path, device)

    assert result.answer == "synthetic answer"
    assert result.device == device
    assert result.gpu_name == ("mock GPU" if device == "cuda" else None)
    assert result.peak_allocated_vram_gib == (3.0 if device == "cuda" else None)
    backends["load_processor"].assert_called_once_with(
        prototype.MODEL_ID, cache_dir=str(prototype.CACHE_DIR)
    )
    dtype = backends["torch"].bfloat16 if device == "cuda" else backends["torch"].float32
    backends["load_model"].assert_called_once_with(
        prototype.MODEL_ID, cache_dir=str(prototype.CACHE_DIR), dtype=dtype
    )
    backends["model"].to.assert_called_once_with(device)
    backends["model"].eval.assert_called_once_with()
    backends["input_ids"].to.assert_called_once_with(device)
    backends["pixels"].to.assert_called_once_with(device)
    backends["model"].generate.assert_called_once_with(
        input_ids=backends["input_ids"],
        pixel_values=backends["pixels"],
        max_new_tokens=128,
        do_sample=False,
        return_dict_in_generate=False,
    )
    backends["generated"].__getitem__.assert_called_once_with((slice(None), slice(3, None)))
    backends["processor"].batch_decode.assert_called_once_with(
        backends["generated"].__getitem__.return_value, skip_special_tokens=True
    )
    backends["source"].__exit__.assert_called_once_with(None, None, None)
    backends["rgb"].__exit__.assert_called_once_with(None, None, None)
    if device == "cpu":
        assert backends["torch"].cuda.method_calls == []
    else:
        backends["torch"].cuda.reset_peak_memory_stats.assert_called_once_with()
        backends["torch"].cuda.synchronize.assert_called_once_with()


@pytest.mark.parametrize("directory", [False, True])
def test_missing_file_or_directory_fails_before_backend_access(
    backends: dict[str, MagicMock], tmp_path: Path, directory: bool
) -> None:
    path = tmp_path / "not-an-image"
    if directory:
        path.mkdir()
    with pytest.raises(FileNotFoundError, match="not a regular file"):
        prototype.run_prototype(path, "cpu")
    backends["image"].open.assert_not_called()
    backends["load_processor"].assert_not_called()
    backends["load_model"].assert_not_called()


def test_decode_failure_precedes_pretrained_loading(
    backends: dict[str, MagicMock], image_path: Path
) -> None:
    error = OSError("synthetic decode failure")
    backends["image"].open.side_effect = error
    with pytest.raises(OSError) as caught:
        prototype.run_prototype(image_path, "cpu")
    assert caught.value is error
    backends["load_processor"].assert_not_called()
    backends["load_model"].assert_not_called()


@pytest.mark.parametrize("available,bf16", [(False, True), (True, False)])
def test_cuda_requirements_fail_before_pretrained_loading_without_cpu_fallback(
    backends: dict[str, MagicMock], image_path: Path, available: bool, bf16: bool
) -> None:
    backends["torch"].cuda.is_available.return_value = available
    backends["torch"].cuda.is_bf16_supported.return_value = bf16
    with pytest.raises(RuntimeError, match="CUDA"):
        prototype.run_prototype(image_path, "cuda")
    backends["load_processor"].assert_not_called()
    backends["load_model"].assert_not_called()
    backends["model"].to.assert_not_called()
    backends["rgb"].__exit__.assert_called_once()


def test_generation_failure_preserves_exception_and_closes_images(
    backends: dict[str, MagicMock], image_path: Path
) -> None:
    error = RuntimeError("synthetic generation failure")
    backends["model"].generate.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        prototype.run_prototype(image_path, "cpu")
    assert caught.value is error
    backends["processor"].batch_decode.assert_not_called()
    backends["source"].__exit__.assert_called_once()
    backends["rgb"].__exit__.assert_called_once()


def test_missing_token_ids_fail_before_generation(
    backends: dict[str, MagicMock], image_path: Path
) -> None:
    backends["processor"].return_value = {"pixel_values": backends["pixels"]}
    with pytest.raises(ValueError, match="input_ids"):
        prototype.run_prototype(image_path, "cpu")
    backends["model"].generate.assert_not_called()


@pytest.mark.parametrize("answers", [[], [""], ["one", "two"]])
def test_unexpected_decoded_answers_are_explicit_failures(
    backends: dict[str, MagicMock], image_path: Path, answers: list[str]
) -> None:
    backends["processor"].batch_decode.return_value = answers
    with pytest.raises(RuntimeError, match="one nonempty"):
        prototype.run_prototype(image_path, "cpu")


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--image", "synthetic.png"],
        ["--device", "cpu"],
        ["--image", "synthetic.png", "--device", "tpu"],
    ],
)
def test_cli_requires_image_and_valid_explicit_device(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    run = Mock()
    monkeypatch.setattr(prototype, "run_prototype", run)
    with pytest.raises(SystemExit) as caught:
        prototype.main(argv)
    assert caught.value.code == 2
    run.assert_not_called()


def test_cli_help_does_not_execute_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock()
    monkeypatch.setattr(prototype, "run_prototype", run)
    with pytest.raises(SystemExit) as caught:
        prototype.main(["--help"])
    assert caught.value.code == 0
    run.assert_not_called()


@pytest.mark.parametrize("device", ["cpu", "cuda"], ids=["cpu-mock", "gpu-mock"])
def test_cli_delegates_and_identifies_prototype_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], device: Device
) -> None:
    report = prototype.PrototypeResult(
        "synthetic answer",
        device,
        "mock GPU" if device == "cuda" else None,
        3.0 if device == "cuda" else None,
    )
    run = Mock(return_value=report)
    monkeypatch.setattr(prototype, "run_prototype", run)
    assert prototype.main(["--image", "synthetic.png", "--device", device]) == 0
    run.assert_called_once_with(Path("synthetic.png"), device)
    output = capsys.readouterr().out
    assert "NOT THE CONTROLLED BASELINE" in output
    assert "synthetic answer" in output
    if device == "cuda":
        assert "GPU: mock GPU" in output
        assert "Peak allocated VRAM: 3.00 GiB" in output
    else:
        assert "GPU:" not in output
        assert "Peak allocated VRAM" not in output


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing image"),
        ModuleNotFoundError("optional runtime"),
        RuntimeError("model failure"),
        ValueError("invalid processing"),
    ],
)
def test_cli_reports_failure_without_success_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    monkeypatch.setattr(prototype, "run_prototype", Mock(side_effect=error))
    assert prototype.main(["--image", "synthetic.png", "--device", "cpu"]) == 1
    assert capsys.readouterr().out == ""
    assert "Prototype inference failed" in caplog.text
    assert str(error) in caplog.text


def test_legacy_wrapper_preserves_entry_point_exit_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = Mock(return_value=7)
    monkeypatch.setattr(prototype, "main", main)
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(ROOT / "src/baseline_inference.py"), run_name="__main__")
    assert caught.value.code == 7
    main.assert_called_once_with()
