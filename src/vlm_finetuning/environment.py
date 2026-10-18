"""Small runtime checks; importing this module never loads a model or initializes CUDA."""

from dataclasses import dataclass
from typing import Literal

Device = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class DeviceReport:
    """Observed runtime properties and the result of a tiny backward pass."""

    device: Device
    torch_version: str
    cuda_runtime: str | None
    gpu_name: str | None
    total_vram_gib: float | None
    compute_capability: tuple[int, int] | None
    bf16_supported: bool | None
    gradient_check_passed: bool


def check_device(device: Device) -> DeviceReport:
    """Require the requested device and verify finite matrix-operation gradients.

    This is an environment diagnostic, not model training or a VLM memory benchmark.
    CUDA requests fail explicitly when unavailable; they never fall back to the CPU.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported device: {device}")

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device.")

    values = torch.ones((4, 4), device=device, requires_grad=True)
    loss = (values @ values.T).square().mean()
    loss.backward()
    if (
        not bool(torch.isfinite(loss).item())
        or values.grad is None
        or not bool(torch.isfinite(values.grad).all().item())
    ):
        raise RuntimeError(f"The {device} forward/backward check produced non-finite results.")

    gpu_name = None
    total_vram_gib = None
    compute_capability = None
    bf16_supported = None
    if device == "cuda":
        torch.cuda.synchronize()
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_name = str(properties.name)
        total_vram_gib = float(properties.total_memory) / 1024**3
        compute_capability = (int(properties.major), int(properties.minor))
        bf16_supported = bool(torch.cuda.is_bf16_supported())

    return DeviceReport(
        device=device,
        torch_version=str(torch.__version__),
        cuda_runtime=torch.version.cuda,
        gpu_name=gpu_name,
        total_vram_gib=total_vram_gib,
        compute_capability=compute_capability,
        bf16_supported=bf16_supported,
        gradient_check_passed=True,
    )
