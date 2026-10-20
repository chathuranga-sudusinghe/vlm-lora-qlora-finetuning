"""Explicit single-image, free-form prototype; not the controlled task baseline."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, cast

from vlm_finetuning.environment import Device

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from torch import Tensor

LOGGER = logging.getLogger(__name__)
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
CACHE_DIR = Path(".cache/huggingface")
MAX_NEW_TOKENS = 128
PROMPT = (
    "Analyze this product image. "
    "Identify the product category, main color, "
    "and visible usage/style."
)


class _Processor(Protocol):
    """The small part of the dynamically selected processor used here."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str: ...

    def __call__(
        self, *, text: str, images: list[PILImage], return_tensors: str
    ) -> dict[str, Tensor]: ...

    def batch_decode(self, sequences: Tensor, *, skip_special_tokens: bool) -> list[str]: ...


class _Model(Protocol):
    """The inference interface supplied by the selected auto-model factory."""

    def to(self, device: str) -> Self: ...

    def eval(self) -> Self: ...

    def generate(self, **kwargs: object) -> Tensor: ...


@dataclass(frozen=True)
class PrototypeResult:
    """One prototype answer and optional process-level CUDA memory diagnostics."""

    answer: str
    device: Device
    gpu_name: str | None
    peak_allocated_vram_gib: float | None


def run_prototype(image_path: Path, device: Device) -> PrototypeResult:
    """Run only when explicitly called with owner-authorized image/model access.

    Run from the repository root so the model cache stays in .cache/huggingface.
    Imports are lazy. File/image and device checks precede pretrained loading.
    Exceptions retain their original causes for callers; the CLI reports failures.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported device: {device}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image is not a regular file: {image_path}")

    from PIL import Image

    with Image.open(image_path) as source, source.convert("RGB") as image:
        import torch

        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device.")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("The requested CUDA prototype requires BF16 support.")

        from transformers import AutoModelForImageTextToText, AutoProcessor

        # These auto-factories have dynamic return types in the pinned Transformers
        # release. Keep the typing boundary here instead of suppressing project errors.
        load_processor = cast(Callable[..., _Processor], AutoProcessor.from_pretrained)
        load_model = cast(Callable[..., _Model], AutoModelForImageTextToText.from_pretrained)
        processor = load_processor(MODEL_ID, cache_dir=str(CACHE_DIR))
        model = load_model(
            MODEL_ID,
            cache_dir=str(CACHE_DIR),
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).to(device)
        model.eval()

        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        if "input_ids" not in inputs:
            raise ValueError("Processor did not return input_ids.")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        gpu_name = None
        peak_vram = None
        if device == "cuda":
            gpu_name = str(torch.cuda.get_device_name())
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                return_dict_in_generate=False,
            )
        # This decoder-only candidate returns prompt tokens followed by its answer.
        answer_tokens = generated[:, inputs["input_ids"].shape[1] :]
        decoded = processor.batch_decode(answer_tokens, skip_special_tokens=True)
        if len(decoded) != 1 or not decoded[0].strip():
            raise RuntimeError("Expected one nonempty prototype answer.")
        if device == "cuda":
            torch.cuda.synchronize()
            peak_vram = float(torch.cuda.max_memory_allocated()) / 1024**3

    return PrototypeResult(decoded[0], device, gpu_name, peak_vram)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse explicit inputs and report prototype failures with a nonzero status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Owner-authorized image path.")
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    device: Device = args.device
    try:
        result = run_prototype(args.image, device)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        LOGGER.error("Prototype inference failed: %s", error)
        return 1

    print(f"Prototype model: {MODEL_ID}")
    print(f"Device: {result.device}")
    if result.gpu_name is not None:
        print(f"GPU: {result.gpu_name}")
    print("\n--- FREE-FORM PROTOTYPE OUTPUT (NOT THE CONTROLLED BASELINE) ---")
    print(result.answer)
    if result.peak_allocated_vram_gib is not None:
        print(f"\nPeak allocated VRAM: {result.peak_allocated_vram_gib:.2f} GiB")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
