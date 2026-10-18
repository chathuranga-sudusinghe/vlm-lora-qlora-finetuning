"""Command-line entry point for a small CPU or CUDA environment diagnostic."""

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict

from vlm_finetuning.environment import Device, check_device

LOGGER = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """Print a JSON device report, returning a nonzero status on runtime failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    device: Device = "cuda" if args.device == "cuda" else "cpu"
    try:
        report = check_device(device)
    except (ModuleNotFoundError, RuntimeError) as error:
        LOGGER.error("Device check failed: %s", error)
        return 1
    print(json.dumps(asdict(report), indent=2))
    return 0
