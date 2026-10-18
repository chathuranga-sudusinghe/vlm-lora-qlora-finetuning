"""Keep GPU execution explicit, even on machines with an available GPU."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-cuda", action="store_true", default=False, help="Run tests requiring a CUDA device."
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-cuda"):
        return
    skip_cuda = pytest.mark.skip(reason="CUDA tests require --run-cuda")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)
