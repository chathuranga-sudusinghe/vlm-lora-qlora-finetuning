# Small VLM adaptation for product attributes

A focused vision-language model (VLM) project comparing a base model with Low-Rank
Adaptation (LoRA) and quantized LoRA (QLoRA) for product attribute extraction.
The selected dataset is
[Fashion Product Images Small](https://huggingface.co/datasets/ashraq/fashion-product-images-small).

**Status:** working rules and the code license are merged. This change adds project
orientation and architecture documentation. Dependency tooling, diagnostics, tests,
and continuous integration have local drafts assigned to separate issues; they are
not yet merged project capabilities. Dataset preparation, training, evaluation, and
a demo remain planned. There are no adaptation results yet.

The existing `src/baseline_inference.py` is a standalone local experiment, preserved
for a separate migration.

## Environment

Target: Python 3.12 on Linux x86-64, including Ubuntu in Windows Subsystem for Linux
(WSL), with an **NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell, approximately 8 GiB)**.
Keep the checkout and `.venv` inside the WSL filesystem.

The [hardware and CUDA assessment](docs/architecture.md#target-gpu-and-cuda-compatibility)
records the actual GPU, driver capability, installed PyTorch runtime, and proposed
upgrade candidate. The CUDA version displayed by `nvidia-smi` is distinct from the
runtime used by a PyTorch wheel.

[Issue #5](https://github.com/chathuranga-sudusinghe/vlm-lora-qlora-finetuning/issues/5)
will introduce `pyproject.toml`, uv locking, and explicit CPU/CUDA profiles. Setup
commands will be documented when that configuration is reviewed and merged.

## Checks

The agreed tools are Ruff, MyPy, and pytest, with isolated unit tests, CPU integration
tests, and explicit opt-in CUDA checks. Tool configuration belongs to issue #5;
diagnostics and tests belong to
[issue #6](https://github.com/chathuranga-sudusinghe/vlm-lora-qlora-finetuning/issues/6),
and hosted checks belong to
[issue #7](https://github.com/chathuranga-sudusinghe/vlm-lora-qlora-finetuning/issues/7).
These checks are planned to avoid model/dataset downloads or model training.

Follow [AGENTS.md](AGENTS.md) for owner-first execution and review requirements.

## Repository map

```text
AGENTS.md                     Working rules for assistants and contributors
LICENSE                       MIT license for project code and documentation
README.md                     Orientation, current status, and repository map
docs/
  README.md                  Documentation index and ownership
  architecture.md            Hardware target, boundaries, and planned components
  decisions/
    README.md                Decision index and maintenance rules
    DR-001-...md             Foundation rationale and decision status
```

This map describes the files introduced by issue #3 and the already-merged working
rules/license. The remaining foundation files are separated into the issues below:

| Issue | Planned paths |
|---|---|
| #4: dataset and experiment documentation | `docs/dataset.md`, `docs/experiments.md`, `docs/references.md`, dataset decision |
| #5: dependency and quality-tool configuration | `pyproject.toml`, `uv.lock`, `.python-version`, ignore rules, minimal package initialization, `data/README.md` |
| #6: diagnostics and tests | `src/vlm_finetuning/`, `tests/unit/`, `tests/integration/`, `docs/development.md` |
| #7: continuous integration | `.github/workflows/ci.yml` |

Local drafts are not part of an issue's commit merely because they exist in the
working directory. Preserve them and stage only files assigned to the active issue.

## Next milestone

Review and merge the remaining foundation issues separately. Then implement
deterministic dataset preparation with duplicate-aware split assignment and a saved
manifest, migrate baseline inference, and establish evaluation before adapter training.

## License

Project code and documentation use the [MIT license](LICENSE). This does not relicense
third-party datasets, images, dependencies, model weights, or adapters; retain and
check their applicable terms before redistributing them. Detailed dataset provenance
will be documented in issue #4.
