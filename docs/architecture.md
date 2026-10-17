# Architecture

## Goal and boundary

Adapt a small vision-language model (VLM) to produce product attributes from a
single product image, then measure the quality and resource tradeoffs of Low-Rank
Adaptation (LoRA) and quantized LoRA (QLoRA).

The initial target is one Linux/Windows Subsystem for Linux (WSL) laptop with an
**NVIDIA GeForce RTX 5060 Laptop GPU**, using the **Blackwell** architecture.
VLM training memory fit, training time, and adaptation quality remain unmeasured.
A database, cloud service, web API, and distributed training are outside the current scope.

## Target GPU and CUDA compatibility

Hardware observations from the owner's screenshots and the earlier local device
inspection, with installed package metadata read on 2026-10-17:

| Property | Value and evidence |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU; the owner's terminal display truncates the suffix, while the earlier full device report includes it |
| Architecture | Blackwell |
| Compute capability | 12.0, kernel target `sm_120`; consistent with NVIDIA's RTX 5060 listing and the earlier device report |
| Reported video memory | 8151 MiB, approximately 7.96 GiB; an 8 GB-class laptop GPU |
| Driver version | 592.01, shown in the owner's `nvidia-smi` screenshot |
| Driver CUDA capability | 13.1, shown by `nvidia-smi`; this is not the installed PyTorch runtime |
| Existing PyTorch package | `2.11.0+cu128`, read from `.venv/lib/python3.12/site-packages/torch/version.py` |
| Existing PyTorch CUDA runtime | 12.8, recorded in the same package metadata |

[NVIDIA's GPU table](https://developer.nvidia.com/cuda/gpus) identifies RTX 5060 as
compute capability 12.0. Libraries must provide compatible kernels for this GPU;
the word Blackwell alone does not establish that a particular wheel supports `sm_120`.

The [NVIDIA management-tool documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html#cuda-version)
defines the displayed CUDA version as driver capability. It does not establish
which toolkit is installed or which runtime a Python package uses. A supported
older runtime can run with a newer driver under
[NVIDIA's compatibility rules](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

### Runtime assessment for issue #5

CUDA 12.8 is already a compatible starting point for this GPU; it is not rejected
merely because the driver displays 13.1. The official
[PyTorch 2.11.0 installation matrix](https://pytorch.org/get-started/previous-versions/#v2110)
offers both CUDA 12.8 (`cu128`) and CUDA 13.0 (`cu130`) wheels with torchvision 0.26.0.
The current [bitsandbytes build table](https://huggingface.co/docs/bitsandbytes/main/en/installation)
also lists `sm120` for CUDA 12.8 and 13.0 builds; main-branch documentation is not
proof that every pinned release has been exercised on this laptop.

**Proposed upgrade candidate:** evaluate PyTorch 2.11.0/torchvision 0.26.0 with CUDA
13.0 wheels in issue #5, retaining the existing CUDA 12.8 environment until owner
review and validation. This offers an updated runtime without simultaneously
changing the PyTorch release. It is a recommendation, not an accepted migration
or a claim of better performance. Do not choose a wheel solely to match `13.1`.

Before accepting a runtime change, verify the specific locked builds, device
forward/backward operations, BF16 (bfloat16) execution, and bitsandbytes quantization
under the owner-first rules. This documentation issue changes no dependencies,
driver, local toolkit, environment, or lockfile and runs no GPU diagnostics.

## Current state and planned foundation

Working rules (`AGENTS.md`) and the MIT code license are merged. Issue #3 introduces
this architecture and its companion documentation. The components below have local
drafts but remain assigned to separate, unmerged implementation issues:

- Issue #6, `src/vlm_finetuning/environment.py`: explicit CPU/CUDA selection, a tiny synthetic
  forward/backward check, and an immutable runtime report. It imports PyTorch only
  when called and does not load a model.
- Issue #6, `src/vlm_finetuning/cli.py`: command parsing, JSON output, and failure exit status.
- Issue #6, `src/vlm_finetuning/__main__.py`: module command entry point.
- Issue #6, `tests/unit/`: command dispatch, report serialization, and failure handling
  without requiring PyTorch or a GPU.
- Issue #6, `tests/integration/`: real PyTorch and the module command. CUDA execution
  is opt-in and requires an actual accessible device.
- Issue #5, `pyproject.toml` and `uv.lock`: package metadata, dependency profiles, and quality tools.
- Issue #7, `.github/workflows/ci.yml`: quality checks and CPU integration tests.

The owner's existing `src/baseline_inference.py` remains a standalone experiment.
Its migration and quality-check scope will be reviewed separately. The planned
device diagnostics do not validate its model inference behavior.

## Intended pipeline — not implemented yet

```text
Pinned Hugging Face dataset revision
  -> validate images and selected labels
  -> identify duplicate/product groups
  -> assign train / validation / test groups
  -> save manifest and build instruction examples
  -> evaluate base model
  -> train language-model adapters
  -> evaluate selected adapters with the same protocol
  -> save predictions, metrics, run configuration, and hardware measurements
```

Issue #4 will define dataset targets and split rules in `docs/dataset.md`, and the
proposed model comparison and metrics in `docs/experiments.md`. Those documents
are outside this issue's commit.

Add preparation, inference, training, and evaluation modules to `vlm_finetuning`
as each milestone is implemented. Shared helpers belong in the package, not copied
between scripts or notebooks. A small demo follows working evaluation.

## Artifact boundaries

Source, small configuration, tests, documentation, and the dependency lockfile belong
in Git. Raw/derived data must stay in ignored `data/` paths; model downloads use a local
cache. Checkpoints, adapters, predictions, and run evidence belong in ignored
`outputs/`, `checkpoints/`, or `artifacts/` directories. Issue #5 will commit the
corresponding ignore rules and local-data instructions.

Later runs must save data/model revision identifiers, exact split membership,
configuration, seed, dependency versions, and device measurements. The code MIT
license does not replace third-party artifact terms.
