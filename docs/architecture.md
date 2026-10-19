# Architecture

## Goal and boundary

Investigate whether fine-tuning a small vision-language model (VLM) improves accurate,
structured fashion-product attribute extraction over the same unadapted model.
Low-Rank Adaptation (LoRA) and quantized LoRA (QLoRA) are the adaptation methods;
their comparison supports the main task question.

The [project overview](../README.md#research-questions) summarizes the question
hierarchy. The detailed question-to-evidence mapping and experimental protocol
belong in `docs/experiments.md` under issue #4; its current draft requires alignment.
This document owns component boundaries and the planned flow, not a second protocol.

The owner-selected output fields are `masterCategory`, `subCategory`, `articleType`,
and `baseColour`, using the exact dataset column names under issue #23. The planned
contract allows explicit `unknown` values for insufficient visual evidence. The
[dataset specification](dataset.md#initial-task) owns the contract and vocabulary
policy; final labels and eligibility rules remain to be derived from training data
and frozen before evaluation. Inputs contain only the image and instructions,
never product names or target metadata.

The initial target is one Linux/Windows Subsystem for Linux (WSL) laptop with an
**NVIDIA GeForce RTX 5060 Laptop GPU**, using the **Blackwell** architecture.
VLM training memory fit, training time, and adaptation quality remain unmeasured.
Catalog ingestion, indexing, search, and data quality motivate machine-readable
outputs; downstream systems are outside this implementation. Keep one dataset and
one small model initially. Recommendation systems, retrieval-augmented generation
(RAG), agents, databases/vector stores, cloud services, marketplace applications,
and distributed training are outside scope. A tiny local demo is optional after evaluation.

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

Working rules (`AGENTS.md`), the MIT code license, and the original architecture
introduced by issue #3 are merged. Issue #13 aligns the overview and architecture;
issue #4 owns the unmerged dataset/experiment specification, which still needs the
refined questions and task contract. The components below have local drafts but
remain assigned to separate, unmerged implementation issues:

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
Its migration and quality-check scope will be reviewed separately. It is not yet
the shared structured baseline. The planned device diagnostics do not validate its
model inference behavior or establish VLM/quantization/training feasibility.

## Intended pipeline

The following pipeline is planned, not implemented or validated:

```text
Verified dataset revision or explicitly identified local snapshot
  -> audit schema, images, labels, dimensions, and exclusion reasons
  -> identify duplicate/product groups
  -> assign candidate train / validation / test groups
  -> derive vocabulary and eligibility rules from candidate training data
  -> freeze mappings, exclusions, split manifest, and instruction examples
  -> validate shared task, processor, and run configuration
  -> verify actual VLM inference and adapter-training feasibility
  -> evaluate floating-point and four-bit bases on development data
  -> pilot and select LoRA / QLoRA using training and validation only
  -> freeze final configurations and evaluate all four conditions on held-out data
  -> save predictions, quality/resource metrics, and reproducibility evidence
  -> report the main task result, supporting questions, and remaining errors
```

### Data and task boundary

Dataset provenance, the audit specification, raw-to-canonical mappings, and split
integrity belong in planned `docs/dataset.md` under
[issue #4](https://github.com/chathuranga-sudusinghe/vlm-lora-qlora-finetuning/issues/4).
Record corrupt/missing records, conflicting labels, grouping decisions, retained
class support, and exclusions. Audit the `masterCategory` / `subCategory` /
`articleType` hierarchy using relationships derived from candidate training data.
These are related catalog levels, not independent visual capabilities. Predict all
four fields explicitly; do not silently derive a predicted category from another
predicted field during the main evaluation.

Split related images before generating multiple instructions. Choose class subsets
and normalization from candidate training data, apply fixed rules to the other
partitions, and save exact membership. Use validation for selection; held-out test
labels and outcomes must not guide prompts, mappings, exclusions, or hyperparameters.

One shared task contract must validate required keys, string types, allowed values,
and abstention values across every condition. Preserve raw outputs and distinguish
valid known predictions, valid abstentions, and invalid outputs. Catalog labels alone
do not establish whether an image supports a field; the detailed protocol must
define how any small human-reviewed visibility subset is used.

### Actual-model feasibility gate

Start with `HuggingFaceTB/SmolVLM-500M-Instruct` as a candidate, not a verified fit.
Before adapter pilots, owner-approved checks must exercise the actual model and:

- Processor/chat template compatibility, image-token handling, and answer-loss masks.
- Floating-point inference in the chosen BF16 or FP16 (16-bit floating-point) mode
  and four-bit loading/inference.
- Exact language-only adapter targets and trainable parameter counts, with the vision
  encoder and projector initially frozen. Verify matched modules rather than relying
  on projection-name suffixes that may also match vision layers.
- A tiny adapter forward/backward pass, finite gradients, peak memory, and save/reload.

Declare which components are quantized; freezing weights does not define their precision.
Measure the supported image resize/crop and sequence policy rather than assuming
small source images imply low memory use. Record baseline output validity; perfect
base-model JSON is not a prerequisite for studying improvement through fine-tuning.
A successful generic tensor probe is only an earlier environment check.

### Shared adaptation and evaluation boundary

Use one reusable loading/processing/generation path and one evaluation implementation:

| Condition | Role |
|---|---|
| Floating-point base, BF16 or FP16 as verified | Unadapted task reference |
| Four-bit base | Quantization-only control |
| LoRA with floating-point base | Adaptation without base quantization |
| QLoRA with four-bit base | Adaptation with quantized base weights |

The floating-point base is not full-parameter fine-tuning. Compare each adapter with
its corresponding base to distinguish adaptation from quantization. Keep model
revision, data, task instructions, processor policy, generation, and scoring aligned.
Match adapter settings and training/update budgets where applicable; document
unavoidable differences and keep method-specific tuning separate.

Use a small validated experiment configuration rather than constants scattered
through scripts. Add responsibilities to the existing `vlm_finetuning` package as
implemented; do not introduce a configuration framework or parallel notebook pipeline.
Optional adapter-target ablations and extra models can wait until the core result.

The planned evaluation must distinguish structural validity from attribute correctness.
Measure complete-record and per-field quality, parse/schema failures, invalid labels,
and abstention with explicit denominators. Invalid outputs remain failures in the
main task scores. Valid JSON or an accepted `unknown` value alone does not establish
visual correctness, calibrated uncertainty, or readiness for unattended catalog use.

Record peak allocated/reserved GPU memory, training time and throughput, trainable
parameters, adapter size, and final reconstruction/deployment footprint. Measure
inference latency under a shared protocol if practical. Agree practical runtime
limits before final runs, and report measured fit/failures on this laptop rather
than generalizing to every consumer GPU.

Planned `docs/experiments.md` owns exact metric definitions, primary-outcome selection,
checkpoint rules, sample/update budgets, seed confirmation, and the small error review.
Its existing draft and DR-002 require alignment before the protocol is finalized.
This architecture does not authorize preparation, model execution, or training;
all execution follows `AGENTS.md`.

## Artifact boundaries

Source, small configuration, tests, documentation, and the dependency lockfile belong
in Git. Raw/derived data must stay in ignored `data/` paths; model downloads use a local
cache. Checkpoints, adapters, predictions, and generated run evidence belong in ignored
`outputs/`, `checkpoints/`, or `artifacts/` directories. Issue #5 will commit the
corresponding ignore rules and local-data instructions.

Each evidence-producing run must save its resolved configuration, data/model revisions
or snapshot identifiers, exact split membership, vocabulary/manifest hashes, seeds,
processor/generation settings, adapter/quantization settings, package builds, device
details, and code revision with any dirty state. Preserve predictions, metrics,
resource measurements, and exclusion/audit records or immutable references to them.
Distinguish adapter bytes from the base weights needed to reconstruct the model.
Small intentional aggregate reports may enter Git after owner review; generated
predictions and model/data artifacts remain local. The code MIT license does not
replace third-party artifact terms.
