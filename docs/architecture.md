# Architecture

## Goal and boundary

Investigate whether fine-tuning a small vision-language model (VLM) improves accurate,
structured fashion-product attribute extraction over the same unadapted model.
Low-Rank Adaptation (LoRA) and quantized LoRA (QLoRA) are the adaptation methods;
their comparison supports the main task question.

The [project overview](../README.md#research-questions) summarizes the question
hierarchy. The detailed question-to-evidence mapping and experimental protocol
belong in `docs/experiments.md`; final experimental settings remain proposed.
This document owns component boundaries and the planned flow, not a second protocol.

The owner-selected output fields are `masterCategory`, `subCategory`, `articleType`,
and `baseColour`, using the exact dataset column names under issue #23. The planned
contract allows explicit `unknown` values under a declared inference/decision policy.
The [dataset specification](dataset.md#initial-task) owns the contract and vocabulary
policy; final labels and eligibility rules remain to be derived from Development
and frozen before Test access. Inputs contain only the image and instructions,
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

Working rules (`AGENTS.md`), the MIT code license, architecture, and dataset/experiment
documents are tracked. Final experimental settings and actual-model feasibility
remain unvalidated. The current repository includes these foundation components:

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
  This issue is completed; hosted evidence is recorded in `docs/development.md`.

Source inspection (#22) is completed in `notebooks/01_dataset_overview.ipynb`, and
the four-field output scope (#23) is reflected in the dataset/experiment documents.
Issue #28 adds `notebooks/02_create_canonical_dataset.ipynb`: source-preserving merge,
round-trip validation, local provenance manifest, and inspection of the saved artifact.
Reruns reuse a complete, checksum-matching artifact pair without overwriting; the
saved schema/count/ID checks still run. Incomplete or invalid pairs stop execution.
Bounded synthetic checks cover that workflow. The owner's saved execution outputs
report successful merge validation; the saved row/image previews have been inspected.
The [dataset evidence record](dataset.md#saved-execution-evidence--2026-10-19) owns
the observed counts, provenance, and limitations. No new real-data run was performed
to review those outputs. Issue #28 / PR #29 is merged. Issue #30 documents the
Test-first policy, but no Test seal, Development selector, preparation code, or
notebook execution guard is implemented by that documentation change.

Issue #33 moves the earlier free-form experiment into
`src/vlm_finetuning/prototype_inference.py`; `src/baseline_inference.py` is a thin
compatibility entry point. Explicit calls validate the image and requested device
before pretrained loading; imports and help do not load optional ML libraries.
CUDA never silently falls back to CPU. Mocked unit tests cover orchestration and
failure handling, not actual-model compatibility or quality. This prototype does
not implement the controlled four-field baseline, rejection policy, or Test access
guards. Its [development guide](development.md#inference-prototype) owns invocation
and limitations. Device diagnostics still do not establish VLM/training feasibility.

## Intended pipeline

The merge/inspection step is implemented with synthetic checks and owner-run saved
execution evidence. The subsequent audit, splitting, model, and evaluation steps
remain planned:

```text
Two retained raw shards, with verified revision or explicitly identified local snapshot
  -> concatenate in fixed order, preserving every raw column, value, image byte, and row
  -> validate saved canonical merged Parquet and provenance manifest in data/interim/
  -> historical pre-seal inspection of the saved merged artifact (#28)
  -> select fixed-seed random Test membership (~10%); freeze membership metadata
  -> logically seal Test; expose only the Development complement (~90%)
  -> audit Development images/labels/groups and version task policies
  -> split Development into Train/Validation (~8/9 and ~1/9), mainly by articleType
     with meaningful Development duplicate/related-image groups kept disjoint
  -> freeze development artifacts before instruction expansion; fit statistics on Train
  -> validate shared task, processor, and run configuration
  -> verify actual VLM inference and adapter-training feasibility
  -> evaluate floating-point and four-bit bases on development data
  -> pilot and select LoRA / QLoRA using training and validation only
  -> select decision policy and thresholds on Validation
  -> freeze weights and the complete system, including unknown/rejection policy
  -> recover exact sealed Test records for authorized final evaluation of all four conditions
  -> save predictions, quality/resource metrics, and reproducibility evidence
  -> report the main task result, supporting questions, and remaining errors
```

### Data and task boundary

Dataset provenance, the canonical merged artifact, audit specification,
raw-to-canonical label mappings, and split integrity belong in
[docs/dataset.md](dataset.md#canonical-merged-artifact). Subsequent processing uses
the validated merged Parquet file; the raw shards are retained for provenance and
rebuilding. The merge retains all 11 columns and performs no label canonicalization,
selection, or split assignment. The four prediction fields do not restrict its schema.
Record corrupt/missing records, conflicting labels, grouping decisions, retained
class support, and exclusions within Development only after Test sealing. Audit the
`masterCategory` / `subCategory` / `articleType` hierarchy using Development-derived
relationships.
These are related catalog levels, not independent visual capabilities. Predict all
four fields explicitly; do not silently derive a predicted category from another
predicted field during the main evaluation.

The [preparation policy](dataset.md#preparation-and-splitting-rules) owns selection,
reproducibility, group constraints, and sealing restrictions. Future Development
loaders must enforce membership before audits, previews, or model use. The two
existing whole-source notebooks are historical pre-seal workflows; prose warnings
do not enforce this boundary. Unique IDs alone do not protect against image
duplicates, and Development-only grouping does not establish a duplicate-free Test
boundary. Choose categorical policies using Development; fit learned statistics
with Train only. Use Validation for selection and never Test for policy revision.

One shared task contract must validate required keys, string types, allowed values,
and abstention values across every condition. Preserve raw outputs and distinguish
valid known predictions, valid abstentions, and invalid outputs. A shared source-code
decision layer owns supported-input checks and field/record rejection; known-label
validation alone cannot establish unsupported-input detection. Preserve raw validation
results separately from final policy outputs, with reasons in sidecars. Catalog
labels alone do not establish whether an image supports a field; the detailed protocol must
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
and abstention with explicit denominators. Separate in-scope prediction from
unsupported-input rejection, accounting for every sealed Test member, including
unreadable/unscorable records. Invalid raw outputs remain failures in raw task
scores even when final policy outputs differ. Valid JSON or an accepted `unknown`
value alone does not establish visual correctness, calibrated uncertainty, or
readiness for unattended catalog use.

Record peak allocated/reserved GPU memory, training time and throughput, trainable
parameters, adapter size, and final reconstruction/deployment footprint. Measure
inference latency under a shared protocol if practical. Agree practical runtime
limits before final runs, and report measured fit/failures on this laptop rather
than generalizing to every consumer GPU.

`docs/experiments.md` owns exact metric definitions, primary-outcome selection,
checkpoint rules, sample/update budgets, seed confirmation, and the small error review.
Final experimental settings still require owner review before the protocol is frozen.
This architecture does not authorize preparation, model execution, or training;
all execution follows `AGENTS.md`.

## Artifact boundaries

Source, small configuration, tests, documentation, and the dependency lockfile belong
in Git. Raw/derived data must stay in ignored `data/` paths; model downloads use a local
cache. Checkpoints, adapters, predictions, and generated run evidence belong in ignored
`outputs/`, `checkpoints/`, or `artifacts/` directories. The committed ignore rules
exclude these generated artifacts. The canonical merged Parquet and merge manifest
use ignored `data/interim/`; this manifest records preservation/provenance, not split
membership or a completed image-duplicate audit.

Future Test sealing should store immutable membership metadata bound to that
canonical artifact, not a physical Test Parquet by default. Development artifacts
must identify their policy and Train/Validation membership versions. Keep these
generated records ignored; the detailed requirements belong in
[dataset.md](dataset.md#test-membership-and-logical-sealing).

Each evidence-producing run must save its resolved configuration, data/model revisions
or snapshot identifiers, exact split membership, vocabulary/manifest hashes, seeds,
processor/generation settings, adapter/quantization settings, package builds, device
details, and code revision with any dirty state. Preserve predictions, metrics,
resource measurements, and exclusion/audit records or immutable references to them.
Distinguish adapter bytes from the base weights needed to reconstruct the model.
Small intentional aggregate reports may enter Git after owner review; generated
predictions and model/data artifacts remain local. The code MIT license does not
replace third-party artifact terms.

A future production extension could collect new rejected inputs for human review
and separately authorized retraining with fresh evaluation data. This is documentation
only: no production feedback loop or second model is implemented, and rejected
records from the current Test set must not become training data.
