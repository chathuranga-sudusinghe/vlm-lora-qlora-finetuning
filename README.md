# Resource-Efficient VLM Fine-Tuning for Structured Fashion Product Attribute Extraction

This project investigates whether fine-tuning a small vision-language model (VLM)
improves its ability to extract accurate, machine-readable attributes from fashion
product images. Low-Rank Adaptation (LoRA) and quantized LoRA (QLoRA) are the
parameter-efficient fine-tuning (PEFT) methods used to investigate that task under
an approximately 8 GiB GPU constraint.

Structured attributes can support product-catalog ingestion, indexing, search, and
data-quality workflows. Those workflows motivate the task; this repository focuses
on extraction and its evaluation.

**Status:** working rules, the code license, architecture, and the dataset/experiment
specification are merged through issues #13 and #4. Their unmeasured settings remain
proposals. Issue #5 reviews dependency tooling and runtime profiles; its configuration
is a local draft with owner-verified CPU/CUDA dependency imports and scoped quality
checks, awaiting final review. Diagnostics, tests, and continuous integration remain
separate drafts under #6 and #7. Dataset preparation, controlled evaluation, and
adapter training remain planned. Model fit,
practical training cost, and extraction improvements have not been established.

## Research questions

**Main question:** Does fine-tuning a small VLM improve the accuracy and structural
reliability of fashion-product attribute extraction compared with the same unadapted
model on held-out images?

The supporting questions are:

1. **Method trade-offs:** how do LoRA and QLoRA compare in prediction quality and
   resource efficiency for the selected model, dataset, and controlled training budget?
2. **GPU feasibility:** can these approaches run practically on the target RTX 5060
   Laptop GPU with approximately 8 GiB VRAM, and what trade-offs are required?
3. **Output reliability:** how consistently do adapted models satisfy the output
   contract, and what semantic errors, invalid outputs, and abstention failures remain?

The detailed questions, comparison controls, metrics, and interpretation belong in
the [experiment plan](docs/experiments.md), introduced by issue #4. Final run settings
still require review. Conclusions will be limited to the measured setup; the study
does not seek a universal winning method.

## Task and bounded scope

Use a single image and task instructions to generate structured attributes. The
preferred initial fields are illustrated below; these values are examples, not a
final canonical vocabulary:

```json
{
  "article_type": "shirt",
  "base_colour": "blue",
  "subcategory": "topwear"
}
```

Plan for an explicit `unknown` value when visual evidence is insufficient. Exact
allowed values, normalization, exclusions, and abstention evaluation remain proposed
under issue #4. Valid JSON alone does not establish correct visual extraction or
readiness for unattended catalog use.

The selected dataset is
[Fashion Product Images Small](https://huggingface.co/datasets/ashraq/fashion-product-images-small).
Start with one small model candidate, `HuggingFaceTB/SmolVLM-500M-Instruct`, subject
to actual-model feasibility checks. Product names, year, and other non-visual
metadata must not become image-grounded targets or leak into model inputs.

The planned comparison includes a floating-point base, a four-bit base, LoRA, and
QLoRA using shared task and evaluation rules. The floating-point base is an
unadapted reference, not full-parameter fine-tuning. See the
[architecture](docs/architecture.md#intended-pipeline) for boundaries and data flow.

Keep one dataset, one model, and a bounded experiment. Recommendation systems,
retrieval-augmented generation (RAG), agents, databases/vector stores, marketplace
applications, cloud infrastructure, and distributed training are outside scope.
A tiny local demonstration is optional after evaluation is complete.

## Environment

Target: Python 3.12 on Linux x86-64, including Ubuntu in Windows Subsystem for Linux
(WSL), with an **NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell, approximately 8 GiB)**.
Keep the checkout and `.venv` inside the WSL filesystem.

The [hardware and CUDA assessment](docs/architecture.md#target-gpu-and-cuda-compatibility)
records the actual GPU, driver capability, installed PyTorch runtime, and proposed
upgrade candidate. The CUDA version displayed by `nvidia-smi` is distinct from the
runtime used by a PyTorch wheel.

### Dependency profiles (issue #5 draft)

Use Python 3.12 and uv 0.13.0 with `pyproject.toml` and `uv.lock`. The migration keeps
the existing direct ML dependency pins and the CUDA 12.8 baseline. CUDA 13.0 remains
an evaluation candidate in the hardware assessment, not an automatic upgrade.

The base package has no ML dependencies. Choose exactly one runtime extra for ML
work; each includes the shared pinned Transformers, Datasets, Accelerate, PEFT, TRL,
and Pillow stack. `cpu` selects official CPU Torch/torchvision builds without
bitsandbytes or NVIDIA runtime packages. `cuda` selects official CUDA 12.8 builds
and bitsandbytes. Requesting both is rejected. Add `--group notebook` to a chosen
profile only when notebook support is needed.

Use the repository's `.venv` for local development. When `.venv` does not yet exist,
create it with a short, project-specific terminal label:

```bash
uv venv --python 3.12 --prompt fashion-vlm .venv
```

Install the CUDA profile and development tools into that environment. This command
adds/removes packages to match the lockfile and can download substantial GPU runtime
packages when they are not already cached:

```bash
UV_PROJECT_ENVIRONMENT=.venv uv sync --locked --extra cuda --group dev
```

Activate it for interactive work:

```bash
source .venv/bin/activate
```

On CPU-only machines, select `--extra cpu` in place of `--extra cuda`. CUDA builds
also support CPU operations, so local CPU unit tests do not require a separate
CPU-only installation. A plain sync without a runtime extra, or a sync using
`--only-group dev`, removes ML packages from the selected environment. Keep the
CUDA extra when synchronizing the laptop's development environment.

Follow [AGENTS.md](AGENTS.md): the owner runs each first lock/sync/check command
and supplies its output. Any use of a location outside the repository requires
prior explicit owner approval; temporary validation is not an exception.

**Owner-run validation (2026-10-18):**

- `uv lock` resolved 116 packages. Static lockfile inspection confirms the intended
  Torch 2.11.0 CPU/CUDA variants and removal of the unintended Torch 2.14.1 fallback.
- The CPU profile installed and passed offline dependency imports and package-isolation
  assertions: CPU Torch/torchvision builds, no CUDA runtime, and no bitsandbytes,
  Triton, NVIDIA, or CUDA distributions.
- A locked dry run selecting both `cpu` and `cuda` extras was rejected with the
  expected declared-conflicts error; the repository environment was not changed.
- In the repository `.venv`, `uv pip check` verified compatibility of 89 installed
  packages. Offline imports passed with Torch `2.11.0+cu128`, torchvision
  `0.26.0+cu128`, PyTorch CUDA runtime `12.8`, and bitsandbytes `0.50.1`, including
  the Transformers auto-model/processor classes and TRL `SFTTrainer`.
- Ruff lint, Ruff formatting, and strict MyPy checks passed for the issue #5 package
  initializer, `src/vlm_finetuning/__init__.py`. Pytest availability was verified as
  `9.1.1`; unit and integration suites were not run in these checks.

These results establish dependency installation/imports and the stated tooling
checks. They do not establish GPU execution, quantization kernels, model fit,
correct loss masking, or training feasibility. Generic diagnostics belong to #6;
actual-model feasibility remains a later gate.

Declared dependency bounds alone do not establish the intended loss masks. The
[recorded TRL caveat](docs/references.md#implementation-guidance-follow-up-2026-10-18)
requires inspecting the pinned trainer/processor and verifying answer-only masks
in the later actual-model gate. The inspected TRL 1.10.0 code rejects the
`assistant_only_loss=True` flag for vision datasets; this issue neither upgrades
dependencies nor implements an alternative masking path.

Regenerate the lockfile after approved dependency edits, review its sources and
transitive changes, then verify the profiles before removing the requirements files
from the committed project. Commit the approved configuration and generated lockfile
together; do not maintain parallel dependency lists after the verified migration.

## Checks

The agreed tools are Ruff, MyPy, and pytest, with isolated unit tests, CPU integration
tests, and explicit opt-in CUDA checks. Tool configuration belongs to issue #5;
diagnostics and tests belong to
[issue #6](https://github.com/chathuranga-sudusinghe/vlm-lora-qlora-finetuning/issues/6),
and hosted checks belong to
[issue #7](https://github.com/chathuranga-sudusinghe/vlm-lora-qlora-finetuning/issues/7).
These checks are planned to avoid model/dataset downloads or model training.
Generic CPU/CUDA diagnostics do not establish actual VLM feasibility or extraction quality.

Issue #5 configures the tools and pytest marker conventions. MyPy targets the available
package; issue #6 adds test files to its scope and introduces the diagnostic command.
Do not report tests or an installed CLI as committed capabilities while their files
remain local drafts. `docs/development.md` remains assigned to #6; it does not replace
the owner-first review of the setup examples above.

Follow [AGENTS.md](AGENTS.md) for owner-first execution and review requirements.

## Repository map

```text
AGENTS.md                     Working rules for assistants and contributors
LICENSE                       MIT license for project code and documentation
README.md                     Orientation, current status, and repository map
docs/
  README.md                  Documentation index and ownership
  architecture.md            Hardware target, boundaries, and planned components
  dataset.md                 Provenance, proposed task contract, and split policy
  experiments.md             Full questions, proposed controls, metrics, and run plan
  references.md              Source verification and its limits
  decisions/
    README.md                Decision index and maintenance rules
    DR-001-...md             Foundation rationale and decision status
    DR-002-...md             Dataset and proposed experiment rationale
```

This map shows the merged documentation, including the proposed issue #4 protocol.
The existing `src/baseline_inference.py` remains a standalone local experiment for
later migration; it is not the controlled structured baseline. Work is separated as follows:

| Issue | Paths and status |
|---|---|
| #13: overview and architecture direction, merged | `README.md`, `docs/architecture.md`, `docs/README.md` |
| #4: detailed specification, merged; run settings proposed | `docs/dataset.md`, `docs/experiments.md`, `docs/references.md`, `docs/decisions/DR-002-product-attribute-experiment.md` |
| #5: dependency tooling, local validation recorded; final review pending | `pyproject.toml`, `uv.lock`, `.python-version`, ignore rules, minimal package initialization, `data/README.md`, setup guidance |
| #6: diagnostics and tests, planned | `src/vlm_finetuning/`, `tests/unit/`, `tests/integration/`, `docs/development.md` |
| #7: continuous integration, planned | `.github/workflows/ci.yml` |

Local drafts are not part of an issue's commit merely because they exist in the
working directory. Preserve them and stage only files assigned to the active issue.

## Next milestone

Validate and merge dependency tooling under issue #5, then diagnostics and CPU CI
under #6 and #7. Then implement the audit and frozen manifests,
verify the actual model, establish both base conditions, and run owner-approved
adapter pilots before the final comparison. Keep held-out test data out of tuning.

## License

Project code and documentation use the [MIT license](LICENSE). This does not relicense
third-party datasets, images, dependencies, model weights, or adapters; retain and
check their applicable terms before redistributing them. See the
[dataset provenance and image-use limitations](docs/dataset.md#provenance-and-use-terms).
