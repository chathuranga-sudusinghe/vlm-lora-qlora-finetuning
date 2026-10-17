# Resource-Efficient VLM Fine-Tuning for Structured Fashion Product Attribute Extraction

This project investigates whether fine-tuning a small vision-language model (VLM)
improves its ability to extract accurate, machine-readable attributes from fashion
product images. Low-Rank Adaptation (LoRA) and quantized LoRA (QLoRA) are the
parameter-efficient fine-tuning (PEFT) methods used to investigate that task under
an approximately 8 GiB GPU constraint.

Structured attributes can support product-catalog ingestion, indexing, search, and
data-quality workflows. Those workflows motivate the task; this repository focuses
on extraction and its evaluation.

**Status:** working rules, the code license, and the original architecture documentation
are merged. Issue #13 aligns the overview and architecture with this direction.
Dataset and experiment documents have local, unmerged drafts under issue #4 that
still need alignment. Dependency tooling, diagnostics, tests, and continuous
integration also have drafts assigned to separate issues. Dataset preparation,
controlled evaluation, and adapter training remain planned. Model fit, practical
training cost, and extraction improvements have not been established.

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
`docs/experiments.md` under
[issue #4](https://github.com/chathuranga-sudusinghe/vlm-lora-qlora-finetuning/issues/4).
Its draft must be updated before the experiment protocol is finalized. Conclusions
will be limited to the measured setup; the study does not seek a universal winning method.

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
Generic CPU/CUDA diagnostics do not establish actual VLM feasibility or extraction quality.

Follow [AGENTS.md](AGENTS.md) for owner-first execution and review requirements.

## Repository map

```text
AGENTS.md                     Working rules for assistants and contributors
LICENSE                       MIT license for project code and documentation
README.md                     Orientation, current status, and repository map
docs/
  README.md                  Documentation index and ownership
  architecture.md            Hardware target, boundaries, and planned components
  dataset.md                 Draft in #4: provenance, task contract, and split policy
  experiments.md             Draft in #4: full questions, controls, metrics, and run plan
  references.md              Draft in #4: source verification and its limits
  decisions/
    README.md                Decision index and maintenance rules
    DR-001-...md             Foundation rationale and decision status
    DR-002-...md             Draft in #4: dataset and experiment rationale
```

This map includes merged documents and the local drafts explicitly marked above.
The existing `src/baseline_inference.py` remains a standalone local experiment for
later migration; it is not the controlled structured baseline. Work is separated as follows:

| Issue | Planned paths |
|---|---|
| #13: project overview and architecture alignment | `README.md`, `docs/architecture.md`, `docs/README.md` |
| #4: dataset and experiment specification | `docs/dataset.md`, `docs/experiments.md`, `docs/references.md`, `docs/decisions/DR-002-product-attribute-experiment.md` |
| #5: dependency and quality-tool configuration | `pyproject.toml`, `uv.lock`, `.python-version`, ignore rules, minimal package initialization, `data/README.md` |
| #6: diagnostics and tests | `src/vlm_finetuning/`, `tests/unit/`, `tests/integration/`, `docs/development.md` |
| #7: continuous integration | `.github/workflows/ci.yml` |

Local drafts are not part of an issue's commit merely because they exist in the
working directory. Preserve them and stage only files assigned to the active issue.

## Next milestone

Align the detailed issue #4 specification with this overview and review each
documentation scope separately. Complete dependency tooling, diagnostics, and CPU
CI under issues #5, #6, and #7. Then implement the audit and frozen manifests,
verify the actual model, establish both base conditions, and run owner-approved
adapter pilots before the final comparison. Keep held-out test data out of tuning.

## License

Project code and documentation use the [MIT license](LICENSE). This does not relicense
third-party datasets, images, dependencies, model weights, or adapters; retain and
check their applicable terms before redistributing them. Dataset provenance and
image-use limitations belong in `docs/dataset.md`, pending review under issue #4.
