# DR-002: A bounded product-attribute experiment

- **Date:** 2026-10-17
- **Updated:** 2026-10-18
- **Status:** dataset accepted; project direction merged through #13 / PR #14;
  detailed task contract and experiment settings proposed under #4

## Context

The local baseline already uses SmolVLM-500M-Instruct to describe product images.
The owner wants a meaningful, manageable project on an RTX 5060 Laptop GPU and
selected Fashion Product Images Small after reviewing the proposal.

The refined goal is to investigate whether fine-tuning improves accurate, reliable
structured fashion-product extraction over the same unadapted model. LoRA and QLoRA
are methods supporting that question. Their task-specific quality/resource trade-off,
practical operation on approximately 8 GiB VRAM, and output reliability are supporting
questions, not separate platform projects. Structured outputs motivate catalog,
indexing, search, and ingestion use cases without implementing those systems here.

The two raw Parquet shards are now present in `data/raw/`. Their exact inventory
and verification limits are maintained in [dataset.md](../dataset.md). Local presence
does not establish schema correctness, image readability, or a verified source revision.

## Decision and proposal

Use `ashraq/fashion-product-images-small` as the selected source. Keep the existing
small model candidate initially; model fit and pinned-stack compatibility remain
unverified. Propose `article_type`, `base_colour`, and `subcategory` from the catalog
fields `articleType`, `baseColour`, and `subCategory`, respectively. This replaces
the earlier broad-category target. Require image-only inputs, a canonical vocabulary,
and a strict three-field JSON contract with explicit `unknown` semantics.

Compare four conditions: floating-point unadapted base, four-bit unadapted base,
Low-Rank Adaptation (LoRA), and quantized LoRA (QLoRA). Match each adapter to its
corresponding base to separate adaptation from quantization. Use shared structured
instructions, processing, and evaluation rather than comparing a free-form baseline
against structured fine-tuned answers. Freeze the vision encoder/projector initially
and adapt verified language-attention targets; avoid all-component adaptation by default.

Require a documented audit, training-derived label/eligibility rules, and frozen
duplicate/product-aware manifests before instruction expansion. Establish actual-model
inference/backward/quantization/save-reload feasibility before adapter pilots; generic
device diagnostics are insufficient. Runtime evidence remains owned by
[architecture.md](../architecture.md#target-gpu-and-cuda-compatibility) and issue #5.

[dataset.md](../dataset.md) owns the proposed contract, provenance, vocabulary, and
preparation policy. [experiments.md](../experiments.md) owns the full research questions,
comparisons, metrics, proposed primary outcome, budgets, configuration/artifacts,
seed confirmation, and small review protocol. [references.md](../references.md)
records sources and verification limits. This record explains the rationale rather
than duplicating those specifications.

Canonical labels, class subset, split/sample budgets, hyperparameters, and practical
acceptance criteria remain proposed until reviewed. Documentation acceptance is not
evidence of dataset validation or permission to execute preparation/training.

## Alternatives considered

- Document question answering: practical but optical character recognition and
  high-resolution document processing increase scope and memory pressure.
- Chart question answering: measurable but adds chart reading and numerical reasoning.
- A conventional image classifier: likely simpler for a fixed schema; the VLM project
  is justified as an investigation of structured generation and efficient adaptation.
- Larger product datasets or models: defer until the initial evaluation shows a need.
- LoRA versus QLoRA as the main goal: too method-centric; retain it as a supporting
  comparison serving the practical extraction question.
- External-photo benchmarking, extensive ablations, or a web platform: defer; the
  core result needs a controlled held-out comparison and a small error review.

## Consequences

Small inspected images warn against fine-detail claims without a full dimension audit.
Catalog labels may be noisy or visually unsupported, and subcategory may depend on
article type. Missing metadata is not abstention supervision; inspect any such claims
with a small predefined visibility review rather than inventing unknown labels.

Good formatting alone does not establish visual accuracy or unattended catalog
readiness. Preserve invalid predictions in score denominators and evaluate semantic
quality separately. QLoRA may offer modest memory savings and no speed improvement
at this model size; measure rather than promise a benefit. Record resource costs and
negative results, and use affordable final seed confirmation without expanding into
a broad search. Claims remain limited to the measured setup.

No new implementation, configuration framework, service, cloud workload, or research
governance layer is required by this decision proposal.
