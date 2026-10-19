# Experiment plan

**Status (2026-10-19):** detailed protocol proposed under issue #4, aligned with the
direction merged through #13 / PR #14 and the owner-selected four-field scope in #23.
The dataset is owner-selected; the final canonical vocabulary, sample/update budgets,
hyperparameters, acceptance thresholds, and final recipe still require review. This
documentation change claims no new audit results, prepared split, controlled model
evaluation, adapter training, or measured VLM fit.

## Research questions

**Main question:** Does fine-tuning a small Vision-Language Model (VLM) improve the
accuracy and structural reliability of fashion-product attribute extraction compared
with the same unadapted model on held-out images?

The supporting questions are:

1. For the selected model, dataset, and task, how do Low-Rank Adaptation (LoRA) and
   quantized LoRA (QLoRA) compare in prediction quality and resource efficiency under
   a controlled training budget?
2. Can these approaches run practically on the target RTX 5060 Laptop GPU with
   approximately 8 GiB VRAM, and what configuration and resource trade-offs are required?
3. How consistently do adapted models satisfy the required structured-output
   contract, and what semantic errors, invalid outputs, and abstention failures remain?

Fine-tuning here means the selected parameter-efficient methods, not all possible
fine-tuning approaches. Structured catalog ingestion, indexing, search, and data
quality motivate the task; downstream systems are outside scope. This file owns
the detailed questions/protocol; the [overview](../README.md#research-questions)
summarizes them and the [architecture](architecture.md) owns component boundaries.

| Question | Comparison and evidence |
|---|---|
| Main: task improvement | Each adapter versus its matched unadapted base: joint exact match, per-field quality, and structural validity |
| 1: method trade-offs | LoRA versus QLoRA under matched settings: quality, memory, time, throughput, and artifact sizes |
| 2: practical feasibility | Actual model/training runs: fit or failure, peak memory, precision, batch/accumulation, duration, and throughput |
| 3: output reliability | All conditions: parse/schema validity, key coverage, invalid labels, abstention, and a small error review |

The [hardware assessment](architecture.md#target-gpu-and-cuda-compatibility) owns
observed device/runtime facts. Issue #5 owns runtime selection. This plan does not
establish application compatibility or select a CUDA upgrade.

## Scope and proposed budget

Use one dataset, `ashraq/fashion-product-images-small`, and initially one candidate,
`HuggingFaceTB/SmolVLM-500M-Instruct`. Pin its exact revision before evidence-producing
runs. The task contract is owned by [dataset.md](dataset.md#initial-task):
`masterCategory`, `subCategory`, `articleType`, and `baseColour`, with source-label
spelling/capitalization and explicit `unknown` semantics. All four conditions use
this same four-field contract and frozen training-derived vocabularies.

- Target 2,000 training, 250 validation, and 250 held-out test images, subject to
  audit, group isolation, eligibility, and available support.
- Propose roughly 8–12 article types selected from candidate training data only.
  Document retained/excluded populations; do not use held-out class frequencies
  to choose the subset. Freeze final membership before evaluation.
- Propose a 30–50-image visibility/error-review subset of the frozen test partition,
  with its rubric developed on training/validation examples. This is supplementary
  analysis, not a new external-photo domain-shift benchmark or a large annotation task.
- After authorized tiny feasibility checks, propose one pilot per method of 50–100
  optimizer steps, capped at 100. Estimate final cost from measured memory/time.
  Record pilot compute in the total project budget; a pilot is not a final result.
- Propose at most one initial epoch per final adapter run, with a shared final
  optimizer-step cap for both methods. Agree the cap and practical time/memory limits
  after the pilot and before final runs. Final runs
  start from the original checkpoint, not undeclared pilot-trained weights.

Extra models, method-specific searches, external-photo benchmarking, and target-module
ablations are optional future work. A tiny demo follows completed evaluation.

## Model feasibility and execution order

1. Review the environment and generic CPU/CUDA diagnostics under issues #5–#7.
   These tests do not prove VLM or quantization compatibility.
2. Implement the proposed audit, task validator, canonical mapping, and deterministic
   preparation with synthetic tests. Freeze duplicate-aware manifests before prompt
   expansion. No implementation is introduced by this documentation issue.
3. Implement shared loading, processing, answer generation, and scoring; migrate the
   standalone baseline rather than keeping a separate free-form evaluation path.
4. Obtain owner authorization for actual-model checks on a few training examples:
   floating-point and four-bit inference; correct processor/chat template and image
   tokens; actual BF16 (bfloat16) or FP16 (16-bit floating-point) execution; exact
   adapter matches; answer-only masks; finite adapter gradients in a tiny backward
   pass for each method; peak memory; and adapter save/reload behavior.
5. Bound image resizing/crops, text, and generated tokens consistently. Do not truncate
   image tokens. Inspect the actual loss masks: only intended answer tokens contribute,
   excluding prompt, image, and padding tokens. A trainer flag is not verification.
6. Record both bases on development data, then run approved adapter pilots. Low base
   schema validity is an outcome, not by itself a failed compatibility check. Use
   measured pilot costs to request approval for longer runs.
7. Select the common recipe/checkpoint policy on validation, then freeze all settings,
   final seeds, and evaluation rules. Train final adapters only with owner approval.
8. Evaluate the frozen conditions on held-out data, save evidence, and conduct the
   predeclared supplementary review. Do not tune or restart selection from test results.

The [TRL reference](references.md#implementation-guidance-follow-up-2026-10-18)
documents template/version conditions for assistant-only loss and image-token
truncation. Inspect the pinned stack and collator rather than assuming current
main-branch examples work. Dependency changes belong to #5; this issue changes none.
No compatibility, performance improvement, or training success is presumed.

## Four-condition comparison

| Condition | Role |
|---|---|
| Floating-point unadapted base | Original task reference in the verified BF16/FP16 mode |
| Four-bit unadapted base | Quantization-only control |
| LoRA, floating-point base | Adaptation without quantizing the base |
| QLoRA, four-bit base | Adaptation with quantized base weights |

The floating-point base is not full-parameter fine-tuning. Interpret contrasts as:

- LoRA minus floating-point base: adaptation without base quantization.
- QLoRA minus four-bit base: adaptation with base quantization.
- Four-bit minus floating-point base: quantization effect.
- LoRA versus QLoRA: quality/resource trade-off for this setup.

Comparing QLoRA only with the floating-point base combines adaptation and quantization.
Both unadapted models receive the same structured instructions and label vocabulary
as the adapted models, not a weaker unrestricted-description prompt.

### Controls and adapter strategy

Hold fixed the checkpoint revision, manifest, instruction examples, task vocabulary,
processor/image policy, answer masks, validation selection rule, and generation/scoring
settings. Match adapter rank, alpha, dropout, exact targets, initialization policy,
effective batch, optimizer, learning rate/schedule, and optimizer-step budget. Use
the same seed for each paired LoRA/QLoRA run. Report unavoidable differences; label
any later method-specific optimization separately from the controlled comparison.

Start with a frozen vision encoder/projector and language-attention adapters only.
Rank 8, microbatch 1, accumulation 8, and gradient checkpointing remain starting
candidates, not verified settings. Resolve alpha, dropout, optimizer, learning rate,
image/sequence limits, and module paths before pilots. Verify qualified language-only
matches and trainable counts; short attention suffixes can also match vision modules.
An attention-plus-MLP ablation requires a justified, separately approved small budget.

For the four-bit paths, propose NF4 (4-bit NormalFloat), with compute dtype, double
quantization, and module scope explicitly configured and verified. Keep quantization
settings identical between the four-bit base and QLoRA. Use the same verified compute
dtype across methods where supported and record any required precision differences.
Freezing vision/projector weights does not imply that they are unquantized.
Do not silently dequantize or merge
an adapter into a different precision for final scoring. The original method papers
provide rationale, not evidence for this laptop/task; see [references](references.md).

## Evaluation

**Proposed primary outcome:** joint exact match on all four required fields over
every eligible test image. Approve the protocol and practical success criteria before
observing final test results; no arbitrary improvement threshold is asserted here.

Let N be the number of evaluated images. Require one prediction or a recorded failure
for each image. Apply the frozen [output contract](dataset.md#proposed-structured-output-contract)
to the generated answer only. Compare canonical labels exactly; JSON key order and
whitespace do not matter. The core experiment uses no silent repair or model-specific
constrained decoding. Any later such procedure must be a declared, shared secondary
condition. Keep raw generations and validation reasons.

### Structural and attribute quality

| Metric | Proposed definition |
|---|---|
| Parse success | Answers accepted by strict single-object JSON parsing / N |
| Schema validity | Parsed answers satisfying all keys, types, allowed values, and abstention rules / N |
| Required-key coverage | Required keys present in parsed objects / (4 × N); presence alone is not correctness |
| Invalid-label rate | Images with at least one parsed field string that is neither a known label for that field nor `unknown` / N; separately report missing/type/parse failures |
| Per-field accuracy | Correct predictions from schema-valid records / N, reported separately for `masterCategory`, `subCategory`, `articleType`, and `baseColour` |
| Per-field macro precision/recall/F1 | Equal-weight class metrics over each of the four fields' frozen training-derived known-label sets |
| Joint exact match | Schema-valid records with all four fields correct / N |
| Hierarchy consistency | Schema-valid records with all three category fields known and their (`masterCategory`, `subCategory`, `articleType`) triple in the frozen training-derived relation / N; report hierarchy abstentions and invalid records separately |

Required-key coverage counts each required key at most once per strictly parsed
object, regardless of its value or extra keys. A parse failure contributes zero;
missing keys remain in the fixed `4 × N` denominator. Schema validity is separate
and still rejects extra/missing/duplicate keys, wrong types, and invalid labels.

The [dataset hierarchy policy](dataset.md#initial-task) owns the training-derived
category relation. Hierarchy consistency is a separate diagnostic: individually
allowed labels can form an inconsistent triple without making the record
schema-invalid. An `unknown` in any category field leaves the hierarchy unassessed;
an `unknown` in `baseColour` alone does not. Keep all images in N and separately
report known consistent triples, known inconsistent triples, hierarchy abstentions,
and invalid records. Do not repair or fill category fields from predicted parents
or children before scoring.

A schema-invalid record fails every field and joint exact match, even if one field
is recoverable. Never drop it from N. For known catalog targets, `unknown` is not a
correct field answer. Invalid/abstaining predictions contribute false negatives for
the true known classes; keep the full known-label set when computing macro metrics.
Use zero for undefined class precision/recall/F1 and report class supports. Any
valid-output-only quality diagnostic must declare its smaller denominator.

### Abstention and visual support

For each field, classify each image as schema-valid known, schema-valid `unknown`,
or invalid record. Report:

- Coverage = schema-valid non-abstaining predictions / N.
- Abstention rate = schema-valid `unknown` predictions / N.
- Invalid rate = schema-invalid records / N.
- Answered correctness = correct non-abstaining predictions / schema-valid
  non-abstaining predictions; report undefined, not zero, if there are no answers.

Coverage, abstention, and invalid rate sum to one per field; coverage is not simply
one minus abstention. Report ordinary accuracy/joint match alongside answered
correctness so a model cannot look successful just by refusing difficult examples.

Catalog labels alone do not establish which fields are visually answerable. For the
small visibility subset, freeze sample IDs and the rubric before viewing model outputs.
Review each field as supported or unsupported, with a reason, independently of the
predictions. Supplementary unsupported-assertion rate is valid non-abstaining answers
on unsupported fields / all reviewed unsupported fields. Report the counts, invalid
outputs, and abstention on supported fields too. If no unsupported fields exist,
report this rate as undefined. Do not equate a catalog mismatch with hallucination.
Keep the original catalog benchmark unchanged; review labels are supplementary.

### Resource measurements and final confirmation

Record trainable parameters and adapter bytes; peak allocated and reserved GPU memory;
wall-clock training time and optimizer steps/sec or examples/sec; and the saved
deployment/reconstruction footprint, distinguishing shared base weights from adapters.
Include quantization metadata, processor/configuration, and required base revision.
Do not report adapter bytes as the whole deployed model or silently change precision.

Use the same hardware/runtime and measurement boundaries across methods. Reset memory
peaks before each measured phase, synchronize CUDA timing, and separate loading,
warm-up, training, and inference. Record batch/accumulation and image/token limits.
If measuring latency, fix the input set, batch size, decoding and warm-up policy,
report generated lengths, and state whether preprocessing/transfer is included.
Record available VRAM and failures; allocated/reserved tensor memory is not a complete
measure of all GPU processes. Feasibility is a measured result within an owner-agreed
time/memory budget, not merely successful model loading.

Explore with one seed. If affordable, use three matched final seeds per adapter
method after freezing the common recipe: six adapter runs on the same split. Evaluate
all selected final runs and report individual scores plus mean and sample standard
deviation, not the best seed. Deterministic base evaluation needs no repeated training
seeds. A single-seed result must disclose that limitation. Training-seed variation
does not measure dataset uncertainty or establish statistical significance.

### Small error review and interpretation

Use the fixed review subset and a small declared development-error sample. Record
colour confusion, confusion at each catalog-category level, hierarchy violations,
ambiguous or low-resolution images, catalog disagreement, invalid output, and unsupported claims.
Allow multiple reasons and report how cases were selected. Do not turn this into
a large annotation project or a post-test relabelling/tuning exercise.

Report mixed or negative outcomes. Better formatting with unchanged attribute quality
supports a narrower conclusion than better extraction. With 250 test images, five
predictions change a rate by two percentage points; show supports and avoid overstating
small differences. Public-data pretraining exposure may be unknown. Conclusions cover
this model, eligible catalog population, recipe, and laptop, not unseen fashion domains
or a universally superior fine-tuning method. A conventional classifier remains a
reasonable alternative for a fixed label set; this study investigates VLM adaptation.

## Configuration and reproducibility

Propose one YAML experiment specification validated into typed configuration, not a
configuration framework. Require dataset revision/snapshot and manifest/vocabulary
IDs; model revision; task/prompt versions; method and exact adapter targets; precision
and quantization policy; seed; training/update settings; evaluation and measurement
settings; and output location. Reject missing/inconsistent required settings and
save the resolved configuration. Declare any YAML parser directly when implemented.

Each run preserves local configuration, raw predictions with sample IDs and failure
reasons, quality/resource metrics, environment/package builds, device details, code
revision/dirty state, data/model revisions, seeds, and references/hashes for immutable
manifests, vocabulary, and audit decisions. Reusing a shared immutable manifest/base
is sufficient; do not duplicate them into every run directory. Generated predictions,
raw/derived data, checkpoints, and adapters stay ignored. Small intentional aggregate
reports may enter Git after owner review.

## Decisions before execution

| Item | Current state |
|---|---|
| Dataset | Selected; historical local inventory in `dataset.md` |
| Provenance/audit | Exact download revision, checksums, row/image validation pending |
| Contract/vocabulary | Four-field scope selected under #23; contract documented in `dataset.md`; final canonical values and review rubric unresolved |
| Splits/budget | Candidate sizes only; no validated manifest or selection implemented |
| Model/runtime | SmolVLM candidate; exact revision, pinned-stack support, fit and practical limits unverified; runtime choice belongs to #5 |
| Training | Candidate adapters only; essential settings and final seeds/budget pending |
| Execution | Every first command follows `AGENTS.md`; training/download/publication require explicit authorization |

This documentation does not create configuration/artifact files, approve runtime
changes, or authorize preparation, model execution, tests, or training. Implementation
and execution remain separate owner-approved work.
