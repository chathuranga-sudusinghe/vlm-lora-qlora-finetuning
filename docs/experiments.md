# Experiment plan

**Status (2026-10-20):** the four-field task and four-condition comparison are retained.
Issue #30 documents the owner's Test-first preparation and sealed evaluation policy
under [DR-003](decisions/DR-003-sealed-test-development-policy.md), superseding the
earlier preparation assumptions. Final vocabularies, sample/update budgets,
hyperparameters, rejection method/thresholds, and practical acceptance criteria still
require review. No seal, prepared split, model evaluation, training, or new real-data
audit is implemented or executed by this documentation change.

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
this same four-field contract and frozen Development-derived vocabularies.

- Partition membership follows the [Test-first policy](dataset.md#preparation-and-splitting-rules):
  approximately 10% Test first, then approximately 80% Train and 10% Validation of
  the original source through a Development-only split. These are membership
  targets, not compute budgets; actual counts depend on rounding, exclusions, and groups.
- The earlier 250-image Test budget is superseded. Evaluate every sealed Test member
  through the final accounting protocol; do not resample Test for convenience.
  Estimate full-Test inference time from Development throughput measurements and
  the planned number of final runs before requesting final-evaluation authorization.
  The earlier 2,000 Train / 250 Validation figures remain possible compute-subset
  proposals within those partitions, subject to owner review. Declare exact subset
  IDs, selection rules, group isolation, and update budgets before use.
- Roughly 8–12 article types remain a proposal, selected using Development only.
  Document covered populations and exclusions without consulting Test frequencies.
- Propose a 30–50-image visibility/error-review subset of the frozen test partition,
  with its rubric and sampling rule developed on Train/Validation and frozen before
  Test access. Draw and inspect it only during final evaluation. This is supplementary
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
2. Under a separate approved implementation issue, implement and synthetically test
   Test membership selection/sealing, Development-only access, audit/policy design,
   and grouped, mainly `articleType`-stratified Train/Validation preparation in that
   order. Freeze their versioned artifacts before prompt expansion. No preparation
   code or notebook execution guard is introduced by this documentation issue.
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

Use one declared, versioned source-code decision policy across all four conditions,
including field-level `unknown`/rejection behavior. Design and validate it on
Development only. Preserve raw generations and raw validation failures separately
from final policy outputs; do not make a base look structurally valid through fallback.

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

**Proposed primary outcome:** raw joint exact match on all four required fields over
the in-scope, reference-scorable Test population. Report unsupported-input rejection
and final decision-policy quality separately. Approve practical success criteria
before observing final Test results; no arbitrary improvement threshold is asserted.

Let `N_all` be every sealed Test member. At final access, apply the frozen reference
rules to account for in-scope/scorable, out-of-scope/scorable, and reference-unscorable
records with counts, IDs, and reasons. These mutually exclusive whole-record statuses
sum to `N_all`; missing required reference labels make a record unscorable, with any
known out-of-scope fields also recorded. An input decode or inference failure does
not remove a record with sufficient reference labels from its scoring population.
Let **N** be the fully in-scope, reference-scorable population, shared by all methods.
Require one prediction or recorded failure per Test member, never only per success.
Report undefined scores with their zero denominators when a population is empty.

Apply the frozen [output contract](dataset.md#proposed-structured-output-contract)
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
| Per-field macro precision/recall/F1 | Equal-weight class metrics over each of the four fields' frozen Development-derived known-label sets |
| Joint exact match | Schema-valid records with all four fields correct / N |
| Hierarchy consistency | Schema-valid records with all three category fields known and their (`masterCategory`, `subCategory`, `articleType`) triple in the frozen Development-derived relation / N; report hierarchy abstentions and invalid records separately |

Required-key coverage counts each required key at most once per strictly parsed
object, regardless of its value or extra keys. A parse failure contributes zero;
missing keys remain in the fixed `4 × N` denominator. Schema validity is separate
and still rejects extra/missing/duplicate keys, wrong types, and invalid labels.

The [dataset hierarchy policy](dataset.md#initial-task) owns the Development-derived
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
the true known classes. When N > 0, compute macro precision/recall/F1 over the full
frozen known-label set, assigning zero to undefined per-class ratios by a declared
zero-division convention. Report every class's reference support, including zero.
A class with no reference examples has mathematically undefined recall; its assigned
zero is an averaging convention, not measured evidence of recall on that class.
Its F1 is correctly zero if false positives exist. Keep these classes in the fixed
macro denominator rather than silently changing the metric after Test access.
When N = 0, report the population's scores as undefined, not an average of assigned
class zeros. Any valid-output-only quality diagnostic must declare its smaller denominator.

### Abstention and visual support

Within N, for each field, classify each image as schema-valid known, schema-valid
`unknown`, or invalid record. Report:

- Coverage = schema-valid non-abstaining predictions / N.
- Abstention rate = schema-valid `unknown` predictions / N.
- Invalid rate = schema-invalid records / N.
- Answered correctness = correct non-abstaining predictions / schema-valid
  non-abstaining predictions; report undefined, not zero, if there are no answers.

Coverage, abstention, and invalid rate sum to one per field; coverage is not simply
one minus abstention. Report ordinary accuracy/joint match alongside answered
correctness so a model cannot look successful just by refusing difficult examples.

Catalog labels alone do not establish which fields are visually answerable. For the
small visibility subset, use the pre-frozen sampling rule and rubric at final access,
save exact sample IDs, and review before viewing model outputs. Review each field as
visually supported or visually unsupported, with a reason, independently of the
predictions. Supplementary unsupported-assertion rate is valid non-abstaining answers
on unsupported fields / all reviewed unsupported fields. Report the counts, invalid
outputs, and abstention on supported fields too. If no unsupported fields exist,
report this rate as undefined. Do not equate a catalog mismatch with hallucination.
Keep the original catalog benchmark unchanged; review labels are supplementary.

### Decision-policy and unsupported-input evaluation

The [dataset contract](dataset.md#canonical-vocabulary-and-abstention) owns the
meaning of known labels and `unknown`. A generative model can confidently assign
a known label to an unsupported product; vocabulary validation alone is not a
rejection method. Select the actual source-code decision mechanism and any thresholds
using Development examples, including unsupported-input examples where available.
Do not claim unsupported-input detection without relevant validation evidence.

Development and Test come from the same catalog, so unsupported classes may occur
in both. No class-disjoint or external-domain rejection challenge is defined.
These scores measure rejection within this catalog; they do not establish detection
of classes unseen during Development or robustness to novel/open-set inputs.

Keep raw text, parse/schema results, decision-policy version, final four-field
output or record rejection, and field/record reasons for each ID. Final outputs
retain exactly the four keys when an object is returned; rejection reasons belong
in sidecars. Do not count malformed raw outputs converted to `unknown` as raw schema
successes or as evidence of detecting an unsupported class.

Alongside the raw metrics above, report:

| Population and denominator | Final decision-policy evidence |
|---|---|
| N: fully in-scope, reference-scorable records | Final per-field accuracy and joint exact match, with rejection, `unknown`, invalid output, and execution failure counted as errors for known targets; report false record-rejection rate / N |
| `N_in,f`: Test records with a supported, scorable reference for field f | Field accuracy and false abstention/rejection rate; include supported fields of otherwise out-of-scope records, keeping the denominator explicit |
| `N_out,f`: Test records with a scorable reference outside the frozen support for field f | Correct unsupported-field rejection / `N_out,f`; known-label false acceptance / `N_out,f`; separately report invalid outputs and execution failures |
| All reference-scorable out-of-scope records | Record-level rejection rate, with its count and denominator; declare how field support determines whole-record scope before Test access |
| `N_all`: every sealed member | Counts of all reference statuses, decode/inference failures, raw parse/schema failures, policy outputs/rejections, and unscorable reasons |

For `N_in,f`, `unknown`, record rejection, malformed output, and execution failure
are not correct known-label predictions; report invalid/failure rates separately
from explicit false abstentions/rejections. For `N_out,f`, only a frozen policy's
explicit unsupported-field or unsupported-record rejection counts as correct
rejection; generic parse/decode/inference failures or visual-uncertainty fallbacks
do not earn detection credit. Define the sidecar reason precedence and field-to-record
rejection rule during Development. Field-level populations may overlap across fields;
do not sum them as if they were disjoint whole-record counts.

Always state supports and denominators. If no unsupported reference examples exist
in a population, rejection performance is undefined there; do not reseed Test to
obtain examples. Out-of-scope records stay in the sealed accounting even though
they are outside the known-class joint-accuracy denominator. Never silently drop
them or present in-scope accuracy as accuracy across every Test member.

### Final freeze and Test access

Before opening Test, freeze all final weights/checkpoints, model revision, learned
preprocessing, class eligibility, vocabularies/normalization, hierarchy rules,
prompts, generation settings, output-schema validation, source-code decision policy,
`unknown`/rejection logic and thresholds, metrics/denominators, final run seeds,
and review sampling/rubric. Save versions/hashes and bind them to the fixed membership
and code revision. Training seeds are distinct from the immutable split seed.

Only an explicitly owner-authorized final command may recover the exact sealed IDs
from the canonical artifact. Apply frozen reference/scoring rules without Test-driven
repair or selection. Save predictions and failures for all members. Test results,
including the small review, must not feed prompt/threshold changes, retraining,
checkpoint selection, or a replacement split. Report limitations and negative results.

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
supports a narrower conclusion than better extraction. Show actual population sizes
and class supports rather than assuming the former 250-image Test budget; avoid
overstating small differences. The [holdout limitations](dataset.md#final-access-and-interpretation-limits)
include prior source inspection and unverified cross-boundary duplicates. Public-data
pretraining exposure may be unknown. Conclusions cover
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
| Provenance/audit | #28 records local snapshot checksums and merge validation; upstream revision unresolved; Development image/duplicate/label audit pending |
| Contract/vocabulary | Four-field scope retained; Development-derived policies, final canonical values, decision mechanism, and review rubric unresolved |
| Splits/budget | Test-first policy selected under #30; algorithm/seed/rounding unresolved; no seal or split implementation; compute budgets remain proposals |
| Model/runtime | SmolVLM candidate; exact revision, pinned-stack support, fit and practical limits unverified; runtime choice belongs to #5 |
| Training | Candidate adapters only; essential settings and final seeds/budget pending |
| Execution | Every first command follows `AGENTS.md`; training/download/publication require explicit authorization |

This documentation does not create configuration/artifact files, approve runtime
changes, or authorize preparation, model execution, tests, or training. Implementation
and execution remain separate owner-approved work.
