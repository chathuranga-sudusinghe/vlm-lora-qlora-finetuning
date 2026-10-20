# Dataset and integrity policy

**Status (2026-10-20):** source inspection (#22), the four-field scope (#23), and
canonical merge (#28 / PR #29) are completed. The saved execution evidence below
remains historical. Issue #30 documents the owner's Test-first preparation policy
in [DR-003](decisions/DR-003-sealed-test-development-policy.md). Test sealing,
Development-only auditing, final vocabularies, and Train/Validation preparation
are not implemented or executed by this documentation change.
The [experiment plan](experiments.md) owns run budgets and evaluation.

## Selected source

**Fashion Product Images Small**, Hugging Face repository
[`ashraq/fashion-product-images-small`](https://huggingface.co/datasets/ashraq/fashion-product-images-small),
is selected for a focused product-attribute project.

Read-only Dataset Viewer inspection on 2026-10-17 reported:

- one `train` split containing 44,072 records;
- 271,496,441 bytes of source Parquet files;
- image, product identifier, category, article type, color, usage, and other metadata;
- the three inspected records had 60 by 80 pixel images.

These are snapshot observations, not a guarantee about every image or future revisions.
The dataset does not provide an independent validation/test split. Preparation must
establish artifact provenance and create a reproducible split manifest before training.

## Local download inventory

Read-only filesystem inspection on 2026-10-17 found these files under `data/raw/`:

| Dataset shard | Local size in bytes |
|---|---:|
| `train-00000-of-00002-6cff4c59f91661c3.parquet` | 136,091,680 |
| `train-00001-of-00002-bb459e5ac5f01e71.parquet` | 135,404,761 |
| **Total** | **271,496,441** |

The total matches `download_size` in the downloaded `data/raw/README.md`, which also
declares 44,072 examples and one source `train` split. This is agreement between
file sizes and source metadata, not a row-count or integrity result from this work.
The completed [overview notebook](../notebooks/01_dataset_overview.ipynb) inspects
raw schemas, full counts, IDs, row previews, and sampled images under #22. Its cleared
outputs do not supply new numeric evidence here. No real data was read during #28's
implementation or synthetic checks; the later owner-run evidence is recorded below.

`data/raw/README.md` and the downloaded `data/raw/gitattributes` are supporting
metadata, not extra training examples. No standalone `sample.jpg` is required.
Both Parquet files are ignored by the committed ignore rules. Never stage `data/raw/`.

**Upstream revision remains unresolved.** The owner-run merge records checksums for
both shards and identifies a local snapshot, as recorded below. A filename suffix,
matching byte count, or the Hub's current `main` revision does not prove the revision
of an earlier download. Preserve the local snapshot identity until an upstream
revision can be established; do not invent one.

## Canonical merged artifact

[Issue #28](https://github.com/chathuranga-sudusinghe/vlm-lora-qlora-finetuning/issues/28)
is implemented in
[`02_create_canonical_dataset.ipynb`](../notebooks/02_create_canonical_dataset.ipynb).
Here, **canonical** means one preserved source artifact, not canonicalized labels
or a training-ready dataset. The notebook uses the existing repository `.venv` and
supports working directories at the repository root or `notebooks/`.

The notebook concatenates the two filenames listed above in that exact order,
retaining every row and all 11 original columns in the first shard's column order.
There is no join on `id`, filtering, deduplication, label normalization, label repair,
or conversion to the four prediction fields. Embedded image bytes and raw metadata
remain intact. Both raw shards stay unchanged as provenance and rebuild inputs.

An authorized run creates these ignored local artifacts:

- `data/interim/fashion_products_merged.parquet`: input for subsequent audit and
  separately approved preparation.
- `data/interim/fashion_products_merged.manifest.json`: ordered source paths,
  byte sizes, row counts, SHA-256 checksums, schema including Arrow metadata, output
  checksum/count, runtime versions, and validation results. An unresolved upstream
  revision is explicitly identified by a checksum-based local snapshot ID.

Before writing, the notebook checks both schemas and refuses to overwrite either
existing artifact. It streams batches to Parquet and rereads the result to compare
every record in order, including image payloads. It checks the full count against
the sum of the input counts and verifies unchanged raw checksums. Ordinary failures
remove only incomplete artifacts created by that invocation. A process interruption
can leave partial output; inspect it before deciding how to recover. The manifest
is created last, after merge validation succeeds.

On rerun, a wrapper catches `FileExistsError` and reuses a completed artifact pair
without overwriting it. Both files must exist, the manifest must identify the
expected output, and the merged checksum must match. The remaining inspection
cells validate schema, full count, and IDs against that manifest. Incomplete pairs,
invalid manifests, and checksum failures stop execution without changing existing
files. Reuse preserves the original snapshot provenance; it does not rescan current
raw shards or rebuild the merged dataset.

Missing and repeated IDs are explicit quality failures in the manifest and notebook
output; they do not cause records to be removed or repaired. Successful record
preservation does not imply clean IDs, unique images, or training eligibility.

The inspection cells read the **saved merged file** and display its column names
and types, column count, full scanned row count, per-column missingness, ID checks,
first 100 raw rows, and 10 distinct sampled row positions with images and all original
metadata. The configurable fixed seed defaults to 42. Missingness counts top-level
Arrow nulls and numeric NaNs; it does not reclassify empty strings or nested fields.
Long table representations are shortened without changing stored values. Only the
10 selected images are decoded; this is not an all-image decode or duplicate audit.
The manifest records the sample seed and exact sampled row positions/IDs but leaves
visual inspection pending;
running display code alone does not establish owner review.

The notebook includes bounded synthetic checks for preservation, ID defects,
missingness, sampling, schema/count/record failures, overwrite protection, and
failure cleanup. These do not establish the real snapshot's row counts or quality.
Real-data execution requires exact commands, impact explanation, and explicit owner
authorization under [AGENTS.md](../AGENTS.md). Clear notebook outputs and execution
counts before commit; keep datasets and generated manifests out of Git.

Issue #28 did not select split ratios or create partitions. The later owner-selected
[preparation policy](#preparation-and-splitting-rules) seals Test membership first,
then restricts detailed audits and task-policy decisions to Development. The
inspection seed 42 above is not an agreed split seed. After sealing, do not rerun
the full-source overview or merge/inspection workflows: they expose Test content.
Their Markdown warnings are documentation; runtime access guards remain future work.

### Saved execution evidence — 2026-10-19

The owner executed the notebook in the existing environment. The coding agent
reviewed its saved outputs, including the rendered table and ten image/metadata
previews, from an ignored local backup before clearing the tracked notebook.
These are observations from that saved run, not an independent rerun or verification
of the artifacts' current on-disk contents. The formatting correction after that
run preserved all eight code cells' Python abstract syntax trees. The later
owner-requested create-or-reuse behavior changes execution flow and has bounded
synthetic coverage. The owner's later saved rerun also reports skipping the merge,
a matching output checksum, and successful inspection of 44,072 rows, 11 columns,
and ten rendered images with no error outputs. This establishes saved execution
evidence for reuse as well as creation. Reuse does not recompute raw-shard checksums;
the original source-integrity results below belong to the creation run.

| Check | Saved result |
|---|---|
| First / second source rows | 22,036 / 22,036 |
| Saved merged rows | 44,072; equals the combined source count |
| Schema | All 11 original columns, original order/types, and schema metadata preserved |
| Record preservation | All 44,072 records compared in order, including embedded image bytes |
| Raw source integrity | Both source SHA-256 checksums unchanged during the run |
| IDs | 44,072 unique non-null IDs; no missing or repeated IDs |
| Missingness | Zero top-level nulls/numeric NaNs in each of the 11 columns |
| Raw preview | 100 rows with all 11 original column headers |
| Image sample | Ten distinct saved row positions, seed 42; all ten previews rendered with raw metadata |
| Merged Parquet size | 271,346,014 bytes |

The schema is `id: int64`, `year: double`, `image: struct<bytes: binary, path: string>`,
and eight string columns. The local snapshot identifier is
`sha256:35424e03d1289d9c9e56d4fd0796171075c91cf1e7258e94596de804d1392587`.
The saved manifest records both source checksums and the output checksum. Its
inspection status remains a pending marker by design; this evidence record describes
the later saved-output review without rewriting that generated manifest.

After the reuse and exception-chaining changes, the owner supplied passing results
for `.venv/bin/ruff check notebooks/02_create_canonical_dataset.ipynb` and
`.venv/bin/ruff format --check notebooks/02_create_canonical_dataset.ipynb`.
The prior configured MyPy result (eight source files) remains applicable because
its source/configuration scope is unchanged. Bounded synthetic top-to-bottom
creation and rerun checks passed from both supported working directories, including
unchanged artifact bytes/timestamps, preview displays, and failure cases.
No model run, all-image decode audit,
duplicate-image audit, label repair, or split preparation is established by this
evidence. Unique IDs and zero reported missingness do not prove unique images or
visually correct labels. Owner acceptance remains separate from these checks.

## Initial task

Extract four attributes from a single product image and task instructions, using
the exact source column names as JSON keys. The owner-selected scope is recorded in
[issue #23](https://github.com/chathuranga-sudusinghe/vlm-lora-qlora-finetuning/issues/23)
and summarized in the [project overview](../README.md):

| Output key | Source field | Role |
|---|---|---|
| `masterCategory` | `masterCategory` | Broad catalog category |
| `subCategory` | `subCategory` | Intermediate catalog category |
| `articleType` | `articleType` | Fine product type within the selected task scope |
| `baseColour` | `baseColour` | Main catalog colour, subject to visual ambiguity |

Only these four fields are output targets. `gender`, `season`, `year`, `usage`, and
`productDisplayName` remain outside the output contract. Inputs must not contain
product names, labels, IDs that encode answers, or other target metadata. Keep source
identifiers only for provenance and grouping.

The hierarchy runs from `masterCategory` to `subCategory` to `articleType`, for
example `Apparel` / `Topwear` / `Shirts`. Derive the allowed category triples from
Development, restricted by its versioned eligibility rules; audit
whether article types determine their parent categories rather than assuming so.
Keep all four requested predictions but report category dependence and hierarchy
consistency. Do not describe these related category levels as independent visual
capabilities or silently derive one predicted field from another in the main evaluation.

### Proposed structured-output contract

Require exactly one JSON object containing exactly these four keys, each with a
string value. This example illustrates the shape, not a finalized vocabulary:

```json
{
  "masterCategory": "Apparel",
  "subCategory": "Topwear",
  "articleType": "Shirts",
  "baseColour": "Blue"
}
```

Each value must belong to that field's frozen canonical vocabulary or be the exact
reserved string `unknown`. Key order and JSON whitespace do not affect correctness.
Extra/missing/duplicate keys, non-string values, unrecognized labels, Markdown fences,
and prose outside the object fail the proposed contract. Preserve the raw generated
answer; do not silently repair it or infer missing fields. Score generated answer
tokens rather than the echoed prompt. Preserve any separately declared decision-policy
output alongside the raw answer and its validation result, never in place of them.
[experiments.md](experiments.md#evaluation) defines their separate metrics.

### Canonical vocabulary and abstention

Retain raw labels alongside canonical targets in local preparation artifacts.
Canonical known-label strings retain source spelling and capitalization, as in
`Apparel`, `Topwear`, `Shirts`, and `Blue`. Propose trimming surrounding whitespace
from source labels only; do not case-fold or lowercase the target vocabulary.
Any additional alias mapping requires explicit recording and review. Do not assume
spelling, capitalization, or punctuation changes preserve meaning. Select allowed
labels for all four fields, aliases, supported article types, eligibility rules, and
category triples using Development only. Version these policies during development
and freeze their versions/hashes before final Test access. Apply the frozen rules
without Test-driven changes. These categorical policy decisions differ from learned
preprocessing statistics, which must be fitted on Train only.
Reserve the exact lowercase string `unknown` separately from known classes and check
for label collisions. Dataset-label normalization does not repair generated labels;
predictions must use the exact canonical strings required by the contract, including
capitalization and without silently trimming generated string values.

The planned source-code inference/decision layer must explicitly define when to
return field-level `unknown` or reject a record, including insufficient visual
evidence and detected unsupported/out-of-scope input. Keep reasons in sidecar
records, not additional JSON keys. A new product from a supported class is still
in scope; a wrong known-label prediction is a prediction error, not automatically
an unsupported-input event. A vocabulary-membership check cannot detect an
unsupported product that the model assigns a valid known label. A rejection method
requires Development-only design and validation, with thresholds selected on
Validation; no reliable detector or calibrated confidence is established here.

Distinguish visual insufficiency, outside-project class support, missing reference
metadata, and malformed generation. Missing metadata is not evidence for `unknown`.
Apply the frozen reference-eligibility rule at final evaluation and account for every
Test member, including unsupported and unscorable records; do not delete them to
improve scores. An invalid raw answer remains structurally invalid even if the
decision layer emits a safe fallback. Rejection and raw/final scoring are specified
in [the evaluation policy](experiments.md#decision-policy-and-unsupported-input-evaluation).

Catalog metadata does not supply independent visibility annotations. Derive ordinary
training answers from eligible catalog labels after splitting, but do not fabricate
unknown training targets. If the owner adds a small visibility-labelled training/dev
subset, preserve its provenance, group isolation, and place in the budget. Otherwise
describe abstention as prompt-supported behavior rather than demonstrated learned
or calibrated uncertainty. A separate small held-out review can assess visual support
under the [evaluation protocol](experiments.md#evaluation).

Catalog labels are imperfect image-grounded annotations. Review missing labels,
ambiguous color names, class frequencies, and image/label disagreement within
Development before fixing the mapping and class subset. Test never guides those choices.

Do not train the first version to guess brand, season, year, intended gender,
material composition, or occasion from these low-resolution images. Fine texture,
small logos, and subtle material details are outside the reliable initial scope.

## Development-only audit

After Test is sealed, the future audit uses only Development records. Produce one
local report plus machine-readable record-level decisions, not an additional data
platform. Source checksums identify provenance without inspecting Test labels/images.

| Area | Required evidence |
|---|---|
| Source | Dataset ID, verified revision or unresolved-revision local snapshot ID, shard sizes/checksums |
| Structure | Actual row count, field names/types, missing IDs and required labels |
| Images | Decode failures, dimensions/aspect ratios, and exact-duplicate hashes |
| Grouping | Product/duplicate membership, near-duplicate/variant policy, conflicting labels |
| Labels | Raw/canonical values, missingness, class support, hierarchy inconsistencies |
| Selection | Retained/excluded counts, record IDs, reasons, and covered population |
| Reproduction | Audit/preparation configuration, seed, mapping and manifest identifiers |

All image, label, frequency, conflict, and grouping investigations in this table are
Development-only. Unique IDs identify records; assigning every unique ID its own
group provides no image-duplicate protection. Discover meaningful exact-duplicate,
related-product, variant, and near-duplicate groups where evidence supports them.
Define conflict/exclusion rules before evaluating models. Do not inspect Test to
complete this audit or edit catalog labels after inspecting final predictions.

## Preparation and splitting rules

The owner-selected sequence is Test first, then Development policy design, then
Train/Validation. It uses a random same-catalog holdout, not chronological or
newest-data splitting. The immutable canonical source retains all 44,072 rows and
all original values; exclusions apply only to derived development views.
Test members are never removed; final evaluation assigns frozen reference statuses
under [the evaluation policy](experiments.md#evaluation).

### Test membership and logical sealing

1. Verify canonical snapshot/output identity and the record-ID/count prerequisites
   needed for reproducible selection. This permits integrity checks, not a new
   global image/label audit. Do not filter, normalize, or repair the source first.
2. Select approximately 10% of source records as Test using a fixed-seed random
   holdout, leaving approximately 90% as Development. Fix the seed before selection;
   do not try alternative seeds to improve class coverage or observed balance.
3. Save immutable Test membership metadata rather than a separate physical Test
   Parquet. Bind exact IDs and canonical row references to the source snapshot and
   output checksum, ordered input-ID identity, split algorithm and implementation
   version, random generator/version, seed, fraction, rounding rule, actual counts,
   and deterministic membership manifest/hash. Seed alone is insufficient. Record
   the Development complement and verify disjoint, exhaustive, unique membership.
   Any physical Test copy requires a documented technical reason and owner agreement.
4. Seal this membership immediately. Later sessions load and verify it, rather than
   regenerate it. A changed source, identity mismatch, or conflicting existing seal
   must stop execution. Changing task policy never changes Test membership.

Logical sealing is a workflow restriction, not encryption of the local canonical
file. Development loaders must select the Development complement before exposing
records to audit, preview, sampling, or training code. Keep Test images/metadata,
class frequencies, plots, and content-derived diagnostics unavailable to those
workflows. No Test-driven rebalancing, repairs, exclusions, or policy decisions are
allowed after sealing. A manifest integrity check need not expose Test content.
The exact algorithm, seed, serialization, rounding rule, and access-guard design
remain unresolved implementation choices requiring owner agreement. No membership
manifest, sealed partition, or enforcement code is created by issue #30.

### Development policies and Train/Validation

Use the [Development-only audit](#development-only-audit) to define and version class
support, vocabularies, normalization, eligibility, hierarchy, duplicate/conflict
handling, and preprocessing policies. Unsupported classes remain identifiable for
rejection evaluation rather than being silently converted to known classes.

Split Development into approximately **8/9 Train and 1/9 Validation**, targeting
approximately 80%/10% of the original source before exclusions and group constraints.
Use `articleType` as the main stratification variable and monitor the other three
target distributions. Do not force sparse joint stratification across all four
fields. Keep every discovered meaningful Development duplicate/related-image group
wholly in Train or Validation, before generating multiple instruction examples.

Agree rare-class, conflicting-label group, and imperfect-balance handling from
Development before accepting a split. Group isolation takes priority over exact
ratios; do not split a duplicate group to improve balance. Record unresolved cases
for owner decision rather than silently disabling grouping or dropping classes.
Save exact memberships, group assignments, policy versions/hashes, rounding,
actual counts, distributions, exclusions with IDs/reasons, and reproducibility checks.
Any policy revision must be versioned and its downstream development artifacts
revalidated while Test membership remains fixed.

Fit learned preprocessing statistics on Train only. Fit/fine-tune weights with
Train; use Validation for prompts, thresholds, hyperparameters, checkpoints, and
model/configuration selection. Because categorical task policies can use the whole
Development pool, Validation is a development selection set, not an independent
assessment of the entire policy-design process. Experiment sample/update budgets
are separate from full partition membership; [experiments.md](experiments.md) owns them.

### Final access and interpretation limits

Only after the [complete system freeze](experiments.md#final-freeze-and-test-access)
may an explicitly authorized final evaluation recover the exact sealed Test records
from the unchanged canonical source. Apply frozen rules, account for every member,
and preserve failures. Final outcomes do not reopen development or membership selection.

The source and sampled images were inspected before this policy was adopted, as
recorded above. The seal is prospective; this is not a claim of never-inspected data.
Random Test selection before Development-only duplicate auditing does **not** prove
duplicate-free Development/Test boundaries or evaluation on wholly unseen products.
Do not silently add a global Test audit to claim that guarantee. This is a
pseudo-future same-catalog holdout: it does not establish future-domain robustness,
chronological generalization, or absence of public-model pretraining exposure.

## Provenance and use terms

The Hugging Face card points to the
[upstream Kaggle dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small).
The upstream listing was previously observed to declare MIT; the Hugging Face card
itself provides limited licensing and provenance detail. This does not independently
establish rights to every underlying image. Preserve attribution and the applicable
upstream notice, and verify terms for the exact revision before redistributing
images or prepared data.

Repository `LICENSE` covers project code/documentation. It does not relicense source
images, model weights, or adapters. Dataset downloads and derived records remain
local; no dataset image needs to be committed for the project to be reproducible.

See [references](references.md) for source verification limits.
