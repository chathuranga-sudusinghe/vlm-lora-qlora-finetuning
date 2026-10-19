# Dataset and integrity policy

**Status (2026-10-19):** source inspection (#22) and the four-field contract (#23)
are completed. Issue #28 adds the canonical merge and saved-artifact inspection
notebook, checked with synthetic records. The owner's saved execution outputs were
reviewed on 2026-10-19: merge validation passed, and the saved table/image previews
were inspected without rerunning real data. The evidence and its limits are recorded
below. Owner acceptance, the full audit, final canonical vocabulary, exclusions,
and split preparation remain future work.
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

After merged-data inspection, agree a separate preparation scope covering product
and image-duplicate grouping, near duplicates/variants, conflicting-label and
unreadable-image policy, reproducible group-disjoint partitions, training-only
vocabulary/eligibility decisions, and held-out test protection. No split ratios or
partitions are selected by #28. The proposed rules below apply to that later work.

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
the candidate training partition, restricted by its frozen eligibility rules; audit
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
tokens rather than the echoed prompt. [experiments.md](experiments.md#evaluation)
defines how invalid records affect metrics.

### Canonical vocabulary and abstention

Retain raw labels alongside canonical targets in local preparation artifacts.
Canonical known-label strings retain source spelling and capitalization, as in
`Apparel`, `Topwear`, `Shirts`, and `Blue`. Propose trimming surrounding whitespace
from source labels only; do not case-fold or lowercase the target vocabulary.
Any additional alias mapping requires explicit recording and review. Do not assume
spelling, capitalization, or punctuation changes preserve meaning. Derive allowed
labels for all four fields, aliases, class selection, eligibility rules, and category
triples from the candidate training partition only, then freeze their versions/hashes
before evaluation. Apply the same frozen rules to every split.
Reserve the exact lowercase string `unknown` separately from known classes and check
for label collisions. Dataset-label normalization does not repair generated labels;
predictions must use the exact canonical strings required by the contract, including
capitalization and without silently trimming generated string values.

`unknown` means insufficient visual evidence for that field. It is a valid abstention,
not a missing key, malformed answer, missing source label, or out-of-vocabulary class.
Exclude missing required metadata using a declared rule; handle out-of-scope classes
with the frozen eligibility policy and report coverage/exclusions. Do not relabel
either case as visually unknown merely to retain the record.

Catalog metadata does not supply independent visibility annotations. Derive ordinary
training answers from eligible catalog labels after splitting, but do not fabricate
unknown training targets. If the owner adds a small visibility-labelled training/dev
subset, preserve its provenance, group isolation, and place in the budget. Otherwise
describe abstention as prompt-supported behavior rather than demonstrated learned
or calibrated uncertainty. A separate small held-out review can assess visual support
under the [evaluation protocol](experiments.md#evaluation).

Catalog labels are imperfect image-grounded annotations. Review missing labels,
ambiguous color names, class frequencies, and image/label disagreement before fixing
the mapping and class subset. Freeze mappings before test evaluation.

Do not train the first version to guess brand, season, year, intended gender,
material composition, or occasion from these low-resolution images. Fine texture,
small logos, and subtle material details are outside the reliable initial scope.

## Proposed dataset audit

The future audit should produce one local report plus machine-readable record-level
decisions, not an additional data platform:

| Area | Required evidence |
|---|---|
| Source | Dataset ID, verified revision or unresolved-revision local snapshot ID, shard sizes/checksums |
| Structure | Actual row count, field names/types, missing IDs and required labels |
| Images | Decode failures, dimensions/aspect ratios, and exact-duplicate hashes |
| Grouping | Product/duplicate membership, near-duplicate/variant policy, conflicting labels |
| Labels | Raw/canonical values, missingness, class support, hierarchy inconsistencies |
| Selection | Retained/excluded counts, record IDs, reasons, and covered population |
| Reproduction | Audit/preparation configuration, seed, mapping and manifest identifiers |

Structural inspection and grouping cover the source before partitioning. Keep global
audit summaries descriptive; task vocabulary and class-selection choices use candidate
training data only. Define conflict/exclusion rules before model evaluation. Do not
drop test errors or edit catalog labels after inspecting model predictions.

## Preparation and splitting rules

1. Establish the local snapshot's provenance and record checksums before preparation.
   Validate declared versus actual schema, row counts, image decoding, and required
   labels under an owner-approved preparation/validation command. Save audit counts
   and record-level decisions, distinguishing source facts from planned checks.
2. Define structural exclusions before evaluating models. Reject unreadable images
   and examples missing required labels; record the exclusion reasons and counts.
3. Identify exact image duplicates and product identifiers before assigning splits.
   Document how related variants and near duplicates are handled.
4. Assign all members of a product/duplicate group to one candidate partition with
   a recorded seed. Do this before instruction expansion and class-frequency-based
   task selection. A product ID alone does not prove that near duplicates are absent.
5. Select eligible article types and derive label mappings from the training
   candidate partition. Apply the same frozen eligibility/normalization rules to
   validation and test; do not change them based on those partitions' frequencies
   or outcomes. Sample within eligible partitions with a fixed, recorded procedure.
6. Generate multiple instruction examples only after image groups have been split.
7. Keep held-out test membership fixed. Do not tune prompts, thresholds, settings,
   preprocessing, or label rules from its outcomes. A small visibility/error-review
   subset may be drawn from the frozen test partition; it is not a tuning set.
8. Save source shard/row identifiers, product IDs, group assignments, split membership,
   provenance, transformations, exclusions, and hashes in the local manifest. Bind
   membership to the vocabulary and preparation-policy versions. Verify group
   disjointness, unique membership, valid image references, and reproducibility
   before training; test these rules with synthetic fixtures during implementation.
   No split-preparation implementation, validated split, or split manifest exists
   yet. The merged-source manifest above is not a split manifest.

The proposed first-run sample budget is in [experiments.md](experiments.md); it is
not a claim that the dataset has already been filtered or split.

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
