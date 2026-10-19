# Dataset and integrity policy

**Status (2026-10-19):** the dataset and four-field output scope are owner-selected;
issue #23 aligns the contract with that scope. The audit, final canonical vocabulary,
exclusions, and preparation rules remain proposed under issue #4. This documentation
change introduces no new dataset measurements, prepared split, or manifest.
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
file sizes and source metadata, not a locally verified row count or integrity check.
Neither Parquet rows nor image payloads were loaded during this documentation work.

`data/raw/README.md` and the downloaded `data/raw/gitattributes` are supporting
metadata, not extra training examples. No standalone `sample.jpg` is required.
Both Parquet files are ignored by the current local ignore rules; committing those
rules and the local-data README belongs to issue #5. Never stage `data/raw/`.

**Provenance still to establish:** the exact revision used for the existing download
and the cryptographic checksums of both shards. A filename suffix, matching byte
count, or the Hub's current `main` revision does not prove the revision or integrity
of an earlier download. Before preparation, record the verified artifact revision
and checksums, or explicitly identify a local snapshot with unresolved upstream
revision. Do not invent a revision or claim checksum verification has happened.

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
   No preparation implementation, validated split, or manifest exists yet.

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
