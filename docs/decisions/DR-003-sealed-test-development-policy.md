# DR-003: Test-first preparation and sealed evaluation

- **Date:** 2026-10-20
- **Status:** owner-selected methodology documented under issue #30;
  implementation choices and execution remain pending
- **Supersedes:** the preparation ordering, categorical task-policy derivation,
  and abstention-policy assumptions in [DR-002](DR-002-product-attribute-experiment.md)

## Context

Issue #28 / PR #29 completed a source-preserving canonical merge. Its saved run
reports 44,072 rows, all 11 original columns, and preserved image bytes. The source
and samples were inspected before this decision; historical measurements remain
in [dataset.md](../dataset.md#saved-execution-evidence--2026-10-19).

The earlier proposal audited/grouped the source before splitting and derived
categorical policies from candidate Train. The owner instead selected a random
Test holdout first, then Development-only investigation and task design. This
record changes those assumptions without reopening issues #4, #22, #23, or #28.
It retains the dataset, image-only four-field task, and four-condition comparison.

## Decision

Keep the canonical merged artifact immutable. Select approximately 10% random
Test membership with a preselected fixed seed, save reproducible membership
metadata, and logically seal it. Use the remaining Development pool for audits
and versioned policies, then prepare approximately 8/9 Train and 1/9 Validation
within Development, mainly stratified by `articleType`, with meaningful discovered
Development groups kept disjoint. This targets approximately 80/10/10 overall
before exclusions and group constraints, not an exact count promise.

Prefer metadata that recovers exact Test IDs from the canonical source at final
evaluation over a second physical Test dataset. Keep membership immutable while
Development policies evolve. Fit learned preprocessing statistics and weights
using Train only; select configurations using Validation. Freeze the complete
system, including source-code `unknown`/rejection decisions, before final Test access.
Final results cannot feed further development.

[dataset.md](../dataset.md#preparation-and-splitting-rules) owns sealing, membership
reproducibility, Development audits, grouping, and task-policy definitions.
[experiments.md](../experiments.md#evaluation) owns final access, raw versus final
outputs, supported-class scores, unsupported-input rejection, and denominators
that account for every sealed Test member. Keep future production human review
and retraining as a separate documentation-only extension; do not train from
rejected records in the current Test set.

## Alternatives considered

- **Global duplicate audit before Test selection:** could support duplicate-aware
  boundaries but differs from the selected Test-first, Development-only order.
  It is not silently included; cross-boundary duplicate protection is unproven.
- **Physical Test Parquet:** unnecessary by default because the immutable source
  and exact membership support recovery. An exception needs a documented technical
  reason and owner agreement.
- **Train-only categorical task design:** superseded by Development-derived
  policies. This does not permit fitting learned preprocessing statistics on
  Validation or presenting Validation as an independent policy-design assessment.
- **Chronological/newest-data holdout:** not selected; this policy uses a random
  same-catalog holdout and does not establish future-domain performance.
- **Vocabulary checks as unsupported-input detection:** insufficient when an
  unsupported product receives a valid known label. A separate decision mechanism
  needs Development-only validation, with raw failures still visible.

## Consequences and unresolved choices

The seal is prospective, logical, and tied to an exact source snapshot; it is not
encryption or proof that source content was never inspected. Record-random Test
selection before Development-only duplicate auditing cannot establish duplicate-free
Development/Test boundaries or wholly unseen products. Group constraints can also
prevent exact Train/Validation balance. Report these limits without adding Test
audits or choosing a more favorable Test seed.

Keep partition membership distinct from compute budgets. The earlier 250-image
Test budget no longer defines Test membership or evaluation coverage. Every sealed
member remains accounted for, including unsupported inputs, missing reference
labels, and execution failures. Distinguish raw schema validity, known-class
prediction quality, and final rejection behavior; neither valid JSON nor `unknown`
alone proves reliable unsupported-input detection.

Before implementation, agree the algorithm/random generator and versions, seed,
rounding and manifest serialization, access guards, rare-class/conflicting-group
handling, and decision mechanism/thresholds. Final task vocabularies and actual
counts require Development evidence. This documentation creates no seal, partitions,
guard code, preparation implementation, model execution, or approved run command.

## Supporting guidance

The scikit-learn [data-leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
supports separating held-out evaluation from development and fitting transforms
on training data. Its [cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)
explains random-split and group limitations; stratification does not itself remove
dependent observations. These support the stated cautions, not evidence that this
project's chosen holdout is duplicate-free or that any rejection method works.
