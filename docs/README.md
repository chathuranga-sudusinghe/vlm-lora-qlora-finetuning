# Documentation

The project investigates whether fine-tuning improves reliable structured
fashion-product attribute extraction. The [root README](../README.md#research-questions)
summarizes the main question and the supporting method, GPU-feasibility, and
output-reliability questions.

Keep each detailed definition in its owner document and link instead of duplicating
the full protocol. This index describes document ownership; merged documentation
does not imply that every planned capability is implemented.

| Document | Owns |
|---|---|
| [Architecture](architecture.md) | Hardware evidence, boundaries, planned data flow, actual-model feasibility gate, and shared training/evaluation responsibilities |
| [Dataset](dataset.md) | Source/snapshot provenance, canonical artifact and task contract, Test-first membership policy, Development-only audit, vocabulary/eligibility rules, and split integrity |
| [Development](development.md) | Environment profiles, execution permissions, quality checks, synthetic tests, device diagnostics, and continuous integration evidence |
| [Experiments](experiments.md) | Detailed research questions, four-condition controls, raw/final metrics, rejection evaluation, final freeze, resource measurements, run budgets, and seed/error-analysis protocol |
| [References](references.md) | External sources, verification scope, and limitations |
| [Decisions](decisions/README.md) | Dated rationale, alternatives, status, and consequences |
| [Root README](../README.md) | Task-first introduction, question summary, illustrative output, scope, current status, environment orientation, and repository map |
| [AGENTS.md](../AGENTS.md) | Working rules and authority boundaries |

These owner documents are merged. Dependency tooling (#5), device diagnostics and
synthetic tests (#6), and CPU continuous integration (#7) are implemented. The
[development guide](development.md#verification-and-hosted-evidence) records the
validation evidence and its limits.

Source inspection (#22), the four-field scope (#23), and canonical merging (#28)
are completed. The [dataset document](dataset.md#canonical-merged-artifact) owns
the saved merge/inspection evidence and limitations. The Test-first preparation
policy (#30) is documented in
[DR-003](decisions/DR-003-sealed-test-development-policy.md).

Test sealing, Development-only preparation, controlled model evaluation, and adapter
training remain future work. Final experiment settings still require owner review;
a proposed experiment is not a measured result. Dataset selection and merged
documentation do not authorize data/model execution.

The [decision index](decisions/README.md) preserves the foundation and experiment
rationale, including DR-003's partial supersession of DR-002. Keep working rules
and execution permissions in [AGENTS.md](../AGENTS.md).
