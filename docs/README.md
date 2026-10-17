# Documentation

The project investigates whether fine-tuning improves reliable structured
fashion-product attribute extraction. The root README summarizes the main question
and the supporting method, GPU-feasibility, and output-reliability questions.

Keep each detailed definition in its owner document and link instead of duplicating
the full protocol. This index distinguishes merged documents from local drafts.

| Document | Owns |
|---|---|
| [Architecture](architecture.md) | Hardware evidence, boundaries, planned data flow, actual-model feasibility gate, and shared training/evaluation responsibilities |
| Planned in issue #4: `docs/dataset.md` | Source/snapshot provenance, audit, canonical task contract and vocabulary, exclusions, visibility limitations, and frozen split integrity |
| Planned in issue #6: `docs/development.md` | Environment profiles, checks, test boundaries, and continuous integration; issue #7 adds hosted-check details |
| Planned in issue #4: `docs/experiments.md` | Canonical detailed research-question hierarchy, question-to-evidence mapping, four-condition controls, metric/abstention definitions, resource measurements, run budgets, and seed/error-analysis protocol |
| Planned in issue #4: `docs/references.md` | External sources, verification scope, and limitations |
| [Decisions](decisions/README.md) | Dated rationale, alternatives, status, and consequences |
| [Root README](../README.md) | Task-first introduction, question summary, illustrative output, scope, current status, environment orientation, and repository map |
| [AGENTS.md](../AGENTS.md) | Working rules and authority boundaries |

Documentation describes current implementation honestly. A planned component is
not an implemented capability, and a proposed experiment is not a measured result.
Paths marked as planned exist as local drafts but are not merged capabilities or
completed specifications. The issue #4 drafts still use the earlier task/question
wording and must align with the new direction before the experiment is finalized.
Their future ownership above does not claim that the current drafts already contain
the refined protocol. Dataset selection does not approve every proposed setting or
authorize execution.

[Issue #13](https://github.com/chathuranga-sudusinghe/vlm-lora-qlora-finetuning/issues/13)
owns this index, the root README, and architecture alignment.
[Issue #4](https://github.com/chathuranga-sudusinghe/vlm-lora-qlora-finetuning/issues/4)
owns dataset/experiment/reference documents and the proposed DR-002 rationale;
DR-001 remains the foundation record. Dependency tooling (#5), diagnostics/tests
(#6), and CPU CI (#7) remain separate scopes. Keep the working rules in
[AGENTS.md](../AGENTS.md); no additional governance or documentation layer is needed.
