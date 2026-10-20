# Decision records

Record material choices under an owner-approved issue and a suitable local branch,
following [AGENTS.md](../../AGENTS.md). Keep the process proportionate to the project.
Use stable `DR-NNN-short-title.md` identifiers. Each record states its date, status,
context, decision, alternatives, and consequences.

| Record | Status | Owns rationale for |
|---|---|---|
| [DR-001](DR-001-project-foundation.md) | Working rules/license accepted; implementation choices pending issue-specific review | Dependency tooling, quality gates, runtime profiles, documentation, and code license |
| [DR-002](DR-002-product-attribute-experiment.md) | Dataset selected; detailed experiment settings proposed | Dataset choice, bounded task, model continuity, and initial comparison |
| [DR-003](DR-003-sealed-test-development-policy.md) | Test-first methodology selected; implementation/execution pending | Test sealing, Development-derived policies, final freeze, and evaluation limits; partially supersedes DR-002 |

Keep approval claims truthful. Proposals can be refined in place while clearly marked
as proposals. If an accepted direction changes, add a new record that links to and
supersedes the old one; keep identifiers stable. Do not rewrite saved results.
