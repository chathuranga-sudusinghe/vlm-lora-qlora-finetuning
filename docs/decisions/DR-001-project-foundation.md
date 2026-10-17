# DR-001: A proportionate project foundation

- **Date:** 2026-10-17
- **Status:** working rules and MIT license accepted/merged; implementation choices pending issue-specific review

## Context

The repository initially contained two requirements files, an ignore file, and a local
standalone baseline. The owner selected `pyproject.toml` and requested Ruff, MyPy,
pytest unit/integration tests, CUDA support, architecture/decision documentation,
working rules, a suitable license, and continuous integration (CI).

The owner's separate research repository supplies useful integrity and reproducibility
principles. The owner subsequently required an issue-first workflow, separate local
branches, explicit assistant roles, and owner-first execution with an independent
read-only inspection exception. These requirements are authoritative in
[AGENTS.md](../../AGENTS.md); the university-specific research protocol is outside scope.

## Decision

The agreed foundation scope is Python 3.12, uv with a committed lockfile, strict
project typing, and the requested quality tools. Issue #5 will review the actual
dependency configuration, mutually exclusive CPU/CUDA profiles, and explicit PyTorch
indexes. Keep notebook tooling optional and preserve the existing baseline until
its migration.

The hardware target is the RTX 5060 Laptop GPU. The
[architecture hardware assessment](../architecture.md#target-gpu-and-cuda-compatibility)
owns its observed properties and CUDA compatibility evidence. The current environment
uses CUDA 12.8 PyTorch wheels; CUDA 13.0 wheels for the same PyTorch release are a
proposed upgrade candidate for issue #5. The final runtime choice and validation
remain owner-controlled rather than being fixed by this documentation change.

Issue #6 will review a small typed device diagnostic as a bounded test boundary.
Issue #7 will add CI for quality and CPU integration, with offline Hugging Face
runtime access. The intended local CUDA tests are opt-in; default checks must not
download a model/dataset or train a model. These are planned implementation boundaries.

Use MIT for original project code and documentation, preserving the copyright notice
and keeping third-party data/model licensing separate. The owner accepted this
choice through issue #2 and its merged license pull request.

## Alternatives considered

- Keep both requirements files: minimal migration, but duplicates tooling concerns
  and omits a shared package/tool configuration.
- CUDA-only setup: matches the laptop but makes hosted CI unnecessarily heavy.
- Full research governance: useful for the original study, disproportionate here.
- Apache-2.0 for original code: a reasonable alternative with explicit patent terms;
  MIT keeps this small codebase's licensing simple. Third-party terms remain separate.

## Consequences

Once issue #5 is implemented, select a runtime profile explicitly when syncing.
Future CI cannot prove local GPU or VLM training behavior. The foundation adds no
service/database infrastructure. Dependencies, platform expansion, and changes to
accepted working rules require owner review and an appropriately scoped issue.
