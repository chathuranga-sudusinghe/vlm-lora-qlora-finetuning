# Working rules

Read this file and the relevant owner documents before changing the repository.
These rules apply to all assistants and contributors. The project owner remains
responsible for accepting code, design decisions, experiment results, and Git operations.

## Project ownership and assistant roles

- The human project owner owns and directs this project. It is not an AI tool's
  project. Final decisions on scope, design, methodology, implementation acceptance,
  experiments, licensing, and Git/GitHub operations belong to the project owner.
- Codex is the coding agent: it may implement explicitly authorized changes within
  the agreed issue and branch. It cannot approve its own work or finalize decisions.
- Claude and ChatGPT are independent review tools: they provide findings, technical
  challenges, and recommendations for the owner to evaluate. They do not edit files
  or finalize changes unless the owner explicitly changes their assigned roles.
- All assistants follow the same owner-first command rules and approval boundaries.
  Their recommendations do not become decisions until the owner accepts them.
- Use tool names only where technically relevant, such as defining these roles.
  Do not add AI tool/vendor branding, promotional text, generated-by labels, or
  AI co-author trailers to project artifacts unless the owner explicitly requests it.

## Scope and proportionality

- This is a small, reproducible vision-language model (VLM) adaptation project for
  product attributes, not a university research protocol or a production platform.
- Follow [architecture](docs/architecture.md), [dataset policy](docs/dataset.md), and
  the [experiment plan](docs/experiments.md). Distinguish implemented behavior from proposals.
- Prefer the smallest correct change. Reuse conventions, avoid premature abstraction,
  unrelated refactoring, service scaffolding, and formatting-only churn.
- Preserve unrelated and untracked user work, including standalone experiments.
- Read existing decisions before asking the owner to decide the same question again.
  Record new material decisions with context, alternatives, consequences, and truthful status.

## Human control and Git

- Every meaningful change requires a GitHub issue and a suitable local issue-specific
  branch before implementation. Group only closely related files under one issue;
  split distinct concerns into separate issues and branches. Do not place the entire
  project foundation on one branch merely because the files were created together.
- Agree the issue scope, affected files, and branch with the project owner before
  coding. Approval of a general plan is not approval to create all proposed files.
- Before creating each GitHub issue, present its proposed title, scope, affected
  files, and acceptance criteria, then ask the project owner for explicit permission.
  Wait for approval before creating it. Permission to inspect GitHub, propose issues,
  or implement a change is not permission to create an issue. Do not infer approval
  from silence or create additional issues beyond the scope the owner approved.
- Assistants may inspect content through read-only file tools and make explicitly
  authorized file edits. Terminal execution follows the owner-first rules below.
  Code-editing permission does not authorize training, large downloads, or publication.
- Do not commit, push, pull, merge, rebase, create/delete branches, create/merge pull
  requests, create/update GitHub issues, or release unless the owner explicitly
  requests that operation. Such authorization does not waive the terminal rules.
- Never commit or push directly to `main`. Use an issue-specific local branch and
  a reviewed pull request. After the issue number is known, propose a meaningful
  task-based branch name: `<type>/issue-<number>-<short-topic>`, using a relevant
  type such as `docs`, `chore`, `feat`, `fix`, or `test`. Do not use an AI tool or
  vendor name as the branch prefix.
- Commit messages must describe the actual issue-scoped change, using a suitable
  type and a clear subject, for example `docs: define project workflow rules (#<number>)`.
  Proposed Git commands must name the actual approved branch, relevant file paths,
  and a meaningful commit message. Replace placeholders only after the real issue
  number and scope are known; never invent issue numbers or add tool branding.
- Stage only files belonging to the active issue. Do not use `git add .` or
  `git add -A` to collect unrelated or pre-existing work. Do not switch branches
  with mixed pending work until the owner has reviewed a safe separation plan.
- Never interpret an assistant's own proposal as a reviewed experimental result or
  an owner-approved methodology change. Mark unsettled design choices as proposed.

## Owner-first terminal execution

- No AI assistant runs a terminal command for the first time in this project.
  This applies to coding, reviews, read-only inspection, Git, checks, installs,
  locking/syncing, scripts, diagnostics, and commands issued through agent tools.
- Give the project owner the exact suitable command, one command per code block,
  with its purpose, working directory, impact, and expected result. The owner runs
  it first and supplies the output. Check that output before proposing another change.
- Only after the owner has run the command and supplied its output may an assistant
  repeat that same command to verify the result. Prior execution by an assistant
  does not count as owner execution or establish permission.
- Before repeating, confirm that the command, arguments, environment settings,
  working directory, and relevant implementation still match the owner's run.
  A changed script, materially changed implementation, or changed command needs
  a fresh owner run and output first. Ask when the evidence is unclear.
- The repeat permission is for verification. It does not authorize additional edits,
  installs, data generation, training, Git mutations, or external publication.
  Any consequential operation still needs the owner's explicit authorization.
- Never bypass these rules by moving a command into Python, a script, subprocess,
  another shell, a scheduled job, a tool wrapper, or another agent.
- When the required owner execution/output is missing, provide the command and
  wait. Do not execute it, claim it passed, or infer permission from silence.

## Environment and implementation

- Use Python 3.12, uv 0.13.0, `pyproject.toml`, and `uv.lock`; do not maintain parallel
  requirements files. Follow the runtime-profile commands in [development](docs/development.md).
- Keep code and virtual environments in Linux/WSL paths. Do not share a Windows
  virtual environment with WSL. Never assume Windows and Linux paths are interchangeable.
- Keep reusable implementation in `src/vlm_finetuning/`. Imports must not download
  files, initialize GPUs, run inference, or start training. Use explicit entry points.
- Use type hints and the configured Ruff, MyPy, and pytest checks. Keep missing-import
  exceptions narrowly scoped to external libraries; do not silence project type errors.
- Use module loggers for diagnostics; configure logging only at entry points.
  Command output may deliberately use stdout for user-facing text or JSON.
- Raise clear validation errors. Do not silently swallow exceptions or fall back
  from a requested CUDA device to CPU. Preserve useful causes and failure exit codes.
- Never expose credentials, tokens, secrets, or private data in source, logs, or reports.

## Data and experiment integrity

- Raw images, derived datasets, model weights, adapters, checkpoints, caches, and
  generated predictions stay out of Git. Use tiny synthetic fixtures in tests.
- Preserve dataset provenance and third-party terms. The project MIT license does
  not grant rights to third-party images or model weights.
- Split by product/duplicate group before creating multiple prompts for an image.
  Fit preprocessing and label mappings from training data only; document exclusions.
- Use validation for model/configuration selection. Do not use held-out test labels
  to tune prompts, thresholds, preprocessing, or hyperparameters.
- Keep target metadata out of model inputs. Do not infer non-visible attributes from
  product names and describe them as image-grounded truth.
- Record seeds, exact sample IDs, dataset/model revisions, configuration, package
  versions, device, training duration, and memory measurements for evidence-producing runs.
- Never fabricate sources, results, performance gains, timings, or hardware support.
  Separate observation, sourced evidence, and recommendation.
- Compare adapters using the same data and evaluation protocol. Do not assume
  quantized Low-Rank Adaptation (QLoRA) is faster or more accurate than LoRA.

## Validation and execution limits

- Ask the owner to run targeted checks after meaningful code changes, then the
  configured quality gates. An assistant may repeat them only under the owner-first
  terminal rules; writing a check does not authorize executing it.
  Unit tests are isolated; integration tests exercise real component boundaries.
- Default tests require no network, real datasets, downloaded weights, or GPU.
  CUDA tests require `--run-cuda`; an explicit CUDA test must fail when CUDA is unavailable.
- Ordinary continuous integration (CI) runs CPU checks only. Never add model training,
  external-data scoring, or secret-dependent tests to pull-request CI by default.
- Explain the impact before large installs/downloads or expensive checks. Obtain explicit
  owner authorization before model training, sweeps, costly cloud work, or large data jobs.
- A tiny synthetic device forward/backward diagnostic follows the same owner-first
  command rules. It is not evidence that a VLM training configuration fits or that
  all CUDA extensions work.

## Documentation and handoff

- Keep each rule/design definition in one owner document; link instead of duplicating
  full specifications. [docs/README.md](docs/README.md) lists document ownership.
- Update relevant docs and decision records when behavior or accepted scope changes.
  Use clear English, expand unfamiliar abbreviations, and verify external sources.
- Keep decision numbers stable. Supersede an accepted decision with a linked new record
  when its direction changes; do not rewrite measured evidence to match a later design.
- End meaningful editing work with status, changed files, checks and limitations,
  current branch/state, suggested Git action and commands, and the best next engineering step.
  Never imply a check ran or passed unless it did.
