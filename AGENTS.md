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
- Google Antigravity, Claude, and ChatGPT are independent review tools: they provide
  findings, technical challenges, and recommendations for the owner to evaluate.
  They do not edit files or finalize changes unless the owner explicitly changes
  their assigned roles.
- All assistants follow the same owner-first command rules and approval boundaries.
  Their recommendations do not become decisions until the owner accepts them.
- Use tool names only where technically relevant, such as defining these roles.
  Do not add AI tool/vendor branding, promotional text, generated-by labels, or
  AI co-author trailers to project artifacts unless the owner explicitly requests it.

## Independent review workflow

- In the owner's current setup, Codex and Google Antigravity can inspect the local
  repository and GitHub; ChatGPT reviews GitHub content only. Confirm actual access
  in each session, including Claude's available access. These are workflow assignments,
  not guarantees about a product's capabilities or permission to perform operations.
- The owner selects the reviewer and review scope. Reviewers may independently
  inspect files and run the routine commands permitted under Terminal execution
  permissions, including eligible unit/integration tests, but may not edit
  implementation files. Real-model/data commands require explicit owner permission;
  other owner-first commands follow the rules below. These boundaries apply equally
  to reviewers and coding agents. Do not contact another tool or send repository
  content to it without the owner's explicit authorization.
- Identify the issue, branch, commit SHA, and files under review. For local reviews,
  distinguish committed code, uncommitted changes, and relevant untracked drafts.
  For GitHub reviews, identify the pull request and head commit when available;
  a GitHub-only reviewer cannot verify changes that exist only on the laptop.
- Reviewers independently inspect the available implementation, issue criteria,
  and tests. Report actionable findings with file/line references, severity, and
  proposed fixes, following the owner's file-by-file review format. State what was
  inspected, what checks actually ran, and what access or evidence was unavailable.
  A review of an earlier revision does not establish the correctness of later edits.
- Codex independently evaluates each reviewer finding against the current code,
  requirements, tests, and available evidence. Do not automatically agree with a
  reviewer or reject a finding to defend existing work. Distinguish confirmed
  defects, evidence gaps, optional improvements, and preferences; reuse valid
  outputs already supplied for unchanged behavior.
- Before applying reviewer recommendations, present a comparison table containing
  the reviewer finding, Codex's assessment (agree, partly agree, or disagree),
  supporting evidence, and the proposed action. Explain disagreements and identify
  which changes are required and which are optional.
- After presenting the table, ask the owner for explicit approval of the proposed
  changes and wait before editing files to implement them. Receiving a review,
  agreeing with a reviewer, or earlier general coding permission does not authorize
  reviewer-driven edits. Implement only the approved changes within the agreed
  scope. If no changes are needed, state that without requesting unnecessary approval.
- Recheck affected behavior after meaningful approved fixes. Reviewer agreement
  does not replace tests or owner approval; the owner decides whether findings are
  accepted and whether the change is ready to commit or merge.

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
- Assistants may independently inspect content and run routine terminal checks
  permitted below, and may make explicitly authorized file edits. Protected
  execution follows the terminal permission rules below.
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

## Terminal execution permissions

### Commands agents may run independently

- Assistants may run read-only inspection commands without asking permission and
  without requiring a prior owner run or supplied output. This includes
  `git status --short --branch`, other read-only `git status` forms, `git diff`,
  `git log`, `git show`, `git ls-files`, and `git branch --show-current`.
- Assistants may read and search ordinary project files using `cat`, `sed -n`,
  `head`, `tail`, `rg`, `ls`, and equivalent read-only file tools. Inspect files
  directly instead of asking the owner to run a command merely to display content.
- Independent command permission applies within this repository. Reading or
  searching files or folders outside it requires prior explicit owner permission,
  as described under Environment and implementation.
- Coding agents and independent reviewers may also run routine, bounded checks
  using the existing environment: command help/version queries, installed-package
  metadata inspection, import probes, and tiny synthetic CPU/CUDA diagnostics.
  No prior owner run or output is required for these permitted checks. Inspect
  unfamiliar scripts before executing them to establish their actual effects.
- Unit and integration tests, including pytest, may run independently when they
  use only mocks, synthetic fixtures, or tiny synthetic tensor operations, with
  no real project model, pretrained weights, real dataset, or external download.
  Inspect the selected tests and fixtures first; a test name or marker alone does
  not establish permission. Run only the eligible subset if a suite mixes safe
  tests with real-model/data workloads. Keep resource use bounded.
- These routine checks must not install packages, download data or model weights,
  change project source/data or Git state, train models, or perform costly work.
  Use repository-local paths for any generated output. Tool caches do not authorize
  unapproved outside-repository access. Protected commands cannot be disguised as
  routine checks or hidden inside scripts, aliases, or chained commands.
- Protect secrets and sensitive data during inspection. Read-only access does not
  authorize exposing credentials, tokens, private records, or restricted data.

### Real-model and real-data commands: explicit owner permission

- No AI tool may run commands involving real project models or real datasets
  without prior explicit permission from the human project owner. This includes
  model/weight downloads and loading, training/fine-tuning, inference, validation,
  testing, evaluation, benchmarking, and real-data preparation or processing.
  The rule applies to coding agents, independent reviewers, and delegated tools.
- Calling a command a unit test, integration test, smoke test, or validation check
  does not exempt it. Using synthetic inputs with a real project model still
  requires permission. If a command's effects are unclear, inspect it first and
  obtain permission before running any real-model/data workload.
- Present the exact commands, working directory, models/data involved, scope,
  and expected compute, download, and disk impact before requesting approval.
  The owner may run the commands or explicitly authorize the agent to run them.
  General coding/review permission and earlier execution outputs do not authorize
  a new real-model/data run. Stay within the approved operation and scope.

### Other owner-first commands and consequential operations

- Ruff lint/format commands and MyPy type checks remain owner-first for both coding
  agents and independent reviewers. The owner must receive the commands, run them
  first, and supply the output before agent verification. Eligible synthetic unit
  and integration tests follow the independent permission above.
- Builds, dependency installs, lock generation, and environment synchronization
  also remain owner-first. Real-model/data workloads follow the explicit-permission
  section above.
- Git-changing commands also remain owner-first and require explicit authorization.
  Examples include `git add`, `git commit`, `git fetch`, `git pull`, `git push`,
  `git switch`, `git checkout`, `git merge`, `git rebase`, `git stash`, `git reset`,
  `git restore`, `git clean`, and branch creation/deletion. A tool's name alone does
  not make an operation read-only; classify it by its actual effect.
- Give the owner the complete known set of owner-first commands for the current task
  in one ordered batch, with one command per code block. State the working directory,
  purpose, impact, expected results, and when to stop on failure. The owner runs the
  batch and supplies the combined outputs. Do not split an already-known sequence
  into repeated one- or two-command handoffs. A later command may be withheld when
  it genuinely depends on an earlier result; explain that dependency.
- Review the owner's outputs before making dependent changes. While waiting,
  continue useful independent inspection and permitted routine checks. Do not
  repeatedly ask for commands or outputs already supplied for unchanged code and
  environment, or stop a review merely because a permitted command has not yet
  been run by the owner.
- Only after the owner has run an owner-first command and supplied its output may
  an assistant repeat that same command to verify the result. Prior execution by
  an assistant does not count as owner execution or establish permission.
- Before repeating, confirm that the command, arguments, environment settings,
  working directory, and relevant implementation still match the owner's run.
  A changed script, materially changed implementation, or changed command needs
  a fresh owner run and output first. Ask when the evidence is unclear.
- The repeat permission is for verification. It does not authorize additional edits,
  installs, data generation, training, Git mutations, or external publication.
  Any consequential operation still needs the owner's explicit authorization.
- Never bypass these rules by moving a command into Python, a script, subprocess,
  another shell, a scheduled job, a tool wrapper, or another agent.
- When required owner execution/output is missing for an owner-first command,
  provide the complete applicable batch and leave dependent validation pending.
  Do not execute it, claim it passed, or infer permission from silence. Complete
  the remaining independent work and clearly identify the pending evidence.

## Environment and implementation

- Use Python 3.12, uv 0.13.0, `pyproject.toml`, and `uv.lock`; do not maintain parallel
  requirements files. Follow the runtime-profile commands in [development](docs/development.md).
- Keep code and virtual environments in Linux/WSL paths. Do not share a Windows
  virtual environment with WSL. Never assume Windows and Linux paths are interchangeable.
- Keep project-related files, environments, and outputs inside this repository by
  default. Files and folders outside it may be inspected or used only with the
  owner's prior explicit permission. Before reading, searching, creating, changing,
  deleting, or installing anything outside the repository, explain the paths,
  purpose, and intended operation, ask the owner, and wait for approval. This also
  applies before giving the owner commands that use those locations. Permission to
  inspect does not authorize changes or installation.
- This project-work rule is not a blanket ban on separate owner-approved tasks
  involving personal or global configuration outside the repository. Such access
  still requires prior explicit permission for the relevant paths and operations.
- Example: creating `/tmp/vlm-issue-5-cpu`, `/tmp/vlm-issue-5-cuda`, or
  `/tmp/vlm-issue-5-quality` to install dependencies and run temporary checks requires
  prior explicit owner permission. Approval to check dependencies does not authorize
  using folders outside the repository.
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

- After meaningful code changes, run eligible synthetic unit/integration tests
  independently and give the owner the remaining owner-first quality gates as an
  ordered batch. Real-model/data tests require explicit approval before execution.
  Unit tests are isolated; integration tests exercise real component boundaries.
- Read-only inspection and permitted routine runtime checks may be performed
  independently by coding agents and reviewers under the terminal execution rules.
  Report static inspection, runtime probes, and test-suite results separately.
- Default tests require no network, real datasets, downloaded weights, or GPU.
  CUDA tests require `--run-cuda`; an explicit CUDA test must fail when CUDA is unavailable.
- Ordinary continuous integration (CI) runs CPU checks only. Never add model training,
  external-data scoring, or secret-dependent tests to pull-request CI by default.
- Explain the impact before large installs/downloads or expensive checks. Obtain explicit
  owner authorization before model training, sweeps, costly cloud work, or large data jobs.
- A tiny synthetic device forward/backward diagnostic may run independently using
  the existing environment, including through an eligible unit/integration test.
  CUDA opt-in controls still apply. Neither establishes that a VLM training
  configuration fits or that all CUDA extensions work.

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
