# Development and validation

## Supported environment

Use Linux x86-64, including Ubuntu in Windows Subsystem for Linux (WSL), Python 3.12,
and uv 0.13.0. Other platforms and Python minor versions are outside the tested target.
The lockfile's supported environment is restricted accordingly.

The foundation preserves the existing machine-learning (ML) dependency versions.
MyPy is added for strict checking of project code. Notebook support remains an
optional dependency group rather than part of every development installation.

## Project environment

Run commands from the repository root. Use its `.venv`; the terminal label is
`fashion-vlm`. If the environment does not exist yet, create it before syncing:

```bash
uv venv --python 3.12 --prompt fashion-vlm .venv
```

On the CUDA laptop, install the locked runtime and development tools:

```bash
UV_PROJECT_ENVIRONMENT=.venv uv sync --locked --extra cuda --group dev
```

For a CPU-only machine, replace `--extra cuda` with `--extra cpu`. Select exactly one
profile. The CUDA profile uses CUDA 12.8 wheels and bitsandbytes; the CPU profile
excludes NVIDIA runtime packages and bitsandbytes. Shared packages come from PyPI;
only explicitly mapped PyTorch packages use the official PyTorch indexes.

Activate the existing environment for interactive work:

```bash
source .venv/bin/activate
```

CUDA PyTorch builds can also run CPU tests; the laptop does not need a second
environment for them. Syncing without a runtime extra, or with `--only-group dev`,
removes ML packages from the selected environment. Keep the CUDA extra when syncing
the laptop's `.venv`. Syncing after an entry-point change registers the console
command without recreating the environment.

Add `--group notebook` to the chosen runtime sync command only when using notebooks.
After intentional dependency edits, run `uv lock`, inspect the lockfile diff, then
sync the selected profile. Commit `pyproject.toml` and `uv.lock` together. Do not
reintroduce manually maintained requirements files.

## Execution permissions and quality gates

Follow [AGENTS.md](../AGENTS.md). Package installation/sync, lock generation, Ruff,
and MyPy are owner-first. Provide those commands together in an ordered batch,
one command per code block, and review the supplied outputs. Agents may independently
run bounded unit/integration tests after confirming they use only mocks, synthetic
fixtures, or tiny tensor operations. Real-model/data commands always require prior
explicit owner permission. Outside-repository paths also require prior approval.

After syncing the chosen runtime profile, the owner runs these quality gates:

```bash
.venv/bin/ruff check src/vlm_finetuning tests
```

```bash
.venv/bin/ruff format --check src/vlm_finetuning tests
```

```bash
.venv/bin/mypy
```

Run the isolated unit tests:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m pytest tests/unit
```

Run the CPU integration tests, including the installed console-command boundary:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m pytest tests/integration -m "not cuda"
```

Ruff and MyPy check the new package and tests. The existing standalone
`src/baseline_inference.py` is deliberately outside their scope until its migration.
MyPy checks project code and tests strictly while skipping external PyTorch internals;
passing MyPy does not establish the validity of tensor shapes or CUDA kernels.

| Check | Verifies | Does not verify |
|---|---|---|
| Unit tests | CLI selection, JSON reporting, error statuses, no silent CUDA fallback, safe entry-point import | Real devices, model inference, dataset quality |
| CPU integration tests | Synthetic PyTorch forward/backward, non-finite loss/gradient rejection, module and installed console boundaries | GPU behavior, quantization, model quality |
| CUDA tests | Actual GPU forward/backward, runtime and hardware report | VLM memory fit, bitsandbytes training, speed or quality |
| Ruff / MyPy | Configured style, static issues, and project type contracts | Numerical correctness or experimental validity |

Default pytest runs unit and CPU integration tests and skips CUDA tests. The console
test requires the project to be installed in the same environment as the test runner;
it deliberately fails when the entry point is missing. Unit tests use mocked runtime
boundaries and do not need PyTorch. CPU integration uses the installed framework with
synthetic tensors, not a pretrained model or real dataset. No current test downloads
data/weights or trains a model. Future real-model/data tests require owner approval
and must not silently join the default suite.

## Diagnostic command

The console command and module entry point both accept `--device cpu` or
`--device cuda`. The default is `cuda`; CPU-only environments must select `cpu`
explicitly. CUDA failure never silently selects CPU. Successful execution prints
a JSON report; errors use stderr and a nonzero exit code.

```bash
.venv/bin/vlm-check-device --device cpu
```

```bash
.venv/bin/python -m vlm_finetuning --device cpu
```

`cuda_runtime` describes the installed PyTorch build even during a CPU check.
`total_vram_gib` reports total GPU memory, not peak usage or free memory. A reported
BF16 capability flag is not a BF16 computation or model-training benchmark.

## CUDA checks

Small diagnostic:

```bash
.venv/bin/vlm-check-device --device cuda
```

GPU integration test:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m pytest tests/integration -m cuda --run-cuda
```

`--run-cuda` opts into the marked synthetic tests; on a machine without CUDA they fail
instead of skipping or selecting CPU. It does not authorize real-model/data workloads.
The operations use a tiny synthetic tensor, not a model.

If CUDA fails, inspect `nvidia-smi`, then the installed PyTorch version and
`torch.version.cuda`. The CUDA version displayed by `nvidia-smi` describes driver
capability, not necessarily the runtime in the installed wheel. Do not reinstall a
driver, local CUDA toolkit, or downgrade PyTorch without diagnosing the exact error.

See the [hardware assessment](architecture.md#target-gpu-and-cuda-compatibility) for
the selected Blackwell runtime. The small diagnostic does not verify every extension;
quantization and actual-model feasibility require separate owner-approved checks.

## Continuous integration

[Issue #7](https://github.com/chathuranga-sudusinghe/vlm-lora-qlora-finetuning/issues/7)
owns the hosted workflow, action-version verification, and dependency-cache behavior.
The [workflow](../.github/workflows/ci.yml) was merged through
[PR #18](https://github.com/chathuranga-sudusinghe/vlm-lora-qlora-finetuning/pull/18).
Hosted checks and cold/warm cache behavior have been observed; dependency-file
invalidation remains pending. It runs on pull requests, pushes to `main`, and
manual dispatch. Feature-branch pushes rely on the pull-request trigger to avoid
duplicate push/PR jobs.

| Job | Locked environment | Checks |
|---|---|---|
| Quality and unit tests | `uv sync --locked --only-group dev` | Scoped Ruff lint/formatting, strict MyPy, isolated unit tests |
| CPU integration | `uv sync --locked --extra cpu --group dev` | Assert PyTorch has no CUDA runtime, then run integration tests excluding CUDA |

Both jobs use ordinary `ubuntu-latest` x86-64 runners, Python from `.python-version`,
and uv 0.13.0. They have read-only repository permissions, checkout credentials are
not persisted, and job timeouts are bounded. Dependency installation requires
network access; Hugging Face model/dataset access is offline during execution.
Neither job loads a project model, uses real data, or requests a GPU.

### Action releases

Verified against official releases on 2026-10-18; the workflow pins their complete
commit SHAs rather than mutable version tags:

| Action | Release |
|---|---|
| `actions/checkout` | [v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1) |
| `actions/setup-python` | [v7.0.0](https://github.com/actions/setup-python/releases/tag/v7.0.0) |
| `astral-sh/setup-uv` | [v10.3.0](https://github.com/astral-sh/setup-uv/releases/tag/v10.3.0) |

These actions use Node.js 24; checkout/setup-python document runner 2.327.1 or later.
The workflow targets GitHub-hosted runners, not an unverified self-hosted installation.
The action version is separate from the installed uv 0.13.0 version. See the
[checkout requirements](https://github.com/actions/checkout/blob/v7.0.1/README.md)
and [setup-python requirements](https://github.com/actions/setup-python/blob/v7.0.0/README.md).

### Dependency cache

The setup-uv action caches dependency artifacts in the runner's repository-local
`.cache/uv`, with distinct `quality` and `cpu` key suffixes. Its key includes the
platform/OS version, architecture, selected Python version, and a hash of `uv.lock`
and `pyproject.toml`. The resolved setup-python version is supplied to setup-uv.
The installed `.venv` is recreated and synchronized on each fresh runner.

Cache pruning is explicitly disabled to retain downloaded wheels, including the
CPU runtime. This trades cache storage/transfer size for fewer package downloads;
measure both before changing the policy. The implementation follows setup-uv's
[versioned caching contract](https://github.com/astral-sh/setup-uv/blob/v10.3.0/docs/caching.md).
A cache miss or eviction still permits a normal locked installation, and no check
is conditional on a cache hit.

Each job writes the cache key and restored status to its GitHub run summary.
The install step uses shell `time` to report sync duration; setup-uv restoration
and its post-job save durations remain visible separately in the Actions step view.
Compare their combined cost, not just installation time.

### Verification and hosted evidence

On the laptop, keep the CUDA profile in `.venv` and use the existing quality commands
and synthetic unit/CPU tests above. Do not run the CI-only sync commands against
that environment: the quality sync would remove ML packages, and the CPU sync would
replace the CUDA profile. The two clean CI environments are created on separate
GitHub runners. Reuse valid local quality results while their Python files,
configuration, and environment remain unchanged.

Use this procedure to collect hosted CI and cache evidence:

1. After owner-authorized publication, inspect both jobs on an actual pull-request
   run. A local YAML parse cannot validate GitHub expressions or hosted actions.
2. After a successful run has saved its caches, rerun on the same ref and record
   both jobs' cache status and keys. Do not assume a cache hit across unrelated refs.
3. On an owner-approved change to `uv.lock` or `pyproject.toml`, confirm each job's
   dependency hash/key changes and checks still execute. Keep this item pending if
   no such change has been exercised; do not add a dependency just for this check.
4. Fill the evidence table using actual run URLs and observed timings. An unchanged
   key does not guarantee a hit after cache eviction, and caching is not a guaranteed
   speed improvement.

Hosted results inspected on 2026-10-18 from GitHub job steps and logs:

- The initial [PR #18 run](https://github.com/chathuranga-sudusinghe/vlm-lora-qlora-finetuning/actions/runs/39110499138)
  at `d7b7befe8c392770fc14039a6fd41ed076112446` passed both jobs with cache misses.
- Both `main` runs below passed Ruff lint/formatting, MyPy (8 files), 7 isolated
  unit tests, and 5 CPU integration tests with 1 CUDA test deselected. The CPU
  profile assertion passed with PyTorch `2.11.0+cpu`; all checks still ran on cache hits.

In the table, **setup** is the setup-uv action duration, including uv setup and
cache lookup/restoration; **sync** is the install command's shell `time` real value;
**post** is the setup-uv post-job step, including cache saving on misses. Setup/post
durations come from whole-second GitHub step timestamps; a recorded 0 s can represent
sub-second work. Checkout, Python setup, and tests are excluded from these timings.

| Hosted scenario | Run URL / commit | Quality cache / setup, sync, post | CPU cache / setup, sync, post |
|---|---|---|---|
| Cold cache on `main` | [Run 39110554936](https://github.com/chathuranga-sudusinghe/vlm-lora-qlora-finetuning/actions/runs/39110554936), `37fc5080ff71324ef517e2da80378bb432963d2e` | Miss; 1 s, 0.648 s, 2 s | Miss; 1 s, 3.387 s, 3 s |
| Warm cache on `main` | [Run 39111882266](https://github.com/chathuranga-sudusinghe/vlm-lora-qlora-finetuning/actions/runs/39111882266), `9e4f66015bc8fd5a48be09beffe34f802b08dae3` | Hit; 2 s, 0.034 s, 0 s | Hit; 5 s, 1.045 s, 0 s |
| Dependency-file hash change | Pending | Pending | Pending |

Both `main` runs used the same keys, with separate quality/CPU suffixes:

```text
setup-uv-2-x86_64-unknown-linux-gnu-ubuntu-24.04-3.12.15-035d4d4dc58da012e75e90d98909b5d5678bf1d4710f202b5f1baffe479bbd58-quality
setup-uv-2-x86_64-unknown-linux-gnu-ubuntu-24.04-3.12.15-035d4d4dc58da012e75e90d98909b5d5678bf1d4710f202b5f1baffe479bbd58-cpu
```

Warm syncs were shorter, but cache restoration adds overhead; these observations
do not establish a consistent overall speedup. Dependency-file invalidation has
not been exercised, so issue #7 is not complete. Keep that check pending until an
owner-approved dependency-file change; no dependency change is needed solely for it.
Hosted CPU checks do not establish CUDA or actual-model feasibility.

## Verification record

### 2026-10-18: issue #7 local checks

On `ci/issue-7-quality-cpu-checks`, the coding agent checked the workflow with the
existing environment; no dependencies were installed and the CUDA profile was
preserved:

- YAML parsing and targeted structural checks passed for triggers, job boundaries,
  verified release SHAs, cache paths/keys, and unconditional check steps.
- Every embedded `run` block passed `bash -n` syntax checking without executing
  its installation or quality commands. No dedicated `actionlint` binary was
  installed; these checks are not complete GitHub workflow-schema validation.
- `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m pytest tests/unit tests/integration -m 'not cuda' -p no:cacheprovider`:
  12 passed, 1 CUDA test deselected. CPU tensors ran in the existing CUDA-profile
  environment, not a newly installed CPU-only environment.
- Python files, dependency/tool configuration, and the lockfile are unchanged by
  #7. The owner-run Ruff/MyPy evidence from #6 remains applicable locally; the
  new workflow will rerun those checks on GitHub.

### 2026-10-18: issue #7 independent review

The owner supplied a Google Antigravity review of the uncommitted issue #7 files on
`ci/issue-7-quality-cpu-checks`, based on commit `72e3ac4`. It reported no confirmed
defects in the workflow, development guide, or README and concluded ready to commit.
The reviewer reported successful YAML parsing and embedded Bash syntax checks,
plus 12 passing synthetic unit/CPU integration tests with 1 CUDA test deselected.
Existing owner-run Ruff/MyPy evidence was reused for unchanged Python code and
configuration.

The coding agent checked the findings against the current files and agreed that
no workflow fixes were needed. The owner approved this documentation-only update
to record the completed review; this later note was not independently reviewed.
Opening or updating the pull request triggers CI; pushing this feature branch
alone does not.

At the time of this pre-publication review, hosted jobs, clean runner profile
isolation, cache hits/misses, dependency-hash invalidation, and timing measurements
were pending. Later hosted results are recorded above; dependency-file invalidation
remains unobserved. The earlier review does not cover this evidence follow-up.

### Issue #5 dependency evidence

Issue #5's owner-run dependency results are recorded in the
[root README](../README.md#dependency-profiles). The earlier foundation snapshot
has been removed because it does not establish verification of this revision.
Record each new result with its date, executor, command scope, and limitations.

### 2026-10-18: coding-agent synthetic checks

These runs used the existing repository `.venv` (Python 3.12.3, pytest 9.1.1), with
Hugging Face offline flags enabled and pytest's cache provider disabled.

| Scope | Observed result |
|---|---|
| `tests/unit`, with `sys.modules["torch"] = None` before pytest startup | 7 passed; unit tests do not require an importable PyTorch runtime |
| `tests/unit tests/integration --run-cuda` after owner package sync and assertion wrapping | 13 passed; includes the installed console command, import guard, CPU error checks, and tiny CUDA diagnostic |
| TOML metadata parsing | `pyproject.toml` and `uv.lock` parsed successfully; this does not establish lockfile freshness |

CPU integration used CPU tensors in the CUDA-profile environment; it was not a
clean CPU-only installation. CUDA runs set `CUDA_CACHE_DISABLE=1` and did not load
a model, dataset, or weights. These are synthetic diagnostics, not model
training, quantization, or BF16 numerical benchmarks.

### 2026-10-18: owner-run installation and quality checks

- Locked CUDA/development synchronization passed, rebuilding and reinstalling the
  editable project so `vlm-check-device` is registered. The dependency lock remained
  unchanged.
- MyPy passed for 8 source files. The subsequent code edit only wrapped an existing
  assertion message; it changed no types or behavior.
- Ruff lint and formatting initially identified the same 101-character assertion
  line. After wrapping it, the owner reran
  `.venv/bin/ruff check src/vlm_finetuning tests`: all checks passed.
- The owner reran `.venv/bin/ruff format --check src/vlm_finetuning tests`:
  8 files already formatted.

### 2026-10-18: independent review supplied by the owner

The Google Antigravity review inspected the issue #6 working tree on
`feat/issue-6-device-diagnostics`, based on commit `afdea5d`, including the
uncommitted implementation and tests. It reported:

- Synthetic unit/CPU integration tests: 12 passed, 1 CUDA test deselected.
- Explicit synthetic CUDA test: 1 passed.
- The console entry point, MyPy test scope, import guard, non-finite loss/gradient
  checks, and assertion formatting finding were resolved.

The review's remaining request for Ruff evidence is satisfied by the owner outputs
above. Its request to repeat MyPy is unnecessary: the owner already supplied a
passing result, and the subsequent assertion wrapping changed no types or behavior.
No further implementation changes are proposed from this review. These records
do not imply that the later documentation updates were independently reviewed.

The full synthetic suite passed, including the console boundary. No real-model/data
or hosted-CI results were claimed. Issue #6 was subsequently merged; #7 owns hosted
workflow and caching verification.
