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
The planned jobs cover quality/unit checks and CPU integration. The local CI draft
is outside issue #6. Local validation does not establish that hosted CI has passed.

## Verification record

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

The full synthetic suite passes, including the console boundary. No real-model/data
or hosted-CI results are claimed. Owner acceptance and merge remain pending.
