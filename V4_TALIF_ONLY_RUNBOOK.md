# TA-LIF-only v4 RTX 5090 execution runbook

This is the only authorized execution order for protocol v4. Pilot and health
artifacts are non-reportable. Do not run either health command on the local RTX
4070: creating the attempt receipt consumes that dataset's pilot seed even if
the command fails or is interrupted.

## Frozen scope

- Protocol: `configs/protocol_v4_talif_only.yaml`
- Protocol hash: `68bb14e892a696bc87b3a2d3f196b81fc58b299dd0c485cf72a50bb5fd0b0a54`
- Conditions: C1 LIF and C2 TA-LIF only, both with the repaired 19-neuron graph
- Datasets: CIFAR-100 primary and CIFAR10-DVS external-validity replication
- Pilot: four non-reportable 120-epoch runs
- Formal: twenty runs, five dataset-specific C1/C2 seed pairs per dataset
- Target environment: RTX 5090, PyTorch `2.9.1+cu128`, CUDA runtime `12.8`,
  deterministic float32, AMP disabled

One-shot health/pilot seeds:

- CIFAR-100: `1975342236`
- CIFAR10-DVS: `1983855948`

The longitudinal health gate uses mean residual-block-by-batch gradient
coverage, the minimum per-layer surrogate support, and validation-loss change
on the same four fixed validation batches. A global front-layer average cannot
hide a zero-coverage deep layer.

## 1. Phase A source state

Run from the repository root in PowerShell. Use the repository virtual
environment, not the Windows Store Python.

```powershell
$python = ".\.venv\Scripts\python.exe"
$protocol = "configs/protocol_v4_talif_only.yaml"
$phaseACommit = (git rev-parse HEAD).Trim()

git status --short --untracked-files=no
git rev-parse HEAD
& $python scripts/preflight.py --protocol $protocol --mode pilot
```

Required before continuing:

- `git status --short --untracked-files=no` prints nothing.
- `HEAD` is the reviewed v4 Phase A source commit.
- Pilot preflight prints `PASS`.
- `nvidia-smi` identifies an RTX 5090.
- The Python environment reports PyTorch `2.9.1+cu128` and CUDA `12.8`.

Generate the two isolated matrices only if they are absent in this checkout:

```powershell
& $python scripts/generate_run_configs.py --protocol $protocol --stage pilot
& $python scripts/generate_run_configs.py --protocol $protocol --stage formal
```

Expected output is exactly four files plus two manifests in
`configs/v4_talif_only_pilot_generated`, and twenty files plus two manifests in
`configs/v4_talif_only_generated`. The generator refuses an existing or partial
destination instead of overwriting it.

## 2. CIFAR-100 health and pilot

The next command claims seed `1975342236` before compute. Run it once only.

```powershell
& $python scripts/pilot_health_gate_v4.py `
  --protocol $protocol --dataset cifar100 --device cuda:0
```

Continue only if it prints `V4_HEALTH_PASS dataset=cifar100`. Then launch the
complete ordered C1/C2 block:

```powershell
& $python scripts/run_matrix.py `
  --protocol $protocol --allow-unfrozen-pilot --dataset cifar100 `
  --device cuda:0 --stop-on-error
```

Do not add `--condition`, `--limit`, `--experiment`, `--output-root`,
`--limit-batches`, or `--dry-run`.

## 3. CIFAR10-DVS health and pilot

The next command claims seed `1983855948` before compute. Run it once only.

```powershell
& $python scripts/pilot_health_gate_v4.py `
  --protocol $protocol --dataset cifar10dvs --device cuda:0
```

Continue only if it prints `V4_HEALTH_PASS dataset=cifar10dvs`. Then run:

```powershell
& $python scripts/run_matrix.py `
  --protocol $protocol --allow-unfrozen-pilot --dataset cifar10dvs `
  --device cuda:0 --stop-on-error
```

## 4. Aggregate pilot decision

Run this only after both two-run blocks complete:

```powershell
& $python scripts/validate_v4_pilot.py `
  --protocol $protocol `
  --config-dir configs/v4_talif_only_pilot_generated
```

The only release verdict is `V4_PILOT_VALIDATION_PASS`. `FAIL` means the v4
formal experiment is permanently blocked; `INVALID` means evidence must be
audited without rerunning or substituting either seed.

## 5. Formal freeze

Create the manifest only after aggregate PASS. The command re-runs the pilot
validator and binds the protocol, author record, Phase A commit, formal matrix,
health/pilot evidence, and all v4 gate and analysis source hashes.

```powershell
$phaseACommit = (git rev-parse HEAD).Trim()

& $python scripts/create_freeze_manifest.py `
  --project-root . `
  --protocol $protocol `
  --signoff PREREGISTRATION_SIGNOFF_V4_TALIF_ONLY.md `
  --matrix-dir configs/v4_talif_only_generated `
  --output FREEZE_MANIFEST_V4_TALIF_ONLY.json `
  --freeze-commit $phaseACommit

& $python scripts/create_freeze_manifest.py `
  --project-root . `
  --protocol $protocol `
  --signoff PREREGISTRATION_SIGNOFF_V4_TALIF_ONLY.md `
  --matrix-dir configs/v4_talif_only_generated `
  --output FREEZE_MANIFEST_V4_TALIF_ONLY.json `
  --verify

& $python scripts/preflight.py --protocol $protocol --mode full
```

Formal training is authorized only when manifest verification and full
preflight both pass.

## 6. Formal twenty-run matrix

```powershell
& $python scripts/run_matrix.py `
  --protocol $protocol --device cuda:0 --stop-on-error
```

No dataset, condition, experiment, run-count, batch-count, config, output-root,
or dry-run override is permitted. The orchestrator must select the complete
protocol-bound formal matrix and output root.

For a technical interruption, resume on the same machine and software
environment only when that run has a valid `last.pt`:

```powershell
& $python scripts/run_matrix.py `
  --protocol $protocol --device cuda:0 --stop-on-error --resume-matrix
```

Never start a fresh retry under a consumed run identity, substitute a seed, or
resume a checkpoint in a different environment.

## 7. One-time final-test access and analysis

Only after all twenty training runs and best-checkpoint audits are complete:

```powershell
& $python scripts/preflight.py --protocol $protocol --mode final-test

& $python scripts/evaluate_checkpoints.py `
  --protocol $protocol `
  --results-root results/formal_v4_talif_only `
  --config-dir configs/v4_talif_only_generated `
  --device cuda:0

& $python scripts/analyze_v4_results.py --protocol $protocol
```

The evaluator permits one committed test transaction per checkpoint. If a
transaction stops at `results_ready`, use its documented `--recover-run RUN_ID`
path; that recovery commits already-recorded results without reading the test
loader again. For an ambiguous interruption before `results_ready`, stop and
record the incident rather than repeating test access.

Reportable outputs are restricted to
`results/analysis/v4_talif_only/v4_paired_accuracy_analysis.json` and the bound
raw paired-difference CSV. Pilot results cannot enter the manuscript analysis.
