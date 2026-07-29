# TA-LIF-only v5 RTX 5090 execution runbook

This is the only authorized execution order for protocol v5. It applies only
to the reviewed v5 release commit and its complete-history bundle. Do not use
these commands against an earlier checkout: the release must contain the v5
health gate, pilot validator, trainer/orchestrator bindings, freeze gate,
one-time evaluator, analysis code, and the two generated matrices.

Development probes, health artifacts, and pilots are non-reportable. Preserve
every receipt, report, checkpoint, event log, and failed-attempt record.

## Permanent exclusions and retired work

Protocol v4 is permanently terminated. Its CIFAR-100 health seed `1975342236`
was consumed by a valid `FAIL`; its CIFAR10-DVS seed `1983855948` is
`RETIRED_UNEXECUTED`. Never run a v4 health, pilot, or formal command, and never
edit, replace, or delete the v4 receipt, failure report, or
`V4_TALIF_ONLY_TERMINATION.md`.

The two v5 development-only design probes have already completed once at
commit `609a505445d68e9961778999b5f7a4bc79cc83a8`:

| Dataset | Canonical result | SHA-256 |
|---|---|---|
| CIFAR-100 | `results/development/v5_health_design_probe/cifar100_attempt01.json` | `421dd0f1f08a7cac631df23d23815632acdb58eac6d654a2c8cd52383aab5eaa` |
| CIFAR10-DVS | `results/development/v5_health_design_probe/cifar10dvs_attempt01.json` | `b3837bfb98948edb180f133a8f5f23bc2ab6e1d764c71dd81a4f987057afefd5` |

Do not run `scripts/calibrate_v5_health.py` again. The development seeds
`1515323759`, `214400889`, `117569744`, and `563701053` are permanently
excluded from health, pilot, formal, and bootstrap roles. Development loss and
accuracy are descriptive only and must never enter manuscript results or a
confirmatory analysis.

## Frozen scope and release order

- Protocol: `configs/protocol_v5_talif_only.yaml`
- Protocol hash: `a5b2a664de5142438061f479a37f3b55401499104abb28ee3cf1d674ec1d4527`
- Conditions: C1 LIF and C2 TA-LIF only
- Datasets: CIFAR-100 primary and CIFAR10-DVS prespecified replication
- Target runtime: RTX 5090, PyTorch `2.9.1+cu128`, CUDA runtime `12.8`,
  deterministic float32, AMP disabled
- Health: two dataset-specific one-shot implementation gates, with no
  loss/accuracy performance threshold
- Pilot: four non-reportable full 120-epoch runs, one C1/C2 pair per dataset
- Formal: exactly twenty full 120-epoch runs, five paired C1/C2 blocks per
  dataset
- Test access: one transaction per audited formal `best.pt`, after all twenty
  training runs pass the frozen audit
- Analysis: one isolated v5 paired analysis after final-test completion

The mandatory gate sequence is:

`reviewed source -> CIFAR-100 health PASS -> CIFAR-100 C1/C2 pilot -> CIFAR10-DVS health PASS -> CIFAR10-DVS C1/C2 pilot -> aggregate pilot PASS -> formal freeze PASS -> 20 formal runs -> final-test preflight -> one-time checkpoint evaluation -> v5 analysis`

Do not skip, reorder, or run later stages speculatively.

## Frozen seed roles

Health and pilot seeds are deliberately different. A health command consumes
only its health seed; it never authorizes reuse of that seed for training.

| Role | CIFAR-100 | CIFAR10-DVS |
|---|---|---|
| One-shot health | `897211180` | `1280028821` |
| Non-reportable pilot | `1968690143` | `1721988285` |
| Formal | `2141022414, 252611743, 1147962214, 931150728, 2106521224` | `157805125, 2131477013, 1311303835, 417519743, 2086335735` |
| Bootstrap | `894719475` | `159287562` |

Any health `PASS`, `FAIL`, `ERROR`, or interruption after its exclusive attempt
receipt exists consumes that health seed. Never retry it. A failed pilot run
also cannot be restarted fresh under the same identity. A technical
interruption may continue only from a valid `last.pt` on the identical machine
and software environment.

## 1. Reviewed cloud source and read-only checks

Run from the repository root on the RTX 5090 Ubuntu instance. Use the fixed
interpreter path throughout every stage.

```bash
cd /hy-tmp/ta-lif-msresnet

PYTHON=/root/venvs/talif-msresnet/bin/python
PROTOCOL=configs/protocol_v5_talif_only.yaml
PILOT_MATRIX=configs/v5_talif_only_pilot_generated
FORMAL_MATRIX=configs/v5_talif_only_generated
PHASE_A_COMMIT="$(git rev-parse HEAD)"

git status --short --untracked-files=no
git rev-parse HEAD
"$PYTHON" -m pip install --no-deps -e .
"$PYTHON" - <<'PY'
from pathlib import Path
import torch
import talif_msresnet.config as config

expected = Path.cwd().resolve() / "src" / "talif_msresnet" / "config.py"
observed = Path(config.__file__).resolve()
print("CONFIG_MODULE", observed)
print("TORCH", torch.__version__)
print("CUDA_RUNTIME", torch.version.cuda)
print("CUDA_AVAILABLE", torch.cuda.is_available())
print("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if observed != expected:
    raise SystemExit(f"reviewed checkout is shadowed: expected {expected}, got {observed}")
if torch.__version__ != "2.9.1+cu128" or torch.version.cuda != "12.8":
    raise SystemExit("frozen PyTorch/CUDA identity mismatch")
if not torch.cuda.is_available() or "RTX 5090" not in torch.cuda.get_device_name(0):
    raise SystemExit("frozen RTX 5090 runtime is unavailable")
PY

sha256sum \
  results/development/v5_health_design_probe/cifar100_attempt01.json \
  results/development/v5_health_design_probe/cifar10dvs_attempt01.json

"$PYTHON" scripts/generate_run_configs.py \
  --protocol "$PROTOCOL" --stage pilot --dry-run
"$PYTHON" scripts/generate_run_configs.py \
  --protocol "$PROTOCOL" --stage formal --dry-run
"$PYTHON" scripts/preflight.py --protocol "$PROTOCOL" --mode pilot
```

Required before any health seed is claimed:

- Tracked status is clean; untracked frozen evidence may exist.
- `HEAD` is the reviewed release commit and remains unchanged through health,
  pilots, freeze, formal training, final-test, and analysis.
- The imported package resolves to this checkout's `src` directory.
- The two development files have exactly the SHA-256 values shown above.
- Dry-run validation reports exactly 4 pilot configurations and 20 formal
  configurations. Both generated matrix directories and both manifests are
  already present in the release; do not regenerate or hand-edit them on the
  cloud host.
- Pilot preflight prints `PASS`.
- Neither canonical v5 health receipt nor health output exists before its
  corresponding one-shot command.

If any prerequisite fails, stop before running a health command. A blocked
health invocation is not permission to delete evidence or retry automatically.

## 2. CIFAR-100 one-shot health

The following command exclusively claims health seed `897211180`. Run it once
only:

```bash
"$PYTHON" scripts/pilot_health_gate_v5.py \
  --protocol "$PROTOCOL" \
  --config-dir "$PILOT_MATRIX" \
  --dataset cifar100 \
  --device cuda:0
```

Continue only if it exits zero and prints
`V5_HEALTH_GATE_PASS dataset=cifar100`. The canonical non-reportable evidence
is:

- `results/pilot/v5_talif_only/health_cifar100_s897211180.attempt.json`
- `results/pilot/v5_talif_only/health_cifar100_s897211180.json`

The health gate evaluates deterministic mechanism and numerical integrity,
including finite shared gradients, C2 TA routing and parameter updates,
absolute warmup LR, absent TA optimizer state while frozen, and exact
`last.pt` continuation across the activation boundary. C1 must report
`ta_enabled=false` throughout. C2 must report `ta_enabled=false` for zero-based
epochs 0-4 and `ta_enabled=true` at zero-based epoch 5; in one-based progress
text, activation therefore first appears at epoch 6. This is expected behavior.

If the verdict is not `PASS`, or execution is interrupted after the receipt is
created, protocol v5 is blocked. Preserve both artifacts, do not retry health,
and do not consume the pilot seed.

## 3. CIFAR-100 full pilot pair

Only after the CIFAR-100 health `PASS`, run the complete ordered C1/C2 pilot.
Both runs use the distinct pilot seed `1968690143` and must complete all 120
epochs:

```bash
"$PYTHON" scripts/run_matrix.py \
  --protocol "$PROTOCOL" \
  --allow-unfrozen-pilot \
  --dataset cifar100 \
  --device cuda:0 \
  --stop-on-error
```

Do not add `--config`, `--config-dir`, `--output-root`, `--pilot-plan`,
`--condition`, `--experiment`, `--limit`, `--limit-batches`, or `--dry-run`.
The orchestrator must bind the exact matrix, health report, pilot plan, output
root, data identity, and runtime identity itself.

## 4. CIFAR10-DVS one-shot health

Run this only after the CIFAR-100 pilot pair has completed. It exclusively
claims health seed `1280028821`:

```bash
"$PYTHON" scripts/pilot_health_gate_v5.py \
  --protocol "$PROTOCOL" \
  --config-dir "$PILOT_MATRIX" \
  --dataset cifar10dvs \
  --device cuda:0
```

Continue only if it exits zero and prints
`V5_HEALTH_GATE_PASS dataset=cifar10dvs`. Its canonical evidence is:

- `results/pilot/v5_talif_only/health_cifar10dvs_s1280028821.attempt.json`
- `results/pilot/v5_talif_only/health_cifar10dvs_s1280028821.json`

The same no-retry and TA-activation rules from the CIFAR-100 health block
apply. Any non-PASS verdict or post-receipt interruption blocks v5.

## 5. CIFAR10-DVS full pilot pair

Only after the CIFAR10-DVS health `PASS`, run both 120-epoch runs with the
distinct pilot seed `1721988285`:

```bash
"$PYTHON" scripts/run_matrix.py \
  --protocol "$PROTOCOL" \
  --allow-unfrozen-pilot \
  --dataset cifar10dvs \
  --device cuda:0 \
  --stop-on-error
```

Use no selection, path, run-count, batch-count, or dry-run override.

For a genuine technical interruption in either pilot block, preserve the
attempt and continue only on the same environment from a valid `last.pt` by
adding `--resume-matrix` to that exact dataset command. Never launch a fresh
retry after an attempt has started.

## 6. Aggregate pilot decision

Run the aggregate validator only after all four pilot runs have completed all
120 epochs:

```bash
"$PYTHON" scripts/validate_v5_pilot.py \
  --protocol "$PROTOCOL" \
  --config-dir "$PILOT_MATRIX"
```

The canonical output is
`results/pilot/v5_talif_only/validation.json`. The only release verdict is
`V5_PILOT_VALIDATION_PASS`.

The validator requires, independently for C1 and C2:

- CIFAR-100: best validation accuracy at least `0.30`, final ten-epoch mean at
  least `0.25`, late/best ratio at least `0.80`, and post-warmup residual
  gradient coverage at least `0.95`.
- CIFAR10-DVS: best validation accuracy at least `0.50`, final ten-epoch mean
  at least `0.45`, late/best ratio at least `0.80`, and post-warmup residual
  gradient coverage at least `0.95`.

`FAIL` terminates v5 and permanently blocks formal execution. `INVALID` or
`ERROR` requires evidence audit; it does not authorize a rerun, seed
substitution, artifact deletion, or manual correction of the validation JSON.

## 7. Formal freeze

Create the freeze manifest only after aggregate `PASS`. `PHASE_A_COMMIT` must
still equal the reviewed release `HEAD`. The command revalidates the health and
pilot evidence and binds the protocol, author record, exact formal matrix,
runtime/gate sources, pilot validation, repository identity, and commit.

```bash
test "$(git rev-parse HEAD)" = "$PHASE_A_COMMIT"
git status --short --untracked-files=no

"$PYTHON" scripts/create_freeze_manifest.py \
  --project-root . \
  --protocol "$PROTOCOL" \
  --signoff PREREGISTRATION_SIGNOFF_V5_TALIF_ONLY.md \
  --matrix-dir "$FORMAL_MATRIX" \
  --output FREEZE_MANIFEST_V5_TALIF_ONLY.json \
  --freeze-commit "$PHASE_A_COMMIT"

"$PYTHON" scripts/create_freeze_manifest.py \
  --project-root . \
  --protocol "$PROTOCOL" \
  --signoff PREREGISTRATION_SIGNOFF_V5_TALIF_ONLY.md \
  --matrix-dir "$FORMAL_MATRIX" \
  --output FREEZE_MANIFEST_V5_TALIF_ONLY.json \
  --verify

"$PYTHON" scripts/preflight.py --protocol "$PROTOCOL" --mode full
```

Do not recreate, overwrite, or edit an existing freeze manifest. Formal
training is authorized only when manifest verification and full preflight both
pass in the same frozen environment.

## 8. Formal twenty-run matrix

Run the complete matrix without filters or overrides:

```bash
"$PYTHON" scripts/run_matrix.py \
  --protocol "$PROTOCOL" \
  --device cuda:0 \
  --stop-on-error
```

This must select exactly twenty runs: five CIFAR-100 C1/C2 pairs and five
CIFAR10-DVS C1/C2 pairs. Do not use a dataset, condition, experiment,
run-count, batch-count, config, matrix, output-root, pilot, or dry-run override.

For a technical interruption, use only the same host, checkout, Python/CUDA
environment, and valid per-run `last.pt`:

```bash
"$PYTHON" scripts/run_matrix.py \
  --protocol "$PROTOCOL" \
  --device cuda:0 \
  --stop-on-error \
  --resume-matrix
```

Never restart a consumed formal run identity from scratch, substitute a seed,
resume cross-environment, exclude an outlier, or analyze an incomplete pair.

## 9. One-time final-test access

Do not run this section until all twenty formal runs and all twenty `best.pt`
checkpoints pass the frozen audit. Training, health, and pilot stages must not
access the test set.

```bash
"$PYTHON" scripts/preflight.py --protocol "$PROTOCOL" --mode final-test

"$PYTHON" scripts/evaluate_checkpoints.py \
  --protocol "$PROTOCOL" \
  --results-root results/formal_v5_talif_only \
  --config-dir "$FORMAL_MATRIX" \
  --device cuda:0
```

The evaluator permits one committed test transaction per checkpoint. It is
not a tunable or repeatable evaluation pass. If a transaction stops after its
journal reaches `results_ready`, commit the already-recorded result without
touching the test loader again:

```bash
"$PYTHON" scripts/evaluate_checkpoints.py \
  --protocol "$PROTOCOL" \
  --results-root results/formal_v5_talif_only \
  --config-dir "$FORMAL_MATRIX" \
  --device cuda:0 \
  --recover-run RUN_ID
```

For an ambiguous interruption before `results_ready`, stop, preserve the
transaction journal, document the incident, and obtain an accountable-author
decision. Never rerun the test loader speculatively.

## 10. One-time v5 analysis

After all twenty final-test transactions are complete, run the isolated
prespecified analysis once:

```bash
"$PYTHON" scripts/analyze_v5_results.py --protocol "$PROTOCOL"
```

The analyzer refuses to overwrite existing outputs. Reportable analysis
artifacts are restricted to:

- `results/analysis/v5_talif_only/v5_paired_accuracy_analysis.json`
- `results/analysis/v5_talif_only/v5_paired_seed_differences.csv`

No health, development, or pilot metric may enter these files or the manuscript
results. CIFAR-100 is the single confirmatory primary dataset; CIFAR10-DVS is
the prespecified replication and cannot rescue a failed primary result.
