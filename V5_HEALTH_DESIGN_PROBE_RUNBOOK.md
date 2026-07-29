# Prospective v5 health-design probe runbook

This runbook is development-only. It does not define or freeze protocol v5,
does not release a pilot or formal run, and produces no manuscript-eligible
result.

## Binding stop rule for v4

Protocol v4 terminated after the consumed CIFAR-100 health FAIL recorded in
`V4_TALIF_ONLY_TERMINATION.md`. Do not rerun that seed, do not run the v4
CIFAR10-DVS health command, and do not start any v4 pilot or formal command.

The probe requires the original cloud evidence to remain at these paths:

- `results/pilot/v4_talif_only/health_cifar100_s1975342236.attempt.json`
- `results/pilot/v4_talif_only/health_cifar100_s1975342236.json`

It verifies their exact SHA-256 identities before GPU execution. It also
requires all v4 post-health artifacts to remain absent: both pilot roots and
plans, CIFAR10-DVS health evidence, aggregate validation, formal freeze,
formal results, and analysis results. CIFAR10-DVS seed `1983855948` is
`RETIRED_UNEXECUTED` and forbidden for every future purpose.

## What this probe does

For C1 and C2 on each dataset, the probe collects two independent development
trajectories:

1. A fixed real batch at base LR with optimizer warmup disabled and no
   scheduler steps. Metrics are sampled at steps 0, 80, 160, 320, and 640.
2. Eight short control-flow epochs using the reviewed formal optimizer and
   scheduler, covering TA-frozen epochs 0-4 and TA-enabled epochs 5-7.

The hard observations are finite/shared gradients, TA routing, actual parameter
updates, exact checkpoint continuation across the activation boundary, frozen
TA parameters and optimizer state before epoch 5, TA gradients and updates from
epoch 5 onward, and the TA/base LR ratio. Natural-data loss and accuracy are
descriptive only. The probe evaluates no learning-performance cutoff.

The four development seeds are deterministically recorded in
`configs/v5_health_calibration.yaml` and must be excluded from every future v5
health, pilot, and formal seed ledger.

## Cloud precheck

From the reviewed checkout, use the project virtual environment and verify the
tracked worktree is clean:

```bash
cd /hy-tmp/ta-lif-msresnet
PYTHON=/root/venvs/talif-msresnet/bin/python

git status --short --untracked-files=no
"$PYTHON" -m pip install -e .
"$PYTHON" scripts/calibrate_v5_health.py --help

sha256sum \
  results/pilot/v4_talif_only/health_cifar100_s1975342236.attempt.json \
  results/pilot/v4_talif_only/health_cifar100_s1975342236.json
```

The expected evidence hashes are:

```text
8115975674e4e1952f0573f691de887dee142018067c7388bd466220125ab6fb
547ea7b24da9976359605eb67467f89a100b482f3c2728fb2d760564d7f2a4b1
```

## Execute once per dataset

Run only one dataset at a time on the RTX 5090. The output files are
exclusive-create and can never be overwritten.

```bash
cd /hy-tmp/ta-lif-msresnet
PYTHON=/root/venvs/talif-msresnet/bin/python

"$PYTHON" scripts/calibrate_v5_health.py \
  --dataset cifar100 \
  --device cuda:0

"$PYTHON" scripts/calibrate_v5_health.py \
  --dataset cifar10dvs \
  --device cuda:0
```

Expected outputs:

- `results/development/v5_health_design_probe/cifar100_attempt01.json`
- `results/development/v5_health_design_probe/cifar10dvs_attempt01.json`

Do not automatically retry an `ERROR` or integrity anomaly. Preserve the JSON,
use a fresh attempt filename only after the cause has been reviewed, and never
change any v4 evidence file.

## Return evidence

After both commands complete, return the command output plus:

```bash
sha256sum results/development/v5_health_design_probe/*_attempt01.json

"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("results/development/v5_health_design_probe")
for path in sorted(root.glob("*_attempt01.json")):
    report = json.loads(path.read_text())
    print(path)
    print(" status:", report.get("status"))
    print(" commit:", report.get("git_commit"))
    print(" seeds:", report.get("development_seeds"))
    print(" anomalies:", report.get("integrity_anomalies"))
PY
```

Only after both artifacts are reviewed can new and disjoint v5 health, pilot,
and formal seeds be derived and the actual protocol v5 be frozen.
