# TA-LIF in Directly Trained Spiking Residual Networks

Reference implementation and reproducibility materials for the frozen paired
evaluation reported in the manuscript:

> TA-LIF in Directly Trained Spiking Residual Networks: A Frozen Paired
> Evaluation on CIFAR-100 and CIFAR10-DVS

The current V5 study is strictly a C1/C2 comparison in one directly trained
20-layer spiking residual network. It does not evaluate MS-ResNet topology,
neuron-by-topology interactions, energy, latency, memory, SynOps, or isolated
TA-LIF component effects.

## Study status

The formal experiment and one-time final-test evaluation are complete.

- C1: fixed LIF operator package.
- C2: complete TA-LIF operator package, including the trainable threshold bank,
  TA-specific optimizer group, delayed activation schedule, and implemented
  gradient route.
- Datasets: CIFAR-100 as the sole confirmatory primary dataset and CIFAR10-DVS
  as a prespecified external-validity replication.
- Design: five paired seeds per dataset, two conditions, 20 formal training
  runs, 120 epochs per run, and one committed final-test transaction per run.
- Training environment: NVIDIA GeForce RTX 5090, PyTorch `2.9.1+cu128`, CUDA
  runtime `12.8`, deterministic float32, AMP disabled.

### Final paired results

| Dataset | Role | Pairs | Mean C2-C1 | 95% CI | One-sided paired t-test |
|---|---|---:|---:|---:|---:|
| CIFAR-100 | Sole confirmatory primary | 5 | +9.336 pp | [8.859, 9.813] pp | p = 3.44e-7 |
| CIFAR10-DVS | Prespecified external validity | 5 | -1.890 pp | [-5.187, 1.407] pp | p = 0.9066 |

The CIFAR-100 result supports a fixed-budget accuracy improvement under the
implemented package contrast. None of the ten CIFAR-100 formal runs reached the
prespecified 0.60 validation threshold by epoch 120, so the result is not framed
as asymptotic or fully converged superiority. CIFAR10-DVS did not replicate a
positive effect.

## Reproducibility identity

- Formal experiment commit:
  `b238baf0d807dd0916e998fbc47298d3c2f37772`
- Protocol: `configs/protocol_v5_talif_only.yaml`
- Protocol SHA-256:
  `a5b2a664de5142438061f479a37f3b55401499104abb28ee3cf1d674ec1d4527`
- Freeze-manifest SHA-256:
  `242d8633e0da1de082029713bf4e469e659f9731d10f1b0615196c4bb3d5c13b`
- Training-environment SHA-256:
  `f3b837c1554615bb97af91183ec450bf8f4bb43610ad47123aa1f161791d4fb6`
- Clean formal evidence archive:
  `v5-formal-reproducibility-clean-20260803.tar.gz`
- Clean archive SHA-256:
  `a354dd7c632b07885f75f7b206e1ea5336d94258005c42e7f927f88ad3913ece`

The experiment commit is immutable evidence of the code used for execution.
The public release commit may contain only later administrative additions such
as `LICENSE`, citation metadata, and this release README; it must retain the
experiment commit as an ancestor and identify both commits explicitly.

## Repository map

- `src/talif_msresnet/`: model, neuron, training, audit, and analysis code.
- `configs/protocol_v5_talif_only.yaml`: frozen V5 protocol.
- `configs/v5_talif_only_generated/`: 20 formal run configurations.
- `configs/v5_talif_only_pilot_generated/`: four non-reportable pilot configs.
- `scripts/`: gated execution, evaluation, and analysis entry points.
- `tests/`: integrity, protocol, model, and release-flow tests.
- `V5_TALIF_ONLY_RUNBOOK.md`: exact historical execution and recovery record.
- `PREREGISTRATION_SIGNOFF_V5_TALIF_ONLY.md`: accountable-author freeze record.

The Git repository intentionally excludes benchmark data, checkpoints, formal
results, development/pilot outputs, virtual environments, and local machine
snapshots. The formal evidence is distributed as a separately checksummed
release archive.

## Installation

Use Python 3.10+ and install the PyTorch build appropriate for the local
CUDA driver before installing this package:

```text
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -e ".[dev,event]"
python -m pytest -q
```

The formal environment identity above is required to reproduce the recorded
runtime exactly. Other environments may be used for code inspection and tests
but are not interchangeable with the formal evidence.

## Data boundary

CIFAR-100 and CIFAR10-DVS are third-party benchmark datasets and are not
redistributed. Users must obtain them from their authorized sources and follow
their original terms. CIFAR10-DVS preparation and provenance verification are
implemented by:

```text
python scripts/prepare_cifar10dvs.py --help
python scripts/verify_cifar10dvs.py --help
```

The archived formal results are derived non-data evidence, not copies of the
source datasets.

## Formal evidence use

The consumed health, pilot, formal, and final-test identities must not be rerun
or overwritten. Use the clean archive to audit the frozen protocol, per-run
manifests, checkpoints, one-time `final_test.json` records, seed-level metrics,
paired analysis, and transaction logs. See `V5_TALIF_ONLY_RUNBOOK.md` for the
historical sequence and its no-retry rules.

After downloading the release archive, verify its digest before inspection:

```text
sha256sum -c v5-formal-reproducibility-clean-20260803.tar.gz.sha256
```

The archive preserves some platform-specific `/hy-tmp/...` and `/root/...`
paths as execution provenance. They are not credentials. Rewriting them would
change the frozen archive checksum.

## Historical protocols

Files for v1-v4 and the earlier C1-C4 factorial route are retained only for
incident history, auditability, and protocol provenance. They are not current
execution instructions and must not be combined with the V5 results.

## Citation and license

The software is released under the MIT License; the ownership decision is
recorded in `RELEASE_LICENSE_DECISION.md`. Runtime dependencies and benchmark
datasets remain subject to their own licenses and terms; see
`THIRD_PARTY_NOTICES.md`. Release `v1.0.0` and the assigned archive DOI
`10.5281/zenodo.21808938` are recorded in `CITATION.cff`.
