# TA-LIF in Directly Trained Spiking Residual Networks

Reference implementation and reproducibility materials for the V7 frozen
mechanism study reported in the manuscript:

> Spike-History-Driven Threshold Adaptation in Spiking Residual Networks:
> A Controlled Mechanism Study

V7 evaluates the disclosed TA-LIF construction under a controlled six-condition
protocol. It is a mechanism study, not a claim that TA-LIF is a newly invented
neuron model, a state-of-the-art benchmark, or an energy-efficiency result.

## Study status

The formal experiment and one-time final-test evaluation are complete.

- Conditions: M0 standard LIF, M1 route-matched LIF, M2 shared trainable
  window, M3 time-indexed trainable bank, M4 count-history TA-LIF, and a
  protocol-matched PLIF-style control.
- Dataset: CIFAR-100, with Spiking ResNet-20 at six simulation steps.
- Design: eight paired seeds, six conditions, 48 formal training runs, and one
  committed final-test transaction per run.
- Training environment: NVIDIA GeForce RTX 5090, PyTorch `2.9.1+cu128`, CUDA
  runtime `12.8`, deterministic float32, AMP disabled.

### Final V7 results

| Condition | Accuracy (mean +/- SD) |
|---|---:|
| M0 standard LIF | 37.890 +/- 0.426% |
| M1 route-matched LIF | 43.785 +/- 0.473% |
| M2 shared trainable window | 47.003 +/- 0.504% |
| M3 time-indexed trainable bank | 45.588 +/- 0.475% |
| M4 count-history TA-LIF | 47.509 +/- 0.673% |
| PLIF-style control | 38.784 +/- 0.472% |

The primary replication contrast M4-M0 is +9.619 percentage points (95% CI
[9.018, 10.220], one-sided paired p = 1.17e-9). M4 exceeds the matched
time-indexed bank M3 by +1.921 points (Holm-adjusted p = 0.000584). Its
0.506-point advantage over the shared trainable window M2 is inconclusive
(Holm-adjusted p = 0.0606). The prespecified aggregate decision is
`partial_mechanism_support`.

## Reproducibility identity

- V7 source commit: `f7ac34c56b09201e6e85799b33a931739a2493a3`
- Protocol: `configs/protocol_v7_mechanism.yaml`
- Protocol SHA-256: `ce8240e28147a87ee76a3953f8097613ecc33d683339ad66d9e73a126104265c`
- Formal design: CIFAR-100, Spiking ResNet-20, T=6, six conditions, eight paired
  seeds, 48/48 completed runs and 48/48 final-test evaluations.
- Evidence archive SHA-256: `a199316c30186c07ce81a1ea2ae4c8ccb5353deda1163825e7fc83f051e1c5ab`
- Session-log archive SHA-256: `1402bc91450dc5dc8095c5da6382b868100e9e9b782369ae970c46104ea62934`

The V7 source commit is immutable evidence of the code used for execution. The
public release must retain it as the release target and identify the exact tag,
archive DOI/URL, and checksums explicitly.

## Repository map

- `src/talif_msresnet/`: model, neuron, training, audit, and analysis code.
- `configs/protocol_v7_mechanism.yaml`: frozen V7 protocol.
- `configs/v7_mechanism_generated/`: 48 formal run configurations.
- `configs/v7_mechanism_pilot_generated/`: six non-reportable pilot configs.
- `scripts/`: gated execution, evaluation, and analysis entry points.
- `tests/`: integrity, protocol, model, and release-flow tests.
- `V7_MECHANISM_5090_RUNBOOK.md`: exact V7 execution and recovery record.
- `PREREGISTRATION_SIGNOFF_V7_MECHANISM.md`: accountable-author freeze record.

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
or overwritten. Use the V7 evidence archive to audit the frozen protocol,
per-run manifests, checkpoints, one-time `final_test.json` records, seed-level
metrics, paired analysis, and transaction logs. See
`V7_MECHANISM_5090_RUNBOOK.md` for the exact sequence and its no-retry rules.

After downloading the release archive, verify its digest before inspection:

```text
sha256sum -c ta-lif-msresnet-v7-evidence.tar.gz.sha256
```

The archive preserves some platform-specific `/hy-tmp/...` and `/root/...`
paths as execution provenance. They are not credentials. Rewriting them would
change the frozen archive checksum.

## Historical protocols

Files for v1-v6 and the earlier C1-C4 factorial route are retained only for
incident history, auditability, and protocol provenance. They are not current
execution instructions and must not be combined with the V7 results.

The failed V6 health transaction remains historical and is not pooled with V7.
V7 is the only active formal evidence set in this release.

## Citation and license

The software is released under the MIT License; the ownership decision is
recorded in `RELEASE_LICENSE_DECISION.md`. Runtime dependencies and benchmark
datasets remain subject to their own licenses and terms; see
`THIRD_PARTY_NOTICES.md`. The V7 tag and its persistent archive DOI are recorded
in `CITATION.cff` once the public release and archive have been verified. The
older V5 release `v1.0.0` and DOI `10.5281/zenodo.21808938` are historical.
