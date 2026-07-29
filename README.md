# TA-LIF x MS-ResNet

> **Current manuscript route (2026-07-29):** the current paper remains strictly
> TA-LIF-only and C1/C2-only. MS-ResNet, C3/C4, topology effects, interactions,
> synergy, complementarity, and composability remain closed and non-reportable.
> The v3 CIFAR-100 pilot failed its prespecified scientific threshold. Under the
> no-retry v3 protocol, the remaining v3 DVS pilot, aggregate validation, freeze
> manifest, formal training, and test access are permanently blocked. The
> unconsumed v3 DVS seed `799312121` must not be run or transferred to v4.
> Protocol v4 is now frozen under the accountable-author authorization recorded
> in `PREREGISTRATION_SIGNOFF_V4_TALIF_ONLY.md`. Its isolated CIFAR-100 and
> CIFAR10-DVS health gates and four-run non-reportable pilot are authorized only
> in the exact RTX 5090 sequence in `V4_TALIF_ONLY_RUNBOOK.md`. Formal v4 remains
> blocked until both dataset pilots aggregate to PASS and
> `FREEZE_MANIFEST_V4_TALIF_ONLY.json` verifies. Preserve all historical files
> and failure evidence unchanged. The remaining v1/v2/v3 execution instructions
> in this README are historical documentation and must not launch new work.

> **Historical study status (through 2026-07-27):** formal v1 is withdrawn from reporting
> after a structural no-learning defect and a deterministic TA-backward
> performance incident were identified. Preserve its artifacts for audit only;
> do not resume its checkpoints or combine them with repaired runs. See
> `FORMAL_V1_INCIDENT.md`. Development had proceeded through the separate,
> non-reportable `configs/protocol_v2r2_seed88_of80_e120.yaml` gate. The failed
> seed-77 v2r1 gate is retained for audit and must not be rerun. No v2 formal
> protocol is frozen yet.

Reference implementation and reproducibility package for the planned
Knowledge-Based Systems experiment. It implements the controlled 2 x 2 cells:

| Condition | Neuron | Topology |
|---|---|---|
| C1 | LIF | Spiking ResNet |
| C2 | TA-LIF | Spiking ResNet |
| C3 | LIF | MS-ResNet |
| C4 | TA-LIF | MS-ResNet |

This is a reconstructed implementation from the manuscript and dissertation
description. It is not the unavailable original source code. The frozen
`configs/protocol.yaml` identifies the withdrawn v1 study and is retained only
for reproducibility and incident audit; it is not the active v2 protocol.

## Environment

Create a clean Python 3.10+ environment and install the PyTorch build matching
the target CUDA driver. From this project root:

```text
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -e ".[dev,event]"
python -m pytest -q
```

Record `python -m pip freeze`, the GPU/driver, CUDA, cuDNN, OS, compiler, and
the source commit before a reportable run.

## Protocol gate

The checked-in protocol is intentionally provisional. Before full training:

1. Review the reconstructed model against the intended equations.
2. Confirm the neuron decay/window settings.
3. Confirm basic-only versus strong static augmentation.
4. Confirm optimizer, schedule, and TA-LIF delayed-update settings.
5. Confirm CIFAR10-DVS frame conversion and 70/10/20 allocation.
6. Review and enter the proposed validation thresholds, interaction SESOI, and
   descriptive efficiency tolerances listed below; do not tune them from joint results.
7. Have Yanzhao Deng, Peng Yan, and Song Wang complete the evidence/date fields
   and checklist in `PREREGISTRATION_SIGNOFF.md`.
8. In the same reviewed change set, manually set every
   `protocol_status.confirmations` value to `true`, enter all three names and a
   timezone-aware confirmation time, and set `protocol_status.frozen: true`.
   The freeze tooling never performs these human-approval steps.

The numerical analysis contract proposed for author freeze is:

- fixed seeds: `11, 22, 33, 44, 55`;
- provisional optimizer batch size: `64` (the previous `256`/`128` values
  failed the recorded 8 GB CUDA smoke and remain documented in the pre-freeze
  hardware report);
- validation-accuracy proportions: CIFAR-100 `0.60` and CIFAR10-DVS `0.60`;
  first threshold-reaching epoch is the convergence epoch,
  and failure to reach it by epoch 240 is recorded as `converged=0` without
  excluding the run; CIFAR-10 `0.70` remains in the protocol only as an unused
  compatibility value;
- interaction smallest effect size of interest: `0.50` percentage points;
- confirmatory groups: CIFAR-100/depth-20/T=6 and
  CIFAR10-DVS/depth-20/T=10, each with C1-C4 and all five fixed seeds;
- excluded scope: no CIFAR-10 group, no depth-56 group, and no E4 T=2/T=4
  group; this design does not support depth-robustness or time-step-sensitivity
  claims;
- descriptive engineering tolerances: B=1 latency `10%`, B=128 latency `10%`,
  peak training memory `5%`, and explicitly modeled energy `10%`;
- complete-seed-block bootstrap: `10000` resamples with seed `20260719` and a
  two-sided 95% percentile interval, used as sensitivity analysis only; cellwise
  resampling is forbidden.

These values are a prespecified draft until the authors approve them and the
protocol hash is frozen. The efficiency percentages are engineering tolerances,
not formal noninferiority margins.

The current energy-model status is `not_assessed`. To report modeled energy,
freeze a repository-relative constants JSON, its SHA-256, and its cited source
under `benchmark.energy_model` before the first reportable run. A CLI argument
cannot introduce or replace energy constants after protocol freeze.

The selected provisional static package is random crop, horizontal flip,
normalization, no AutoAugment, no CutMix, and no label smoothing. These choices
are explicit config fields and must not be changed after the first reportable run.

### Two-stage freeze

The signed record cannot contain its own hash or the commit that contains it,
and a pre-generation record cannot contain the later matrix hash. The repository
therefore uses two stages:

1. **Phase A (human/source):** complete the author record and YAML fields
   manually, review the exact source, and commit them together. Do not include
   provisional generated configs in this commit. Keep the full 40-character
   commit ID.
2. **Phase B (generated artifacts):** with `HEAD` still at that clean commit,
   generate a fresh matrix and create the independent manifest. The manifest
   binds the Phase A commit, signed-record and protocol hashes, and every
   generated artifact without modifying any signed input.

Use a new or empty `configs/generated` directory. If `origin` is not yet set,
pass the intended remote with `--repository-url`.

```powershell
$freezeCommit = (git rev-parse HEAD).Trim()
python scripts/generate_run_configs.py --output configs/generated
python scripts/create_freeze_manifest.py --freeze-commit $freezeCommit
python scripts/create_freeze_manifest.py --verify
```

Creation fails closed when the author record is incomplete, the protocol is not
already frozen, `HEAD` differs from the explicit commit, tracked source is
dirty, unexpected untracked source exists, any of the 40 generated configs
differs, or either generated matrix manifest differs.

An existing `FREEZE_MANIFEST.json` is never overwritten. The command prints the
manifest's SHA-256 for an external archive checksum list; the manifest does not
recursively contain its own hash.

Check the gate:

```text
python scripts/preflight.py --mode smoke
python scripts/preflight.py --mode full
```

`--mode full` must be blocked until the protocol, CIFAR10-DVS data, and Phase B
`FREEZE_MANIFEST.json` are ready. Once the author fields are frozen, the full
preflight and all formal execution/benchmark/diagnostic entry points verify the
manifest and the executable source against the Phase A commit; a missing or
stale manifest cannot be bypassed by invoking a lower-level script directly.

## CIFAR10-DVS preparation

The following command reflects the provisional protocol. Change both the
command and YAML together before freezing if the authors choose other values.

```text
python scripts/prepare_cifar10dvs.py `
  --data-root data/cifar10dvs/raw `
  --output data/cifar10dvs/processed `
  --time-bins 10 --height 48 --width 48 `
  --test-fraction 0.2 --split-seed 2024
```

The converter uses a stratified trainval/test split, writes one tensor per
sample plus `index.csv`, preserves source indices, and records SHA-256 hashes.
Training creates a second stratified train/validation manifest inside trainval.
Existing output directories are never overwritten.

For the audited pre-extracted AEDAT4 conversion currently stored under
`data/cifar10dvs/raw/CIFAR10DVS`, use the explicit alternative source mode:

```text
python scripts/prepare_cifar10dvs.py `
  --aedat4-root data/cifar10dvs/raw/CIFAR10DVS `
  --output data/cifar10dvs/processed `
  --time-bins 10 --height 48 --width 48 `
  --test-fraction 0.2 --split-seed 2024
```

This mode never invokes Tonic's downloader. It accepts exactly the 10 standard
class directories, exactly 1,000 numerically named AEDAT4 samples per class,
and a `README.txt`. It writes `source_manifest.json` with the ordered source
inventory, file hashes, README hash, and an explicit disclosure that byte
identity with the official Figshare `CIFAR10DVS.zip` was not verified. The
official-archive mode and the third-party AEDAT4 conversion are not treated as
interchangeable provenance claims.

After conversion, run the mandatory full scan:

```text
python scripts/verify_cifar10dvs.py --root data/cifar10dvs/processed
```

Only a successful scan writes
`data/cifar10dvs/processed/verification_receipt.json`. Pilot/full/final-test
preflight verifies the receipt and its current data hashes; a missing or stale
receipt blocks execution.

## Generate and smoke-test configs

```text
python scripts/generate_run_configs.py --dry-run
python scripts/generate_run_configs.py --output configs/generated
python scripts/run_matrix.py `
  --config configs/generated/E1_cifar100_d20_t6_C1_s11.yaml `
  --dry-run --limit-batches 2 --device cpu
```

The matrix contains 40 unique E1 training jobs: two retained dataset/depth/
time-step groups x four conditions x five fixed seeds. No E4 configurations
are generated.

`configs/smoke_generated` contains pre-reduction engineering artifacts and is
not a source of formal 40-run configurations. Do not substitute it for the
fresh `configs/generated` directory bound by the Phase B manifest.

The seed-77 v2r1 health gate is a retained failed engineering record and is no
longer an active pilot identity. Its protocol, report, and benchmark evidence
must not be overwritten or rerun. The v2r2 seed-88 C1-C4 block in
`configs/protocol_v2r2_seed88_of80_e120.yaml` is also historical and must not be
rerun. `V2R2_PILOT_RUNBOOK.md` is retained only to document that closed route.

## Full validation-only training

After both freeze stages, run full preflight and then the matrix. Full preflight
must pass before formal execution, and training never constructs an official
test loader.

```text
python scripts/preflight.py --mode full
python scripts/create_freeze_manifest.py --verify
python scripts/run_matrix.py --config-dir configs/generated `
  --output-root results/runs --device cuda --stop-on-error
```

Each run writes `best.pt`, `last.pt`, events, failure artifacts, the scientific
config hash, the execution hash, the shared-weight hash, a canonical training
hardware/software/device/precision identity and SHA-256, and seed metrics.
Existing completed run directories and matrix logs are not silently overwritten.

After an interruption, continue the matrix explicitly:

```text
python scripts/run_matrix.py --config-dir configs/generated `
  --output-root results/runs --device cuda --resume-matrix --stop-on-error
```

`--resume-matrix` verifies each planned scientific hash, skips completed runs,
and resumes incomplete runs only from `last.pt`. Every process invocation writes
new `attempt_NNN.stdout.log` and `attempt_NNN.stderr.log` files. If a failed run
has scientific artifacts but no `last.pt`, the orchestrator blocks it; archive
that whole run directory and document the attempt before an explicit restart.

Resume an interrupted run directly from its `last.pt`; `best.pt` and `failed.pt`
are rejected because they are not safe epoch boundaries. Execution-only fields
do not change the scientific hash, and dry-run checkpoints are rejected:

```text
python -m talif_msresnet.train `
  --config configs/generated/E1_cifar100_d20_t6_C1_s11.yaml `
  --resume results/runs/E1_cifar100_d20_t6_C1_s11/last.pt `
  --output-dir results/runs --device cuda
```

## Fixed validation batches

Export 128 validation samples for benchmark activity accounting. The E3 script
uses the first 8 samples by default to control Jacobian memory.

```text
python scripts/export_representative_batch.py `
  --config configs/generated/E1_cifar100_d20_t6_C1_s11.yaml `
  --output results/representative/cifar100.pt --batch-size 128
python scripts/export_representative_batch.py `
  --config configs/generated/E1_cifar10dvs_d20_t10_C1_s11.yaml `
  --output results/representative/cifar10dvs.pt --batch-size 128
```

These commands never access a test split.

## E3 diagnostics

Run both retained groups across C1-C4 and all five seeds:

```text
python scripts/run_diagnostics.py --dataset cifar100 --experiment E1 --depth 20 `
  --time-steps 6 --input-batch results/representative/cifar100.pt --device cuda
python scripts/run_diagnostics.py --dataset cifar10dvs --experiment E1 --depth 20 `
  --time-steps 10 --input-batch results/representative/cifar10dvs.pt --device cuda
```

Outputs include block-input gradient norms, cross-block CV, local block
Jacobian spectral moments, layer firing rates, membrane quantiles, TA-LIF
windows, and spike-count occupancy. The Jacobian is explicitly a local
state-conditioned derivative, not an end-to-end temporal Jacobian. Scalar E3
summaries are merged into the matching seed row; long tables remain auditable.

## CUDA benchmark

Benchmark both retained groups separately:

```text
python scripts/run_benchmarks.py --dataset cifar100 --experiment E1 --depth 20 `
  --time-steps 6 --input-tensor results/representative/cifar100.pt `
  --output results/benchmark_results.csv --device cuda
python scripts/run_benchmarks.py --dataset cifar10dvs --experiment E1 --depth 20 `
  --time-steps 10 --input-tensor results/representative/cifar10dvs.pt `
  --output results/benchmark_results.csv --device cuda
```

Latency uses synchronized CUDA events at B=1 and B=128. Activity, SyOPs, MACs,
threshold-bank accesses, and count updates require the representative tensor.
Energy stays blank while `benchmark.energy_model.status` is `not_assessed`.
When a sourced constants JSON is bound in the protocol before freeze, the wrapper
uses that exact path/hash/source; `--energy-constants` may only repeat the same
frozen path. Modeled energy is never labeled as measured energy.

Efficiency overhead is paired by seed and topology within each retained group:
C2/C1 and C4/C3. Each metric is analyzed on the log-ratio scale and reported as
geometric percentage overhead with a paired two-sided 95% t confidence interval
for the mean log ratio. The independent unit is the five-seed checkpoint pair,
not the 100 CUDA timing
iterations. Firing rate is descriptive and has no one-directional tolerance.
For each estimable metric, the frozen reporting rule is:

- point estimate above the margin: `exceeds_tolerance`;
- point estimate at or below the margin but CI upper bound above it:
  `within_tolerance_uncertain`;
- point estimate and CI upper bound both at or below the margin:
  `supported_within_tolerance`.

For the row-level summary, known exceedance precedes uncertain, unavailable,
and supported states, so missing energy cannot hide a measured latency breach.
Peak training memory is assessed only when every selected training row has the
same verified hardware/software/device/precision identity; mixed environments
block Table 6 rather than being treated as comparable.

Missing sourced energy constants produce `not_assessed`; they are never imputed.

## One-time final test

Only after all 40 `best.pt` files pass the freeze audit:

```text
python scripts/preflight.py --mode final-test
python scripts/evaluate_checkpoints.py --results-root results/runs `
  --config-dir configs/generated --device cuda
```

The evaluator constructs only the independent test loader, verifies every
checkpoint protocol/config hash first, requires exactly 40 unique consolidated
seed rows, and cross-checks protocol, split, shared-weight, and test fields
against each per-run manifest. Pre-populated unaudited test fields block test
access. The evaluator writes an immutable
`final_test.json`. A restart skips an already completed identical checkpoint;
a changed checkpoint or pre-populated unaudited test metric is rejected.

Before test access, each run creates `final_test.in_progress.json` and
`final_test.lock`. The atomic `final_test.json` is the commit record. If the
process stops and the journal says `stage: results_ready`, commit the recorded
result without reading test data again:

```text
python scripts/evaluate_checkpoints.py --results-root results/runs `
  --config-dir configs/generated --recover-run RUN_ID
```

For `prepared` or `test_access_started`, whether the test set affected execution
is ambiguous. The evaluator will not retry automatically. Do not delete the
journal or lock; preserve them, document the incident, and obtain an explicit
author decision before any manual recovery.

## Tables 4-6

```text
python scripts/analyze_results.py --input results/runs `
  --benchmark results/benchmark_results.csv `
  --output results/statistics
```

The confirmatory unit is the seed-level direct difference-in-differences,
`(C4-C3)-(C2-C1)`, in each of the two fixed primary groups. A one-sample t test
on the five seed contrasts tests
`H0: delta <= 0.50 pp` against `H1: delta > 0.50 pp`; the two one-sided raw
p-values are Holm-adjusted as one family at FWER 0.05, and two-sided 95%
confidence intervals are also reported. The effect-coded model
`accuracy ~ neuron * topology + C(seed)` is secondary/descriptive. The fixed
10,000-resample bootstrap resamples complete seed blocks with seed `20260719`
and is sensitivity-only.

Every primary group must contain C1-C4 for all five fixed seeds. A missing or
failed cell blocks confirmatory inference. Failed attempts stay in the manifest;
accuracy outliers are not deleted, and seeds are not replaced or changed.
Table 5 requires complete, protocol-homogeneous E3 diagnostics for both retained
groups and refuses incomplete or mixed blocks. Table 6 keeps missing
efficiency measurements explicit rather than imputing them and reports the
paired engineering-tolerance states defined above.

## Release blockers

Before repository release, resolve `LICENSE_CHOICE_REQUIRED.md`, complete
`THIRD_PARTY_NOTICES.md`, create `CITATION.cff`, archive the exact frozen
configs/results/checksums/environment, and insert the repository commit and
archive DOI into the manuscript.
