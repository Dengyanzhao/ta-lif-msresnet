# Prespecified protocol sign-off record

Status: **SIGNED - AUTHOR APPROVALS COMPLETE; PROTOCOL FROZEN**

This record accompanies `configs/protocol.yaml`. The numerical choices below
are author-approved and frozen in the working tree. Formal C1-C4 training
remains blocked until the authors commit this signed record and frozen protocol
as Phase A, then generate a fresh configuration matrix and complete the
independent Phase B manifest.

## Study identity

- Working study title: TA-LIF x MS-ResNet controlled factorial evaluation
- Implementation status: reconstructed reference implementation; not the
  unavailable dissertation source code
- Intended venue: Knowledge-Based Systems
- Authors expected to review: Yanzhao Deng; Peng Yan; Song Wang
- Accountable protocol confirmer(s): `Yanzhao Deng; Peng Yan; Song Wang`
- Date on which joint confirmatory outcomes were first accessible, if any:
  `NONE as of 2026-07-23T17:09:00+08:00`

## Frozen numerical protocol

- Seeds: `11, 22, 33, 44, 55`; every seed is a complete C1-C4 block.
- Optimizer batch size: `64`. A pre-freeze CUDA smoke found `256` and `128`
  out of memory on the recorded 8 GB RTX 4070 Laptop; `64` completed the
  tested static and DVS-shaped C2/C4 batches. This is a provisional hardware
  feasibility decision and requires explicit author approval.
- Confirmatory groups: CIFAR-100/depth-20/T=6 and
  CIFAR10-DVS/depth-20/T=10. Each group contains C1-C4 for all five fixed
  seeds, giving 40 planned training runs.
- CIFAR-10, depth-56, and E4 T=2/T=4 experiments are outside the revised
  confirmatory scope. This study will not claim depth robustness or time-step
  sensitivity from the 40-run design.
- Validation convergence thresholds are proportions: CIFAR-100 `0.60` and
  CIFAR10-DVS `0.60`. The YAML retains CIFAR-10 `0.70` for repository
  compatibility only; no confirmatory configuration uses that value.
- Convergence epoch: first validation epoch at or above the dataset threshold.
  A run that does not reach the threshold by epoch 240 is nonconverged but is
  not excluded from the accuracy analysis.
- Accuracy interaction SESOI: `0.50` percentage points.
- Primary interaction estimand: seed-level
  `(C4-C3)-(C2-C1)` difference in differences, in percentage points.
- Primary null/alternative: one-sided `H0: delta <= 0.50 pp` versus
  `H1: delta > 0.50 pp`, blocked by seed, alpha `0.05`.
- Multiplicity family: the two confirmatory interaction tests; Holm correction.
- Confidence level: `0.95`.
- Bootstrap: 10,000 complete-seed-block resamples, base seed `20260719`,
  two-sided 95% percentile interval, sensitivity analysis only; cellwise
  resampling is forbidden.
- Efficiency assessment: descriptive. Reference margins are latency B=1
  `10%`, latency B=128 `10%`, peak memory `5%`, and modeled energy `10%`;
  efficiency confidence level is `0.95`. C2/C1 and C4/C3 are paired within
  seed and simulation length on the natural-log-ratio scale. The geometric
  overhead uses a paired two-sided 95% t interval on the mean log ratio.
  Point estimate above tolerance is `exceeds_tolerance`; otherwise, a CI upper
  bound above tolerance is `within_tolerance_uncertain`, and an upper bound at
  or below tolerance is `supported_within_tolerance`. Missing or invalid inputs
  are `not_assessed`. Known exceedance takes precedence over unavailable
  metrics in the row-level summary. Peak training memory is analyzed only when
  all rows share one verified hardware/software/device/precision identity;
  mixed or unverifiable training environments block the analysis.
- Energy is currently specified as `not_assessed`. To report modeled energy,
  the authors must change this before freeze by recording a repository-relative
  constants file, its SHA-256, and its cited source in `benchmark.energy_model`.
  A benchmark CLI argument cannot add or replace that model after freeze.
- E3 diagnostics: one fixed validation batch of 8 samples for each retained
  confirmatory group, 8 Hutchinson/Rademacher probes, probe seed `20260719`,
  local Jacobian time index `-1`.
- Benchmark: CUDA, float32, B=1 and B=128, 25 warm-up iterations, 100 timed
  iterations, synchronized CUDA events, with input resident on the device
  before warm-up and timing.

## Wording gates

Evaluate the gates independently. `synergistic` has interpretation precedence;
`complementary` and `composable_additive` may both be reported when both gates
are true. Do not choose wording after seeing which label is more favorable.

1. `synergistic`: the SESOI-shifted one-sided interaction test rejects after
   Holm correction.
2. `complementary`: synergy is unsupported and the lower bounds of the 95%
   confidence intervals for both C4-C3 and C4-C2 are above zero.
3. `composable_additive`: the 90% interaction confidence interval lies wholly
   within +/-0.50 pp, and the one-sided 95% lower bounds for C4 versus C2 and C3
   are both above -0.50 pp.
4. `inconclusive`: no applicable gate is supported or the confirmatory block is
   incomplete.

## Run handling and exclusions

- Pilot runs are non-reportable engineering checks. They must use a separate
  output root such as `results/pilot`, must never be copied into
  `results/runs`, and must not enter confirmatory or sensitivity analyses. The
  first invocation creates one global audited C1-C4 plan for a single seed;
  later invocations cannot change its block, protocol, config hashes, or output
  root and cannot add a fifth unique run.
- A failed or interrupted attempt remains in the run directory and manifest.
  It may be continued only from that run's `last.pt` with the same frozen
  config and seed; every attempt log is retained.
- A failed seed is not replaced by a new seed. No seed is changed after the
  first confirmatory run begins.
- No run is deleted or excluded as an outlier. Any data-integrity failure is
  documented and leaves the confirmatory block unresolved; no partial-block
  analysis is substituted.
- Nonconvergence at the frozen validation threshold does not exclude the run.
- Hyperparameters, preprocessing, stopping rules, outcomes, margins, or
  wording gates are not changed after the first confirmatory run begins.
- Test partitions remain inaccessible until all 40 best checkpoints and the
  frozen selection rule pass the one-time final-test audit.

## Author review checklist

Each item maps to `protocol_status.confirmations` in the YAML. Every item must
be checked before `frozen: true` is permitted.

- [x] `reference_implementation_reviewed`: C1-C4 graphs and intervention sites
  match the intended study.
- [x] `neuron_parameters`: LIF/TA-LIF equations, `tau=0.5`, window
  initialization, reset-gradient distinction, and delayed TA update are
  accepted.
- [x] `static_augmentation_package`: crop, flip, normalization, and the decision
  to omit AutoAugment, CutMix, and label smoothing are accepted.
- [x] `optimizer_and_schedule`: 240 epochs, optimizer, milestones, learning
  rates, weight decay, batch size `64`, and deterministic settings are
  accepted, including the documented change from the infeasible batch sizes.
- [x] `cifar10dvs_preprocessing_and_split`: T=10, two polarities, 48x48 nearest
  resize, raw event counts, and stratified 70/10/20 allocation are accepted.
- [x] `outcome_thresholds_and_margins`: thresholds, SESOI, one-sided test,
  multiplicity family, descriptive efficiency margins, diagnostics, benchmark,
  energy-model status/source/hash, wording gates, and run-handling rules are
  accepted without inspecting joint confirmatory outcomes.

## Two-stage freeze record

This signed record intentionally does **not** contain its own SHA-256, the
commit that contains it, or a hash of a matrix that does not yet exist. Those
computed identifiers belong in the independent `FREEZE_MANIFEST.json` created
in Phase B. Do not paste them back into this file after signing.

### Phase A - human approval and source commit

In one reviewed change set, the authors must:

1. Resolve every bracketed pending field in this record and check all six
   author-review items.
2. Enter all three approval-evidence locations and dates below. The evidence
   may be a signed paper record, institutional e-signature record, or archived
   written approval; this file does not authenticate that external evidence.
3. Set all six YAML confirmations to `true`, enter all three names in
   `protocol_status.confirmed_by`, enter a timezone-aware ISO-8601
   `protocol_status.confirmed_at`, and set `protocol_status.frozen: true`.
4. Replace the top status value with
   `SIGNED - AUTHOR APPROVALS COMPLETE; PROTOCOL FROZEN`.
5. Commit the protocol, this record, and the exact experiment source. Do not
   include a provisional or stale `configs/generated` directory in this Phase A
   commit. The resulting 40-character commit ID is the freeze commit.

No program may perform steps 1-4 on behalf of the authors.

### Phase B - generated-artifact binding

With `HEAD` still at the clean Phase A commit, generate a new 40-run matrix in
an otherwise empty `configs/generated` directory. Then run
`scripts/create_freeze_manifest.py` with that explicit 40-character commit ID.
The command validates the signed-state marker, all approval fields, the frozen
YAML, Git commit, all 40 YAML files, both generated manifests, and every
recorded hash before it creates `FREEZE_MANIFEST.json`. It refuses to overwrite
an existing manifest and never edits this file or the protocol.

The independent manifest binds the repository URL, freeze commit/tree, protocol
blob and canonical hashes, signed-record blob hash, matrix hash, both matrix
manifest hashes, and the aggregate set of 40 configuration hashes. Its own
file hash is printed by the command and may be stored in an external archive
checksum list; it is not embedded recursively in the manifest.

Run full preflight after CIFAR10-DVS and the execution environment are ready.
The first confirmatory-run time belongs in the immutable run logs, not in this
pre-run author record. Any later amendment requires a new protocol version,
timestamp, reason, hashes, and replacement manifest. An amendment made after
outcome access cannot be represented as prespecified.

## Signatures

- Yanzhao Deng - approval evidence/location: `Private author archive / AUTHOR_PROTOCOL_APPROVAL_20260723_SIGNED.docx`; date: `2026-07-23`
- Peng Yan - approval evidence/location: `Private author archive / AUTHOR_PROTOCOL_APPROVAL_20260723_SIGNED.docx`; date: `2026-07-23`
- Song Wang, corresponding author - approval evidence/location: `Private author archive / AUTHOR_PROTOCOL_APPROVAL_20260723_SIGNED.docx`; date: `2026-07-23`
