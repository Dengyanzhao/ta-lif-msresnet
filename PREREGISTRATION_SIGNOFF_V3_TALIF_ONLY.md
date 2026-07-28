# TA-LIF-only v3 protocol sign-off record

Status: **SIGNED - AUTHOR APPROVALS COMPLETE; PROTOCOL FROZEN**

This prospective record accompanies `configs/protocol_v3_talif_only.yaml`.
It does not transfer, reuse, or imply any approval from the historical v1/v2
protocols. The user supplied collective written confirmation on behalf of all
three authors for Phase A. Phase B and all runtime gates remain mandatory before
any GPU health gate, pilot, formal training, or test-set access.

## Study identity

- Working study title: TA-LIF versus LIF on a fixed Spiking ResNet-20
- Current manuscript scope: C1/C2 only
- Excluded current-paper scope: MS-ResNet, C3/C4, topology effects, interaction,
  synergy, complementarity, and composability claims
- Authors expected to review: Yanzhao Deng; Peng Yan; Song Wang
- Accountable protocol confirmer(s): Yanzhao Deng; Peng Yan; Song Wang
- Joint v3 formal outcomes accessible at sign-off: `NONE`

## Prospective numerical contract

- Formal matrix: CIFAR-100/depth-20/T=6 and CIFAR10-DVS/depth-20/T=10,
  each with C1/C2 and five paired seeds, for 20 runs total.
- Formal seeds: `1882214332, 836017246, 126468528, 2075015328,
  419442269`.
- Dataset-specific pilot seeds: CIFAR-100 `474123945`; CIFAR10-DVS
  `799312121`. Pilot results are non-reportable.
- Retired seeds forbidden in v3: `11, 22, 33, 44, 55, 77, 88, 314159`.
- Health gates: 80 fixed-batch steps, accuracy at least `0.50`, final/initial
  loss at most `0.90`, C2 routed-gradient coverage at least `0.90`, and C2
  TA-enabled/frozen CUDA step-time ratio at most `3.0`.
- Pilot schedule: 120 epochs; every C1/C2 pilot run in both datasets must reach
  best validation accuracy at least `0.60`.
- Primary outcome: one-time independent test accuracy from `best.pt`, expressed
  in percentage points. Training never accesses the test set.
- Model selection: highest validation accuracy, then lowest validation loss,
  then earliest epoch on an exact tie. Test-based reselection is forbidden.
- Sole confirmatory analysis: CIFAR-100 paired seed-level C2-C1 differences,
  one-sided paired t test at alpha `0.05`, with a two-sided 95% t interval and
  all five raw paired differences.
- CIFAR10-DVS is a prespecified replication/external-validity analysis and
  cannot rescue a failed CIFAR-100 primary result. There is no cross-dataset
  Holm family.
- Sensitivity analyses: exhaustive 32-assignment seed-level sign flips and
  10,000 complete-pair bootstrap resamples per dataset. The fixed PCG64 seeds
  are CIFAR-100 `1033863572` and CIFAR10-DVS `1367073951`; linear percentile
  intervals are used. Sensitivity analyses cannot replace or rescue the primary
  t-test decision.
- Improvement wording requires CIFAR-100 one-sided `p < 0.05` and positive mean
  C2-C1 difference. Otherwise the primary result is described as inconclusive.
- Test access occurs only after all 20 frozen runs and checkpoints pass the v3
  audit, once per checkpoint, without outcome-based repetition.

## Author review checklist

Every item maps to `protocol_status.confirmations` in the v3 YAML. The checked
boxes record the user-attested collective review and approval of the complete
executable protocol and source on behalf of all three authors.

- [x] `talif_only_scope_and_legacy_preservation`: C1/C2-only scope, closure of
  the MS-ResNet route, and preservation of all historical evidence are accepted.
- [x] `reference_implementation_reviewed`: the fixed Spiking ResNet C1/C2 graphs
  and intervention sites are accepted.
- [x] `neuron_parameters`: LIF/TA-LIF equations, neuron constants, delayed TA
  activation, and deterministic reduction behavior are accepted.
- [x] `data_preprocessing_and_splits`: CIFAR-100 and CIFAR10-DVS preprocessing,
  manifests, and train/validation/test partitions are accepted.
- [x] `optimizer_schedule_and_pilot_gates`: optimizer, batch size, 120-epoch
  schedule, health thresholds, pilot seeds, and no-retry policy are accepted.
- [x] `paired_accuracy_analysis_and_claim_gates`: dataset roles, paired t test,
  sensitivity analyses, fixed analysis seeds, intervals, and wording gates are
  accepted.
- [x] `deterministic_seeds_and_run_handling`: formal seeds, retired seeds,
  failure retention, no substitution, and resume restrictions are accepted.
- [x] `model_selection_and_one_time_test_access`: validation-only checkpoint
  selection and the one-time final-test transaction are accepted.

## Author approvals

The following records document the collective written confirmation supplied by
the user on behalf of all three authors. They are user-attested approval records,
not three independently collected signatures.

- Yanzhao Deng - approval evidence/location: `Codex task transcript, collective written confirmation supplied by the user on behalf of all three authors`; date: `2026-07-28`
- Peng Yan - approval evidence/location: `Codex task transcript, collective written confirmation supplied by the user on behalf of all three authors`; date: `2026-07-28`
- Song Wang, corresponding author - approval evidence/location: `Codex task transcript, collective written confirmation supplied by the user on behalf of all three authors`; date: `2026-07-28`

## Two-stage freeze boundary

Phase A was completed from the user-attested collective author approval: every
checklist item and approval record is complete, all three names and a
timezone-aware timestamp are recorded in the YAML, and `frozen: true` is set.
The approval originated from the human confirmation recorded in the Codex task;
it was not generated autonomously by a program.

Phase B must use only the isolated paths declared by the v3 protocol:
`configs/v3_talif_only_generated` and
`FREEZE_MANIFEST_V3_TALIF_ONLY.json`. Historical generated matrices, manifests,
results, checkpoints, and failure evidence must not be overwritten or mixed
with v3 artifacts.
