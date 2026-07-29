# TA-LIF-only v4 accountable-author authorization record

Status: **AUTHORIZED - ACCOUNTABLE AUTHOR/USER APPROVAL RECORDED; PROTOCOL FROZEN**

This prospective record accompanies
`configs/protocol_v4_talif_only.yaml`. On 2026-07-29, the Codex task user and
accountable author, Yanzhao Deng, selected the dual-dataset scope (option B),
selected the `topology_required` terminal-neuron graph (option A), and then
gave full authorization to complete the prerequisites and proceed through the
frozen gates to formal training. The authorization covers the exact values in
the executable v4 YAML and this runbook; it does not waive any PASS gate.

This record intentionally does **not** represent Peng Yan or Song Wang as having
provided independent signatures. Their names are not entered in
`protocol_status.confirmed_by`, and no approval is inferred from the v1-v3
records. Any journal or institutional requirement for independent co-author
approval remains outside this engineering authorization record.

## Study identity and scope

- Accountable author and task user: Yanzhao Deng
- Other manuscript authors, without a v4 signature asserted here: Peng Yan;
  Song Wang
- Active contrast: C2 TA-LIF versus C1 LIF on one fixed Spiking ResNet-20 graph
- Active datasets: CIFAR-100 primary; CIFAR10-DVS prespecified external-validity
  replication
- Closed scope: C3/C4, MS-ResNet, topology effects, interaction, synergy,
  complementarity, and composability
- Reportable formal matrix: 20 runs; five paired C1/C2 blocks per dataset
- Pilot artifacts: non-reportable and ineligible for manuscript inference

## Frozen scientific contract

- Every v4 run uses `terminal_neuron_mode: topology_required`. C1 and C2 each
  contain 19 neuron layers; legacy `always` remains available only for audit.
- Optimizer: SGD, momentum `0.9`, batch size `64`, base learning rate `0.025`,
  global gradient clipping `1.0`, five warmup epochs, milestones
  `[75, 90, 105]`, and gamma `0.1`.
- Weight decay `5e-4` applies only to convolutional and linear weights.
  Normalization parameters, biases, and TA parameters receive zero decay.
- TA parameters are frozen for the five-epoch warmup, then use learning-rate
  scale `0.1` and zero weight decay.
- Both pilots run for 120 epochs. The CIFAR-100 C1 and C2 runs must each reach
  best validation accuracy `0.30`, final-ten-epoch mean `0.25`, and late/best
  ratio `0.80`. CIFAR10-DVS must reach `0.50`, `0.45`, and `0.80`, respectively.
- Each pilot also requires post-warmup residual-gradient batch coverage at
  least `0.95`. Both conditions and both datasets must pass.
- Longitudinal health observes three epochs, 16 train batches and four
  validation batches per epoch. Required residual-gradient coverage is `0.95`,
  surrogate-support coverage is `0.001`, loss reduction is `0.01`, parameter
  norm ratio remains in `[0.50, 2.00]`, and nonfinite observations equal zero.
- The retained fixed-batch and CUDA timing checks remain frozen exactly as in
  the YAML. Every PASS, FAIL, ERROR, or interruption consumes the dataset's
  pilot seed; a fresh retry requires a new protocol version and unused seed.
- Only same-environment continuation from `last.pt` is permitted after a
  technical interruption. Cross-environment resume and seed substitution are
  forbidden.

## Seed derivation and ledger

The namespace is
`ta-lif-msresnet/v4-author-freeze/2026-07-29`. Each role/dataset/index label was
hashed with SHA-256; the first eight digest bytes were mapped to a positive
31-bit integer, rejecting every historical value and every collision. Dataset
seed sets are disjoint.

| Role | Dataset | Frozen values |
|---|---|---|
| Pilot | CIFAR-100 | `1975342236` |
| Pilot | CIFAR10-DVS | `1983855948` |
| Formal paired blocks | CIFAR-100 | `375760402, 595643067, 671880744, 1045910319, 615268440` |
| Formal paired blocks | CIFAR10-DVS | `1230259817, 1487387499, 856172396, 1846577336, 1248366457` |
| Bootstrap sensitivity | CIFAR-100 | `874085245` |
| Bootstrap sensitivity | CIFAR10-DVS | `2019339204` |

All v1-v3 training, diagnostic, pilot, and analysis seeds listed in
`V4_RETIRED_SEEDS` are forbidden for v4 training. Pilot seeds cannot enter the
formal matrix, and formal seeds cannot cross dataset boundaries.

## Analysis and test-access contract

- The sole confirmatory result is the CIFAR-100 seed-paired C2-C1 test-accuracy
  difference, evaluated by a one-sided paired t test at alpha `0.05` and
  accompanied by the two-sided 95% t interval and all five differences.
- CIFAR10-DVS repeats the paired analysis as external-validity evidence and
  cannot rescue a failed CIFAR-100 result.
- Exhaustive 32-assignment sign flips and 10,000 complete-pair PCG64 bootstrap
  resamples are sensitivity analyses only and cannot replace the primary test.
- `best.pt` is selected by highest validation accuracy, then lowest validation
  loss, then earliest epoch. Test-based selection is forbidden.
- No training or pilot command may access a test set. Each of the 20 frozen
  `best.pt` checkpoints receives one test evaluation only after the complete
  formal audit passes.
- Failed runs remain recorded; no outlier exclusion, seed substitution, or
  incomplete-pair analysis is permitted.

## Accountable-author review checklist

Each checked item maps exactly to `protocol_status.confirmations` in the v4
YAML. The checks record the task user's accountable authorization, not three
independently collected signatures.

- [x] `talif_only_dual_dataset_scope`: C1/C2-only dual-dataset scope and the
  continued closure of C3/C4 and MS-ResNet are accepted.
- [x] `terminal_neuron_graph`: the shared 19-neuron-layer
  `topology_required` graph and legacy audit behavior are accepted.
- [x] `optimizer_and_scheduler`: parameter groups, learning rate, warmup,
  clipping, milestones, and gamma are accepted.
- [x] `ta_activation_contract`: delayed TA activation, LR scale, and zero
  TA weight decay are accepted.
- [x] `data_preprocessing_and_splits`: dataset provenance, preprocessing, and
  frozen train/validation/test partitions are accepted.
- [x] `health_and_pilot_gates`: dataset-specific thresholds, longitudinal
  health measurements, one-shot consumption, and non-reportable status are
  accepted.
- [x] `deterministic_seed_derivation`: namespace, algorithm, exact pilot/formal/
  bootstrap seeds, exclusions, and dataset separation are accepted.
- [x] `paired_accuracy_analysis_and_claim_gates`: primary/replication roles,
  paired tests, sensitivity analyses, and wording gates are accepted.
- [x] `environment_and_run_handling`: RTX 5090 runtime binding, deterministic
  float32 execution, resume restrictions, and artifact isolation are accepted.
- [x] `model_selection_and_one_time_test_access`: validation-only selection
  and one-time post-audit test access are accepted.

## Authorization evidence

- Authorizer: Yanzhao Deng, accountable author and Codex task user
- Evidence/location: Codex task transcript containing `B`, `A`, and the explicit
  instruction `I fully authorize; complete the prerequisites quickly so I can
  run the formal experiments` (English rendering of the Chinese user message)
- Recorded at: `2026-07-29T16:36:09+08:00`
- Scope: freeze the exact v4 protocol and permit prerequisite validation,
  dataset-specific health gates, non-reportable pilots, and conditional formal
  release after all frozen machine gates pass
- Independent approval from Peng Yan: not asserted in this record
- Independent approval from Song Wang: not asserted in this record

## Conditional release boundary

This authorization is prospective and conditional. It permits generation of
the isolated four-run pilot matrix and twenty-run formal matrix, followed by
the exact health and pilot workflow in `V4_TALIF_ONLY_RUNBOOK.md`. Formal
training remains blocked unless all of the following are verified without
altering the protocol: both dataset health reports are PASS, all four pilot
runs satisfy their dataset thresholds, aggregate pilot validation is PASS, the
source and runtime identity match the pilot environment, the formal matrix has
exactly 20 valid runs, and `FREEZE_MANIFEST_V4_TALIF_ONLY.json` verifies.

Any change to a scientific setting, threshold, seed, graph, analysis rule, or
artifact path invalidates this release and requires a new protocol version.
