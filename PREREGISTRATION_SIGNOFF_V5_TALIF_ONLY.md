# TA-LIF-only v5 accountable-author authorization record

Status: **AUTHORIZED - ACCOUNTABLE AUTHOR/USER APPROVAL RECORDED; PROTOCOL FROZEN**

This prospective record accompanies `configs/protocol_v5_talif_only.yaml`.
Protocol v4 terminated after its consumed, non-reporting CIFAR-100 health
failure. The Codex task user and accountable author, Yanzhao Deng, then gave
full authorization to complete the prerequisites and proceed through frozen
gates toward formal training. That authorization permits this v5 repair but
does not waive any health, pilot, freeze, or final-test gate.

This record does not represent Peng Yan or Song Wang as having supplied an
independent signature. Any journal or institutional co-author approval
requirement remains outside this engineering authorization record.

## Bound development evidence

The following development-only probes were executed once at Git commit
`609a505445d68e9961778999b5f7a4bc79cc83a8`. They are forbidden from manuscript
results and from confirmatory analysis.

| Dataset | Result SHA-256 | Status | Integrity anomalies |
|---|---|---|---|
| CIFAR-100 | `421dd0f1f08a7cac631df23d23815632acdb58eac6d654a2c8cd52383aab5eaa` | `COMPLETED` | none |
| CIFAR10-DVS | `b3837bfb98948edb180f133a8f5f23bc2ab6e1d764c71dd81a4f987057afefd5` | `COMPLETED` | none |

Both reports bind plan hash
`25dd909f58ddf7408d688c0f102e058191e73612564c07b674bc516147ea4c93`
and a clean tracked worktree. Their loss and accuracy trajectories are
descriptive only. No observed learning value from v4 or these probes was used
to choose a v5 health criterion.

The four development seeds `1515323759`, `214400889`, `117569744`, and
`563701053` are permanently excluded from v5 health, pilot, formal, and
bootstrap roles.

## Frozen implementation health gate

- Health uses one new dataset-specific seed that is never used by a pilot or
  formal run.
- The gate checks finite shared gradients, C2 TA routing, actual parameter
  updates, deterministic float32 operation, the absolute warmup learning-rate
  schedule, TA frozen state and absent optimizer state before epoch 5, TA
  activation and update at epoch 5, and exact `last.pt` continuation across
  the epoch 4 to epoch 5 boundary.
- Natural-data loss and accuracy are retained as descriptive observations but
  have no minimum, maximum, improvement, or convergence cutoff in health.
- Any PASS, FAIL, ERROR, or interruption after the attempt receipt consumes the
  health seed. There is no automatic or fresh retry under protocol v5.

## Frozen pilot and formal contract

- The graph, data, optimizer, 120-epoch schedule, TA activation fraction, model
  selection, test-access rules, and paired C2-C1 analysis remain the reviewed
  repair-study contract.
- One disjoint pilot seed per dataset produces four non-reportable runs. Each
  pilot must complete all 120 epochs.
- CIFAR-100 C1 and C2 must each reach best validation accuracy `0.30`, final
  ten-epoch mean `0.25`, late/best ratio `0.80`, and post-warmup residual
  gradient coverage `0.95`.
- CIFAR10-DVS C1 and C2 must each reach best validation accuracy `0.50`, final
  ten-epoch mean `0.45`, late/best ratio `0.80`, and post-warmup residual
  gradient coverage `0.95`.
- Formal execution contains exactly 20 runs: five dataset-specific paired
  C1/C2 blocks for CIFAR-100 and five for CIFAR10-DVS.
- Same-environment continuation from `last.pt` is permitted after a technical
  interruption. Fresh restart after an attempt, cross-environment resume,
  seed substitution, and partial-pair analysis are forbidden.
- The test set remains inaccessible during health, pilot, and training. Each
  audited formal `best.pt` receives one final test evaluation only after every
  formal run and checkpoint passes the frozen audit.

## Seed ledger

All v5 seeds are derived from
`ta-lif-msresnet/v5-author-freeze/2026-07-29` with the SHA-256 algorithm and
collision policy recorded in the protocol. The ledger explicitly excludes 37
historical, split, analysis, failed-health, and development values.

| Role | CIFAR-100 | CIFAR10-DVS |
|---|---|---|
| Health | `897211180` | `1280028821` |
| Pilot | `1968690143` | `1721988285` |
| Formal | `2141022414, 252611743, 1147962214, 931150728, 2106521224` | `157805125, 2131477013, 1311303835, 417519743, 2086335735` |
| Bootstrap | `894719475` | `159287562` |

## Accountable-author review checklist

- [x] `talif_only_dual_dataset_scope`: C1/C2-only, dual-dataset scope remains accepted.
- [x] `terminal_neuron_graph`: the shared `topology_required` graph remains accepted.
- [x] `optimizer_and_scheduler`: parameter groups, warmup, clipping, milestones, and gamma remain accepted.
- [x] `ta_activation_contract`: epoch-5 activation, TA LR scale, and zero TA weight decay remain accepted.
- [x] `data_preprocessing_and_splits`: dataset provenance and frozen partitions remain accepted.
- [x] `development_probe_evidence`: both exact development reports and their non-reportable role are accepted.
- [x] `deterministic_health_and_pilot_gates`: threshold-free health and full pilot gates are accepted.
- [x] `disjoint_seed_ledger`: derivation records, exclusions, and role/dataset separation are accepted.
- [x] `paired_accuracy_analysis_and_claim_gates`: primary, replication, sensitivity, and wording rules remain accepted.
- [x] `environment_and_run_handling`: RTX 5090 binding, deterministic float32, and continuation rules remain accepted.
- [x] `model_selection_and_one_time_test_access`: validation-only selection and one-time test access remain accepted.

## Authorization evidence

- Authorizer: Yanzhao Deng, accountable author and Codex task user
- Evidence/location: Codex task transcript containing the explicit instruction to fully authorize and rapidly complete prerequisites for formal experiments
- Recorded at: `2026-07-29T21:10:31+08:00`
- Scope: freeze protocol v5 and permit its development-evidence audit, one-shot health gates, non-reportable pilots, conditional formal release, and prespecified analysis
- Independent approval from Peng Yan: not asserted in this record
- Independent approval from Song Wang: not asserted in this record

## Conditional release boundary

Formal training remains mechanically blocked until both v5 health reports are
`PASS`, all four 120-epoch pilot runs pass, the aggregate pilot validation is
`PASS`, the complete 20-run formal matrix is verified, and
`FREEZE_MANIFEST_V5_TALIF_ONLY.json` is created from and verifies against the
committed source tree. A failed health or pilot gate terminates protocol v5 and
requires a new protocol version with unused seeds.
