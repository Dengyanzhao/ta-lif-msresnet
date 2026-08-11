# V7 mechanism study preregistration sign-off

Status: **AUTHORIZED - ACCOUNTABLE AUTHOR/USER APPROVAL RECORDED; PROTOCOL FROZEN**

This record authorizes the isolated V7 CIFAR-100 mechanism study described by
`configs/protocol_v7_mechanism.yaml`. It preserves the scientific design of V6
while replacing every execution identity after the consumed non-reportable V6
health failure. It does not amend, reopen, pool with, or reinterpret the V5
release or any V6 evidence.

## Authorization record

- Authorizer: Yanzhao Deng, accountable author and Codex task user
- Independent approval from Peng Yan: not asserted in this record
- Independent approval from Song Wang: not asserted in this record
- Authorization date: 2026-08-11 (Asia/Shanghai)
- Scope: complete all zero-seed V7 prerequisites, then execute the frozen
  sequence on a newly rented single RTX 5090 instance

## Predecessor disposition

- V5 release `v1.0.0`, DOI `10.5281/zenodo.21808938`, its one-time test
  transactions, results, and conclusions remain immutable and out of scope.
- V6 remains canonically `FAIL`; implementation findings do not retroactively
  convert the frozen failure into a pass.
- V6 health seed `1707261715` is
  `CONSUMED_NONREPORTING_V6_HEALTH_NEVER_RETRY`.
- The other 26 V6 health/pilot/formal/bootstrap/benchmark/conditional values
  are `RESERVED_UNUSED_RETIRED_WITH_V6` and must never be executed under V7.
- V6 failure evidence must never be copied into a V7 result root.

The immutable predecessor identities, all 26 retired values, and prohibited
actions are recorded in `V6_MECHANISM_TERMINATION.md` and bound into the V7
protocol under `predecessor_termination`.

## Frozen checklist

- [x] `v5_evidence_remains_immutable_and_out_of_scope`: V5 remains unchanged.
- [x] `v6_failure_is_immutable_and_all_v6_seeds_retired`: V6 remains failed;
  its consumed seed and all unused reserved values have terminal dispositions.
- [x] `six_condition_mechanism_design`: CIFAR-100 uses exactly M0, M1, M2, M3,
  M4, and PLIF in that order, with eight complete paired seed blocks.
- [x] `condition_level_confound_audit`: topology, shared initialization, data
  split, optimizer schedule, augmentation, precision, and budget remain frozen.
- [x] `plif_style_parameterization_and_optimizer_policy`: PLIF remains a
  protocol-matched learnable-decay control, not an exact Fang et al. replica.
- [x] `time_index_definition_k_equals_min_t_tminus1`: M3 remains
  `k=min(t,T-1)` with zero-based within-sample time and no modulo indexing.
- [x] `paired_seed_and_shared_initialization_design`: all six conditions share
  convolution, normalization, and classifier initialization within each seed;
  the formal matrix contains exactly 48 runs.
- [x] `health_pilot_and_formal_run_handling`: health and pilot are
  non-reportable; no seed substitution or failed-run fresh retry is permitted.
- [x] `holm_family_equivalence_and_wording_gates`: M4-M0 is the independent
  replication, the Holm family is exactly M4-M2 and M4-M3, and equivalence
  requires the frozen paired TOST.
- [x] `conditional_cifar10_extension`: CIFAR-10 stays inactive until the full
  CIFAR-100 activation receipt passes.
- [x] `validation_only_benchmark_and_one_time_test_access`: resource/activity
  proxies use validation data; energy claims are forbidden.

## Corrected health-integrity contract

The following fields are part of the frozen V7 protocol and therefore part of
its protocol hash:

- `expected_adaptive_update_by_condition`: M0 and M1 are `false`; M2, M3, M4,
  and PLIF are `true`. An empty adaptive-parameter list is a valid expected
  non-update, not a positive update.
- `fixed_batch_identity.seed_source`:
  `same_frozen_health_seed_for_preclaim_and_execution`.
- `fixed_batch_identity.rng_reset`:
  `isolated_cpu_torch_rng_reset_immediately_before_each_loader_construction`.
- `fixed_batch_identity.digest`:
  `sha256_over_fixed_input_and_target_tensors`.
- `fixed_batch_identity.required_relation`:
  `preclaim_sha256_exactly_equals_execution_sha256`.
- `fixed_batch_identity.mismatch_action`:
  `block_before_any_condition_probe`.

The fixed batch must be reconstructed from an isolated CPU Torch RNG reset
immediately before both loader constructions. A digest mismatch blocks before
the health seed is claimed or any condition probe begins.

## Core condition audit

| Condition | Neuron | Trainable adaptive parameters | Routing/index | Expected adaptive update |
|---|---|---:|---|---:|
| M0 | fixed LIF | none | none | false |
| M1 | route-matched fixed LIF | none | none | false |
| M2 | shared trainable threshold window | center + width, one entry | none | true |
| M3 | trainable threshold bank | center + width, T entries | time step `min(t,T-1)` | true |
| M4 | full TA-LIF threshold bank | center + width, T entries | pre-spike count `min(c_(t-1),T-1)` | true |
| PLIF | protocol-matched learnable decay | scalar decay per neuron | none | true |

## Fixed execution order

1. Verify the V7 source release and offline data package in a blank instance.
2. Verify the exact Python, RTX 5090, PyTorch, torchvision, CUDA, CUBLAS, data,
   and repository-module identities.
3. Run all read-only zero-seed source, protocol, matrix, and health prechecks.
4. Confirm both V7 health receipt paths are absent.
5. Claim and run V7 health seed `1068798027` exactly once.
6. If and only if health passes, run all six V7 pilot conditions with pilot
   seed `1673127435`, then validate the complete pilot.
7. Create and verify the V7 formal freeze manifest.
8. Run all 48 formal E9 trainings and retain every attempt record.
9. Audit complete checkpoints, environment, split, shared initialization, and
   configs before any test access.
10. Run validation-only resource/activity proxy benchmarking.
11. Execute exactly one committed final-test transaction per audited checkpoint.
12. Run E10 paired analysis and export the evidence package; consider the
    separately gated CIFAR-10 extension only if its activation receipt passes.

No result-dependent protocol edit, condition deletion, seed replacement,
outlier exclusion, partial-block analysis, test retry, or automatic failed-run
retry is authorized. No V7 seed may be used before every zero-seed gate passes.
