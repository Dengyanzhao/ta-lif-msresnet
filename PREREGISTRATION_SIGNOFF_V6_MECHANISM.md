# V6 mechanism study preregistration sign-off

Status: **AUTHORIZED - ACCOUNTABLE AUTHOR/USER APPROVAL RECORDED; PROTOCOL FROZEN**

This record authorizes the isolated V6 CIFAR-100 mechanism study described by
`configs/protocol_v6_mechanism.yaml`. It does not amend, reopen, pool with, or
reinterpret the released V5 study, tag `v1.0.0`, or DOI
`10.5281/zenodo.21808938`.

## Authorization record

- Authorizer: Yanzhao Deng, accountable author and Codex task user
- Independent approval from Peng Yan: not asserted in this record
- Independent approval from Song Wang: not asserted in this record
- Authorization date: 2026-08-10 (Asia/Shanghai)
- Scope: complete all V6 prerequisites, then execute the frozen sequence on a
  newly rented single RTX 5090 instance

## Frozen checklist

- [x] `v5_evidence_remains_immutable_and_out_of_scope`: V5 source release,
  results, one-time test transactions, DOI, and conclusions remain unchanged.
- [x] `six_condition_mechanism_design`: CIFAR-100 uses exactly M0, M1, M2, M3,
  M4, and PLIF in that order, with eight complete paired seed blocks.
- [x] `condition_level_confound_audit`: topology, initialization, data split,
  optimizer schedule, augmentation, precision, and training budget are frozen;
  each intended condition difference is declared explicitly.
- [x] `plif_style_parameterization_and_optimizer_policy`: PLIF is a
  protocol-matched learnable decay control and is not described as an exact
  Fang et al. PLIF reproduction; its scalar `raw_decay` follows the same
  delayed adaptive-parameter optimizer policy as trainable threshold banks.
- [x] `time_index_definition_k_equals_min_t_tminus1`: M3 uses
  `k=min(t,T-1)` with zero-based within-sample time and never modulo indexing.
- [x] `paired_seed_and_shared_initialization_design`: all six conditions share
  convolution, normalization, and classifier initialization within each seed;
  the formal matrix contains exactly 48 runs.
- [x] `health_pilot_and_formal_run_handling`: health and pilot are
  non-reportable; no seed substitution or fresh retry is allowed; a technical
  interruption may resume only from `last.pt` in the identical recorded
  environment.
- [x] `holm_family_equivalence_and_wording_gates`: M4-M0 is an independent
  replication; the Holm family is exactly M4-M2 and M4-M3; equivalence requires
  the frozen paired TOST; M1 follows the predeclared three-way branch.
- [x] `conditional_cifar10_extension`: CIFAR-10 remains inactive unless the
  complete CIFAR-100 integrity, replication, and mechanism gates pass; it is a
  boundary triangulation study, not a causal modality test.
- [x] `validation_only_benchmark_and_one_time_test_access`: resource/activity
  proxies use validation data only; all 48 checkpoints must pass the frozen
  audit before one test transaction per checkpoint; energy claims are
  forbidden.

## Core condition audit

| Condition | Neuron | Trainable adaptive parameters | Routing/index | Delayed adaptive group |
|---|---|---:|---|---|
| M0 | fixed LIF | none | none | N/A |
| M1 | route-matched fixed LIF | none | none | N/A |
| M2 | shared trainable threshold window | center + width, one entry | none | yes |
| M3 | trainable threshold bank | center + width, T entries | time step `min(t,T-1)` | yes |
| M4 | full TA-LIF threshold bank | center + width, T entries | pre-spike count `min(c_(t-1),T-1)` | yes |
| PLIF | protocol-matched learnable decay | scalar decay per neuron | none | yes |

M0 and M1 intentionally have no empty adaptive optimizer group. M2, M3, M4,
and PLIF use the same activation epoch, learning-rate scale, and zero adaptive
weight decay. M4-M3 is the bank-capacity-matched routing contrast; M4-M2 also
changes bank capacity and must be described at the condition level.

## Fixed execution order

1. Verify the committed release bundle in a blank `/hy-tmp` checkout.
2. Install and verify RTX 5090, PyTorch `2.9.1+cu128`, torchvision
   `0.24.1+cu128`, CUDA runtime `12.8`, deterministic float32, and the
   repository module path.
3. Generate and byte-validate the six pilot and 48 formal configurations.
4. Claim and run the one-time CIFAR-100 V6 health seed.
5. If and only if health passes, run all six 120-epoch pilot conditions.
6. Validate the complete pilot and create/verify the formal freeze manifest.
7. Run all 48 formal trainings; preserve every attempt and interruption record.
8. Audit environment, split, shared initialization, configs, and `best.pt` for
   the complete matrix.
9. Run validation-only resource/activity proxy benchmarking.
10. Execute exactly one committed final-test transaction per audited checkpoint,
    each bound to the frozen CIFAR-100 test-pickle and source-provenance hashes.
11. Run the frozen paired analysis and export its evidence package.
12. Consider the separately frozen CIFAR-10 extension only if its activation
    receipt is PASS.

No result-dependent protocol edit, condition deletion, seed replacement,
outlier exclusion, partial-block analysis, speculative test retry, or automatic
failed-run retry is authorized by this record.
