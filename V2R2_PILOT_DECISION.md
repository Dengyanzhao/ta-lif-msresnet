# v2r2 pilot engineering decision record

Recorded on 2026-07-27 (Asia/Shanghai), before the first execution using seed 88.
This is a non-reporting engineering record, not an author-approved confirmatory
protocol and not evidence eligible for the manuscript results.

## Inputs retained as historical evidence

- Production implementation baseline commit:
  `0bc078d0dd5153b9f0002a4e6c480cead4012b18`.
- Failed v2r1 seed-77 health report:
  `results/pilot/v2_seed77_e120_health.json`.
- Failed report SHA-256:
  `6c79e0b721c787b29ae2c4dbdd2a7ae104848397a92cbab94aab4b1cf987097e`.
- The v2r1 functional gate passed C1, C3, and C4. C2 reduced its fixed-batch
  loss to 0.3365 of the initial loss, and its shared/TA gradient and checkpoint
  checks passed, but its 40-step final accuracy was 0.25 rather than the fixed
  minimum 0.50. Seed 77 is therefore consumed and must not be rerun.
- The original TA reduction also failed the fixed timing gate. The production
  reduction benchmark at commit `0bc078d` subsequently passed without changing
  the scientific TA selection rule: C2 ratio 1.6853 and C4 ratio 1.6847, both
  below the pre-existing maximum 3.0.
- Production benchmark evidence directory:
  `results/pilot/engineering_benchmark_0bc078d`.
- Production benchmark stdout SHA-256:
  `39f20a564446c6738c6f83daa8f08e39a55b7debed833831972d035ce553edc5`.
- Production benchmark stderr SHA-256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Production benchmark summary SHA-256:
  `0f6470f0526652c79d3643e64bd82b3746ae41edd002b20c24b5b66731105453`.

The paths above are ignored runtime evidence and are not asserted to be present
in every checkout. Their hashes were recorded from the operator's RTX 5090
execution and must be verified against the downloaded evidence archive.

## Prospective v2r2 identity

- Identity: `v2r2_seed88_of80_e120`.
- Protocol: `configs/protocol_v2r2_seed88_of80_e120.yaml`.
- Protocol hash:
  `77109228d1f6479d47a97a011bf52a60987e9083ff8bab5d7ed64c7176c6af8a`.
- Acceptance hash:
  `5ef063ed53d00b70f4d8b16aec55bf8353f00afb529db517b8bd540cb5bdee29`.
- Seed: 88, disjoint from formal seeds 11/22/33/44/55 and retired pilot seed 77.
- Fixed-batch overfit window: 80 steps.
- Candidate schedule: 120 epochs for each of C1, C2, C3, and C4.

Repository review found no tracked experiment identity using seed 88. The first
GPU use of seed 88 must occur only after the protocol and generated matrix are
committed. This repository check does not claim knowledge of untracked or
external experiments.

## Rationale and non-changes

The overfit window is doubled from 40 to 80 steps because the failed C2 run had
valid gradients, a valid checkpoint transition, and a substantial loss decline,
while the 40-step memorization accuracy gate was not met. Eighty is a fixed
prospective diagnostic window, not a threshold chosen to match an observed
passing step.

The following acceptance criteria are unchanged:

- minimum fixed-batch accuracy: 0.50;
- maximum final-to-initial loss fraction: 0.90;
- minimum TA routed-gradient coverage: 0.90;
- maximum TA-enabled/frozen CUDA step ratio: 3.0;
- required best pilot validation accuracy: 0.60 for every condition;
- 120-epoch candidate schedule and fresh 160-epoch fallback rule.

The accuracy threshold is not reduced to the observed 0.25, and the timing
threshold is neither relaxed after the legacy failure nor tightened around the
observed 1.68 ratios. Model, optimizer, augmentation, data split, and TA schedule
settings are unchanged from v2r1.

## Consumption and stop rules

- The old protocol, seed-77 report, and both old and production benchmark
  evidence are immutable historical records.
- A v2r2 health JSON, PASS or FAIL, consumes seed 88. A FAIL cannot be retried,
  reinterpreted by changing a threshold, or followed by the 120-epoch pilot.
- A launch blocked before GPU execution and before JSON creation may be corrected
  without changing the protocol; any ambiguous interruption requires an audit.
- The four-condition pilot may begin only after a bound health PASS on the same
  commit, GPU UUID, software stack, data, and deterministic environment.
- A purely technical pilot interruption may resume only from that run's own
  `last.pt` on the same environment. Scientific failure, cross-environment
  continuation, seed substitution, and checkpoint reuse across identities are
  forbidden.
- A failed 120-epoch schedule decision requires a new protocol identity, a new
  previously unused seed, fresh runs, and no resume into the 160-epoch fallback.
