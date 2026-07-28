# TA-LIF-only v3 protocol design record

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-28
- Verification Status: AUTHOR-APPROVED AND PHASE A-FROZEN
- Version Label: ta_lif_only_v3_phase_a_20260728

## Design status

This document records the agreed prospective design accompanying the signed and
Phase A-frozen `configs/protocol_v3_talif_only.yaml`. The filename retains
`DRAFT` only for traceability to its pre-sign-off development history; the YAML
and `PREREGISTRATION_SIGNOFF_V3_TALIF_ONLY.md` are the authoritative current
records. Phase A permits progression into the isolated Phase B health-gate and
non-reportable pilot workflow, but it does not authorize formal training or
test-set access before the aggregate pilot PASS and Phase B freeze manifest.

The historical C1--C4 code and evidence remain intact under
`MS_RESNET_ROUTE_CLOSURE.md`.

## Research question

For a fixed conventional Spiking ResNet-20 topology, does replacing LIF with
TA-LIF improve held-out classification accuracy under the same data split,
initialization, optimisation schedule, and seed?

This question excludes MS-ResNet. It supports a TA-LIF-versus-LIF claim only;
it cannot support a topology, interaction, synergy, complementarity, or
composability claim.

## Fixed study scope

| Element | v3 draft |
|---|---|
| Baseline condition | C1: conventional Spiking ResNet + LIF |
| Treatment condition | C2: conventional Spiking ResNet + TA-LIF |
| Excluded conditions | C3/C4 and all `ms_resnet` reporting claims |
| Datasets | CIFAR-100 (ResNet-20, T=6) and CIFAR10-DVS (ResNet-20, T=10) |
| Dataset roles | CIFAR-100 is the sole confirmatory primary analysis; CIFAR10-DVS is a pre-specified replication/external-validity analysis |
| Pairing unit | Same seed within a dataset, including identical split, initialization and data-order controls |
| Primary outcome candidate | One-time independent test accuracy of the validation-selected best checkpoint, expressed in percentage points |
| Secondary outcomes | Validation convergence, training time, CUDA latency, and peak memory; descriptive unless a later frozen contract says otherwise |
| Excluded scope | CIFAR-10, depth robustness, time-step sensitivity, energy claims without a sourced frozen model |

## Prospective execution structure

### Non-reportable pilot

- Four runs split into two independent C1/C2 blocks: one CIFAR-100 block and
  one CIFAR10-DVS block. Each block has its own new pilot seed.
- Both pilot seeds are disjoint from every retired and formal v3 seed.
- The pilot may establish implementation health and schedule feasibility only.
- Pilot accuracy, timing, and diagnostics are forbidden from the manuscript
  results and from formal statistical inference.
- A failed or completed pilot seed is consumed; it is never reused in v3.

On 2026-07-28, the v3 design selected an independent full C1/C2 health gate for
each dataset (option B). The CIFAR-100 gate and CIFAR10-DVS gate must each use
that dataset's actual input shape and preprocessing path. Each health report is
bound to its own dataset-specific pilot seed, software/hardware environment,
data identity, source commit, protocol hash, and acceptance hash. Creation of a
PASS, FAIL, ERROR, or interrupted report consumes that pilot seed. A PASS
authorizes only the matching dataset's two-run pilot; it cannot authorize the
other dataset.

### Formal matrix after a passing pilot and author freeze

- Twenty runs: two datasets x C1/C2 x five new paired formal seeds.
- Every formal seed produces exactly one C1 and one C2 run for each dataset.
- No seed substitution, outcome-based exclusion, partial-block analysis, or
  cross-environment checkpoint resume is allowed.
- Formal test access occurs once, only after all frozen runs/checkpoints pass
  the v3 audit.

### Confirmed deterministic seed derivation

The seed rule was confirmed on 2026-07-28 and independently reproduced with
Node and PowerShell implementations.

- Namespace: `ta-lif-only-v3|2026-07-28|seed-plan-v1`
- Namespace SHA-256:
  `a178ec8b74255a74dc392671819d28872cfd08aaa8996454c520793e36833052`
- For each label, hash the UTF-8 bytes of
  `namespace + "|" + label + "|" + decimal_counter` with SHA-256.
- Interpret digest bytes 0--3 as an unsigned big-endian integer, then apply
  `value & 0x7fffffff` to remain in the portable positive signed-31-bit range.
- Reject zero, a value in the retired set
  `{11,22,33,44,55,77,88,314159}`, or a value already selected; increment the
  counter and repeat. All selected values used counter zero.

| Role/label | Input SHA-256 | Confirmed seed |
|---|---|---:|
| `pilot|cifar100` | `9c428ea9099e787aabac160b9f761b92274080c15d59c5bff7fdd4dc558cd829` | 474123945 |
| `pilot|cifar10dvs` | `2fa488f9d5aa4c746d1b6d49fd2f1d401dff37a18c508286ba46698bc5d42201` | 799312121 |
| `formal|1` | `70304fbcafcd19c33811b31d8c8bf58e67b17ed1d711a4a2023e352e0906cee5` | 1882214332 |
| `formal|2` | `b1d49c5e7e1aa90a4f08ce82b77231fc1de5f5161ab550a794202bb26590e5af` | 836017246 |
| `formal|3` | `8789c1b016f63f0ab51673c528239486e7beef68cb38dd3ce8f11c54b65f379b` | 126468528 |
| `formal|4` | `fbae38a04ccc987a7953e4da40fce8458298946209698e47c686e85c6825e5fd` | 2075015328 |
| `formal|5` | `99002e5d466c9a9ed00a0321d8363b838b3de5bab820cb859eaaa40519a1fe99` | 419442269 |
| `bootstrap|cifar100` | `bd9f81942134081081204387558fff5118858f9a1c5977464dbea99f89c25083` | 1033863572 |
| `bootstrap|cifar10dvs` | `d17be49fb69d74b7f0d1da6602c339e881acf6ab679a1311e73350e342cef6bd` | 1367073951 |

The five formal seeds are used as the same paired blocks in both datasets.
Neither pilot seed is eligible for formal training or analysis.
The two bootstrap seeds are dataset-specific analysis seeds. They are not
training seeds and may be used only for the pre-specified sensitivity analysis.

## Confirmed pilot gates

On 2026-07-28, the v3 design confirmed the following requirements. They retain
the v2r2 engineering thresholds rather than changing them after observation:

| Gate | Confirmed design requirement |
|---|---|
| Determinism | Strict PyTorch deterministic algorithms enabled before any health-gate compute |
| Fixed-batch overfit | Within each required health block, C1 and C2 each reach accuracy >= 0.50 and final/initial loss <= 0.90 in 80 steps |
| TA gradient routing | C2 routed-gradient coverage >= 0.90 |
| Checkpoint transition | Exact required boundary/checkpoint audit passes for C1 and C2 |
| TA compute overhead | C2 TA-enabled/frozen CUDA step-time ratio <= 3.0 |
| Pilot schedule | 120 epochs; every C1/C2 run in both datasets must reach best validation accuracy >= 0.60 |

A pilot failure would stop v3 formal execution. Any schedule or threshold
revision would require a new protocol version, a new unused pilot seed, and a
new author decision; it is not an automatic retry. These values cannot be
relaxed after the first dataset-specific health report is created.

## Author-approved prespecified analysis contract

For dataset `d` and formal seed `s`, define the paired effect as:

```text
Delta(d, s) = test_accuracy(C2, d, s) - test_accuracy(C1, d, s)
```

The estimator is the mean of the five seed-level `Delta` values per dataset.
This removes the obsolete four-cell difference-in-differences and keeps the
seed pairing explicit. The design choices below were recorded as confirmed on
2026-07-28 and are bound by the user-attested collective written confirmation
on behalf of all three authors. The YAML was frozen before any v3 seed was
consumed.

1. **Dataset role and multiplicity -- confirmed 2026-07-28.** CIFAR-100 is the
   sole confirmatory primary analysis. CIFAR10-DVS is a pre-specified
   replication/external-validity analysis and is not a second confirmatory
   primary endpoint. There is no cross-dataset Holm family. A CIFAR10-DVS
   result cannot rescue a failed CIFAR-100 primary test; a discordant
   CIFAR10-DVS result limits any cross-domain wording even if the CIFAR-100
   primary test passes.
2. **Hypothesis threshold and sidedness -- confirmed 2026-07-28.** The
   CIFAR-100 primary hypothesis is the prospective paired, one-sided
   superiority test `H0: mean(Delta) <= 0` against `H1: mean(Delta) > 0` at
   `alpha = 0.05`. The analysis must also report a two-sided 95% confidence
   interval and all five raw seed-level contrasts. The former 0.50-percentage-
   point interaction threshold does not transfer to v3.
3. **Analysis method -- confirmed 2026-07-28.** The declared primary method is
   a paired seed-level t test over the five CIFAR-100 `Delta` values (`df = 4`).
   Its one-sided p value is evaluated against `alpha = 0.05`, and its two-sided
   t interval is reported at 95% confidence. An exhaustive one-sided sign-flip
   calculation over all `2^5 = 32` assignments and a complete-seed-block
   bootstrap are sensitivity analyses only. They cannot replace, override, or
   rescue the primary t-test decision. All five raw paired differences remain
   visible. Each dataset uses 10,000 complete seed-block bootstrap resamples.
   CIFAR-100 uses analysis seed `1033863572`, derived from label
   `bootstrap|cifar100`; CIFAR10-DVS uses the independently derived analysis
   seed `1367073951`, from label `bootstrap|cifar10dvs`. Both used counter zero
   under the confirmed SHA-256 seed rule. For each dataset, initialize
   `numpy.random.Generator(numpy.random.PCG64(seed))`, then draw indices with
   `rng.integers(0, 5, size=(10000, 5), endpoint=False)`. Each selected index
   resamples the complete paired C1/C2 seed block, and the statistic is the mean
   of the five resampled `Delta` values. Report the two-sided 95% percentile
   interval as `numpy.quantile(bootstrap_means, [0.025, 0.975],
   method="linear")`. This is not a BCa, basic, or studentized interval and
   remains sensitivity-only.
4. **Claim gate -- confirmed 2026-07-28.** Use `TA-LIF improved accuracy on
   CIFAR-100` only when the frozen CIFAR-100 primary test has a one-sided
   `p < 0.05` and `mean(Delta) > 0`. Otherwise report the observed estimate and
   interval as inconclusive; do not use equivalence, no-effect, or improvement
   language. CIFAR10-DVS cannot rescue a failed CIFAR-100 primary test. It may
   support cross-domain wording only when its estimated `mean(Delta) > 0`, and
   any such wording must retain its replication/external-validity status rather
   than presenting a second confirmatory claim.

## Completed Phase A author sign-off

1. Confirm the C1/C2-only scope and that MS-ResNet is future work only.
2. Dataset role is confirmed: CIFAR-100 is the sole confirmatory primary;
   CIFAR10-DVS is the pre-specified replication analysis.
3. Hypothesis threshold, sidedness, and implementation are confirmed as the
   zero-threshold, one-sided paired t test at `alpha = 0.05`, with a two-sided
   95% t interval and all raw seed contrasts. Exact sign-flip and complete-block
   bootstrap results are sensitivity-only. Each dataset uses 10,000 bootstrap
   resamples, its independently derived fixed analysis seed, NumPy `PCG64`, and
   the two-sided 95% linear-percentile implementation recorded above.
4. The manuscript claim gate is confirmed: improvement language requires the
   CIFAR-100 primary test to have one-sided `p < 0.05` and positive
   `mean(Delta)`. Otherwise the result is inconclusive. CIFAR10-DVS cannot
   rescue the primary test and supports cross-domain wording only when its
   estimated mean effect is positive.
5. Health-gate scope is confirmed as one independent full C1/C2 gate per
   dataset (option B). The 80-step health thresholds, ratio limit, 120-epoch
   pilot schedule, and 0.60 validation threshold are confirmed.
6. The deterministic SHA-256 rule, two pilot seeds, and five paired formal
   seeds are confirmed. They cannot be regenerated or substituted.
7. The v3 author sign-off and source freeze are complete before Phase B
   generates any configuration. Phase B audits the generated configurations
   before health-gate, pilot, or formal use. No existing v1/v2 sign-off
   transfers to v3.

## Implemented isolation boundary

The v3 code path is intentionally isolated:

- preserve the legacy C1--C4 schema and validators for audit;
- add a versioned v3 protocol profile with `active_conditions: [C1, C2]`;
- add v3-only generation, health/preflight, final-test, and paired-analysis
  paths; and
- add tests proving that v3 cannot generate C3/C4 or consume retired seeds.

Until all three authors complete the v3 sign-off and Phase A is committed, do
not run `run_matrix.py`, `pilot_health_gate_v3.py`, or any training command for
v3. Formal execution additionally requires the Phase B manifest.
