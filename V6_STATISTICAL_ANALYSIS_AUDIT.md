# V6 statistical analysis audit

## Contract status

The executable statistical implementation follows
`configs/protocol_v6_mechanism.yaml` and `src/talif_msresnet/config_v6.py`.
It is isolated from V5: V5 artifacts, seeds, protocol fields, and conclusions
are neither modified nor pooled into V6.

The executable entry point rejects replacement seed lists, a changed alpha,
mislabelled contrasts, incomplete Holm families, and unregistered route
decisions. The formal seed order and statistical constants are imported from
the validated V6 configuration contract rather than maintained as a second
editable copy.

The V6 primary unit is one complete six-condition seed block. The frozen core
contains eight formal seeds and therefore 48 formal CIFAR-100 runs. A condition
run is not an independent replicate. Missing one condition leaves that seed
block incomplete and blocks confirmatory analysis; condition-wise deletion,
seed replacement, and outlier deletion are forbidden.

## Conditions and interpretation

| ID | Frozen condition | Interpretation boundary |
|---|---|---|
| `M0` | LIF | Original reference. |
| `M1` | Route-matched LIF | Same forward threshold behavior with the TA-LIF quiet-branch gradient route. |
| `M2` | Shared trainable threshold window | Window trainability without an index; parameter count is lower than M4. |
| `M3` | Time-indexed trainable threshold bank | Bank-size and optimizer-matched routing control for M4. |
| `M4` | Count-history-indexed TA-LIF | Full target mechanism. |
| `PLIF` | Protocol-matched `plif_style` neuron | Specific learnable-decay control, not an exact Fang et al. PLIF reproduction. |

The strongest history-specific contrast is `M4-M3`, because it changes the
routing signal while retaining a trainable bank of the same size. `M4-M2`
tests the full count-indexed bank against one shared window, but it also changes
bank capacity; its wording must remain at the condition level. An all-versus-M0
comparison family would not identify these incremental contributions. A
difference between significance labels is never treated as a significant
difference between conditions.

## Frozen inferential families

### Independent replication

`M4-M0` is an independent confirmatory replication of the V5 CIFAR-100
direction. It uses a one-sided paired t test of mean difference greater than
zero at alpha `.05`. It passes only when raw `p < .05` and the mean is positive.
No mechanism or secondary result may rescue a failed replication.

### Two-member Holm mechanism family

The direct family is exactly:

1. `M4-M2`: count-indexed bank versus a shared trainable window.
2. `M4-M3`: count-history index versus a time-step index with matched bank size.

Each uses a one-sided paired t test of a positive mean difference. Raw p-values
are Holm-adjusted as one complete two-test family at family-wise alpha `.05`;
the ordered thresholds are `.025` and `.05`. A missing or non-estimable member
blocks the family. Raw seed differences, mean, paired SD, SE, and the ordinary
two-sided 95% t interval must also be reported.

The `0.50-pp` SESOI is a practical interpretation threshold and the frozen
equivalence bound; the main Holm nulls remain zero as specified by the protocol.

### Secondary estimates and M1 branch

`M1-M0`, `PLIF-M0`, and `M4-PLIF` are secondary estimates and cannot rescue the
Holm family. `PLIF-M0` and `M4-PLIF` must not be promoted into unregistered
confirmatory PLIF claims.

`M1-M0` has a mandatory three-way branch:

| Gate | Permitted route conclusion |
|---|---|
| Two-sided 95% CI lower bound `> 0` and mean `>= 0.50 pp` | `route_positive_evidence` |
| Otherwise, paired TOST passes within `[-0.50, +0.50] pp` at alpha `.05` | `route_practical_equivalence` |
| Neither passes | `route_contribution_inconclusive` |

TOST uses `max(p_lower, p_upper)` and passes only when both one-sided tests
reject. A non-significant superiority result is never called equivalence.

### Sensitivity analyses

Every registered contrast (`M4-M0`, both Holm contrasts, and all three
secondary estimates) receives the frozen exhaustive 256-assignment one-sided
seed-level sign-flip analysis and a 10,000-draw complete-block PCG64 percentile
bootstrap using its contrast-specific seed. These are sensitivity analyses
only. Neither a sign-flip p-value nor a bootstrap interval can replace, rescue,
or change the paired-t, Holm, replication, TOST, or wording decisions.

## Study wording gates

- `mechanism_supported`: replication passes and both Holm contrasts reject.
- `partial_mechanism_support`: replication passes and exactly one Holm contrast
  rejects; name the supported direct contrast rather than generalizing.
- `mechanism_inconclusive`: replication passes but neither Holm contrast rejects.
- `replication_not_confirmed_mechanism_claim_blocked`: replication fails,
  regardless of Holm or secondary results.

The M1 branch is always reported alongside the study branch, but it cannot
change the main branch. A history-index claim specifically requires `M4-M3`;
`M4-M2` alone does not isolate history from bank capacity.

## MDES and eight-seed limitation

Prospective sensitivity uses the first Holm threshold `.05 / 2 = .025`, 80%
power, eight pairs, and the exact noncentral-t power function.

| Assumed paired SD | Detectable mean difference |
|---:|---:|
| `0.3844867 pp` | about `0.444 pp` |
| `0.50 pp` | about `0.578 pp` |
| `0.75 pp` | about `0.867 pp` |
| `1.00 pp` | about `1.156 pp` |

The reference SD is inherited from the observed V5 total-package contrast and
may be optimistic for new component contrasts. The sensitivity rows, not the
single reference row, should frame the limitation. These values are design
sensitivity, not post-hoc power claims.

With eight pairs, the minimum exhaustive one-sided sign-flip p-value is
`1/256 = 0.00390625`; the conservative two-test Holm floor is `0.0078125`, so
the sensitivity analysis has enough numerical resolution to corroborate a
family-wise `.05` result. The paired t assumptions remain difficult to diagnose
with eight differences; raw differences and sensitivity results are mandatory.

## Isolation and gate order

1. Implementation tests and six-condition confound audit.
2. Development-only debugging and any validation-only tuning.
3. Non-reportable health seed across all six conditions.
4. Non-reportable pilot seed across all six conditions.
5. Author freeze of code, protocol, exact formal seeds, split, condition table,
   analysis code, test-access rule, and wording gates.
6. All 48 formal training runs without formal-test access.
7. Complete-block, environment, split, initialization, and checkpoint audit.
8. One-time final-test evaluation after every prior gate passes.
9. Frozen replication, Holm, M1/TOST, secondary-estimate, and wording analysis.

Development, health, and pilot seeds are permanently excluded from formal
analysis. Formal seed substitution and post-outcome sample extension require a
new protocol version; they cannot be repaired inside the frozen V6 study.

## Validation-only resource proxy accounting

The validation-only benchmark distinguishes the control state actually used by
each frozen neuron condition. M2 records two shared-window boundary reads per
logical neuron-layer time step and no bank or spike-history update. M3 records
two time-indexed bank boundary reads per logical neuron-layer time step and no
spike-history update. M4 records two count-indexed bank boundary reads plus one
spike-count update per neuron state. The three access fields are mutually
exclusive where appropriate and are audited against the condition identity in
each immutable benchmark result.

These counts are high-level control-operation proxies. They are neither
instruction counts nor GPU memory-traffic traces, and they do not support an
energy claim. CUDA latency and peak memory are measured separately on the same
validation-only batch.

## Implementation

`src/talif_msresnet/v6_statistics.py` implements the exact block validator,
replication test, two-member Holm family, paired TOST, M1 conclusion branch,
study wording gates, exhaustive sign-flip analysis, complete-block bootstrap,
and MDES sensitivity grid. The implementation deliberately does not import,
modify, or write V5 evidence.
