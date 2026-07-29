# TA-LIF-only v4 termination record

Status: TERMINATED AFTER CONSUMED NON-REPORTING CIFAR-100 HEALTH FAILURE

This is a post-hoc administrative record. It does not alter the frozen v4
protocol or either canonical cloud evidence file.

## Immutable evidence identity

- Git commit: `3d29236c80d80ce6d2cf6a594e561137708d7ff9`
- Protocol hash: `68bb14e892a696bc87b3a2d3f196b81fc58b299dd0c485cf72a50bb5fd0b0a54`
- Dataset: `cifar100`
- Consumed health seed: `1975342236`
- Attempt receipt SHA-256: `8115975674e4e1952f0573f691de887dee142018067c7388bd466220125ab6fb`
- Health report SHA-256: `547ea7b24da9976359605eb67467f89a100b482f3c2728fb2d760564d7f2a4b1`
- Health status: `FAIL`
- Runtime failure: none

## Frozen-gate outcome

Both C1 and C2 failed the fixed-batch response thresholds and the three-epoch
validation-loss threshold. The following implementation checks passed:

- Shared convolution/classifier gradients: 23/23 for C1 and C2.
- C2 TA routed-gradient coverage: 228/228, aggregate coverage 1.0.
- Checkpoint continuation: exact, with zero mismatches for C1 and C2.
- Residual-gradient batch coverage: 1.0 throughout both trajectories.
- Minimum reported surrogate support: approximately 0.19 for both conditions
  under the v4 report's then-recorded aggregation.
- Non-finite observations: zero.
- Parameter-norm ratios: 0.9987 (C1) and 0.9988 (C2).
- Conservative TA-enabled/frozen timing ratio: 1.6623, below 3.0.

The fixed-batch final loss fractions were 0.9923 (C1) and 0.9873 (C2), while
the frozen maximum was 0.90. Final fixed-batch accuracy was 0.0 for both, while
the frozen minimum was 0.50. Three-epoch fixed-validation loss changes were
-0.0171% (C1) and -0.0313% (C2), where a positive reduction of at least 1% was
required.

## Classification

The v4 FAIL is a valid decision under the frozen v4 contract and permanently
blocks v4 pilot and formal execution. It is not evidence of a runtime crash,
gradient disconnection, checkpoint corruption, numerical collapse, or excessive
TA cost.

The failed performance thresholds are not a valid standalone diagnosis of model
health. The fixed-batch thresholds were inherited from a different optimizer
regime, while v4 used a lower learning rate, five-epoch warmup, and gradient
clipping. The separate 1% longitudinal validation-loss rule was newly added in
v4 without independent calibration. The fixed-batch probe never advanced the
epoch scheduler, and the longitudinal probe ended at epoch 2 even though formal
C2 first enables TA at epoch 5. Therefore v5 must replace these
construct-misaligned checks rather than tune their values to the observed v4
outcome.

## Prohibited actions

- Never rerun CIFAR-100 v4 health seed `1975342236`.
- Treat CIFAR10-DVS v4 seed `1983855948` as `RETIRED_UNEXECUTED`; never run it
  for health, diagnosis, calibration, pilot, formal training, or any other use.
- Never run any v4 pilot or formal job.
- Never run the v4 CIFAR10-DVS health block as a way to rescue v4.
- Never edit, replace, or delete the canonical cloud receipt or health report.
- Never report v4 health or development-calibration values as manuscript results.

The only permitted continuation is a new protocol version with development
calibration isolated from all future pilot and formal seeds.
