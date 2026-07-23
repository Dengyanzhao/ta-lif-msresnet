# Implementation decisions requiring author review

This file separates source-grounded behavior from choices introduced by the
reconstructed KBS experiment package.

## Source-grounded behavior

- LIF integration and decay follow dissertation Eqs. (2.1)-(2.3).
- A spike occurs only when `U > (V1+V2)/2`, matching Eq. (2.12).
- Rectangular derivatives for `U`, `V1`, and `V2` follow Eqs. (2.11)-(2.14).
- TA-LIF selects a layer-shared window using each element's cumulative
  pre-decision spike count.
- Original LIF isolates the reset indicator. TA-LIF additionally retains the
  non-spiking surrogate membrane path in Eqs. (2.18) and (2.21).
- TA parameters start after 5% of 240 epochs at `0.1 x` the network learning
  rate with zero weight decay.
- MS-ResNet keeps the shortcut/addition in the membrane domain; projection
  shortcuts contain no neuron; the conventional topology retains its
  post-addition spike gate.

## Reconstructed choices

- CIFAR ResNet-20 widths, stem, projection, terminal readout, BatchNorm
  placement, and mean-over-time classifier are a new reference implementation.
  Generic ResNet-56 support is not part of the confirmatory design.
- TA windows use `center +/- width/2`, with
  `width = softplus(raw_width) + delta_min`.
- The provisional decay coefficient is `tau=0.5`.
- The provisional static package uses crop/flip/normalization only. The code
  also supports AutoAugment, CutMix, and label smoothing, but they are off.
- CIFAR10-DVS provisionally uses 10 polarity-separated bins, 48 x 48 nearest
  resize, and a stratified 70/10/20 train/validation/test allocation.
- The proposed validation-accuracy proportions are 0.60 for CIFAR-100 and 0.60
  for CIFAR10-DVS. The convergence epoch is the first epoch
  reaching the threshold; a run below threshold after all 240 epochs is retained
  with `converged=0`. These values remain pending author sign-off and protocol
  freeze. The protocol retains CIFAR-10 `0.70` as an unused compatibility value;
  no confirmatory configuration uses it.
- The proposed interaction smallest effect size of interest is 0.50 percentage
  points. Confirmatory inference uses the five direct seed-level
  `(C4-C3)-(C2-C1)` contrasts for seeds 11, 22, 33, 44, and 55 in each of two
  fixed primary groups: CIFAR-100/depth-20/T=6 and
  CIFAR10-DVS/depth-20/T=10. One-sample t tests on the five seed contrasts test
  `H0: delta <= 0.50 pp` one-sided and are Holm-adjusted at family-wise alpha
  0.05; two-sided 95% confidence intervals are reported. Effect-coded OLS is
  secondary/descriptive.
- The result wording is gated before execution. `synergistic` requires the
  SESOI-shifted one-sided Holm test to pass. If it does not, `complementary`
  requires positive lower bounds for both C4-C3 and C4-C2 two-sided 95%
  intervals. `composable_additive` requires the interaction 90% interval
  within +/-0.50 pp and C4 accuracy noninferiority to C2 and C3 at a 0.50 pp
  margin. Failure to pass these gates is `inconclusive`, not evidence of no
  interaction.
- The fixed sensitivity analysis uses 10,000 complete-seed-block bootstrap
  resamples with seed 20260719 and a two-sided 95% percentile interval; cellwise
  resampling is forbidden. It cannot replace the confirmatory method after
  results are observed.
- The confirmatory matrix contains 40 E1 runs. It contains no CIFAR-10 group,
  no depth-56 group, and no E4 T=2/T=4 group. Accordingly, depth robustness and
  time-step sensitivity are outside the planned claims; the bootstrap above is
  an inferential robustness check, not a time-step experiment.
- The proposed descriptive engineering tolerances are 10% for B=1 latency,
  10% for B=128 latency, 5% for peak training memory, and 10% for explicitly
  modeled energy. They are not formal noninferiority margins. Overheads are
  paired as C2/C1 and C4/C3 within seed and simulation length, analyzed as log
  ratios, and reported with paired two-sided 95% t confidence intervals for the
  mean log ratio. Firing
  rate is descriptive and has no directional tolerance. A point overhead above
  its margin is `exceeds_tolerance`; otherwise an upper interval bound above the
  margin is `within_tolerance_uncertain`, and an upper bound at or below the
  margin is `supported_within_tolerance`.
- The row-level efficiency status uses the fixed precedence `exceeds`,
  `uncertain`, `not_assessed`, then `supported`; unavailable energy cannot hide
  a known latency or memory exceedance. Peak training memory requires one
  verified hardware/software/device/precision identity across all compared
  training rows; mixed identities block Table 6.
- The current energy model is `not_assessed`. Any modeled-energy analysis must
  bind a repository-relative constants JSON, its SHA-256, and its cited source
  in `benchmark.energy_model` before protocol freeze; CLI substitution after
  freeze is forbidden.
- Local Jacobian moments use fixed-batch Hutchinson/Rademacher probes and are
  conditioned on the captured pre-invocation SNN state. They are not the full
  temporal network Jacobian.
- Historical engineering record: pre-freeze CUDA feasibility smoke on the
  recorded RTX 4070 Laptop found
  CIFAR-10/depth-56/T=6 C2 OOM at batch sizes 256 and 128; batch size 64
  completed C2 and C4. A synthetic, explicitly non-reportable DVS tensor with
  shape [B, T=10, C=2, H=48, W=48] also completed C2 and C4 at batch size 64.
  See `environment/pre_freeze_gpu_smoke_20260721.md`; the depth-56 checks are
  outside the revised 40-run design. These runs do not supply scientific
  results, support a depth claim, or replace the authors' batch-size approval.

## Sign-off evidence

Before setting `protocol_status.frozen: true`, retain a dated author record that
confirms:

1. the reconstructed C1-C4 graph is the intended intervention;
2. the LIF versus TA-LIF gradient-path distinction is correct;
3. all provisional numerical and data-preprocessing choices are accepted;
4. validation thresholds, the 0.50-percentage-point interaction SESOI, wording
   gates, and descriptive efficiency tolerances were accepted without seeing
   the joint test results;
5. the test-set gate and one-time checkpoint evaluation rule are accepted.
