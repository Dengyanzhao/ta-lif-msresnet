# TA-LIF-only v4 protocol draft

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-29T13:37:48+08:00
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1
- Upstream Dependencies: `MS_RESNET_ROUTE_CLOSURE.md`;
  `PREREGISTRATION_SIGNOFF_V3_TALIF_ONLY.md`;
  `configs/protocol_v3_talif_only.yaml`; source commit
  `0cecd61250c1e7b4650a925e420320f7fe4f8f10`; failure archive SHA-256
  `aa3a8f0c6c602cfc03e4d13a569182b10205beb81c9dbf8f2f235657b5dd4a00`

Protocol Status: **DRAFT - AUTHOR CONFIRMATION REQUIRED; EXECUTION NOT AUTHORIZED**

Protocol Version: `v4-draft`

## Status and authority boundary

This file is a planning artifact for a possible v4 engineering repair. It is
not a preregistration sign-off, executable protocol, frozen configuration,
pilot authorization, or formal-run authorization. It does not amend or reopen
v3. No v4 YAML, generated matrix, freeze manifest, training command, or seed
claim is authorized by this draft.

The candidate changes below remain engineering hypotheses until the authors
confirm the complete model graph, optimizer contract, monitoring gates, new
seed, output paths, environment, and exact command. Passing code tests alone
must not be treated as author approval or permission to run a pilot.

## Closed v3 outcome and evidence boundary

The v3 CIFAR-100 C1/C2 pilot, using seed `474123945`, completed but failed its
prespecified scientific acceptance threshold. The archived run showed best
validation accuracy of `0.0738` for C1 and `0.1818` for C2; both runs later
collapsed to chance-level behavior. These values and all v3 pilot diagnostics
are non-reportable engineering evidence and must not enter the manuscript's
Results, figures, tables, or formal statistical analysis.

The local failure archive is:

```text
ta_lif_v3_cifar100_pilot_FAIL_0cecd612_s474123945_20260729_111556.tar.gz
SHA-256: aa3a8f0c6c602cfc03e4d13a569182b10205beb81c9dbf8f2f235657b5dd4a00
```

Its integrity was verified, and the root-cause review is `ANALYZED` without a
reproducibility rerun. The v3 no-retry rule remains in force. Under v3, formal
execution and test-set access are permanently blocked. The v3 CIFAR10-DVS
pilot was not run; its seed `799312121` remains unconsumed and must not be
started, reused, or transferred to v4.

## Fixed current-manuscript scope

| Element | Required v4 boundary |
|---|---|
| Research comparison | TA-LIF versus LIF on one fixed conventional Spiking ResNet topology |
| Baseline | C1: Spiking ResNet + LIF |
| Treatment | C2: Spiking ResNet + TA-LIF |
| Active conditions | C1/C2 only |
| Closed conditions | C3/C4 and every MS-ResNet reporting route |
| Forbidden claims | Topology effects, interactions, synergy, complementarity, and composability |
| Historical evidence | Preserve unchanged for audit; never pool with v4 output |
| v4 pilot role | Non-reportable engineering validation only |

MS-ResNet remains future work requiring a separate protocol, implementation
review, new seeds, pilot, and author freeze. It is not part of v4.

## Author-confirmed scientific decisions

### Dataset and formal-matrix scope

On 2026-07-29, the author selected scope option **B: dual-dataset journal
version**. This decision fixes the intended scientific role and size of the
formal matrix, but it does not select any seed value, approve a configuration,
freeze the matrix, or authorize execution.

| Dataset | Scientific role | Paired formal blocks | Formal trainings |
|---|---|---:|---:|
| CIFAR-100 | Primary experiment | 5 new C1/C2 paired-seed blocks | 10 |
| CIFAR10-DVS | Second-stage external-validity replication | 5 new C1/C2 paired-seed blocks | 10 |
| **Total** | Two dataset-specific analyses | **10 paired-seed blocks** | **20** |

Within each paired block, C1 and C2 must use the same approved seed and aligned
split, initialization, and data-order controls. Each dataset therefore requires
five distinct, author-approved seed identities. Exact seed values, derivation,
historical-value exclusions beyond those already forbidden below, and whether
the two dataset seed sets must be disjoint remain
`AUTHOR-CONFIRMATION-REQUIRED`.

CIFAR-100 proceeds first. CIFAR10-DVS is not an auxiliary run appended to the
CIFAR-100 gate: it requires its own candidate configuration review, health
gate, and non-reportable pilot. The complete 20-training formal matrix may be
frozen only after both dataset-specific health gates and both dataset-specific
pilots pass their prospectively approved criteria. Pilot outputs cannot be
counted among the 20 formal trainings or reported as manuscript results.

### Terminal-neuron graph

On 2026-07-29, the author selected model-graph option **A**. Every v4 C1 and C2
health gate, pilot, and formal run on both CIFAR-100 and CIFAR10-DVS must use:

```yaml
terminal_neuron_mode: topology_required
```

For the conventional Spiking ResNet topology used by C1/C2, the final residual
block already emits spikes. The repeated terminal spike gate is therefore
replaced by an identity operation, giving both C1-LIF and C2-TA-LIF 19 neuron
layers instead of the legacy 20. This graph is shared across the two conditions;
the neuron type remains the intended independent variable.

The legacy `always` mode remains available only for historical compatibility
and audit reproduction and is not eligible for any v4 experimental run. C3/C4
and MS-ResNet remain closed and are not reopened by this decision. This author
confirmation fixes the scientific graph but does not authorize creation of a
YAML, checkpoint conversion, health gate, pilot, or formal run.

## Seed ledger

| Seed or set | Status | v4 disposition |
|---|---|---|
| `11, 22, 33, 44, 55, 77, 88, 314159` | Historical/retired | Forbidden |
| `474123945` | v3 CIFAR-100 pilot consumed by the failed attempt | Forbidden |
| `799312121` | v3 CIFAR10-DVS pilot reserved and unconsumed | Keep reserved in v3; forbidden for v4 |
| `1882214332, 836017246, 126468528, 2075015328, 419442269` | Frozen to the unexecuted v3 formal design | Do not transfer automatically; v4 policy requires author confirmation |
| `1033863572, 1367073951` | v3 bootstrap-analysis seeds, not training seeds | Forbidden as v4 training seeds |
| v4 CIFAR-100 pilot seed | Not selected | New, unused, disjoint seed; `AUTHOR-CONFIRMATION-REQUIRED` |
| v4 CIFAR10-DVS pilot seed | Not selected | Eligibility and relationship to other v4 seeds require separate author confirmation |
| v4 CIFAR-100 formal paired-seed set | Five values required; not selected | Exact values and eligibility require separate author confirmation |
| v4 CIFAR10-DVS formal paired-seed set | Five values required; not selected | Exact values and cross-dataset relationship require separate author confirmation |

The exact v4 seed derivation namespace, labels, collision exclusions, and
resulting integer values are not specified by this draft. They must be fixed
prospectively and confirmed before any seed-claiming artifact is created.

## v4 engineering decision register

Only entries explicitly marked `AUTHOR-CONFIRMED` are fixed. Every other entry
remains a candidate rather than an approved value.

| Area | Candidate | Status |
|---|---|---|
| Terminal neuron graph | `terminal_neuron_mode: topology_required`; omit the repeated terminal spike gate for C1/C2 on both datasets while preserving the legacy audit path | `AUTHOR-CONFIRMED: OPTION A, 2026-07-29` |
| Surrogate function | Retain the current equation initially and add measured support coverage; no equation change is approved | `AUTHOR-CONFIRMATION-REQUIRED` |
| Weight decay grouping | Apply decay to convolution/linear weights; exclude normalization parameters, all biases, and TA parameters | `AUTHOR-CONFIRMATION-REQUIRED` |
| Optimizer | SGD, momentum candidate `0.9` | `AUTHOR-CONFIRMATION-REQUIRED` |
| Batch size | Candidate `64` | `AUTHOR-CONFIRMATION-REQUIRED` |
| Base learning rate | Candidate `0.025`; linear-scaling rationale remains an engineering hypothesis | `AUTHOR-CONFIRMATION-REQUIRED` |
| Warmup | Candidate `5` epochs | `AUTHOR-CONFIRMATION-REQUIRED` |
| Gradient clipping | Candidate global norm `1.0` | `AUTHOR-CONFIRMATION-REQUIRED` |
| Scheduler | Warmup interaction, milestones, gamma, and resume semantics not yet fixed | `AUTHOR-CONFIRMATION-REQUIRED` |
| TA activation | Start epoch/fraction, TA learning-rate scale, and freeze behavior not yet re-approved | `AUTHOR-CONFIRMATION-REQUIRED` |
| Pilot duration | Not selected | `AUTHOR-CONFIRMATION-REQUIRED` |
| Pilot acceptance | Best accuracy, late-epoch stability, and failure thresholds not selected | `AUTHOR-CONFIRMATION-REQUIRED` |

The v3 `0.60` pilot threshold is not silently carried forward or relaxed. In
particular, the observed v3 C2 value `0.1818` must not become a post hoc v4
success threshold. The authors must approve an independently justified v4
criterion before execution.

## Required pre-pilot verification

Code-level verification may proceed without consuming a seed, but it cannot
authorize training. Before author sign-off, the engineering repair should have
tests that establish at least the following:

1. The legacy/default terminal-neuron behavior remains available for audit.
2. The candidate C1/C2 graph omits only the specified repeated terminal gate.
3. Surrogate support coverage is computed from exact numerator and denominator
   counts and is reset correctly between measurements.
4. Optimizer parameter groups contain every trainable parameter exactly once.
5. Convolution/linear weights receive the intended decay while normalization,
   bias, and TA parameters receive none.
6. Warmup learning-rate sequencing and checkpoint resume are deterministic.
7. Resume preflight binds the prior run ID, configuration hash, environment,
   and optimizer-group identity to `last.pt`; rejection leaves every existing
   run artifact byte-for-byte unchanged.
8. Zero residual-block gradients, nonfinite metrics, and parameter-norm collapse
   remain visible instead of being dropped from summaries.
9. Existing regression tests, static checks, and import compilation pass.

## Draft health and futility gate requirements

The v3 80-step fixed-batch overfit gate was insufficient because it did not
test gradient survival across epochs. A v4 health gate must evaluate both C1
and C2 on the confirmed graph and the eventual author-confirmed optimizer. At
minimum it should record:

- nonzero residual-block gradient coverage by batch and epoch;
- batches with all residual-block gradients equal to zero;
- per-layer spike rate and surrogate support coverage;
- convolution, normalization, classifier, and TA parameter norms;
- finite train/validation loss and accuracy;
- TA activation-boundary behavior; and
- best-epoch performance together with late-epoch stability.

The observation window, aggregation rule, minimum gradient coverage, minimum
surrogate coverage, parameter-norm bounds, chance-level futility rule, late-
epoch stability criterion, and stop action are all
`AUTHOR-CONFIRMATION-REQUIRED`. No implementation default may silently become
the scientific gate.

## Candidate non-reportable pilot structure

| Field | Draft value |
|---|---|
| Pilot sequence | Stage 1: CIFAR-100; Stage 2: CIFAR10-DVS external-validity replication |
| Conditions | One paired C1/C2 block |
| Pairing | Identical new seed, split, initialization, and data-order controls |
| Seed | One seed per dataset pilot; exact values, eligibility, and relationship are `AUTHOR-CONFIRMATION-REQUIRED` |
| Dataset gate | Each dataset requires its own approved health gate and pilot acceptance criteria |
| Epochs | `AUTHOR-CONFIRMATION-REQUIRED` |
| Output root | New isolated v4 path, exact path `AUTHOR-CONFIRMATION-REQUIRED` |
| Retry rule | Candidate one-shot seed consumption on PASS, FAIL, ERROR, or interruption |
| Reportability | Forbidden from manuscript Results and formal inference |
| Formal release | Not provided by this draft or by a pilot alone |

The CIFAR10-DVS stage is retained by the scope decision, but remains blocked
until its separate author-confirmed gate, configuration, and new v4 pilot seed
are fixed. The unconsumed v3 DVS seed is not transferable. Passing only the
CIFAR-100 gate or pilot cannot release either dataset's formal matrix.

## Setup and command gate

- Language/framework: Python and PyTorch versions must be recorded from the
  final environment lock.
- Working directory: repository root at the author-approved source commit.
- Device/environment candidate: one homogeneous RTX 5090 environment using
  float32 and strict deterministic algorithms; exact runtime identity remains
  `AUTHOR-CONFIRMATION-REQUIRED`.
- Entry command: `AUTHOR-CONFIRMATION-REQUIRED; DO NOT RUN`.
- Inputs: author-approved v4 YAML, exact source commit, data provenance and split
  manifest, environment receipt, and confirmed seed receipt.
- Outputs: isolated health report, attempt receipt, event log, resolved config,
  checkpoints if authorized, and an archive with recorded SHA-256.
- Monitoring: process-alive and hard timeout plus all gradient, coverage, norm,
  loss, accuracy, and TA-boundary metrics listed above.

## Author confirmation gates

### Pre-health/pilot authorization gate

All open boxes must be completed from explicit author approval before any v4
GPU health-gate or pilot command is issued:

- [x] C1/C2-only scope and continued C3/C4 and MS-ResNet closure confirmed.
- [x] CIFAR-100 primary role, CIFAR10-DVS external-validity role, five paired
      formal seeds per dataset, and 20-training target matrix confirmed.
- [x] Exact C1 and C2 model graphs confirmed as
      `terminal_neuron_mode: topology_required` for both datasets.
- [ ] Exact optimizer groups, batch size, learning rate, warmup, scheduler,
      gradient clipping, and TA activation schedule confirmed.
- [ ] Exact dataset-specific health, futility, late-stability, and pilot
      acceptance thresholds confirmed.
- [ ] New v4 CIFAR-100 and CIFAR10-DVS pilot seeds and deterministic derivation
      records confirmed.
- [ ] Environment identity, input manifests, output paths, timeout, and
      monitoring files confirmed.
- [ ] One-shot/no-retry handling and non-reportable status confirmed.
- [ ] Exact health-gate and pilot commands confirmed verbatim.
- [ ] A separate v4 executable protocol and sign-off record reviewed and
      committed before health-gate or pilot execution.

### Post-pilot formal-release gate

This gate cannot be completed prospectively by the present scope decision. It
opens only after both dataset-specific health gates and pilots have completed:

- [ ] CIFAR-100 health gate and non-reportable pilot passed their frozen
      criteria.
- [ ] CIFAR10-DVS health gate and non-reportable pilot passed their frozen
      criteria.
- [ ] Two five-seed formal sets, their cross-dataset relationship, eligibility
      policy, and deterministic derivation records confirmed.
- [ ] Dataset-specific formal configurations, analysis plan, test-access rule,
      output paths, timeout, and monitoring contract frozen.
- [ ] All 20 exact formal-training commands confirmed verbatim.
- [ ] A separate formal executable protocol, freeze manifest, and sign-off
      record reviewed and committed before formal execution.

Until the pre-health/pilot gate is complete, do not create a frozen pilot YAML
or pilot freeze manifest and do not start either dataset's health gate or
pilot. Until the post-pilot formal-release gate is complete, do not create a
formal frozen YAML or formal freeze manifest, do not access a test set, and do
not start formal training. Both dataset-specific health gates and pilots must
pass before the 20-training formal matrix can be frozen or released.
