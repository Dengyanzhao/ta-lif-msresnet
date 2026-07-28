# TA-LIF-only v3 execution runbook

This runbook is intentionally staged. The checked-in v3 protocol is currently
unsigned and unfrozen; no v3 training command is authorized yet.

## Phase A: author/source freeze

1. Review `configs/protocol_v3_talif_only.yaml`, the source changes, the pilot
   and formal matrix rules, and `PREREGISTRATION_SIGNOFF_V3_TALIF_ONLY.md`.
2. All three authors complete the checklist and approval evidence. Set
   `protocol_status.frozen: true`, enter all names and a timezone-aware
   `confirmed_at`, and replace the sign-off status with the exact signed marker.
3. Commit only the reviewed protocol/sign-off/source change. Record the full
   40-character commit ID. Do not include generated v3 matrices or a freeze
   manifest in this commit.

## Phase B: pilot and generated artifacts

1. In the clean Phase A checkout, run the v3 pilot matrix generator for the
   isolated path `configs/v3_talif_only_pilot_generated`, then generate the
   formal matrix at `configs/v3_talif_only_generated`.
2. Run the ordinary preflight and verify that the two isolated matrices contain
   exactly four pilot and twenty formal C1/C2 configurations.
3. Run exactly one dataset-specific v3 health gate for CIFAR-100 and exactly
   one for CIFAR10-DVS. Each gate claims its own pilot seed before compute;
   PASS, FAIL, ERROR, and interruption all consume that seed and are never
   retried.
4. Launch only the matching two-run pilot block for a dataset whose health
   report is a strict PASS. Run `scripts/validate_v3_pilot.py` only after both
   dataset blocks complete. The aggregate report must be `PASS` and is
   non-reportable.
5. Create and verify `FREEZE_MANIFEST_V3_TALIF_ONLY.json`. It binds the Phase A
   commit, signed record, protocol, both pilot health blocks, pilot plans, and
   formal matrix. Archive its SHA-256 before formal training.

## Formal execution and analysis

1. Run full preflight and the complete isolated 20-run matrix. Resume only an
   interrupted run from its same-environment `last.pt`; never substitute a
   seed, start a fresh retry, or resume across environments.
2. Audit every run and checkpoint. Access each independent test set exactly
   once only after all twenty frozen runs pass the audit.
3. Run `scripts/analyze_v3_results.py` on the frozen formal root. Keep pilot,
   failed, and historical MS-ResNet artifacts outside the reportable analysis.

Any protocol, threshold, schedule, seed, or output-path change after Phase A
requires a new protocol version and a new author decision.
