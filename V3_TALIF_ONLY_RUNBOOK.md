# TA-LIF-only v3 execution runbook

> **Terminal status (2026-07-29): V3 PILOT FAILED; V3 FORMAL PERMANENTLY
> BLOCKED.** The CIFAR-100 C1/C2 pilot consumed seed `474123945` and failed its
> prespecified scientific acceptance threshold. Under the signed no-retry rule,
> no remaining command in this runbook is authorized. Do not start the v3
> CIFAR10-DVS gate or pilot, run aggregate validation, create the v3 freeze
> manifest, start formal training, or access a test set. CIFAR10-DVS seed
> `799312121` remains unconsumed and reserved to v3; it must not be run, reused,
> or transferred to v4.

This runbook is retained as a historical record of the prospective v3 workflow.
The failure archive and all generated v3 artifacts must remain non-reportable
engineering evidence. Any continuation requires a new protocol version, new
unused seed, explicit author decision, and exact-command authorization. The
`UNVERIFIED` `V4_TALIF_ONLY_PROTOCOL_DRAFT.md` does not itself provide that
authorization.

## Phase A: author/source freeze (completed)

1. `configs/protocol_v3_talif_only.yaml`, the source changes, the pilot and
   formal matrix rules, and `PREREGISTRATION_SIGNOFF_V3_TALIF_ONLY.md` were
   reviewed.
2. The user supplied the collective written confirmation on behalf of all three
   authors. The checklist, approval evidence, names, timezone-aware
   `confirmed_at`, `protocol_status.frozen: true`, and exact signed marker are
   complete.
3. The final Phase A commit contains only the reviewed
   protocol/sign-off/source change. Generated v3 matrices and the freeze
   manifest remain absent; record the final 40-character commit ID after this
   corrected state is committed.

## Phase B: pilot and generated artifacts

Entering Phase B authorizes only the non-reportable validation workflow in this
section. It does not authorize formal training or test-set access.

1. In a clean checkout of the final Phase A commit, run the v3 pilot matrix generator for the
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
