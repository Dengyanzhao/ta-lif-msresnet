# Formal v1 validity incident and preservation record

Incident ID: `formal-v1-20260727`

Status: **WITHDRAWN FROM REPORTING; PRESERVE FOR AUDIT ONLY**

This record separates the first frozen formal experiment (formal v1) from all
later engineering, pilot, and formal runs. It is an incident record, not an
attempt to reinterpret or repair v1 results.

## Frozen identity

- Phase A freeze commit:
  `369a96b142532e9084bffbae008942477e05fcfd`
- Canonical protocol SHA-256:
  `bec83ee5145518f2dd81e02bdf3d3499021c5c86489b5d54655ff9f2f57fcf1e`
- Phase B freeze-manifest file SHA-256:
  `3137389accdbfddfecd50ff1cfa74ce4f1179ccfc41d2a0b2e8b9001e924778d`
- Planned design: two dataset groups, C1-C4, five fixed seeds, 240 epochs,
  40 runs in total.

These identifiers define v1. A result with another protocol or configuration
identity is not part of v1 even if its run identifier has the same spelling.

## Triggering observations

The first five completed CIFAR-100 C1 runs did not learn. Validation accuracy
remained at approximately `0.01`, the 100-class chance level, and final training
loss was approximately `4.605178`, effectively `ln(100)`. The affected runs
were the complete C1 seed block (`11, 22, 33, 44, 55`). This is a structural
validity failure, not an accuracy outlier and not a failed seed that may be
silently replaced.

In the first observed CIFAR-100 C2 run, epochs 1-12 (before the delayed TA-LIF
update became active) took approximately 170-178 seconds each. Epochs 13-16
(after activation) took approximately 10,493-10,498 seconds each. The observed
step-time change is about `60.4x`. Accuracy remained at approximately `0.01`.
This makes the frozen implementation both scientifically invalid for its
intended comparison and operationally impractical.

The numerical observations above were transcribed from the preserved cloud
run logs and operator console output. The archived files, rather than this
summary, are the primary audit evidence.

## Disposition

1. No v1 accuracy, convergence, diagnostic, latency, memory, or energy result
   may be reported in the Knowledge-Based Systems manuscript.
2. No completed, failed, or partial v1 run may be combined with a repaired v2
   run. In particular, the five completed C1 runs cannot be reused.
3. V1 run directories, checkpoints, manifests, standard streams, event logs,
   process metadata, and environment records must be retained unchanged.
4. A code repair creates a new implementation and therefore requires a new
   non-reportable pilot, a new protocol identity, renewed three-author approval,
   and new Phase A/Phase B freeze artifacts before formal execution.
5. V2 must use a separate output root. It must never resume from a v1
   checkpoint or write into `results/runs` while that path contains v1 data.
6. The v1 incident archive is audit evidence only. Its existence does not make
   any v1 result reportable.

## Controlled stop and archive

`scripts/archive_formal_v1.py` is designed for the cloud repository that holds
the v1 run directory. Its default action is read-only inspection:

```text
python scripts/archive_formal_v1.py
```

It discovers relevant `run_matrix.py` and `talif_msresnet.train` processes from
the live process table and resolves their output paths using `/proc/<pid>/cwd`.
It does not trust a copied PID.

Stopping is a separate, explicit operation. It sends `SIGTERM` only; it never
sends `SIGKILL`, deletes a run directory, or restarts a run. The receipt path
must be new, and the exact confirmation token is required:

```text
python scripts/archive_formal_v1.py --stop \
  --confirm-stop STOP_FORMAL_V1_369A96B \
  --stop-receipt environment/formal_v1_stop_20260727.json
```

Inspect the receipt and confirm that no matching or ambiguous writer remains.
If a process does not exit after `SIGTERM`, preserve that fact and investigate;
do not delete locks or checkpoints and do not escalate to a forced kill without
a separate, documented author decision.

Create the archive only after the inspection reports no active or ambiguous v1
writer. Choose a new destination outside every archived input:

```text
python scripts/archive_formal_v1.py \
  --archive /hy-tmp/ta_lif_formal_v1_incident_20260727.tar.gz
```

Archive creation refuses an existing destination and refuses identity conflicts.
The archive contains the formal result tree plus its protocol, generated matrix,
freeze manifest, sign-off record, available environment records, and the
incident/archive-tool sources. The `_audit` directory inside the archive holds:

- the incident identity and non-reportable disposition;
- repository metadata and tracked-worktree status;
- a result/run summary and protocol-identity scan;
- process snapshots taken immediately before and after capture;
- a complete inventory and SHA-256 for every archived regular file;
- line counts and termination status for log and event-stream files.

Record the archive SHA-256 printed by the command in the external transfer
receipt. After download, verify it independently with `sha256sum` before the
cloud copy is retired. The archive command never removes the source tree.

## Follow-up boundary

Root-cause repair, pilot acceptance thresholds, reduced v2 design, and v2
freezing belong to separate records. They must cite this incident but must not
alter this record or the archived v1 evidence.
