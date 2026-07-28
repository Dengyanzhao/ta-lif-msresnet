# MS-ResNet route closure and evidence-preservation record

Decision date: 2026-07-28 (Asia/Shanghai)

Status: **CLOSED FOR THE CURRENT MANUSCRIPT; RETAINED AS FUTURE WORK ONLY**

This is a scope decision for the current TA-LIF paper. It does not delete,
reinterpret, repair, or otherwise change any historical experiment, source
code, checkpoint, or audit artifact.

## Scope now closed

The current manuscript will not make a result claim about MS-ResNet. In
particular, it will not report or analyse the C3 (`MS-ResNet + LIF`) or C4
(`MS-ResNet + TA-LIF`) conditions, the topology main effect, the TA-LIF by
topology interaction, or wording such as `synergistic`, `complementary`, or
`composable`.

No unfinished C3/C4 run may be resumed. No result from formal v1, v2r1, v2r2,
or the MS-ResNet diagnostic may enter the manuscript's result tables, figures,
or confirmatory analysis.

## Preservation requirements

1. Retain the `ms_resnet` implementation, diagnostics, tests, and legacy
   C1--C4 protocol/configuration support unchanged for audit and future work.
2. Retain all historical failure records, attempt logs, checkpoints, manifests,
   standard streams, and environment receipts. Do not overwrite or delete them.
3. Keep historical outputs separate from any v3 output root. A v3 run must
   never resume a v1/v2 checkpoint or write into a historical result directory.
4. The previously used seeds `11, 22, 33, 44, 55, 77, 88, and 314159` are
   retired from v3. They cannot be rerun, substituted, or used for v3 pilot or
   formal analysis.

## Evidence inventory

The following are retained audit evidence, not reportable scientific results.
Some pilot artifacts are kept in downloaded archives and therefore are not
assumed to be present in every checkout.

| Record | Disposition | Integrity reference |
|---|---|---|
| Formal v1 incident | Withdrawn from reporting | `FORMAL_V1_INCIDENT.md` |
| Formal v1 cloud archive | Preserve for audit | `results/archives/formal_v1_20260727/ta_lif_formal_v1_incident_20260727.tar.gz`, SHA-256 `04ebe645cb1181b2d004ef4e561717b5f34077c14fc011b9020119d686485be6` |
| v2r1 seed-77 health failure | Engineering-only; seed consumed | health report SHA-256 `6c79e0b721c787b29ae2c4dbdd2a7ae104848397a92cbab94aab4b1cf987097e`; downloaded archive SHA-256 `d8b310644615abaea24453752671b44a8dce41d7e58028890bac98d75be05ada` |
| v2r2 seed-88 health failure | Engineering-only; seed consumed | health report SHA-256 `6aaa1d0ae2c046187e6c348449dd998668a522f87769ddaac88f5df1f511412f`; downloaded archive SHA-256 `3c4927153161d7240f742a6bdc87b025470ff0e9a7f25fcabefe490517b46d75` |
| MS-ResNet BatchNorm diagnostic, seed 314159 | Engineering-only error; seed consumed | result SHA-256 `9ec5bebff4d218d8bbe00566b75103a2e35274a594e072f9370d5324a11e5f8a`; attempt receipt SHA-256 `680ebc7aed1e76642579a8ab1f51ae3e707f43177c6159a5da885dd79c3c8a47`; archive SHA-256 `d0426df3bcd3cf7b5ca56dbe5871b90f18930cfaa64f8dd8875b21ad882642ee` |

The seed-314159 diagnostic stopped before a root-cause conclusion because
strict PyTorch deterministic algorithms were not enabled. It is not evidence
that establishes, rules out, or explains a scientific MS-ResNet effect.

## Future-work boundary

MS-ResNet may be revisited only under a separately approved future-work
protocol with a new implementation review, fresh seeds, a new pilot, and a
new author freeze. Such work must remain separate from the current TA-LIF-only
v3 manuscript route unless the authors make a later, documented scope change
before any reportable v3 result is used.

## Active transition

The active design path is `V3_TALIF_ONLY_PROTOCOL_DRAFT.md`. It defines a
two-condition C1/C2 study draft only. It is not frozen, it authorizes no GPU
execution, and it does not alter any legacy protocol.
