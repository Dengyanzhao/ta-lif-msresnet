# V6 mechanism-study termination record

Status: TERMINATED AFTER CONSUMED NON-REPORTING CIFAR-100 HEALTH FAILURE

This is a post-hoc administrative record. It does not alter the frozen V6
protocol, its attempt receipt, its health report, or its terminal log. The V6
gate result remains `FAIL`; the implementation defect documented below must not
be used to rewrite or reinterpret that canonical outcome as a pass.

## Immutable evidence identity

- Git commit: `d83857786e2470bccfd2dac52c8b4e50b0f00b6f`
- Protocol SHA-256: `c13b9cc568c9a26303c39de80447b1ac1048f8c1d46add6c9f747bad7691fecf`
- Dataset: `cifar100`
- Consumed health seed: `1707261715`
- Reserved but unconsumed pilot seed: `22068314`
- Unused-seed disposition: `RESERVED_UNUSED_RETIRED_WITH_V6`
- Attempt receipt SHA-256: `adbe836d70b9b04c1afc7b2d6e9ca24f7dc426078baf669bbadef7bfcbf71a41`
- Health report SHA-256: `3ce50865947dfa53ed3072ce24c88cbb3d7bbc9383c4a52172ed74b4908da55d`
- Terminal log SHA-256: `9194eed41e6e7b6da5fc02767bd6446040b77a11fe5642d92de7315220caa66d`
- Exit-code file SHA-256: `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`
- Health status: `FAIL`
- Runtime failure: none

## Frozen-gate outcome

M0 and M1 were recorded as failing the aggregate implementation-health check.
M2, M3, M4, and PLIF passed. For both M0 and M1, every underlying numerical,
fixed-batch, schedule, gradient-route, and exact-resume probe recorded by the
report passed. Initial-forward equivalence and shared initialization also
passed for the complete six-condition block.

The report recorded different identities for the preclaim and execution fixed
batches:

- Preclaim fixed-batch SHA-256: `0d006507545d6ff270e598091c10f9ef7e33796f7ecc7d72479ddac4bcd9edfe`
- Execution fixed-batch SHA-256: `e6cabb52787acc266a74eae0a7d876ce89a8dce4ee4777e59c1349b82469a9da`

## Implementation finding

The V6 aggregate condition predicate interpreted an empty adaptive-parameter
list for nonadaptive M0/M1 as a positive adaptive update and then compared it
with the opposite expected value. This produced a false aggregate failure even
though the underlying no-adaptive-parameter update probe passed. Separately,
the V6 preclaim loader was constructed before the deterministic health seed was
applied, so its fixed-batch identity was not bound to the execution batch.

These are implementation findings about the V6 gate. They do not authorize a
V6 retry, pilot, formal run, or post-hoc change to V6 evidence.

## Required continuation

The frozen V6 run-handling rule requires a new protocol version and new unused
seeds after a health failure. Continuation therefore uses V7 with an isolated
namespace, artifact paths, run identifiers, health seed, pilot seed, formal
seeds, bootstrap seeds, and benchmark seeds. The six-condition scientific
design remains unchanged unless a future author-signed protocol explicitly
states otherwise.

The following 26 V6 values were reserved but never executed and are retired
with the exact disposition `RESERVED_UNUSED_RETIRED_WITH_V6`:

`22068314`, `1587277406`, `187544621`, `990604905`, `969985360`,
`1718581689`, `465608478`, `1670924910`, `1825076945`, `1128314682`,
`65044423`, `774715468`, `508189787`, `1956664330`, `367144660`,
`1169352105`, `1537480873`, `2048258242`, `786414499`, `1691781719`,
`897483192`, `2138200647`, `1641516093`, `1811716963`, `1134185216`, and
`38481167`.

## Prohibited actions

- Never rerun V6 health seed `1707261715`.
- Never run V6 pilot seed `22068314`; it is retired unexecuted.
- Never run any V6 formal, bootstrap, benchmark, or conditional-extension seed.
- Never edit, replace, delete, or relabel the canonical V6 receipt, report, log,
  or exit-code file.
- Never copy V6 failure evidence into a V7 result root.
- Never report V6 health values as manuscript results.
