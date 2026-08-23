# V7 Benchmark Compatibility Recovery

## Scope

This record documents a post-training evaluator defect discovered after the
48-run CIFAR-100 formal matrix completed. It does not alter the protocol,
training seeds, checkpoints, validation split, or test-data boundary.

## Evidence

- Formal training release commit: `b780b38322b491f849592d71814b7eb44de16948`
- Failed benchmark log: `/hy-tmp/v7-benchmark.log`
- Failed benchmark exit sidecar: `/hy-tmp/v7-benchmark.exit`
- Fixed validation batch file SHA-256: `cf1557a2192d337d9f01f18c85c190fd4ad98b8eb750bde0b2fe80150597906f`
- Fixed validation batch content SHA-256: `c71537580ee9f8d17f7d66d85027f73691f12a533fd87379bf268d28aa9f3869`

The benchmark stopped at the first checkpoint with `KeyError: 'total'` while
building the in-memory measurement payload. No benchmark result JSON or
benchmark receipt was written before the failure. No test loader was created
or accessed.

## Defect and correction

`CIFARSpikingResNet.parameter_report()` canonically emits
`total_parameters` and `trainable_parameters`. The V7 benchmark adapter
incorrectly looked up `total` and `trainable`. The compatibility correction
maps the canonical model fields to the frozen benchmark result schema fields:

```text
total_parameters     -> measurements.parameters.total
trainable_parameters -> measurements.parameters.trainable
```

This correction is evaluator-only and is applied after formal training. The
formal checkpoints remain bound to the original training release and are not
retrained or replaced.

## Verification

The correction has a regression test covering the adapter mapping. The V7
benchmark schema continues to require the result fields `total` and
`trainable`; no schema relaxation or result fabrication is permitted.
