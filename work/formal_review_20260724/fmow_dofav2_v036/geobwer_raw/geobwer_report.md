# GeoBWER Audit Report

- Metric version: `geobwer_fractional_1.1`
- Protocol hash: `f51e772730ce48a8a203e9768112419e396e96a7fd88bb299342896456a31a16`
- Primary beta: `0.1`
- Audit measure: `balanced`
- Partition rule: `one_axis_at_a_time`
- Estimand scope: `fixed_slice_universe`
- Dependence design: `independent_clusters`
- Inference: `cluster_maxt`

| Axis | Validity | Mean risk | Tail risk | GeoBWER | 95% CI | LCB |
|---|---|---:|---:|---:|---:|---:|
| country | descriptive_only | 0.824161 | 1.000000 | 0.175839 | [0.000000, 0.883144] | 0.000000 |
| class_label | descriptive_only | 0.817119 | 0.997566 | 0.180447 | [0.000000, 0.726985] | 0.000000 |
| country_class | descriptive_only | 0.731231 | 0.998534 | 0.267302 | [0.000000, 0.900000] | 0.000000 |

`apparent` point estimates describe the observed audit table. A positive LCB is required for a certified positive disparity claim.
