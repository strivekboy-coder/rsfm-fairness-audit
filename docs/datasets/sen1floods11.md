# Sen1Floods11 Dataset Note

## Official Sources

- Official repository: https://github.com/cloudtostreet/Sen1Floods11
- Paper: http://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html

## Verified Facts

The official repository describes Sen1Floods11 as a georeferenced dataset for training and testing flood algorithms for Sentinel-1. It is hosted in a Google Cloud Storage bucket and includes data, splits, and a STAC catalog.

Official documentation lists Sentinel-1 imagery with VV and VH bands, Sentinel-2 Level-1C imagery with 13 bands, and hand-labeled water masks. Files are projected to WGS84 at 10 m ground resolution.

## Fairness Use

This dataset is well suited for flood-event, country, orbit, sensor, and label-source fairness. The first subset should use hand-labeled chips only, grouped by flood event and ISO country.

For this project, chip-level Sen1Floods11 classification is a sanity audit only.
The paper-grade path is native pixel-level flood segmentation with Sentinel-2
hand-labeled chips, `LabelHand` masks, valid-pixel handling, and event-level
aggregation from TP/FP/FN/TN counts. BWER is a support-aware,
composition-standardised, CVaR-style tail-risk statistic for
deployment-relevant remote sensing slices; in Sen1Floods11, `event_id` should be
read as an operational disaster-event slice rather than a causal country
fairness attribute.

The current 512 native segmentation prepared zip can be reused by both the
official Prithvi TL audit and the supervised U-Net baseline. For U-Net,
`LabelHand` pixels are interpreted as `0=background`, `1=water/flood`, and
`-1=ignore`; ignore pixels are excluded from loss and from TP/FP/FN/TN metrics.
Chip-level macro IoU is auxiliary. Formal BWER inputs should be event-level
rows aggregated from valid pixel counts and confusion counts.

Claim guardrail: Sen1Floods11 results in this project support event-level
tail-risk evidence for this specific case study. They should not be generalized
to all disaster segmentation tasks, all flood settings, or all disaster
geographies without additional datasets and protocols.

## Open Items

- Current official license or terms in the GCS bucket: to_verify.
- Exact current train/validation/test split files in v1.1: to_verify.
- Mapping from event metadata to biome/ecoregion slices: to_verify.
