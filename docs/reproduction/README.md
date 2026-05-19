# Reproduction Notes

Official-source findings for real models and datasets should be recorded here
before implementation. Milestone 1 intentionally avoids guessing checkpoint
names, input shapes, preprocessing, licenses, or download procedures.

Operational Colab and Google Drive artifact-management lessons are recorded in
`colab_drive_lessons_learned.md`.

Post-hoc BWER-Audit v2 enrichment for completed segmentation runs is documented
in `bwer_v2_posthoc.md`.

The supervised U-Net Sen1Floods11 native segmentation baseline is documented in
`unet_sen1floods11.md`. It is Protocol C (`adaptation_protocol =
supervised_baseline`) and produces the same BWER-compatible event-level outputs
as the Prithvi TL segmentation run. The same guide documents the standalone
Prithvi-vs-U-Net comparison workflow; comparison outputs live under
`outputs/comparisons/` rather than inside either single-model run directory.
