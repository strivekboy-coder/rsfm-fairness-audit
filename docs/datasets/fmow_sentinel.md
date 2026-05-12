# fMoW / fMoW-Sentinel Dataset Note

## Official Sources

- Official fMoW GitHub organization: https://github.com/fmow
- Official dataset repository: https://github.com/fMoW/dataset
- IARPA challenge page: https://www.iarpa.gov/challenges/fmow.html
- Paper: https://arxiv.org/abs/1711.07846

## Verified Facts

The official fMoW dataset provides global functional land-use classification data. The official repository documents two variants: `fMoW-full`, a large TIFF dataset with 4-band and 8-band multispectral imagery, and `fMoW-rgb`, a smaller JPEG RGB version.

The official README states that metadata and ground-truth releases include raw metadata and GPS coordinates, which can support geographic fairness slicing.

## fMoW-Sentinel Status

An official fMoW-Sentinel dataset release was not verified in this pass. Community checkpoints or derived datasets exist, but they are not recorded as official dataset sources for this project. The registry therefore uses official fMoW as the global classification fallback and marks Sentinel-specific fMoW usage as unavailable/to_verify.

## Fairness Use

Use official fMoW only after a metadata-first audit. Start with RGB or metadata-only subsets for global slice design, then decide whether multispectral data is feasible.

## Open Items

- Official Sentinel-derived fMoW release, if any: to_verify.
- Country/coordinate field quality and invalid country-code handling: to_verify.
- License constraints for publication and model training: to_verify with official license text.
