# Sen1Floods11 全19模型统一描述性后处理

该步骤只读取已经冻结并通过产物审计的 U-Net、Prithvi 和 TerraMind 概率图，不训练、不推理、不运行 bootstrap，也不生成正式推断性 GeoBWER。

它统一使用 `per_chip_one_minus_flood_iou_at_probability_0.5`，输出 validation89、standard test90、Bolivia15 和 combined held-out105 的跨架构可比结果，同时保留 pooled-pixel IoU 作为次级诊断。

在全新 CPU Colab 中 clone/pull 冻结提交后运行一格：

```python
from google.colab import drive
drive.mount('/content/drive')

import subprocess

PROJECT = '/content/rsfm-fairness-audit'
DRIVE = '/content/drive/MyDrive/rsfm_fairness_audit'

command = [
    'python', '-u',
    f'{PROJECT}/scripts/colab/postprocess_sen1_19model_descriptive_colab.py',
    '--unet-root', f'{DRIVE}/outputs/geobwer_final_v3/sen1_geobwer_v0428/supervised',
    '--prithvi-root', f'{DRIVE}/outputs/geobwer_final_v3/sen1_geobwer_v0432/prithvi_final',
    '--terramind-root', f'{DRIVE}/outputs/geobwer_final_v3/sen1_geobwer_v0434/terramind_final',
    '--core-metadata', f'{DRIVE}/cache/sen1_prithvi_tl/metadata.csv',
    '--bolivia-metadata', f'{DRIVE}/cache/sen1_prithvi_tl_bolivia15_v0432/metadata.csv',
    '--geospatial-metadata', f'{DRIVE}/cache/sen1floods11/sen1_geospatial_metadata_446_v0426.csv',
    '--unet-audit', f'{DRIVE}/outputs/geobwer_final_v3/00_audit_evidence/sen1_v0428_unet_artifact_audit_v1.json',
    '--prithvi-audit', f'{DRIVE}/outputs/geobwer_final_v3/00_audit_evidence/sen1_v0432_prithvi_artifact_audit_v1.json',
    '--terramind-audit', f'{DRIVE}/outputs/geobwer_final_v3/00_audit_evidence/sen1_v0434_terramind_descriptive_artifact_audit_v1.json',
    '--output-dir', '/content/sen1_19model_descriptive_v2',
    '--persistent-output-dir', f'{DRIVE}/outputs/geobwer_final_v3/sen1_19model_descriptive_v2',
]
subprocess.run(command, check=True)
```

唯一成功标志：

```text
SEN1_19MODEL_DESCRIPTIVE_POSTPROCESS=PASS
```

主要输出：

- `unified_19model_metrics.csv`
- `three_seed_architecture_modality_summary.csv`
- `event_level_metrics.csv`
- `same_seed_modality_rankings.csv`
- `modality_ranking_stability.csv`
- `prediction_degeneracy_diagnostics.csv`
- `source_contract.json`
- `postprocess_manifest.json`
- `completion_contract.json`
- `scientific_interpretation_report.md`

冻结的 `sen1_geospatial_metadata_446_v0426.csv` 是 latitude/longitude 的唯一来源；core431 与 Bolivia15 metadata 只提供 event、split 等非坐标属性。

脚本先检查 `/content/sen1_19model_descriptive_work/staged_probability_sources`。若现有57组 staging 的源索引、逐概率文件 SHA 和冻结 Drive 源完全一致，则直接复用；任一不一致都会硬失败且不会重新复制。全新 runtime 才会首次复制概率导出到 `/content`。Drive 冻结源不会被修改，非空但不完整的输出目录也会硬失败。
