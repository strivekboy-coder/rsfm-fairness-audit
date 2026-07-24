# v0.4.0 最终 Colab 运行顺序

## 总原则

1. GitHub checkout 后确认 `rsfm_fairness_audit.__version__ == "0.4.0"`，并把 `git rev-parse HEAD` 与 Codex 最终交接 SHA 对齐。
2. Drive 只作持久缓存；训练、LMDB 读取和推理都在 `/content`。
3. 新正式根目录使用 `geobwer_final_v3`，不得写回 `geobwer_final_v2`。
4. smoke 使用独立的 `/content/smoke_v040`，不得把 smoke 产物复制进正式根。
5. 任一正式作业失败立即停止；不要降低 checkpoint coverage、support、共同空间尺度或 bootstrap 门槛。

```bash
export REPO=/content/rsfm-fairness-audit
export LOCAL=/content/outputs/geobwer_final_v3
export DRIVE=/content/drive/MyDrive/rsfm_fairness_audit
export PERSIST=$DRIVE/outputs/geobwer_final_v3
cd "$REPO"
python - <<'PY'
import rsfm_fairness_audit
assert rsfm_fairness_audit.__version__ == "0.4.0"
print(rsfm_fairness_audit.__version__)
PY
git rev-parse HEAD
```

下面命令中的数据路径沿用已经通过真实 smoke 的本地路径；如果新 runtime 改了挂载位置，只改路径，不改协议参数。

## 1. fMoW

### 1.1 复用已验证的 DOFAv2 embeddings

v0.3.6 的 checkpoint、9-band 预处理和 split 已严格通过，因此可以只复用 embeddings；旧 probe、旧 formal outputs 和旧 BWER 表不得复制。

```bash
mkdir -p "$LOCAL/fmow_dofav2/embedding_cache"
cp -a "$DRIVE/outputs/geobwer_final_v2/fmow_dofav2/embedding_cache/." \
  "$LOCAL/fmow_dofav2/embedding_cache/"
```

新 runner 会重新验证 cache signature。若签名不符应重新导出 embeddings，而不是修改 manifest。

### 1.2 DOFAv2 三种子 train-only probe selection

```bash
python scripts/colab/run_fmow_dofav2_final_colab.py \
  --metadata-csv /content/fmow_formal_split_v1/fmow_formal_manifest_train_calibration_test.csv \
  --data-root /content/data/fmow_sentinel_clean_subset_30k_location_disjoint_v3_merged \
  --model-config configs/models/dofav2_fmow_sentinel.yaml \
  --dofa-repo /content/rsfm_model_repos/dofa \
  --checkpoint /content/rsfm_model_assets/dofav2_vit_base_e150.pth \
  --output-dir "$LOCAL/fmow_dofav2" \
  --persistent-output-dir "$PERSIST/fmow_dofav2" \
  --seeds 42,73,101 \
  --probe-learning-rates 0.0001,0.0003,0.001,0.003 \
  --probe-patience 20 \
  --probe-epochs 200 \
  --device cuda
```

### 1.3 common-9-band ResNet-50

```bash
python scripts/colab/run_fmow_resnet50_common9_final_colab.py \
  --metadata-csv /content/fmow_formal_split_v1/fmow_formal_manifest_train_calibration_test.csv \
  --data-root /content/data/fmow_sentinel_clean_subset_30k_location_disjoint_v3_merged \
  --output-dir "$LOCAL/fmow_resnet50_common9" \
  --persistent-output-dir "$PERSIST/fmow_resnet50_common9" \
  --seeds 42,73,101 \
  --device cuda
```

### 1.4 同种子成对面板

```bash
python scripts/colab/finalize_fmow_extended_panel_colab.py \
  --dofav2-root "$LOCAL/fmow_dofav2" \
  --resnet50-root "$LOCAL/fmow_resnet50_common9" \
  --output-dir "$LOCAL/fmow_extended_panel" \
  --persistent-output-dir "$PERSIST/fmow_extended_panel" \
  --seeds 42,73,101
```

建议 A100；复用 embeddings 后 DOFAv2 主要是小型 probe，ResNet-50 正式训练约为数小时量级。若不复用 embeddings，再加一次 DOFAv2 全 split 推理。

## 2. reBEN

正式 runner 一次完成 CROMA、TerraMind、supervised ResNet-50 × S1/S2/S1+S2 × 3 seeds。每个架构/模态的 foundation embeddings 在种子间共享，不重复提取。

```bash
python scripts/colab/run_reben_geofm_full_panel_colab.py \
  --lmdb-root /content/data/reben/BigEarthNetEncoded.lmdb \
  --metadata-parquet /content/data/reben/metadata.parquet \
  --metadata-snow-cloud-parquet /content/data/reben/metadata_for_snow_cloud.parquet \
  --croma-repo /content/rsfm_model_repos/croma \
  --croma-checkpoint /content/rsfm_model_assets/CROMA_base.pt \
  --terramind-checkpoint /content/rsfm_model_assets/TerraMind_v1_base.pt \
  --output-dir "$LOCAL/reben_full_panel" \
  --persistent-output-dir "$PERSIST/reben_full_panel" \
  --seeds 42,73,101 \
  --s1-unit-policy already_db \
  --device cuda
```

建议 A100 80GB 高 RAM。LMDB 和 metadata 必须先复制到 `/content`；不要从 Drive 直接训练。完成时间主要由三模态 embeddings 与九条监督训练路线决定，预计 8–20 小时，取决于实际 LMDB 吞吐。

## 3. Sen1Floods11

固定路径：

```bash
export S1=/content/data/sen1/S1GRDHand
export S2=/content/data/sen1/S2L1CHand
export LABEL=/content/data/sen1/LabelHand
export TRAIN=/content/data/sen1/splits/flood_train_data.txt
export VAL=/content/data/sen1/splits/flood_valid_data.txt
export TEST=/content/data/sen1/splits/flood_test_data.txt
export META=/content/data/sen1/sen1_geospatial_metadata.csv
```

### 3.1 监督 U-Net 三模态、三种子

```bash
python scripts/colab/run_sen1_supervised_panel_colab.py \
  --s1-root "$S1" --s2-root "$S2" --label-root "$LABEL" \
  --train-split "$TRAIN" --val-split "$VAL" --test-split "$TEST" \
  --output-dir "$LOCAL/sen1_supervised" \
  --persistent-output-dir "$PERSIST/sen1_supervised" \
  --seeds 42,73,101 \
  --device cuda
```

正式主基线从头训练，保持 S1/S2/fusion 传感器对称；`--pretrained-encoder` 只用于另开目录的 ImageNet sensitivity，不进入主比较。

### 3.2 Prithvi TL 概率图迁移

先把 Drive 的 `sen1floods11_prithvi_tl_official_full_512.zip` 解压到 `/content/data/sen1_prithvi_tl`。

```bash
python scripts/colab/run_prithvi_sen1_geobwer_migration_colab.py \
  --prepared-data-root /content/data/sen1_prithvi_tl \
  --prepared-metadata-csv /content/data/sen1_prithvi_tl/metadata.csv \
  --model-config configs/models/prithvi_tl_sen1floods11.yaml \
  --val-split "$VAL" --test-split "$TEST" \
  --output-dir "$LOCAL/sen1_prithvi_tl" \
  --persistent-output-dir "$PERSIST/sen1_prithvi_tl" \
  --device cuda
```

### 3.3 TerraMind 九路线与全模型共同空间校准

```bash
python scripts/colab/run_terramind_sen1floods11_final_colab.py \
  --s1-root "$S1" --s2-root "$S2" --label-root "$LABEL" \
  --train-split "$TRAIN" --val-split "$VAL" --test-split "$TEST" \
  --metadata-csv "$META" \
  --checkpoint /content/rsfm_model_assets/TerraMind_v1_base.pt \
  --output-dir "$LOCAL/sen1_terramind" \
  --persistent-output-dir "$PERSIST/sen1_terramind" \
  --seeds 42,73,101 \
  --additional-validation-export resnet34_unet_s1_seed_42="$LOCAL/sen1_supervised/s1/seed_42/probabilities/validation" \
  --additional-validation-export resnet34_unet_s1_seed_73="$LOCAL/sen1_supervised/s1/seed_73/probabilities/validation" \
  --additional-validation-export resnet34_unet_s1_seed_101="$LOCAL/sen1_supervised/s1/seed_101/probabilities/validation" \
  --additional-validation-export resnet34_unet_s2_seed_42="$LOCAL/sen1_supervised/s2/seed_42/probabilities/validation" \
  --additional-validation-export resnet34_unet_s2_seed_73="$LOCAL/sen1_supervised/s2/seed_73/probabilities/validation" \
  --additional-validation-export resnet34_unet_s2_seed_101="$LOCAL/sen1_supervised/s2/seed_101/probabilities/validation" \
  --additional-validation-export resnet34_unet_s1_plus_s2_seed_42="$LOCAL/sen1_supervised/s1_plus_s2/seed_42/probabilities/validation" \
  --additional-validation-export resnet34_unet_s1_plus_s2_seed_73="$LOCAL/sen1_supervised/s1_plus_s2/seed_73/probabilities/validation" \
  --additional-validation-export resnet34_unet_s1_plus_s2_seed_101="$LOCAL/sen1_supervised/s1_plus_s2/seed_101/probabilities/validation" \
  --additional-validation-export prithvi_tl_sen1floods11="$LOCAL/sen1_prithvi_tl/probabilities/validation"
```

### 3.4 扩展面板最终化

```bash
python scripts/colab/finalize_sen1_extended_panel_colab.py \
  --terramind-root "$LOCAL/sen1_terramind" \
  --supervised-root "$LOCAL/sen1_supervised" \
  --prithvi-root "$LOCAL/sen1_prithvi_tl" \
  --metadata-csv "$META" \
  --output-dir "$LOCAL/sen1_extended_panel" \
  --persistent-output-dir "$PERSIST/sen1_extended_panel" \
  --seeds 42,73,101
```

Sen1 建议 A100；九条 TerraMind fine-tuning 加九条 U-Net 训练预计 8–24 小时。每个已完成 U-Net seed 和 TerraMind fit/prediction stage 都可复用；中断后用完全相同命令恢复。

## 4. AlphaEarth

AlphaEarth 的现有完整概率和空间块结果可以做 GeoBWER 1.1 下游回算，无需重新请求 GEE embedding。保持原有 all-split probability 资产，使用新版：

```bash
python scripts/colab/run_alphaearth_geobwer_final_colab.py \
  --all-split-predictions /content/alphaearth/all_split_predictions.npz \
  --output-dir "$LOCAL/alphaearth" \
  --persistent-output-dir "$PERSIST/alphaearth"
```

真正的距离加权 GeoConformal comparator 尚未纳入这次正式 campaign；不要把 LAC/APS/RAPS 或 CRC 改名为 GeoConformal。

## 5. 每阶段验收

必须检查：

- `formal_evidence=true`，无 diagnostic manifest 混入；
- 三个 seed 均有完整 calibration/test probability；
- calibration 与 test sample IDs 无交集；
- `outer_validation_used_for_model_selection=false`；
- `dataset_signature` 在同一任务的所有模型间一致；
- `protocol_hash` 一致；
- strict standardisation 不可识别时保留状态，并同时报告 partial bounds；
- Selective 任一预注册组零接受时不得报告成有效公平分数；
- conformal 同时报 coverage、set size/accepted fraction、target violation 与 GeoBWER；
- common-support 至少 95%，否则修复缺失预测；
- Sen1 `common_spatial_block_calibration.json` 的 `all_models_passed=true` 且包含 19 个模型条目（9 TerraMind、9 U-Net、1 Prithvi）。

首次正式结果全部完成后，再统一生成跨任务表图；不要在单个任务跑完时提前改公式、切片或阈值。
