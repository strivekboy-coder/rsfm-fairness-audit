# Sen1Floods11 v0.4.26：Bolivia 独立留出协议

## 冻结设计

Sen1Floods11 hand-labeled core 固定为四个互斥集合：

- train：252；
- validation：89；
- standard test：90；
- Bolivia held-out test：15。

四者并集必须恰好为 446 张、11 个事件。Bolivia 只能存在于独立
holdout；模型训练只使用 train，checkpoint 选择、CRC 与共同空间尺度
校准只使用 validation。论文审计同时保留：

1. standard test 90；
2. Bolivia holdout 15；
3. combined held-out 105（固定 11-event deployment universe）。

旧 v0.4.24 GPU smoke 与 v0.4.25 formal-gate 证据保持不变，只证明已经
通过的调用链；正式 v0.4.26 结果必须写入全新目录，不能复用旧431样本
正式输出。

## 路径

```bash
export S1=/content/data/sen1/S1GRDHand
export S2=/content/data/sen1/S2L1CHand
export LABEL=/content/data/sen1/LabelHand
export TRAIN=/content/data/sen1/splits/flood_train_data.txt
export VAL=/content/data/sen1/splits/flood_valid_data.txt
export TEST=/content/data/sen1/splits/flood_test_data.txt
export BOLIVIA=/content/data/sen1/splits/flood_bolivia_data.txt
export META=/content/data/sen1/sen1_geospatial_metadata_446.csv
export LOCAL=/content/sen1_geobwer_v0426
export PERSIST=/content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/sen1_geobwer_v0426
```

`$BOLIVIA` 可以是官方两列 CSV 或一行一个 prefix 的文件；runner 会
验证其恰好15行且事件集合为 `{"Bolivia"}`。`$META` 必须能为105个评估
样本提供坐标，或概率索引中的绝对 GeoTIFF 路径必须仍可读取。

## Prithvi Bolivia 独立补充资产

保留既有431行 Prithvi prepared root 不变，从446张原始 hand-labeled
GeoTIFF 中单独建立15行 Bolivia 补充资产：

```bash
python -u scripts/prepare_sen1floods11_subset.py \
  --source-root /content/data/sen1 \
  --output-dir /content/data/sen1_prithvi_tl_bolivia15 \
  --event-filter Bolivia \
  --max-samples 0 \
  --target-size 224 \
  --band-profile prithvi_tl_sen1floods11
```

runner 会硬验证该补充资产没有多余行，并与既有431行核心资产零重叠。

## 正式运行顺序

### 1. ResNet34-U-Net：三模态 × 三种子

```bash
python -u scripts/colab/run_sen1_supervised_panel_colab.py \
  --s1-root "$S1" --s2-root "$S2" --label-root "$LABEL" \
  --train-split "$TRAIN" --val-split "$VAL" --test-split "$TEST" \
  --bolivia-split "$BOLIVIA" \
  --output-dir "$LOCAL/supervised" \
  --persistent-output-dir "$PERSIST/supervised" \
  --seeds 42,73,101 \
  --device cuda
```

### 2. Prithvi TL：冻结 checkpoint 概率迁移

```bash
python -u scripts/colab/run_prithvi_sen1_geobwer_migration_colab.py \
  --prepared-data-root /content/data/sen1_prithvi_tl \
  --prepared-metadata-csv /content/data/sen1_prithvi_tl/metadata.csv \
  --bolivia-prepared-data-root /content/data/sen1_prithvi_tl_bolivia15 \
  --bolivia-prepared-metadata-csv /content/data/sen1_prithvi_tl_bolivia15/metadata.csv \
  --model-config configs/models/prithvi_tl_sen1floods11.yaml \
  --train-split "$TRAIN" --val-split "$VAL" --test-split "$TEST" \
  --bolivia-split "$BOLIVIA" \
  --output-dir "$LOCAL/prithvi" \
  --persistent-output-dir "$PERSIST/prithvi" \
  --device cuda
```

`--train-split` 仅用于证明既有431行 prepared asset 恰好等于
252+89+90；Prithvi runner 不会对 train 做推理、训练或校准。

### 3. TerraMind：九路线与19模型共同空间尺度

沿用原正式命令的19个 `--additional-validation-export`，但必须增加：

```bash
python -u scripts/colab/run_terramind_sen1floods11_final_colab.py \
  --s1-root "$S1" --s2-root "$S2" --label-root "$LABEL" \
  --train-split "$TRAIN" --val-split "$VAL" --test-split "$TEST" \
  --bolivia-split "$BOLIVIA" \
  --metadata-csv "$META" \
  --checkpoint /content/rsfm_model_assets/TerraMind_v1_base.pt \
  --output-dir "$LOCAL/terramind" \
  --persistent-output-dir "$PERSIST/terramind" \
  --seeds 42,73,101 \
  --additional-validation-export prithvi_tl_sen1floods11="$LOCAL/prithvi/probabilities/validation"
```

实际正式命令仍须追加9个 U-Net validation exports。共同空间尺度只读取
validation；Bolivia 在尺度冻结后才推理和审计。

### 4. 扩展面板

```bash
python -u scripts/colab/finalize_sen1_extended_panel_colab.py \
  --terramind-root "$LOCAL/terramind" \
  --supervised-root "$LOCAL/supervised" \
  --prithvi-root "$LOCAL/prithvi" \
  --metadata-csv "$META" \
  --output-dir "$LOCAL/extended_panel" \
  --persistent-output-dir "$PERSIST/extended_panel" \
  --seeds 42,73,101
```

## 必须看到的合同

- split preflight：252/89/90/15、446、11 events；
- standard test events：10个非 Bolivia 事件全部覆盖；
- `evaluation_sample_count=105`；
- `standard_test_count=90`；
- `bolivia_holdout_count=15`；
- `no_training_or_calibration_leakage=true`；
- 每个模型分别存在 `standard_test`、`bolivia_holdout` 和
  `combined_held_out` 概率/正式输出；
- combined audit 的角色计数严格为90+15；
- 旧431样本正式输出不得作为 v0.4.26 completion。
