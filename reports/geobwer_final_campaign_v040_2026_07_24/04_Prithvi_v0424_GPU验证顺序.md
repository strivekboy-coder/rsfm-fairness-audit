# Prithvi v0.4.24 GPU 验证顺序

## 冻结推理合同

- 官方 Sen1Floods11 TL 推理使用 512×512 空间窗口。
- 输入布局为 `[B,C,T,H,W]`；只对 H/W 做 NumPy reflect 右侧/底部填充，T 不变。
- 224×224 输入填充为 512×512，推理后严格裁回 224×224。
- `--device cuda` 是强合同：模型参数、buffer 和输入必须全部位于同一 CUDA device，禁止静默回退 CPU。
- 该修复不改变 checkpoint、band profile、prepared data、252/89/90 split、CRC 或 GeoBWER。

官方依据：

- 模型与数据说明：https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11
- 正式推理脚本：https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/blob/main/inference.py
- 正式配置：https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/blob/main/config.yaml

## 第一步：Prithvi-only A100 probe

继续复用 v0.4.23 已准备的数据、split、checkpoint 和配置。不要覆盖失败目录。

```bash
cd /content/rsfm-fairness-audit
git pull
python -m pip install -e .

python -u scripts/colab/run_prithvi_sen1_geobwer_migration_colab.py \
  --prepared-data-root /content/data/sen1_prithvi_tl \
  --prepared-metadata-csv /content/data/sen1_prithvi_tl/metadata.csv \
  --model-config configs/models/prithvi_tl_sen1floods11.yaml \
  --val-split /content/data/sen1/splits/flood_valid_data.txt \
  --test-split /content/data/sen1/splits/flood_test_data.txt \
  --output-dir /content/prithvi_only_gpu_probe_v0424 \
  --persistent-output-dir /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/00_smoke_evidence/prithvi_only_gpu_probe_v0424 \
  --batch-size 1 \
  --device cuda \
  --diagnostic-max-samples 1
```

必须看到：

```text
resolved=cuda:0
model_parameter_device=cuda:0
model_input_device=cuda:0
PRITHVI_ONLY_GPU_PROBE=PASS
```

并检查 `diagnostic_probe/validation_full_probabilities.npz` 与
`diagnostic_probe/test_full_probabilities.npz` 均为 `[1,2,H,W]`、类别和为 1、
target 空间尺寸一致。

## 第二步：统一完整 GPU smoke

仅当 Prithvi-only probe 通过后运行。输出使用全新目录
`sen1_gpu_smoke_v0424`，不得覆盖 v0.4.23。

```bash
python -u scripts/colab/run_sen1floods11_gpu_smoke_colab.py \
  --s1-root /content/data/sen1/S1GRDHand \
  --s2-root /content/data/sen1/S2L1CHand \
  --label-root /content/data/sen1/LabelHand \
  --train-split /content/data/sen1/splits/flood_train_data.txt \
  --val-split /content/data/sen1/splits/flood_valid_data.txt \
  --test-split /content/data/sen1/splits/flood_test_data.txt \
  --terramind-checkpoint /content/rsfm_model_assets/TerraMind_v1_base.pt \
  --prithvi-prepared-data-root /content/data/sen1_prithvi_tl \
  --prithvi-prepared-metadata-csv /content/data/sen1_prithvi_tl/metadata.csv \
  --prithvi-model-config configs/models/prithvi_tl_sen1floods11.yaml \
  --output-dir /content/sen1_gpu_smoke_v0424 \
  --persistent-output-dir /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/00_smoke_evidence/sen1_gpu_smoke_v0424 \
  --seed 42 \
  --diagnostic-max-samples 12 \
  --batch-size 2 \
  --num-workers 2
```

正式 19-model campaign 仍以 `SEN1_GPU_SMOKE=PASS` 为启动门槛。
