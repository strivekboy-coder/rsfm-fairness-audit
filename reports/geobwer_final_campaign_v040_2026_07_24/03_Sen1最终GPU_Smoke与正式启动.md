# Sen1Floods11 最终真实 GPU 门槛

## 判定原则

正式 campaign 启动前必须由同一次非正式 smoke 验证：

- TerraMind：S1、S2、S1+S2 的真实训练 batch、验证 batch、checkpoint 保存/重载，以及 validation/test 概率图 writer；
- ResNet34-U-Net：S1、S2、S1+S2 的事件互斥内部选择、训练、refit 和 validation/test 完整概率图；
- Prithvi TL：官方 S2 任务 checkpoint 的 validation/test 概率图；
- 三条路线均实际使用 CUDA；
- 概率有限且位于 `[0,1]`，两类逐像素和为 1；
- mask 与概率图尺寸一致且包含有效手工标注像素；
- validation/test `sample_id` 零重叠；
- 所有 smoke 产物均标记 `formal_evidence=false`。

只有最终日志出现 `SEN1_GPU_SMOKE=PASS` 且
`completion_contract.json` 存在时，才允许启动正式训练。

## Colab 环境

使用 A100 High-RAM。Drive 只保存准备包和持久镜像；数据、训练和推理均在
`/content` 本地盘运行。

```bash
cd /content/rsfm-fairness-audit
git pull
python -m pip install -e .

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
  --prithvi-model-config /content/rsfm-fairness-audit/configs/models/prithvi_tl_sen1floods11.yaml \
  --output-dir /content/sen1_gpu_smoke_v0421 \
  --persistent-output-dir /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v3/00_smoke_evidence/sen1_gpu_smoke_v0421 \
  --seed 42 \
  --diagnostic-max-samples 12 \
  --batch-size 2 \
  --num-workers 2
```

路径若与当前 runtime 不同，只修改资产路径，不修改 seed、模型或科学参数。
失败时保留原目录作为诊断证据，修复后使用新的版本化 smoke 目录，不在原目录上覆盖。
v0.4.21 固定每个 TerraMind smoke 阶段运行 2 个 batch；这是为了避免
Lightning 将 `1` 解析为 `1.0`（即 100% batches），不改变任何正式训练参数。
TerraMind validation/test prediction 通过仓库内的 CLI 包装器启动；包装器在
CLI 实例化完成、`trainer.predict()` 开始前，精确移除 TerraTorch 1.2.10
自动注入且不支持 Mapping 输出的 `terratorch.cli_tools.CustomWriter`，
并硬校验 `GeoBWERProbabilityWriter` 恰好保留一个。fit 和 checkpoint
生命周期仍使用 TerraTorch 原生 CLI。
官方 split 中合法的全 `-1`（ignore/no-label）芯片会原样保留。诊断验收逐行
检查标签值域只能为 `{-1,0,1}`，但允许单行有效像素数为零；整个 validation
或 test 导出仍必须至少有一行及大于零的聚合有效像素支持。

## 正式启动

smoke 通过后，依次运行原冻结流程：

1. `run_sen1_supervised_panel_colab.py`：三模态 × 三种子；
2. `run_prithvi_sen1_geobwer_migration_colab.py`：完整 validation/test；
3. `run_terramind_sen1floods11_final_colab.py`：三模态 × 三种子，并使用全部 19 个模型的 validation 概率冻结共同空间尺度；
4. `finalize_sen1_extended_panel_colab.py`：CRC、GeoBWER 及配对模型面板。

正式命令继续以
`02_最终Colab运行顺序.md` 第 3 节为准，但必须使用本次 smoke 所对应的冻结 commit。
