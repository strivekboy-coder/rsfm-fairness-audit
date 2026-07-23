# Colab 一次性实施命令与作业顺序

这里的“一次”指**一个冻结的正式实验 campaign**，不是一个 Colab 会话、一次点击或完全不调试。每个模型×数据集仍是独立、可恢复的作业；smoke 只修路径、依赖、显存和 adapter，不允许根据结果改变 \(\beta\)、切片、权重或主比较。

所有实时训练/推理写 `/content`，完成阶段后同步 Drive。不要直接在 Drive 上训练。

## 0. 从 GitHub 固定同一份代码

工作流仍然是 **Codex 本地完成并测试 → push 到 GitHub → Colab clone/pull**。正式 campaign 不跟随会移动的 `master`，而应记录并 checkout 本次硬化完成后的提交 SHA。

首次进入新的 Colab runtime：

```bash
cd /content
git clone https://github.com/strivekboy-coder/rsfm-fairness-audit.git
cd /content/rsfm-fairness-audit
git fetch origin
git checkout <FROZEN_COMMIT_SHA>
git rev-parse HEAD
```

若仓库已经存在：

```bash
cd /content/rsfm-fairness-audit
git fetch origin
git status --short
git checkout <FROZEN_COMMIT_SHA>
git rev-parse HEAD
```

`git rev-parse HEAD` 必须与实验登记表中的 SHA 完全一致。不要在 Colab 修改正式协议后继续沿用同一个输出目录。

## 1. 挂载 Drive 并安装仓库

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
cd /content/rsfm-fairness-audit
python -m pip install -e .
```

运行 fMoW / DOFAv2 smoke 前，在该 runtime 安装冻结的 DOFA 环境：

```bash
python -m pip install -r requirements-dofa.txt
python - <<'PY'
import timm
assert timm.__version__ == "1.0.15", timm.__version__
print("timm", timm.__version__)
PY
```

DOFAv2 Base 使用作者实现的 ViT-B/14（不是 DOFAv1 的 ViT-B/16）。正式加载必须得到
`194/194` 个 state-dict 键、`105432320/105432320` 个参数完全匹配；不得降低 coverage
门槛或用 `strict=False` 掩盖结构不一致。

TerraMind 作业还需安装冻结兼容窗 `terratorch>=1.2.5,<1.3`；若安装后出现 `numpy.dtype size changed`，重启 runtime 后再继续。

## 2. 只读盘点 Drive，先判断哪些概率可复用

```bash
python scripts/colab/preflight_geobwer_drive_artifacts_colab.py \
  --output-dir /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v2/00_drive_preflight
```

只有包含样本级损失、完整概率/概率图、独立单元、cluster/block、类别映射和 split lineage 的产物才能零成本进入新正式流程。聚合 CSV 不能反推这些字段。

## 3. 一次准备固定模型资产

```bash
python scripts/colab/prepare_geobwer_model_assets_colab.py
```

默认输出：

```text
/content/rsfm_model_assets/dofav2_vit_base_e150.pth
/content/rsfm_model_assets/CROMA_base.pt
/content/rsfm_model_assets/TerraMind_v1_base.pt
/content/rsfm_model_repos/dofa
/content/rsfm_model_repos/croma
```

脚本会先在 Drive 缓存，再复制到 `/content`，并校验三份权重 SHA-256 与两个源码 Git commit。已有文件哈希不匹配时直接失败，不静默覆盖。

若旧冻结提交曾在首次 clone 后报 `Tracked files are modified`，不要让 Colab AI 执行 `git reset --hard`、`git clean` 或无条件清理任意仓库。该问题是旧资产准备器在 `--no-checkout` 状态下的检查顺序错误。更新并 checkout 新冻结提交后，推荐直接重启 Colab runtime，再运行本节原命令：`/content` 中的半成品会消失，Drive 中已验证的 checkpoint 缓存会保留。新脚本会自行创建挂载 Drive 下尚不存在的项目缓存目录。

## 4. 生成 fMoW 正式三分割

```bash
python scripts/colab/prepare_fmow_formal_splits_colab.py \
  --source-manifest /content/data/fmow_sentinel_clean_subset_30k_location_disjoint_v3_merged/final_clean_subset_manifest_30k_location_disjoint_v3_merged.csv \
  --output-dir /content/fmow_formal_split_v1 \
  --train-source-split train \
  --holdout-source-split val \
  --calibration-fraction 0.5 \
  --seed 42
```

先检查 `fmow_formal_split_contract.json`：三个 site overlap 必须为 0，calibration/test 应均保留 62 类。若失败，停止；不要降低要求或按单图随机拆分。

## 5. 真实 GPU smoke：先跑这些，不产生正式证据

### fMoW / DOFAv2

```bash
python scripts/colab/run_fmow_dofav2_final_colab.py \
  --metadata-csv /content/fmow_formal_split_v1/fmow_formal_manifest_train_calibration_test.csv \
  --data-root /content/data/fmow_sentinel_clean_subset_30k_location_disjoint_v3_merged \
  --model-config configs/models/dofav2_fmow_sentinel.yaml \
  --dofa-repo /content/rsfm_model_repos/dofa \
  --checkpoint /content/rsfm_model_assets/dofav2_vit_base_e150.pth \
  --output-dir /content/smoke/fmow_dofav2 \
  --device cuda \
  --diagnostic-max-per-split 32
```

验收日志中的 `checkpoint_load_report` 必须同时满足：

```text
parameter_coverage = 1.0
matched_keys = model_keys = checkpoint_keys = 194
model_keys_missing_from_checkpoint = []
checkpoint_keys_missing_from_model = []
same_name_shape_mismatches = []
patch_size = 14
timm = 1.0.15
```

若仍失败，先运行结构诊断，不要重新训练或降低门槛：

```bash
python scripts/diagnose_dofa_checkpoint_compatibility.py \
  --repo-path /content/rsfm_model_repos/dofa \
  --checkpoint /content/rsfm_model_assets/dofav2_vit_base_e150.pth \
  --constructor dofav2_base_patch14 \
  --output-json /content/dofav2_checkpoint_compatibility.json
```

### reBEN / CROMA×TerraMind×三模态

先抽查真实 LMDB 中 S1 数值范围，确认 `already_db`、`linear_power_to_db` 或 `linear_amplitude_to_db`。不要根据名称猜。

```bash
python scripts/colab/run_reben_geofm_full_panel_colab.py \
  --lmdb-root /content/data/reben \
  --metadata-parquet /content/data/reben/metadata.parquet \
  --croma-repo /content/rsfm_model_repos/croma \
  --croma-checkpoint /content/rsfm_model_assets/CROMA_base.pt \
  --terramind-checkpoint /content/rsfm_model_assets/TerraMind_v1_base.pt \
  --output-dir /content/smoke/reben_2x3 \
  --device cuda \
  --s1-unit-policy already_db \
  --diagnostic-max-samples 32
```

### Sen1Floods11 / TerraMind 三模态

```bash
python scripts/colab/run_terramind_sen1floods11_final_colab.py \
  --s1-root /content/data/sen1/S1GRDHand \
  --s2-root /content/data/sen1/S2L1CHand \
  --label-root /content/data/sen1/LabelHand \
  --train-split /content/data/sen1/splits/flood_train_data.txt \
  --val-split /content/data/sen1/splits/flood_valid_data.txt \
  --test-split /content/data/sen1/splits/flood_test_data.txt \
  --checkpoint /content/rsfm_model_assets/TerraMind_v1_base.pt \
  --output-dir /content/smoke/sen1_terramind \
  --smoke-only
```

smoke 验收：真实图像可读、band/channel 顺序正确、输出维度正确、无 NaN、CUDA 可用，且日志中的模型参数和输入 tensor 都在同一 GPU。smoke 输出目录在确认后可删除；正式目录必须使用新路径。

## 6. 正式 campaign

smoke 通过后，去掉三个 diagnostic/smoke 参数，改用新正式本地目录及对应 Drive mirror：

```text
/content/outputs/geobwer_final_v2/<campaign>
/content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v2/<campaign>
```

推荐顺序：

1. AlphaEarth 下游回算（若 Drive 已有完整 all-split probabilities）；
2. fMoW DOFAv2；
3. reBEN 2×3 面板；
4. Sen1 TerraMind 三模态；
5. common-support model panels、统一报告卡和论文表图。

AlphaEarth：

```bash
python scripts/colab/run_alphaearth_geobwer_final_colab.py \
  --all-split-predictions /content/alphaearth/all_split_predictions.npz \
  --output-dir /content/outputs/geobwer_final_v2/alphaearth \
  --persistent-output-dir /content/drive/MyDrive/rsfm_fairness_audit/outputs/geobwer_final_v2/alphaearth
```

其余正式命令与 smoke 相同，但删除 `--diagnostic-max-per-split`、`--diagnostic-max-samples` 或 `--smoke-only`，并增加 `--persistent-output-dir`。正式 runner 均可按阶段恢复；若同一输出目录中的模型、split、checkpoint、protocol 或 cache signature 改变，会硬失败并要求新目录。

## 7. 失败时允许改什么

允许修：路径、依赖版本、batch size、num workers、OOM、加载器实现、缓存恢复和明确的 adapter bug。  
不得静默改：主 \(\beta\)、切片定义、部署权重、正式 split、损失定义、标准化目标、block 选择依据、主比较族。后者若必须变化，应升级协议/metric version 并保留旧目录。
