# RSFM BWER-Audit 技术答辩手册

用途：帮助汇报者理解当前项目中已经实现和已经完成的技术细节，以便回答导师关于公式、输入、波段、数据处理、模型输出、参数设计和验证逻辑的问题。

本文以仓库代码、`docs/experiments/scientific_findings.md`、`docs/experiments/fmow_step3_scientific_findings.md`、复现实验文档和最终 canonical 结果为依据。它不是新的实验结果，也不替代最终 ZIP 中的报告和 CSV。

---

## 1. 一分钟理解整个项目

### 1.1 研究对象

本项目研究 Remote Sensing Foundation Models（RSFM）及相关 baseline 在不同真实部署条件下的可靠性差异。

这里的“公平性”定义为：

> 按真实部署条件划分的数据分组，是否承担了不均匀的模型错误风险。

后文使用的 `slice` 或“部署分组”，指把测试数据按一个与部署有关的条件分组。例如：

- 灾害事件：Sen1Floods11 中每个 flood event 是一个分组。
- 地理位置：fMoW-Sentinel 中每个 country、region 或 latitude band 是一个分组。
- 传感器或模态条件：reBEN/CROMA 中比较 S1-only、S2-only 和 S1+S2 fusion 的运行条件。

它不是 demographic fairness，也不声称因果歧视。

### 1.2 先理解几个常用词

| 术语 | 本项目中的通俗含义 |
|---|---|
| GeoFM / RSFM | 在大量地球观测或遥感数据上预训练，再适配到具体任务的 foundation model |
| deployment group / slice | 按事件、国家、类别等真实部署条件划分的一组测试数据 |
| aggregate performance | 把全部测试数据合在一起计算的总体性能 |
| risk | 错误程度；不同任务用不同方式定义，例如分类错误率或 `1 - IoU` |
| support | 一个分组中可用于可靠计算的数据量；不是“模型支持某功能”的 support |
| worst tail | 风险最高的一组有效部署分组，而不是只看单个最差分组 |
| composition | 一个分组内部由哪些类型的数据组成以及各自比例，例如容易类别与困难类别的比例 |
| standardisation | 假设各部署分组拥有相同的已测量 composition，再比较其风险 |
| image patch / chip | 一小块遥感影像。本文统一称“影像块”；代码或原始报告中可能出现 `chip` |

技术实现中有时会使用 `risk primitive`。它不是本研究需要强调的新概念，只表示“每种任务最基础的错误或风险如何定义”。为避免生硬，本文主要使用 **task-specific risk metric** 或“任务对应的错误定义”。

### 1.3 核心方法

项目把模型输出转成统一 audit table，然后执行：

```text
模型预测
→ 定义部署分组
→ 检查每个分组的数据量和缺失组合
→ 选择适合该任务的错误定义
→ Raw-BWER
→ Standardised-BWER
→ 参数敏感性与置信度条件审计
→ 明确说明评估条件和适用范围的结论
```

`audit table` 是把每条预测与它所属的事件、国家、类别等分组信息，以及相应错误或风险连接起来的标准化表格。BWER 在这张表上运行，不直接读取原始遥感影像。

### 1.4 当前三条证据链

| 证据轴 | 数据集与任务 | Foundation model 与对照 | 正式风险单位 |
|---|---|---|---|
| Event / disaster | Sen1Floods11，洪水分割 | Prithvi TL；U-Net、S2 ResNet34-U-Net 与 MNDWI 对照 | 事件内聚合像素混淆计数后的 `1 - micro IoU` |
| Geography / location | fMoW-Sentinel，62 类场景分类 | DOFA frozen encoder + linear probe；从头训练的 ResNet50 baseline | 分组内样本分类错误率 |
| Sensor / modality | BigEarthNet v2 / reBEN，19 标签多标签分类 | 同一 CROMA foundation model 的 S1、S2、S1+S2 受控比较；BIFOLD reference 当前不可用 | label-level BCE；binary error 为次要诊断 |

---

## 2. BWER 到底是什么

项目 canonical 名称是 **Balanced Worst-slice Excess Risk**。

### 2.1 第一步：定义任务特定风险

对部署切片 \(g\)，先计算切片风险 \(R_g\)。

#### 分割任务

Sen1Floods11 先在事件内聚合全部有效像素的混淆计数：

\[
\mathrm{IoU}_g =
\frac{TP_g}{TP_g + FP_g + FN_g}
\]

\[
R_g = 1-\mathrm{IoU}_g
\]

关键点：正式事件风险不是“每个影像块 IoU 的简单平均”，而是先聚合 TP、FP、FN，再计算事件级 micro IoU。

辅助指标：

\[
\mathrm{Dice}_g =
\frac{2TP_g}{2TP_g+FP_g+FN_g}
\]

标签为：

- `1`：water / flood。
- `0`：background / no water。
- `-1`：ignore，不进入 loss 和混淆计数。

#### 单标签分类

fMoW-Sentinel 对样本 \(i\)：

\[
r_i = \mathbb{1}(\hat y_i \neq y_i)
\]

切片风险是切片内错误率：

\[
R_g = \frac{1}{n_g}\sum_{i \in g} r_i
\]

#### 多标签分类

reBEN 把每个 sample 展开成 19 个 sample-label 行。

主要概率风险：

\[
r_{ic}^{BCE}
=
-y_{ic}\log p_{ic}
-(1-y_{ic})\log(1-p_{ic})
\]

次要阈值风险：

\[
r_{ic}^{binary}
=
\mathbb{1}(\hat y_{ic}\neq y_{ic})
\]

BCE 利用概率大小，能反映高置信错误；binary error 只反映阈值后的对错。两者回答不同问题，不能混为一个指标。

### 2.2 第二步：只保留支持充分的切片

设有效部署分组集合为 \(\mathcal G_{valid}\)。只有数据量达到预先定义要求的分组，才进入正式 BWER。

分类任务主要使用 sample-level support，也就是一个分组中有多少有效样本。

分割任务使用 pixel-aware support，也就是除了影像块数量，还检查可用于计算的像素数量：

- effective support：有效像素数。
- positive support：\(TP+FN\)，即真实正类像素数。
- predicted-positive support：\(TP+FP\)，用于诊断。

预检还会拒绝：

- 切片变量缺失。
- balance variable 与 slice variable 相同。
- balance variable 是 slice variable 的确定性代理。
- 有效切片数量不足。
- slice × balance 单元过度稀疏。

### 2.3 第三步：定义最坏尾部

将有效切片按风险降序排列。主设定：

\[
\alpha=0.1
\]

最坏尾部数量：

\[
k=\max\left(1,\left\lceil \alpha|\mathcal G_{valid}|\right\rceil\right)
\]

最坏尾部集合 \(T_\alpha\) 是风险最高的 \(k\) 个有效切片。

例如 Sen1Floods11 有 11 个事件：

\[
\lceil 11 \times 0.1\rceil=2
\]

因此进入两个最坏事件，而不是只看一个最差事件。

### 2.4 BWER 主公式

仓库当前 canonical 实现是：

\[
\mathrm{BWER}_\alpha
=
\frac{1}{|T_\alpha|}
\sum_{g\in T_\alpha}R_g
-
\frac{1}{|\mathcal G_{valid}|}
\sum_{g\in\mathcal G_{valid}}R_g
\]

直观解释：

> 最坏的支持充分切片尾部，比全部支持充分切片的平均风险多承担多少风险。

BWER 越低表示切片风险分布越均匀，但不代表模型绝对性能越好。一个整体表现很差但各切片都同样差的模型，也可能有较低 BWER。因此必须同时报告 aggregate performance 与 BWER。

辅助的 `max_bwer` 为：

\[
\max_g R_g-\mathrm{mean}_{g\in\mathcal G_{valid}}R_g
\]

它只依赖单个最坏切片，比 BWER 更容易受噪声影响。

---

## 3. Raw、Standardised 与 Selective BWER

### 3.1 Raw-BWER

Raw-BWER 直接使用观察到的切片风险 \(R_g\)。

它回答：

> 在当前真实数据组成下，哪些部署分组承担了更高风险？

### 3.2 Standardised-BWER

#### 先用一个例子理解为什么需要 standardisation

假设模型在两个地区都要识别“容易类别”和“困难类别”，并且模型在同一种类别上的表现没有地区差异：

| 数据类型 | 模型错误率 |
|---|---:|
| 容易类别 | 10% |
| 困难类别 | 50% |

但是两个地区的数据组成不同：

| 地区 | 容易类别占比 | 困难类别占比 | Raw error |
|---|---:|---:|---:|
| 地区 A | 90% | 10% | \(0.9\times0.1+0.1\times0.5=14\%\) |
| 地区 B | 10% | 90% | \(0.1\times0.1+0.9\times0.5=46\%\) |

Raw 结果会显示地区 B 风险更高，但这个差异完全来自地区 B 包含更多困难类别，而不是模型在地区 B 内处理同类数据时更差。

Standardised comparison 会假设两个地区使用同一个共同组成，例如都按 50% 容易类别和 50% 困难类别重新加权：

\[
0.5\times0.1+0.5\times0.5=30\%
\]

此时两个地区的标准化风险相同。直观上，Standardised-BWER 回答：

> 如果不同部署分组拥有相同的已测量数据组成，高风险分组是否仍然更差？

在实际实验中：

- Sen1Floods 使用 flood-extent bin，检查事件风险是否只是因为洪水覆盖范围不同。
- fMoW 使用 class composition，检查国家风险是否只是因为各国类别比例不同。
- reBEN 可按 class composition 进行比较，但 label-expanded schema 会影响其解释。

#### 正式计算

假设 \(z\) 是组成变量，例如类别、洪水范围分箱。先计算 slice × composition cell 风险 \(R_{gz}\)，再用共同参考权重标准化：

\[
R_g^{std}
=
\sum_{z\in A_g}\tilde w_{gz}R_{gz}
\]

其中 \(A_g\) 是切片 \(g\) 可使用的 composition levels，\(\tilde w_{gz}\) 是按缺失策略处理后的归一化参考权重。

然后：

\[
\mathrm{Standardised\ BWER}_\alpha
=
\mathrm{BWER}_\alpha(R_g^{std})
\]

主设定使用：

- reference weighting：`uniform`。
- missing policy：`renormalize`。

`uniform` 表示每个组成 level 获得相同参考权重，而不是让数据量最大的类别自动主导结果。

缺失策略：

- `renormalize`：对该切片已有组成单元的权重重新归一化。
- `overlap`：只使用所有支持有效切片共同拥有的组成 levels。
- `invalidate`：只要缺失要求的组成 level，就将该切片判为无效。

Standardised-BWER 只能控制已经测量并纳入的 composition difference。它不是 causal adjustment，也不能证明已经消除所有 confounding。

### 3.3 Selective Risk 与 Selective-BWER

对 reBEN 的每个 sample-label 预测：

\[
\mathrm{confidence}_{ic}=\max(p_{ic},1-p_{ic})
\]

目标 coverage 为 \(q\) 时，用全局置信度分位点确定阈值：

\[
\tau_q = Q_{1-q}(\mathrm{confidence})
\]

保留：

\[
\mathcal A_q=\{(i,c):\mathrm{confidence}_{ic}\ge\tau_q\}
\]

再在保留预测上计算 overall risk、slice risk 和 selective BWER proxy。

注意：

- reBEN coverage 是 **sample × label 层面**，不是保留图像比例。
- 各切片实际 retained coverage 可能不同。
- 当前 selective BWER 是 post-hoc proxy，主要正式证据来自 reBEN/CROMA。

---

## 4. 参数设计：为什么这样设，怎么验证

### 4.1 为什么主 tail fraction 是 \(\alpha=0.1\)

理由：

- 代表最差的 10% 支持充分切片，容易解释。
- 避免只依赖单个 worst slice。
- 在切片数量较少时通过 `ceil` 保证至少一个尾部切片。
- 与 CVaR 风格的尾部平均风险思想一致。

验证：

- reBEN 明确运行 \(\alpha \in \{0.1,0.2,0.3\}\)。
- Sen1Floods BWER v2 输出 `alpha_sensitivity.csv`。
- fMoW 输出可按同一 BWER pipeline 进行 alpha sensitivity。

回答导师时应说：0.1 是主报告设定，不是唯一正确值；研究重点是结论对合理 alpha 范围是否稳定。

### 4.2 为什么最低 support threshold 不是所有任务统一

支持单位取决于任务：

- fMoW：主 taxonomy 为每个 slice 至少 20 个样本，`min_units_required=20`。
- reBEN：主设定每个 label-expanded slice 至少 20 行，敏感性为 10、20、30。
- Sen1Floods 原始 segmentation taxonomy 要求至少 1000 有效像素和 1000 正类像素；BWER v2 的事件级 post-hoc 路径使用 `min_units_required=10`，但事件的实际有效像素远高于这一最低门槛。

原因是样本数量和像素数量不是同一种有效数据量。对分割只看影像块数量，会错误地把几乎没有正类像素的事件视为数据充分。

### 4.3 为什么默认使用 uniform slice weighting

BWER 的参考均值对有效切片等权：

- 防止大国家、大事件或高频类别因样本多而自动主导。
- 指标关注“部署分组之间的风险差异”，而不是重新计算样本级总体风险。

同时保留 empirical reference weighting 作为标准化敏感性检查。

### 4.4 缺失单元合理性如何验证

项目不默默填补缺失的 slice × balance cell，而是：

1. 输出 `support_diagnostics.csv`。
2. 记录 missing cell count 与 missing fraction。
3. 比较 `renormalize`、`overlap`、`invalidate`。
4. 当缺失过多时将结果降为 diagnostic。

Sen1Floods 的 `flood_extent_bin` 没有缺失单元，所以对缺失策略稳定；`invalid_pixel_ratio_bin` 有缺失单元并对 overlap 更敏感，因此只作为次要 robustness 证据。

### 4.5 稳定性和不确定性检查

已实现或已输出：

- Bootstrap CI：默认可用 1000 次；Sen1 BWER v2 的 post-hoc event bootstrap 重采样保存的事件行，不重新运行模型。
- Stabilised BWER：

\[
\tilde R_g
=
\bar R+
\frac{n_g}{n_g+\tau}(R_g-\bar R)
\]

其中 \(\tau\in\{10,20,50\}\)。低支持切片会向总体切片均值收缩。

- Leave-one-slice-out：每次移除一个切片，检查最坏尾部和 BWER 是否被单个切片完全驱动。
- Protocol-matched：在完全相同评估样本交集上重新计算。
- LOEO：对 Sen1Floods 监督基线执行留一事件训练与评估。
- Random split 对 location-disjoint：验证 fMoW 评估协议难度。
- Tiny overfit：验证 fMoW ResNet 训练、标签和 loss plumbing 没有损坏。
- DOFA input scaling 与 pooling diagnostics：验证预处理和表示输出契约。

---

## 5. 数据与输出的统一技术契约

### 5.1 Normalized audit table

所有模型最终都需要产生标准化 audit rows，至少包含：

- 样本或 label-level 标识。
- `dataset`、`model`、`task`、`split`。
- slice 字段，例如 `event_id`、`country`、`class_label`。
- 任务输出和风险字段，例如 `correct`、`risk`、`TP/FP/FN/TN`、`risk_bce`。
- 协议字段，例如 `input_mode`、`adaptation_protocol`、`split_protocol`。
- 支持和置信度字段，如果可用。

BWER 只读取 audit table，不读取原始像素。这使模型推理与审计解耦，完成的模型输出可以 post-hoc 重跑审计而无需重新推理。

### 5.2 输出阶段

典型输出链：

```text
predictions / segmentation_metrics
→ audit_table.csv
→ slice_support_recommendations.csv
→ bwer_summary.csv
→ bwer_by_slice.csv
→ support_diagnostics.csv
→ sensitivity CSVs
→ reports / figures
→ final evidence ZIP
```

Packaging 默认排除原始影像、embeddings、大型数组和 checkpoint，保留报告、表格、诊断、图与 provenance。

---

## 6. Sen1Floods11：事件/灾害部署轴

### 6.1 数据处理

正式数据：

- 446 个手工标注 Sentinel-2 影像块。
- 11 个洪水事件。
- 512 × 512 prepared resolution。
- 输入来自 `S2Hand`，标签来自 `LabelHand`。
- mask：`-1=ignore`、`0=background`、`1=water/flood`。

Prithvi TL 六波段：

| 名称 | Sentinel-2 source index |
|---|---:|
| BLUE | 1 |
| GREEN | 2 |
| RED | 3 |
| NIR_NARROW | 8 |
| SWIR_1 | 11 |
| SWIR_2 | 12 |

关键数据修复记录：

- 旧 prepared cache 使用非 TL 的 `B02-B07` 兼容波段，导致正式 TL checkpoint 输入不匹配。
- 修复后建立独立 `prithvi_tl_sen1floods11` band profile，并拒绝不兼容 cache。
- 第二次 all-background 问题来自遗漏官方 TerraTorch `test_transform` 和 `aug` 路径。
- 恢复官方预处理和 512 × 512 windowed inference 后，正式结果恢复正常。

### 6.2 Prithvi TL 输入与输出处理

正式模型：

`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`

协议标签：

- `adaptation_protocol=task_adapted_decoder`
- `training_budget=official_sen1floods11_finetune`
- `checkpoint_source=official_huggingface`

预处理：

1. 输入 reflectance 缩放到 0–1。
2. 执行 TerraTorch datamodule `test_transform`。
3. 执行 datamodule `aug`。
4. 恢复单时间轴，输入模型为 `[B,C,T,H,W]`。
5. 使用 512 × 512 windowed inference 匹配官方推理路径。

输出：

- dense logits / probabilities。
- 类别 1 被解释为 water / flood。
- 使用预测 mask 与有效标签 mask，计算每个影像块的 TP、FP、FN、TN。
- 再按 event 聚合计数，计算正式 event-level micro IoU 风险。

### 6.3 基线

#### Vanilla U-Net

- 6-band S2 input。
- masked BCE-with-logits + soft Dice loss。
- `-1` 像素不进入 loss。
- 最多 50 epochs。
- AdamW。
- batch size 4。
- learning rate \(10^{-3}\)。
- validation IoU 选择最佳 checkpoint。
- ReduceLROnPlateau。
- early stopping patience 10。
- 默认 random image-patch split。

#### S2 ResNet34-U-Net

- 更强的 U-Net-family 基线。
- 6-band S2。
- ResNet34 encoder + U-Net decoder。
- 可选 ImageNet encoder；若使用，前三波段复制 RGB 权重，额外波段使用 RGB filter 均值初始化。

#### MNDWI 诊断基线

\[
\mathrm{MNDWI}
=
\frac{GREEN-SWIR_1}{GREEN+SWIR_1+\epsilon}
\]

正式 closure 使用固定 threshold 0.0。它是物理可解释 diagnostic baseline，不是主模型。

### 6.4 组成标准化

主 balance variable：

```text
flood_extent_bin
```

每个影像块的洪水范围：

\[
\mathrm{ground\ truth\ positive\ ratio}
=
\frac{\mathrm{positive\ pixel\ count}}
{\mathrm{valid\ pixel\ count}}
\]

使用全体可用影像块的 1/3 和 2/3 分位点分成：

- low flood extent。
- medium flood extent。
- high flood extent。

然后对每个 event × flood extent cell 聚合像素混淆计数并重新计算风险，使用 uniform reference 和 renormalize missing policy 得到主 Standardised-BWER。

### 6.5 当前主要事实

Prithvi TL：

- aggregate micro IoU：0.8052。
- aggregate Dice：0.8921。
- Raw-BWER(event)：0.1175。
- Raw tail：Pakistan、Bolivia。
- Standardised-BWER(event | flood extent)：0.1566。
- Standardised tail：Bolivia、Pakistan。

解释：

- 强平均性能没有消除事件级风险差异。
- 尾部在控制已测量 flood extent composition 后仍存在。
- 不能说所有 confounder 已被排除。

Closure：

- aggregate IoU ranking：Prithvi TL > S2 ResNet34-U-Net > Vanilla U-Net > MNDWI。
- Raw-BWER ranking：Prithvi TL < MNDWI < S2 ResNet34-U-Net < Vanilla U-Net。
- Standardised-BWER ranking：MNDWI < Prithvi TL < S2 ResNet34-U-Net < Vanilla U-Net。
- MNDWI 的低 Standardised-BWER 不表示它是最佳分割模型；它说明低绝对性能和低风险离散度可以同时出现。

图中可直接讲解的具体事件：

- Prithvi：Mekong 最好，IoU 约 0.916；Pakistan 最差，IoU 约 0.655；Bolivia 约 0.677。Raw tail 为 Pakistan、Bolivia。
- Vanilla U-Net：Mekong 最好，IoU 约 0.932；Pakistan 最差，IoU 约 0.216；Raw tail 同样为 Pakistan、Bolivia。
- S2 ResNet34-U-Net：Mekong 最好，IoU 约 0.921；Ghana 最差，IoU 约 0.495；Raw tail 为 Ghana、Bolivia。
- MNDWI：Nigeria 最好，IoU 约 0.812；Paraguay 最差，IoU 约 0.431；Raw tail 为 Paraguay、India。

这组结果说明部分困难事件会跨 learned model 重复出现，例如 Bolivia；但最差事件仍明显依赖模型。

验证：

- 89 个 exact image-patch match 上排序差异仍存在。
- Vanilla U-Net LOEO Raw-BWER = 0.1291，最差事件 Somalia。
- LOEO standardised BWER = 0.1531。
- LOEO 证明未知事件评估下仍有尾部，但不能与 random split 直接作同协议数值比较。

---

## 7. fMoW-Sentinel：地理/位置部署轴

### 7.1 数据构建

正式数据：

- 30,000 rows。
- 62 scene categories。
- train：21,046。
- validation：8,954。
- 195 countries。
- 其中 145 个 country 在 validation 中满足正式 country BWER 的 `min_samples_per_slice=20`。
- validation 中所有 62 categories 均有覆盖。
- 最低 validation category support：33。

正式 split：

```text
location_disjoint
group = category + location_id
```

训练和验证 group overlap 为 0。

数据构建原则：

- 从官方 Stanford PURL `fmow-sentinel.tar.gz` 下载到 Colab 本地。
- 不完全解压大 tar。
- 按 metadata target path 提取 clean subset。
- 验证每一个 raster 的存在性和可读性。
- 使用最终 self-contained `v3_merged` archive；早期 v1/v2 不作为正式输入。

### 7.2 地理 metadata 如何产生

SatMAE fMoW-Sentinel 原 CSV 主要有 category、location_id、image_id、timestamp，没有完整 country / region。

项目使用外部 fMoW metadata 回连：

- `category + location_id + image_id` 的 image-level join 不完整。
- `category + location_id` 的 location-level join 达到 100%。
- country：同一 location group 的多数值。
- latitude / longitude：同一 location group polygon centroid 的中位数。
- continent、UN region、region：country-region map。

重要边界：

> geography metadata 只进入 audit slicing 和 reporting，不作为模型输入。

### 7.3 Sentinel-2 13 波段

顺序与中心波长：

| Band | μm |
|---|---:|
| B01 | 0.443 |
| B02 | 0.490 |
| B03 | 0.560 |
| B04 | 0.665 |
| B05 | 0.705 |
| B06 | 0.740 |
| B07 | 0.783 |
| B08 | 0.842 |
| B8A | 0.865 |
| B09 | 0.945 |
| B10 | 1.373 |
| B11 | 1.610 |
| B12 | 2.190 |

DOFA 的关键不是任意固定 band order，而是每个输入 channel 必须与正确 wavelength 对应。

### 7.4 ResNet50 正式基线

输入：

- 13-band Sentinel-2 image only。
- resize 到 96 × 96。
- 第一层卷积从 3 channel 改为 13 channel。
- `weights=None`，从头训练。
- 使用 training split 计算每波段 mean/std，验证集不参与 normalization statistics。

训练：

- AdamW。
- cross-entropy。
- epochs：20。
- batch size：32。
- learning rate：\(10^{-3}\)。
- weight decay：\(10^{-4}\)。
- checkpoint metric：macro-F1；若不可用则回退 accuracy。
- seed：42。

输出：

- 62 类 softmax。
- prediction。
- max softmax confidence。
- `correct` 与 `risk=1-correct`。
- geography fields 被复制到 audit rows，但不输入模型。

### 7.5 DOFA 正式 frozen-probe

模型：

- DOFA ViT-base。
- frozen encoder。
- `forward_features` embedding。
- 当前 adapter 输出已是 pooled 768-dimensional representation。
- 线性分类 probe。

输入：

- 13-band Sentinel-2。
- resize 到 224 × 224。
- raw reflectance-like TIFF 首先除以 10000：

\[
x'=\frac{x}{10000}
\]

- 当前 13-band profile 对 DOFA 使用 identity mean/std；关键变化是 reflectance scaling 和 wavelength correspondence。

Probe：

- 先缓存 train / val embeddings。
- 用 train embeddings 的 mean/std 标准化 embedding。
- `torch.nn.Linear`。
- AdamW + cross-entropy。
- 200 epochs。
- learning rate \(10^{-2}\)。
- weight decay \(10^{-4}\)。
- 输出 softmax 与 max probability confidence。

为什么 `input_scale=10000` 合理：

- DOFA 要求输入在 `[0,1]` 或 `[-1,1]` 附近。
- 未缩放 TIFF 值约 0–3000+，超出 frozen encoder 预期。
- 未缩放 diagnostic accuracy 约 0.109、macro-F1 约 0.063。
- 缩放后 accuracy 约 0.178、macro-F1 约 0.169。
- embeddings 无 NaN/Inf、方差未坍缩、norm 合理。

Pooling diagnostics：

- `flatten` 与 `mean_tokens` 结果完全相同。
- 原因不是 pooling 无意义，而是当前 adapter 已输出二维 pooled embedding，没有 token 维度可供 `mean_tokens` 改变。
- CLS 不可用，因此没有编造 CLS 结果。

### 7.6 fMoW BWER

主要风险：

\[
r_i=\mathbb{1}(\hat y_i\neq y_i)
\]

正式主切片：

- continent。
- UN region。
- region。
- latitude band。
- season。
- category。

country：

- 主要使用 `min_samples_per_slice=20`。
- 也可看至少 30 的支持敏感性。

Standardised geography BWER：

- 对 geography slice 按 `class_label` 或 `category` 组成进行共同参考加权。
- country × class 高度稀疏时只作 diagnostic。

### 7.7 当前主要事实

Location-disjoint：

| 模型 | Accuracy | Macro-F1 | Country Raw-BWER | Country \| class Std-BWER |
|---|---:|---:|---:|---:|
| ResNet50 | 0.2000 | 0.1725 | 0.1736 | 0.1423 |
| DOFA | 0.1777 | 0.1687 | 0.1614 | 0.1270 |

解释：

- ResNet50 aggregate accuracy 较高。
- DOFA 在 country、region 和部分标准化地理 BWER 上较低。
- continent BWER 等并非全部由 DOFA 更低。
- 支持的结论是：aggregate ranking 与部分 geography-tail-risk ranking 不一致。

`country | class Standardised-BWER` 的含义：

- 最终仍按 country 比较风险。
- `class` 是用于 standardisation 的 composition variable。
- 它提出的问题是：如果每个国家拥有相同的 62 类场景组成，国家尾部风险是否仍存在？
- 它不同于 `country × class`。后者表示一个具体国家中的一个具体场景类别，当前 fMoW 中由于稀疏性主要作为 diagnostic。

图中使用的国家代码与具体尾部：

- ResNet50 最差国家为 `UGA`，即 Uganda；图中尾部例子包括 `AGO` Angola、`FJI` Fiji、`GMB` Gambia、`KGZ` Kyrgyzstan。
- DOFA 最差国家为 `SVK`，即 Slovakia；图中尾部例子包括 `CIV` Côte d’Ivoire、`DJI` Djibouti、`GHA` Ghana、`GMB` Gambia。
- 两个模型的尾部都包含 Gambia，但大部分尾部国家不同，说明错误地理结构具有模型依赖性。
- ResNet50 的 continent Raw-BWER 为 0.0567，低于 DOFA 的 0.0689；DOFA 的 country 和 region Raw-BWER 更低。因此不能用单一地理尺度概括所有结果。

协议验证：

- ResNet random split accuracy：0.7119。
- DOFA random split accuracy：0.3843。
- 两者都显著高于 location-disjoint，证明跨位置评估更难。
- ResNet tiny overfit accuracy：0.96875，证明训练、标签和 loss plumbing 可工作。
- Random split 仍有 geography BWER，因此较高 average performance 不自动消除地理风险。

Patch-size diagnostic：

- 30,000 rasters 全部可读。
- width min / median / max：50 / 51 / 502。
- height min / median / max：18 / 44 / 505。
- area min / median / max：918 / 2244 / 253005。
- resize 统一输入张量大小，但不能恢复原始小 patch 缺失的上下文。

---

## 8. reBEN/CROMA：传感器/模态部署轴

### 8.1 数据与协议

任务：

- BigEarthNet v2 / reBEN。
- 19-label multi-label scene classification。
- official split。
- train 用于 probe 训练，validation 用于当前评估。

当前数据源：

- `hackelle/BigEarthNetV2-LMDB` 的 safetensors-style LMDB。
- 仓库使用 direct LMDB + safetensors loader。
- 这是相对官方 ConfigILM pickle-LMDB 的 protocol risk，必须保留记录。

Sensor mode 是跨运行实验条件：

- S1-only。
- S2-only。
- S1+S2 fusion。

它不是每条样本内部的普通 slice variable。

当前 split 样本数：

- train：237,871。
- validation：122,342，当前 PPT 主结果使用此 split。
- test：119,825，当前主结果未在 test 上报告。
- country metadata 覆盖 10 个欧洲国家。

### 8.2 波段与输入

S1：

- VV。
- VH。
- 2 channels。

CROMA S2 12 bands：

```text
B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
```

即移除 cirrus B10。

输入：

- 统一 resize 到 120 × 120。
- CROMA patch size 要求 image size 能被 8 整除。

CROMA 官方 channel scaling：

对每个样本、每个 channel：

1. 计算空间 mean 与 std。
2. clip 到 `mean ± 2 × std`。
3. 线性缩放到 `[0,1]`。

### 8.3 CROMA frozen encoder

使用：

- official `antofuller/CROMA` implementation。
- `CROMA_base.pt`。
- S1：`SAR_GAP`。
- S2：`optical_GAP`。
- fusion：`joint_GAP`。

三种模式使用相同类型的 frozen encoder + linear multi-label probe 协议。

Fusion 的具体含义：

- 每个样本具有空间配对的 Sentinel-1 与 Sentinel-2 输入。
- S1 分支接收 VV、VH 两个 SAR channels。
- S2 分支接收 12 个 optical bands。
- `input_modality=both` 时，两种输入同时送入官方 CROMA。
- 下游 probe 使用 CROMA 输出的 `joint_GAP` 预训练联合表示。
- 它不是把 S1 与 S2 的最终预测概率简单平均，也不是在审计阶段拼接两个独立结果。

Probe：

- train embeddings 按训练集 mean/std 标准化。
- linear layer 输出 19 logits。
- AdamW。
- BCE-with-logits。
- epochs：100。
- learning rate：\(10^{-2}\)。
- weight decay：\(10^{-4}\)。
- full-run runner batch size：64。
- seed：42。

### 8.4 阈值和多标签输出如何处理

logits 经过 sigmoid：

\[
p_{ic}=\sigma(\ell_{ic})
\]

每个 label 使用独立阈值。当前 validation 路径在候选网格：

\[
\{0.05,0.10,\ldots,0.95\}
\]

上选择使该 label F1 最大的阈值；并列时选择更低阈值。

重要限制：

- 当前阈值在同一个 validation eval split 上选择。
- 若未来有正式 test evaluation，应先在 validation 固定阈值，再应用到 test。

每个样本展开为 19 行，每行保存：

- `class_label`。
- `label_true`。
- `label_probability`。
- `label_prediction`。
- `threshold`。
- `risk_bce`。
- `risk_binary_error`。
- `confidence=max(p,1-p)`。

### 8.5 reBEN BWER

主切片：

- class。
- country。
- country × class。
- cloud / snow / shadow，仅在 metadata 和支持量充分时使用。

主标准化：

\[
\mathrm{BWER}(\mathrm{country}\mid \mathrm{class\ label})
\]

在当前 label-expanded schema 中，每个样本对全部 19 个 class label 都贡献一行。因此每个 country 通常拥有相同的 class-label level 集合。使用 uniform class weighting 时，country Raw-BWER 与 country \| class Standardised-BWER 可能数值完全相同。这是预期的 schema effect，不是标准化代码失效。country × class 交互风险仍然提供更细粒度的信息。

主参数：

- alpha：0.1。
- minimum support：20 label-expanded rows。
- missing policy：renormalize。

敏感性：

- alpha：0.1、0.2、0.3。
- support：10、20、30。
- missing policy：renormalize、overlap、invalidate。
- BCE 与 binary-error 分开报告。

### 8.6 当前主要事实

Aggregate：

| Mode | Macro AP | Macro F1 | Mean BCE risk |
|---|---:|---:|---:|
| S1 | 0.4958 | 0.5191 | 0.3208 |
| S2 | 0.5818 | 0.5755 | 0.2819 |
| S1+S2 | 0.6080 | 0.5950 | 0.2614 |

Broad BCE geography risk：

- S1 country Raw-BWER 约 0.159。
- S2 country Raw-BWER 约 0.106。
- fusion country Raw-BWER 约 0.072，优于 S1 与 S2。
- fusion 改善 aggregate 与 broad probability-aware tail risk。

Residual risk：

- `class`：将所有国家中同一个土地覆盖标签的 label-level 判断合并，例如所有 `Pastures` 判断。
- `country`：将一个国家中的全部 label-level 判断合并。
- `country × class`：一个具体国家中的具体土地覆盖标签，例如 `Kosovo × Transitional woodland, shrub`。
- country × class BWER 明显高于 country BWER。
- fusion 后仍存在支持充分的细粒度交互尾部。
- binary-error BWER 与 BCE-BWER 可能给出不同 sensor-mode preference。

图中可直接讲解的具体结果：

| Mode | Country BCE-BWER | Country × class BCE-BWER | Worst country | Worst supported country × class |
|---|---:|---:|---|---|
| S1 | 0.1588 | 0.7501 | Luxembourg | Luxembourg × Pastures |
| S2 | 0.1055 | 0.5939 | Portugal | Switzerland × Broad-leaved forest |
| S1+S2 | 0.0724 | 0.6091 | Portugal | Kosovo × Transitional woodland, shrub |

解读：

- Fusion 的 country BWER 最低，说明 broad country-level reliability 最好。
- 在 country × class 粒度，S2 的 BWER 0.5939 略低于 Fusion 的 0.6091，因此 aggregate-best 或 country-BWER-best 不保证在每个更细粒度指标上仍然最佳。
- S1 的 class-level tail 包含 `Land principally occupied by agriculture, with significant areas of natural vegetation` 与 `Broad-leaved forest`。
- S2 和 Fusion 的 class-level tail 都包含 `Transitional woodland, shrub` 与上述 agriculture/natural-vegetation 类别。

Selective risk：

- 降低 label-level coverage 会降低总体 retained BCE risk。
- worst-country、worst-class 和 selective BWER proxy 仍不为零。
- 各切片实际 retained coverage 不同，说明 abstention 可能重新分配拒绝负担。

当前边界：

- BIFOLD ResNet101 supervised reference 因 `reben_publication` 代码不可用而被阻塞。
- 没有使用 torchvision ResNet101 替代。
- 因此当前是 CROMA-only sensor-mode case study，状态为 formal-partial。

---

## 9. 三个实验为什么能够放在同一研究中

统一之处：

- 都从模型预测构造 audit table。
- 都定义与真实部署条件相关的数据分组。
- 都使用支持量预检。
- 都取最坏支持充分切片尾部。
- 都计算 tail excess relative to valid-slice mean。
- 都保留 Raw、Standardised、sensitivity 和可用的 selective view。

不同之处：

- task-specific risk metric 不同，也就是各任务定义错误的方式不同。
- 支持单位不同。
- 标准化组成变量不同。
- 评估协议和模型适配方式不同。

所以统一的是审计逻辑，不是把不同任务的风险值强行数值等同。

---

## 10. 当前结果应该怎样表述

### 可以说

- Aggregate performance 不能单独描述部署切片可靠性。
- BWER-Audit 提供支持感知、任务适配的部署尾部风险审计。
- Sen1Floods 的事件尾部在 measured flood extent standardisation 后仍可见。
- fMoW 中 aggregate ranking 与部分 geography BWER ranking 不一致。
- location-disjoint 是比 random split 更困难且更贴近跨位置泛化的正式协议。
- CROMA fusion 改善 aggregate 与 broad BCE tail risk，但细粒度残余尾部仍存在。
- 置信度过滤降低总体 retained risk，但不保证切片差异或拒绝负担消失。

### 不应该说

- 某个国家受到模型歧视。
- 已经证明 global fairness。
- Standardised-BWER 消除了全部 confounding。
- DOFA 在所有维度上更公平或更好。
- MNDWI 是最佳洪水分割模型。
- Random split 是正式 deployment protocol。
- Selective-BWER 已是所有任务的正式主指标。
- 三种任务的绝对 BWER 值可以直接横向排名。

---

## 11. 导师高频技术问题与建议回答

### Q1：BWER 与 worst-group accuracy 有什么区别？

BWER 不只看一个最差切片，而是对支持充分切片的最坏 \(\alpha\) 尾部求平均，并减去全部有效切片平均风险。它更关注尾部额外风险，也减少单个极端小切片造成的波动。项目仍会把 worst-group、max-min gap 和 group std 作为对照。

### Q2：为什么叫 Balanced？

因为框架支持用共同组成参考权重计算标准化切片风险，避免大样本切片或不同类别组成自动主导解释。主 Standardised-BWER 使用 uniform composition weighting。Raw-BWER 本身不进行组成标准化。

### Q3：为什么 BWER 低不一定代表模型好？

BWER 衡量风险分布差异，不衡量绝对能力。一个所有切片都很差的模型可能 BWER 很低。因此必须同时报告 aggregate score 和 BWER。

### Q4：为什么分割不能直接平均每张图 IoU？

每张图的正类比例和有效像素数量差异很大。简单平均会让几乎无洪水的影像块与大面积洪水影像块等权。正式风险先在事件内聚合 TP、FP、FN，再计算 micro IoU，更符合事件部署风险。

### Q5：为什么 fMoW geography metadata 不输入模型？

研究目标是审计 image-only 模型在地理切片上的表现，而不是训练 metadata-aware 模型。若把 country 或坐标输入模型，会改变研究问题并可能引入直接地理捷径。

### Q6：为什么 DOFA 要除以 10000？

fMoW TIFF 是 reflectance-like 原始值，DOFA frozen encoder 预期接近 `[0,1]` 或 `[-1,1]` 的输入。未缩放运行性能显著下降；缩放后 embedding diagnostics 和性能恢复，因此未缩放运行被标为 invalid diagnostic。

### Q7：为什么 ResNet 和 DOFA 输入大小不同？

两者遵循各自合理的模型协议：ResNet baseline 使用 96 × 96，DOFA 使用 224 × 224。当前比较是 protocol-aware model comparison，不是严格只改变 architecture 的消融。研究通过明确记录协议避免把差异误写成纯架构效应。

### Q8：为什么 CROMA fusion 不是 sample-level slice？

同一个样本分别通过 S1、S2 和 S1+S2 三个独立运行条件。sensor mode 是跨运行条件，而不是某条样本内部自然变化的 metadata 字段。

### Q9：reBEN 的 coverage 为什么不是图像覆盖率？

多标签预测中，每张图有 19 个 label decisions。当前置信度、风险和拒绝都定义在 sample × label 行上，因此 80% coverage 表示保留 80% label decisions，而不是 80% images。

### Q10：阈值会不会导致 reBEN 结果偏乐观？

BCE 主风险不依赖二值阈值。binary error 和 F1 使用 validation 上每标签 F1 最优阈值。当前 validation-only 路径确实有阈值选择与评估同 split 的限制，所以未来正式 test 应先在 validation 固定阈值。

### Q11：Standardised-BWER 为什么可能比 Raw-BWER 更高？

标准化重新定义了参考组成，不是重新计算总体模型性能。若高风险切片在某些组成单元中风险特别高，而这些单元在共同参考分布中获得更大权重，标准化后的切片差异可以增加。这不表示 aggregate IoU 下降。

### Q12：为什么选择 flood extent 作为 Sen1Floods 主要标准化变量？

洪水正类比例会直接影响分割难度和 IoU 行为，而且能够从保存的 image-patch-level ground truth 与有效像素可靠计算。它是可测量、非 event proxy 的组成变量。它只能控制 flood extent，不能代表全部环境 confounders。

### Q13：如何证明结果不是少数样本或坏数据造成？

项目使用 support preflight、缺失单元诊断、敏感性分析、exact image-patch matching、LOEO、location-disjoint、random split contrast、tiny overfit、输入 scaling diagnostics 和 canonical provenance。每个检查针对不同失败模式，没有单个检查能证明全部正确。

### Q14：为什么 country × class 常被标成 diagnostic？

交互切片数量快速增加，许多 country-class cell 缺失或支持量不足。即使能计算数值，也不一定支持正式结论。因此只有支持充分的交互切片用于解释，完整稀疏表保留为诊断。

### Q15：为什么不直接比较三套数据的 BWER 大小？

三套任务风险尺度不同：`1-IoU`、0/1 classification error 和 BCE 不具备直接数值等价性。可以比较它们各自的 aggregate-tail 关系，但不能说某数据集因为 BWER 数值更大就“更不公平”。

### Q16：当前最强的研究贡献是什么？

最强贡献不是某一个反转数字，而是把 support preflight、适合不同任务的错误定义、composition standardisation、尾部风险、敏感性和置信度审计连接为一套可跨部署轴复用的 protocol，并展示它能识别分离、持续和共同改善三种不同关系。

### Q17：当前最需要补强的部分是什么？

优先是 fMoW 的 confidence-conditioned geography audit、替代 subgroup 指标与更完整 bootstrap；同时将地理风险空间化并研究与 benchmark coverage、人口暴露和社会经济变量的关联。

### Q18：为什么 reBEN 的 country Raw-BWER 和 country \| class Standardised-BWER 可能完全相同？

因为 audit table 是 label-expanded 的：每个样本对 19 个 class label 都产生一行，所以每个 country 在表中通常覆盖同一组 class-label levels。uniform class weighting 后，标准化 country risk 可以与原始 country risk 相同。这是预期 schema effect。若要观察更细的交互风险，应查看支持充分的 country × class slices。

### Q19：fMoW 的 location-disjoint accuracy 很低，BWER 还有意义吗？

有意义，但解释必须同时报告绝对能力。低 accuracy 说明这个跨位置任务很难；BWER 进一步说明错误在地理切片间如何分布。Random split 和 tiny-overfit diagnostics 已降低“训练流程损坏”的担忧，但当前结果不能被包装成高准确率模型中的隐藏风险故事。

### Q20：为什么没有用普通 torchvision ResNet101 替代 reBEN 的 BIFOLD reference？

因为官方 BIFOLD reference 包含特定输入协议、训练方式和 checkpoint。用 torchvision ResNet101 会悄悄改变比较对象，并让结果看似完整但协议不可信。项目选择明确标记 blocked，使当前结论保持为 CROMA-only formal-partial sensor-mode case study。

### Q21：Standardised-BWER 到底控制了什么，没有控制什么？

它只控制被选为 balance variable 的已测量组成。例如 Sen1Floods 控制 flood extent bin，fMoW 控制 class composition。它没有控制未观测环境因素、标签质量、训练数据覆盖、成像条件或因果机制，因此只能说“风险在该已测量组成标准化后仍存在”。

---

## 12. 技术文件定位

### 核心指标和审计

- `src/rsfm_fairness_audit/bwer.py`：BWER 主实现、tail、standardisation、missing policy、bootstrap。
- `src/rsfm_fairness_audit/bwer_v2.py`：Sen1Floods post-hoc standardisation、stabilisation、LOO 和报告。
- `src/rsfm_fairness_audit/audit_pipeline.py`：audit table 到 BWER family 的统一入口。
- `src/rsfm_fairness_audit/slice_support.py`：支持预检和推荐。
- `configs/slice_taxonomy.yaml`：各数据集主切片、支持门槛与默认 alpha。

### Sen1Floods

- `src/rsfm_fairness_audit/adapters/prithvi.py`：Prithvi 非 TL 与正式 TL adapter。
- `src/rsfm_fairness_audit/segmentation.py`：像素混淆计数、event aggregation、metrics。
- `src/rsfm_fairness_audit/unet_baseline.py`：U-Net 数据拆分、loss、训练和 early stopping。
- `docs/reproduction/prithvi.md`
- `docs/reproduction/unet_sen1floods11.md`
- `docs/reproduction/sen1floods11_closure.md`

### fMoW

- `src/rsfm_fairness_audit/fmow_sentinel_classification.py`：ResNet、DOFA probe、prediction 和 BWER 入口。
- `src/rsfm_fairness_audit/adapters/dofa.py`：DOFA scaling、wavelength、embedding。
- `src/rsfm_fairness_audit/band_profiles.py`：13-band 顺序和 wavelength。
- `src/rsfm_fairness_audit/fmow_sentinel_enrichment.py`：地理 enrichment。
- `docs/experiments/fmow_step3_scientific_findings.md`
- `docs/experiments/fmow_step3_analysis_plan.md`

### reBEN/CROMA

- `src/rsfm_fairness_audit/adapters/reben.py`：LMDB/safetensors、S1/S2 bands、split 与 metadata。
- `src/rsfm_fairness_audit/adapters/croma.py`：CROMA 输入缩放和 embedding。
- `src/rsfm_fairness_audit/reben_sensor_audit.py`：probe、threshold、multi-label metrics、BWER、selective risk。
- `scripts/colab/run_reben_croma_sensor_mode_audit_colab.py`
- `docs/reproduction/croma.md`

### 科学结论与边界

- `docs/experiments/scientific_findings.md`
- `docs/experiments/fmow_step3_scientific_findings.md`

---

## 13. 汇报前最后检查清单

在老师面前至少能够清楚解释：

- BWER 公式中的 \(R_g\)、\(T_\alpha\)、有效切片和 reference mean。
- 为什么 segmentation 使用 event 内聚合像素计数。
- 为什么 fMoW geography 不是模型输入。
- 为什么 DOFA 必须 scale by 10000。
- 为什么 reBEN 是 label-level coverage。
- Raw、Standardised、Selective 三者分别回答什么。
- alpha、support 和 missing policy 如何做敏感性验证。
- formal、formal-partial、sanity/diagnostic 的区别。
- 每个实验最强结论和最重要限制。

一句话总答：

> BWER-Audit 的目标不是用一个新指标替代平均性能，而是在分组数据量、任务错误定义和评估条件都明确时，补充说明表现最差的一组部署条件比典型分组额外承担多少风险。
