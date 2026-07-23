# 冻结式实施、模型矩阵与 Colab 重跑路线

> **实施状态更新（2026-07-22）：** 本路线的 A–E 本地部分已经执行，正式命令、真实 GPU smoke 和最新问题修复已迁移至 `reports/geobwer_final_implementation_2026_07_22/02_Colab一次性实施命令与作业顺序.md`。本文件保留为设计历史，不再作为当前运行手册。

目标：先一次性冻结指标、推断选择器、输出合同和模型矩阵，再执行一个版本化的正式运行 campaign；campaign 由多个可恢复的模型×数据集作业组成，不是一次点击或一个 Colab 会话。

## 1. 代码需要重写多少

结论：**指标层是中等规模重构，整个项目不是推倒重写。**

| 层 | 复用程度 | 主要工作 |
|---|---:|---|
| 数据缓存、manifest、prepared zip | 90%+ | 只补协议 hash 和存在性检查 |
| 已有模型适配器/训练流程 | 70%–90% | 增加完整概率/概率图输出和模型签名 |
| audit table 与预检 | 70%–80% | 增加 independent unit、空间块、概率路径、validity 字段 |
| BWER core | 约 30% | 新建 fractional weighted core；旧函数成为兼容层 |
| standardisation | 约 20% | strict/common-support/partial identification 需要重写 |
| bootstrap / inference | 约 10%–20% | 需要实质重写，现有实现不足以支持证书 |
| reporting/package | 60%–75% | 保留框架，新增 profile、CI、protocol card 和 invalid-state 表 |
| 新模型 | 新增 | TerraMind、DOFAv2；THOR 可选 |

粗略工程量不是几行 post-processing：

- 新 core 与 protocol：约 400–700 行；
- standardisation、common support、inference：约 700–1,200 行；
- 测试：约 800–1,200 行；
- CLI、报告和现有任务集成：约 500–1,000 行；
- TerraMind/DOFAv2/可选 THOR notebook 与 adapter：另计约 1,000–2,500 行。

这些是规划量级，不是承诺行数。真正耗时的不是 fractional 公式，而是空间/聚类推断校准、任务适配和新模型同协议复现。

## 2. 最终任务合同

### 2.1 Sen1Floods11

- 主任务：事件级洪水分割；
- 主风险：先在事件内聚合 TP/FP/FN，再算 (1-IoU)；
- 次风险：事件 FNR、(1-Dice)；
- 主切片：event；机制切片：event×flood support / climate/region（有可靠元数据时）；
- 主部署权重：equal event；
- 推断目标必须分开：固定 11 事件描述 vs 未来事件超总体；
- 独立单位：事件是最高层部署单位；对固定 11 事件的事件风险，用原始 scene/tile 与事件内相邻空间块估计误差；不能把 11 个事件伪装成大量独立未来灾害；
- Conformal：CRC 控制事件级洪水漏检或分割风险；逐像素 `{flood, non-flood}` 仅作可视化。

模型：

1. 监督 U-Net/ResNet34；
2. 当前 Prithvi task-learning；
3. TerraMind S1、S2、S1+S2；
4. THOR S1、S2、S1+S2（可选高价值第三架构）。

所有主比较必须统一 split、分辨率、训练预算、decoder protocol 和标签预处理；旧冻结诊断不进入同一模型胜负表。

### 2.2 fMoW-Sentinel

- 主风险：0/1 error；
- 次风险：clipped log loss、top-k error；
- 主切片：country 或预注册 geographic unit；location ID 是 split 和 paired inference 的最低独立单元，若校准残差仍显示跨 location 空间相关则升格为地表空间超块；
- 主部署权重：equal country；敏感性：经验 deployment mass；
- 主现象：mean-risk ranking 与 BWER ranking divergence；
- Conformal：APS/RAPS 类多类 prediction set，校准集必须 location-disjoint。

模型：

1. ResNet-50；
2. DOFA v1 历史消融；
3. DOFAv2 正式现代 GFM。

必须重新导出完整 62 维 probabilities/logits；只保存 max confidence 不能支持正式 conformal set。

### 2.3 reBEN

- 主风险：样本级 Hamming loss；
- 部署敏感次风险：false-negative loss；
- `1-macro AP`：只作模型/模态能力汇总，不作为唯一样本级 BWER 输入；
- 主切片：country/region；同一 patch 的 S1/S2/fusion 必须配对，导出 Sentinel tile/source scene 和坐标，以 patch/scene 或校准后的空间块作为独立单位；
- 主因子：architecture × modality；
- 主部署权重：equal country；
- Conformal：CRC 控制平均漏标风险或预注册 FNR。

模型：

1. CROMA S1/S2/S1+S2；
2. TerraMind S1/S2/S1+S2；
3. THOR 仅在 Sen1 完成且适配成本可接受时扩展，不是 reBEN 必选。

### 2.4 AlphaEarth

- 主风险：相对 WorldCover 的 0/1 map disagreement；
- 主切片：country 与 class；
- country×class：discovery + independent confirmation，不直接当主总体分数；
- 主部署权重：country 宏权重；class 分析使用预声明 equal-class；
- 推断：基于 H3/S2 等地表空间块的 studentized spatial multiplier max-T；旧 0.5°经纬网格只作敏感性；
- 机制：WorldCover–Dynamic World disagreement、land-cover ambiguity、scale sensitivity；
- Conformal：标准 spatial-block split CP 为主；GeoCP/GeoSIMCP 为可选局部不确定性 comparator。

当前不新增 AlphaEarth 模型。科学增益主要来自 lineage、reference ambiguity 和 coverage debt 推断，而不是再训练一个 tabular classifier。

### 2.5 最终主区间与空间块选择：已经确定什么

四任务统一使用以下主推断层级：

| 分析类型 | 正式方法 | 作用 |
|---|---|---|
| 预注册固定切片 | 95% studentized cluster/spatial multiplier max-T 同时风险带，传播到 BWER | 主区间 |
| 正差距声明 | 上述构造的一侧 95% BWER 下界 | `certified disparity` |
| 模型比较/排名反转 | common-support 配对 cluster multiplier 95% $\Delta$BWER 区间；主要比较 max-T 或 Holm 控制 | 主比较 |
| 自动交叉切片/空间热点 | discovery→independent confirmation | 避免搜索后自证 |
| 小 cluster/方法失效 | simultaneous 有界损失界或 `inference_not_certified` | 严格保底 |
| 直接 BWER bootstrap | 与主区间并列展示的敏感性 | 不作唯一证书 |

空间块**不写死一个跨全球、跨任务的公里数**。选择算法现在固定为：

1. 候选层级来自原 scene/tile/location 和约 1×、1.5×、2×校准相关范围的空间块；
2. 相关范围只用 calibration/validation 输出或与测试效应符号无关的信息估计；
3. 在零差异、弱尾和中等尾模拟上计算 coverage、false-positive 与 power；
4. 选择通过覆盖/假阳性门且功效最高的最小候选；
5. 没有候选通过就不认证，绝不按哪个尺度让 BWER 更显著来挑。

任务映射：Sen1 用 event→scene/tile→事件内空间块；fMoW 用 country→location→必要时空间超块；reBEN 用 country→source scene/Sentinel tile→patch 空间块；AlphaEarth 用 country/class→H3/S2 地表块。精确尺度是数据导出的协议结果，不再需要人工临场决策。

## 3. 一次性输出合同

所有正式运行统一保存：

```text
sample_id
independent_unit_id
spatial_block_id
event/location/country/region/class hierarchy
target and validity mask
hard prediction
full logits/probabilities or probability-map path
per-sample/per-tile primary and secondary losses
confidence / entropy / uncertainty fields
train-calibration-validation-test role
model/checkpoint/preprocessing signature
dataset/split/reference-product signature
source artifact hashes
metric_version / protocol_hash / run_id
```

任务专属：

- 分割：TP/FP/FN/TN、有效/前景像素数、nodata、probability-map path；
- 多标签：全标签概率、完整 multi-hot target、每标签 valid mask；
- 单标签：稳定 class order、全类别概率、logits；
- AlphaEarth：坐标、country、WorldCover、Dynamic World、年份、空间尺度和 block ID。

所有概率数组必须保存 `class_mapping_hash`，防止模型间类别顺序不一致。

## 4. 最优实施顺序

### 阶段 A：核心代码与最小 API，不跑大模型

1. 新建 `bwer_core.py` 和协议数据类；
2. 实现 exact fractional weighted tail；
3. 实现 BWER profile、tie/boundary/tail effective groups；
4. 保留 `legacy_whole_slice_bwer`；
5. 改 `bwer.py` 为兼容 façade；
6. 实现 `audit`、`compare`、四类 task adapter 和 CLI 骨架；
7. 通过数学性质测试。

**验收：**`design_validation.py` 的全部性质为 True；旧整数等权案例精确一致；11 事件尾部质量精确为 0.1。

### 阶段 B：可识别性和模型比较

1. 实现 strict standardisation；
2. 实现 model-independent common support；
3. 实现 protocol signature/hash；
4. 增加 invalid-state vocabulary；
5. 为每个模型比较生成 comparison contract。
6. formal 模式缺失独立单位、cluster、概率映射或协议签名时 hard fail，禁止静默降级。

**验收：**`renormalize` 反例正式结果返回 `not_identified` 或共同支持 0 差距；不同支持模型不能无提示排名。

### 阶段 C：推断与认证

1. 修复 cluster clone ID；
2. 明确 `fixed_slice_universe` 与 `slice_superpopulation` 两种目标；
3. 实现 apparent BWER；
4. 实现 A→B/B→A honest confirmed contrast；
5. 实现 cluster/spatial multiplier simultaneous risk bands；
6. 实现 paired model-difference CI；
7. 实现预注册 block selector，比较候选尺度、small-cluster 权重、直接 bootstrap 和严格有限样本保底下界；
8. 将 AlphaEarth 的主空间块从 0.5°经纬格升级为地表索引/等面积候选。

**验收：**

- 关键零假设 95% 覆盖与 Monte Carlo 误差相容；
- 假阳性不超过预注册容许范围；
- 中等/强尾 power 可接受；
- 结论对合理 block-size 范围不反复翻转；
- 空间相关情景不允许退回 i.i.d. interval。

### 阶段 D：现有 CSV 零成本回算

在不重新推理的情况下：

1. 回算 Raw-BWER 和可识别的 Standardised-BWER；
2. 回算 Selective-BWER；
3. 回算 AlphaEarth 现有 conformal；
4. 对旧结论做 `survives / changes / not identifiable / missing fields` 表；
5. 同步 AlphaEarth Drive full-v2，废止本地 placeholder 作为正式来源；
6. 生成一次性缺字段清单。

该阶段必须先做，因为部分运行可能已有足够概率或中间缓存，不需要盲目重推理。

### 阶段 E：模型适配与 dry run

1. DOFAv2 fMoW adapter；
2. TerraMind Sen1 与 reBEN adapter/config；
3. 可选 THOR Sen1 adapter；
4. 每个任务只用极小真实样本做 schema、device、class order 和概率输出 dry run；
5. 对 cache key 加入 input scale、band order、wavelength、resolution、checkpoint、manifest hash。

**验收：**GPU、模型参数和输入 tensor 设备一致；完整概率和 class mapping 可读；不产生伪数据；不开始正式训练。

### 阶段 F：同一冻结协议下的 Colab 正式运行 campaign

只有 A–E 全通过后，才开始纸面结果所用的正式作业。每个模型×数据集独立执行，可分多次 Colab 会话、可断点恢复、可重试失败作业：

1. fMoW：DOFAv2 + 完整概率；必要时为旧模型仅重做 inference/probe 输出；
2. Sen1：同协议模型面板 + 概率图；
3. reBEN：TerraMind 三模态；复用现有 CROMA 概率；
4. AlphaEarth：通常不重做 GEE embedding/inference，只同步正式包并下游重算；
5. 每个协议写新 output directory，不覆盖旧结果；
6. 运行结束立即压缩并同步 Drive，记录 hashes。

正式 campaign 的“一次”只表示 `metric_version + protocol_hash + split + output schema` 不再变化。TerraMind×Sen1、TerraMind×reBEN、DOFAv2×fMoW 和可选 THOR 都应分别运行；每项先做 GPU smoke，再做正式 run。遇到 OOM、路径、权重下载或 adapter bug 可以修复并重跑该项，但任何会改变 estimand、切片、$\beta$、权重或模型比较协议的变化都必须升级协议版本并说明。

### 阶段 G：统一审计和论文资产

1. 四任务统一生成 Certified BWER Profile；
2. 生成 mean/tail/gap、profile、CI、support、missing mass、tail stability；
3. 生成 common-support paired ranking；
4. 生成 conformal coverage debt 与 efficiency；
5. 生成 reference ambiguity 和 modality mechanism；
6. 运行 negative controls、permutation、partition/scale sensitivity；
7. 冻结 canonical manifest、claim IDs、表图和 paper assets。

## 5. 是否需要全部重新跑

答案分三类：

### 只需 post-processing

- fractional Raw-BWER；
- 大部分 strict/common-support 检查；
- 已保存损失和独立单位时的 cluster inference；
- AlphaEarth 现有 Raw/scale/ambiguity；
- 现有 CROMA 的大部分 Raw/Selective 分析。

### 需要重新推理，通常不需要重新训练

- fMoW 旧模型缺完整 62 维概率时；
- Sen1 已有 checkpoint 但缺概率图时；
- 需要补模型签名、class mapping 或 probability-map hashes 时。

### 需要新训练/正式 fine-tuning

- TerraMind Sen1；
- TerraMind reBEN；
- DOFAv2 的新 probe/fine-tuning；
- 可选 THOR。

因此不建议“一键把旧四实验全部从数据下载开始重跑”。正确做法是复用 prepared zip 和 checkpoint，完成一份冻结 schema 所要求的缺失输出与新增模型，再统一做下游审计。BWER1 基础使旧任务成功概率明显高于从零开始，但新增 TerraMind/DOFAv2 adapter 仍应预期真实 GPU 环境中的若干次调试。

## 6. 新模型和数据是否现在一起做

### 必做模型

1. **TerraMind × Sen1Floods11**：最高价值；填补现代 multimodal flood anchor；
2. **TerraMind × reBEN**：与 CROMA 形成 architecture × modality 复制；
3. **DOFAv2 × fMoW**：验证排名反转不依赖旧版本。

### 可选模型

- **THOR × Sen1Floods11**：若前述三项稳定，作为 2026 当前模型和第三架构 anchor；
- 不建议再无结构地加入多个 optical-only GFM。

### 数据集

- 不加第五主任务；
- EarthShift 的一个 paired shift 只作为完成主研究后的外部效度门；
- 若无法复用其数据/协议且会延误核心认证，则放弃，不降低论文主体完整性。

## 7. 预期结果会怎样变化

新方法不保证数值更大或图更震撼。更可能出现：

- fMoW 的 BWER1→fractional BWER 数值变化小，但一部分 apparent ranking divergence 被证书筛掉；剩余反转可信度大幅提升；
- Sen1 的 BWER 数值可明显变化，因为精确 10% 尾部不再机械取两个事件；但有效尾部组数仍很小；
- reBEN 的 fusion 优势可能保留；若在 TerraMind 重现，“融合优先修复尾部”的机制主张会显著增强；
- AlphaEarth country×class 的 0.3498 很可能缩小或部分失效，但稳定 coverage debt 与 reference ambiguity 会成为更强、更难反驳的现象；
- Standardised-BWER 变化可能最大，因为旧 `renormalize` 并非同一目标分布。

最有顶刊价值的结果不是“新公式让分数更大”，而是以下任一经过认证的现象：

1. 平均排名与可确认尾部排名反转；
2. 多模态融合对均值改善有限，却显著减少 certified tail disparity；
3. 全球 marginal coverage 接近目标，但特定国家/类别承担稳定 coverage debt；
4. apparent 大差距大部分由选择噪声造成，认证框架纠正了现有 GeoFM 评估的系统性过度解释。

## 8. 何时算设计真正冻结

只有以下条件全部满足，才允许正式 Colab 全量运行：

- 核心数学性质单测通过；
- strict standardisation 和 common-support 测试通过；
- cluster clone bug 修复；
- 各任务推断目标、block size 规则和主 CI 写入 config；
- 四任务 loss/slice/$\mu$/$\beta$ 合同冻结；
- 新模型的 class/band/preprocessing 经过小真实样本验证；
- 输出合同包含未来 Raw、Selective、Conformal/CRC 所需全部字段；
- canonical output 路径和 Drive package 名称冻结。

当前已经可以开始 A 阶段编码，但尚不应直接进入 F 阶段正式全量运行。

## 9. 自动化边界与用户需要做什么

阶段 A–D 以及阶段 E 的绝大多数工作可以在本地由 Codex 自动完成：代码、单测、模拟、协议/config、API/CLI、旧 CSV 回算、缺字段清单、notebook 生成和 CPU schema dry run。能在本地取得小样本与权重时，也可完成小规模 adapter 测试。

仍需用户打开 Colab 的部分是：GPU/高内存依赖、Google Drive 中的大型 prepared zip 与 checkpoint、GEE/Drive 授权，以及正式训练/推理。最稳妥的协作方式是：

1. Codex 先完成 A–E 的本地可验证部分并生成逐作业 notebook/config；
2. 用户按顺序在 Colab 运行一个作业；
3. 每个作业先输出设备、样本、band/class order、概率维度和路径自检；
4. 失败时只修该 adapter 或运行环境，成功产物立即落 Drive；
5. 全部完成后由统一 API 自动审计，不再手工拼表。

所以“第六步以前都能自动完成”在代码与协议层面基本成立，但不能在看不到 Drive 大文件、GPU 和真实权重的情况下诚实保证新模型端到端零错误。目标应是把 Colab 阶段缩减为**短 smoke + 若干独立正式作业**，而不是承诺不现实的单次无调试运行。
