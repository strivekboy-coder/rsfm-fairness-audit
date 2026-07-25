# GeoBWER 实验完成后的权威路线与 Agent 集成计划

日期：2026-07-24  
适用指标版本：`geobwer_fractional_1.1`  
状态：**冻结为当前正式 campaign 完成后的执行路线；不得在真实实验尚未完成时用产品化改动干扰冻结实验协议。**

## 1. 当前边界

当前本地代码已经实现：

- exact fractional GeoBWER、固定 beta profile、风险包络、支持/尾部稳定性诊断；
- strict standardisation、partial bounds、common-support comparison；
- cluster/spatial simultaneous inference、LCB、paired comparison 和 honest confirmation；
- 分类、多标签、分割和 conformal 风险转换器；
- Selective-BWER、split conformal、multilabel/segmentation CRC；
- 地理核局部 conformal comparator 及空间适用性门控；
- fMoW、reBEN、Sen1Floods11、AlphaEarth 正式 runner、模型/数据签名和版本化报告卡；
- DOFAv2、CROMA、TerraMind、监督基线和 Prithvi 外部参考所需的主要运行代码。

这表示“能够在本地完成的核心方法和 runner 升级基本完成”，不表示真实实验已经完成。当前仍需通过 Colab/Drive 完成真实 GPU smoke、失败作业修复、正式多种子运行、完整概率/概率图导出和最终统计验收。

## 2. 当前 campaign 完成门

只有以下条件全部满足，才进入 SDK/Agent 产品化：

1. 所有预注册主路线完成真实 GPU smoke；
2. 所有正式训练模型具有至少三个随机种子；
3. calibration/test、inner selection/outer evaluation 隔离通过；
4. 完整概率、logits、prediction sets 或概率图已保存；
5. model/dataset/class mapping/protocol signatures 一致；
6. audit table、独立单位、cluster/spatial block 和支持量通过 preflight；
7. 正式输出未混入 diagnostic 或旧 metric version；
8. 结果包已持久化到 Drive，并完成文件哈希和完整性检查。

真实 smoke 暴露的路径、依赖、显存、数据生命周期或 adapter bug可以修复并升级软件版本；不得根据观察到的科学结果修改 beta、主切片、损失、比较族或推断目标。

## 3. Colab 完成后的第一阶段：结果冻结与论文证据

优先级高于 SDK 产品化：

1. 汇总四任务正式结果并运行 common-support paired comparisons；
2. 生成跨任务 GeoBWER Audit Cards、beta profiles、CI/LCB 和 tail/support diagnostics；
3. 汇总 Raw、Standardised、Selective、Conformal/CRC 与 spatial comparator；
4. 验证多随机种子稳定性、平均性能与尾部性能排名反转；
5. 完成 BWER1 legacy 消融和 GeoBWER 1.1 方法差异解释；
6. 检查参考地图歧义、空间尺度、模态和 protocol sensitivity；
7. 冻结主文/附录表图、结果叙事和不可识别/未认证结论；
8. 只有在完整结果支持时，决定 GeoConformal comparator 进入主文还是附录。

条件性扩展：

- 仅当地理核 comparator 显示“地理邻近不能解释误差机制”时，再评估 GeoSIMCP；
- 仅当预注册切片遗漏成为主要证据缺口时，再增加 discovery/confirmation 分离的自动切片发现；
- 默认不增加第五任务或更多 GeoFM，除非正式结果揭示当前模型矩阵存在明确的外部有效性缺口。

## 4. 第二阶段：面向外部用户的 GeoBWER Research SDK

目标：第三方模型无须接入本仓库的训练/推理代码，只需提供逐独立单位预测与部署元数据，即可生成可验证的 GeoBWER Audit Card。

### 4.1 AuditTable v1

冻结机器可验证的数据合同：

- JSON Schema；
- CSV 与 Parquet 表示；
- pandas/NumPy 入口；
- task、model、dataset、class mapping、protocol signatures；
- 独立单位、cluster/spatial block、split role 和 calibration lineage；
- 分类/多标签概率、分割 count/probability map、conformal set/coverage 字段；
- invalid-state vocabulary 和 formal/diagnostic evidence 标记。

### 4.2 一站式任务 API

新增并稳定：

```python
audit_multiclass(...)
audit_multilabel(...)
audit_segmentation(...)
audit_conformal(...)
audit_predictions(...)
```

这些函数应自动：

- 校验概率、标签维度和类别顺序；
- 生成任务正确的逐单位风险；
- 构建 AuditTable v1；
- 计算 signatures 和 protocol hash；
- 执行 preflight；
- 调用现有 `audit_rows`；
- 输出完整 Audit Card，而不是只返回一个 GeoBWER 数字。

核心 `audit`、`audit_rows`、`compare`、`confirm` 保持为高级接口。SDK 易用层不得通过自动降级破坏 formal fail-closed 规则。

### 4.3 Formal / Exploratory 双模式合同

SDK 和 CLI 应共享同一套 GeoBWER 数学核心，但明确区分两种证据模式。模式必须由用户显式选择并写入 AuditTable、protocol、manifest、Audit Card 和输出目录；不得根据 preflight 结果从 Formal 静默退化为 Exploratory。

#### Formal 模式

用途：论文主结果、正式 benchmark、模型排名和可引用的确认性结论。

要求：

- 审计轴、交互切片、风险定义、beta profile、部署测度、support 规则、缺失规则和依赖设计预注册；
- protocol 要求的字段和 split/calibration lineage 完整；
- 模糊或无法验证的国家、区域、坐标、季节及社会属性不得自动猜测；
- 不得静默删除样本、合并切片、降低 support threshold 或退化为 i.i.d. 推断；
- 任何映射或 enrichment 必须具有版本、来源、映射表哈希、覆盖摘要和 unresolved 清单；
- calibration、selection 与 test 严格隔离；
- 只有通过 schema、support、independent-unit、cluster/spatial、common-support 和 protocol-hash 检查的结果才能标记 `formal_evidence=true`；
- 不可识别、支持不足或区间无法认证属于正式科学结果，应保留明确 invalid state，不得用空值、零值或描述性估计替代。

Formal 失败时，runner 应在昂贵的模型加载、embedding、训练和推理之前完成能够提前执行的 metadata preflight，并给出机器可读的失败报告。

#### Exploratory 模式

用途：第三方数据首次接入、可用性检查、候选切片发现、调试和假设生成。

行为：

- 自动盘点可用字段与可运行审计轴，但不自动补造缺失语义；
- 仅在实际可用的轴上运行描述性 GeoBWER、support profile 和必要的诊断；
- 输出每个字段和轴的 row coverage、independent-unit coverage、cluster coverage、有效部署质量、排除样本/切片及原因；
- 对无法识别或歧义值保留 unresolved 类别和逐值报告，不把它们自动映射为真实地理群体；
- 允许用户显式提供可审计的映射表后重新运行，但必须记录映射来源和哈希；
- 所有结果强制标记 `formal_evidence=false`、`evidence_mode=exploratory` 和 `cross_run_comparable=false`；
- 不得进入正式 leaderboard、确认性模型排名、主文 CI/LCB 结论或与 Formal 结果混合汇总。

Exploratory 模式可以推荐下一步需要补充的字段、支持量或依赖设计，但不得依据观察到的 test 结果自动选择 beta、切片、阈值或正式结论。

#### 两种模式的共同输出

Audit Card 至少报告：

- evidence mode、metric version、protocol hash 和数据/模型 signatures；
- 可用字段 coverage 与预注册字段 completeness；
- 每个轴的有效/排除切片、有效部署质量和缺失原因；
- mean risk、tail risk、GeoBWER、beta profile 及适用时的 CI/LCB；
- descriptive、not identified、not certified、formal valid 等明确状态；
- enrichment/mapping lineage、unresolved values 和所有禁止比较条件。

核心原则：

```text
Formal 模式保护科学有效性；
Exploratory 模式提高接入可用性；
两者共享计算引擎，但不共享证据等级。
```

### 4.4 统一 CLI 和示例

新增稳定入口：

```text
rsfm-audit audit-predictions
rsfm-audit validate-audit-table
rsfm-audit compare-predictions
```

至少提供：

- 五分钟多分类示例；
- 五分钟多标签示例；
- 五分钟分割示例；
- conformal/CRC 示例；
- 从任意第三方 GeoFM 导出的预测表开始的完整示例；
- 每类常见失败的反例与修复说明。

SDK 升级只改变软件包版本；若 GeoBWER 数学定义和正式 estimand 不变，不升级 `metric_version`。

## 5. 第三阶段：GeoBWER Agent Integration Kit

目标：让 Codex、Claude Code 或其他编程代理能够在不猜测数据含义、不降低审计标准的前提下，帮助第三方模型完成 GeoBWER 接入。

### 5.1 交付物

```text
AGENTS.md
docs/agent_integration_protocol.md
templates/integration/AGENTS.md
templates/integration/task_manifest.yaml
templates/integration/model_manifest.yaml
templates/integration/column_mapping.yaml
schemas/audit_table_v1.schema.json
examples/agent_integration/
```

根目录 `AGENTS.md` 服务本仓库开发；可复制模板服务第三方项目。供应商相关入口应保持简短，并统一指向 `docs/agent_integration_protocol.md`，避免维护多套矛盾规则。

### 5.2 用户向代理提供的最小资产

```text
predictions.csv 或 predictions.parquet
metadata.csv 或 metadata.parquet
task_manifest.yaml
model_manifest.yaml
可选 calibration predictions / probability maps
```

### 5.3 代理必须执行的步骤

1. 盘点任务、预测输出、标签、独立单位和空间元数据；
2. 生成字段映射建议，并明确所有未识别字段；
3. 推荐候选切片，但未经确认不得注册为正式切片；
4. 构建 AuditTable v1；
5. 检查 split/calibration 泄漏、重复独立单位、类别映射和共同支持；
6. 检查 cluster/spatial dependence design；
7. 先运行 diagnostic preflight；
8. 冻结 protocol 和 signatures；
9. 运行 formal audit；
10. 输出 integration manifest、命令、验证日志和 Audit Card。

### 5.4 代理禁止事项

- 不得根据文件名猜测国家、坐标、季节或社会属性；
- 不得在缺少 cluster/block 时静默退化为 i.i.d. 推断；
- 不得使用测试标签选择 beta、阈值、切片、带宽或空间尺度；
- 不得把像素当作相互独立样本；
- 不得丢弃完整概率后声称支持 conformal；
- 不得自动降低 support threshold；
- 不得混用 calibration、selection 和 evaluation；
- 不得把 descriptive、diagnostic 或 screened-not-run 标记成 formal evidence；
- 不得因为某条路线失败而修改冻结科学协议。

### 5.5 Agent Kit 的验证

- 四任务 golden fixtures；
- 人工构造的 leakage、错误 class order、重复 unit、缺 cluster、零 selective coverage 和空间支持不足反例；
- 相同输入跨代理重复运行得到相同 AuditTable/signatures；
- 代理生成的所有正式表必须通过同一个 schema/CLI，而不是依赖自然语言自我声明正确。

## 6. 论文前后边界

论文结果稳定后、投稿前应完成：

- AuditTable v1；
- 一站式四任务 API；
- CSV/Parquet/DataFrame 入口；
- 统一 CLI；
- 最小文档、示例和 Agent Integration Kit；
- 可复现的本地安装与测试。

可以在论文后完成：

- PyPI/Conda 正式发布与长期版本策略；
- Docker/GitHub Action；
- `geobwer-thor`、`geobwer-clay` 等第三方模型插件；
- 本地 HTML 应用；
- REST/云端服务。

云端 API 不是近期优先级。GeoBWER 的首要产品形态应是 local-first、模型无关、机器可验证的 Python SDK、CLI 和 AuditTable。

## 7. 后续对话触发语

当用户在正式 campaign 完成后询问“下一步还能做什么”“如何让别人复用”或“API/Agent 怎么做”时，应优先读取本文件，并按以下顺序继续：

1. 结果冻结与论文证据；
2. GeoBWER Research SDK；
3. Agent Integration Kit；
4. PyPI/插件生态；
5. 只有经结果触发才考虑 GeoSIMCP、自动切片发现或新增模型/数据集。
