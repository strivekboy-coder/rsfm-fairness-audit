# 验证记录、剩余风险与 BWER 1.0 迁移

## 已完成的本地验证

- 数学性质：fractional tail 的精确质量分配、等权整数嵌入、\(\beta=1\)、小 \(\beta\) 极限、平移不变性、ties 和权重边界。
- 风险包络：保持审计测度的 clone invariance 与一般 refinement monotonicity；审计测度、分区规则、estimand scope 和 dependence design 均进入协议哈希。
- 可识别性：strict/overlap/partial-bounds、共同支持、不同模型协议/类别映射/数据签名不一致时拒绝比较。
- 推断：cluster/spatial simultaneous max-T、weighted 与 total-variation 双 envelope 认证区间、参数范围截断与单侧下界、paired difference、honest confirmation、small-cluster gate。
- 反例：修复旧 cluster bootstrap 重复抽到同一 cluster 后因原 ID 被 group-by 合并的问题。
- schema：分类完整概率、多标签概率、分割概率图、校准/测试 sample ID 泄漏、class mapping hash、protocol hash 和 metric version。
- campaign：AlphaEarth、fMoW、reBEN 2×3、Sen1 三模态的 CPU 级配置、缓存、恢复、产物完整性和 CLI 路径。
- 生产级模拟（`geobwer_fractional_1.1`）：200 个 Monte Carlo repetitions、每次 500 个 multiplier/bootstrap；null group-band coverage 0.935、null BWER interval coverage 1.00、false-positive rate 0，alternative coverage/power 1.00，imbalance coverage 0.925，paired coverage 1.00，honest confirmation power 1.00；sharpened radius 从未比旧有效半径更宽，平均未截断宽度比例 0.945527。该模拟验证实现与预注册门，不替代四任务实证结果。

最终全库测试：`python -m pytest -q`，**316 passed、10 skipped**（2026-07-23）。其中新增的 5 项资产准备器回归覆盖首次 no-checkout clone、固定 revision checkout、二次复用、真实 dirty 拒绝和 Drive 缓存目录初始化。跳过项均为需要外部真实资产或可选运行环境的测试，不是失败项。最终生产模拟证据位于 `outputs/geobwer_validation_production_2026_07_23_v2/`。

## 固定外部资产

| 资产 | 固定版本 | SHA-256/说明 |
|---|---|---|
| DOFAv2 base | HF `earthflow/DOFA@67e355727ca732ff0d6ca3ebcd86d399cd6b3c15` | `e1be9d50...5314d`，官方 LFS 记录 |
| DOFA source | GitHub commit `0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9` | 检查 `dofa_v1.py`、`wave_dynamic_layer.py` tracked clean |
| CROMA base | HF `antofuller/CROMA@0dd28e3d633bd6715856ae9890e8c49360040598` | `0238d814...3b63` |
| CROMA source | GitHub commit `59505a6bcadbf36ba20767270154bf9f3067c5e7` | 检查 `use_croma.py` tracked clean |
| TerraMind v1 base | HF `ibm-esa-geospatial/TerraMind-1.0-base@fb96c70d0a5f68dcc44030b89cbfd8ec3fb0c67a` | `83c3a093...d7ec` |

官方来源：

- [DOFA 官方仓库](https://github.com/zhu-xlab/DOFA)；[DOFAv2 权重固定提交与 LFS 哈希](https://huggingface.co/earthflow/DOFA/commit/67e355727ca732ff0d6ca3ebcd86d399cd6b3c15)
- [CROMA 官方仓库](https://github.com/antofuller/CROMA)；[CROMA 权重固定提交](https://huggingface.co/antofuller/CROMA/commit/0dd28e3d633bd6715856ae9890e8c49360040598)
- [TerraMind 官方示例](https://github.com/IBM/terramind)；[TerraMind 固定模型版本](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base/commit/fb96c70d0a5f68dcc44030b89cbfd8ec3fb0c67a)

## 旧产物盘点结论

本目录的 `legacy_inventory_*` 是针对旧正式包/论文资产的定向盘点；`task_inventory_*` 与 `inventory_*` 是开发期宽扫描证据。当前本地结果是：

- AlphaEarth：发现若干候选，但主要是 placeholder 或聚合表，不是完整 all-split 样本级概率来源；
- fMoW：旧表主要是聚合指标，没有 62 维完整概率，不能生成 prediction set；
- reBEN：本地没有可进入新正式输出合同的完整概率包；
- Sen1：论文资产中的表为聚合结果，没有足够的像素概率图。

因此，阶段 D 的正确结论不是“必须全部重训”，也不是“汇总 CSV 可以回算一切”，而是：挂载 Drive 后按字段决定。若已有匹配 split/checkpoint/class mapping 的完整概率，只重跑后处理；缺概率时重新 inference/probe；只有新增模型或下游训练协议改变时才训练。

## BWER 1.0 保留与迁移规则

1. 不删除旧代码、旧配置、旧 canonical packages 和论文资产。
2. BWER 1.0 作为 legacy 消融，展示 `ceil(beta*G)` 与 exact fractional boundary 的差异。
3. 旧结果如果只有聚合切片风险，可以计算 legacy point estimate 和部分新 point estimate，但不能补造样本级区间、common-support paired CI、Selective 或 Conformal 结果。
4. 所有新正式输出使用 `geobwer_final_v2` 新目录，并保存 `metric_version`、`protocol_hash`、dataset/model signature、class mapping hash 和文件 SHA。
5. 论文中可以用旧结果说明工程演进，但主 claim 只来自新冻结协议。

## 仍然存在、且只能由真实运行回答的风险

| 风险 | 处理方式 |
|---|---|
| fMoW 某类别在旧 val 中不足两个独立 site | 三分割器 hard fail；必要时回到官方更大 holdout，而不是图像级随机拆分 |
| reBEN S1 单位不明 | 真实数值范围 preflight 后固定 `s1_unit_policy`；错误单位 hard fail |
| reBEN 官方 split 的 source tile 跨 split 重叠 | 明确报告；用于依赖 cluster，不把 reBEN 冒充 location-disjoint 泛化实验 |
| CROMA train-only 统计扫描开销 | 只扫描一次并用 data-contract hash 缓存；不从 test 估计 |
| Sen1 有效事件/空间 cluster 太少 | 固定 11-event estimand；simulation gate 未通过则不认证区间 |
| AlphaEarth 参考地图歧义 | 风险命名为 map disagreement；Dynamic World/尺度/产品敏感性分层 |
| Conformal 的空间可交换性不严格 | 把 coverage 当审计结果；报告 assumption 和空间分块敏感性，不把 nominal 90% 写成无条件保证 |
| DOFAv2 9-band native 输入与 13-band ResNet 信息量不同 | 主文把比较解释为部署 pipeline；至少报告输入差异，若资源允许做 common-9-band baseline sensitivity，不因此新增主任务 |

## 进入正式运行的 Go/No-Go 门

只有以下全部为 Go 才开始正式作业：权重和源码身份通过；真实 smoke 无 NaN 且设备一致；数据/标签全部可读；类别顺序可验证；三分割无 site 泄漏；正式表具备 independent unit 与 cluster/block；校准/测试 sample ID 不重叠；协议和输出目录是新版本。任何一项失败都先修事实问题，不降低 support 或推断要求来强行出图。
