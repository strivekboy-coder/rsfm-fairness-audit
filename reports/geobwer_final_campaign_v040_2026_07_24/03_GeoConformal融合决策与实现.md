# GeoConformal 融合决策与实现（v0.4.1）

## 最终结论

本项目不把 Lou、Luo 与 Meng 的 GeoConformal 直接复制成四类任务共用的“正式覆盖保证”。最终采用两层结构：

1. **正式理论锚点**：fMoW 与 AlphaEarth 保留普通 split conformal（LAC/APS/RAPS）；reBEN 与 Sen1Floods11 保留任务正确的 Conformal Risk Control（CRC）。
2. **空间局部化层**：四个任务都运行同一空间适用性门控；门控通过的单标签分类任务额外运行地理核局部 prediction-set comparator，再由 GeoBWER 同时审计误覆盖尾部和集合大小尾部。多标签与分割只记录适用性结论，不把未经充分建立的“地理局部 CRC”包装成正式方法。

这不是折中，而是当前证据下最强的统一：统一的是问题、数据契约、支持诊断、产物和审计方式，不强迫不同任务共享错误的数学对象。

## 为什么没有照搬原始 GeoConformal

GeoConformal 的核心启发是：对每个测试地点，根据与校准点的地理距离加权非符合度，得到位置相关阈值。它让“全球平均覆盖率正常、局部地区却持续失效”成为可直接观察的问题。

但后续理论给出三条重要边界：

- [Weighted conformal prediction under covariate shift（NeurIPS 2019）](https://papers.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html)的有限样本保证依赖可解释为目标/源分布密度比的权重或相应的加权可交换性；任意地理核不是天然的密度比。
- [Localized conformal prediction（Biometrika 2023）](https://academic.oup.com/biomet/article/110/1/33/6647831)说明局部阈值需要经过专门校准，不能把普通加权分位数自动解释为无条件的点态覆盖。
- [Conformal prediction with local weights（JRSS B 2025）](https://academic.oup.com/jrsssb/article/87/2/549/7889096)进一步证明，朴素测试点中心的 local CP 可能任意欠覆盖或过覆盖；其 RLCP 用随机局部化恢复边际保证，但引入输出随机性，重复平均又会削弱保证或造成过度保守。

因此，v0.4.1 把地理核方法准确标为：

> empirical geographic localization comparator；除非另行证明加权可交换性/密度比条件，否则不宣称无条件有限样本点态覆盖。

普通 split CP/CRC 仍是形式保证锚点。GeoBWER 的新作用是比较“形式上边际有效”与“空间上更自适应”两类方案分别把失败和低效率分配给了谁。

## 从相关高质量工作借鉴了什么

| 来源 | 借鉴内容 | 在本项目中的实现 |
|---|---|---|
| [GeoConformal Prediction（Annals of the AAG, 2025）](https://doi.org/10.1080/24694452.2025.2516091) | 测试地点中心的高斯地理权重、位置相关不确定性 | `geo_kernel_conformal_{lac,aps,raps}` comparator |
| [Covariate-shift weighted CP（NeurIPS 2019）](https://papers.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html) | 加权分位数中的测试点原子；严格区分权重有效性的前提 | 加入权重为 1 的 `+∞` 测试原子，并在协议中冻结 validity scope |
| [Localized CP（Biometrika 2023）](https://academic.oup.com/biomet/article/110/1/33/6647831) | 局部支持与带宽会改变条件覆盖和效率 | 带宽只由校准坐标的 leave-one-out ESS 门控确定 |
| [Randomly localized CP（JRSS B 2025）](https://academic.oup.com/jrsssb/article/87/2/549/7889096) | 朴素局部化可能失去边际保证；随机化存在稳定性代价 | 不把地理核 comparator 冒充正式锚点；本轮不引入 RLCP 随机自由度 |
| [Neighborhood CP（AAAI 2023）](https://ojs.aaai.org/index.php/AAAI/article/view/25936) | 用邻域限制计算与提高局部适应性 | 使用固定最大邻居数和 BallTree；邻域/带宽均进入可复核协议 |
| [Conformal Risk Control（ICLR 2024）](https://research.google/pubs/conformal-risk-control/) | 用单调损失控制多标签漏标与分割漏检风险 | reBEN/Sen1 继续使用 CRC，不套用单标签 prediction-set 公式 |
| [EO LULC conformal statistics（RSE 2023）](https://doi.org/10.1016/j.rse.2023.113682) | 土地覆盖必须同时解释 coverage 与 prediction-set size | 所有分类 CP 同时运行 miscoverage-GeoBWER 和 efficiency-GeoBWER |
| [EO probabilistic ML + CP（Scientific Reports 2024）](https://www.nature.com/articles/s41598-024-65954-w) | EO 中模型无关、易部署的不确定性报告 | 以标准概率产物和公开 API 实现，不依赖特定 GeoFM 内部结构 |

## v0.4.1 的冻结实现

### 1. 全球正确的距离

使用 EPSG:4326 经纬度和 Haversine 大圆距离。没有沿用经纬度平面欧氏距离，因此正确处理反经线和高纬度。

### 2. 无标签、仅校准集的带宽选择

候选带宽固定为：

`25, 50, 100, 200, 400, 800, 1600, 3200, 6400 km`

对校准坐标做确定性的 leave-one-out 邻域分析，选择使局部有效样本量第 10 百分位首次达到 50 的最小带宽。测试标签完全不参与选择，也不按模型挑选最有利带宽。

### 3. 支持失败时保守而不伪装

- 校准样本少于 60：不运行局部 comparator；该门槛只做最外层防错，真正的局部支持仍由 ESS≥50 决定；
- 没有任何候选带宽通过 ESS 门槛：不运行；
- 某个测试地点 ESS 小于 50：返回全类别集合（阈值 `+∞`），同时写入 `spatial_support_identified=false`；
- 报告 `identified_fraction`、最小/中位 ESS、最近校准点距离和无穷阈值比例。

全类别集合可能覆盖很好但没有实用价值，所以不能只看 miscoverage。v0.4.1 强制同时审计：

1. `miscoverage_loss` 的 GeoBWER；
2. `prediction_set_fraction` 的 efficiency-GeoBWER。

### 4. 四任务决策

| 任务 | 正式锚点 | 空间扩展 |
|---|---|---|
| AlphaEarth | LAC/APS/RAPS | 正式运行地理核 comparator；最自然、全球范围最大 |
| fMoW-Sentinel | LAC/APS/RAPS | 坐标与 ESS 门控通过后运行；可检验地理泛化失效是否被局部集合吸收 |
| reBEN | multilabel CRC | 运行统一 preflight；当前不运行局部 CRC。原始产物缺少可靠点坐标时会明确不可识别 |
| Sen1Floods11 | segmentation CRC | 传入 validation/test 芯片坐标运行 preflight；样本/事件支持不足或任务几何不适合时正式停止 |

“screened_not_run”是实验结果，不是缺失实现：它证明同一个扩展为什么在某种任务和支持结构下不可识别。

## 产物与判定规则

每个不确定性目录新增：

- `spatial_localization_preflight.json`；
- 分类任务通过门控后新增 `geo_kernel_conformal_lac/aps/raps`；
- 每种全局/空间 CP 都新增 `*_efficiency/geobwer`；
- 空间派生表逐样本保存 bandwidth、ESS、最近校准距离、支持状态；
- `uncertainty_summary.csv` 明确区分 `formal_marginal_anchor` 与 `empirical_spatial_localization_comparator`。

已有完整概率不需要重新推理。v0.4.1 提供：

```bash
rsfm-audit geobwer-spatial-conformal-upgrade \
  --calibration-probabilities <旧calibration_probabilities.npz> \
  --calibration-metadata <含sample_id/latitude/longitude的原始metadata.csv> \
  --calibration-manifest <旧calibration_manifest.json> \
  --test-formal-dir <旧formal_outputs> \
  --protocol <对应configs/geobwer/*.yaml> \
  --group-column country \
  --output-dir <新的v041输出目录>
```

该命令按 `sample_id` 精确连接坐标，复制而不修改原 NPZ，验证全部坐标后只重算便宜的 conformal 与 GeoBWER 层。AlphaEarth 和已完成的 fMoW 因而无需重新运行 encoder/probe；新的正式 campaign 本身也会直接保存校准坐标。协议扩展产物必须写入新的 v0.4.1 输出目录，不能与 v0.4.0 结果混放。

主文纳入空间 comparator 必须同时满足：

1. 测试空间支持识别率足够高，且没有靠大量全类别集合制造覆盖率；
2. 相比全局 CP，局部 coverage 或尾部 coverage violation 有实质改善；
3. prediction-set efficiency 没有出现同等或更严重的尾部恶化；
4. 结论在 LAC/APS/RAPS 和预注册带宽敏感性中方向稳定；
5. 不把经验改善表述为未经证明的点态有限样本保证。

若不满足，放入附录作为重要负结果：地理邻近不一定等于误差机制相似，GeoBWER 揭示了局部 UQ 的公平性—效率代价。

## 没有采用的高复杂度方案

- **RLCP**：理论更强，但单次输出随机，聚合又会削弱保证；会引入与本文公平性主线无关的随机化争议。
- **calLCP**：计算和校准结构明显更复杂，且局部覆盖仍可能不均；本轮的边际收益不足。
- **GeoSIMCP（地理＋特征相似度）**：有潜力，但增加特征距离、融合权重和表示选择三个自由度。只有当地理核 comparator 显示“地理相近但语义不同”是主要失败机制时，才作为后续机制实验。
- **四任务局部 CRC**：目前没有足够成熟、统一且任务正确的保证；强做会降低而不是提高方法学可信度。

## 科学上最重要的新问题

空间局部化不只是“让集合更准”的后处理。它提供了一个新的可证伪问题：

> 当 GeoFM 的误差具有空间非平稳性时，地理局部 conformal 能否降低最差地区的误覆盖，而不把代价转移为这些地区更大的、近乎无用的预测集合？

这正好由 GeoBWER 的双风险审计回答。无论结果是改善、无效还是代价转移，都比只报告全球 coverage 更有科学信息。
