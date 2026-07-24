# v0.4.0 冻结实验矩阵与科学边界

## 主比较矩阵

| 任务 | 主 GeoFM | 协议匹配基线 | 外部参考 | 主切片 |
|---|---|---|---|---|
| fMoW-Sentinel 单标签分类 | DOFAv2，3 probe seeds | common-9-band ResNet-50，3 seeds | 旧 DOFA/BWER1 仅历史对照 | country、class、country×class |
| reBEN 多标签分类 | CROMA、TerraMind；各 S1/S2/S1+S2、3 probe seeds | supervised ResNet-50；S1/S2/S1+S2、3 seeds | 无需再加模型 | country |
| Sen1Floods11 分割 | TerraMind；S1/S2/S1+S2、3 seeds | ResNet34-U-Net；S1/S2/S1+S2、3 seeds | Prithvi TL S2，任务专用强参考 | event |
| AlphaEarth 全球土地覆盖 | AlphaEarth | 现有 ML/probe 对照 | Dynamic World/WorldCover 用于参考图歧义 | country、class、country×class、spatial block |

Prithvi 不被删除，也不替代 TerraMind。它的已发布任务适配 checkpoint 与本文重新训练的系统不共享训练预算，因此是外部有效性参考，不是架构公平胜负的主检验。

## 模型选择与校准严格隔离

v0.4.0 修复了一个会影响 conformal 解释的隐蔽风险：同一个 validation 不能既用于 early stopping，又被当作 conformal/CRC 校准集。

- fMoW DOFAv2 probe：仅在 outer train 内按 category-scoped site 做分组内验证，选择学习率和 epoch；随后用完整 outer train 重拟合。
- fMoW ResNet-50：仅在 outer train 内做 category-site-disjoint 选择；随后完整 train 重拟合。
- reBEN supervised ResNet-50：仅在 official train 内按 source tile 分组选择；随后完整 train 重拟合。
- Sen1 supervised U-Net：仅在 official train 内按洪水事件分组选择；随后完整 train 重拟合。
- Sen1 TerraMind：在 official train 内按事件划分 fit/selection；official validation 只用于空间尺度和 CRC 校准。
- CROMA/TerraMind reBEN frozen probe：使用冻结的训练协议与多个 seed；official validation 只做标签阈值和 CRC 校准。

所有由本文训练的正式模型必须至少三个随机种子，并保存完整 logits/probabilities 或概率图。单种子只允许 smoke。

## 同协议主比较与次级比较

主比较：

- fMoW：相同 seed、相同 9 bands、相同样本的 DOFAv2 vs ResNet-50；
- reBEN：相同 seed、相同 sensor mode、相同样本的 CROMA vs TerraMind vs ResNet-50；
- Sen1：相同 seed、相同 sensor mode、相同 official split 的 TerraMind vs U-Net。

次级比较：

- 同一模型家族的 S1、S2、S1+S2 模态差异；
- DOFAv2 三 seed 概率平均 ensemble，作为部署系统结果而非架构主检验；
- Prithvi TL 与 TerraMind S2 的外部有效性对照；
- 旧 BWER1 结果仅用于说明协议升级改变了哪些结论。

## 跨模型空间尺度

Sen1 的空间块尺度必须由所有正式模型的 official validation 概率共同通过：

1. 空间相关范围充分性；
2. 模拟 coverage；
3. false-positive-rate gate；
4. 全模型共同通过后，才按最小 power、平均 power 和较小尺度依次排序。

不允许每个模型自行选择最有利的空间尺度。S1/S2/架构信息属于 model lineage，不能进入 model-independent dataset signature；参考掩膜内容哈希必须进入 dataset signature。

## Conformal 的准确表述

四个任务现在都能做 conformal-family 审计，但不是四个任务都实现了 Lou、Luo 与 Meng 的 GeoConformal 原文算法。

| 任务 | 当前正式方法 | 作用 |
|---|---|---|
| fMoW、AlphaEarth | split conformal：LAC/APS/RAPS | 生成多类 prediction sets，并审计切片 miscoverage 与集合大小 |
| reBEN | multilabel Conformal Risk Control | 控制多标签漏标风险并审计国家尾部违反量 |
| Sen1 | segmentation Conformal Risk Control | 控制像素洪水漏检风险并审计事件尾部违反量 |

当前共同点是：固定模型后用独立校准集确定预测集合或风险阈值，再用 GeoBWER 审计哪些部署切片承担不成比例的失败。

GeoConformal 原文最有辨识度的组成是针对每个测试地点按地理距离加权校准误差，产生位置相关阈值。当前正式四任务代码没有把这一算法强行移植到分类、多标签和分割；把现有方法称为“GeoConformal 原算法”是不准确的。

最佳策略是：

- 主框架保留任务正确的 split CP/CRC；
- 真正的地理加权 GeoConformal 只在 AlphaEarth 做预注册 comparator；
- 若它在 held-out spatial blocks 上同时改善局部 coverage、集合效率和切片稳定性，再进入主文；否则留在附录，不影响 GeoBWER 核心。

## 是否还需要更多模型或数据集

不需要再增加任务。当前四任务已经覆盖事件、地理、传感器模态和全球地图协议四类部署轴。

模型多样性已足够：

- DOFAv2：波段自适应视觉基础模型；
- CROMA：SAR/光学多模态预训练；
- TerraMind：统一多模态 GeoFM；
- AlphaEarth：全球嵌入产品；
- Prithvi：任务专用外部参考；
- 三套监督深度基线。

继续加入 THOR 或新数据集的边际价值低于把现有矩阵的置信区间、common-support、seed 稳定性、标准化部分识别和机制分析做扎实。
