# BWER 框架中的 Conformal Prediction、GeoConformal 与 Selective-BWER

## 一、先回答最关键的问题

目前项目已经实现了 **standard split conformal prediction 的分类版本**，并在 AlphaEarth 土地覆盖实验中完成了：

- 独立的空间块训练集、校准集和测试集；
- 利用校准集确定 conformal 阈值；
- 为每个测试样本生成土地覆盖预测集合；
- 计算总体 coverage、各地理/类别切片 coverage、平均 set size；
- 将“真实类别是否落入预测集合”转换为损失，再用 BWER 审计最差部署切片。

但是，目前实现的还不是 Lou、Luo 和 Meng 提出的 **GeoConformal Prediction（GeoCP）完整算法**。GeoCP 最有辨识度的一步，是根据每个测试地点与校准样本之间的地理距离，对校准误差赋予不同权重，使不同地点得到不同的局部阈值和预测区间。当前 AlphaEarth 代码对全部校准分数计算一个全局阈值，没有进行逐测试地点的地理距离加权。

因此，最准确的状态是：

> 基础 conformal 分类流程和 BWER 覆盖差距审计已经实现；GeoConformal 论文中的位置相关地理加权校准尚未实现。

这并不意味着之前的工作做错了。standard split conformal 是 GeoConformal 和许多其他 conformal 扩展的共同基础，现有数据划分、概率输出、coverage 计算和 BWER 管线都可以继续使用。缺少的是 GeoConformal 的核心空间加权层，而不是需要推翻整个实现。

---

## 二、GeoConformal 论文主要在做什么

> Lou, X., Luo, P., & Meng, L. (2025). *GeoConformal Prediction: A Model-Agnostic Framework for Measuring the Uncertainty of Spatial Prediction*. **Annals of the American Association of Geographers**.  
> 正式论文：[https://doi.org/10.1080/24694452.2025.2516091](https://doi.org/10.1080/24694452.2025.2516091)  
> 预印本：[https://arxiv.org/abs/2412.08661](https://arxiv.org/abs/2412.08661)  
> 官方代码：[https://github.com/pengtum/geoconformal](https://github.com/pengtum/geoconformal)

### 最通俗的例子：预测房价

假设模型预测某个地点的房价。

普通模型只输出：

> 预测房价：50 万美元

普通 conformal prediction 可能输出：

> 预测区间：45 万至 55 万美元

这比单一预测值多表达了一层信息：模型认为真实房价大概率落在这个区间中。

但是，城市中心和偏远地区的预测难度并不相同：

- 城市中心附近训练样本多，规律可能较稳定；
- 偏远地区样本少，模型误差可能更大；
- 不同区域可能存在不同的局部空间过程；
- 一个全球统一的误差区间，可能在容易地区过宽，在困难地区过窄。

GeoConformal 的核心想法是：

> 预测某个新地点时，主要参考附近校准样本过去犯过的错误，较少参考距离很远的样本。

对于测试地点 \(x_*\)，GeoConformal 根据地理距离给第 \(i\) 个校准样本分配权重。常见形式是高斯距离衰减：

\[
w_i(x_*) \propto
\exp\!\left[-\frac{1}{2}
\left(\frac{d(x_i,x_*)}{b}\right)^2\right],
\]

其中 \(b\) 是 bandwidth，决定“多远还算邻近”。随后，它计算附近校准误差的加权分位数：

\[
q_{\mathrm{geo}}(x_*)=
\operatorname{WeightedQuantile}_{1-\epsilon}
\{e_i;w_i(x_*)\}.
\]

每个测试地点因此拥有自己的区间：

\[
C(x_*)=
[\hat y(x_*)-q_{\mathrm{geo}}(x_*),
 \hat y(x_*)+q_{\mathrm{geo}}(x_*)].
\]

一句话概括：

> 普通 conformal 给所有地点使用一把全局“误差尺子”；GeoConformal 根据附近的历史误差，为每个地点制作一把本地尺子。

论文主要在空间回归和空间插值上验证方法。作者报告，在房价空间回归案例中，GeoConformal 得到 93.67% coverage，而其 bootstrap 对比最高为 81.00%；在空间插值中，GeoConformal 的不确定性与 Kriging variance 较为一致。论文还利用不确定性分析说明，加入局部特征可以降低强局部依赖区域的不确定性。

### 这篇论文与“预测类别集合”的关系

GeoConformal 原论文的主要案例是连续值预测，因此输出的是数值区间，例如“45 万至 55 万”。

你的 AlphaEarth 和 fMoW 是单标签分类，因此更自然的输出是类别集合，例如：

> {草地，灌木地}

将 GeoConformal 从回归区间迁移到分类集合，核心思想不变：

1. 用分类模型产生各类别概率；
2. 在校准集计算不符合度分数；
3. 根据测试地点与各校准点的地理距离分配权重；
4. 为每个测试地点计算位置相关的加权分位数；
5. 生成位置相关的类别预测集合。

这不是对论文代码的原样复制，而是将论文的地理加权 conformal 原理迁移到 GeoFM 分类审计。

---

## 三、另一篇 Conformal Risk Control 论文在做什么

另一条相关路线是：

> Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024). *Conformal Risk Control*. ICLR 2024.  
> 论文页面：[https://research.google/pubs/conformal-risk-control/](https://research.google/pubs/conformal-risk-control/)  
> 官方代码：[https://github.com/aangelopoulos/conformal-risk](https://github.com/aangelopoulos/conformal-risk)

普通 conformal prediction 最典型的目标是控制：

> 真实答案是否落在预测集合中。

Conformal Risk Control（CRC）把目标扩展到更一般的单调损失，例如：

- 假阴性率；
- 多标签漏标率；
- 分割区域损失；
- token-level F1 等任务风险。

因此 CRC 特别适合 reBEN 多标签分类和 Sen1Floods11 分割，但它不是导师所提 GeoConformal 论文的同一个方法：

- GeoConformal 重点解决“空间不同地点应该有不同的不确定性范围”；
- CRC 重点解决“不只控制覆盖错误，还能控制更一般的任务损失”。

二者未来甚至可以结合：先用地理距离对校准样本加权，再控制位置相关的任务风险。但这属于进一步研究，不是当前实现。

---

## 四、目前实现到底属于哪一部分

目前 AlphaEarth 实现采用的是 basic split conformal classification：

1. 将数据按空间块分为 train、calibration 和 test；
2. 在校准集计算 \(1-p_{\mathrm{true}}\)；
3. 把所有校准分数放在一起排序；
4. 对每个目标 coverage 计算一个全局 \(q_{\mathrm{hat}}\)；
5. 所有测试地点使用同一个阈值生成预测集合；
6. 按国家、地区、土地类别等切片审计 coverage。

代码证据见：

- 空间块拆分：[run_alphaearth_landcover_full_audit.py](../scripts/analysis/run_alphaearth_landcover_full_audit.py#L183)
- 全局 conformal 阈值和预测集合：[run_alphaearth_landcover_full_audit.py](../scripts/analysis/run_alphaearth_landcover_full_audit.py#L356)

目前已经实现的共同基础包括：

| 组成部分 | Standard split CP | GeoConformal | 当前项目 |
|---|---:|---:|---:|
| 独立校准集 | 需要 | 需要 | 已实现 |
| 不符合度分数 | 需要 | 需要 | 已实现 |
| 覆盖率目标 | 需要 | 需要 | 已实现 |
| 预测集合/区间 | 需要 | 需要 | AlphaEarth 已实现 |
| 总体 coverage 和效率 | 需要 | 需要 | 已实现 |
| 地理切片 coverage 审计 | 非必需 | 可分析 | 已实现 |
| 每个测试点的地理距离权重 | 不需要 | 核心步骤 | 未实现 |
| 每个地点不同的 \(q_{\mathrm{geo}}(x)\) | 不需要 | 核心步骤 | 未实现 |
| bandwidth 选择和敏感性 | 不需要 | 需要 | 未实现 |

所以最准确的判断是：

> 工程基础与 GeoConformal 很接近，但论文最有辨识度的科学步骤尚未接入。

---

## 五、最通俗的 Conformal Prediction 例子

假设模型判断一张遥感图像的土地类型。

普通预测只输出：

> 草地

Conformal Prediction 可能输出：

> {草地，灌木地}

对于非常确定的图像，可能仍然只输出：

> {水体}

对于很困难的图像，可能输出：

> {草地，灌木地，农田}

提前规定目标覆盖率，例如 90%。Conformal Prediction 用一批独立的校准数据决定“集合应该放多宽”，目标是：

> 对未来与校准数据条件相似的样本，真实类别平均约有至少 90% 的概率包含在预测集合里。

它不是让模型突然变准，而是让模型能够诚实地表达：

> “我不确定是草地还是灌木地，所以两个都报出来。”

代价是集合可能变大。因此必须同时看两个指标：

- **Coverage：**真实答案有多少比例落在集合里；
- **Set size：**平均每次给出多少个候选答案。

如果 coverage 很高，但每次都把所有类别放进去，就没有实际价值。

### GeoConformal 在这个例子中增加了什么

普通 conformal 可能对全球所有地点使用同一个阈值。

GeoConformal 会进一步考虑：

> 这张图像所在地点附近，历史校准样本是否特别容易混淆草地和灌木地？

如果附近误差很大，预测集合可能放宽为：

> {草地，灌木地，农田}

如果附近模型一直很稳定，集合可能只需要：

> {草地}

因此，GeoConformal 输出的是位置相关的不确定性，而不是只有一套全球阈值。

---

## 六、AlphaEarth 实现做了什么

可以把当前实现理解为四步。

### 第一步：模型输出各类别概率

例如：

| 类别 | 概率 |
|---|---:|
| 草地 | 0.55 |
| 灌木地 | 0.30 |
| 农田 | 0.10 |
| 其他 | 0.05 |

### 第二步：用校准集决定集合阈值

AlphaEarth 实现不是拿测试集调阈值，而是单独划出空间块校准集。

校准集告诉程序：

> 如果要达到约 90% coverage，预测集合需要放宽到什么程度？

当前版本计算的是一个全局阈值；GeoConformal 升级版需要为每个测试地点根据附近校准点重新计算地理加权阈值。

### 第三步：在测试集上生成预测集合

当前版本可能得到：

> {草地，灌木地}

然后检查 WorldCover 标签是否在集合中。

### 第四步：把覆盖失败交给 BWER 审计

普通 conformal 主要看：

> 全球平均 coverage 是否接近 90%？

BWER 框架继续追问：

> 全球平均达到 90%，但非洲、南美洲、草地类别或某些国家是否只有 70%？哪些部署切片承担了最多的覆盖失败？

两者的分工是：

```text
Conformal / GeoConformal
    ↓
产生预测集合和“是否覆盖真实标签”
    ↓
计算每个国家、类别和地区的误覆盖率
    ↓
BWER
    ↓
衡量最差部署切片是否明显比平均水平更糟
```

这就是 Conformal-BWER 的核心意义。升级为 GeoConformal 后，还可以研究：

> 使用地理局部校准以后，哪些地区的 coverage gap 被修复，哪些地区仍然失效？

---

## 七、为什么不能只看 Conformal-BWER 一个数

假设目标是 90% coverage，即允许 10% 误覆盖率。

| 模型 | 各地区误覆盖率 | BWER 差距 |
|---|---|---:|
| A | 所有地区都是 10% | 接近 0 |
| B | 所有地区都是 30% | 也接近 0 |

模型 B 的各地区“同样差”，所以公平差距可能接近零，但它完全没有兑现 90% coverage。

正式结果至少应同时报告：

- 总体 coverage；
- 最差切片 coverage；
- 平均 set size；
- Conformal-BWER；
- 最差切片超过目标误覆盖率多少；
- GeoConformal 中各切片的平均 set size 或区间宽度。

最后一项也很重要：两个地区都达到 90% coverage，但如果一个地区平均只需 1.2 个候选标签，另一个地区平均需要 6 个标签，后者获得的预测服务明显更模糊。公平审计需要同时衡量“是否覆盖”和“为了覆盖付出了多大的不确定性代价”。

也就是说：

> BWER 衡量差异，coverage violation 衡量是否合格，set size/interval width 衡量预测是否有用。

---

## 八、Selective-BWER 与 Conformal-BWER 是否重复

二者相关，但不是同一个指标，也不应该互相替代。

### Selective-BWER

模型仍然只输出一个类别，但允许拒绝回答：

> 置信度足够高：输出“草地”；置信度不足：拒绝预测。

它回答：

- 只保留最有信心的样本后，平均错误率是否下降？
- 哪些国家在保留样本中仍然错误最多？
- 哪些地区被拒绝得更多？

### Conformal-BWER

模型不一定拒绝，而是扩大候选集合：

> 不确定时输出 {草地，灌木地}。

它回答：

- 真实类别是否达到目标覆盖率？
- 哪些地理切片更容易落在集合之外？
- 哪些切片必须获得更大的集合才能达到相同 coverage？

### GeoConformal-BWER

它进一步让集合大小由局部地理误差决定：

> 每个地点根据附近历史误差获得不同的阈值和集合。

可以把三者记成：

- **Selective：**不确定就不回答；
- **Conformal：**不确定就多给几个答案；
- **GeoConformal：**根据这个地点附近有多难预测，决定多给几个答案；
- **BWER：**检查哪些部署群体仍然获得更差、更容易失败或更模糊的服务。

因此 GeoConformal 不是 Selective-BWER 的替代品。二者还可以组合：使用 GeoConformal 的 set size 或 interval width 作为拒绝依据，再用 Selective-BWER 检查拒绝负担是否集中在某些群体。

---

## 九、它是不是像测评参数一样，大部分数据集都能用

原理上可以广泛使用，但不像 accuracy 一样只拿到硬标签就能直接计算。Conformal 需要：

- 独立校准集；
- 真实标签；
- 合适的不符合度分数，通常来自概率、logits、残差或任务损失；
- GeoConformal 还需要可靠坐标、距离定义和 bandwidth；
- 不同任务需要定义不同的预测集合或风险控制对象。

| 方法 | 需要什么 | 可迁移性 |
|---|---|---|
| BWER | 样本损失＋切片信息 | 很高 |
| Selective-BWER | 损失＋置信度 | 较高 |
| Conformal-BWER | 损失＋校准集＋不符合度分数/概率 | 高，但需要任务适配 |
| GeoConformal-BWER | 上述信息＋坐标＋地理权重＋bandwidth | 高，但要求空间协议 |
| 分组条件 Conformal | 上述信息＋每组足够支持＋更强校准设计 | 要求更高 |

四个实验理论上都能纳入 conformal 审计，但不能把单标签分类的同一段代码原封不动复制到多标签和分割任务。

---

## 十、四个实验分别怎样使用

### 1. AlphaEarth：最自然，也是 GeoConformal 的首选实现

这是单标签土地覆盖分类。

普通预测：

> 草地

Conformal 输出：

> {草地，灌木地}

当前已经可以检查：

- 全球 coverage；
- 各国家 coverage；
- 各土地类型 coverage；
- 国家×类别 coverage；
- 平均 set size；
- Conformal-BWER。

当前流程已经完成，但使用的是全局 conformal 阈值。升级为 GeoConformal 后，应增加：

- 每个测试地点到校准点的地理距离；
- 高斯核或其他预声明空间权重；
- bandwidth 的独立验证和敏感性；
- 每个测试地点不同的 \(q_{\mathrm{geo}}(x)\)；
- standard CP 与 GeoCP 的同测试集比较；
- coverage、set size 和 BWER 的共同报告。

AlphaEarth 有全球坐标、大样本和空间块设计，是回答导师问题最有说服力的实验。

### 2. fMoW-Sentinel：适合，但缺完整概率字段

它也是单标签分类，有 62 个类别。

理论上可以输出：

> {机场，港口，工业设施}

旧实验只保存了：

- 最高概率；
- 预测类别；
- 是否正确。

没有保存全部 62 类 probability vector，因此无法知道第二、第三候选类别，也不能正式构造分类预测集合。当前实现只能做到：

> 置信度超过阈值时接受，否则拒绝。

这属于 selective prediction 或 calibrated confidence-threshold diagnostic，不是完整 conformal prediction，也不是 GeoConformal。

修复方式通常不需要重新训练，只需重新推理并保存完整 62 维概率向量。随后可以利用 location coordinates 构造地理加权预测集合。

### 3. reBEN/CROMA：可以使用，但定义不同

reBEN 是多标签任务，一张图可能同时属于：

> {农田，草地，水体}

它的预测本来就像一个标签集合。因此需要回答：

- 真实标签是否全部包含在输出集合中？
- 平均漏掉多少真实标签？
- 能否控制 false negative rate？
- 能否控制每张图的漏标比例？

如果直接复制单标签 conformal，可能产生非常大的无用集合。更适合的是 Conformal Risk Control，例如规定：

> 预测标签集合的平均漏标率不超过 10%。

然后用 BWER 检查：

> 哪些国家或类别的漏标风险特别高？

如果进一步引入 GeoConformal，则可让不同地点根据附近多标签校准误差获得不同阈值。但这属于 GeoCP 与 CRC 的新组合，不是导师论文中已经直接验证的算法。

### 4. Sen1Floods11：理论可行，但最困难

这是像素级洪水分割。

最直接的像素集合可能是：

- {洪水}
- {非洪水}
- {洪水，非洪水}

但大量边界像素可能得到 {洪水，非洪水}，结果不一定有实用价值。更好的目标可能是：

- 控制整张图的洪水漏检率；
- 控制事件级 false negative rate；
- 输出高置信洪水区、模糊边界区和高置信非洪水区；
- 使用 Conformal Risk Control 控制事件级分割损失；
- 使用地理或事件距离进行局部校准。

然后 BWER 检查：

> 哪些洪水事件的覆盖或漏检控制明显失效？

理论上可做，但现有正式输出没有保存足够的像素/图块概率，需要重新推理或重新导出概率图；空间相关和计算规模也明显高于 AlphaEarth。

---

## 十一、GeoConformal 会不会改变其他 BWER 分数

如果管线设计正确，不会改变 Raw-BWER、Standardised-BWER 或已有 Selective-BWER 的定义。推荐保持平行分支：

```text
固定 final test 上的原始预测
├── 原始损失 → Raw-BWER
├── 组成标准化损失 → Standardised-BWER
├── 置信度筛选 → Selective-BWER
├── 全局 conformal 集合 → Conformal-BWER
└── 地理加权 conformal 集合 → GeoConformal-BWER
```

Conformal/GeoConformal 相对于模型属于 **post-hoc calibration / 后处理层**；相对于 BWER 属于生成新损失的前置步骤。它通常不修改模型权重，也不会回头改变原始预测。

唯一需要控制的是测试集一致性。如果为了建立校准集而从旧测试集中拿走部分样本，Raw-BWER 也会因为测试总体变化而变化。正式比较应该在同一个固定 final test 上同时计算 Raw、Selective、Conformal 和 GeoConformal-BWER，校准集不能参与最终测试。

---

## 十二、最准确的整体理解

- **Raw-BWER：**哪些部署切片的原始错误风险最高？
- **Standardised-BWER：**控制类别组成后，哪些地理切片仍然更差？
- **Selective-BWER：**模型只回答有信心的样本后，哪些切片仍然错误更多，哪些切片被拒绝更多？
- **Conformal-BWER：**模型给出目标覆盖率的预测集合后，哪些切片仍然更容易落在集合之外？
- **GeoConformal-BWER：**根据局部地理误差调整集合后，哪些切片仍然覆盖不足，或者必须得到特别大的集合才能达到相同 coverage？

Conformal 可以成为 BWER 的通用不确定性扩展，但需要为不同任务建立适配器：

1. 单标签分类适配器：AlphaEarth、fMoW；
2. 多标签风险控制适配器：reBEN；
3. 分割风险控制适配器：Sen1Floods11；
4. 在前三类适配器上可选加入地理加权层，形成 GeoConformal-BWER。

一旦输入规范、校准划分、输出字段和报告指标固定，未来其他 GeoFM 数据集就可以根据任务类型复用这套审计协议。

---

## 十三、建议的下一步

不需要立即在四个实验上全部重跑。最有信息量的顺序是：

1. **先在 AlphaEarth 加入真正的 GeoConformal 地理加权。**现有训练、概率、空间块和 BWER 结果全部保留。
2. 在同一 final test 上比较 standard split CP 与 GeoCP：总体 coverage、最差切片 coverage、平均 set size、set-size disparity、Conformal-BWER。
3. 用独立 validation 选择 bandwidth，并报告多个尺度的敏感性；不能用 final test 调参。
4. 加入普通 CP、GeoCP、无空间随机权重作为控制，检验地理加权是否真的带来新增价值。
5. 只有 AlphaEarth 显示明确新增价值后，才补 fMoW 完整 probability vector。
6. reBEN 与 Sen1 优先保留 CRC/任务风险扩展，不必为了形式统一强行复刻单标签集合。

如果 GeoCP 相比普通 CP不能改善切片 coverage、set size 或 held-out 空间稳定性，那么它不应成为主文核心；如果能够改善，GeoConformal-BWER 将成为连接“不确定性校准”和“GeoFM 部署公平性”的强扩展。

---


