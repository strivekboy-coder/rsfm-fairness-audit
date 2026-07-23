# GeoBWER 公式、指标家族与编码规范

## 1. 核心科学问题

GeoBWER 回答的不是“模型总体准确吗”，而是：

> 在预先声明的部署分布中，风险最高的 $\beta$ 比例部署质量，相对平均部署风险额外承担了多少损失？

设切片集合为 $\mathcal G$，有界任务损失 $L\in[0,1]$，切片风险为

\[
R_g=\mathbb E[L\mid G=g],
\]

预先声明的部署权重为 $\mu_g\ge0$，且 $\sum_g\mu_g=1$。部署平均风险为

\[
m_\mu(R)=\sum_g\mu_gR_g.
\]

定义风险最高的 $\beta$ 比例部署质量的尾部风险：

\[
T_{\beta,\mu}(R)
=\inf_{\eta\in\mathbb R}
\left\{
\eta+\frac1\beta\sum_g\mu_g(R_g-\eta)_+
\right\},
\qquad 0<\beta\le1.
\]

核心指标为

\[
\boxed{
\operatorname{BWER}_{\beta,\mu}(R)
=T_{\beta,\mu}(R)-m_\mu(R)
}
\]

它的单位与原任务损失相同。BWER 为 0 只表示所声明切片下没有可见风险差异，不表示模型准确，也不表示社会公平已经实现。因此任何正式结果必须同时报告：

\[
\boxed{m_\mu,\quad T_{\beta,\mu},\quad \operatorname{BWER}_{\beta,\mu}}
\]

## 2. 风险包络解释

尾部风险等价于

\[
T_{\beta,\mu}(R)
=\sup_{w}
\sum_g\mu_gw_gR_g,
\]

其中

\[
0\le w_g\le1/\beta,
\qquad
\sum_g\mu_gw_g=1.
\]

所以 BWER 是：在允许向高风险切片进行不超过 $1/\beta$ 倍的最坏合法部署重加权时，风险相对原部署均值最多上升多少。该定义直接来自 AVaR/CVaR 与 generalized deviation 理论，而不是声称重新发明 CVaR。[Rockafellar 与 Uryasev](https://doi.org/10.1016/S0378-4266(02)00271-6)、[Rockafellar、Uryasev 与 Zabarankin](https://doi.org/10.1007/s00780-005-0165-8)、[Williamson 与 Menon](https://proceedings.mlr.press/v97/williamson19a.html)

## 3. 必须证明和测试的性质

### 3.1 旧 BWER 嵌入

当 $G$ 个切片等权且 $\beta=k/G$ 时，fractional BWER 精确等于旧版“最差 $k$ 个完整切片均值减全切片均值”。旧 BWER1 因而是最终参数的离散特例，不是独立竞争指标。

### 3.2 端点与 profile

\[
\lim_{\beta\downarrow0}\operatorname{BWER}_{\beta,\mu}
=\max_gR_g-m_\mu,
\qquad
\operatorname{BWER}_{1,\mu}=0.
\]

BWER 随 $\beta$ 不增。正文主值固定为 `BWER@10`；`{.05,.10,.20,.30}` 形成预注册 profile，不允许事后挑最显著的 $\beta$。

### 3.3 质量保持的同风险克隆不变性

一个切片若被拆为多个相同风险子切片，且总部署质量不变，BWER 不变。这修复等权 `ceil(βG)` 对切片数量的机械敏感。

### 3.4 ties 的数值唯一性

边界风险并列时，最优尾部成员权重可能不唯一，但 $T$ 与 BWER 的值唯一。API 不应通过字符串排序把某个并列国家伪装成唯一尾部成员；应输出 tie set、边界质量和聚合尾部质量。

### 3.5 plug-in 上偏

令 $D(R)=T_{\beta,\mu}(R)-m_\mu(R)$。因为 $D$ 是凸函数，若固定支持上的 $\widehat R$ 对 $R$ 无偏，则

\[
\mathbb E[D(\widehat R)]\ge D(R).
\]

这就是“先选看起来最坏的组，再在同一数据上报告”的 winner's curse。它不仅发生在 11 个洪水事件，也会发生在 145 个国家和数百个国家×类别切片。

## 4. 精确数值算法

不需要数值优化器。对 $R_g$ 从高到低稳定排序，令剩余尾部质量初始为 $\beta$，依次取

\[
a_g=\min(\mu_g,\text{remaining}),
\]

直到 $\sum_ga_g=\beta$。然后

\[
T_{\beta,\mu}=\frac1\beta\sum_ga_gR_g,
\qquad
\operatorname{BWER}=T_{\beta,\mu}-\sum_g\mu_gR_g.
\]

输出中应保存 `selected_mass=a_g`，而不只保存布尔 `is_tail`。

## 5. 指标家族：同一泛函，不同风险输入

### 5.1 Raw-BWER

\[
R_g^{\rm raw}=\mathbb E[L\mid G=g].
\]

回答原始任务错误是否集中在某些事件、国家、类别、位置或传感器条件。

### 5.2 Standardised-BWER

给定要控制的组成变量 $Z$ 和所有模型共用的目标分布 $q_z$：

\[
R_g^{\rm std}=\sum_zq_zR_{gz}.
\]

正式可识别性要求：对所有 $q_z>0$，每个被比较切片都具有足够支持。实现规则：

- `strict`：缺任何目标层即返回 `not_identified`；
- `common_support`：在看模型性能前冻结共同可观测层，并重新声明新的 $q$；结果名必须标明 overlap estimand；
- `partial_bounds`：有可靠损失上下界时可给部分识别区间；
- 旧 `renormalize` 仅保留为 `legacy_diagnostic`，不得称正式 Standardised-BWER。

目标 $q$ 必须由训练/校准数据、外部部署人口或均匀政策目标预先确定，不能每个模型各自从测试结果产生。

### 5.3 Selective-BWER

在独立校准数据上选择全局阈值 $\tau_c$，使总体业务覆盖率接近预注册 $c$。令

\[
A_i(c)=\mathbf1\{s_i\ge\tau_c\},
\qquad
R_g^{\rm sel}(c)
=\frac{\mathbb E[L_iA_i(c)\mid G=g]}
{\mathbb E[A_i(c)\mid G=g]}.
\]

同时必须报告

\[
C_g(c)=\mathbb E[A_i(c)\mid G=g],
\]

即各组被服务/接受的比例。只报接受样本的低错误率会掩盖某些群体被大量拒绝。阈值不得按测试组分别调节；接受支持不足时返回无效状态。

### 5.4 Conformal-BWER

单标签分类的预测集合记为 $\mathcal C_\lambda(X)$，误覆盖损失为

\[
L_i^{\rm cov}=\mathbf1\{Y_i\notin\mathcal C_\lambda(X_i)\},
\qquad
R_g^{\rm conf}=\mathbb E[L_i^{\rm cov}\mid G=g].
\]

然后对 $R_g^{\rm conf}$ 计算同一 BWER。还需报告：

- marginal coverage；
- worst-slice coverage；
- 平均/分位数 set size；
- empty、singleton、multi-set rate；
- `coverage_debt_g=[R_g^{conf}-α]_+`；
- coverage-debt 的 worst slice 和尾部均值。

BWER 测量“覆盖失败是否不均”，coverage debt 测量“是否兑现目标”。两者不可合并成一个数。

多标签和分割任务不复制单标签集合算法，而使用 [Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html) 控制预注册的单调损失，例如多标签漏标率或事件级洪水漏检风险，再用 BWER 审计该承诺在哪些切片失效。

### 5.5 GeoConformal-BWER：空间适配器，不是新核心

[GeoConformal Prediction](https://doi.org/10.1080/24694452.2025.2516091) 用地理距离对校准残差加权，使每个测试地点得到局部区间；原论文验证的是空间回归和插值。其典型权重为

\[
w_i(x)\propto K\!\left(\frac{d(s_i,s_x)}h\right).
\]

本项目当前 AlphaEarth 流程属于空间块 split conformal，并没有实现该论文的局部地理加权。若增加 GeoConformal-BWER，应作为 AlphaEarth 的高价值 comparator：

1. 用训练、校准、带宽验证、最终测试四分；
2. 仅在验证集调 $h$，不得在测试集挑覆盖最好带宽；
3. 报告局部有效样本量、区间/集合效率和稀疏区失效；
4. 比较 standard CP、GeoCP 和必要时 GeoSIMCP；
5. 再用 BWER 检查局部方法是否真正减少 country/class coverage debt。

把 GeoCP 扩展到多类 prediction set 需要额外方法验证，不能声称是原论文原样实现。[官方代码](https://github.com/pengtum/geoconformal)

## 6. Certified BWER Profile

### 6.1 apparent 描述性点估计

全样本 plug-in BWER 保留，名称必须显式为 `apparent_bwer`。它回答“样本中观察到多少”，不冒充无偏总体估计。

### 6.2 独立确认下界

发现集 A 仅选择风险包络中的可行权重 $w^A$。在独立评估集 B 估计

\[
d(w^A,R)=\sum_g\mu_g(w_g^A-1)R_g.
\]

因为 $w^A$ 是风险包络的可行解，必有

\[
d(w^A,R)\le\operatorname{BWER}_{\beta,\mu}(R).
\]

因此对固定 $w^A$ 的 $d$ 构造一侧下置信界，就得到总体 BWER 的有效保守证书。该方法主要用于自动搜索、交叉切片和连续空间热点等**数据驱动发现**：可预先声明 A→B 与 B→A 的对称交叉确认，但不得把它称为无偏 oracle BWER。国家、事件、类别、模态等预注册固定切片不强制拆半，以免无必要地损失低支持任务的功效；它们使用下一节的同时风险带。

### 6.3 同时风险带传播

若以至少 $1-\alpha$ 的概率同时满足

\[
|\widehat R_g-R_g|\le e_g,\quad\forall g,
\]

则

\[
|D(\widehat R)-D(R)|
\le T_{\beta,\mu}(e)+\mathbb E_\mu[e].
\]

实现上用 **studentized cluster/spatial multiplier max-T** 构建切片风险同时带，再以确定性界传播到 BWER。这是四任务固定切片的正式主区间：两侧 95% 区间用于估计，一侧 95% 下界用于声明“存在可确认的正差距”。普通 i.i.d. bootstrap 只在协议证明样本本身即独立单位时允许；地理任务正式模式默认禁止静默退化。

### 6.4 模型差值与排名反转

对模型 A、B，在性能未知时冻结共同切片集 $\mathcal G_{AB}$、同一 $\mu$ 和 $\beta$，计算

\[
\Delta D=D(R^A;\mathcal G_{AB})-D(R^B;\mathcal G_{AB}).
\]

同一事件、地点或空间块必须成对重采样，并在每个 replicate 中重算完整非线性泛函。两个独立 CI 是否重叠不能替代差值 CI。

正式模型排序在共同支持上使用配对 cluster/spatial multiplier 95% 差值区间。预注册多个主要模型对时采用 max-T 联合控制；若实现条件不满足，使用 Holm 校正作为透明保底。直接对非光滑 BWER 做 percentile/studentized bootstrap 只作为敏感性，不取代主同时风险带。

### 6.5 必报稳定性诊断

- `deployment_effective_groups = 1/Σμ_g²`；
- 令 $q_g=a_g/\beta$，报告 `tail_effective_groups = 1/Σq_g²`；
- `max_tail_atom_share=max(q_g)`；
- tail membership / selected-mass stability；
- valid deployment mass、missing mass、最小/中位支持；
- tie mass 与 boundary mass。

例如 11 个等权事件、$\beta=.1$ 时，fractional 算法虽然精确审计 10% 部署质量，但有效尾部组数约为 1.20，仍需明确提示“证据主要由一个事件支配”。这比只报切片数更诚实。

### 6.6 已冻结的全任务推断层级

“全任务主区间”指统一的**推断逻辑**，不是强迫四个数据集使用同一个公里数或同一个 cluster 字段：

1. 固定切片：全评估集 apparent BWER + studentized cluster/spatial max-T 同时风险带；
2. 数据驱动切片：discovery→independent confirmation，或预注册的双向交叉确认；
3. 模型差值：共同支持、同一 $\mu$/$\beta$、配对 cluster 区间；
4. 严格保底：有界损失的 simultaneous Hoeffding/Bernstein 界；过宽时如实报告“可描述但未认证”；
5. 敏感性：直接 BWER bootstrap、相邻 block scales、经验质量权重和 legacy BWER1。

cluster 数的软件门是预注册 guardrail，不冒充普适定理：独立 cluster 至少 30 时可进入默认 multiplier 候选；15–29 时只有任务模拟覆盖率通过才认证，并强制 small-cluster 敏感性；少于 15 时不做未来超总体的渐近证书，只给固定部署集描述、严格界或 leave-one-cluster/event 稳定性。

空间块选择器也已冻结：只使用校准/验证数据或与测试效应符号无关的信息估计损失相关范围；在原生独立单元、约 1×、1.5×、2×相关范围的预注册候选中，选择通过名义覆盖/假阳性门且功效最高的最小尺度。若没有候选通过，输出 `inference_not_certified`，不得按测试集显著性挑尺度。

## 7. 编码模块

建议新增模块，而不是在现有 630 行 `bwer.py` 内继续叠加条件：

```text
src/rsfm_fairness_audit/
  bwer_core.py              # 总体参数、fractional tail、profile、legacy
  bwer_standardization.py   # strict/common-support/partial bounds
  bwer_inference.py         # honest confirmation、simultaneous bands、paired CI
  bwer_risk_adapters.py     # raw/selective/conformal/CRC/task losses
  bwer_protocol.py          # 签名、hash、invalid states、comparison contracts
  bwer_report.py            # Certified BWER Profile 表与图
  bwer.py                   # 兼容 façade，转调新模块
```

现有 `bwer_v2.py` 是 Sen1 post-hoc 流水线版本，不应再承担“BWER2 指标”含义；短期保留文件名以兼容 CLI，内部改调新 core，并在文档标为 `posthoc_pipeline_v2`。

核心数据类建议：

```python
@dataclass(frozen=True)
class BWERProtocol:
    beta: float
    deployment_weighting: str
    support_rule: str
    missingness_rule: str
    inference_target: str
    inference_method: str = "cluster_maxt"
    confidence_level: float = 0.95
    metric_version: str = "geobwer_fractional_1.0"

@dataclass(frozen=True)
class BWERResult:
    mean_risk: float
    tail_risk: float
    bwer: float
    selected_mass: np.ndarray
    protocol_signature: str
    validity: str
```

API 目标采用“风险核心 + 任务适配器”，使新数据集只需提供损失、切片和独立推断单位，而不必复制四条实验管线：

```python
from rsfm_fairness_audit.geobwer import audit, compare, Protocol

result = audit(
    loss=loss,
    groups={"country": country},
    unit_id=location_id,
    spatial_block_id=block_id,
    protocol=Protocol(beta=0.10, group_weight="equal"),
)
result.to_report("audit_out/")
```

标准适配器：

```text
from_multiclass(...)    # fMoW、AlphaEarth
from_multilabel(...)    # reBEN
from_segmentation(...)  # Sen1Floods11
from_conformal(...)     # prediction set / CRC 的 miscoverage 与效率
```

最小 CLI：

```text
geobwer audit predictions.parquet --protocol geobwer.yaml --out audit_out/
geobwer compare model_a.parquet model_b.parquet --paired-on independent_unit_id --out compare_out/
```

输出至少包含 versioned JSON/CSV/Parquet report card、`metric_version`、`protocol_hash`、point/CI/LCB、support、tail stability 和 invalid states。API MVP 必须在正式四任务运行前实现，让四任务本身成为兼容性测试；完善文档、示例数据和 PyPI/Conda 发布可在正式结果稳定后完成。

## 8. 无效状态

不得用 NaN 或 warning 混合所有失败原因。最少支持：

```text
valid
insufficient_slices
insufficient_independent_units
insufficient_tail_effective_groups
not_identified_missing_standardization_cells
no_common_support
missing_probability_output
calibration_leakage
invalid_inference_target
reference_product_not_comparable
missing_independent_unit
inference_not_certified
spatial_block_not_calibrated
```

## 9. 实现测试门

### 数学单测

- 旧 BWER 嵌入；
- $\beta\downarrow0$、$\beta=1$；
- profile 单调；
- weighted fractional boundary 质量精确等于 $\beta$；
- 同风险质量保持克隆不变；
- ties 排列不变；
- 常量风险返回 0；
- 正线性缩放和平移不变；
- invalid weight/support/missingness 状态。

### 统计单测与模拟

- 同质零差异；
- 连续弱尾、强尾、多尾；
- fMoW 长尾支持；
- Sen1 小 $G$；
- 空间相关强度和 block size 梯度；
- cluster=slice clone ID；
- strict standardisation 与 positivity；
- paired model difference 和 common support；
- ties、交叉切片、多重发现；
- 分类、多标签和分割损失。

名义 95% 方法只有在关键零假设与替代情景的 Monte Carlo 置信区间与 95% 相容、且具有可接受 power 时，才能成为默认主区间。
