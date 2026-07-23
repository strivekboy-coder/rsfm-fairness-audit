# GeoBWER 冻结公式、推断协议与公开 API

## 1. 唯一核心总体参数

给定预先定义的部署切片分区 \(g=1,\ldots,G\)，单元损失 \(L\)，切片风险

\[
R_g = \mathbb E[L\mid G=g],
\]

以及和为 1 的**审计目标测度** \(\mu_g\)，定义部署平均风险

\[
m_\mu = \sum_g \mu_g R_g.
\]

上侧 \(\beta\) 部署质量的 exact fractional tail risk 为

\[
T_{\beta,\mu}(R)
=
\sup_{q}
\left\{
\sum_g q_gR_g:
q_g\ge0,\ \sum_gq_g=1,\ q_g\le \mu_g/\beta
\right\}.
\]

核心参数为

\[
\boxed{
\operatorname{BWER}_{\beta,\mu}
=T_{\beta,\mu}(R)-m_\mu
}
\]

正式名称使用 **GeoBWER / Geospatial β-Weighted Excess Risk**。它回答的是：最坏的 \(\beta\) 部署质量相对总体部署平均额外承担多少风险。主值固定为 `BWER@10`，同时报告 \(\beta\in\{0.05,0.10,0.20,0.30\}\) 的 profile。

这里的 \(\mu\) 不是可忽略的实现细节，而是参数定义的一部分：

- `balanced/equal`：每个预注册切片获得相同审计质量，是四任务主分析；
- `observed/empirical`：按测试总体中的观测频率加权，只作敏感性；
- `external/custom`：使用预先给定的部署暴露、人口或业务权重。

不同 \(\mu\) 回答不同问题，不能把它们混成一个分数。正式输出将 `audit_measure`、`partition_rule`、`estimand_scope` 和 `dependence_design` 纳入 protocol hash。

## 2. 与 BWER 1.0 的关系

当切片等权且 \(\beta=k/G\) 时，fractional BWER 与 BWER 1.0 取最坏 \(k\) 个完整切片的结果完全相同。非整数 \(\beta G\) 时，新定义只取边界切片达到精确 \(\beta\) 质量所需的分数质量，避免 `ceil(beta*G)` 的阶跃。

同时有：

- \(\operatorname{BWER}_{1,\mu}=0\)；
- \(\beta\downarrow0\) 时趋向 \(\max_gR_g-m_\mu\)；
- 风险整体平移不改变 BWER；
- BWER 非负，但 **BWER=0 只说明切片间没有尾部超额，不说明模型风险合格**，所以必须同时报告 mean risk 和 tail risk。

底层 tail risk 与 weighted AVaR/CVaR 风险度量家族相连；项目的方法贡献不是声称重新发明 CVaR，而是把它变成 GeoFM 场景中具有切片构造、严格标准化、空间推断、common-support 比较、任务风险适配和可复算输出合同的公平性审计参数。

## 3. 风险包络与切片细分性质

令

\[
\mathcal Q_{\beta,\mu}
=
\left\{q:\ q_g\ge0,\ \sum_gq_g=1,\ q_g\le\mu_g/\beta\right\}.
\]

则

\[
\operatorname{GeoBWER}_{\beta,\mu}(R)
=
\sup_{q\in\mathcal Q_{\beta,\mu}}
\sum_g(q_g-\mu_g)R_g.
\]

这个风险包络给出直接的部署解释：对手可以把审计质量重新分配到较差切片，但任何切片最多被放大到原质量的 \(1/\beta\)；GeoBWER 是该受约束最坏重加权相对原审计测度的超额风险。

**保持审计测度的细分单调性。** 若一个粗切片被细分为若干子切片，子切片权重之和等于父切片权重，父风险等于子风险的测度加权均值，且 \(\beta\) 与总体审计测度不变，则细分后的 GeoBWER 不小于细分前。若子切片风险完全相同，则取等号。这个结论只针对 measure-preserving refinement；任意改权重、删除支持或改变目标总体不属于该定理。

## 4. 一个核心泛函，多种风险输入

| 名称 | 送入 GeoBWER 的风险 | 必须共同报告 | 角色 |
|---|---|---|---|
| Raw-GeoBWER | 每个独立单元的预注册任务损失 | mean/tail/BWER、支持、cluster 数 | 主结果 |
| Standardised-GeoBWER | \(R_g^{std}=\sum_c\pi_c^*E[L\mid g,c]\) | 目标组成、缺失 cell、可识别状态 | 区分地理差异与类别组成差异 |
| Selective-GeoBWER | 校准集确定阈值后，在 accepted 样本上的任务损失 | 总体及分组 coverage、selective risk | 回答拒答后尾部差异是否仍在 |
| Conformal-GeoBWER（分类） | prediction set 的 miscoverage loss | marginal/slice coverage、set size、target violation | 审计不确定性覆盖债务 |
| CRC-GeoBWER（多标签/分割） | 校准控制后的 per-unit false-negative risk | 校准风险、阈值、效率和 slice risk | 任务适配的不确定性扩展 |

这些不是五个互相竞争的总指标。核心风险泛函只有一个；Standardised、Selective、Conformal/CRC 改变的是风险构造或目标总体，并通过独立 protocol hash 标记。

## 5. 正式统计协议

1. 主审计测度为切片等权宏平均；经验样本权重只作敏感性，外部暴露/人口权重仅在其 estimand 有权威依据时使用。
2. 每个审计轴单独形成互斥完备分区；`country`、`class`、`sensor` 等不同轴分别计算，交叉切片必须预先显式构造，不能把重叠集合当成同一个概率分布。
3. 固定切片使用完整测试集估计；不通过同一噪声重复“选最坏再声称确认”。主区间使用 cluster/spatial studentized multiplier max-T 同时风险带，再传播到 BWER。
4. 自动发现的切片必须 discovery/confirmation 分离；`confirm` 实现 A→B 与 B→A 双向 honest confirmation。
5. 模型差异只在模型无关的 common units/common groups、同一 \(\mu\) 上做 paired cluster inference；面板对预注册比较族做 family-wise 校正。
6. 空间任务没有 block/cluster 时 formal mode 硬失败，不退回 i.i.d. bootstrap。
7. cluster 数低于默认门槛时，除非该任务的 small-cluster simulation 已通过，否则状态为 `inference_not_certified`，不伪造窄区间。
8. strict standardisation 要求所有正目标权重 cell 有数据；缺失时返回 `not_identified`。`overlap` 和 partial bounds 是明确的敏感性估计，不冒充同一总体。

### 同时风险带到 GeoBWER 的认证

设同时风险带满足 \(|\widehat R_g-R_g|\le\delta_g\) 对所有固定切片同时成立。旧的有效半径为

\[
r_{\mathrm{weighted}}
=T_{\beta,\mu}(\delta)+\sum_g\mu_g\delta_g.
\]

风险包络还给出

\[
\lVert q-\mu\rVert_1\le 2(1-\beta),
\qquad
r_{\mathrm{TV}}
=2(1-\beta)\lVert\delta\rVert_\infty.
\]

正式实现使用两个有效上界的较小者：

\[
\boxed{r=\min(r_{\mathrm{weighted}},r_{\mathrm{TV}})}.
\]

若损失范围为 \([a,b]\)，参数本身满足

\[
0\le \operatorname{GeoBWER}_{\beta,\mu}\le(1-\beta)(b-a),
\]

所以认证区间为

\[
\left[
\max(0,\widehat{\operatorname{GeoBWER}}-r),
\min((1-\beta)(b-a),\widehat{\operatorname{GeoBWER}}+r)
\right].
\]

它在以下预注册条件下具有所声明的覆盖含义：风险带对全部固定切片同时有效、\(\mu\) 和分区固定、cluster/spatial block 与实际依赖结构相符。代码同时输出旧半径、新半径、采用的 envelope 和参数上界，便于审稿复算；使用更紧半径不改变点估计。

## 6. 最小 API

```python
from rsfm_fairness_audit import BWERProtocol, audit, compare, confirm

protocol = BWERProtocol(
    beta=0.10,
    beta_profile=(0.05, 0.10, 0.20, 0.30),
    deployment_weighting="equal",
    audit_measure="balanced",
    partition_rule="one_axis_at_a_time",
    missingness_rule="strict",
    estimand_scope="fixed_slice_universe",
    dependence_design="independent_clusters",
    group_variable="country",
    independent_unit_column="sample_id",
    cluster_column="site_id",
    inference_method="cluster_maxt",
)

result = audit(
    loss=loss,
    groups={"country": countries, "class": classes},
    unit_id=sample_ids,
    cluster_id=site_ids,
    protocol=protocol,
    n_bootstrap=2000,
)
result.to_report("outputs/geobwer")
```

CLI：`rsfm-audit geobwer-audit`、`geobwer-compare`、`geobwer-inventory`、`geobwer-multiclass-uncertainty` 和 `geobwer-multilabel-uncertainty`。

## 7. 适用域边界

- 公平性仍依赖“国家、事件、sensor、类别或交叉切片”这一构念选择；指标不能替研究者自动决定社会上正确的群体。
- 极少事件或 cluster 是信息不足，不是软件 bug；Sen1 的 11 个事件只支持固定事件宇宙主张，不能自动外推所有未来洪灾。
- WorldCover/Dynamic World 是参考产品，不是无噪声真值；AlphaEarth 主风险应表述为 map disagreement，并做 ambiguity/product sensitivity。
- 普通 split conformal 的有限样本保证依赖校准/测试可交换性。空间分块降低泄漏但不自动证明空间样本可交换；论文必须把 coverage 当作被审计对象，并把空间相关下的保证范围写清楚。
- `BWER@10` 是预注册主尺度而非自然常数；固定 profile 和 partition/scale sensitivity 用于检查结论是否只存在于单一尺度。
