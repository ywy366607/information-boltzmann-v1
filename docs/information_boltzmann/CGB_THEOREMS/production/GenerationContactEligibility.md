# GenerationContactEligibility

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactEligibility.lean) · [返回导航](../README.md)

模块源码说明：Local contact eligibility is a forward response, not occupancy alone For a column contact `w` with local participation `s`, the next activity is `hardSat (base + w * s)`.

## CGB.Generation.hardSat_eq_self

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactEligibility.lean#L26)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：对处于 [-1, 1] 区间内的任意实数 x，硬饱和函数 hardSat x 的值严格等于其自身 x。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {x : ℝ}, -1 ≤ x → x ≤ 1 → CGB.Generation.hardSat x = x
```

## CGB.Generation.hardSat_eq_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactEligibility.lean#L31)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：对任意实数 x ≥ 1，硬饱和函数 hardSat x 的值严格恒等于 1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {x : ℝ}, 1 ≤ x → CGB.Generation.hardSat x = 1
```

## CGB.Generation.contact_response_unsaturated_increment

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactEligibility.lean#L38)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在未饱和响应线性段内，参与度标量精确对应接触权重微扰下的前向有限差分比率。

源码说明：On one unsaturated response segment, participation is the exact local forward finite-difference factor of a contact-weight perturbation.

形式陈述：

```lean
∀ (base weight participation δ : ℝ),
  -1 ≤ base + weight * participation →
    base + weight * participation ≤ 1 →
      -1 ≤ base + (weight + δ) * participation →
        base + (weight + δ) * participation ≤ 1 →
          CGB.Generation.contactResponse base (weight + δ) participation -
              CGB.Generation.contactResponse base weight participation =
            δ * participation
```

## CGB.Generation.contact_response_saturated_increment_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactEligibility.lean#L53)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：有界响应饱和面不携带额外局部接触证据：只要两次扰动后的响应均达到 +1 饱和态，二者的差分增量精确为零。

源码说明：A bounded activity face carries no additional local contact evidence: as long as both responses remain saturated at `+1`, their difference is zero.

形式陈述：

```lean
∀ (base weight participation δ : ℝ),
  1 ≤ base + weight * participation →
    1 ≤ base + (weight + δ) * participation →
      CGB.Generation.contactResponse base (weight + δ) participation -
          CGB.Generation.contactResponse base weight participation =
        0
```

## CGB.Generation.contact_score_unsaturated_increment

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactEligibility.lean#L66)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当局部读出系数为 q 时，未饱和分值的增量等于 q 乘以接触扰动项，构成下一字节前向资格因子的显式构成。

源码说明：A score whose local read coefficient is `q` inherits both cases. This is the one-contact part of the next-byte forward eligibility before its language innovation is used as the third factor.

形式陈述：

```lean
∀ (q base weight participation δ : ℝ),
  -1 ≤ base + weight * participation →
    base + weight * participation ≤ 1 →
      -1 ≤ base + (weight + δ) * participation →
        base + (weight + δ) * participation ≤ 1 →
          q * CGB.Generation.contactResponse base (weight + δ) participation -
              q * CGB.Generation.contactResponse base weight participation =
            q * δ * participation
```
