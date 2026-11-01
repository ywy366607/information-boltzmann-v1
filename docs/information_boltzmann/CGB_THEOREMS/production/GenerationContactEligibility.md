# GenerationContactEligibility

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactEligibility.lean) · [返回导航](../README.md)

模块源码说明：Local contact eligibility is a forward response, not occupancy alone For a column contact `w` with local participation `s`, the next activity is `hardSat (base + w * s)`.

## CGB.Generation.hardSat_eq_self

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactEligibility.lean#L26)

解释：输入hardSat_eq_self形式化前提参量，经底层算子变换与代数化简，输出形式陈述所保证的严格等式或不变性结论。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {x : ℝ}, -1 ≤ x → x ≤ 1 → CGB.Generation.hardSat x = x
```

## CGB.Generation.hardSat_eq_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactEligibility.lean#L31)

解释：输入hardSat_eq_one形式化前提参量，经底层算子变换与代数化简，输出形式陈述所保证的严格等式或不变性结论。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {x : ℝ}, 1 ≤ x → CGB.Generation.hardSat x = 1
```

## CGB.Generation.contact_response_unsaturated_increment

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactEligibility.lean#L38)

解释：On one unsaturated response segment, participation is the exact local forward finite-difference factor of a contact-weight perturbation.

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

解释：A bounded activity face carries no additional local contact evidence: as long as both responses remain saturated at `+1`, their difference is zero.

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

解释：A score whose local read coefficient is `q` inherits both cases. This is the one-contact part of the next-byte forward eligibility before its language innovation is used as the third factor.

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
