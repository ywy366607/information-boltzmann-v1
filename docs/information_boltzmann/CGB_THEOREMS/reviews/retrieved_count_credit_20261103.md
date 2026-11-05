# retrieved_count_credit_20261103

探索区：仅按源码盘点；不在生产区全量审计覆盖内，不能据此声称已通过当前构建或信任审计。

[源文件](../../../../lean/CGBAxioms/reviews/retrieved_count_credit_20261103.lean) · [返回导航](../README.md)

模块源码说明：A retrieved association receives later evidence through its actual mixture read.

## RetrievedCountCreditReview.normalized_read_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/retrieved_count_credit_20261103.lean#L20)

解释：Changing one participant's mixture weight changes the normalized read by its value relative to the other participants; the denominator must be nonzero.

源码说明：Changing one participant's mixture weight changes the normalized read by its value relative to the other participants; the denominator must be nonzero.

形式陈述：

```lean
theorem normalized_read_deriv (a b v c : ℝ) (h : a + c ≠ 0) :
    HasDerivAt (fun x => (b + x * v) / (a + x))
      ((v * a - b) / (a + c)^2) c
```

## RetrievedCountCreditReview.count_trust_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/retrieved_count_credit_20261103.lean#L28)

解释：The existing count-to-trust map has this local derivative. It is not a derivative of the old sequence of writes that produced the count.

源码说明：The existing count-to-trust map has this local derivative. It is not a derivative of the old sequence of writes that produced the count.

形式陈述：

```lean
theorem count_trust_deriv (c : ℝ) (h : c + 1 ≠ 0) :
    HasDerivAt (fun x : ℝ => x / (x + 1)) (1 / (c + 1)^2) c
```

## RetrievedCountCreditReview.partial_exchange_total

[源码位置](../../../../lean/CGBAxioms/reviews/retrieved_count_credit_20261103.lean#L38)

解释：Any common fractional move toward normalized reinvestment preserves the eligible pool's total. Shortening the common move at a numerical ceiling also preserves this identity; ordinary boundary writes are outside this statement. `GenerationReinvestment.reinvest` and `predicted` are unfolded in the statement to avoid importing unrelated logarithmic-wealth proofs into this runtime check.

源码说明：Any common fractional move toward normalized reinvestment preserves the eligible pool's total. Shortening the common move at a numerical ceiling also preserves this identity; ordinary boundary writes are outside this statement. `GenerationReinvestment.reinvest` and `predicted` are unfolded in the statement to avoid importing unrelated logarithmic-wealth proofs into this runtime check.

形式陈述：

```lean
theorem partial_exchange_total {A : Type*} [Fintype A]
    (c w : A → ℝ) (r : ℝ) (h : (∑ i, c i * w i) ≠ 0) :
    (∑ i, ((1-r)*c i + r*(∑ j, c j)*(c i*w i/(∑ j, c j*w j)))) =
      ∑ i, c i
```
