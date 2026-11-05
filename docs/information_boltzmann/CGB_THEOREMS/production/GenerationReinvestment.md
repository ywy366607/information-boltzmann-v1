# GenerationReinvestment

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean) · [返回导航](../README.md)

模块源码说明：Reinvestment as one balanced exchange (language organism, 2026-11-03) User decisions of 2026-11-03: occupancy is reinvested multiplicatively, and metabolism is gross balanced exchange.

## CGB.Reinvestment.reinvest_total

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L45)

分类：design；适用范围：candidate；审核：draft；候选：Reinvestment

分类说明：把乘性再投资写成份额上的 gross 等量交换的候选机制（用户 11-03 决定）；固定下注，未含输运，未落到任何身体。

解释：Formal contract for reinvest_total; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (a w : A → ℝ),
  CGB.Reinvestment.predicted a w ≠ 0 → ∑ i, CGB.Reinvestment.reinvest a w i = 1
```

## CGB.Reinvestment.gains_eq_losses

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L51)

分类：design；适用范围：candidate；审核：draft；候选：Reinvestment

分类说明：把乘性再投资写成份额上的 gross 等量交换的候选机制（用户 11-03 决定）；固定下注，未含输运，未落到任何身体。

解释：Two distributions of the same total: the summed gains equal the summed losses.

源码说明：Two distributions of the same total: the summed gains equal the summed losses.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (a b : A → ℝ), ∑ i, a i = ∑ i, b i → ∑ i, max (b i - a i) 0 = ∑ i, max (a i - b i) 0
```

## CGB.Reinvestment.gross_balanced

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L64)

分类：design；适用范围：candidate；审核：draft；候选：Reinvestment

分类说明：把乘性再投资写成份额上的 gross 等量交换的候选机制（用户 11-03 决定）；固定下注，未含输运，未落到任何身体。

解释：**Gross balanced exchange**: reinvestment imports into the winners exactly what it exports from the losers.

源码说明：**Gross balanced exchange**: reinvestment imports into the winners exactly what it exports from the losers.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (a w : A → ℝ),
  ∑ i, a i = 1 →
    CGB.Reinvestment.predicted a w ≠ 0 →
      ∑ i, max (CGB.Reinvestment.reinvest a w i - a i) 0 = ∑ i, max (a i - CGB.Reinvestment.reinvest a w i) 0
```

## CGB.Reinvestment.grows_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L69)

分类：design；适用范围：candidate；审核：draft；候选：Reinvestment

分类说明：把乘性再投资写成份额上的 gross 等量交换的候选机制（用户 11-03 决定）；固定下注，未含输运，未落到任何身体。

解释：An address with a positive share grows exactly when its bet beats the prediction.

源码说明：An address with a positive share grows exactly when its bet beats the prediction.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (a w : A → ℝ) (i : A),
  0 < a i →
    0 < CGB.Reinvestment.predicted a w → (a i < CGB.Reinvestment.reinvest a w i ↔ CGB.Reinvestment.predicted a w < w i)
```

## CGB.Reinvestment.no_exchange_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L77)

分类：design；适用范围：candidate；审核：draft；候选：Reinvestment

分类说明：把乘性再投资写成份额上的 gross 等量交换的候选机制（用户 11-03 决定）；固定下注，未含输运，未落到任何身体。

解释：**Exact prediction exchanges nothing**, and only then: the shares stay put exactly when every live address bet the prediction.

源码说明：**Exact prediction exchanges nothing**, and only then: the shares stay put exactly when every live address bet the prediction.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (a w : A → ℝ),
  CGB.Reinvestment.predicted a w ≠ 0 →
    ((∀ (i : A), CGB.Reinvestment.reinvest a w i = a i) ↔ ∀ (i : A), a i ≠ 0 → w i = CGB.Reinvestment.predicted a w)
```

## CGB.Reinvestment.mean_multiplier_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L93)

分类：design；适用范围：candidate；审核：draft；候选：Reinvestment

分类说明：把乘性再投资写成份额上的 gross 等量交换的候选机制（用户 11-03 决定）；固定下注，未含输运，未落到任何身体。

解释：**Adaptation of the total**: the share-weighted multiplier is one for every byte.

源码说明：**Adaptation of the total**: the share-weighted multiplier is one for every byte.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (a w : A → ℝ),
  CGB.Reinvestment.predicted a w ≠ 0 → ∑ i, a i * (w i / CGB.Reinvestment.predicted a w) = 1
```

## CGB.Reinvestment.reinvest_scale

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L100)

分类：design；适用范围：candidate；审核：draft；候选：Reinvestment

分类说明：把乘性再投资写成份额上的 gross 等量交换的候选机制（用户 11-03 决定）；固定下注，未含输运，未落到任何身体。

解释：**Adaptation of the level**: multiplying every bet by the same positive number changes nothing. Only the bet relative to the prediction acts.

源码说明：**Adaptation of the level**: multiplying every bet by the same positive number changes nothing. Only the bet relative to the prediction acts.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (a w : A → ℝ) (c : ℝ),
  c ≠ 0 → ∀ (i : A), CGB.Reinvestment.reinvest a (fun j => c * w j) i = CGB.Reinvestment.reinvest a w i
```

## CGB.Reinvestment.predicted_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L107)

分类：design；适用范围：candidate；审核：draft；候选：Reinvestment

分类说明：把乘性再投资写成份额上的 gross 等量交换的候选机制（用户 11-03 决定）；固定下注，未含输运，未落到任何身体。

解释：Formal contract for predicted_pos; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] [Nonempty A] (a w : A → ℝ),
  (∀ (i : A), 0 < a i) → (∀ (i : A), 0 < w i) → 0 < CGB.Reinvestment.predicted a w
```

## CGB.Reinvestment.reinvest_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L111)

分类：design；适用范围：candidate；审核：draft；候选：Reinvestment

分类说明：把乘性再投资写成份额上的 gross 等量交换的候选机制（用户 11-03 决定）；固定下注，未含输运，未落到任何身体。

解释：Formal contract for reinvest_pos; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] [Nonempty A] (a w : A → ℝ),
  (∀ (i : A), 0 < a i) → (∀ (i : A), 0 < w i) → ∀ (i : A), 0 < CGB.Reinvestment.reinvest a w i
```

## CGB.Reinvestment.wealth_eq_mixture

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L124)

分类：design；适用范围：candidate；审核：draft；候选：Reinvestment

分类说明：把乘性再投资写成份额上的 gross 等量交换的候选机制（用户 11-03 决定）；固定下注，未含输运，未落到任何身体。

解释：**Kelly equals Bayes**: the product of the predictions equals the mixture of the addresses' own products of bets.

源码说明：**Kelly equals Bayes**: the product of the predictions equals the mixture of the addresses' own products of bets.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] {B : Type u_2} [Nonempty A] (w : B → A → ℝ),
  (∀ (y : B) (i : A), 0 < w y i) →
    ∀ (ys : List B) (a : A → ℝ),
      (∀ (i : A), 0 < a i) →
        ∑ i, a i = 1 → CGB.Reinvestment.wealth w a ys = ∑ i, a i * (List.map (fun y => w y i) ys).prod
```

## CGB.Reinvestment.wealth_ge_address

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L138)

分类：design；适用范围：candidate；审核：draft；候选：Reinvestment

分类说明：把乘性再投资写成份额上的 gross 等量交换的候选机制（用户 11-03 决定）；固定下注，未含输运，未落到任何身体。

解释：The wealth is at least what one address would have made from its first share.

源码说明：The wealth is at least what one address would have made from its first share.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] {B : Type u_2} [Nonempty A] (w : B → A → ℝ),
  (∀ (y : B) (i : A), 0 < w y i) →
    ∀ (ys : List B) (a : A → ℝ),
      (∀ (i : A), 0 < a i) →
        ∑ i, a i = 1 → ∀ (i : A), a i * (List.map (fun y => w y i) ys).prod ≤ CGB.Reinvestment.wealth w a ys
```

## CGB.Reinvestment.predicted_update

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L148)

分类：design；适用范围：candidate；审核：draft；候选：Reinvestment

分类说明：把乘性再投资写成份额上的 gross 等量交换的候选机制（用户 11-03 决定）；固定下注，未含输运，未落到任何身体。

解释：Changing one bet changes the prediction by the share times the change.

源码说明：Changing one bet changes the prediction by the share times the change.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] [inst_1 : DecidableEq A] (a w : A → ℝ) (i : A) (θ : ℝ),
  CGB.Reinvestment.predicted a (Function.update w i (w i + θ)) = CGB.Reinvestment.predicted a w + a i * θ
```

## CGB.Reinvestment.log_predicted_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L161)

分类：design；适用范围：candidate；审核：draft；候选：Reinvestment

分类说明：把乘性再投资写成份额上的 gross 等量交换的候选机制（用户 11-03 决定）；固定下注，未含输运，未落到任何身体。

解释：**Credit from the same event**: the derivative of the arrived byte's log prediction with respect to one bet is the share over the prediction.

源码说明：**Credit from the same event**: the derivative of the arrived byte's log prediction with respect to one bet is the share over the prediction.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] [inst_1 : DecidableEq A] (a w : A → ℝ) (i : A),
  CGB.Reinvestment.predicted a w ≠ 0 →
    HasDerivAt (fun θ => Real.log (CGB.Reinvestment.predicted a (Function.update w i (w i + θ))))
      (a i / CGB.Reinvestment.predicted a w) 0
```

## CGB.Reinvestment.credit_eq_new_share

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L173)

分类：design；适用范围：candidate；审核：draft；候选：Reinvestment

分类说明：把乘性再投资写成份额上的 gross 等量交换的候选机制（用户 11-03 决定）；固定下注，未含输运，未落到任何身体。

解释：The same credit written with the new share: the update needs only the share after the arrival and the bet itself.

源码说明：The same credit written with the new share: the update needs only the share after the arrival and the bet itself.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (a w : A → ℝ) (i : A),
  w i ≠ 0 → a i / CGB.Reinvestment.predicted a w = CGB.Reinvestment.reinvest a w i / w i
```
