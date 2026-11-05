# GenerationCountedPopulation

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean) · [返回导航](../README.md)

模块源码说明：Reinvestment by counted units (language organism, 2026-11-03) `GenerationReinvestment` treats the shares as real numbers.

## CGB.CountedPopulation.moran_expected_change

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L51)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：One birth in proportion to reinvestment and one recycling in proportion to share: the mean change of the share of `i` is the reinvestment step.

源码说明：One birth in proportion to reinvestment and one recycling in proportion to share: the mean change of the share of `i` is the reinvestment step.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] [inst_1 : DecidableEq A] (a w : A → ℝ),
  ∑ i, a i = 1 →
    CGB.Reinvestment.predicted a w ≠ 0 →
      ∀ (i : A),
        ∑ j, ∑ k, CGB.Reinvestment.reinvest a w j * a k * ((if j = i then 1 else 0) - if k = i then 1 else 0) =
          CGB.Reinvestment.reinvest a w i - a i
```

## CGB.CountedPopulation.exchange_moves_a_unit

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L66)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：Counts of a fixed total change only by moving at least one whole unit.

源码说明：Counts of a fixed total change only by moving at least one whole unit.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (n n' : A → ℕ), ∑ i, n i = ∑ i, n' i → n ≠ n' → 1 ≤ ∑ i, (n' i - n i)
```

## CGB.CountedPopulation.extinct_stays_extinct

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L81)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：An address at zero stays at zero. New addresses cannot come from reinvestment.

源码说明：An address at zero stays at zero. New addresses cannot come from reinvestment.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (a w : A → ℝ) (i : A), a i = 0 → CGB.Reinvestment.reinvest a w i = 0
```

## CGB.CountedPopulation.concentrated_absorbing

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L85)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：A population on one address never leaves it.

源码说明：A population on one address never leaves it.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] [inst_1 : DecidableEq A] (w : A → ℝ) (c : A),
  w c ≠ 0 → ∀ (i : A), CGB.Reinvestment.reinvest (Pi.single c 1) w i = Pi.single c 1 i
```

## CGB.CountedPopulation.reinvest_ratio

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L102)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：The ratio of two shares moves by the ratio of the bets.

源码说明：The ratio of two shares moves by the ratio of the bets.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (a w : A → ℝ) (i j : A),
  CGB.Reinvestment.predicted a w ≠ 0 →
    w j ≠ 0 → CGB.Reinvestment.reinvest a w i / CGB.Reinvestment.reinvest a w j = a i / a j * (w i / w j)
```

## CGB.CountedPopulation.ratio_after_le

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L118)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：**Lock-in**: when each byte pays `i` at most `ρ` times what it pays `j`, the ratio of `i` to `j` falls by at least `ρ` per byte, however good `i` was before.

源码说明：**Lock-in**: when each byte pays `i` at most `ρ` times what it pays `j`, the ratio of `i` to `j` falls by at least `ρ` per byte, however good `i` was before.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] {B : Type u_2} [Nonempty A] (w : B → A → ℝ),
  (∀ (y : B) (i : A), 0 < w y i) →
    ∀ (i j : A) (ρ : ℝ),
      (∀ (y : B), w y i ≤ ρ * w y j) →
        ∀ (ys : List B) (a : A → ℝ),
          (∀ (k : A), 0 < a k) →
            CGB.CountedPopulation.after w a ys i / CGB.CountedPopulation.after w a ys j ≤ a i / a j * ρ ^ ys.length
```

## CGB.CountedPopulation.mix_total

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L143)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：Formal contract for mix_total; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] [Nonempty A] (ε : ℝ) (a : A → ℝ),
  ∑ i, a i = 1 → ∑ i, CGB.CountedPopulation.mix ε a i = 1
```

## CGB.CountedPopulation.mix_floor

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L153)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：Every address keeps a floor after one transport.

源码说明：Every address keeps a floor after one transport.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A],
  ∀ ε ≤ 1, ∀ (a : A → ℝ), (∀ (i : A), 0 ≤ a i) → ∀ (i : A), ε / ↑(Fintype.card A) ≤ CGB.CountedPopulation.mix ε a i
```

## CGB.CountedPopulation.mix_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L159)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：Formal contract for mix_pos; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] [Nonempty A] (ε : ℝ),
  0 < ε → ε ≤ 1 → ∀ (a : A → ℝ), (∀ (i : A), 0 ≤ a i) → ∀ (i : A), 0 < CGB.CountedPopulation.mix ε a i
```

## CGB.CountedPopulation.mixWealth_ge_address

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L170)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：Wealth with transport tracks any one address, paying `1 - ε` per byte.

源码说明：Wealth with transport tracks any one address, paying `1 - ε` per byte.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] {B : Type u_2} [Nonempty A] (ε : ℝ),
  0 < ε →
    ε ≤ 1 →
      ∀ (w : B → A → ℝ),
        (∀ (y : B) (i : A), 0 < w y i) →
          ∀ (ys : List B) (a : A → ℝ),
            (∀ (i : A), 0 < a i) →
              ∑ i, a i = 1 →
                ∀ (i : A),
                  a i * (1 - ε) ^ ys.length * (List.map (fun y => w y i) ys).prod ≤
                    CGB.CountedPopulation.mixWealth ε w a ys
```

## CGB.CountedPopulation.recovers_from_floor

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L209)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：**Recovery after any history**: from any state with the floor, the wealth over a later stretch is at least the floor times what the address earns there, minus the transport toll. The cost does not depend on how long or how badly the address lost before.

源码说明：**Recovery after any history**: from any state with the floor, the wealth over a later stretch is at least the floor times what the address earns there, minus the transport toll. The cost does not depend on how long or how badly the address lost before.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] {B : Type u_2} [Nonempty A] (ε : ℝ),
  0 < ε →
    ε ≤ 1 →
      ∀ (w : B → A → ℝ),
        (∀ (y : B) (i : A), 0 < w y i) →
          ∀ (ys : List B) (a : A → ℝ),
            (∀ (i : A), 0 ≤ a i) →
              ∑ i, a i = 1 →
                ∀ (i : A),
                  ε / ↑(Fintype.card A) * (1 - ε) ^ ys.length * (List.map (fun y => w y i) ys).prod ≤
                    CGB.CountedPopulation.mixWealth ε w (CGB.CountedPopulation.mix ε a) ys
```

## CGB.CountedPopulation.price_equation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L224)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：**Price equation**: the mean of any carried trait changes by the covariance of the trait with the multiplier minus one.

源码说明：**Price equation**: the mean of any carried trait changes by the covariance of the trait with the multiplier minus one.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (a w z : A → ℝ),
  CGB.Reinvestment.predicted a w ≠ 0 →
    ∑ i, CGB.Reinvestment.reinvest a w i * z i - ∑ i, a i * z i =
      ∑ i, a i * (w i / CGB.Reinvestment.predicted a w - 1) * z i
```

## CGB.CountedPopulation.groupGain_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L248)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：Formal contract for groupGain_sum; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] {C : Type u_3} [inst_1 : Fintype C] [inst_2 : DecidableEq C] (g : A → C)
  (a w : A → ℝ), ∑ c, CGB.CountedPopulation.groupGain g a w c = CGB.Reinvestment.predicted a w
```

## CGB.CountedPopulation.coarse_predicted

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L254)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：Groups predict as addresses do.

源码说明：Groups predict as addresses do.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] {C : Type u_3} [inst_1 : Fintype C] [inst_2 : DecidableEq C] (g : A → C)
  (a w : A → ℝ),
  (∀ (c : C), CGB.CountedPopulation.groupShare g a c ≠ 0) →
    CGB.Reinvestment.predicted (CGB.CountedPopulation.groupShare g a) (CGB.CountedPopulation.groupBet g a w) =
      CGB.Reinvestment.predicted a w
```

## CGB.CountedPopulation.coarse_reinvest

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L262)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：**The same law at the next scale**: reinvestment of the group total is reinvestment of the group as one address.

源码说明：**The same law at the next scale**: reinvestment of the group total is reinvestment of the group as one address.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] {C : Type u_3} [inst_1 : Fintype C] [inst_2 : DecidableEq C] (g : A → C)
  (a w : A → ℝ),
  (∀ (c : C), CGB.CountedPopulation.groupShare g a c ≠ 0) →
    ∀ (c : C),
      CGB.Reinvestment.reinvest (CGB.CountedPopulation.groupShare g a) (CGB.CountedPopulation.groupBet g a w) c =
        ∑ i with g i = c, CGB.Reinvestment.reinvest a w i
```

## CGB.CountedPopulation.local_keeps_group_total

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L281)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：**A compartment keeps its total**: the sum of a group after a local exchange is its sum before, whatever the bets are.

源码说明：**A compartment keeps its total**: the sum of a group after a local exchange is its sum before, whatever the bets are.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] {C : Type u_3} [inst_1 : DecidableEq C] (g : A → C) (a w : A → ℝ) (c : C),
  CGB.CountedPopulation.groupGain g a w c ≠ 0 →
    ∑ i with g i = c, CGB.CountedPopulation.localReinvest g a w i = CGB.CountedPopulation.groupShare g a c
```

## CGB.CountedPopulation.local_ignores_outside

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L297)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：**A compartment reads only its own bets**: changing bets outside the group changes nothing inside it.

源码说明：**A compartment reads only its own bets**: changing bets outside the group changes nothing inside it.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] {C : Type u_3} [inst_1 : DecidableEq C] (g : A → C) (a w w' : A → ℝ) (i : A),
  (∀ (j : A), g j = g i → w j = w' j) →
    CGB.CountedPopulation.localReinvest g a w i = CGB.CountedPopulation.localReinvest g a w' i
```

## CGB.CountedPopulation.step_smul

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L320)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：Formal contract for step_smul; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (T : A → A → ℝ) (w u : A → ℝ) (c : ℝ) (j : A),
  CGB.CountedPopulation.step T w (fun i => c * u i) j = c * CGB.CountedPopulation.step T w u j
```

## CGB.CountedPopulation.lik_smul

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L326)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：Formal contract for lik_smul; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] {B : Type u_2} (T : A → A → ℝ) (w : B → A → ℝ) (ys : List B) (c : ℝ) (u : A → ℝ),
  CGB.CountedPopulation.lik T w (fun i => c * u i) ys = c * CGB.CountedPopulation.lik T w u ys
```

## CGB.CountedPopulation.fwd_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L346)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：Formal contract for fwd_nonneg; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] [Nonempty A] (T : A → A → ℝ),
  (∀ (i j : A), 0 ≤ T i j) →
    ∀ (w a : A → ℝ),
      (∀ (i : A), 0 < w i) →
        (∀ (i : A), 0 ≤ a i) → 0 < CGB.Reinvestment.predicted a w → ∀ (j : A), 0 ≤ CGB.CountedPopulation.fwd T w a a j
```

## CGB.CountedPopulation.fwd_total

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L353)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：Formal contract for fwd_total; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (T : A → A → ℝ),
  (∀ (i : A), ∑ j, T i j = 1) →
    ∀ (w a : A → ℝ), CGB.Reinvestment.predicted a w ≠ 0 → ∑ j, CGB.CountedPopulation.fwd T w a a j = 1
```

## CGB.CountedPopulation.predicted_pos_of_total

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L364)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：Formal contract for predicted_pos_of_total; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] [Nonempty A] (a w : A → ℝ),
  (∀ (i : A), 0 < w i) → (∀ (i : A), 0 ≤ a i) → ∑ i, a i = 1 → 0 < CGB.Reinvestment.predicted a w
```

## CGB.CountedPopulation.tag_fraction_eq_posterior

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L379)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：**Delayed credit without a backward pass**: the tagged fraction of the mass that survives a stream equals the likelihood of that stream through the tag over its likelihood overall, which is the posterior of the tag given every later byte. The mass itself is the history carrier.

源码说明：**Delayed credit without a backward pass**: the tagged fraction of the mass that survives a stream equals the likelihood of that stream through the tag over its likelihood overall, which is the posterior of the tag given every later byte. The mass itself is the history carrier.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] {B : Type u_2} [Nonempty A] (T : A → A → ℝ),
  (∀ (i j : A), 0 ≤ T i j) →
    (∀ (i : A), ∑ j, T i j = 1) →
      ∀ (w : B → A → ℝ),
        (∀ (y : B) (i : A), 0 < w y i) →
          ∀ (ys : List B) (a b : A → ℝ),
            (∀ (i : A), 0 ≤ a i) →
              ∑ i, a i = 1 →
                CGB.CountedPopulation.tagFraction T w a b ys =
                  CGB.CountedPopulation.lik T w b ys / CGB.CountedPopulation.lik T w a ys
```

## CGB.CountedPopulation.step_add

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L394)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：Formal contract for step_add; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (T : A → A → ℝ) (w u v : A → ℝ) (j : A),
  CGB.CountedPopulation.step T w (fun i => u i + v i) j =
    CGB.CountedPopulation.step T w u j + CGB.CountedPopulation.step T w v j
```

## CGB.CountedPopulation.lik_add

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L400)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：Formal contract for lik_add; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] {B : Type u_2} (T : A → A → ℝ) (w : B → A → ℝ) (ys : List B) (u v : A → ℝ),
  CGB.CountedPopulation.lik T w (fun i => u i + v i) ys =
    CGB.CountedPopulation.lik T w u ys + CGB.CountedPopulation.lik T w v ys
```

## CGB.CountedPopulation.lik_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L411)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：Formal contract for lik_pos; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] {B : Type u_2} [Nonempty A] (T : A → A → ℝ),
  (∀ (i j : A), 0 ≤ T i j) →
    (∀ (i : A), ∑ j, T i j = 1) →
      ∀ (w : B → A → ℝ),
        (∀ (y : B) (i : A), 0 < w y i) →
          ∀ (ys : List B) (a : A → ℝ), (∀ (i : A), 0 ≤ a i) → ∑ i, a i = 1 → 0 < CGB.CountedPopulation.lik T w a ys
```

## CGB.CountedPopulation.tag_is_exact_credit

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L428)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：**The tag is the exact credit**: the derivative of the log likelihood of every later byte with respect to mass added at an earlier event equals the tag's fraction of the surviving mass. This is the derivative of the future with respect to a past contact, for fixed bets. The price is one more field per tag: this is forward sensitivity, and it costs what forward sensitivity costs.

源码说明：**The tag is the exact credit**: the derivative of the log likelihood of every later byte with respect to mass added at an earlier event equals the tag's fraction of the surviving mass. This is the derivative of the future with respect to a past contact, for fixed bets. The price is one more field per tag: this is forward sensitivity, and it costs what forward sensitivity costs.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] {B : Type u_2} [Nonempty A] (T : A → A → ℝ),
  (∀ (i j : A), 0 ≤ T i j) →
    (∀ (i : A), ∑ j, T i j = 1) →
      ∀ (w : B → A → ℝ),
        (∀ (y : B) (i : A), 0 < w y i) →
          ∀ (ys : List B) (a b : A → ℝ),
            (∀ (i : A), 0 ≤ a i) →
              ∑ i, a i = 1 →
                HasDerivAt (fun θ => Real.log (CGB.CountedPopulation.lik T w (fun i => a i + θ * b i) ys))
                  (CGB.CountedPopulation.tagFraction T w a b ys) 0
```

## CGB.CountedPopulation.tagReinvest_total_bracket

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L456)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：Formal contract for tagReinvest_total_bracket; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (a w b : A → ℝ) (lo hi : ℝ),
  CGB.CountedPopulation.Bracketed a b lo hi →
    (∀ (i : A), 0 < w i) →
      0 < CGB.Reinvestment.predicted a w →
        lo ≤ ∑ i, CGB.CountedPopulation.tagReinvest a w b i ∧ ∑ i, CGB.CountedPopulation.tagReinvest a w b i ≤ hi
```

## CGB.CountedPopulation.bracket_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L479)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：**One step of forgetting**: uniform transport pulls the bracket of a tag toward its mean, by `ε / n` of the gap at every byte.

源码说明：**One step of forgetting**: uniform transport pulls the bracket of a tag toward its mean, by `ε / n` of the gap at every byte.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] [Nonempty A] (a w b : A → ℝ) (ε lo hi : ℝ),
  0 < ε →
    ε ≤ 1 →
      CGB.CountedPopulation.Bracketed a b lo hi →
        (∀ (i : A), 0 ≤ a i) →
          ∑ i, a i = 1 →
            (∀ (i : A), 0 < w i) →
              ∃ lo' hi',
                CGB.CountedPopulation.Bracketed (CGB.CountedPopulation.mix ε (CGB.Reinvestment.reinvest a w))
                    (CGB.CountedPopulation.tagMix ε (CGB.CountedPopulation.tagReinvest a w b)) lo' hi' ∧
                  hi' - lo' = (1 - ε / ↑(Fintype.card A)) * (hi - lo)
```

## CGB.CountedPopulation.bracket_after

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L541)

分类：design；适用范围：candidate；审核：draft；候选：CountedPopulation

分类说明：乘性再投资的计数形式、输运、携带标签的前向信用与信用地平线（11-03）；固定下注，未落到任何身体。

解释：**The credit horizon**: after `L` bytes the tag is the same fraction of the mass at every address, up to `(1 - ε / n) ^ L` of the original gap. Later bytes can only read a difference that is still there, so the transport that gives recovery also bounds how far back credit reaches.

源码说明：**The credit horizon**: after `L` bytes the tag is the same fraction of the mass at every address, up to `(1 - ε / n) ^ L` of the original gap. Later bytes can only read a difference that is still there, so the transport that gives recovery also bounds how far back credit reaches.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] {B : Type u_2} [Nonempty A] (ε : ℝ),
  0 < ε →
    ε ≤ 1 →
      ∀ (w : B → A → ℝ),
        (∀ (y : B) (i : A), 0 < w y i) →
          ∀ (ys : List B) (a b : A → ℝ) (lo hi : ℝ),
            CGB.CountedPopulation.Bracketed a b lo hi →
              (∀ (i : A), 0 ≤ a i) →
                ∑ i, a i = 1 →
                  ∃ lo' hi',
                    CGB.CountedPopulation.Bracketed (CGB.CountedPopulation.tagAfter ε w (a, b) ys).1
                        (CGB.CountedPopulation.tagAfter ε w (a, b) ys).2 lo' hi' ∧
                      hi' - lo' = (1 - ε / ↑(Fintype.card A)) ^ ys.length * (hi - lo)
```
