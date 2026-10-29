# GenerationPredictiveHarvest

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean) · [返回导航](../README.md)

模块源码说明：Prediction is metabolism under reinvestment (language-stream life, T1) World contract (declared, a niche like the E.

## CGB.PredictiveHarvest.growth_eq_log_card_sub_crossEntropy

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L41)

解释：当 p 总和为一且各分配 c 为正时，定义的增长量等于 log|B| 减交叉熵。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {B : Type u_1} [inst : Fintype B] [Nonempty B] (p c : B → ℝ),
  ∑ b, p b = 1 →
    (∀ (b : B), 0 < c b) →
      CGB.PredictiveHarvest.growth p c = Real.log ↑(Fintype.card B) - CGB.PredictiveHarvest.crossEntropy p c
```

## CGB.PredictiveHarvest.gibbs_term

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L53)

解释：若 x 非负且 y 为正，则 x(log y−log x) 不超过 y−x。

源码说明：One term of Gibbs' inequality: `x (log y - log x) ≤ y - x` (from `log t ≤ t - 1`).

形式陈述：

```lean
∀ (x y : ℝ), 0 ≤ x → 0 < y → x * (Real.log y - Real.log x) ≤ y - x
```

## CGB.PredictiveHarvest.gibbs_inequality

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L63)

解释：对非负归一 p 和严格正、归一的 c，p 加权的 log c 不超过 p 加权的 log p。

源码说明：**Gibbs' inequality**: `Σ p log c ≤ Σ p log p` for a positive allocation summing to one.

形式陈述：

```lean
∀ {B : Type u_1} [inst : Fintype B] (p c : B → ℝ),
  (∀ (b : B), 0 ≤ p b) →
    ∑ b, p b = 1 → (∀ (b : B), 0 < c b) → ∑ b, c b = 1 → ∑ b, p b * Real.log (c b) ≤ ∑ b, p b * Real.log (p b)
```

## CGB.PredictiveHarvest.growth_le

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L75)

解释：若 p 非负归一、c 严格正且归一，则指定增长量不超过 log|B| 减去 p 的熵。

源码说明：**Kelly bound**: no allocation grows faster than `log |B| - H(p)`.

形式陈述：

```lean
∀ {B : Type u_1} [inst : Fintype B] [Nonempty B] (p c : B → ℝ),
  (∀ (b : B), 0 ≤ p b) →
    ∑ b, p b = 1 →
      (∀ (b : B), 0 < c b) →
        ∑ b, c b = 1 → CGB.PredictiveHarvest.growth p c ≤ Real.log ↑(Fintype.card B) - CGB.PredictiveHarvest.entropy p
```

## CGB.PredictiveHarvest.growth_at_law

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L83)

解释：当 p 的总和为一时，定义的 growth p p 等于 log|B| 减去定义的熵。

源码说明：Placing capacity as the stream's law attains the bound.

形式陈述：

```lean
∀ {B : Type u_1} [inst : Fintype B] [Nonempty B] (p : B → ℝ),
  ∑ b, p b = 1 → CGB.PredictiveHarvest.growth p p = Real.log ↑(Fintype.card B) - CGB.PredictiveHarvest.entropy p
```

## CGB.PredictiveHarvest.uniform_growth_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L96)

解释：在非空有限符号集上，均匀分配代入所定义的 growth 时结果为零。

源码说明：A body that knows nothing about the stream (uniform allocation) does not grow.

形式陈述：

```lean
∀ {B : Type u_1} [inst : Fintype B] [Nonempty B] (p : B → ℝ),
  (CGB.PredictiveHarvest.growth p fun x => 1 / ↑(Fintype.card B)) = 0
```

## CGB.PredictiveHarvest.no_allocation_persists

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L106)

解释：对非负归一分布，若其熵超过 log|B|−m，则任意正且归一的分配都使指定对数增长量减 m 为负。

源码说明：**No body persists** on a stream whose entropy exceeds `log |B| - m`: every allocation's net growth (growth minus a maintenance cost `m` per arrival) is negative.

形式陈述：

```lean
∀ {B : Type u_1} [inst : Fintype B] [Nonempty B] (p : B → ℝ) (m : ℝ),
  (∀ (b : B), 0 ≤ p b) →
    ∑ b, p b = 1 →
      Real.log ↑(Fintype.card B) - m < CGB.PredictiveHarvest.entropy p →
        ∀ (c : B → ℝ), (∀ (b : B), 0 < c b) → ∑ b, c b = 1 → CGB.PredictiveHarvest.growth p c - m < 0
```

## CGB.PredictiveHarvest.persists_of_crossEntropy

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L113)

解释：在所列归一化与正分配条件下，若交叉熵低于 log|B|−m，则定义的对数增长量减 m 严格为正。

源码说明：A body whose cross-entropy is below `log |B| - m` has positive net growth.

形式陈述：

```lean
∀ {B : Type u_1} [inst : Fintype B] [Nonempty B] (p c : B → ℝ) (m : ℝ),
  ∑ b, p b = 1 →
    (∀ (b : B), 0 < c b) →
      CGB.PredictiveHarvest.crossEntropy p c < Real.log ↑(Fintype.card B) - m → 0 < CGB.PredictiveHarvest.growth p c - m
```

## CGB.PredictiveHarvest.additive_payoff_mode

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L122)

解释：若 bs 的权重不小于其他符号且分配非负归一，则加性加权收益不超过把全部分配放在 bs 的收益。

源码说明：Without reinvestment (payoff additive in the capacity placed on the arrived byte) the best allocation puts everything on a most likely byte: probability matching is a consequence of the multiplicative contract, not of prediction alone.

形式陈述：

```lean
∀ {B : Type u_1} [inst : Fintype B] [inst_1 : DecidableEq B] (p c : B → ℝ) (bs : B),
  (∀ (b : B), p b ≤ p bs) → (∀ (b : B), 0 ≤ c b) → ∑ b, c b = 1 → ∑ b, p b * c b ≤ ∑ b, p b * if b = bs then 1 else 0
```
