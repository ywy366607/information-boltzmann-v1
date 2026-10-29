# GenerationLearning

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean) · [返回导航](../README.md)

模块源码说明：When does a world reward learning? (generation chain, L-1 to L-3) C-1 (`GenerationLeaveRule`) says when a forager must compare now with the past within one life.

## CGB.Generation.innate_optimal_of_constant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L27)

解释：在指定觅食收益模型中，固定环境均值已知时，以该均值为阈值的固定规则达到每个斑块的最优收益。

源码说明：**L-1**: a constant average across generations makes the innate threshold optimal.

形式陈述：

```lean
∀ (c q : ℝ), CGB.Generation.leaveValue c c q = max q c
```

## CGB.Generation.learner_loses_of_constant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L31)

解释：在指定恒定环境模型中，已知均值的固定规则严格优于额外支付正学习成本的同一最优收益。

源码说明：**L-1**: in a constant world a learner that pays `κ > 0` loses to the instinct at every patch.

形式陈述：

```lean
∀ {κ : ℝ}, 0 < κ → ∀ (c q : ℝ), max q c - κ < CGB.Generation.leaveValue c c q
```

## CGB.Generation.innate_loss_quarter

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L37)

解释：对于两个不同环境均值，任意固定阈值都在其中某个世界的某个斑块损失至少两均值差的四分之一。

源码说明：**L-2**: with two different averages every innate threshold loses at least a quarter of their gap at some patch in one of the two worlds.

形式陈述：

```lean
∀ {c₁ c₂ : ℝ},
  c₁ < c₂ →
    ∀ (τ : ℝ),
      ∃ q,
        CGB.Generation.leaveValue τ c₁ q ≤ max q c₁ - (c₂ - c₁) / 4 ∨
          CGB.Generation.leaveValue τ c₂ q ≤ max q c₂ - (c₂ - c₁) / 4
```

## CGB.Generation.learner_beats_innate

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L53)

解释：在两个不同均值的指定觅食模型中，若学习成本小于均值差的四分之一，则每个固定阈值都在某处严格劣于付费学习者。

源码说明：**L-2**: a learner (who uses the world's own average after paying `κ`) beats every instinct somewhere when its cost is below a quarter of the gap.

形式陈述：

```lean
∀ {c₁ c₂ κ : ℝ},
  c₁ < c₂ →
    κ < (c₂ - c₁) / 4 →
      ∀ (τ : ℝ), ∃ q, CGB.Generation.leaveValue τ c₁ q < max q c₁ - κ ∨ CGB.Generation.leaveValue τ c₂ q < max q c₂ - κ
```

## CGB.Generation.history_useless_of_independent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L61)

解释：未来环境与历史独立且每种固定规则的期望收益均有同一上界时，按历史选择规则的期望收益也不超过该界。

源码说明：**L-3**: if the next world state `c` (law `p`) is independent of the history `e` (law `r`) from which the rule `σ e` is chosen, the expected value is at most that of the best fixed rule.

形式陈述：

```lean
∀ {E : Type u_1} {Cw : Type u_2} {R : Type u_3} [inst : Fintype E] [inst_1 : Fintype Cw] [Nonempty R] (r : E → ℝ)
  (p : Cw → ℝ),
  (∀ (e : E), 0 ≤ r e) →
    ∑ e, r e = 1 →
      (∀ (c : Cw), 0 ≤ p c) →
        ∀ (v : R → Cw → ℝ) (σ : E → R) {M : ℝ}, (∀ (ρ : R), ∑ c, p c * v ρ c ≤ M) → ∑ e, ∑ c, r e * p c * v (σ e) c ≤ M
```

## CGB.Generation.settled_fixed_optimal

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L84)

解释：若环境均值从某时刻起恒定，则使用该均值的固定觅食阈值在之后每个时刻都达到模型最优。

源码说明：**O-1**: if the world settles (its average constant from `T` on), the fixed rule `τ = c T` is optimal at every later time and patch: nothing is left to compare or learn.

形式陈述：

```lean
∀ (c : ℕ → ℝ) (T : ℕ), (∀ t ≥ T, c t = c T) → ∀ t ≥ T, ∀ (q : ℝ), CGB.Generation.leaveValue (c T) (c t) q = max q (c t)
```

## CGB.Generation.beaten_forever_iff_never_settles

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L92)

解释：在指定阈值觅食模型中，任意未来时期每个固定规则仍会在某处落后，等价于环境均值从未最终恒定。

源码说明：**O-2**: every fixed rule is beaten after every time if and only if the world never settles (after every time it still takes two different averages).

形式陈述：

```lean
∀ (c : ℕ → ℝ),
  (∀ (T : ℕ) (τ : ℝ), ∃ t ≥ T, ∃ q, CGB.Generation.leaveValue τ (c t) q < max q (c t)) ↔
    ∀ (T : ℕ), ∃ t₁ ≥ T, ∃ t₂ ≥ T, c t₁ ≠ c t₂
```
