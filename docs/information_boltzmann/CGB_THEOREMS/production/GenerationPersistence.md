# GenerationPersistence

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean) · [返回导航](../README.md)

模块源码说明：Activity without input: driven circulation, adaptation, latches (generation chain, 2026-10-27) Three different things keep a body active when the input says nothing, and they must not be confused (eco-sphere audit: 154 o

## CGB.Generation.cycle_rates_balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L46)

解释：正平衡权重满足逐边细致平衡时，任一有限循环的正向速率乘积等于反向乘积。

源码说明：**A (Kolmogorov, one direction)**: detailed balance with a positive weight balances the rate products around every cycle `c 0 → c 1 → … → c (n-1) → c 0`.

形式陈述：

```lean
∀ {X : Type u_1} {n : ℕ} (K : X → X → ℝ) (π : X → ℝ),
  (∀ (x : X), 0 < π x) →
    (∀ (x y : X), π x * K x y = π y * K y x) →
      ∀ (c : Fin n → X), ∏ i, K (c i) (c ((finRotate n) i)) = ∏ i, K (c ((finRotate n) i)) (c i)
```

## CGB.Generation.driven_cycle_not_balanced

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L61)

解释：若某有限循环正反速率乘积不相等，则不存在满足所有边细致平衡的严格正权重。

源码说明：**A**: a cycle with unequal forward and reverse rate products (net circulation) admits no positive detailed-balance weight: self-repeating activity needs a drive.

形式陈述：

```lean
∀ {X : Type u_1} {n : ℕ} (K : X → X → ℝ) (c : Fin n → X),
  ∏ i, K (c i) (c ((finRotate n) i)) ≠ ∏ i, K (c ((finRotate n) i)) (c i) →
    ¬∃ π, (∀ (x : X), 0 < π x) ∧ ∀ (x y : X), π x * K x y = π y * K y x
```

## CGB.Generation.integral_feedback_adapts

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L73)

解释：非零增益的积分反馈达到固定点时，其反馈观测恰等于目标值。

源码说明：**B (integral feedback adapts perfectly)**: if the reference update `m ↦ m + γ (f (s - m) - a₀)` with `γ ≠ 0` has `m` as a fixed point, the activity there is `a₀`, whatever the input `s`.

形式陈述：

```lean
∀ (f : ℝ → ℝ) (γ a₀ s m : ℝ), γ ≠ 0 → m + γ * (f (s - m) - a₀) = m → f (s - m) = a₀
```

## CGB.Generation.adapted_gap_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L82)

解释：积分反馈达到固定点且增益非零时，输出偏离目标为零，故任何在零处取零的偏差函数也为零。

源码说明：**B**: an act driven only by the gap `g (a - a₀)` with `g 0 = 0` stops at every adapted fixed point of a constant input.

形式陈述：

```lean
∀ (f g : ℝ → ℝ), g 0 = 0 → ∀ (γ a₀ s m : ℝ), γ ≠ 0 → m + γ * (f (s - m) - a₀) = m → g (f (s - m) - a₀) = 0
```

## CGB.Generation.contraction_unique_rest

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L92)

解释：若实函数以严格小于一的统一常数收缩距离，则它的任意两个不动点必相同。

源码说明：**C**: a loop map with gain `L < 1` has at most one rest state: tonic activity without enough loop gain is monostable.

形式陈述：

```lean
∀ (f : ℝ → ℝ), ∀ L < 1, (∀ (x y : ℝ), |f x - f y| ≤ L * |x - y|) → ∀ (x y : ℝ), f x = x → f y = y → x = y
```

## CGB.Generation.latch_needs_unit_gain

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L102)

解释：可微实函数若有两个不同不动点，则两点之间至少有一点的导数等于一。

源码说明：**C**: two distinct rest states of a differentiable loop map force loop gain (slope) exactly one somewhere between them: a latch needs unit gain.

形式陈述：

```lean
∀ (f : ℝ → ℝ), Differentiable ℝ f → ∀ (x y : ℝ), x < y → f x = x → f y = y → ∃ z ∈ Set.Ioo x y, deriv f z = 1
```
