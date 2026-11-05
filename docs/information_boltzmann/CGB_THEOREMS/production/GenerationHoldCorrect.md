# GenerationHoldCorrect

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean) · [返回导航](../README.md)

模块源码说明：Holding versus correcting, and the latch transition (generation chain, 2026-11-04) The user separated two abilities of a memory: * **holding**: when nobody rewrites it, the content stays; * **attractor correction**: when

## CGB.Generation.HoldCorrect.hold_accumulates

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L74)

分类：semantics；适用范围：universal；审核：draft

分类说明：纯保持累积每次扰动：误差随扰动次数无界增长（保持≠纠错）。

解释：**Holding keeps every disturbance**: under the pure hold the state is the start plus the sum of all disturbances.

源码说明：**Holding keeps every disturbance**: under the pure hold the state is the start plus the sum of all disturbances.

形式陈述：

```lean
∀ (ε : ℕ → ℝ) (x₀ : ℝ) (n : ℕ), CGB.Generation.HoldCorrect.orbit id ε x₀ n = x₀ + ∑ t ∈ Finset.range n, ε t
```

## CGB.Generation.HoldCorrect.hold_error_unbounded

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L82)

分类：semantics；适用范围：universal；审核：draft

分类说明：无收缩的保持在持续小扰动下误差无界。

解释：**A held content drifts without bound** under any constant disturbance `e > 0`: for every bound `B` some time exceeds it.

源码说明：**A held content drifts without bound** under any constant disturbance `e > 0`: for every bound `B` some time exceeds it.

形式陈述：

```lean
∀ {e : ℝ}, 0 < e → ∀ (x₀ B : ℝ), ∃ n, B < |CGB.Generation.HoldCorrect.orbit id (fun x => e) x₀ n - x₀|
```

## CGB.Generation.HoldCorrect.contraction_noise_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L94)

分类：semantics；适用范围：universal；审核：draft

分类说明：收缩动力学把扰动误差限制在有界范围（吸引子式纠错）。

解释：**Contraction corrects**: if `f` pulls every state toward `r` by the factor `L < 1`, disturbances bounded by `e` leave an error of at most `L^n |x₀ - r| + e / (1 - L)` at every time, a bound that does not grow with `n`.

源码说明：**Contraction corrects**: if `f` pulls every state toward `r` by the factor `L < 1`, disturbances bounded by `e` leave an error of at most `L^n |x₀ - r| + e / (1 - L)` at every time, a bound that does not grow with `n`.

形式陈述：

```lean
∀ (f : ℝ → ℝ) {r L e : ℝ},
  0 ≤ L →
    L < 1 →
      (∀ (x : ℝ), |f x - r| ≤ L * |x - r|) →
        ∀ (ε : ℕ → ℝ),
          (∀ (t : ℕ), |ε t| ≤ e) →
            ∀ (x₀ : ℝ) (n : ℕ), |CGB.Generation.HoldCorrect.orbit f ε x₀ n - r| ≤ L ^ n * |x₀ - r| + e / (1 - L)
```

## CGB.Generation.HoldCorrect.contraction_merges

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L115)

分类：semantics；适用范围：universal；审核：draft

分类说明：收缩会把相近内容合并到同一吸引子，因此纠错只能保持离散状态。

解释：**Correction merges contents**: a map contracting all pairs by `L` brings any two contents within `L^n` of their initial gap.

源码说明：**Correction merges contents**: a map contracting all pairs by `L` brings any two contents within `L^n` of their initial gap.

形式陈述：

```lean
∀ (f : ℝ → ℝ) {L : ℝ},
  0 ≤ L → (∀ (x y : ℝ), |f x - f y| ≤ L * |x - y|) → ∀ (x y : ℝ) (n : ℕ), |f^[n] x - f^[n] y| ≤ L ^ n * |x - y|
```

## CGB.Generation.HoldCorrect.fb_odd

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L139)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：feedback_delta 标量图是奇函数（两个锁存态对称）。

解释：Two copies of the latch: the renewal is odd, so `-r*` mirrors `r*`.

源码说明：Two copies of the latch: the renewal is odd, so `-r*` mirrors `r*`.

形式陈述：

```lean
∀ (a c g r : ℝ), CGB.Generation.HoldCorrect.fb a c g (-r) = -CGB.Generation.HoldCorrect.fb a c g r
```

## CGB.Generation.HoldCorrect.fb_nonneg_eq

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L142)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：feedback_delta 标量图在非负半轴的显式形式。

解释：Formal contract for fb_nonneg_eq; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {a c g x : ℝ}, 0 < g → 0 ≤ x → CGB.Generation.HoldCorrect.fb a c g x = a * x + c * (g * x / (1 + g * x))
```

## CGB.Generation.HoldCorrect.fb_sub_fixed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L147)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：恒等式 fb(x) - r* = (x - r*)(a + (1-a)/(1+gx))。

解释：**The identity**: for `x ≥ 0`, `fb x - r* = (x - r*) · (a + (1-a)/(1+g x))`.

源码说明：**The identity**: for `x ≥ 0`, `fb x - r* = (x - r*) · (a + (1-a)/(1+g x))`.

形式陈述：

```lean
∀ {a c g x : ℝ},
  a < 1 →
    0 < g →
      0 ≤ x →
        CGB.Generation.HoldCorrect.fb a c g x - CGB.Generation.HoldCorrect.rstar a c g =
          (x - CGB.Generation.HoldCorrect.rstar a c g) * CGB.Generation.HoldCorrect.factor a g x
```

## CGB.Generation.HoldCorrect.factor_lt_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L155)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：超临界时收缩因子小于 1。

解释：Formal contract for factor_lt_one; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {a g x : ℝ}, a < 1 → 0 < g → 0 < x → CGB.Generation.HoldCorrect.factor a g x < 1
```

## CGB.Generation.HoldCorrect.factor_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L160)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：feedback_delta 标量图的收缩因子非负。

解释：Formal contract for factor_nonneg; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {a g x : ℝ}, 0 ≤ a → a < 1 → 0 < g → 0 ≤ x → 0 ≤ CGB.Generation.HoldCorrect.factor a g x
```

## CGB.Generation.HoldCorrect.factor_le_of_le

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L165)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：收缩因子随距离单调。

解释：Formal contract for factor_le_of_le; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {a g ρ x : ℝ},
  a < 1 → 0 < g → 0 ≤ ρ → ρ ≤ x → CGB.Generation.HoldCorrect.factor a g x ≤ CGB.Generation.HoldCorrect.factor a g ρ
```

## CGB.Generation.HoldCorrect.subcritical_forgets

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L172)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：kappa = cg/(1-a) <= 1 时唯一不动点为 0，内容被遗忘。

解释：**κ ≤ 1, forgetting**: every positive content shrinks by at least the factor `a + c g` per event (and stays positive): only 0 is left in the long run.

源码说明：**κ ≤ 1, forgetting**: every positive content shrinks by at least the factor `a + c g` per event (and stays positive): only 0 is left in the long run.

形式陈述：

```lean
∀ {a c g x : ℝ},
  0 ≤ a →
    0 < c →
      0 < g →
        c * g ≤ 1 - a →
          0 < x →
            0 < CGB.Generation.HoldCorrect.fb a c g x ∧
              CGB.Generation.HoldCorrect.fb a c g x ≤ (a + c * g) * x ∧ CGB.Generation.HoldCorrect.fb a c g x < x
```

## CGB.Generation.HoldCorrect.supercritical_fixed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L184)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：kappa > 1 时存在非零锁存不动点 r*。

解释：**κ > 1, the latch exists**: `r* > 0` and it is fixed.

源码说明：**κ > 1, the latch exists**: `r* > 0` and it is fixed.

形式陈述：

```lean
∀ {a c g : ℝ},
  a < 1 →
    0 < g →
      1 - a < c * g →
        0 < CGB.Generation.HoldCorrect.rstar a c g ∧
          CGB.Generation.HoldCorrect.fb a c g (CGB.Generation.HoldCorrect.rstar a c g) =
            CGB.Generation.HoldCorrect.rstar a c g
```

## CGB.Generation.HoldCorrect.supercritical_origin_repels

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L194)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：kappa > 1 时原点排斥，相变只由 kappa 决定。

解释：**κ > 1, the origin repels**: small positive content grows toward `r*`.

源码说明：**κ > 1, the origin repels**: small positive content grows toward `r*`.

形式陈述：

```lean
∀ {a c g x : ℝ},
  a < 1 → 0 < g → 0 < x → x < CGB.Generation.HoldCorrect.rstar a c g → x < CGB.Generation.HoldCorrect.fb a c g x
```

## CGB.Generation.HoldCorrect.latch_converges

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L202)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：超临界锁存从正半轴收敛到 r*。

解释：**κ > 1, convergence inside the basin**: from any `x₀ ≥ ρ`, `0 < ρ ≤ r*`, the undisturbed latch stays above `min x₀ r* ≥ ρ` and approaches `r*` at rate `factor ρ < 1`.

源码说明：**κ > 1, convergence inside the basin**: from any `x₀ ≥ ρ`, `0 < ρ ≤ r*`, the undisturbed latch stays above `min x₀ r* ≥ ρ` and approaches `r*` at rate `factor ρ < 1`.

形式陈述：

```lean
∀ {a c g ρ x₀ : ℝ},
  0 ≤ a →
    a < 1 →
      0 < g →
        0 < ρ →
          ρ ≤ CGB.Generation.HoldCorrect.rstar a c g →
            ρ ≤ x₀ →
              ∀ (n : ℕ),
                ρ ≤ (CGB.Generation.HoldCorrect.fb a c g)^[n] x₀ ∧
                  |(CGB.Generation.HoldCorrect.fb a c g)^[n] x₀ - CGB.Generation.HoldCorrect.rstar a c g| ≤
                    CGB.Generation.HoldCorrect.factor a g ρ ^ n * |x₀ - CGB.Generation.HoldCorrect.rstar a c g|
```

## CGB.Generation.HoldCorrect.latch_corrects

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L234)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：锁存把容限内的扰动拉回 r*（吸引子式纠错）。

解释：**κ > 1, attractor correction under disturbance**: if the state starts within `B` of `r*`, the basin margin covers it (`B ≤ r* - ρ`, `ρ > 0`), and every disturbance is at most `(1 - factor ρ) · B`, the state stays within `B` of `r*` forever. Disturbances larger than this can carry it out of the basin.

源码说明：**κ > 1, attractor correction under disturbance**: if the state starts within `B` of `r*`, the basin margin covers it (`B ≤ r* - ρ`, `ρ > 0`), and every disturbance is at most `(1 - factor ρ) · B`, the state stays within `B` of `r*` forever. Disturbances larger than this can carry it out of the basin.

形式陈述：

```lean
∀ {a c g ρ B e : ℝ},
  0 ≤ a →
    a < 1 →
      0 < g →
        0 < ρ →
          B ≤ CGB.Generation.HoldCorrect.rstar a c g - ρ →
            e ≤ (1 - CGB.Generation.HoldCorrect.factor a g ρ) * B →
              ∀ (ε : ℕ → ℝ),
                (∀ (t : ℕ), |ε t| ≤ e) →
                  ∀ (x₀ : ℝ),
                    |x₀ - CGB.Generation.HoldCorrect.rstar a c g| ≤ B →
                      ∀ (n : ℕ),
                        |CGB.Generation.HoldCorrect.orbit (CGB.Generation.HoldCorrect.fb a c g) ε x₀ n -
                              CGB.Generation.HoldCorrect.rstar a c g| ≤
                          B
```

## CGB.Generation.HoldCorrect.latch_tolerance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L261)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：锁存容限 (sqrt c - sqrt((1-a)/g))^2 内的扰动不翻转状态。

解释：**The latch's noise tolerance** (κ > 1). Write `u = √((1-a)/g)` and `v = √c` (given as positive numbers with these squares, so no square root is needed): `√κ = v/u > 1`. With the margin `ρ = (v/u - 1)/g` the condition of `latch_corrects` holds (`0 < ρ ≤ r*`) and the tolerated disturbance per event, `(1 - factor ρ) · (r* - ρ)`, equals `(v - u)²`.

源码说明：**The latch's noise tolerance** (κ > 1). Write `u = √((1-a)/g)` and `v = √c` (given as positive numbers with these squares, so no square root is needed): `√κ = v/u > 1`. With the margin `ρ = (v/u - 1)/g` the condition of `latch_corrects` holds (`0 < ρ ≤ r*`) and the tolerated disturbance per event, `(1 - factor ρ) · (r* - ρ)`, equals `(v - u)²`.

形式陈述：

```lean
∀ {a c g u v : ℝ},
  a < 1 →
    0 < g →
      0 < u →
        0 < v →
          u ^ 2 = (1 - a) / g →
            v ^ 2 = c →
              u < v →
                have ρ := (v / u - 1) / g;
                0 < ρ ∧
                  ρ ≤ CGB.Generation.HoldCorrect.rstar a c g ∧
                    (1 - CGB.Generation.HoldCorrect.factor a g ρ) * (CGB.Generation.HoldCorrect.rstar a c g - ρ) =
                      (v - u) ^ 2
```

## CGB.Generation.HoldCorrect.tolerance_falls_with_decay

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L287)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：更强衰减（a 更小）降低锁存容限：加大衰减不能保护锁存。

解释：**Decay weakens the latch**: the tolerance `(v - u)²` with `u = √((1-a)/g)` falls as decay grows: a larger `u` (stronger decay, smaller retention `a`) with the latch still present (`u ≤ v`) tolerates no more disturbance.

源码说明：**Decay weakens the latch**: the tolerance `(v - u)²` with `u = √((1-a)/g)` falls as decay grows: a larger `u` (stronger decay, smaller retention `a`) with the latch still present (`u ≤ v`) tolerates no more disturbance.

形式陈述：

```lean
∀ {u₁ u₂ v : ℝ}, u₁ ≤ u₂ → u₂ ≤ v → (v - u₂) ^ 2 ≤ (v - u₁) ^ 2
```

## CGB.Generation.HoldCorrect.rates_vanish_at_transition

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L294)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：相变点两侧恢复速率 (1-a)(1-kappa)、(1-a)(1-1/kappa) 趋于 0（临界慢化）。

解释：**Critical slowing at κ = 1**: the forgetting rate below the transition and the correction rate at `r*` above it are `(1-a)(1-κ)` and `(1-a)(1-1/κ)` with `κ = c g / (1 - a)`; both vanish as κ → 1.

源码说明：**Critical slowing at κ = 1**: the forgetting rate below the transition and the correction rate at `r*` above it are `(1-a)(1-κ)` and `(1-a)(1-1/κ)` with `κ = c g / (1 - a)`; both vanish as κ → 1.

形式陈述：

```lean
∀ {a c g : ℝ},
  a < 1 →
    0 < g →
      0 < c →
        1 - (a + c * g) = (1 - a) * (1 - c * g / (1 - a)) ∧
          1 - CGB.Generation.HoldCorrect.factor a g (CGB.Generation.HoldCorrect.rstar a c g) =
            (1 - a) * (1 - 1 / (c * g / (1 - a)))
```

## CGB.Generation.HoldCorrect.long_timescale_band

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L317)

分类：design；适用范围：candidate；审核：draft；候选：FeedbackDelta

分类说明：靠近相变点存在无尺度的长时间尺度带。

解释：**Scale-free timescales near criticality**: a head's relaxation time exceeds `T` exactly when its `κ` lies within `1 / ((1-a) T)` of 1. The band shrinks as `1/T`, so a population whose `κ` is spread over a neighbourhood of 1 contains every timescale, the share above `T` falling as `1/T`: one mechanism family, many timescales, without assigning them.

源码说明：**Scale-free timescales near criticality**: a head's relaxation time exceeds `T` exactly when its `κ` lies within `1 / ((1-a) T)` of 1. The band shrinks as `1/T`, so a population whose `κ` is spread over a neighbourhood of 1 contains every timescale, the share above `T` falling as `1/T`: one mechanism family, many timescales, without assigning them.

形式陈述：

```lean
∀ {a κ T : ℝ}, a < 1 → 0 < T → κ ≠ 1 → (T < CGB.Generation.HoldCorrect.relaxTime a κ ↔ |κ - 1| < 1 / ((1 - a) * T))
```
