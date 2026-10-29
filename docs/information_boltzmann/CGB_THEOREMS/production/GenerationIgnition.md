# GenerationIgnition

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean) · [返回导航](../README.md)

模块源码说明：Ignition needs drive or interaction (generation chain, G23) Ignition (global-workspace theory) is all-or-none, self-sustaining activation: the number of active units has two peaks (off and ignited) with a valley between

## CGB.Generation.choose_log_concave

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L19)

解释：相邻三个二项式系数满足中间项平方不小于两侧项乘积，即二项式系数序列对数凹。

源码说明：Binomial coefficients are log-concave.

形式陈述：

```lean
∀ (N k : ℕ), N.choose k * N.choose (k + 2) ≤ N.choose (k + 1) ^ 2
```

## CGB.Generation.activeWeight_log_concave

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L47)

解释：活动比率非负时，由二项式系数乘比率幂构成的活跃数量权重仍满足对数凹不等式。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (N : ℕ) {r : ℝ},
  0 ≤ r →
    ∀ (k : ℕ),
      CGB.Generation.activeWeight N r k * CGB.Generation.activeWeight N r (k + 2) ≤
        CGB.Generation.activeWeight N r (k + 1) ^ 2
```

## CGB.Generation.no_valley

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L61)

解释：对该非负活动比率的可加平衡权重，相邻三个活跃数量的中间权重不可能严格低于两侧。

源码说明：**G23 (no ignition in equilibrium with additive energies)**: the distribution of the number of active units has no strict valley, so it cannot have an "off" peak and an "ignited" peak separated by a gap.

形式陈述：

```lean
∀ (N : ℕ) {r : ℝ},
  0 ≤ r →
    ∀ (k : ℕ),
      ¬(CGB.Generation.activeWeight N r (k + 1) < CGB.Generation.activeWeight N r k ∧
          CGB.Generation.activeWeight N r (k + 1) < CGB.Generation.activeWeight N r (k + 2))
```

## CGB.Generation.pulse_rate_le

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L83)

解释：非负速率系数下，刺激引起的二态总松弛速率不超过无刺激速率乘以 exp(|μ|/2)。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {κ : ℝ}, 0 ≤ κ → ∀ (Δ μ : ℝ), CGB.Generation.relaxRate κ (Δ - μ) ≤ Real.exp (|μ| / 2) * CGB.Generation.relaxRate κ Δ
```

## CGB.Generation.equilibrium_memory_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L100)

解释：若写入速率受指定平衡比率约束，则持续 tp 的脉冲经过正延迟 T 后保留的偏离量受与 tp/T 成正比的上界限制。

源码说明：**G24 (equilibrium memory bound)**: loading at rate `lp ≤ e^{|μ|/2} l0` for time `tp`, then relaxing at rate `l0` for time `T`, keeps at most `e^{|μ|/2 - 1} tp / T` of the deviation, for every choice of `l0`.

形式陈述：

```lean
∀ {l0 lp tp T μ : ℝ},
  0 ≤ tp →
    0 < T →
      lp ≤ Real.exp (|μ| / 2) * l0 →
        (1 - Real.exp (-(lp * tp))) * Real.exp (-(l0 * T)) ≤ Real.exp (|μ| / 2 - 1) * tp / T
```
