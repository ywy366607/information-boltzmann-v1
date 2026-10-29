# GenerationAmplifier

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean) · [返回导航](../README.md)

模块源码说明：An equilibrium amplifier: ordered filling of a conserved shell (generation chain, G25) Chemotaxis needs an all-or-none response to a tiny difference (a gain of order ten in *E.

## CGB.Generation.occupancy_le_exp

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L42)

解释：logistic 占用率对任意实输入都不超过该输入的指数。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (y : ℝ), CGB.Generation.occupancy y ≤ Real.exp y
```

## CGB.Generation.one_sub_occupancy_le_exp

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L49)

解释：logistic 空余率对任意实输入都不超过输入负值的指数。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (y : ℝ), 1 - CGB.Generation.occupancy y ≤ Real.exp (-y)
```

## CGB.Generation.occupancy_le_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L61)

解释：logistic 占用率对任意实输入都不超过一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (y : ℝ), CGB.Generation.occupancy y ≤ 1
```

## CGB.Generation.high_off_below_threshold

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L68)

解释：当低层势不超过其阈值时，指定高层占用率不超过 exp(−Δ)。

源码说明：**G25a (H stays off below threshold)**: while the low page is at most half full, the high page is occupied at most `e^{-Δ}`.

形式陈述：

```lean
∀ {l φL Δ : ℝ}, l ≤ φL → CGB.Generation.occupancy (l - (φL + Δ)) ≤ Real.exp (-Δ)
```

## CGB.Generation.switch_width

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L78)

解释：当低层容量非负时，两种势之间的内容增量受高层占用增量与低层指数尾项之和约束。

源码说明：**G25b (switch width)**: raising the potential from `l1` to `l2` adds at most `NH` times the gain in `H`'s occupancy plus `NL e^{-(l1 - φL)}` content.

形式陈述：

```lean
∀ {NL NH φL φH l1 l2 : ℝ},
  0 ≤ NL →
    CGB.Generation.shellContent NL NH φL φH l2 - CGB.Generation.shellContent NL NH φL φH l1 ≤
      NH * (CGB.Generation.occupancy (l2 - φH) - CGB.Generation.occupancy (l1 - φH)) + NL * Real.exp (-(l1 - φL))
```

## CGB.Generation.equilibrium_hill_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L90)

解释：单量子平衡占用率的对数几率等于势差 μ−φ，即对势差的系数为一。

源码说明：**G25c (equilibrium Hill coefficient is one)**: the log-odds of occupation is exactly the potential above the unit's energy, so it moves one-for-one with the sensed potential.

形式陈述：

```lean
∀ (μ φ : ℝ), Real.log (CGB.Generation.occupancy (μ - φ) / (1 - CGB.Generation.occupancy (μ - φ))) = μ - φ
```

## CGB.Generation.quanta_hill

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L99)

解释：指定 s 量子状态与空状态的平衡权重比的对数恰为 sμ。

源码说明：**G25d (cooperativity from multiple quanta)**: a state holding `s` units of content, with energy `φs`, has equilibrium weight `exp (s μ - φs)` relative to the empty state, so its log-odds moves `s`-for-one with the potential: its Hill coefficient is `s`. Binary units give 1 (G25c); a unit that takes up several quanta at once is cooperative at equilibrium, as in Hill's haemoglobin. (Found by selection in the multi-state chemotaxis run before it was written down.)

形式陈述：

```lean
∀ (s μ φs : ℝ), Real.log (Real.exp (s * μ - φs) / Real.exp (s * 0 - φs)) = s * μ
```

## CGB.Generation.occupancy_ge_inv_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L106)

解释：N>1 时，logistic 占用率至少为 1/N 当且仅当 −h 不超过 log(N−1)。

源码说明：Lower edge of the resolvable range: with `N` units, the occupancy is at least one unit's worth (`1/N`) exactly when `-h ≤ log (N - 1)`.

形式陈述：

```lean
∀ {N h : ℝ}, 1 < N → (1 / N ≤ CGB.Generation.occupancy h ↔ -h ≤ Real.log (N - 1))
```

## CGB.Generation.occupancy_le_one_sub_inv_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L114)

解释：N>1 时，logistic 占用率至多为 1−1/N 当且仅当 h 不超过 log(N−1)。

源码说明：Upper edge: the occupancy leaves at least one unit empty exactly when `h ≤ log (N - 1)`.

形式陈述：

```lean
∀ {N h : ℝ}, 1 < N → (CGB.Generation.occupancy h ≤ 1 - 1 / N ↔ h ≤ Real.log (N - 1))
```

## CGB.Generation.resolvable_band

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L140)

解释：N>1 且增益为正时，指定 logistic 输出处于可分辨区间当且仅当输入与阈值之差处于给定范围。

源码说明：**G25e (gain–range bound)**: a switch of gain `s` (a state of `s` quanta, G25d) centred at `θ`, read through `N` units, is resolvable (between one occupied and one empty unit) exactly on `|μ - θ| ≤ log (N - 1) / s`.

形式陈述：

```lean
∀ {N s μ θ : ℝ},
  1 < N →
    0 < s →
      (1 / N ≤ CGB.Generation.occupancy (s * (μ - θ)) ∧ CGB.Generation.occupancy (s * (μ - θ)) ≤ 1 - 1 / N ↔
        |μ - θ| ≤ Real.log (N - 1) / s)
```

## CGB.Generation.gain_times_band

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L151)

解释：非零增益 s 与给定可分辨带宽的乘积恒为 2log(N−1)。

源码说明：The product of gain and resolvable width is `2 log (N - 1)`, independent of the gain: an amplified switch covers a proportionally narrower band of the sensed potential.

形式陈述：

```lean
∀ {N s : ℝ}, s ≠ 0 → s * (2 * (Real.log (N - 1) / s)) = 2 * Real.log (N - 1)
```

## CGB.Generation.fixed_threshold_fails

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L163)

解释：在指定输入范围宽于固定阈值可分辨带宽时，该范围内必有输入使输出不可分辨。

源码说明：**G25f (a fixed threshold fails on a wide world)**: if the sensed potential ranges over an interval wider than `2 log (N - 1) / s`, a switch of gain `s` at any fixed `θ` is unresolvable somewhere in it. With G25c (at equilibrium every unit's threshold is its own fixed energy `φ`), no equilibrium pair table can compare at gain `s` across such a world.

形式陈述：

```lean
∀ {N s θ a b : ℝ},
  1 < N →
    0 < s →
      a ≤ b →
        2 * (Real.log (N - 1) / s) < b - a →
          ∃ μ, a ≤ μ ∧ μ ≤ b ∧ ¬CGB.Generation.Resolvable N (CGB.Generation.occupancy (s * (μ - θ)))
```

## CGB.Generation.centred_threshold_resolves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L178)

解释：当预测误差不超过 δ 且增益与 δ 满足给定界时，以预测为中心的指定阈值保持可分辨。

源码说明：**G25f (a threshold centred on the prediction works everywhere)**: if the threshold sits at a stored prediction `μhat` (G13: the slow interior's conditional) and the innovation `μ - μhat` (G14) stays within `δ` with `s δ ≤ log (N - 1)`, the switch is resolvable whatever the absolute level of `μ`. Moving the threshold with the stored past is adaptation; by G25c it needs drive (or a non-additive energy).

形式陈述：

```lean
∀ {N s μ μhat δ : ℝ},
  1 < N →
    0 < s →
      s * δ ≤ Real.log (N - 1) →
        |μ - μhat| ≤ δ → CGB.Generation.Resolvable N (CGB.Generation.occupancy (s * (μ - μhat)))
```
