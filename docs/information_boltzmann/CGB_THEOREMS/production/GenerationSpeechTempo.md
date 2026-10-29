# GenerationSpeechTempo

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean) · [返回导航](../README.md)

模块源码说明：Speech tempo between two finite bodies (generation chain, 2026-10-28, edge N7) A speaker emits units (bytes, syllables) at rate `r` per tick to a listener.

## CGB.SpeechTempo.unsettled_eq_rpow

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L59)

解释：当 α<1 时，(1−α)^(k/r) 等于指定指数形式 exp(−k·(−log(1−α))/r)。

源码说明：The settling factor is the unified reference's geometric bound: `(1 - α) ^ (k / r)` with `L = -log (1 - α)`.

形式陈述：

```lean
∀ (α k r : ℝ), α < 1 → (1 - α) ^ (k / r) = Real.exp (-(k * -Real.log (1 - α) / r))
```

## CGB.SpeechTempo.retention_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L65)

解释：对任意实数 τ 和 r，所定义的 retention 严格为正。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (τ r : ℝ), 0 < CGB.SpeechTempo.retention τ r
```

## CGB.SpeechTempo.retention_le_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L67)

解释：当 r 和 τ 为正时，所定义的 retention 不超过一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (τ r : ℝ), 0 < r → 0 < τ → CGB.SpeechTempo.retention τ r ≤ 1
```

## CGB.SpeechTempo.settled_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L73)

解释：若 kL/r 非负，则按定义计算的 settled 值非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (k L r : ℝ), 0 ≤ k * L / r → 0 ≤ CGB.SpeechTempo.settled k L r
```

## CGB.SpeechTempo.settled_le_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L80)

解释：对任意实数 k、L 和 r，所定义的 settled 不超过一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (k L r : ℝ), CGB.SpeechTempo.settled k L r ≤ 1
```

## CGB.SpeechTempo.settled_le

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L84)

解释：对任意实数 k、L 和 r，所定义的 settled 不超过 kL/r。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (k L r : ℝ), CGB.SpeechTempo.settled k L r ≤ k * L / r
```

## CGB.SpeechTempo.useful_le_rate

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L93)

解释：在指定非负与正性条件下，useful 值不超过 r。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (C τ k L r : ℝ), 0 ≤ C → 0 < τ → 0 ≤ k * L → 0 < r → CGB.SpeechTempo.useful C τ k L r ≤ r
```

## CGB.SpeechTempo.useful_le_capacity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L105)

解释：在 C≥0、τ>0、kL≥0 和 r>0 的条件下，指定 useful 值不超过参数 C。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (C τ k L r : ℝ), 0 ≤ C → 0 < τ → 0 ≤ k * L → 0 < r → CGB.SpeechTempo.useful C τ k L r ≤ C
```

## CGB.SpeechTempo.useful_le_inv_rate

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L115)

解释：在指定非负与正性条件下，useful 值不超过 CkL/r。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (C τ k L r : ℝ), 0 ≤ C → 0 < τ → 0 ≤ k * L → 0 < r → CGB.SpeechTempo.useful C τ k L r ≤ C * k * L / r
```

## CGB.SpeechTempo.tempo_band

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L127)

解释：当 C≥0、τ>0、kL≥0 且 r>0 时，若 r 小于 u 或大于 CkL/u，则定义的 useful 小于正阈值 u。

源码说明：**N7**: the good tempo is a bounded band. If some tempo reaches the useful rate `u > 0`, every tempo slower than `u` or faster than `C k L / u` does worse.

形式陈述：

```lean
∀ (C τ k L r : ℝ),
  0 ≤ C →
    0 < τ → 0 ≤ k * L → 0 < r → ∀ (u : ℝ), 0 < u → r < u ∨ C * k * L / u < r → CGB.SpeechTempo.useful C τ k L r < u
```

## CGB.SpeechTempo.slow_retention_vanishes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L139)

解释：当 τ 为正时，对任意正 ε，足够小的正速率 r 使 retention 小于 ε。

源码说明：Too slow: per unit, the earlier part is forgotten, whatever the listener's capacity.

形式陈述：

```lean
∀ (τ ε : ℝ), 0 < τ → 0 < ε → ∃ r₀ > 0, ∀ (r : ℝ), 0 < r → r < r₀ → CGB.SpeechTempo.retention τ r < ε
```

## CGB.SpeechTempo.fast_settling_vanishes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L155)

解释：若 kL 非负，则对任意正 ε，足够大的速率 r 使定义的 settled 值小于 ε。

源码说明：Too fast: per unit, the speaker is unsettled, whatever the listener's capacity.

形式陈述：

```lean
∀ (k L ε : ℝ), 0 ≤ k * L → 0 < ε → ∃ r₁, ∀ (r : ℝ), r₁ < r → CGB.SpeechTempo.settled k L r < ε
```

## CGB.SpeechTempo.peaks_wasted

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L166)

解释：对有限序列逐项截断到 C 再求和，结果不超过总和与 nC 两者的较小值。

源码说明：Uniform information density, upper bound: with a per-slot capacity `C`, the slots absorb at most the total sent and at most `n C`; anything above capacity in a slot is lost.

形式陈述：

```lean
∀ {n : ℕ} (x : Fin n → ℝ) (C : ℝ), ∑ i, min (x i) C ≤ min (∑ i, x i) (↑n * C)
```

## CGB.SpeechTempo.uniform_attains

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L173)

解释：对正整数 n，n 个相同的 min(I/n,C) 之和等于 min(I,nC)。

源码说明：Uniform information density, attainment: the even spread reaches the bound.

形式陈述：

```lean
∀ (n : ℕ), 0 < n → ∀ (I C : ℝ), ∑ _i, min (I / ↑n) C = min I (↑n * C)
```
