# GenerationColumnMemory

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean) · [返回导航](../README.md)

模块源码说明：Column memory for the language organism (generation chain, 2026-10-28) The language organism's synaptic page is a set of columns `S_m` written by the gated delta rule (`GenerationLinearMemory`), with the activity page re

## CGB.Generation.dot_self_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L43)

解释：有限实向量与自身点积非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (x : Fin n → ℝ), 0 ≤ CGB.Generation.dot x x
```

## CGB.Generation.eq_zero_of_dot_self

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L46)

解释：有限实向量的自身点积为零时，该向量为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (x : Fin n → ℝ), CGB.Generation.dot x x = 0 → x = 0
```

## CGB.Generation.silent_rest_unique

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L53)

解释：对指定无写入更新 α·deltaStep，单位键、β 在零到二之间且 0≤α<1 时，唯一不动点为零；其他更新需另行核对。

源码说明：**Silence has one rest**: a gated delta step without a write (`α < 1`) fixes only zero.

形式陈述：

```lean
∀ {n : ℕ} (k x : Fin n → ℝ),
  CGB.Generation.dot k k = 1 →
    ∀ {α β : ℝ}, 0 ≤ α → α < 1 → 0 ≤ β → β ≤ 2 → (fun i => α * CGB.Generation.deltaStep k β x i) = x → x = 0
```

## CGB.Generation.tiedWrite_eq_delta_add

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L68)

解释：所定义的绑定写入等于 delta 更新加上目标值沿键方向的写入项。

源码说明：The tied write is the delta step plus the write of the arrival.

形式陈述：

```lean
∀ {n : ℕ} (k x : Fin n → ℝ) (β v : ℝ),
  CGB.Generation.tiedWrite k β v x = fun i => CGB.Generation.deltaStep k β x i + β * v * k i
```

## CGB.Generation.tied_write_is_error_descent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L76)

解释：绑定写入可代数改写为按预测误差乘键方向进行的一步更新。

源码说明：**Error descent**: the tied write is a gradient step of size `β` on `½ (⟨k, x⟩ - v)²`, whose gradient in `x i` is `(⟨k, x⟩ - v) k i` (predicted minus arrived, times the presynaptic key).

形式陈述：

```lean
∀ {n : ℕ} (k x : Fin n → ℝ) (β v : ℝ),
  CGB.Generation.tiedWrite k β v x = fun i => x i - β * ((CGB.Generation.dot k x - v) * k i)
```

## CGB.Generation.tied_write_error_shrinks

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L83)

解释：单位键下，绑定写入后的预测误差等于原预测误差乘以 1−β。

源码说明：With a unit key the prediction error shrinks by `1 - β`.

形式陈述：

```lean
∀ {n : ℕ} (k x : Fin n → ℝ),
  CGB.Generation.dot k k = 1 →
    ∀ (β v : ℝ), CGB.Generation.dot k (CGB.Generation.tiedWrite k β v x) - v = (1 - β) * (CGB.Generation.dot k x - v)
```

## CGB.Generation.binary_delta_crossEntropy_decreases

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L104)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p : ℝ) (y : Bool) (β : ℝ),
  0 < p →
    p < 1 →
      0 < β →
        β ≤ 1 →
          have p' := p + β * ((if y = true then 1 else 0) - p);
          (0 ≤ p' ∧ p' ≤ 1 ∧ p' ≠ p) ∧
            CGB.PredictiveHarvest.crossEntropy (CGB.Generation.binaryPointLaw y) (CGB.Generation.binaryAllocation p') <
              CGB.PredictiveHarvest.crossEntropy (CGB.Generation.binaryPointLaw y) (CGB.Generation.binaryAllocation p)
```

## CGB.Generation.tied_write_cross_read

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L168)

源码说明：The change in any other readout is exactly the prediction error times the key overlap and write gain. Orthogonal readouts are therefore preserved.

形式陈述：

```lean
∀ {n : ℕ} (k q x : Fin n → ℝ) (β v : ℝ),
  CGB.Generation.dot q (CGB.Generation.tiedWrite k β v x) - CGB.Generation.dot q x =
    β * (v - CGB.Generation.dot k x) * CGB.Generation.dot q k
```

## CGB.Generation.reused_key_tracks_and_rewrites

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L187)

源码说明：Every write recovers the latest value. Whenever consecutive arrivals differ, the finite state must change again; reuse is not a monotonically shrinking set of permanently excluded candidates.

形式陈述：

```lean
∀ {n : ℕ} (k x : Fin n → ℝ),
  CGB.Generation.dot k k = 1 →
    ∀ (v : ℕ → ℝ) (t : ℕ),
      CGB.Generation.dot k (CGB.Generation.reusedKeyState k x v (t + 1)) = v t ∧
        (v (t + 1) ≠ v t → CGB.Generation.reusedKeyState k x v (t + 2) ≠ CGB.Generation.reusedKeyState k x v (t + 1))
```

## CGB.Generation.exact_prediction_no_write

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L204)

解释：预测值恰好等于目标值时，绑定写入不改变状态。

源码说明：An exact prediction writes nothing.

形式陈述：

```lean
∀ {n : ℕ} (k x : Fin n → ℝ) (β v : ℝ), CGB.Generation.dot k x = v → CGB.Generation.tiedWrite k β v x = x
```

## CGB.Generation.occurrence_decay_retains

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L211)

解释：当衰减系数位于零到一且每步权重不超过一时，按权重衰减的有限乘积不小于每步完整衰减。

源码说明：**Decay by use keeps rare topics**: decaying by `α ^ w t` with `w t ∈ [0, 1]` keeps at least what decaying by `α` every tick keeps.

形式陈述：

```lean
∀ (α : ℝ), 0 < α → α ≤ 1 → ∀ (w : ℕ → ℝ), (∀ (t : ℕ), w t ≤ 1) → ∀ (T : ℕ), α ^ T ≤ ∏ t ∈ Finset.range T, α ^ w t
```

## CGB.Generation.unused_column_kept

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L221)

解释：未使用方向每步的衰减指数为零，因此任意有限步累计乘积为一。

源码说明：An unused column keeps everything.

形式陈述：

```lean
∀ (α : ℝ) (T : ℕ), ∏ _t ∈ Finset.range T, α ^ 0 = 1
```

## CGB.Generation.no_latch_below_unit_gain

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L226)

解释：固定零点的一 Lipschitz 非线性与小于一的非负增益组合，其不动点只能是零。

源码说明：**No latch below unit gain**: `x ↦ σ (g x)` with a 1-Lipschitz `σ`, `σ 0 = 0` and `0 ≤ g < 1` rests only at zero.

形式陈述：

```lean
∀ (σ : ℝ → ℝ),
  (∀ (a b : ℝ), |σ a - σ b| ≤ |a - b|) → σ 0 = 0 → ∀ {g : ℝ}, 0 ≤ g → g < 1 → ∀ (x : ℝ), σ (g * x) = x → x = 0
```

## CGB.Generation.hardSat_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L235)

解释：指定硬饱和函数在零点的值为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Generation.hardSat 0 = 0
```

## CGB.Generation.hardSat_lipschitz

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L237)

解释：指定硬饱和函数不会增大任意两个输入之间的距离。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (a b : ℝ), |CGB.Generation.hardSat a - CGB.Generation.hardSat b| ≤ |a - b|
```

## CGB.Generation.latch_above_unit_gain

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L241)

解释：在增益至少为一时，硬饱和反馈有 −1、0 和 1 三个所列不动点。

源码说明：**A latch above unit gain**: with saturation and `g ≥ 1`, `± 1` rest besides `0`.

形式陈述：

```lean
∀ {g : ℝ},
  1 ≤ g → CGB.Generation.hardSat (g * 1) = 1 ∧ CGB.Generation.hardSat (g * -1) = -1 ∧ CGB.Generation.hardSat (g * 0) = 0
```

## CGB.Generation.self_listening_increment

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L259)

源码说明：Exact two-trajectory increment, including the feedback contribution.

形式陈述：

```lean
∀ (α β θ θ' x x' : ℝ),
  CGB.Generation.selfListeningStep α β θ' x' - CGB.Generation.selfListeningStep α β θ x =
    (α + β * θ) * (x' - x) + (1 + β * x') * (θ' - θ)
```

## CGB.Generation.self_listening_frozen_state_gap

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L266)

源码说明：Freezing the self-listening input loses the feedback part of state credit.

形式陈述：

```lean
∀ (α β θ sensitivity : ℝ), (α + β * θ) * sensitivity - α * sensitivity = β * θ * sensitivity
```

## CGB.Generation.self_listening_frozen_state_counterexample

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L272)

源码说明：A concrete nonzero feedback path makes the frozen-input credit incorrect.

形式陈述：

```lean
(0 + 1 * 1) * 1 ≠ 0 * 1
```
