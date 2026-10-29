# GenerationColumnMemory

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean) · [返回导航](../README.md)

模块源码说明：Column memory for the language organism (generation chain, 2026-10-28) The language organism's synaptic page is a set of columns `S_m` written by the gated delta rule (`GenerationLinearMemory`), with the activity page re

## CGB.Generation.dot_self_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L41)

解释：有限实向量与自身点积非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (x : Fin n → ℝ), 0 ≤ CGB.Generation.dot x x
```

## CGB.Generation.eq_zero_of_dot_self

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L44)

解释：有限实向量的自身点积为零时，该向量为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (x : Fin n → ℝ), CGB.Generation.dot x x = 0 → x = 0
```

## CGB.Generation.silent_rest_unique

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L51)

解释：单位键、有效 delta 强度和严格小于一的非负门控下，无写入更新的唯一不动点为零。

源码说明：**Silence has one rest**: a gated delta step without a write (`α < 1`) fixes only zero.

形式陈述：

```lean
∀ {n : ℕ} (k x : Fin n → ℝ),
  CGB.Generation.dot k k = 1 →
    ∀ {α β : ℝ}, 0 ≤ α → α < 1 → 0 ≤ β → β ≤ 2 → (fun i => α * CGB.Generation.deltaStep k β x i) = x → x = 0
```

## CGB.Generation.tiedWrite_eq_delta_add

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L66)

解释：所定义的绑定写入等于 delta 更新加上目标值沿键方向的写入项。

源码说明：The tied write is the delta step plus the write of the arrival.

形式陈述：

```lean
∀ {n : ℕ} (k x : Fin n → ℝ) (β v : ℝ),
  CGB.Generation.tiedWrite k β v x = fun i => CGB.Generation.deltaStep k β x i + β * v * k i
```

## CGB.Generation.tied_write_is_error_descent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L74)

解释：绑定写入可代数改写为按预测误差乘键方向进行的一步更新。

源码说明：**Error descent**: the tied write is a gradient step of size `β` on `½ (⟨k, x⟩ - v)²`, whose gradient in `x i` is `(⟨k, x⟩ - v) k i` (predicted minus arrived, times the presynaptic key).

形式陈述：

```lean
∀ {n : ℕ} (k x : Fin n → ℝ) (β v : ℝ),
  CGB.Generation.tiedWrite k β v x = fun i => x i - β * ((CGB.Generation.dot k x - v) * k i)
```

## CGB.Generation.tied_write_error_shrinks

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L81)

解释：单位键下，绑定写入后的预测误差等于原预测误差乘以 1−β。

源码说明：With a unit key the prediction error shrinks by `1 - β`.

形式陈述：

```lean
∀ {n : ℕ} (k x : Fin n → ℝ),
  CGB.Generation.dot k k = 1 →
    ∀ (β v : ℝ), CGB.Generation.dot k (CGB.Generation.tiedWrite k β v x) - v = (1 - β) * (CGB.Generation.dot k x - v)
```

## CGB.Generation.exact_prediction_no_write

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L94)

解释：预测值恰好等于目标值时，绑定写入不改变状态。

源码说明：An exact prediction writes nothing.

形式陈述：

```lean
∀ {n : ℕ} (k x : Fin n → ℝ) (β v : ℝ), CGB.Generation.dot k x = v → CGB.Generation.tiedWrite k β v x = x
```

## CGB.Generation.occurrence_decay_retains

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L101)

解释：当衰减系数位于零到一且每步权重不超过一时，按权重衰减的有限乘积不小于每步完整衰减。

源码说明：**Decay by use keeps rare topics**: decaying by `α ^ w t` with `w t ∈ [0, 1]` keeps at least what decaying by `α` every tick keeps.

形式陈述：

```lean
∀ (α : ℝ), 0 < α → α ≤ 1 → ∀ (w : ℕ → ℝ), (∀ (t : ℕ), w t ≤ 1) → ∀ (T : ℕ), α ^ T ≤ ∏ t ∈ Finset.range T, α ^ w t
```

## CGB.Generation.unused_column_kept

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L111)

解释：未使用方向每步的衰减指数为零，因此任意有限步累计乘积为一。

源码说明：An unused column keeps everything.

形式陈述：

```lean
∀ (α : ℝ) (T : ℕ), ∏ _t ∈ Finset.range T, α ^ 0 = 1
```

## CGB.Generation.no_latch_below_unit_gain

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L116)

解释：固定零点的一 Lipschitz 非线性与小于一的非负增益组合，其不动点只能是零。

源码说明：**No latch below unit gain**: `x ↦ σ (g x)` with a 1-Lipschitz `σ`, `σ 0 = 0` and `0 ≤ g < 1` rests only at zero.

形式陈述：

```lean
∀ (σ : ℝ → ℝ),
  (∀ (a b : ℝ), |σ a - σ b| ≤ |a - b|) → σ 0 = 0 → ∀ {g : ℝ}, 0 ≤ g → g < 1 → ∀ (x : ℝ), σ (g * x) = x → x = 0
```

## CGB.Generation.hardSat_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L125)

解释：指定硬饱和函数在零点的值为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Generation.hardSat 0 = 0
```

## CGB.Generation.hardSat_lipschitz

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L127)

解释：指定硬饱和函数不会增大任意两个输入之间的距离。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (a b : ℝ), |CGB.Generation.hardSat a - CGB.Generation.hardSat b| ≤ |a - b|
```

## CGB.Generation.latch_above_unit_gain

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L131)

解释：在增益至少为一时，硬饱和反馈有 −1、0 和 1 三个所列不动点。

源码说明：**A latch above unit gain**: with saturation and `g ≥ 1`, `± 1` rest besides `0`.

形式陈述：

```lean
∀ {g : ℝ},
  1 ≤ g → CGB.Generation.hardSat (g * 1) = 1 ∧ CGB.Generation.hardSat (g * -1) = -1 ∧ CGB.Generation.hardSat (g * 0) = 0
```
