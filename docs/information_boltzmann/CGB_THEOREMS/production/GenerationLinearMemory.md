# GenerationLinearMemory

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean) · [返回导航](../README.md)

模块源码说明：Linear recurrent memory cannot latch (generation chain, corollary of G17/G24) The results below concern the specific recurrence `x ↦ α (x - β ⟪k, x⟫_ℝ k) + (write)`, with `‖k‖ = 1`, `β ∈ [0, 2]`, `α < 1`.

## CGB.Generation.dot_delta

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L37)

解释：当键向量平方范数为一时，delta 步骤后的状态平方范数精确等于原范数减去 β(2−β) 乘投影平方；该恒等式适用于任意实数 β。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (k x : Fin n → ℝ),
  CGB.Generation.dot k k = 1 →
    ∀ (β : ℝ),
      CGB.Generation.dot (CGB.Generation.deltaStep k β x) (CGB.Generation.deltaStep k β x) =
        CGB.Generation.dot x x - β * (2 - β) * CGB.Generation.dot k x ^ 2
```

## CGB.Generation.delta_nonexpansive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L51)

解释：键向量范数为一且强度 β 位于零到二之间时，delta 更新不会增大状态的平方范数。

源码说明：The delta step never expands (squared norm), for `0 ≤ β ≤ 2`.

形式陈述：

```lean
∀ {n : ℕ} (k x : Fin n → ℝ),
  CGB.Generation.dot k k = 1 →
    ∀ {β : ℝ},
      0 ≤ β →
        β ≤ 2 →
          CGB.Generation.dot (CGB.Generation.deltaStep k β x) (CGB.Generation.deltaStep k β x) ≤ CGB.Generation.dot x x
```

## CGB.Generation.gated_delta_contracts

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L61)

解释：对指定形式 α·deltaStep(k,β,d)，若键为单位向量且 β 在零到二之间，更新后的平方范数不超过原平方范数乘 α²；此式不自动适用于 αd−β⟨k,d⟩k。

源码说明：One gated delta step on the difference of two states (the write cancels): the squared gap shrinks by `α²`. Since `deltaStep` is linear in `x`, this is the step on the gap.

形式陈述：

```lean
∀ {n : ℕ} (k d : Fin n → ℝ),
  CGB.Generation.dot k k = 1 →
    ∀ {α β : ℝ},
      0 ≤ β →
        β ≤ 2 →
          (CGB.Generation.dot (fun i => α * CGB.Generation.deltaStep k β d i) fun i =>
              α * CGB.Generation.deltaStep k β d i) ≤
            α ^ 2 * CGB.Generation.dot d d
```

## CGB.Generation.deltaStep_sub

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L70)

解释：对于固定键与强度，两个状态分别经过 delta 更新后的差，等于对它们的原始差应用同一 delta 更新。这体现该步骤相对状态的线性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (k : Fin n → ℝ) (β : ℝ) (x y : Fin n → ℝ),
  CGB.Generation.deltaStep k β x - CGB.Generation.deltaStep k β y = CGB.Generation.deltaStep k β (x - y)
```

## CGB.Generation.histories_merge

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L78)

解释：若每步历史差距至多乘以非负因子 α，T 步后的差距不超过初始差距乘所有 α 的积。

源码说明：**Histories merge**: if every step shrinks the gap between two states by at most `α t`, the gap after `T` steps is at most `∏ α t` times the initial gap.

形式陈述：

```lean
∀ (gap α : ℕ → ℝ),
  (∀ (t : ℕ), 0 ≤ α t) → (∀ (t : ℕ), gap (t + 1) ≤ α t * gap t) → ∀ (T : ℕ), gap T ≤ (∏ t ∈ Finset.range T, α t) * gap 0
```

## CGB.Generation.histories_merge_uniform

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L90)

解释：若各步收缩因子均不超过统一上界，历史差距受该上界的 T 次幂乘初始差距约束。

源码说明：With a uniform gate bound the gap decays at least geometrically.

形式陈述：

```lean
∀ (gap α : ℕ → ℝ) {ab : ℝ},
  (∀ (t : ℕ), 0 ≤ α t) →
    (∀ (t : ℕ), α t ≤ ab) → 0 ≤ gap 0 → (∀ (t : ℕ), gap (t + 1) ≤ α t * gap t) → ∀ (T : ℕ), gap T ≤ ab ^ T * gap 0
```

## CGB.Generation.linear_hold_keeps_error

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L105)

解释：若线性映射原样保持信号 s，那么沿 s 方向的任意比例扰动也会原样保留。因此该映射不能同时对这类扰动作精确纠错；结论只针对这里的线性映射及扰动方向。

源码说明：**A linear hold cannot clean up**: if a linear map holds the signal `s` (`L s = s`), a perturbation along `s` is held exactly as well, `L (s + t • s) - s = t • s`. Holding and error correction are tied in linear memory, as writing and forgetting are tied in equilibrium (G24); an attractor that maps `s + δ` back to `s` needs nonlinearity.

形式陈述：

```lean
∀ {E : Type u_1} [inst : AddCommGroup E] [inst_1 : _root_.Module ℝ E] (L : E →ₗ[ℝ] E) {s : E},
  L s = s → ∀ (t : ℝ), L (s + t • s) - s = t • s
```
