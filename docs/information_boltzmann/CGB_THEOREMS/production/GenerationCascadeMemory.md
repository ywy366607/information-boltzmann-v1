# GenerationCascadeMemory

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean) · [返回导航](../README.md)

模块源码说明：A bank of two-state pools keeps a power-law trace (generation chain, cascade memory, 2026-10-27) G20a (`two_state_geometric`) makes the trace of one two-state pool exponential.

## CGB.Generation.single_pool_falls_below_power

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L29)

解释：对于正时间常数和正系数，足够晚时单个指数池轨迹小于 c/t。

源码说明：One pool's trace drops below `c / t` once `t > 2 τ / c`.

形式陈述：

```lean
∀ (τ c t : ℝ), 0 < τ → 0 < c → 2 * τ / c < t → Real.exp (-t / τ) / τ < c / t
```

## CGB.Generation.bank_trace_ge_one_term

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L58)

解释：正时间常数的有限池组轨迹总和不小于其中任一单池轨迹。

源码说明：The bank's trace is at least any one of its (nonnegative) terms.

形式陈述：

```lean
∀ {m : ℕ} (τ : ℕ → ℝ),
  (∀ (k : ℕ), 0 < τ k) →
    ∀ (t : ℝ),
      ∀ j ∈ Finset.range (m + 1), Real.exp (-t / τ j) / τ j ≤ ∑ k ∈ Finset.range (m + 1), Real.exp (-t / τ k) / τ k
```

## CGB.Generation.pool_in_band_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L63)

解释：正时间常数 τ 位于 [t,γt] 时，其指数轨迹 exp(−t/τ)/τ 具有给定的 1/t 型下界。

源码说明：A pool with `t ≤ τ ≤ γ t` alone keeps `1 / (e γ t)`.

形式陈述：

```lean
∀ (τ γ t : ℝ), 0 < t → 0 < γ → t ≤ τ → τ ≤ γ * t → 1 / (Real.exp 1 * (γ * t)) ≤ Real.exp (-t / τ) / τ
```

## CGB.Generation.bank_power_law_lower_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L77)

解释：若有限记忆池中有时间常数位于 t 到 γt 之间，则该池组的总轨迹有一个与 1/t 成比例的下界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {m : ℕ} (τ : ℕ → ℝ),
  (∀ (k : ℕ), 0 < τ k) →
    ∀ (γ t : ℝ),
      0 < t →
        0 < γ →
          ∀ j ∈ Finset.range (m + 1),
            t ≤ τ j → τ j ≤ γ * t → 1 / (Real.exp 1 * (γ * t)) ≤ ∑ k ∈ Finset.range (m + 1), Real.exp (-t / τ k) / τ k
```

## CGB.Generation.geometric_bank_covers

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L84)

解释：对于正初始时间常数和大于一的几何比，覆盖范围内的任意 t 都有一个池时间常数落在 [t,γt]。

源码说明：Geometric time constants cover the band: every `t ∈ [τ₀, τ₀ γ^m]` has a level `k ≤ m` with `t ≤ τ₀ γ^k ≤ γ t`.

形式陈述：

```lean
∀ (τ₀ γ t : ℝ) (m : ℕ),
  0 < τ₀ → 1 < γ → τ₀ ≤ t → t ≤ τ₀ * γ ^ m → ∃ k ∈ Finset.range (m + 1), t ≤ τ₀ * γ ^ k ∧ τ₀ * γ ^ k ≤ γ * t
```
