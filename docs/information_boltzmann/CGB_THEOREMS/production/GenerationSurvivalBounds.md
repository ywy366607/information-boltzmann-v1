# GenerationSurvivalBounds

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean) · [返回导航](../README.md)

模块源码说明：Conditional maintenance, not immortality Maintenance means longer expected structural lifetime with drive than without (`CGB_AXIOMS_V0_20261022.md`, section 4).

## CGB.SurvivalBounds.decay_eventually_lt

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L22)

源码说明：Reuse the existing geometric bound; no nonnegativity of `q` is needed for this upper-tail conclusion. For survival, each index can denote a completed block.

形式陈述：

```lean
∀ (q : ℕ → ℝ) (p : ℝ),
  0 < p → p ≤ 1 → q 0 ≤ 1 → (∀ (n : ℕ), q (n + 1) ≤ (1 - p) * q n) → ∀ (τ : ℝ), 0 < τ → ∃ N, ∀ (n : ℕ), N ≤ n → q n < τ
```

## CGB.SurvivalBounds.no_uniform_positive_lower

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L31)

源码说明：Uniform positive irreversible block loss rules out a fixed positive floor.

形式陈述：

```lean
∀ (q : ℕ → ℝ) (p : ℝ),
  0 < p → p ≤ 1 → q 0 ≤ 1 → (∀ (n : ℕ), q (n + 1) ≤ (1 - p) * q n) → ∀ (τ : ℝ), 0 < τ → ¬∀ (n : ℕ), τ ≤ q n
```

## CGB.SurvivalBounds.geometric_lower

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L38)

源码说明：Lower counterpart to the reused geometric upper bound.

形式陈述：

```lean
∀ (q : ℕ → ℝ), ∀ p ≤ 1, 1 ≤ q 0 → (∀ (n : ℕ), (1 - p) * q n ≤ q (n + 1)) → ∀ (n : ℕ), (1 - p) ^ n ≤ q n
```

## CGB.SurvivalBounds.cumulative_risk_lower

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L51)

源码说明：Telescoping lower bound; no independence, absorption, or sign premise is needed for this algebraic statement. Nonnegative `risk` gives its risk reading.

形式陈述：

```lean
∀ (q risk : ℕ → ℝ), (∀ (n : ℕ), q n - risk n ≤ q (n + 1)) → ∀ (n : ℕ), q 0 - ∑ k ∈ Finset.range n, risk k ≤ q n
```

## CGB.SurvivalBounds.risk_budget_uniform_lower

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L62)

源码说明：A prefix budget allows strictly positive but summably bounded per-step risk. It does not require `δ < 1`; that premise is needed only for a positive floor.

形式陈述：

```lean
∀ (q risk : ℕ → ℝ) (δ : ℝ),
  q 0 = 1 →
    (∀ (n : ℕ), q n - risk n ≤ q (n + 1)) → (∀ (n : ℕ), ∑ k ∈ Finset.range n, risk k ≤ δ) → ∀ (n : ℕ), 1 - δ ≤ q n
```

## CGB.SurvivalBounds.geometric_tail_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L82)

源码说明：Uniform scalar decay bounds every finite cut tail, reusing the existing pointwise geometric bound rather than repeating its power induction.

形式陈述：

```lean
∀ (q : ℕ → ℝ) (p : ℝ),
  0 < p →
    p ≤ 1 →
      q 0 ≤ 1 → (∀ (n : ℕ), q (n + 1) ≤ (1 - p) * q n) → ∀ (T L : ℕ), ∑ k ∈ Finset.range L, q (T + k) ≤ (1 - p) ^ T / p
```

## CGB.SurvivalBounds.prefix_gap_persists

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L103)

源码说明：A prefix advantage exceeding the entire cut-tail allowance persists at all later horizons; normal needs only nonnegative scores, not contraction.

形式陈述：

```lean
∀ (normal cut : ℕ → ℝ) (T : ℕ) (tailBound : ℝ),
  (∀ (n : ℕ), 0 ≤ normal n) →
    (∀ (L : ℕ), ∑ k ∈ Finset.range L, cut (T + k) ≤ tailBound) →
      tailBound < ∑ t ∈ Finset.range T, normal t - ∑ t ∈ Finset.range T, cut t →
        ∀ (N : ℕ), T ≤ N → ∑ t ∈ Finset.range N, cut t < ∑ t ∈ Finset.range N, normal t
```

## CGB.SurvivalBounds.finite_window_lifetime_transfer

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L117)

源码说明：Finite-window cumulative errors plus an actual cut-tail certificate transfer reference lifetime advantage to every actual horizon beyond the window.

形式陈述：

```lean
∀ (qN qC rN rC : ℕ → ℝ) (T : ℕ) (E_N E_C tailBound : ℝ),
  (∀ (n : ℕ), 0 ≤ rN n) →
    (∀ (L : ℕ), ∑ k ∈ Finset.range L, rC (T + k) ≤ tailBound) →
      |∑ t ∈ Finset.range T, rN t - ∑ t ∈ Finset.range T, qN t| ≤ E_N →
        |∑ t ∈ Finset.range T, rC t - ∑ t ∈ Finset.range T, qC t| ≤ E_C →
          E_N + E_C + tailBound < ∑ t ∈ Finset.range T, qN t - ∑ t ∈ Finset.range T, qC t →
            ∀ (N : ℕ), T ≤ N → ∑ t ∈ Finset.range N, rC t < ∑ t ∈ Finset.range N, rN t
```

## CGB.SurvivalBounds.expected_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L137)

源码说明：General expected-step identity extracted from the rearrangement in `Energetics.expected_first_law`. The observable is not an energy `Account`.

形式陈述：

```lean
∀ {S : Type u_1} {O : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype O]
  (P : CGB.Interaction.Machine S ℕ O) (initial : S) (v : S → ℝ) (t : ℕ),
  CGB.Energetics.mean (CGB.Energetics.law P initial (t + 1)) v =
    ∑ s, CGB.Energetics.law P initial t s * ∑ o, ∑ s', P.step t s o s' * v s'
```

## CGB.SurvivalBounds.mean_initial

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L149)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {O : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype O]
  (P : CGB.Interaction.Machine S ℕ O) (initial : S) (v : S → ℝ),
  CGB.Energetics.mean (CGB.Energetics.law P initial 0) v = v initial
```

## CGB.SurvivalBounds.mean_unit_interval

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L155)

源码说明：A bounded observable gives a bounded score under the actual process law.

形式陈述：

```lean
∀ {S : Type u_1} {O : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype O]
  (P : CGB.Interaction.Machine S ℕ O) (initial : S) (v : S → ℝ),
  (∀ (s : S), 0 ≤ v s ∧ v s ≤ 1) →
    ∀ (t : ℕ),
      0 ≤ CGB.Energetics.mean (CGB.Energetics.law P initial t) v ∧
        CGB.Energetics.mean (CGB.Energetics.law P initial t) v ≤ 1
```

## CGB.SurvivalBounds.mean_contraction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L168)

源码说明：A local row contraction transfers to the original ensemble score.

形式陈述：

```lean
∀ {S : Type u_1} {O : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype O]
  (P : CGB.Interaction.Machine S ℕ O) (initial : S) (v : S → ℝ) (p : ℝ) (t : ℕ),
  (∀ (s : S), ∑ o, ∑ s', P.step t s o s' * v s' ≤ (1 - p) * v s) →
    CGB.Energetics.mean (CGB.Energetics.law P initial (t + 1)) v ≤
      (1 - p) * CGB.Energetics.mean (CGB.Energetics.law P initial t) v
```

## CGB.SurvivalBounds.mean_risk_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L181)

源码说明：A local additive risk allowance transfers without independence assumptions.

形式陈述：

```lean
∀ {S : Type u_1} {O : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype O]
  (P : CGB.Interaction.Machine S ℕ O) (initial : S) (v : S → ℝ) (risk : ℝ) (t : ℕ),
  (∀ (s : S), v s - risk ≤ ∑ o, ∑ s', P.step t s o s' * v s') →
    CGB.Energetics.mean (CGB.Energetics.law P initial t) v - risk ≤
      CGB.Energetics.mean (CGB.Energetics.law P initial (t + 1)) v
```

## CGB.SurvivalBounds.mean_retention_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L195)

源码说明：Lower multiplicative row bounds transfer along the same actual law.

形式陈述：

```lean
∀ {S : Type u_1} {O : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype O]
  (P : CGB.Interaction.Machine S ℕ O) (initial : S) (v : S → ℝ) (p : ℝ) (t : ℕ),
  (∀ (s : S), (1 - p) * v s ≤ ∑ o, ∑ s', P.step t s o s' * v s') →
    (1 - p) * CGB.Energetics.mean (CGB.Energetics.law P initial t) v ≤
      CGB.Energetics.mean (CGB.Energetics.law P initial (t + 1)) v
```

## CGB.SurvivalBounds.expectedLifetime_strict_of_row_gap

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L216)

源码说明：Positive maintenance without immortality: the same initial individual and observable have strictly longer truncated lifetime under the normal process. The advantage is strict at tick 1 and nonnegative at every tick.

形式陈述：

```lean
∀ {S : Type u_1} {O : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype O]
  (normal cut : CGB.Interaction.Machine S ℕ O) (initial : S) (a : S → ℝ),
  (∀ (s : S), 0 ≤ a s ∧ a s ≤ 1) →
    a initial = 1 →
      ∀ (pN pC : ℝ),
        0 ≤ pN →
          pN < pC →
            pC ≤ 1 →
              (∀ (t : ℕ) (s : S), (1 - pN) * a s ≤ ∑ o, ∑ s', normal.step t s o s' * a s') →
                (∀ (t : ℕ) (s : S), ∑ o, ∑ s', cut.step t s o s' * a s' ≤ (1 - pC) * a s) →
                  ∀ (T : ℕ),
                    2 ≤ T →
                      CGB.SurvivalBounds.expectedLifetime cut initial a T <
                        CGB.SurvivalBounds.expectedLifetime normal initial a T
```

## CGB.SurvivalBounds.normal_cut_eventual_gap

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L246)

源码说明：Stronger, separate all-time-floor criterion: SAME initial state and observable, normal prefix risk budget versus uniform cut contraction. The row premises must be verified on those two processes. No infinite-path measure is assumed.

形式陈述：

```lean
∀ {S : Type u_1} {O : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype O]
  (normal cut : CGB.Interaction.Machine S ℕ O) (initial : S) (v : S → ℝ),
  (∀ (s : S), 0 ≤ v s ∧ v s ≤ 1) →
    v initial = 1 →
      ∀ (risk : ℕ → ℝ) (δ p : ℝ),
        δ < 1 →
          0 < p →
            p ≤ 1 →
              (∀ (n : ℕ), ∑ k ∈ Finset.range n, risk k ≤ δ) →
                (∀ (t : ℕ) (s : S), v s - risk t ≤ ∑ o, ∑ s', normal.step t s o s' * v s') →
                  (∀ (t : ℕ) (s : S), ∑ o, ∑ s', cut.step t s o s' * v s' ≤ (1 - p) * v s) →
                    (∀ (n : ℕ), 1 - δ ≤ CGB.Energetics.mean (CGB.Energetics.law normal initial n) v) ∧
                      ∃ N,
                        ∀ (n : ℕ),
                          N ≤ n →
                            (1 - δ) / 2 <
                              CGB.Energetics.mean (CGB.Energetics.law normal initial n) v -
                                CGB.Energetics.mean (CGB.Energetics.law cut initial n) v
```
