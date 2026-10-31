# GenerationLanguageExistence

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean) · [返回导航](../README.md)

模块源码说明：Necessary bridges for a language-life existence theorem Keep continuous GDN mathematics separate from its physical realization.

## CGB.LanguageExistence.no_global_finite_gdn_codec

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L26)

源码说明：A finite physical code cannot losslessly represent every real GDN state. The real coordinate is an existing head bias, not an extra memory register.

形式陈述：

```lean
∀ {Physical : Type u_1} [Finite Physical] (k d : ℕ) (encode : CGB.GDNLife.State k d → Physical)
  (decode : Physical → CGB.GDNLife.State k d), ¬∀ (s : CGB.GDNLife.State k d), decode (encode s) = s
```

## CGB.LanguageExistence.p1a_no_global_real_codec

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L39)

源码说明：P1a plus a finite number of physical sites rules out a global exact real-state codec; it does not rule out a continuous macro model or finite precision matrix arithmetic.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [Finite U] (M : CGB.Medium U Λ A I),
  M.P1a →
    ∀ (k d : ℕ) (encode : CGB.GDNLife.State k d → U → Λ) (decode : (U → Λ) → CGB.GDNLife.State k d),
      ¬∀ (s : CGB.GDNLife.State k d), decode (encode s) = s
```

## CGB.LanguageExistence.finite_range_nonincreasing_eventually_constant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L48)

源码说明：A nonincreasing real sequence with finite range eventually attains a fixed value. No finiteness of the real number type is assumed.

形式陈述：

```lean
∀ (f : ℕ → ℝ), (Set.range f).Finite → (∀ (n : ℕ), f (n + 1) ≤ f n) → ∃ N, ∀ (n : ℕ), N ≤ n → f n = f N
```

## CGB.LanguageExistence.finite_body_stationary_loss_eventually_constant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L72)

源码说明：A fixed loss on a finite body cannot keep strictly improving while remaining nonincreasing at every step. This is a finite-body result, not a claim that arbitrary real GDN trajectories must freeze.

形式陈述：

```lean
∀ {State : Type u_1} [Finite State] (run : ℕ → State) (loss : State → ℝ),
  (∀ (n : ℕ), loss (run (n + 1)) ≤ loss (run n)) → ∃ N, ∀ (n : ℕ), N ≤ n → loss (run n) = loss (run N)
```

## CGB.LanguageExistence.recurrent_gain_excludes_eventual_stationary_descent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L83)

源码说明：Even *eventually* nonincreasing stationary loss is incompatible with arbitrarily late strict improvements on the same finite body. The theorem does not forbid world-conditioned relearning after drift.

形式陈述：

```lean
∀ {State : Type u_1} [Finite State] (run : ℕ → State) (loss : State → ℝ),
  (∀ (N : ℕ), ∃ n, N ≤ n ∧ loss (run (n + 1)) < loss (run n)) →
    ¬∃ N, ∀ (n : ℕ), N ≤ n → loss (run (n + 1)) ≤ loss (run n)
```

## CGB.LanguageExistence.strict_descent_has_no_finite_exact_readout

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L107)

源码说明：The obstruction also applies on one actual trajectory: a finite body cannot reproduce an endlessly strictly descending fixed real readout exactly. It does not require encoding every mathematically possible real state.

形式陈述：

```lean
∀ {State : Type u_1} {Physical : Type u_2} [Finite Physical] (run : ℕ → State) (loss : State → ℝ)
  (encode : State → Physical) (physicalLoss : Physical → ℝ),
  (∀ (n : ℕ), physicalLoss (encode (run n)) = loss (run n)) → (∀ (n : ℕ), loss (run (n + 1)) < loss (run n)) → False
```

## CGB.LanguageExistence.restricted_step_refines

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L132)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {State : Type u_1} {Input : Type u_2} (domain : Set State) (next : State → Input → State)
  (hclosed : ∀ s ∈ domain, ∀ (u : Input), next s u ∈ domain) (s : ↑domain) (u : Input),
  ↑(CGB.LanguageExistence.restrictedStep domain next hclosed s u) = next (↑s) u
```

## CGB.LanguageExistence.restricted_fold_refines

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L137)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {State : Type u_1} {Input : Type u_2} (domain : Set State) (next : State → Input → State)
  (hclosed : ∀ s ∈ domain, ∀ (u : Input), next s u ∈ domain) (s : ↑domain) (inputs : List Input),
  ↑(List.foldl (CGB.LanguageExistence.restrictedStep domain next hclosed) s inputs) = List.foldl next (↑s) inputs
```

## CGB.LanguageExistence.gdn_prior_preserves_bias

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L147)

源码说明：The physical-time prior never rewrites learned head bias. This is a property of the existing GDN split, not a newly installed protected register.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (elapsed : ℝ),
  (CGB.GDNLife.prior M s elapsed).bias = s.bias
```

## CGB.LanguageExistence.gdn_silent_continuation_bias

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L154)

源码说明：With no new arrived byte, any sequence of physical-time priors preserves the head bias. Real-time passage is not conflated with zero-time freezing.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (times : List ℝ),
  (List.foldl (CGB.GDNLife.prior M) s times).bias = s.bias
```

## CGB.LanguageExistence.gdn_alternative_arrival_bias

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L165)

源码说明：Alternative bytes arriving at the same prepared body install a head-bias difference determined by the actual local update. No teacher, extra history page, or externally written memory is supplied.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (p : CGB.GDNLife.Prepared k d) (a a' : CGB.GDNLife.Arrival)
  (b : CGB.GDNLife.Byte),
  (CGB.GDNLife.finish M p a).bias b - (CGB.GDNLife.finish M p a').bias b =
    M.headStep * ((if b = a.byte then 1 else 0) - if b = a'.byte then 1 else 0)
```

## CGB.LanguageExistence.gdn_written_distinction_survives_silence

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L183)

源码说明：A distinction written by different bytes survives arbitrarily many silent physical-time steps in this real GDN. This supplies a concrete memory mechanism, but does not identify it with the finite-medium CMI life clause.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d),
  M.headStep ≠ 0 →
    ∀ (p : CGB.GDNLife.Prepared k d) (a a' : CGB.GDNLife.Arrival),
      a.byte ≠ a'.byte →
        ∀ (times times' : List ℝ),
          (List.foldl (CGB.GDNLife.prior M) (CGB.GDNLife.finish M p a) times).bias a.byte -
                (List.foldl (CGB.GDNLife.prior M) (CGB.GDNLife.finish M p a') times').bias a.byte =
              M.headStep ∧
            List.foldl (CGB.GDNLife.prior M) (CGB.GDNLife.finish M p a) times ≠
              List.foldl (CGB.GDNLife.prior M) (CGB.GDNLife.finish M p a') times'
```

## CGB.LanguageExistence.precision_assimilation_is_two_state_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L204)

源码说明：The existing two-state population law gives exactly the actual GDN precision-weighted mean update, with up/down probabilities determined by the same local gain. This is not the mixed-key delta-write identity.

形式陈述：

```lean
∀ (p w m v : ℝ),
  0 < p → 0 ≤ w → (p * m + w * v) / (p + w) = CGB.Generation.twoStateStep (w / (p + w) * v) (w / (p + w) * (1 - v)) m
```

## CGB.LanguageExistence.precision_two_state_rates_valid

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L215)

源码说明：For a bounded occupancy target the two rates are valid probabilities; calm `v=1/2` balances them. A complete physical refinement must still expose where the exchanged particles and precision dynamics reside.

形式陈述：

```lean
∀ (p w v : ℝ),
  0 < p →
    0 ≤ w →
      0 ≤ v → v ≤ 1 → 0 ≤ w / (p + w) * v ∧ 0 ≤ w / (p + w) * (1 - v) ∧ w / (p + w) * v + w / (p + w) * (1 - v) < 1
```

## CGB.LanguageExistence.bounded_mean

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L228)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d),
  CGB.LanguageExistence.UnitMemory s →
    ∀ (j : Fin k) (i : Fin d), 0 ≤ CGB.GDNLife.mean M s j i ∧ CGB.GDNLife.mean M s j i ≤ 1
```

## CGB.LanguageExistence.prior_unit_memory

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L238)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d),
  CGB.GDNLife.Positive s →
    CGB.LanguageExistence.UnitMemory s →
      ∀ (elapsed : ℝ), 0 ≤ elapsed → CGB.LanguageExistence.UnitMemory (CGB.GDNLife.prior M s elapsed)
```

## CGB.LanguageExistence.assimilate_unit_memory

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L252)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d),
  CGB.LanguageExistence.UnitMemory s →
    ∀ (a : CGB.GDNLife.Arrival),
      (∀ (i : Fin d), 0 ≤ M.value a.byte i ∧ M.value a.byte i ≤ 1) →
        CGB.LanguageExistence.UnitMemory (CGB.GDNLife.assimilate M s a)
```

## CGB.LanguageExistence.step_unit_memory

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L267)

源码说明：Bounded byte embeddings imply bounded memory means through the *full* time-prior/assimilation/head step. No assumption of contraction or a finite real type is introduced. Head-parameter bounds remain a distinct obligation.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d),
  CGB.GDNLife.Positive s →
    CGB.LanguageExistence.UnitMemory s →
      ∀ (a : CGB.GDNLife.Arrival),
        (∀ (i : Fin d), 0 ≤ M.value a.byte i ∧ M.value a.byte i ≤ 1) →
          CGB.LanguageExistence.UnitMemory (CGB.GDNLife.step M s a)
```
