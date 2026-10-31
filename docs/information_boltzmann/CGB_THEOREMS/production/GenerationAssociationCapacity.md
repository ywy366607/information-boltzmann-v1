# GenerationAssociationCapacity

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean) · [返回导航](../README.md)

模块源码说明：Exact and mixed capacity of an associative linear state A GDN fast-weight state is a superposition of value-weighted keys.

## CGB.Generation.associationState_row

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L29)

源码说明：Each output row of `V Kᵀ` is one scalar key superposition.

形式陈述：

```lean
∀ {dv dk n : ℕ} (k : Fin n → Fin dk → ℝ) (v : Fin n → Fin dv → ℝ) (o : Fin dv),
  CGB.Generation.associationState k v o = CGB.Generation.keySuperposition k fun i => v i o
```

## CGB.Generation.association_slot_retention

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L38)

源码说明：Slotwise natural-state retention transports each stored key by the same slot factors. It applies to arbitrarily many prior associations at once; the list of keys is a mathematical decomposition, not required runtime state.

形式陈述：

```lean
∀ {dv dk n : ℕ} (k : Fin n → Fin dk → ℝ) (v : Fin n → Fin dv → ℝ) (retention : Fin dk → ℝ),
  (fun o j => retention j * CGB.Generation.associationState k v o j) =
    CGB.Generation.associationState (fun i j => retention j * k i j) v
```

## CGB.Generation.association_state_readout

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L56)

源码说明：Reading `V Kᵀ` with a query is the corresponding weighted combination of values; this is the algebraic reference for GDN probe readout.

形式陈述：

```lean
∀ {dv dk n : ℕ} (k : Fin n → Fin dk → ℝ) (v : Fin n → Fin dv → ℝ) (q : Fin dk → ℝ),
  CGB.Generation.associationRead (CGB.Generation.associationState k v) q = fun o => ∑ i, v i o * ∑ j, k i j * q j
```

## CGB.Generation.association_readout_interference_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L74)

源码说明：A single interference matrix bounds the readout error uniformly over all bounded values. Keys and queries are the actual vectors presented to the associative state; the bound is a mathematical certificate, not stored history.

形式陈述：

```lean
∀ {dv dk n : ℕ} (k q : Fin n → Fin dk → ℝ) (v : Fin n → Fin dv → ℝ),
  (∀ (i : Fin n) (o : Fin dv), |v i o| ≤ 1) →
    ∀ (i : Fin n) (o : Fin dv),
      |CGB.Generation.associationRead (CGB.Generation.associationState k v) (q i) o - v i o| ≤
        ∑ j, |∑ t, k j t * q i t - if j = i then 1 else 0|
```

## CGB.Generation.probeFamilyAccurate_mono

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L111)

源码说明：Accuracy is monotone in the declared tolerance.

形式陈述：

```lean
∀ {dv dk m : ℕ} (state : Fin dv → Fin dk → ℝ) (keys : Fin m → Fin dk → ℝ) (targets : Fin m → Fin dv → ℝ) {ε ε' : ℝ},
  CGB.Generation.probeFamilyAccurate state keys targets ε →
    ε ≤ ε' → CGB.Generation.probeFamilyAccurate state keys targets ε'
```

## CGB.Generation.exact_decoder_capacity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L127)

源码说明：Exact decoding itself forces the same key-dimension bound.

形式陈述：

```lean
∀ (d n : ℕ) (k : Fin n → Fin d → ℝ), CGB.Generation.ExactDecoder k → n ≤ d
```

## CGB.Generation.exact_decoder_association_readout

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L158)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {dv dk n : ℕ} (k : Fin n → Fin dk → ℝ) (v : Fin n → Fin dv → ℝ) (D : (Fin dk → ℝ) →ₗ[ℝ] Fin n → ℝ),
  (∀ (a : Fin n → ℝ), D (CGB.Generation.keySuperposition k a) = a) →
    ∀ (o : Fin dv), D (CGB.Generation.associationState k v o) = fun i => v i o
```

## CGB.Generation.bounded_decoder_association_readout

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L167)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {dv dk n : ℕ} (k : Fin n → Fin dk → ℝ) (v : Fin n → Fin dv → ℝ) {ε : ℝ},
  CGB.Generation.BoundedEpsilonDecoder k ε →
    (∀ (i : Fin n) (o : Fin dv), |v i o| ≤ 1) →
      ∃ D, ∀ (o : Fin dv) (i : Fin n), |D (CGB.Generation.associationState k v o) i - v i o| ≤ ε
```

## CGB.Generation.normalized_assimilation_decoder_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L185)

源码说明：A weighted natural-parameter write followed by slotwise normalization is an associative superposition with effective keys `wᵢ kᵢⱼ / pⱼ`, plus the normalized old state. If the existing bounded decoder handles those keys and its decoded old-state contribution is bounded by `ρ`, the full decoded error is at most `ε + ρ`. The event family is a proof-side decomposition, not an additional persistent history.

形式陈述：

```lean
∀ {dv dk n : ℕ} (old : Fin dv → Fin dk → ℝ) (k : Fin n → Fin dk → ℝ) (v : Fin n → Fin dv → ℝ) (w : Fin n → ℝ)
  (p : Fin dk → ℝ) (ε ρ : ℝ),
  (∀ (i : Fin n) (o : Fin dv), |v i o| ≤ 1) →
    (∃ D,
        (∀ (a : Fin n → ℝ),
            (∀ (i : Fin n), |a i| ≤ 1) →
              ∀ (i : Fin n), |D (CGB.Generation.keySuperposition (fun i j => w i * k i j / p j) a) i - a i| ≤ ε) ∧
          ∀ (o : Fin dv) (i : Fin n), |D (fun j => old o j / p j) i| ≤ ρ) →
      ∃ D, ∀ (o : Fin dv) (i : Fin n), |D (fun j => (old o j + ∑ t, w t * v t o * k t j) / p j) i - v i o| ≤ ε + ρ
```

## CGB.Generation.exact_decoder_is_bounded_decoder

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L224)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {d n : ℕ} (k : Fin n → Fin d → ℝ), CGB.Generation.ExactDecoder k → CGB.Generation.BoundedEpsilonDecoder k 0
```

## CGB.Generation.precision_assimilation_is_innovation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L237)

源码说明：A natural-parameter assimilation is an innovation update, with gain `w / (p + w)` for the individual slot. This is a slotwise identity; it does not assert equivalence with a key-directed delta-rule update.

形式陈述：

```lean
∀ (p w m y : ℝ), 0 < p → 0 ≤ w → (p * m + w * y) / (p + w) = m + w / (p + w) * (y - m)
```

## CGB.Generation.precision_assimilation_gain_bounds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L246)

源码说明：The precision-derived slot gain is bounded when prior precision is positive and incoming precision is nonnegative.

形式陈述：

```lean
∀ (p w : ℝ), 0 < p → 0 ≤ w → 0 ≤ w / (p + w) ∧ w / (p + w) < 1
```

## CGB.Generation.process_noise_gain_floor

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L258)

源码说明：With strictly positive process variance and a lower bound on incoming precision, one predict-then-assimilate step has a positive gain floor. The bound is conditional on applying the process step before the observation.

形式陈述：

```lean
∀ (p q w wmin : ℝ), 0 < p → 0 < q → 0 < wmin → wmin ≤ w → wmin / (1 / q + wmin) ≤ w / (p / (1 + q * p) + w)
```

## CGB.Generation.exact_key_capacity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L271)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (d n : ℕ) (k : Fin n → Fin d → ℝ), LinearIndependent ℝ k → n ≤ d
```

## CGB.Generation.collision_of_excess_keys

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L278)

源码说明：Beyond key dimension, linear superposition is not injective: two distinct coefficient assignments produce the same stored state. This is the formal collision statement behind mixed storage in an associative GDN matrix.

形式陈述：

```lean
∀ (d n : ℕ),
  d < n →
    ∀ (k : Fin n → Fin d → ℝ),
      ∃ f g, CGB.Generation.keySuperposition k f = CGB.Generation.keySuperposition k g ∧ ∃ i, f i ≠ g i
```
