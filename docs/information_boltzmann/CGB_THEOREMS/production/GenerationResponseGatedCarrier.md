# GenerationResponseGatedCarrier

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean) · [返回导航](../README.md)

模块源码说明：Response-gated conserved contact carrier `GenerationContactEligibility` establishes that a saturated local H contact has zero forward response.

## CGB.Generation.response_gated_contact_closed_holds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L25)

解释：A response gate closed at the source event leaves every coordinate of the existing shared contact reservoir unchanged.

源码说明：A response gate closed at the source event leaves every coordinate of the existing shared contact reservoir unchanged.

形式陈述：

```lean
∀ {n : ℕ} (occupied target : Fin n → ℝ) (inflow release : ℝ),
  CGB.Generation.sharedContactPoolStep occupied target inflow release 0 = occupied
```

## CGB.Generation.response_gated_contact_write_distinguishes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L34)

解释：If opening the existing shared exchange produces a nonzero local transfer, the S coordinate distinguishes it from a saturated (closed-gate) event. This is exactly the missing condition for a delayed reader to retain the local response distinction; no separate eligibility page is added.

源码说明：If opening the existing shared exchange produces a nonzero local transfer, the S coordinate distinguishes it from a saturated (closed-gate) event. This is exactly the missing condition for a delayed reader to retain the local response distinction; no separate eligibility page is added.

形式陈述：

```lean
∀ {n : ℕ} (occupied target : Fin n → ℝ) (inflow release : ℝ) (i : Fin n),
  inflow * target i * (1 - ∑ j, occupied j) - release * occupied i ≠ 0 →
    CGB.Generation.sharedContactPoolStep occupied target inflow release 0 i ≠
      CGB.Generation.sharedContactPoolStep occupied target inflow release 1 i
```

## CGB.Generation.shared_contact_pool_vector_affine

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L58)

解释：Exact connection of the retention/injection decomposition to the executable coordinatewise-release shared reservoir.

源码说明：Exact connection of the retention/injection decomposition to the executable coordinatewise-release shared reservoir.

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow gate : ℝ) (i : Fin n),
  CGB.Generation.sharedContactPoolVectorStep occupied target release inflow gate i =
    CGB.Generation.contactRetention gate (release i) * occupied i +
      CGB.Generation.sharedContactInjection occupied target inflow gate i
```

## CGB.Generation.contact_retention_bounds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L66)

解释：输入contact_retention_bounds形式化前提参量，经底层算子变换与代数化简，输出形式陈述所保证的严格等式或不变性结论。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (gate release : ℝ),
  0 ≤ gate ∧ gate ≤ 1 →
    0 ≤ release ∧ release ≤ 1 →
      0 ≤ CGB.Generation.contactRetention gate release ∧ CGB.Generation.contactRetention gate release ≤ 1
```

## _private.CGBAxioms.GenerationResponseGatedCarrier.0.CGB.Generation.affine_forward_history

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L78)

解释：输入affine_forward_history形式化前提参量，经底层算子变换与代数化简，输出形式陈述所保证的严格等式或不变性结论。

源码说明：Algebraic forward expansion only. Public results below instantiate both coefficients with the actual vector shared-pool transition.

形式陈述：

```lean
∀ (state retention injection : ℕ → ℝ),
  (∀ (t : ℕ), state (t + 1) = retention t * state t + injection t) →
    ∀ (T : ℕ),
      state T =
        state 0 * ∏ t ∈ Finset.range T, retention t +
          ∑ k ∈ Finset.range T, injection k * ∏ t ∈ Finset.Ico (k + 1) T, retention t
```

## CGB.Generation.response_gated_vector_history_expansion

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L102)

解释：Every actual coordinate is its retained initial value plus its finitely many injections, each weighted by subsequent retention. The formula does not require the runtime to save or replay those injections.

源码说明：Every actual coordinate is its retained initial value plus its finitely many injections, each weighted by subsequent retention. The formula does not require the runtime to save or replay those injections.

形式陈述：

```lean
∀ {n : ℕ} (state target release : ℕ → Fin n → ℝ) (inflow gate : ℕ → ℝ),
  (∀ (t : ℕ),
      state (t + 1) = CGB.Generation.sharedContactPoolVectorStep (state t) (target t) (release t) (inflow t) (gate t)) →
    ∀ (T : ℕ) (i : Fin n),
      state T i =
        state 0 i * ∏ t ∈ Finset.range T, CGB.Generation.contactRetention (gate t) (release t i) +
          ∑ k ∈ Finset.range T,
            CGB.Generation.sharedContactInjection (state k) (target k) (inflow k) (gate k) i *
              ∏ t ∈ Finset.Ico (k + 1) T, CGB.Generation.contactRetention (gate t) (release t i)
```

## CGB.Generation.response_gated_vector_no_injection_retention

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L118)

解释：With no new injection into this type during the interval, only the product of its actual coordinatewise retention acts on its initial content.

源码说明：With no new injection into this type during the interval, only the product of its actual coordinatewise retention acts on its initial content.

形式陈述：

```lean
∀ {n : ℕ} (state target release : ℕ → Fin n → ℝ) (inflow gate : ℕ → ℝ),
  (∀ (t : ℕ),
      state (t + 1) = CGB.Generation.sharedContactPoolVectorStep (state t) (target t) (release t) (inflow t) (gate t)) →
    ∀ (T : ℕ) (i : Fin n),
      (∀ t < T, CGB.Generation.sharedContactInjection (state t) (target t) (inflow t) (gate t) i = 0) →
        state T i = state 0 i * ∏ t ∈ Finset.range T, CGB.Generation.contactRetention (gate t) (release t i)
```

## CGB.Generation.response_gated_vector_zero_without_injection

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L135)

解释：A type absent at birth cannot appear without an injection of that type.

源码说明：A type absent at birth cannot appear without an injection of that type.

形式陈述：

```lean
∀ {n : ℕ} (state target release : ℕ → Fin n → ℝ) (inflow gate : ℕ → ℝ),
  (∀ (t : ℕ),
      state (t + 1) = CGB.Generation.sharedContactPoolVectorStep (state t) (target t) (release t) (inflow t) (gate t)) →
    ∀ (T : ℕ) (i : Fin n),
      state 0 i = 0 →
        (∀ t < T, CGB.Generation.sharedContactInjection (state t) (target t) (inflow t) (gate t) i = 0) → state T i = 0
```

## CGB.Generation.response_gated_vector_history_difference

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L148)

解释：Equal external inputs alone do not guarantee equal injections: the free pool depends on each trajectory. This result explicitly requires equal actual injections as well as the shared retention schedule.

源码说明：Equal external inputs alone do not guarantee equal injections: the free pool depends on each trajectory. This result explicitly requires equal actual injections as well as the shared retention schedule.

形式陈述：

```lean
∀ {n : ℕ} (left right target release : ℕ → Fin n → ℝ) (inflow gate : ℕ → ℝ),
  (∀ (t : ℕ),
      left (t + 1) = CGB.Generation.sharedContactPoolVectorStep (left t) (target t) (release t) (inflow t) (gate t)) →
    (∀ (t : ℕ),
        right (t + 1) =
          CGB.Generation.sharedContactPoolVectorStep (right t) (target t) (release t) (inflow t) (gate t)) →
      ∀ (T : ℕ) (i : Fin n),
        (∀ t < T,
            CGB.Generation.sharedContactInjection (left t) (target t) (inflow t) (gate t) i =
              CGB.Generation.sharedContactInjection (right t) (target t) (inflow t) (gate t) i) →
          left T i - right T i =
            (left 0 i - right 0 i) * ∏ t ∈ Finset.range T, CGB.Generation.contactRetention (gate t) (release t i)
```

## CGB.Generation.response_gated_vector_nonzero_retention_distinguishes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L173)

解释：Nonzero retained product preserves an initial scalar distinction, not a claim that the complete old pre×gate vector can be decoded.

源码说明：Nonzero retained product preserves an initial scalar distinction, not a claim that the complete old pre×gate vector can be decoded.

形式陈述：

```lean
∀ {n : ℕ} (left right target release : ℕ → Fin n → ℝ) (inflow gate : ℕ → ℝ),
  (∀ (t : ℕ),
      left (t + 1) = CGB.Generation.sharedContactPoolVectorStep (left t) (target t) (release t) (inflow t) (gate t)) →
    (∀ (t : ℕ),
        right (t + 1) =
          CGB.Generation.sharedContactPoolVectorStep (right t) (target t) (release t) (inflow t) (gate t)) →
      ∀ (T : ℕ) (i : Fin n),
        (∀ t < T,
            CGB.Generation.sharedContactInjection (left t) (target t) (inflow t) (gate t) i =
              CGB.Generation.sharedContactInjection (right t) (target t) (inflow t) (gate t) i) →
          left 0 i ≠ right 0 i →
            ∏ t ∈ Finset.range T, CGB.Generation.contactRetention (gate t) (release t i) ≠ 0 → left T i ≠ right T i
```

## CGB.Generation.response_gated_vector_source_difference

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L193)

解释：Equal rank-release rates give a direct source-contrast recurrence inside one and the same reservoir; the common free pool is retained in the formula.

源码说明：Equal rank-release rates give a direct source-contrast recurrence inside one and the same reservoir; the common free pool is retained in the formula.

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow gate : ℝ) (i j : Fin n),
  release i = release j →
    CGB.Generation.sharedContactPoolVectorStep occupied target release inflow gate i -
        CGB.Generation.sharedContactPoolVectorStep occupied target release inflow gate j =
      CGB.Generation.contactRetention gate (release i) * (occupied i - occupied j) +
        gate * inflow * (target i - target j) * (1 - ∑ k, occupied k)
```

## CGB.Generation.positive_demand_preserves_contact_distinction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L204)

解释：输入positive_demand_preserves_contact_distinction形式化前提参量，经底层算子变换与代数化简，输出形式陈述所保证的严格等式或不变性结论。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (left right demand : ℝ), 0 < demand → left ≠ right → demand * left ≠ demand * right
```
