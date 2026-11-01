# GenerationContactCompetition

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean) · [返回导航](../README.md)

模块源码说明：A shared finite contact reservoir `S` is already a bounded, typed local contact state.

## CGB.Generation.receipt_rate_in_unit_interval

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L31)

解释：A nonnegative receipt produces a legal exchange rate, even when its raw magnitude is unbounded. This is the precise range premise needed to apply the shared-pool conservation and nonnegativity laws to the executable ``-expm1(-x)`` conversion.

源码说明：A nonnegative receipt produces a legal exchange rate, even when its raw magnitude is unbounded. This is the precise range premise needed to apply the shared-pool conservation and nonnegativity laws to the executable ``-expm1(-x)`` conversion.

形式陈述：

```lean
∀ (x : ℝ), 0 ≤ x → 0 ≤ CGB.Generation.receiptRate x ∧ CGB.Generation.receiptRate x ≤ 1
```

## CGB.Generation.shared_contact_pool_total

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L49)

解释：The whole contact bundle follows one two-state exchange. Coordinate shares may differ, but their sum can draw only from the same implicit pool.

源码说明：The whole contact bundle follows one two-state exchange. Coordinate shares may differ, but their sum can draw only from the same implicit pool.

形式陈述：

```lean
∀ {n : ℕ} (occupied target : Fin n → ℝ) (inflow release gate : ℝ),
  ∑ i, target i = 1 →
    ∑ i, CGB.Generation.sharedContactPoolStep occupied target inflow release gate i =
      ∑ i, occupied i + gate * (inflow * (1 - ∑ i, occupied i) - release * ∑ i, occupied i)
```

## CGB.Generation.full_shared_pool_blocks_admission

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L77)

解释：Once a local bundle is full, no positive admission can add to any of its contacts. Only its release channel can change it.

源码说明：Once a local bundle is full, no positive admission can add to any of its contacts. Only its release channel can change it.

形式陈述：

```lean
∀ {n : ℕ} (occupied target : Fin n → ℝ) (inflow release gate : ℝ),
  ∑ j, occupied j = 1 →
    ∀ (i : Fin n),
      CGB.Generation.sharedContactPoolStep occupied target inflow release gate i =
        occupied i - gate * release * occupied i
```

## CGB.Generation.shared_pool_marginal_admission_is_anti_dominant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L89)

解释：For a positive local admission gate, an already more occupied bundle has strictly less remaining marginal capacity. This is the local anti-dominance law required to prevent unlimited winner growth; it does not prescribe a global balancing scalar.

源码说明：For a positive local admission gate, an already more occupied bundle has strictly less remaining marginal capacity. This is the local anti-dominance law required to prevent unlimited winner growth; it does not prescribe a global balancing scalar.

形式陈述：

```lean
∀ (gate inflow target q₁ q₂ : ℝ),
  0 < gate → 0 < inflow → 0 < target → q₁ < q₂ → gate * inflow * target * (1 - q₂) < gate * inflow * target * (1 - q₁)
```

## CGB.Generation.shared_pool_closed_gate_holds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L100)

解释：When a local evidence gate is closed, the complete bundle holds exactly. This lets delayed evidence regulate the existing S carrier without advancing a separate trace or a global learning-rate state.

源码说明：When a local evidence gate is closed, the complete bundle holds exactly. This lets delayed evidence regulate the existing S carrier without advancing a separate trace or a global learning-rate state.

形式陈述：

```lean
∀ {n : ℕ} (occupied target : Fin n → ℝ) (inflow release : ℝ),
  CGB.Generation.sharedContactPoolStep occupied target inflow release 0 = occupied
```

## CGB.Generation.shared_contact_pool_vector_constant_release

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L114)

解释：Constant vector release recovers the original interface exactly.

源码说明：Constant vector release recovers the original interface exactly.

形式陈述：

```lean
∀ {n : ℕ} (occupied target : Fin n → ℝ) (inflow release gate : ℝ),
  CGB.Generation.sharedContactPoolVectorStep occupied target (fun x => release) inflow gate =
    CGB.Generation.sharedContactPoolStep occupied target inflow release gate
```

## CGB.Generation.shared_contact_pool_vector_total

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L122)

解释：输入shared_contact_pool_vector_total形式化前提参量，经底层算子变换与代数化简，输出形式陈述所保证的严格等式或不变性结论。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow gate : ℝ),
  ∑ i, target i = 1 →
    ∑ i, CGB.Generation.sharedContactPoolVectorStep occupied target release inflow gate i =
      ∑ i, occupied i + gate * inflow * (1 - ∑ i, occupied i) - gate * ∑ i, release i * occupied i
```

## CGB.Generation.shared_contact_pool_vector_conserves_total

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L152)

解释：输入shared_contact_pool_vector_conserves_total形式化前提参量，经底层算子变换与代数化简，输出形式陈述所保证的严格等式或不变性结论。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow gate : ℝ),
  ∑ i, target i = 1 →
    ∑ i, CGB.Generation.sharedContactPoolVectorStep occupied target release inflow gate i +
        CGB.Generation.sharedContactPoolFreeStep occupied release inflow gate =
      1
```

## CGB.Generation.shared_contact_pool_vector_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L161)

解释：输入shared_contact_pool_vector_nonneg形式化前提参量，经底层算子变换与代数化简，输出形式陈述所保证的严格等式或不变性结论。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow gate : ℝ),
  (∀ (i : Fin n), 0 ≤ occupied i) →
    ∑ i, occupied i ≤ 1 →
      (∀ (i : Fin n), 0 ≤ target i) →
        (∀ (i : Fin n), 0 ≤ release i ∧ release i ≤ 1) →
          0 ≤ inflow →
            0 ≤ gate ∧ gate ≤ 1 →
              ∀ (i : Fin n), 0 ≤ CGB.Generation.sharedContactPoolVectorStep occupied target release inflow gate i
```

## CGB.Generation.shared_contact_pool_free_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L179)

解释：输入shared_contact_pool_free_nonneg形式化前提参量，经底层算子变换与代数化简，输出形式陈述所保证的严格等式或不变性结论。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (occupied release : Fin n → ℝ) (inflow gate : ℝ),
  (∀ (i : Fin n), 0 ≤ occupied i) →
    ∑ i, occupied i ≤ 1 →
      (∀ (i : Fin n), 0 ≤ release i) →
        0 ≤ inflow ∧ inflow ≤ 1 →
          0 ≤ gate ∧ gate ≤ 1 → 0 ≤ CGB.Generation.sharedContactPoolFreeStep occupied release inflow gate
```

## CGB.Generation.shared_contact_pool_vector_total_le_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L191)

解释：输入shared_contact_pool_vector_total_le_one形式化前提参量，经底层算子变换与代数化简，输出形式陈述所保证的严格等式或不变性结论。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow gate : ℝ),
  (∀ (i : Fin n), 0 ≤ occupied i) →
    ∑ i, occupied i ≤ 1 →
      ∑ i, target i = 1 →
        (∀ (i : Fin n), 0 ≤ release i) →
          0 ≤ inflow ∧ inflow ≤ 1 →
            0 ≤ gate ∧ gate ≤ 1 →
              ∑ i, CGB.Generation.sharedContactPoolVectorStep occupied target release inflow gate i ≤ 1
```

## CGB.Generation.full_shared_pool_vector_blocks_admission

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L203)

解释：输入full_shared_pool_vector_blocks_admission形式化前提参量，经底层算子变换与代数化简，输出形式陈述所保证的严格等式或不变性结论。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow gate : ℝ),
  ∑ j, occupied j = 1 →
    ∀ (i : Fin n),
      CGB.Generation.sharedContactPoolVectorStep occupied target release inflow gate i =
        (1 - gate * release i) * occupied i
```

## CGB.Generation.shared_pool_vector_closed_gate_holds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L212)

解释：输入shared_pool_vector_closed_gate_holds形式化前提参量，经底层算子变换与代数化简，输出形式陈述所保证的严格等式或不变性结论。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow : ℝ),
  CGB.Generation.sharedContactPoolVectorStep occupied target release inflow 0 = occupied
```

## CGB.Generation.shared_contact_pool_vector_endpoint_agreement

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L220)

解释：Endpoints applying the same inputs to the same shared state cannot write inconsistent copies of that edge's reservoir.

源码说明：Endpoints applying the same inputs to the same shared state cannot write inconsistent copies of that edge's reservoir.

形式陈述：

```lean
∀ {n : ℕ} (left right target release : Fin n → ℝ) (inflow gate : ℝ),
  left = right →
    CGB.Generation.sharedContactPoolVectorStep left target release inflow gate =
      CGB.Generation.sharedContactPoolVectorStep right target release inflow gate
```

## CGB.Generation.shared_edge_response_gate_symmetric

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L229)

解释：输入shared_edge_response_gate_symmetric形式化前提参量，经底层算子变换与代数化简，输出形式陈述所保证的严格等式或不变性结论。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (left right : ℝ), CGB.Generation.sharedEdgeResponseGate left right = CGB.Generation.sharedEdgeResponseGate right left
```

## CGB.Generation.shared_edge_response_gate_bounds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L233)

解释：输入shared_edge_response_gate_bounds形式化前提参量，经底层算子变换与代数化简，输出形式陈述所保证的严格等式或不变性结论。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (left right : ℝ),
  0 ≤ left ∧ left ≤ 1 →
    0 ≤ right ∧ right ≤ 1 →
      0 ≤ CGB.Generation.sharedEdgeResponseGate left right ∧ CGB.Generation.sharedEdgeResponseGate left right ≤ 1
```

## CGB.Generation.shared_edge_response_gate_closed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L238)

解释：输入shared_edge_response_gate_closed形式化前提参量，经底层算子变换与代数化简，输出形式陈述所保证的严格等式或不变性结论。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Generation.sharedEdgeResponseGate 0 0 = 0
```

## CGB.Generation.shared_edge_response_gate_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L241)

解释：输入shared_edge_response_gate_one形式化前提参量，经底层算子变换与代数化简，输出形式陈述所保证的严格等式或不变性结论。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ right ≤ 1, CGB.Generation.sharedEdgeResponseGate 1 right = 1
```
