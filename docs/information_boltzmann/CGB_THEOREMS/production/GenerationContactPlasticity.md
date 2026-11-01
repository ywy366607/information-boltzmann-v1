# GenerationContactPlasticity

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean) · [返回导航](../README.md)

模块源码说明：Compartment tags modulate a current-event spatial pullback The coefficient `c = lambda * eligibility` is fixed during this event's parameter differentiation.

## CGB.Generation.ContactPlasticity.linear_contact_return

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L25)

解释：Transposing a finite linear contact read returns exactly the same scalar contraction; the return is not an independently learned feedback matrix.

源码说明：Transposing a finite linear contact read returns exactly the same scalar contraction; the return is not an independently learned feedback matrix.

形式陈述：

```lean
∀ {I : Type u_1} {J : Type u_2} [inst : Fintype I] [inst_1 : Fintype J] (post : I → ℝ) (contact : I → J → ℝ)
  (pre : J → ℝ), ∑ i, post i * ∑ j, contact i j * pre j = ∑ j, pre j * ∑ i, contact i j * post i
```

## CGB.Generation.ContactPlasticity.periodic_flux_return

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L39)

解释：Incoming-minus-outgoing periodic flux has the next-minus-local adjoint. `next.symm` is the predecessor, so this covers any finite periodic geometry.

源码说明：Incoming-minus-outgoing periodic flux has the next-minus-local adjoint. `next.symm` is the predecessor, so this covers any finite periodic geometry.

形式陈述：

```lean
∀ {I : Type u_1} [inst : Fintype I] (next : Equiv.Perm I) (v flow : I → ℝ),
  ∑ i, v i * (flow ((Equiv.symm next) i) - flow i) = ∑ i, (v (next i) - v i) * flow i
```

## CGB.Generation.ContactPlasticity.two_direction_flux_return

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L53)

解释：Two transport directions add their local adjoints without retaining a transport trajectory or its past pullbacks.

源码说明：Two transport directions add their local adjoints without retaining a transport trajectory or its past pullbacks.

形式陈述：

```lean
∀ {I : Type u_1} [inst : Fintype I] (p q : Equiv.Perm I) (v f g : I → ℝ),
  ∑ i, v i * (f ((Equiv.symm p) i) - f i + (g ((Equiv.symm q) i) - g i)) =
    ∑ i, ((v (p i) - v i) * f i + (v (q i) - v i) * g i)
```

## CGB.Generation.ContactPlasticity.allocation_normalization_cancels

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L63)

解释：Positive total drive cancels between admission and normalized target. No assumption on the capacity divisor is needed for this algebraic identity.

源码说明：Positive total drive cancels between admission and normalized target. No assumption on the capacity divisor is needed for this algebraic identity.

形式陈述：

```lean
∀ (active totalDrive capacity drive : ℝ),
  totalDrive ≠ 0 → active * (totalDrive / capacity) * (drive / totalDrive) = active * drive / capacity
```

## CGB.Generation.ContactPlasticity.normalized_actual_pool

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L73)

解释：The cancellation holds inside the actual shared reservoir, with its old occupation and coordinatewise release unchanged.

源码说明：The cancellation holds inside the actual shared reservoir, with its old occupation and coordinatewise release unchanged.

形式陈述：

```lean
∀ {n : ℕ} (occupied drive release : Fin n → ℝ) (active capacity gate : ℝ),
  ∑ j, drive j ≠ 0 →
    ∀ (i : Fin n),
      CGB.Generation.sharedContactPoolVectorStep occupied (fun j => drive j / ∑ k, drive k) release
          (active * ((∑ j, drive j) / capacity)) gate i =
        occupied i + gate * (active * drive i / capacity) * (1 - ∑ j, occupied j) - gate * release i * occupied i
```

## CGB.Generation.ContactPlasticity.sigmoid_runtime_formula

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L96)

解释：Mathlib's sigmoid is exactly the runtime logistic expression.

源码说明：Mathlib's sigmoid is exactly the runtime logistic expression.

形式陈述：

```lean
∀ (x : ℝ), x.sigmoid = 1 / (1 + Real.exp (-x))
```

## CGB.Generation.ContactPlasticity.endpoint_allocation_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L102)

解释：Exact allocation response to one endpoint's local pre-weight perturbation; `pre` is fixed and the sigmoid supplies the current post-response factor.

源码说明：Exact allocation response to one endpoint's local pre-weight perturbation; `pre` is fixed and the sigmoid supplies the current post-response factor.

形式陈述：

```lean
∀ (left right pre : ℝ),
  HasDerivAt (fun θ => CGB.Generation.ContactPlasticity.endpointAllocation (left + θ * pre) right)
    (((left + right) / 2).sigmoid * (1 - ((left + right) / 2).sigmoid) * (pre / 2)) 0
```

## CGB.Generation.ContactPlasticity.endpoint_retention_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L113)

解释：Exact mean-retention response at one perturbed endpoint. Unlike allocation, the sigmoid is applied before the two endpoints are averaged.

源码说明：Exact mean-retention response at one perturbed endpoint. Unlike allocation, the sigmoid is applied before the two endpoints are averaged.

形式陈述：

```lean
∀ (left right pre : ℝ),
  HasDerivAt (fun θ => CGB.Generation.ContactPlasticity.endpointRetention (left + θ * pre) right)
    (left.sigmoid * (1 - left.sigmoid) * pre / 2) 0
```

## CGB.Generation.ContactPlasticity.local_potential_zero_eligibility

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L131)

解释：Vanishing participation tags make the entire parameter-dependent proxy identically zero, not merely its value at a chosen parameter.

源码说明：Vanishing participation tags make the entire parameter-dependent proxy identically zero, not merely its value at a chosen parameter.

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow gate : ℝ) (lambda eligibility : Fin n → ℝ),
  (∀ (i : Fin n), eligibility i = 0) →
    CGB.Generation.ContactPlasticity.localPotential occupied target release inflow gate lambda eligibility = 0
```

## CGB.Generation.ContactPlasticity.local_potential_zero_return

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L137)

解释：A zero current local return also makes the stopped-coefficient proxy zero.

源码说明：A zero current local return also makes the stopped-coefficient proxy zero.

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow gate : ℝ) (lambda eligibility : Fin n → ℝ),
  (∀ (i : Fin n), lambda i = 0) →
    CGB.Generation.ContactPlasticity.localPotential occupied target release inflow gate lambda eligibility = 0
```

## CGB.Generation.ContactPlasticity.local_potential_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L145)

解释：Exact current-event derivative through the actual inflow and coordinatewise release. Old S, target, gate, lambda and eligibility are constant here; this is not the derivative of a state trajectory or of the current language CE.

源码说明：Exact current-event derivative through the actual inflow and coordinatewise release. Old S, target, gate, lambda and eligibility are constant here; this is not the derivative of a state trajectory or of the current language CE.

形式陈述：

```lean
∀ {n : ℕ} (occupied target lambda eligibility : Fin n → ℝ) (inflow : ℝ → ℝ) (release : ℝ → Fin n → ℝ)
  (gate x dinflow : ℝ) (drelease : Fin n → ℝ),
  HasDerivAt inflow dinflow x →
    (∀ (i : Fin n), HasDerivAt (fun θ => release θ i) (drelease i) x) →
      HasDerivAt
        (fun θ =>
          CGB.Generation.ContactPlasticity.localPotential occupied target (release θ) (inflow θ) gate lambda
            eligibility)
        (∑ i,
          lambda i * eligibility i *
            (gate * target i * (1 - ∑ j, occupied j) * dinflow - gate * occupied i * drelease i))
        x
```

## CGB.Generation.ContactPlasticity.allocation_pool_pre_post_coefficient

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L167)

解释：Actual normalized allocation, allowing target and total admission to vary together. `capacity` is rank K, not the number of source-expanded coordinates. Only the selected source's drive is nonzero in the runtime instantiation.

源码说明：Actual normalized allocation, allowing target and total admission to vary together. `capacity` is rank K, not the number of source-expanded coordinates. Only the selected source's drive is nonzero in the runtime instantiation.

形式陈述：

```lean
∀ {n : ℕ} (occupied release : Fin n → ℝ) (drive : ℝ → Fin n → ℝ) (active capacity gate left right pre c : ℝ)
  (i : Fin n),
  (∀ (θ : ℝ), ∑ j, drive θ j ≠ 0) →
    (∀ (θ : ℝ), drive θ i = CGB.Generation.ContactPlasticity.endpointAllocation (left + θ * pre) right) →
      HasDerivAt
        (fun θ =>
          c *
            CGB.Generation.sharedContactPoolVectorStep occupied (fun j => drive θ j / ∑ k, drive θ k) release
              (active * ((∑ j, drive θ j) / capacity)) gate i)
        (pre * (((left + right) / 2).sigmoid * (1 - ((left + right) / 2).sigmoid) / 2) *
          (c * gate * active * (1 - ∑ j, occupied j) / capacity))
        0
```

## CGB.Generation.ContactPlasticity.retention_pool_pre_post_coefficient

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L201)

解释：The actual release is A*(1-retention), so a retention logit's response has a positive sign and is contracted with old S across all source coordinates. Equal per-rank endpoint logits across sources specialize this sum directly.

源码说明：The actual release is A*(1-retention), so a retention logit's response has a positive sign and is contracted with old S across all source coordinates. Equal per-rank endpoint logits across sources specialize this sum directly.

形式陈述：

```lean
∀ {n : ℕ} (occupied target lambda eligibility left right pre : Fin n → ℝ) (inflow gate active : ℝ),
  HasDerivAt
    (fun θ =>
      CGB.Generation.ContactPlasticity.localPotential occupied target
        (fun i => active * (1 - CGB.Generation.ContactPlasticity.endpointRetention (left i + θ * pre i) (right i)))
        inflow gate lambda eligibility)
    (∑ i,
      pre i * ((left i).sigmoid * (1 - (left i).sigmoid) / 2) * (lambda i * eligibility i * gate * active * occupied i))
    0
```

## CGB.Generation.ContactPlasticity.local_direction_self_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L225)

解释：A parameter direction contracted with itself is nonnegative. This concerns only the local proxy's own direction, not descent of a different CE objective.

源码说明：A parameter direction contracted with itself is nonnegative. This concerns only the local proxy's own direction, not descent of a different CE objective.

形式陈述：

```lean
∀ {P : Type u_1} [inst : Fintype P] (direction : P → ℝ), 0 ≤ ∑ p, direction p * direction p
```
