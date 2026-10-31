# GenerationGDNLife

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean) · [返回导航](../README.md)

模块源码说明：Real-valued GDN carrier on one open trajectory The body is a fixed-dimensional real state, not a finite enumeration.

## CGB.GDNLife.address_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L73)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (b : CGB.GDNLife.Byte) (j : Fin k), 0 < CGB.GDNLife.address M b j
```

## CGB.GDNLife.action_head_score_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L107)

源码说明：The existing G14 log-softmax law instantiated at the GDN language head. Thus the first two factors of delayed action credit are fixed by the pre-action body and emitted byte; only their persistence until an external outcome is a separate causal-carrier obligation.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (emitted coordinate : CGB.GDNLife.Byte) (i : Fin d),
  HasDerivAt
    (fun θ =>
      Real.log
        (CGB.Medium.softmax
          (fun b =>
            CGB.GDNLife.logits s (CGB.GDNLife.feature M s) b +
              if b = coordinate then θ * CGB.GDNLife.feature M s i else 0)
          emitted))
    (CGB.GDNLife.actionHeadScore M s emitted coordinate i) 0
```

## CGB.GDNLife.addressLogitPerturb_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L187)

源码说明：Exact softmax derivative for an address-logit basis coordinate. This is the G14 finite-alphabet score identity applied to the GDN address, rather than an assumed attention-gradient rule.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (b : CGB.GDNLife.Byte) (j₀ j : Fin k),
  HasDerivAt (fun θ => CGB.GDNLife.address (CGB.GDNLife.addressLogitPerturb M b j₀ θ) b j)
    (CGB.GDNLife.address M b j * ((if j = j₀ then 1 else 0) - CGB.GDNLife.address M b j₀)) 0
```

## CGB.GDNLife.assimilate_addressLogitPerturb_natural_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L211)

源码说明：The exact address-side contribution to a GDN natural-parameter write. The address and value paths are separate summands of an encoder's total derivative; neither may be silently discarded.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival) (j₀ j : Fin k)
  (i : Fin d),
  HasDerivAt (fun θ => (CGB.GDNLife.assimilate (CGB.GDNLife.addressLogitPerturb M a.byte j₀ θ) s a).natural j i)
    (CGB.GDNLife.addressEligibility M a j₀ j i) 0
```

## CGB.GDNLife.assimilate_addressLogitPerturb_precision_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L235)

源码说明：The same address perturbation changes precision by the address-side incoming derivative. It is part of the future state sensitivity and cannot be represented by the value-only result.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival) (j₀ j : Fin k),
  HasDerivAt (fun θ => (CGB.GDNLife.assimilate (CGB.GDNLife.addressLogitPerturb M a.byte j₀ θ) s a).precision j)
    (a.observationPrecision * M.likelihoodPrecision a.byte * CGB.GDNLife.address M a.byte j *
      ((if j = j₀ then 1 else 0) - CGB.GDNLife.address M a.byte j₀))
    0
```

## CGB.GDNLife.likelihoodSoftplus_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L262)

源码说明：Exact derivative of the actual softplus confidence parametrization.

形式陈述：

```lean
∀ (r : ℝ), HasDerivAt (fun θ => CGB.GDNLife.likelihoodSoftplus (r + θ)) (CGB.GDNLife.likelihoodSlope r) 0
```

## CGB.GDNLife.assimilate_likelihoodLogitPerturb_natural_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L302)

源码说明：The likelihood side of the current GDN write: a learned confidence logit changes the natural parameter by its softplus slope times the existing address, observation precision and value.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival) (raw : ℝ) (j : Fin k)
  (i : Fin d),
  HasDerivAt (fun θ => (CGB.GDNLife.assimilate (CGB.GDNLife.likelihoodLogitPerturb M a.byte raw θ) s a).natural j i)
    (a.observationPrecision * CGB.GDNLife.address M a.byte j * M.value a.byte i * CGB.GDNLife.likelihoodSlope raw) 0
```

## CGB.GDNLife.assimilate_likelihoodLogitPerturb_precision_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L319)

源码说明：The same learned confidence changes the stored precision. Thus it cannot be treated as a fixed write scale while claiming the encoder is trainable.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival) (raw : ℝ) (j : Fin k),
  HasDerivAt (fun θ => (CGB.GDNLife.assimilate (CGB.GDNLife.likelihoodLogitPerturb M a.byte raw θ) s a).precision j)
    (a.observationPrecision * CGB.GDNLife.address M a.byte j * CGB.GDNLife.likelihoodSlope raw) 0
```

## CGB.GDNLife.assimilate_likelihoodLogitPerturb_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L338)

源码说明：When the raw logit represents the current confidence, the perturbation passes through the actual arrival transition at zero exactly. This establishes that the two derivative theorems are tangents of the existing model, rather than of an auxiliary learning rule.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival) (raw : ℝ),
  M.likelihoodPrecision a.byte = CGB.GDNLife.likelihoodSoftplus raw →
    CGB.GDNLife.assimilate (CGB.GDNLife.likelihoodLogitPerturb M a.byte raw 0) s a = CGB.GDNLife.assimilate M s a
```

## CGB.GDNLife.prepare_valuePerturb

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L346)

源码说明：Altering a byte value cannot alter the prediction prepared before that byte arrives. This is the causal boundary for its later credit.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ elapsed : ℝ)
  (s : CGB.GDNLife.State k d),
  CGB.GDNLife.prepare (CGB.GDNLife.valuePerturb M direction θ) s elapsed = CGB.GDNLife.prepare M s elapsed
```

## CGB.GDNLife.incoming_valuePerturb

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L353)

源码说明：Changing the byte encoder leaves the arrival weight unchanged: it does not silently alter the address or likelihood path.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ : ℝ) (a : CGB.GDNLife.Arrival)
  (j : Fin k), CGB.GDNLife.incoming (CGB.GDNLife.valuePerturb M direction θ) a j = CGB.GDNLife.incoming M a j
```

## CGB.GDNLife.assimilate_valuePerturb_natural

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L362)

源码说明：Exact one-arrival finite difference for a byte encoder coordinate. The coefficient is an arrival-local eligibility seed. It is a theorem about the actual GDN assimilation arithmetic, not a claim that delayed future credit can already be recovered without propagating this sensitivity.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ : ℝ) (s : CGB.GDNLife.State k d)
  (a : CGB.GDNLife.Arrival) (j : Fin k) (i : Fin d),
  (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a).natural j i =
    (CGB.GDNLife.assimilate M s a).natural j i + θ * CGB.GDNLife.valueEligibility M direction a j i
```

## CGB.GDNLife.assimilate_valuePerturb_precision

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L378)

源码说明：The same perturbation leaves precision unchanged. A future learner can therefore not attribute an encoder-value effect to a fabricated precision change in this fixed-address GDN subcase.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ : ℝ) (s : CGB.GDNLife.State k d)
  (a : CGB.GDNLife.Arrival) (j : Fin k),
  (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a).precision j =
    (CGB.GDNLife.assimilate M s a).precision j
```

## CGB.GDNLife.assimilate_valuePerturb_difference

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L387)

源码说明：Comparing two encoder coordinates gives the exact finite difference, without invoking reverse-time differentiation.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ₁ θ₂ : ℝ)
  (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival) (j : Fin k) (i : Fin d),
  (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ₁) s a).natural j i -
      (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ₂) s a).natural j i =
    (θ₁ - θ₂) * CGB.GDNLife.valueEligibility M direction a j i
```

## CGB.GDNLife.valueEligibility_zero_of_no_incoming

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L397)

源码说明：No arriving flow means no encoder-value eligibility at that address.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (a : CGB.GDNLife.Arrival) (j : Fin k)
  (i : Fin d), CGB.GDNLife.incoming M a j = 0 → CGB.GDNLife.valueEligibility M direction a j i = 0
```

## CGB.GDNLife.value_sensitivity_two_arrivals

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L405)

源码说明：The value-side sensitivity is carried by the same natural state across a second arrival. The second local term is added; no history buffer or reverse computation is introduced.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ : ℝ)
  (a₀ a₁ : CGB.GDNLife.Arrival) (j : Fin k) (i : Fin d),
  (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ)
          (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a₀) a₁).natural
      j i =
    (CGB.GDNLife.assimilate M (CGB.GDNLife.assimilate M s a₀) a₁).natural j i +
      θ * (CGB.GDNLife.valueEligibility M direction a₀ j i + CGB.GDNLife.valueEligibility M direction a₁ j i)
```

## CGB.GDNLife.initial_positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L434)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (p : ℝ),
  0 < p →
    ∀ (head : CGB.GDNLife.Byte → Fin d → ℝ) (bias : CGB.GDNLife.Byte → ℝ),
      CGB.GDNLife.Positive (CGB.GDNLife.initial p head bias)
```

## CGB.GDNLife.prior_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L438)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d), CGB.GDNLife.prior M s 0 = s
```

## CGB.GDNLife.prior_natural_difference

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L451)

源码说明：If a perturbation leaves a slot's precision unchanged, the prior transports its natural-state difference by `priorNaturalScale`; no past input is reread.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s s' : CGB.GDNLife.State k d) (elapsed : ℝ) (j : Fin k) (i : Fin d),
  s'.precision j = s.precision j →
    (CGB.GDNLife.prior M s' elapsed).natural j i - (CGB.GDNLife.prior M s elapsed).natural j i =
      CGB.GDNLife.priorNaturalScale M s elapsed j * (s'.natural j i - s.natural j i)
```

## CGB.GDNLife.value_sensitivity_through_prior

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L472)

源码说明：A value-write sensitivity remains in the same state after the time prior. The result is exact for the present GDN path: the prior rescales the existing trace instead of allocating a separate history carrier.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ)
  (θ elapsed : ℝ) (a : CGB.GDNLife.Arrival) (j : Fin k) (i : Fin d),
  (CGB.GDNLife.prior (CGB.GDNLife.valuePerturb M direction θ)
            (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a) elapsed).natural
        j i -
      (CGB.GDNLife.prior M (CGB.GDNLife.assimilate M s a) elapsed).natural j i =
    CGB.GDNLife.priorNaturalScale M (CGB.GDNLife.assimilate M s a) elapsed j *
      (θ * CGB.GDNLife.valueEligibility M direction a j i)
```

## CGB.GDNLife.feature_difference_of_shared_query

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L492)

源码说明：With unchanged precision and causal query, an existing natural-state difference is read out linearly by the same GDN feature map. This is the readout side of forward sensitivity, not an auxiliary eligibility array.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s s' : CGB.GDNLife.State k d),
  (∀ (j : Fin k), s'.precision j = s.precision j) →
    s'.lastByte = s.lastByte →
      ∀ (i : Fin d),
        CGB.GDNLife.feature M s' i - CGB.GDNLife.feature M s i =
          ∑ j, (s'.natural j i - s.natural j i) / max (s.precision j) M.precisionFloor * CGB.GDNLife.query M s j
```

## CGB.GDNLife.feature_valuePerturb

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L509)

源码说明：A perturbation of the byte-value encoder has no direct readout path. Its only possible effect is the natural-state flow written at arrival.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ : ℝ) (s : CGB.GDNLife.State k d),
  CGB.GDNLife.feature (CGB.GDNLife.valuePerturb M direction θ) s = CGB.GDNLife.feature M s
```

## CGB.GDNLife.assimilate_lastByte_of_positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L516)

源码说明：A positive-confidence arrival makes that byte the next causal query. This is an existing `lastByte` state field, rather than a new trace.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  0 < a.observationPrecision → (CGB.GDNLife.assimilate M s a).lastByte = some a.byte
```

## CGB.GDNLife.value_read_eligibility_exact

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L534)

源码说明：One byte-value encoder direction reaches the next causal readout exactly through `valueReadEligibility`. No reverse-time graph, copied history, or separate eligibility state is used: the coefficient is reconstructed from the same material state that already stores the write.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ : ℝ) (s : CGB.GDNLife.State k d)
  (a : CGB.GDNLife.Arrival),
  0 < a.observationPrecision →
    ∀ (i : Fin d),
      CGB.GDNLife.feature (CGB.GDNLife.valuePerturb M direction θ)
          (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a) i =
        CGB.GDNLife.feature M (CGB.GDNLife.assimilate M s a) i + θ * CGB.GDNLife.valueReadEligibility M direction s a i
```

## CGB.GDNLife.value_future_read_eligibility_exact

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L587)

源码说明：The exact encoder-value eligibility at the next prepared readout. A write is transported by the GDN prior and read by the same causal query; no backward graph or unbounded trace is introduced.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ elapsed : ℝ)
  (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  0 < a.observationPrecision →
    ∀ (i : Fin d),
      CGB.GDNLife.feature (CGB.GDNLife.valuePerturb M direction θ)
          (CGB.GDNLife.prior (CGB.GDNLife.valuePerturb M direction θ)
            (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a) elapsed)
          i =
        CGB.GDNLife.feature M (CGB.GDNLife.prior M (CGB.GDNLife.assimilate M s a) elapsed) i +
          θ * CGB.GDNLife.valueFutureReadEligibility M direction s a elapsed i
```

## CGB.GDNLife.assimilate_head

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L658)

源码说明：Assimilation writes only the material memory fields and causal observation cache; it does not create a second copy of the language readout parameters.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  (CGB.GDNLife.assimilate M s a).head = s.head
```

## CGB.GDNLife.assimilate_bias

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L661)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  (CGB.GDNLife.assimilate M s a).bias = s.bias
```

## CGB.GDNLife.prior_head

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L665)

源码说明：The time prior transports memory but retains the same head and bias.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (elapsed : ℝ),
  (CGB.GDNLife.prior M s elapsed).head = s.head
```

## CGB.GDNLife.prior_bias

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L669)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (elapsed : ℝ),
  (CGB.GDNLife.prior M s elapsed).bias = s.bias
```

## CGB.GDNLife.value_future_logit_eligibility_exact

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L684)

源码说明：The next byte-logit displacement is exactly the forward eligibility read through the unchanged affine head. This is still an identity, before an external outcome is used as a third factor.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ elapsed : ℝ)
  (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  0 < a.observationPrecision →
    ∀ (b : CGB.GDNLife.Byte),
      CGB.GDNLife.logits
          (CGB.GDNLife.prior (CGB.GDNLife.valuePerturb M direction θ)
            (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a) elapsed)
          (CGB.GDNLife.feature (CGB.GDNLife.valuePerturb M direction θ)
            (CGB.GDNLife.prior (CGB.GDNLife.valuePerturb M direction θ)
              (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a) elapsed))
          b =
        CGB.GDNLife.logits (CGB.GDNLife.prior M (CGB.GDNLife.assimilate M s a) elapsed)
            (CGB.GDNLife.feature M (CGB.GDNLife.prior M (CGB.GDNLife.assimilate M s a) elapsed)) b +
          θ * CGB.GDNLife.valueFutureLogitEligibility M direction s a elapsed b
```

## CGB.GDNLife.byte_log_softmax_direction_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L721)

源码说明：G14 for a whole finite logit direction. Coordinate G14 is the special case in which `direction` has one nonzero component.

形式陈述：

```lean
∀ (z direction : CGB.GDNLife.Byte → ℝ) (target : CGB.GDNLife.Byte),
  HasDerivAt (fun θ => Real.log (CGB.Medium.softmax (fun b => z b + θ * direction b) target))
    (direction target - ∑ b, CGB.Medium.softmax z b * direction b) 0
```

## CGB.GDNLife.value_future_log_likelihood_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L777)

源码说明：G14 applied to the exact next-byte GDN displacement: the derivative of next-byte log likelihood with respect to an encoder-value direction is the forward eligibility weighted by the observed-minus-predicted language error. This derives the local two factors; a delayed external outcome may only enter later as the already-required third factor.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (elapsed : ℝ)
  (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  0 < a.observationPrecision →
    ∀ (target : CGB.GDNLife.Byte),
      HasDerivAt
        (fun θ =>
          Real.log
            (CGB.GDNLife.allocation (CGB.GDNLife.valuePerturb M direction θ)
              (CGB.GDNLife.prior (CGB.GDNLife.valuePerturb M direction θ)
                (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a) elapsed)
              target))
        (CGB.GDNLife.valueFutureScore M direction s a elapsed target) 0
```

## CGB.GDNLife.value_future_score_factor

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L818)

源码说明：Exact factorization of the encoder-value score. The forward material eligibility and the observed-minus-predicted language error are sufficient statistics for this one-step value path; their product is not a hand-added surrogate for BPTT.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (s : CGB.GDNLife.State k d)
  (a : CGB.GDNLife.Arrival) (elapsed : ℝ) (target : CGB.GDNLife.Byte),
  CGB.GDNLife.valueFutureScore M direction s a elapsed target =
    ∑ i,
      CGB.GDNLife.valueFutureReadEligibility M direction s a elapsed i *
        CGB.GDNLife.nextFeatureLanguageError M s a elapsed target i
```

## CGB.GDNLife.address_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L868)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (b : CGB.GDNLife.Byte) (j : Fin k), 0 ≤ CGB.GDNLife.address M b j
```

## CGB.GDNLife.incoming_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L872)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (a : CGB.GDNLife.Arrival) (j : Fin k), 0 ≤ CGB.GDNLife.incoming M a j
```

## CGB.GDNLife.prior_positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L877)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d),
  CGB.GDNLife.Positive s → ∀ (elapsed : ℝ), 0 ≤ elapsed → CGB.GDNLife.Positive (CGB.GDNLife.prior M s elapsed)
```

## CGB.GDNLife.assimilate_positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L889)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d),
  CGB.GDNLife.Positive s → ∀ (a : CGB.GDNLife.Arrival), CGB.GDNLife.Positive (CGB.GDNLife.assimilate M s a)
```

## CGB.GDNLife.step_positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L894)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d),
  CGB.GDNLife.Positive s → ∀ (a : CGB.GDNLife.Arrival), CGB.GDNLife.Positive (CGB.GDNLife.step M s a)
```

## CGB.GDNLife.zero_precision_keeps_last_byte

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L900)

源码说明：Zero-precision input retains an existing last observation, matching `assimilate`'s live-observation mask rather than silently changing query.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival) (b : CGB.GDNLife.Byte),
  s.lastByte = some b → a.observationPrecision = 0 → (CGB.GDNLife.assimilate M s a).lastByte = some b
```

## CGB.GDNLife.first_observation_sets_last_byte

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L906)

源码说明：The first input still installs the cache, even at zero precision.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  s.lastByte = none → (CGB.GDNLife.assimilate M s a).lastByte = some a.byte
```

## CGB.GDNLife.positive_observation_sets_last_byte

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L912)

源码说明：Any live arrival writes its byte into the existing causal query field. This is a state transition, not a separate eligibility or action-log array.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  0 < a.observationPrecision → (CGB.GDNLife.assimilate M s a).lastByte = some a.byte
```

## CGB.GDNLife.step_positive_observation_sets_last_byte

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L921)

源码说明：The complete GDN step preserves that live causal write through its local head update. A self-produced language byte can therefore use the same body transition as an external byte; source provenance remains a boundary condition for later outcome modulation.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  0 < a.observationPrecision → (CGB.GDNLife.step M s a).lastByte = some a.byte
```

## CGB.GDNLife.distinct_live_bytes_separate_steps

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L931)

源码说明：Distinct live language emissions produce distinct post-step bodies through an existing causal field. This establishes the positive premise needed by `delayed_action_credit_requires_state_separation` for a byte-level speech act; it does not claim that the distinction survives arbitrary later interference.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a₀ a₁ : CGB.GDNLife.Arrival),
  0 < a₀.observationPrecision →
    0 < a₁.observationPrecision → a₀.byte ≠ a₁.byte → CGB.GDNLife.step M s a₀ ≠ CGB.GDNLife.step M s a₁
```

## CGB.GDNLife.step_source_irrelevant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L944)

源码说明：Source labels do not invent a speech fee or a different learning rule. The world must still distinguish external provenance from self-produced text when deciding whether an event is new evidence.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival)
  (src : CGB.GDNLife.Source),
  CGB.GDNLife.step M s
      { elapsed := a.elapsed, elapsed_nonneg := ⋯, byte := a.byte, observationPrecision := a.observationPrecision,
        observation_nonneg := ⋯, source := src } =
    CGB.GDNLife.step M s a
```

## CGB.GDNLife.preparation_uses_only_elapsed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L948)

源码说明：Prediction cannot inspect the not-yet-arrived byte or its source.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a a' : CGB.GDNLife.Arrival),
  a.elapsed = a'.elapsed → CGB.GDNLife.prepare M s a.elapsed = CGB.GDNLife.prepare M s a'.elapsed
```

## CGB.GDNLife.head_update_preserves_memory

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L953)

源码说明：Local head learning does not modify the memory or its uncertainty.

形式陈述：

```lean
∀ {k d : ℕ} (η : ℝ) (s : CGB.GDNLife.State k d) (f : Fin d → ℝ) (b : CGB.GDNLife.Byte),
  (CGB.GDNLife.headUpdate η s f b).natural = s.natural ∧ (CGB.GDNLife.headUpdate η s f b).precision = s.precision
```

## CGB.GDNLife.head_update_logits

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L959)

源码说明：The actual affine parameter step induces a score step of size `eta * (1 + squared feature norm)`, including the learned bias.

形式陈述：

```lean
∀ {k d : ℕ} (η : ℝ) (s : CGB.GDNLife.State k d) (f : Fin d → ℝ) (target b : CGB.GDNLife.Byte),
  CGB.GDNLife.logits (CGB.GDNLife.headUpdate η s f target) f b =
    CGB.GDNLife.logits s f b + η * (1 + ∑ i, f i * f i) * CGB.GDNLife.innovation s f target b
```

## CGB.GDNLife.head_update_improves_saved_feature

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L973)

源码说明：Strict improvement on the saved feature, not a future-language claim.

形式陈述：

```lean
∀ {k d : ℕ} (η : ℝ),
  0 < η →
    ∀ (s : CGB.GDNLife.State k d) (f : Fin d → ℝ) (target : CGB.GDNLife.Byte),
      CGB.Medium.softmax (CGB.GDNLife.logits s f) target <
        CGB.Medium.softmax (CGB.GDNLife.logits (CGB.GDNLife.headUpdate η s f target) f) target
```

## CGB.GDNLife.carrier_next

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1012)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} (M : CGB.GDNLife.Model k d) (s₀ : CGB.GDNLife.State k d) (h₀ : CGB.GDNLife.Positive s₀)
  (policy : CGB.GDNLife.State k d → Act) (response : CGB.GDNLife.State k d → Act → CGB.GDNLife.Arrival)
  (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  CGB.SelfProduction.renewableLanguageNext (CGB.GDNLife.carrier M s₀ h₀ policy response) s a = CGB.GDNLife.step M s a
```

## CGB.GDNLife.carrier_fold

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1020)

源码说明：Actual input histories fold through this same real body, with no reset.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} (M : CGB.GDNLife.Model k d) (s₀ : CGB.GDNLife.State k d) (h₀ : CGB.GDNLife.Positive s₀)
  (policy : CGB.GDNLife.State k d → Act) (response : CGB.GDNLife.State k d → Act → CGB.GDNLife.Arrival)
  (inputs : List CGB.GDNLife.Arrival),
  List.foldl (CGB.SelfProduction.renewableLanguageNext (CGB.GDNLife.carrier M s₀ h₀ policy response)) s₀ inputs =
    List.foldl (CGB.GDNLife.step M) s₀ inputs
```

## CGB.GDNLife.exists_open_real_gdn_trajectory

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1045)

源码说明：An actual continuous GDN, not a finite surrogate, has one infinite open numerical trajectory. Future benefit, active maintenance and the full life definition are not conclusions of positive precision.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (M : CGB.GDNLife.Model k d) (s₀ : CGB.GDNLife.State k d),
  CGB.GDNLife.Positive s₀ →
    ∀ (policy : CGB.GDNLife.State k d → Act) (response : CGB.GDNLife.State k d → Act → CGB.GDNLife.Arrival) (w₀ : World)
      (boundary : World → CGB.GDNLife.State k d → Act → CGB.GDNLife.Arrival)
      (evolve : World → CGB.GDNLife.State k d → Act → CGB.GDNLife.Arrival → World),
      ∃ x,
        x 0 = (s₀, w₀) ∧
          (∀ (n : ℕ),
              x (n + 1) =
                (CGB.GDNLife.step M (x n).1 (boundary (x n).2 (x n).1 (policy (x n).1)),
                  evolve (x n).2 (x n).1 (policy (x n).1) (boundary (x n).2 (x n).1 (policy (x n).1)))) ∧
            ∀ (n : ℕ), CGB.GDNLife.Positive (x n).1
```

## CGB.GDNLife.finish_saved_logits

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1072)

源码说明：Assimilation does not change the head. Thus its local update is exactly on the prediction-time feature, despite the changed memory underneath it.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (p : CGB.GDNLife.Prepared k d) (a : CGB.GDNLife.Arrival),
  CGB.GDNLife.logits (CGB.GDNLife.finish M p a) p.feature =
    CGB.GDNLife.logits (CGB.GDNLife.headUpdate M.headStep p.state p.feature a.byte) p.feature
```

## CGB.GDNLife.byte_softmax_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1082)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (z : CGB.GDNLife.Byte → ℝ) (b : CGB.GDNLife.Byte), 0 < CGB.Medium.softmax z b
```

## CGB.GDNLife.finish_saved_nll_improves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1089)

源码说明：The saved-feature current-byte loss strictly decreases at any positive head step. No assumption about the next byte is hidden in this result.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d),
  0 < M.headStep →
    ∀ (p : CGB.GDNLife.Prepared k d) (a : CGB.GDNLife.Arrival),
      CGB.GDNLife.nll (CGB.GDNLife.logits (CGB.GDNLife.finish M p a) p.feature) a.byte <
        CGB.GDNLife.nll (CGB.GDNLife.logits p.state p.feature) a.byte
```

## CGB.GDNLife.future_ce_improves_of_target_mass

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1102)

源码说明：A distributional, not in-sample, sufficient condition. A known target loss gain beats all possible other-byte loss increases when the *same* future law assigns enough mass to the target.

形式陈述：

```lean
∀ (law old new : CGB.GDNLife.Byte → ℝ) (target : CGB.GDNLife.Byte) (gain bound : ℝ),
  (∀ (b : CGB.GDNLife.Byte), 0 ≤ law b) →
    ∑ b, law b = 1 →
      CGB.GDNLife.nll new target - CGB.GDNLife.nll old target ≤ -gain →
        (∀ (b : CGB.GDNLife.Byte), b ≠ target → CGB.GDNLife.nll new b - CGB.GDNLife.nll old b ≤ bound) →
          bound < law target * (bound + gain) →
            CGB.PredictiveHarvest.crossEntropy law (CGB.Medium.softmax new) <
              CGB.PredictiveHarvest.crossEntropy law (CGB.Medium.softmax old)
```

## CGB.GDNLife.finish_future_ce_improves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1131)

源码说明：A concrete head update can improve a declared future law even after the memory/query changed: the saved-feature gain must exceed the readout drift, and the law must be sufficiently persistent. These are numeric/world conditions, not an assumption that future CE already improved.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (p : CGB.GDNLife.Prepared k d) (a : CGB.GDNLife.Arrival)
  (law : CGB.GDNLife.Byte → ℝ) (δ κ : ℝ),
  (∀ (b : CGB.GDNLife.Byte), 0 ≤ law b) →
    ∑ b, law b = 1 →
      (∀ (b : CGB.GDNLife.Byte),
          |CGB.GDNLife.logits (CGB.GDNLife.finish M p a) (CGB.GDNLife.feature M (CGB.GDNLife.finish M p a)) b -
                CGB.GDNLife.logits (CGB.GDNLife.finish M p a) p.feature b| ≤
            δ) →
        (∀ (b : CGB.GDNLife.Byte),
            |CGB.GDNLife.logits (CGB.GDNLife.finish M p a) (CGB.GDNLife.feature M (CGB.GDNLife.finish M p a)) b -
                  CGB.GDNLife.logits p.state p.feature b| ≤
              κ) →
          2 * κ <
              law a.byte *
                (2 * κ +
                  (CGB.GDNLife.nll (CGB.GDNLife.logits p.state p.feature) a.byte -
                      CGB.GDNLife.nll (CGB.GDNLife.logits (CGB.GDNLife.finish M p a) p.feature) a.byte -
                    2 * δ)) →
            CGB.PredictiveHarvest.crossEntropy law (CGB.GDNLife.allocation M (CGB.GDNLife.finish M p a)) <
              CGB.PredictiveHarvest.crossEntropy law (CGB.Medium.softmax (CGB.GDNLife.logits p.state p.feature))
```

## CGB.GDNLife.real_bias_injective

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1162)

源码说明：The real family already embeds a real coordinate in an existing head bias. No finite body-state enumeration is being smuggled into this carrier.

形式陈述：

```lean
∀ {k d : ℕ}, Function.Injective fun r => CGB.GDNLife.initial 1 (fun x x_1 => 0) fun x => r
```

## CGB.GDNLife.prior_logits

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1167)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (elapsed : ℝ) (f : Fin d → ℝ),
  CGB.GDNLife.logits (CGB.GDNLife.prior M s elapsed) f = CGB.GDNLife.logits s f
```

## CGB.GDNLife.step_ne_of_positive_head_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1176)

源码说明：A positive real head step changes the body on every arriving byte. This alone is not evidence of useful learning; unpredictable input also moves parameters. Finite-precision stalling is a separate numerical issue.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d),
  0 < M.headStep → ∀ (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival), CGB.GDNLife.step M s a ≠ s
```

## CGB.GDNLife.Process.run_initial

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1229)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World),
  P.run 0 = (P.initial, P.worldInitial)
```

## CGB.GDNLife.Process.run_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1231)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (n : ℕ),
  P.run (n + 1) =
    (CGB.GDNLife.step P.model (P.run n).1 (P.arrival n),
      P.evolve (P.run n).2 (P.run n).1 (P.policy (P.run n).1) (P.arrival n))
```

## CGB.GDNLife.Process.run_positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1239)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (n : ℕ),
  CGB.GDNLife.Positive (P.run n).1
```

## CGB.GDNLife.Process.run_body_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1243)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (n : ℕ),
  (P.run (n + 1)).1 = CGB.GDNLife.finish P.model (P.snapshot n) (P.arrival n)
```

## CGB.GDNLife.Process.self_speech_action_contrast_writes_body

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1252)

源码说明：Two candidate language actions that emit different live self-produced bytes write distinct next bodies through the ordinary GDN boundary transition. The byte is the action's existing bodily consequence, not a separately stored action record.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (n : ℕ) (a₀ a₁ : Act),
  (P.boundary (P.run n).2 (P.run n).1 a₀).source = CGB.GDNLife.Source.self →
    (P.boundary (P.run n).2 (P.run n).1 a₁).source = CGB.GDNLife.Source.self →
      0 < (P.boundary (P.run n).2 (P.run n).1 a₀).observationPrecision →
        0 < (P.boundary (P.run n).2 (P.run n).1 a₁).observationPrecision →
          (P.boundary (P.run n).2 (P.run n).1 a₀).byte ≠ (P.boundary (P.run n).2 (P.run n).1 a₁).byte →
            CGB.GDNLife.step P.model (P.run n).1 (P.boundary (P.run n).2 (P.run n).1 a₀) ≠
              CGB.GDNLife.step P.model (P.run n).1 (P.boundary (P.run n).2 (P.run n).1 a₁)
```

## CGB.GDNLife.Process.next_byte_improves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1268)

源码说明：Repetition refers to the world's actual next byte, not a byte chosen for rescoring. The readout-drift budget includes the next physical-time prior.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (n : ℕ) (δ : ℝ),
  (P.arrival (n + 1)).byte = (P.arrival n).byte →
    (∀ (b : CGB.GDNLife.Byte),
        |P.scores (n + 1) b -
              CGB.GDNLife.logits (CGB.GDNLife.finish P.model (P.snapshot n) (P.arrival n)) (P.snapshot n).feature b| ≤
          δ) →
      2 * δ <
          CGB.GDNLife.nll (P.scores n) (P.arrival n).byte -
            CGB.GDNLife.nll
              (CGB.GDNLife.logits (CGB.GDNLife.finish P.model (P.snapshot n) (P.arrival n)) (P.snapshot n).feature)
              (P.arrival n).byte →
        CGB.GDNLife.nll (P.scores (n + 1)) (P.arrival (n + 1)).byte <
          CGB.GDNLife.nll (P.scores n) (P.arrival (n + 1)).byte
```

## CGB.GDNLife.Process.externallyUseful_improves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1302)

源码说明：The useful-event certificate invokes the earlier causal result; it does not inspect the next byte while preparing the current prediction.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (n : ℕ),
  P.externallyUseful n →
    CGB.GDNLife.nll (P.scores (n + 1)) (P.arrival (n + 1)).byte < CGB.GDNLife.nll (P.scores n) (P.arrival (n + 1)).byte
```

## CGB.GDNLife.Process.self_not_externallyUseful

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1311)

源码说明：A copied self byte never becomes external evidence just by passing through the same learning rule. It can still affect the body through `step`.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (n : ℕ),
  (P.arrival n).source = CGB.GDNLife.Source.self → ¬P.externallyUseful n
```

## CGB.GDNLife.Process.externalInformation_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1346)

源码说明：A valid external language law has nonnegative information opportunity. This is finite conditional mutual information, rather than an arbitrary natural-number intake signal.

形式陈述：

```lean
∀ (joint : CGB.GDNLife.Byte → CGB.GDNLife.Byte → ℝ),
  (∀ (now later : CGB.GDNLife.Byte), 0 ≤ joint now later) → 0 ≤ CGB.GDNLife.Process.externalInformation joint
```

## CGB.GDNLife.Process.ExternalLanguageLaw.information_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1355)

源码说明：The state/action-indexed opportunity inherits CMI nonnegativity from the registered external language law.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} {P : CGB.GDNLife.Process k d Act World} (L : P.ExternalLanguageLaw)
  (s : CGB.GDNLife.State k d) (a : Act), 0 ≤ L.information s a
```

## CGB.GDNLife.Process.future_reserve_preserved_keeps_signature_distinction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1394)

源码说明：A preserved future-language reserve cannot silently merge two histories that a registered external continuation could previously distinguish. This is the long-delay retention condition for any causal difference already carried by the body; it adds neither an action log nor a separate eligibility state.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) {m n : ℕ} (Q : CGB.GDNLife.Process.ExternalFutureProbes m)
  (before after : Fin n → CGB.GDNLife.State k d) (i j : Fin n),
  CGB.GDNLife.Process.futureReservePreserved M Q before after →
    CGB.GDNLife.Process.futureLanguageSignature M Q (before i) ≠
        CGB.GDNLife.Process.futureLanguageSignature M Q (before j) →
      CGB.GDNLife.Process.futureLanguageSignature M Q (after i) ≠
        CGB.GDNLife.Process.futureLanguageSignature M Q (after j)
```

## CGB.GDNLife.Process.future_reserve_recoverable_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1437)

源码说明：A future distinction is not dissipated exactly when all of the prior future language signature remains readable from the post-update signature. This reuses the existing causal-carrier theorem instead of introducing a separate retention mechanism.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) {m n : ℕ} (Q : CGB.GDNLife.Process.ExternalFutureProbes m)
  (before after : Fin n → CGB.GDNLife.State k d),
  (∃ restore,
      ∀ (i : Fin n),
        restore (CGB.GDNLife.Process.futureLanguageSignature M Q (after i)) =
          CGB.GDNLife.Process.futureLanguageSignature M Q (before i)) ↔
    CGB.GDNLife.Process.futureReservePreserved M Q before after
```

## CGB.GDNLife.Process.loses_future_language_distinction_iff_not_preserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1452)

源码说明：Merging a prior distinct future-language signature is precisely the failure of that causal retention contract.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) {m n : ℕ} (Q : CGB.GDNLife.Process.ExternalFutureProbes m)
  (before after : Fin n → CGB.GDNLife.State k d),
  CGB.GDNLife.Process.losesFutureLanguageDistinction M Q before after ↔
    ¬CGB.GDNLife.Process.futureReservePreserved M Q before after
```

## CGB.GDNLife.Process.recurrent_external_information_forces_activity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1475)

源码说明：Information-theoretic active necessity for the concrete GDN port. If settling into `passive` eliminates the conditional information available from the external language boundary, while positive external information recurs on the realised trajectory, the policy cannot settle into `passive`. The premise is deliberately a boundary statement, not a synthetic hunger counter. To turn available information into a learning claim, it must also meet the causal `externallyUseful` scoring certificate above.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (L : P.ExternalLanguageLaw)
  (passive : Act),
  (CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive →
      ∃ N, ∀ (n : ℕ), N ≤ n → L.information (P.run n).1 (P.policy (P.run n).1) = 0) →
    (∀ (N : ℕ),
        ∃ n,
          N ≤ n ∧
            (P.arrival n).source = CGB.GDNLife.Source.external ∧ 0 < L.information (P.run n).1 (P.policy (P.run n).1)) →
      ¬CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive
```

## CGB.GDNLife.Process.recurrent_external_evidence_forces_activity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1494)

源码说明：If settling into a passive policy rules out recurrent externally useful language, but such events actually recur on this same trajectory, the policy cannot settle. This is the language form of active necessity: no arbitrary `intake`, opportunity counter, food pool, or maintenance currency is added.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (passive : Act),
  (CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive →
      ∃ N, ∀ (n : ℕ), N ≤ n → ¬P.externallyUseful n) →
    (∀ (N : ℕ), ∃ n, N ≤ n ∧ P.externallyUseful n) →
      ¬CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive
```

## CGB.GDNLife.Process.scored_external_language_trajectory

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1512)

源码说明：Conditional language-level joint result for the real GDN. Every condition is about the actual language trajectory: source provenance, recurrent post-update improvement, the causal effect of action on the next arrival, and a non-uniform prequential margin. The result intentionally says nothing about physical food, material ingestion, or an auxiliary reserve state. It is still not a full `Medium.Lives` construction.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World),
  0 < P.model.headStep →
    ∀ (passive : Act),
      (CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive →
          ∃ N, ∀ (n : ℕ), N ≤ n → ¬P.externallyUseful n) →
        (∀ (N : ℕ), ∃ n, N ≤ n ∧ P.externallyUseful n) →
          (∃ n a a', (P.boundary (P.run n).2 (P.run n).1 a).byte ≠ (P.boundary (P.run n).2 (P.run n).1 a').byte) →
            (∀ (n : ℕ), CGB.GDNLife.nll (P.scores n) (P.arrival n).byte < Real.log 256) →
              (∀ (n : ℕ), CGB.GDNLife.Positive (P.run n).1) ∧
                ¬CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive ∧
                  (¬∃ predict,
                        ∀ (n : ℕ) (a : Act),
                          predict (P.run n).2 (P.run n).1 = (P.boundary (P.run n).2 (P.run n).1 a).byte) ∧
                    (∀ (n : ℕ), 0 < Real.log 256 - CGB.GDNLife.nll (P.scores n) (P.arrival n).byte) ∧
                      (∀ (N : ℕ),
                          ∃ n,
                            N ≤ n ∧
                              CGB.GDNLife.nll (P.scores (n + 1)) (P.arrival (n + 1)).byte <
                                CGB.GDNLife.nll (P.scores n) (P.arrival (n + 1)).byte) ∧
                        ¬∃ N, ∀ (n : ℕ), N ≤ n → (P.run (n + 1)).1 = (P.run n).1
```

## CGB.GDNLife.Process.scored_joint_trajectory

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1554)

源码说明：A conditional scored joint trajectory for this real GDN. Conditions are on the actual world: registered intake, a finite passive opportunity budget under the same intervenable transition, an action contrast, a nonnegative predictive margin, and recurrent next-byte persistence with bounded readout drift. Recurrent future score gain is *derived*, not supplied as `hlearn`. Numeric positivity is still not material/biological viability. This deterministic-world statement scores actual future bytes, not a claim about the conditional law of an unspecified stochastic language world.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World),
  0 < P.model.headStep →
    ∀ (intake : World → CGB.GDNLife.State k d → Act → CGB.GDNLife.Arrival → ℕ) (passive : Act),
      CGB.ActiveInference.unbounded (P.signal intake) →
        ∀ (opportunity : CGB.GDNLife.State k d × World → ℕ),
          (∀ (z : CGB.GDNLife.State k d × World),
              opportunity (P.contract.step z (P.contract.response z passive)) +
                  intake z.2 z.1 passive (P.boundary z.2 z.1 passive) ≤
                opportunity z) →
            (∃ n a a', (P.boundary (P.run n).2 (P.run n).1 a).byte ≠ (P.boundary (P.run n).2 (P.run n).1 a').byte) →
              ∀ (m : ℝ),
                0 ≤ m →
                  (∀ (n : ℕ), CGB.GDNLife.nll (P.scores n) (P.arrival n).byte < Real.log 256 - m) →
                    (∀ (N : ℕ),
                        ∃ n,
                          N ≤ n ∧
                            ∃ δ,
                              (P.arrival (n + 1)).byte = (P.arrival n).byte ∧
                                (∀ (b : CGB.GDNLife.Byte),
                                    |P.scores (n + 1) b -
                                          CGB.GDNLife.logits (CGB.GDNLife.finish P.model (P.snapshot n) (P.arrival n))
                                            (P.snapshot n).feature b| ≤
                                      δ) ∧
                                  2 * δ <
                                    CGB.GDNLife.nll (P.scores n) (P.arrival n).byte -
                                      CGB.GDNLife.nll
                                        (CGB.GDNLife.logits (CGB.GDNLife.finish P.model (P.snapshot n) (P.arrival n))
                                          (P.snapshot n).feature)
                                        (P.arrival n).byte) →
                      (∀ (n : ℕ), CGB.GDNLife.Positive (P.run n).1) ∧
                        ¬CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive ∧
                          (¬∃ predict,
                                ∀ (n : ℕ) (a : Act),
                                  predict (P.run n).2 (P.run n).1 = (P.boundary (P.run n).2 (P.run n).1 a).byte) ∧
                            (∀ (n : ℕ),
                                0 < Real.log 256 - CGB.GDNLife.nll (P.scores n) (P.arrival n).byte ∧
                                  0 < Real.log 256 - CGB.GDNLife.nll (P.scores n) (P.arrival n).byte - m) ∧
                              (∀ (N : ℕ),
                                  ∃ n,
                                    N ≤ n ∧
                                      CGB.GDNLife.nll (P.scores (n + 1)) (P.arrival (n + 1)).byte <
                                        CGB.GDNLife.nll (P.scores n) (P.arrival (n + 1)).byte) ∧
                                ¬∃ N, ∀ (n : ℕ), N ≤ n → (P.run (n + 1)).1 = (P.run n).1
```
