# gdn_operator_tags_20261108

探索区：仅按源码盘点；不在生产区全量审计覆盖内，不能据此声称已通过当前构建或信任审计。

[源文件](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean) · [返回导航](../README.md)

模块源码说明：Installed GDN operator directions Real-arithmetic operator refinement, separately kernel checked.

## CGB.GDNOperatorTags.bilinear_read_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L31)

解释：Same bilinear local law for each projection row, depthwise convolution window and query read. Frozen weights mean dw=0, not dx=0.

源码说明：Same bilinear local law for each projection row, depthwise convolution window and query read. Frozen weights mean dw=0, not dx=0.

形式陈述：

```lean
theorem bilinear_read_deriv (w x : I → ℝ → ℝ) (dw dx : I → ℝ) (θ : ℝ)
    (hw : ∀ i, HasDerivAt (w i) (dw i) θ)
    (hx : ∀ i, HasDerivAt (x i) (dx i) θ) :
    HasDerivAt (fun t => ∑ i, w i t * x i t)
      (∑ i, (dw i * x i θ + w i θ * dx i)) θ
```

## CGB.GDNOperatorTags.square_sum_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L39)

解释：Squared norm source; the factor two is essential for both normalizers.

源码说明：Squared norm source; the factor two is essential for both normalizers.

形式陈述：

```lean
theorem square_sum_deriv (x : I → ℝ → ℝ) (dx : I → ℝ) (θ : ℝ)
    (hx : ∀ i, HasDerivAt (x i) (dx i) θ) :
    HasDerivAt (fun t => ∑ i, (x i t) ^ 2)
      (2 * ∑ i, x i θ * dx i) θ
```

## CGB.GDNOperatorTags.silu_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L57)

解释：SiLU used by the three causal convolutions, MLP and output gate.

源码说明：SiLU used by the three causal convolutions, MLP and output gate.

形式陈述：

```lean
theorem silu_deriv (x : ℝ → ℝ) (dx θ : ℝ) (hx : HasDerivAt x dx θ) :
    HasDerivAt (fun t => silu (x t))
      (dx * (Real.sigmoid (x θ) +
        x θ * Real.sigmoid (x θ) * (1 - Real.sigmoid (x θ)))) θ
```

## CGB.GDNOperatorTags.softplus_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L71)

解释：Smooth branch only. Torch softplus uses a linear branch above its threshold; its boundary and finite-precision behavior are not erased here.

源码说明：Smooth branch only. Torch softplus uses a linear branch above its threshold; its boundary and finite-precision behavior are not erased here.

形式陈述：

```lean
theorem softplus_deriv (x : ℝ → ℝ) (dx θ : ℝ) (hx : HasDerivAt x dx θ) :
    HasDerivAt (fun t => softplus (x t))
      (Real.exp (x θ) * dx / (1 + Real.exp (x θ))) θ
```

## CGB.GDNOperatorTags.beta_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L77)

解释：sigmoid write gate: raw projection directions still have a local path.

源码说明：sigmoid write gate: raw projection directions still have a local path.

形式陈述：

```lean
theorem beta_deriv (b : ℝ → ℝ) (db θ : ℝ) (hb : HasDerivAt b db θ) :
    HasDerivAt (fun t => Real.sigmoid (b t))
      (Real.sigmoid (b θ) * (1 - Real.sigmoid (b θ)) * db) θ
```

## CGB.GDNOperatorTags.log_retention_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L84)

解释：log retention in the installed layer: -exp(A_log)*softplus(a+dt_bias). The caller supplies a+dt_bias as z, including both parameter/input sources.

源码说明：log retention in the installed layer: -exp(A_log)*softplus(a+dt_bias). The caller supplies a+dt_bias as z, including both parameter/input sources.

形式陈述：

```lean
theorem log_retention_deriv (a z : ℝ → ℝ) (da dz θ : ℝ)
    (ha : HasDerivAt a da θ) (hz : HasDerivAt z dz θ) :
    HasDerivAt (fun t => -(Real.exp (a t) * softplus (z t)))
      (-(Real.exp (a θ) * da * softplus (z θ) +
        Real.exp (a θ) * (Real.exp (z θ) * dz / (1 + Real.exp (z θ))))) θ
```

## CGB.GDNOperatorTags.alpha_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L92)

解释：Delta receives alpha=exp(g), not g itself.

源码说明：Delta receives alpha=exp(g), not g itself.

形式陈述：

```lean
theorem alpha_deriv (g : ℝ → ℝ) (dg θ : ℝ) (hg : HasDerivAt g dg θ) :
    HasDerivAt (fun t => Real.exp (g t)) (Real.exp (g θ) * dg) θ
```

## CGB.GDNOperatorTags.smooth_normalizer_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L99)

解释：Chunk q/k norm (d=1), RMS norm (d=feature count); n is the squared sum. This factor is sqrt(n/d+eps), NOT sqrt(n)+eps.

源码说明：Chunk q/k norm (d=1), RMS norm (d=feature count); n is the squared sum. This factor is sqrt(n/d+eps), NOT sqrt(n)+eps.

形式陈述：

```lean
theorem smooth_normalizer_deriv (n : ℝ → ℝ) (dn d eps θ : ℝ)
    (hn : HasDerivAt n dn θ) (hpositive : 0 < n θ / d + eps) :
    HasDerivAt (fun t => Real.sqrt (n t / d + eps))
      ((dn / d) / (2 * Real.sqrt (n θ / d + eps))) θ
```

## CGB.GDNOperatorTags.recurrent_normalizer_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L107)

解释：Fused recurrent short-input q/k norm uses sqrt(n)+eps. Nonzero norm is an explicit premise for this scalar sqrt proof.

源码说明：Fused recurrent short-input q/k norm uses sqrt(n)+eps. Nonzero norm is an explicit premise for this scalar sqrt proof.

形式陈述：

```lean
theorem recurrent_normalizer_deriv (n : ℝ → ℝ) (dn eps θ : ℝ)
    (hn : HasDerivAt n dn θ) (hpositive : 0 < n θ) :
    HasDerivAt (fun t => Real.sqrt (n t) + eps)
      (dn / (2 * Real.sqrt (n θ))) θ
```

## CGB.GDNOperatorTags.normalized_coordinate_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L115)

解释：One normalized coordinate, including the norm's response to all other coordinates. Suppressing dr is not an exact normalization direction.

源码说明：One normalized coordinate, including the norm's response to all other coordinates. Suppressing dr is not an exact normalization direction.

形式陈述：

```lean
theorem normalized_coordinate_deriv (x r : ℝ → ℝ) (dx dr θ : ℝ)
    (hx : HasDerivAt x dx θ) (hr : HasDerivAt r dr θ) (hne : r θ ≠ 0) :
    HasDerivAt (fun t => x t / r t)
      ((dx * r θ - x θ * dr) / (r θ) ^ 2) θ
```

## CGB.GDNOperatorTags.weighted_norm_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L122)

解释：RMS affine weight is trainable in the algebra; a frozen block still passes the incoming direction.

源码说明：RMS affine weight is trainable in the algebra; a frozen block still passes the incoming direction.

形式陈述：

```lean
theorem weighted_norm_deriv (w x r : ℝ → ℝ) (dw dx dr θ : ℝ)
    (hw : HasDerivAt w dw θ) (hx : HasDerivAt x dx θ)
    (hr : HasDerivAt r dr θ) (hne : r θ ≠ 0) :
    HasDerivAt (fun t => w t * (x t / r t))
      (dw * (x θ / r θ) + w θ *
        ((dx * r θ - x θ * dr) / (r θ) ^ 2)) θ
```

## CGB.GDNOperatorTags.shift4_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L141)

解释：The convolution CACHE direction must shift as well as its output. Keeping S tags but dropping these buffers omits a real temporal path.

源码说明：The convolution CACHE direction must shift as well as its output. Keeping S tags but dropping these buffers omits a real temporal path.

形式陈述：

```lean
theorem shift4_deriv (cache : Fin 4 → ℝ → ℝ) (x : ℝ → ℝ)
    (dcache : Fin 4 → ℝ) (dx θ : ℝ)
    (hc : ∀ i, HasDerivAt (cache i) (dcache i) θ)
    (hx : HasDerivAt x dx θ) (i : Fin 4) :
    HasDerivAt (fun t => shift4 (fun j => cache j t) (x t) i)
      (shift4 dcache dx i) θ
```

## CGB.GDNOperatorTags.causal_conv4_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L154)

解释：Four-tap convolution plus SiLU, with weight and incoming-cache sources on the same graph. The current projection enters the last tap.

源码说明：Four-tap convolution plus SiLU, with weight and incoming-cache sources on the same graph. The current projection enters the last tap.

形式陈述：

```lean
theorem causal_conv4_deriv (w cache : Fin 4 → ℝ → ℝ) (x : ℝ → ℝ)
    (dw dcache : Fin 4 → ℝ) (dx θ : ℝ)
    (hw : ∀ i, HasDerivAt (w i) (dw i) θ)
    (hc : ∀ i, HasDerivAt (cache i) (dcache i) θ)
    (hx : HasDerivAt x dx θ) :
    HasDerivAt
      (fun t => silu (∑ i, w i t * shift4 (fun j => cache j t) (x t) i))
      ((∑ i, (dw i * shift4 (fun j => cache j θ) (x θ) i +
        w i θ * shift4 dcache dx i)) *
        (Real.sigmoid (∑ i, w i θ * shift4 (fun j => cache j θ) (x θ) i) +
          (∑ i, w i θ * shift4 (fun j => cache j θ) (x θ) i) *
          Real.sigmoid (∑ i, w i θ * shift4 (fun j => cache j θ) (x θ) i) *
          (1 - Real.sigmoid (∑ i, w i θ * shift4 (fun j => cache j θ) (x θ) i)))) θ
```

## CGB.GDNOperatorTags.delta_next_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L184)

解释：Joint source and transport: alpha,beta,key,value AND old S all vary. No assumption that the old effect stays in its original layer is introduced.

源码说明：Joint source and transport: alpha,beta,key,value AND old S all vary. No assumption that the old effect stays in its original layer is introduced.

形式陈述：

```lean
theorem delta_next_deriv (S : I → J → ℝ → ℝ) (k : I → ℝ → ℝ)
    (v : J → ℝ → ℝ) (α β : ℝ → ℝ) (dS : I → J → ℝ)
    (dk : I → ℝ) (dv : J → ℝ) (dα dβ θ : ℝ)
    (hS : ∀ i j, HasDerivAt (S i j) (dS i j) θ)
    (hk : ∀ i, HasDerivAt (k i) (dk i) θ)
    (hv : ∀ j, HasDerivAt (v j) (dv j) θ)
    (hα : HasDerivAt α dα θ) (hβ : HasDerivAt β dβ θ) (i : I) (j : J) :
    HasDerivAt
      (fun t => deltaNext (fun a b => S a b t) (fun a => k a t)
        (fun b => v b t) (α t) (β t) i j)
      (dα * S i j θ + α θ * dS i j +
        (dβ * k i θ + β θ * dk i) *
          (v j θ - ∑ a, k a θ * (α θ * S a j θ)) +
        β θ * k i θ * (dv j - ∑ a, (dk a * (α θ * S a j θ) +
          k a θ * (dα * S a j θ + α θ * dS a j)))) θ
```

## CGB.GDNOperatorTags.query_read_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L210)

解释：Actual query read has BOTH query credit and transported S credit; scale is the fixed head-dimension factor. Frozen query projections do not kill dq if their inputs depend on the old participation.

源码说明：Actual query read has BOTH query credit and transported S credit; scale is the fixed head-dimension factor. Frozen query projections do not kill dq if their inputs depend on the old participation.

形式陈述：

```lean
theorem query_read_deriv (q : I → ℝ → ℝ) (S : I → J → ℝ → ℝ)
    (dq : I → ℝ) (dS : I → J → ℝ) (scale θ : ℝ)
    (hq : ∀ i, HasDerivAt (q i) (dq i) θ)
    (hS : ∀ i j, HasDerivAt (S i j) (dS i j) θ) (j : J) :
    HasDerivAt (fun t => scale * ∑ i, q i t * S i j t)
      (scale * ∑ i, (dq i * S i j θ + q i θ * dS i j)) θ
```

## CGB.GDNOperatorTags.swish_product_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L223)

解释：SwiGLU intermediate coordinate AND the swish output norm gate share the same product rule. u's source survives even if the gate weights are frozen.

源码说明：SwiGLU intermediate coordinate AND the swish output norm gate share the same product rule. u's source survives even if the gate weights are frozen.

形式陈述：

```lean
theorem swish_product_deriv (g u : ℝ → ℝ) (dg du θ : ℝ)
    (hg : HasDerivAt g dg θ) (hu : HasDerivAt u du θ) :
    HasDerivAt (fun t => silu (g t) * u t)
      (dg * (Real.sigmoid (g θ) +
        g θ * Real.sigmoid (g θ) * (1 - Real.sigmoid (g θ))) * u θ +
        silu (g θ) * du) θ
```

## CGB.GDNOperatorTags.residual_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L232)

解释：Each residual adds the incoming path instead of replacing it with an isolated local loss. Apply twice per GDN block.

源码说明：Each residual adds the incoming path instead of replacing it with an isolated local loss. Apply twice per GDN block.

形式陈述：

```lean
theorem residual_deriv (x f : ℝ → ℝ) (dx df θ : ℝ)
    (hx : HasDerivAt x dx θ) (hf : HasDerivAt f df θ) :
    HasDerivAt (fun t => x t + f t) (dx + df) θ
```

## CGB.GDNOperatorTags.frozen_successor_carries_direction

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L246)

解释：The next frozen block is a map on activation AND its own recurrent cache. Its weights may be fixed while its state/activation directions pass through. This is spatial composition, not a second historical BPTT window.

源码说明：The next frozen block is a map on activation AND its own recurrent cache. Its weights may be fixed while its state/activation directions pass through. This is spatial composition, not a second historical BPTT window.

形式陈述：

```lean
theorem frozen_successor_carries_direction (input : ℝ → E) (e : E) (θ : ℝ)
    (lower : E → F) (upper : F → G)
    (dlower : E →L[ℝ] F) (dupper : F →L[ℝ] G)
    (hi : HasDerivAt input e θ)
    (hl : HasFDerivAt lower dlower (input θ))
    (hu : HasFDerivAt upper dupper (lower (input θ))) :
    HasDerivAt (fun t => upper (lower (input t))) (dupper (dlower e)) θ
```

## CGB.GDNOperatorTags.spatial_stages_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L276)

解释：Assemble any finite number of differentiable stages, hence also 24, from LOCAL stage derivatives. It does not assume the whole-block/whole-stack derivative as a premise. Stage instantiation must cover every cache field.

源码说明：Assemble any finite number of differentiable stages, hence also 24, from LOCAL stage derivatives. It does not assume the whole-block/whole-stack derivative as a premise. Stage instantiation must cover every cache field.

形式陈述：

```lean
theorem spatial_stages_deriv (stage : ℕ → ℝ × E → E) (input : ℝ → E)
    (D : ℕ → (ℝ × E →L[ℝ] E)) (initial : E)
    (hi : HasDerivAt input initial 0)
    (hs : ∀ n, HasFDerivAt (stage n) (D n) (0, spatial stage input 0 n))
    (layers : ℕ) :
    HasDerivAt (fun ε => spatial stage input ε layers)
      (spatialTag D initial layers) 0
```

## CGB.GDNOperatorTags.frozen_stage_transport

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L291)

解释：A frozen spatial stage contributes no new parameter source, but passes the old participation via its actual state/input differential.

源码说明：A frozen spatial stage contributes no new parameter source, but passes the old participation via its actual state/input differential.

形式陈述：

```lean
theorem frozen_stage_transport (D : ℝ × E →L[ℝ] E) (e : E)
    (frozen : D (1, 0) = 0) : D (1, e) = D (0, e)
```

## CGB.GDNOperatorTags.boundary_nll_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L305)

解释：The original language boundary supplies a vector innovation. Contracting it with the old participation's logit direction does not give hidden layers private vocabulary heads or substitute a scalar reward for language error.

源码说明：The original language boundary supplies a vector innovation. Contracting it with the old participation's logit direction does not give hidden layers private vocabulary heads or substitute a scalar reward for language error.

形式陈述：

```lean
theorem boundary_nll_deriv (z : V → ℝ → ℝ) (dz : V → ℝ) (θ : ℝ)
    (target : V) (hz : ∀ i, HasDerivAt (z i) (dz i) θ) :
    HasDerivAt (fun t => Real.log (∑ i, Real.exp (z i t)) - z target t)
      ((∑ i, CGB.Medium.softmax (fun i => z i θ) i * dz i) - dz target) θ
```

## CGB.GDNOperatorTags.boundary_nll_eq_original

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L323)

解释：The declared real-valued boundary loss is the original negative log softmax, not a new auxiliary target for the interior layers.

源码说明：The declared real-valued boundary loss is the original negative log softmax, not a new auxiliary target for the interior layers.

形式陈述：

```lean
theorem boundary_nll_eq_original (z : V → ℝ) (target : V) :
    Real.log (∑ i, Real.exp (z i)) - z target =
      -Real.log (CGB.Medium.softmax z target)
```

## CGB.GDNOperatorTags.torch_softplus_below

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L339)

解释：Formal contract for torch_softplus_below; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem torch_softplus_below (x : ℝ) (h : x < 20) :
    HasDerivAt torchSoftplus (Real.exp x / (1 + Real.exp x)) x
```

## CGB.GDNOperatorTags.torch_softplus_above

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L347)

解释：Formal contract for torch_softplus_above; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem torch_softplus_above (x : ℝ) (h : 20 < x) :
    HasDerivAt torchSoftplus 1 x
```

## CGB.GDNOperatorTags.scalar_branch_preserved

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L355)

解释：Numerical error must fit inside the actual branch margin before the same differential is used. Equality/ties have no smooth admission here.

源码说明：Numerical error must fit inside the actual branch margin before the same differential is used. Equality/ties have no smooth admission here.

形式陈述：

```lean
theorem scalar_branch_preserved (x approximate threshold error : ℝ)
    (he : |x - approximate| ≤ error) (hm : error < |x - threshold|) :
    (threshold < approximate ↔ threshold < x)
```

## CGB.GDNOperatorTags.normalized_coordinate_gain

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L372)

解释：Shared scalar coordinate row for the actual normalized derivative. r is sqrt(sum+eps), sqrt(sum)+eps, or the active recall clamp denominator; the caller must use the corresponding branch's dr.

源码说明：Shared scalar coordinate row for the actual normalized derivative. r is sqrt(sum+eps), sqrt(sum)+eps, or the active recall clamp denominator; the caller must use the corresponding branch's dr.

形式陈述：

```lean
theorem normalized_coordinate_gain (x dx r dr : ℝ) (hr : 0 < r) :
    |(dx * r - x * dr) / r ^ 2| ≤ |dx| / r + |x| * |dr| / r ^ 2
```

## CGB.GDNOperatorTags.clamped_coordinate_below

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L384)

解释：Recall F.normalize has max(norm,eps), not either native normalizer. Strictly below eps its denominator is constant, including at a zero vector.

源码说明：Recall F.normalize has max(norm,eps), not either native normalizer. Strictly below eps its denominator is constant, including at a zero vector.

形式陈述：

```lean
theorem clamped_coordinate_below (x n : ℝ → ℝ) (dx eps θ : ℝ)
    (hx : HasDerivAt x dx θ) (hn : ContinuousAt n θ) (h : n θ < eps) :
    HasDerivAt (fun t => x t / max (n t) eps) (dx / eps) θ
```

## CGB.GDNOperatorTags.recurrent_zero_remainder

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L397)

解释：The recurrent VECTOR map is differentiable at zero, although norm itself is not. This quadratic remainder also records its large 1/eps^2 sensitivity.

源码说明：The recurrent VECTOR map is differentiable at zero, although norm itself is not. This quadratic remainder also records its large 1/eps^2 sensitivity.

形式陈述：

```lean
theorem recurrent_zero_remainder (eps : ℝ) (heps : 0 < eps) (x : E) :
    ‖recurrentNormalize eps x - eps⁻¹ • x‖ ≤ (eps ^ 2)⁻¹ * ‖x‖ ^ 2
```

## CGB.GDNOperatorTags.recurrent_derivative_at_zero

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L409)

解释：Formal contract for recurrent_derivative_at_zero; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem recurrent_derivative_at_zero (eps : ℝ) (heps : 0 < eps) :
    HasFDerivAt (recurrentNormalize (E
```

## CGB.GDNOperatorTags.reciprocal_error

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_operator_tags_20261108.lean#L423)

解释：Reciprocal error becomes unbounded near zero. A denominator floor is an explicit numerical obligation, not something fixed by U/V compression.

源码说明：Reciprocal error becomes unbounded near zero. A denominator floor is an explicit numerical obligation, not something fixed by U/V compression.

形式陈述：

```lean
theorem reciprocal_error (r approximate floor : ℝ) (hf : 0 < floor)
    (hr : floor ≤ r) (ha : floor ≤ approximate) :
    |r⁻¹ - approximate⁻¹| ≤ |r - approximate| / floor ^ 2
```
