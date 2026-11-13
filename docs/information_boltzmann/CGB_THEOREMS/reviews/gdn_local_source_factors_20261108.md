# gdn_local_source_factors_20261108

探索区：仅按源码盘点；不在生产区全量审计覆盖内，不能据此声称已通过当前构建或信任审计。

[源文件](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean) · [返回导航](../README.md)

模块源码说明：Local source factors of the installed decay-before-delta GDN One reached event, fixed incoming activation/cache, real arithmetic.

## CGB.GDNLocalSourceFactors.column_scale

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L27)

解释：Pull a scalar out of the actual key-column contraction.

源码说明：Pull a scalar out of the actual key-column contraction.

形式陈述：

```lean
theorem column_scale (k x : I → ℝ) (c : ℝ) :
    (∑ a, k a * (c * x a)) = c * dot k x
```

## CGB.GDNLocalSourceFactors.retention_edit_exact

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L54)

解释：Varying retention alone changes every preserved column through the same matrix-valued source. This is exact for finite scalar edits with fixed k.

源码说明：Varying retention alone changes every preserved column through the same matrix-valued source. This is exact for finite scalar edits with fixed k.

形式陈述：

```lean
theorem retention_edit_exact (S : I → J → ℝ) (k : I → ℝ)
    (v : J → ℝ) (α β d : ℝ) (i : I) (j : J) :
    native S k v (α + d) β i j - native S k v α β i j =
      d * retentionSource S k β i j
```

## CGB.GDNLocalSourceFactors.write_edit_exact

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L63)

解释：The write-gate source is an outer product k times the current innovation; all weights of its scalar projection row share this source, not its owners.

源码说明：The write-gate source is an outer product k times the current innovation; all weights of its scalar projection row share this source, not its owners.

形式陈述：

```lean
theorem write_edit_exact (S : I → J → ℝ) (k : I → ℝ)
    (v : J → ℝ) (α β d : ℝ) (i : I) (j : J) :
    native S k v α (β + d) i j - native S k v α β i j =
      d * writeSource S k v α i j
```

## CGB.GDNLocalSourceFactors.value_edit_exact

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L72)

解释：A value direction gives one outer product. Its parameter source rank and the rank of a sum over different events are separate quantities.

源码说明：A value direction gives one outer product. Its parameter source rank and the rank of a sum over different events are separate quantities.

形式陈述：

```lean
theorem value_edit_exact (S : I → J → ℝ) (k : I → ℝ)
    (v dv : J → ℝ) (α β : ℝ) (i : I) (j : J) :
    native S k (fun j => v j + dv j) α β i j - native S k v α β i j =
      β * k i * dv j
```

## CGB.GDNLocalSourceFactors.key_edit_with_remainder

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L81)

解释：A key intervention has two first-order outer products AND a quadratic remainder. The first-order tag must not be advertised as a finite-edit law.

源码说明：A key intervention has two first-order outer products AND a quadratic remainder. The first-order tag must not be advertised as a finite-edit law.

形式陈述：

```lean
theorem key_edit_with_remainder (S : I → J → ℝ) (k dk : I → ℝ)
    (v : J → ℝ) (α β t : ℝ) (i : I) (j : J) :
    native S (fun a => k a + t * dk a) v α β i j =
      native S k v α β i j + t * keySource S k dk v α β i j -
        t ^ 2 * β * dk i * dot dk (fun a => α * S a j)
```

## CGB.GDNLocalSourceFactors.native_direction_source_split

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L93)

解释：The native derivative already proved in gdn_operator_tags is here split into transported old S and four actual local sources, without dropping key, retention or old-state terms. Incoming activation effects enter dk,dv,da,db.

源码说明：The native derivative already proved in gdn_operator_tags is here split into transported old S and four actual local sources, without dropping key, retention or old-state terms. Incoming activation effects enter dk,dv,da,db.

形式陈述：

```lean
theorem native_direction_source_split (S dS : I → J → ℝ)
    (k dk : I → ℝ) (v dv : J → ℝ) (α β da db : ℝ) (i : I) (j : J) :
    da * S i j + α * dS i j +
        (db * k i + β * dk i) * residual S k v α j +
        β * k i * (dv j - ∑ a, (dk a * (α * S a j) +
          k a * (da * S a j + α * dS a j))) =
      α * retentionSource dS k β i j +
        da * retentionSource S k β i j + db * writeSource S k v α i j +
        keySource S k dk v α β i j + β * k i * dv j
```

## CGB.GDNLocalSourceFactors.query_read_source

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L111)

解释：Query is absent from the S update; this does not erase its activation, raw-convolution-cache, downstream-layer or stored-record causal paths.

源码说明：Query is absent from the S update; this does not erase its activation, raw-convolution-cache, downstream-layer or stored-record causal paths.

形式陈述：

```lean
theorem query_read_source (S : I → J → ℝ) (q dq : I → ℝ) (j : J) :
    (∑ i, (q i + dq i) * S i j) - (∑ i, q i * S i j) =
      ∑ i, dq i * S i j
```

## CGB.GDNLocalSourceFactors.erase_injective

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L119)

解释：Retention can preserve all directions: away from the exact erase singularity, x -> x-beta*k*(k dot x) is injective. Thus one gate-source column in effect space does NOT imply a rank-one native S matrix.

源码说明：Retention can preserve all directions: away from the exact erase singularity, x -> x-beta*k*(k dot x) is injective. Thus one gate-source column in effect space does NOT imply a rank-one native S matrix.

形式陈述：

```lean
theorem erase_injective (k : I → ℝ) (β : ℝ)
    (h : 1 - β * dot k k ≠ 0) :
    Function.Injective (fun x : I → ℝ => fun i => x i - β * k i * dot k x)
```

## CGB.GDNLocalSourceFactors.retention_matrix_injective

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L160)

解释：The same injectivity holds columnwise for the whole native matrix. No rank-one compression of the retention source follows from scalar gating.

源码说明：The same injectivity holds columnwise for the whole native matrix. No rank-one compression of the retention source follows from scalar gating.

形式陈述：

```lean
theorem retention_matrix_injective (k : I → ℝ) (β : ℝ)
    (h : 1 - β * dot k k ≠ 0) :
    Function.Injective (fun S : I → J → ℝ => retentionSource S k β)
```

## CGB.GDNLocalSourceFactors.alpha_gate_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L186)

解释：The installed smooth softplus branch: a_proj and dt_bias share rawSlope; A_log has logSlope. The thresholded torch branch requires its own admission.

源码说明：The installed smooth softplus branch: a_proj and dt_bias share rawSlope; A_log has logSlope. The thresholded torch branch requires its own admission.

形式陈述：

```lean
theorem alpha_gate_deriv (A z : ℝ → ℝ) (dA dz θ : ℝ)
    (hA : HasDerivAt A dA θ) (hz : HasDerivAt z dz θ) :
    HasDerivAt (fun t => alphaOf (A t) (z t))
      (rawSlope (A θ) (z θ) * dz + logSlope (A θ) (z θ) * dA) θ
```

## CGB.GDNLocalSourceFactors.native_affine_retention

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L201)

解释：For fixed reached S/key/value, the native update is affine in retention. This connects the actual nonlinear gate derivative to its shared S source.

源码说明：For fixed reached S/key/value, the native update is affine in retention. This connects the actual nonlinear gate derivative to its shared S source.

形式陈述：

```lean
theorem native_affine_retention (S : I → J → ℝ) (k : I → ℝ)
    (v : J → ℝ) (α β : ℝ) (i : I) (j : J) :
    native S k v α β i j =
      α * retentionSource S k β i j + β * k i * v j
```

## CGB.GDNLocalSourceFactors.native_retention_gate_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L210)

解释：Original raw retention gate parameters generate one exact current native source per head. Old-state/input directions are additional transported terms.

源码说明：Original raw retention gate parameters generate one exact current native source per head. Old-state/input directions are additional transported terms.

形式陈述：

```lean
theorem native_retention_gate_deriv (S : I → J → ℝ) (k : I → ℝ)
    (v : J → ℝ) (β : ℝ) (A z : ℝ → ℝ) (dA dz θ : ℝ)
    (hA : HasDerivAt A dA θ) (hz : HasDerivAt z dz θ) (i : I) (j : J) :
    HasDerivAt (fun t => native S k v (alphaOf (A t) (z t)) β i j)
      ((rawSlope (A θ) (z θ) * dz + logSlope (A θ) (z θ) * dA) *
        retentionSource S k β i j) θ
```

## CGB.GDNLocalSourceFactors.write_gate_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L221)

解释：One scalar write gate has the ordinary sigmoid slope, shared by all coordinates in its projection row at the current incoming feature.

源码说明：One scalar write gate has the ordinary sigmoid slope, shared by all coordinates in its projection row at the current incoming feature.

形式陈述：

```lean
theorem write_gate_deriv (b : ℝ → ℝ) (db θ : ℝ)
    (hb : HasDerivAt b db θ) :
    HasDerivAt (fun t => Real.sigmoid (b t))
      (Real.sigmoid (b θ) * (1 - Real.sigmoid (b θ)) * db) θ
```

## CGB.GDNLocalSourceFactors.native_write_gate_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L229)

解释：Write projection weights share the innovation outer product after the sigmoid slope. The current input and old S are held only for this source leg.

源码说明：Write projection weights share the innovation outer product after the sigmoid slope. The current input and old S are held only for this source leg.

形式陈述：

```lean
theorem native_write_gate_deriv (S : I → J → ℝ) (k : I → ℝ)
    (v : J → ℝ) (α : ℝ) (b : ℝ → ℝ) (db θ : ℝ)
    (hb : HasDerivAt b db θ) (i : I) (j : J) :
    HasDerivAt (fun t => native S k v α (Real.sigmoid (b t)) i j)
      ((Real.sigmoid (b θ) * (1 - Real.sigmoid (b θ)) * db) *
        writeSource S k v α i j) θ
```

## CGB.GDNLocalSourceFactors.projection_row_direction

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L248)

解释：Dense row ownership is finite: each incoming feature coordinate supplies its own coefficient even though the downstream effect direction is shared.

源码说明：Dense row ownership is finite: each incoming feature coordinate supplies its own coefficient even though the downstream effect direction is shared.

形式陈述：

```lean
theorem projection_row_direction (w dw x : P → ℝ) :
    HasDerivAt (fun t => ∑ p, (w p + t * dw p) * x p)
      (∑ p, dw p * x p) 0
```

## CGB.GDNLocalSourceFactors.retention_projection_family_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L257)

解释：Combine the actual a_proj row, dt_bias and A_log interventions. This proves their local parameter source from operators instead of assuming it.

源码说明：Combine the actual a_proj row, dt_bias and A_log interventions. This proves their local parameter source from operators instead of assuming it.

形式陈述：

```lean
theorem retention_projection_family_deriv (S : I → J → ℝ) (k : I → ℝ)
    (v : J → ℝ) (β : ℝ) (w dw x : P → ℝ) (A bias dlog dbias : ℝ)
    (i : I) (j : J) :
    HasDerivAt (fun t => native S k v
      (alphaOf (A + t * dlog) ((∑ p, (w p + t * dw p) * x p) + bias + t * dbias)) β i j)
      ((rawSlope A ((∑ p, w p * x p) + bias) * ((∑ p, dw p * x p) + dbias) +
        logSlope A ((∑ p, w p * x p) + bias) * dlog) * retentionSource S k β i j) 0
```

## CGB.GDNLocalSourceFactors.retention_owner_contraction

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L272)

解释：Raw retention-projection weights, dt_bias and A_log all use ONE native retention matrix source. Their distinct coefficients are not summed away.

源码说明：Raw retention-projection weights, dt_bias and A_log all use ONE native retention matrix source. Their distinct coefficients are not summed away.

形式陈述：

```lean
theorem retention_owner_contraction (x dw : P → ℝ) (A z dbias dlog : ℝ) :
    rawSlope A z * ((∑ p, dw p * x p) + dbias) + logSlope A z * dlog =
      (∑ p, dw p * (rawSlope A z * x p)) +
        dbias * rawSlope A z + dlog * logSlope A z
```

## CGB.GDNLocalSourceFactors.projection_row_shared_source

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L298)

解释：All weights in ONE raw projection row share its propagated coordinate source. L must include the raw conv cache as well as current S and output.

源码说明：All weights in ONE raw projection row share its propagated coordinate source. L must include the raw conv cache as well as current S and output.

形式陈述：

```lean
theorem projection_row_shared_source (L : (O → ℝ) →ₗ[ℝ] E)
    (o : O) (x : P → ℝ) (p : P) :
    L (coordinate o (x p)) = x p • L (coordinate o 1)
```

## CGB.GDNLocalSourceFactors.projection_all_rows_factor

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L308)

解释：A dense projected direction is represented by finitely many raw-output sources; this is a one-event factorization, not a bound over a lifetime.

源码说明：A dense projected direction is represented by finitely many raw-output sources; this is a one-event factorization, not a bound over a lifetime.

形式陈述：

```lean
theorem projection_all_rows_factor (L : (O → ℝ) →ₗ[ℝ] E)
    (dw : O → P → ℝ) (x : P → ℝ) :
    L (fun o => ∑ p, dw o p * x p) =
      ∑ o, (∑ p, dw o p * x p) • L (coordinate o 1)
```

## CGB.GDNLocalSourceFactors.shared_source_survives_transport

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L322)

解释：Shared ownership survives frozen successor layers and joint record operations whenever they supply the SAME admitted linearized map.

源码说明：Shared ownership survives frozen successor layers and joint record operations whenever they supply the SAME admitted linearized map.

形式陈述：

```lean
theorem shared_source_survives_transport (A : E →ₗ[ℝ] F)
    (U : E) (owner : P → ℝ) (p : P) :
    A (owner p • U) = owner p • A U
```

## CGB.GDNLocalSourceFactors.shared_source_delayed_capture

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L329)

解释：The delayed third factor is contracted with the shared effect once; ownership distributes the resulting scalar to the original parameters.

源码说明：The delayed third factor is contracted with the shared effect once; ownership distributes the resulting scalar to the original parameters.

形式陈述：

```lean
theorem shared_source_delayed_capture (m : E →ₗ[ℝ] ℝ)
    (U : E) (owner : P → ℝ) (p : P) :
    m (owner p • U) = owner p * m U
```

## CGB.GDNLocalSourceFactors.visible_zero_does_not_erase_cache

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_local_source_factors_20261108.lean#L337)

解释：A raw projection may change persistent cache even when its immediate output differential vanishes. Observing only current S/output is insufficient whenever a future read can distinguish that cache direction.

源码说明：A raw projection may change persistent cache even when its immediate output differential vanishes. Observing only current S/output is insufficient whenever a future read can distinguish that cache direction.

形式陈述：

```lean
theorem visible_zero_does_not_erase_cache (visible : E →ₗ[ℝ] F)
    (future : E →ₗ[ℝ] ℝ) (e : E) (_hv : visible e = 0)
    (hf : future e ≠ 0) : e ≠ 0
```
