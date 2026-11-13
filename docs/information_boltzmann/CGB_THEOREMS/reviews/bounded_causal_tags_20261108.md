# bounded_causal_tags_20261108

探索区：仅按源码盘点；不在生产区全量审计覆盖内，不能据此声称已通过当前构建或信任审计。

[源文件](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean) · [返回导航](../README.md)

模块源码说明：Bounded causal tags, not archived whole-body episodes Design review; not a production/body-coverage claim.

## CGB.BoundedCausalTags.tag_does_not_inject

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L45)

解释：Full/held tags have the same physical one-step forward by construction.

源码说明：Full/held tags have the same physical one-step forward by construction.

形式陈述：

```lean
theorem tag_does_not_inject (F : ℝ × E → E) (D : ℝ × E →L[ℝ] E)
    (z e e' : E) : (taggedStep F D z e).1 = (taggedStep F D z e').1
```

## CGB.BoundedCausalTags.one_step_source_transport

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L51)

解释：The same current differential splits into a new participation source and the transported past tag. A source-free future event sets the first term to zero; it must not re-inject an isolated old participation every time.

源码说明：The same current differential splits into a new participation source and the transported past tag. A source-free future event sets the first term to zero; it must not re-inject an isolated old participation every time.

形式陈述：

```lean
theorem one_step_source_transport (D : ℝ × E →L[ℝ] E) (e : E) :
    D (1, e) = D (1, 0) + D (0, e)
```

## CGB.BoundedCausalTags.coupled_tag_exact

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L59)

解释：Arbitrary finite causal age: one-step derivatives generate the delayed direction by forward transport, with no stored trajectory and no BPTT tape.

源码说明：Arbitrary finite causal age: one-step derivatives generate the delayed direction by forward transport, with no stored trajectory and no BPTT tape.

形式陈述：

```lean
theorem coupled_tag_exact (F : ℕ → ℝ × E → E) (initial : ℝ → E)
    (D : ℕ → (ℝ × E →L[ℝ] E)) (e₀ : E)
    (h₀ : HasDerivAt initial e₀ 0)
    (hstep : ∀ n, HasFDerivAt (F n) (D n) (0, path F initial 0 n)) (T : ℕ) :
    HasDerivAt (fun ε => path F initial ε T) (tag D e₀ T) 0
```

## CGB.BoundedCausalTags.delayed_loss_capture

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L74)

解释：A later REAL observation supplies the loss differential. Its contraction with the transported tag is exact for the declared intervention. No loss on an old answer, oracle record, or new output-injection path is introduced.

源码说明：A later REAL observation supplies the loss differential. Its contraction with the transported tag is exact for the declared intervention. No loss on an old answer, oracle record, or new output-injection path is introduced.

形式陈述：

```lean
theorem delayed_loss_capture (F : ℕ → ℝ × E → E) (initial : ℝ → E)
    (D : ℕ → (ℝ × E →L[ℝ] E)) (e₀ : E)
    (h₀ : HasDerivAt initial e₀ 0)
    (hstep : ∀ n, HasFDerivAt (F n) (D n) (0, path F initial 0 n))
    (T : ℕ) (loss : E → ℝ) (modulation : E →L[ℝ] ℝ)
    (hLoss : HasFDerivAt loss modulation (path F initial 0 T)) :
    HasDerivAt (fun ε => loss (path F initial ε T))
      (modulation (tag D e₀ T)) 0
```

## CGB.BoundedCausalTags.transport_sources_split

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L102)

解释：Partitioning participation sources partitions the final credit. This is the algebra required to add only OUTSIDE-window sources to a current BPTT gradient, instead of counting the inside-window participation twice. The actual scheduler must supply the declared disjoint age partition.

源码说明：Partitioning participation sources partitions the final credit. This is the algebra required to add only OUTSIDE-window sources to a current BPTT gradient, instead of counting the inside-window participation twice. The actual scheduler must supply the declared disjoint age partition.

形式陈述：

```lean
theorem transport_sources_split (A : ℕ → E →ₗ[ℝ] E) (old recent : ℕ → E)
    (initial : E) (T : ℕ) :
    transport A (fun n => old n + recent n) initial T =
      transport A old initial T + transport A recent 0 T
```

## CGB.BoundedCausalTags.capture_sources_split

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L115)

解释：Arrival modulation contracts the SAME source partition. This is a conditional no-double-counting identity, not a proof that an existing BPTT implementation has already used the identical masks and boundary.

源码说明：Arrival modulation contracts the SAME source partition. This is a conditional no-double-counting identity, not a proof that an existing BPTT implementation has already used the identical masks and boundary.

形式陈述：

```lean
theorem capture_sources_split (A : ℕ → E →ₗ[ℝ] E) (old recent : ℕ → E)
    (initial : E) (read : E →ₗ[ℝ] ℝ) (T : ℕ) :
    read (transport A (fun n => old n + recent n) initial T) =
      read (transport A old initial T) + read (transport A recent 0 T)
```

## CGB.BoundedCausalTags.compressed_transport_exact

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L124)

解释：Exact compression is permitted when the retained coordinates are closed under actual transport AND retain the participation source. This is a checked commuting condition, not the assumption that two chosen layers are enough.

源码说明：Exact compression is permitted when the retained coordinates are closed under actual transport AND retain the participation source. This is a checked commuting condition, not the assumption that two chosen layers are enough.

形式陈述：

```lean
theorem compressed_transport_exact (A : ℕ → E →ₗ[ℝ] E) (b : ℕ → E)
    (initial : E) (P : E →ₗ[ℝ] Q) (smallA : ℕ → Q →ₗ[ℝ] Q)
    (smallb : ℕ → Q)
    (hclosed : ∀ n x, P (A n x) = smallA n (P x))
    (hsource : ∀ n, P (b n) = smallb n) (T : ℕ) :
    transport smallA smallb (P initial) T = P (transport A b initial T)
```

## CGB.BoundedCausalTags.compressed_capture_exact

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L136)

解释：Compression must also preserve what the arriving outcome reads. A local trace can satisfy its recurrence yet still lose the delayed credit.

源码说明：Compression must also preserve what the arriving outcome reads. A local trace can satisfy its recurrence yet still lose the delayed credit.

形式陈述：

```lean
theorem compressed_capture_exact (A : ℕ → E →ₗ[ℝ] E) (b : ℕ → E)
    (initial : E) (P : E →ₗ[ℝ] Q) (smallA : ℕ → Q →ₗ[ℝ] Q)
    (smallb : ℕ → Q) (read : E →ₗ[ℝ] ℝ) (smallRead : Q →ₗ[ℝ] ℝ)
    (hclosed : ∀ n x, P (A n x) = smallA n (P x))
    (hsource : ∀ n, P (b n) = smallb n)
    (hread : ∀ x, read x = smallRead (P x)) (T : ℕ) :
    smallRead (transport smallA smallb (P initial) T) =
      read (transport A b initial T)
```

## CGB.BoundedCausalTags.credit_code_exists_iff

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L150)

解释：Exact credit through a finite code exists iff histories collapsed to the same code require the same credit. Whole-body snapshots are not a premise.

源码说明：Exact credit through a finite code exists iff histories collapsed to the same code require the same credit. Whole-body snapshots are not a premise.

形式陈述：

```lean
theorem credit_code_exists_iff {H Q C : Type*} (encode : H → Q)
    (required : H → C) (fallback : C) :
    (∃ read : Q → C, ∀ h, read (encode h) = required h) ↔
      (∀ h h', encode h = encode h' → required h = required h')
```

## CGB.BoundedCausalTags.local_zero_does_not_mean_no_credit

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L163)

解释：An old lower-layer intervention remains in the upper recurrent state after its lower-layer trace has become zero.

源码说明：An old lower-layer intervention remains in the upper recurrent state after its lower-layer trace has become zero.

形式陈述：

```lean
theorem local_zero_does_not_mean_no_credit :
    (upperCarry (1, 0) - upperCarry (0, 0)).1 = 0 ∧
      (upperCarry (1, 0) - upperCarry (0, 0)).2 = 1
```

## CGB.BoundedCausalTags.lower_only_capture_impossible

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L170)

解释：No reader of the lower coordinate alone can recover every upper effect. This is the missing read condition in a layer-only compression.

源码说明：No reader of the lower coordinate alone can recover every upper effect. This is the missing read condition in a layer-only compression.

形式陈述：

```lean
theorem lower_only_capture_impossible :
    ¬ ∃ read : ℝ → ℝ, ∀ x : ℝ × ℝ, read (upperCarry x).1 = (upperCarry x).2
```

## CGB.BoundedCausalTags.finite_tag_state_count

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L181)

解释：Count of finite-precision tag arrays: independent of lifetime. This counts K*d coordinates, not parameter sensitivities for every parameter. The actual runtime must separately account for bit width, masters and workspace.

源码说明：Count of finite-precision tag arrays: independent of lifetime. This counts K*d coordinates, not parameter sensitivities for every parameter. The actual runtime must separately account for bit width, masters and workspace.

形式陈述：

```lean
theorem finite_tag_state_count (K d q : ℕ) :
    Fintype.card (Fin K → Fin d → Fin q) = (q ^ d) ^ K
```

## CGB.BoundedCausalTags.native_gap_transport

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L198)

解释：The exact homogeneous transport of an old participation direction when future keys/gates/values are held. In a stacked model, other induced changes are additional coupled sources; this lemma does not freeze them by fiat.

源码说明：The exact homogeneous transport of an old participation direction when future keys/gates/values are held. In a stacked model, other induced changes are additional coupled sources; this lemma does not freeze them by fiat.

形式陈述：

```lean
theorem native_gap_transport (k x y : Fin n → ℝ) (v α β : ℝ) :
    nativeStep k x v α β - nativeStep k y v α β =
      fun i => α * deltaStep k β (x - y) i
```

## CGB.BoundedCausalTags.retention_source

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L211)

解释：Retention participation is the ACTUAL old state after selective erase, not just a stored word or a scalar measure of activity.

源码说明：Retention participation is the ACTUAL old state after selective erase, not just a stored word or a scalar measure of activity.

形式陈述：

```lean
theorem retention_source (k x : Fin n → ℝ) (v α β ε : ℝ) :
    nativeStep k x v (α + ε) β - nativeStep k x v α β =
      fun i => ε * deltaStep k β x i
```

## CGB.BoundedCausalTags.write_gate_source

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L220)

解释：The write/erase gate source contains the old prediction error. A rule that stores only the key omits a required local distinction.

源码说明：The write/erase gate source contains the old prediction error. A rule that stores only the key omits a required local distinction.

形式陈述：

```lean
theorem write_gate_source (k x : Fin n → ℝ) (v α β ε : ℝ) :
    nativeStep k x v α (β + ε) - nativeStep k x v α β =
      fun i => ε * k i * (v - α * dot k x)
```

## CGB.BoundedCausalTags.value_source

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L229)

解释：A value direction has a rank-one key-side source. This does not assert that retention and key/projection derivatives are all rank one.

源码说明：A value direction has a rank-one key-side source. This does not assert that retention and key/projection derivatives are all rank one.

形式陈述：

```lean
theorem value_source (k x : Fin n → ℝ) (v α β ε : ℝ) :
    nativeStep k x (v + ε) α β - nativeStep k x v α β =
      fun i => ε * β * k i
```

## CGB.BoundedCausalTags.orthogonal_tag_holds

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L238)

解释：Orthogonal future writes leave an old direct-path tag unchanged when retention is one. Many steps need not destroy every old distinction.

源码说明：Orthogonal future writes leave an old direct-path tag unchanged when retention is one. Many steps need not destroy every old distinction.

形式陈述：

```lean
theorem orthogonal_tag_holds (k e : Fin n → ℝ) (β : ℝ)
    (horth : dot k e = 0) : (fun i => (1 : ℝ) * deltaStep k β e i) = e
```

## CGB.BoundedCausalTags.aligned_tag_erased

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L245)

解释：Conversely, a unit write along a unit key can erase its direct-path tag exactly. Unlimited causal AGE is not unlimited guaranteed retention.

源码说明：Conversely, a unit write along a unit key can erase its direct-path tag exactly. Unlimited causal AGE is not unlimited guaranteed retention.

形式陈述：

```lean
theorem aligned_tag_erased (k : Fin n → ℝ) (hk : dot k k = 1) :
    deltaStep k 1 k = 0
```

## CGB.BoundedCausalTags.capture_decreases_of_margin

[源码位置](../../../../lean/CGBAxioms/reviews/bounded_causal_tags_20261108.lean#L255)

解释：A capture is not a guarantee of improvement. This sufficient condition spells out what an approximate tag must beat, including omitted-path error. The smooth-loss upper bound is a substantive premise for the actual update.

源码说明：A capture is not a guarantee of improvement. This sufficient condition spells out what an approximate tag must beat, including omitted-path error. The smooth-loss upper bound is a substantive premise for the actual update.

形式陈述：

```lean
theorem capture_decreases_of_margin (before after g e η L : ℝ)
    (hbound : after ≤ before - η * g * e + L * (η * e) ^ 2 / 2)
    (hη : 0 < η) (hmargin : 0 < g * e - L * η * e ^ 2 / 2) :
    after < before
```
