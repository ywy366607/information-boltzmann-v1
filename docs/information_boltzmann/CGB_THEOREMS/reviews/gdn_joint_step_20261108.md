# gdn_joint_step_20261108

探索区：仅按源码盘点；不在生产区全量审计覆盖内，不能据此声称已通过当前构建或信任审计。

[源文件](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean) · [返回导航](../README.md)

模块源码说明：Joint 24-layer cache plumbing and existing recall ports This is a typed conditional assembly, not equality to a CUDA implementation.

## CGB.GDNJointStep.untouched_cache

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L78)

解释：Structural native-cache locality: an inlet is not permission to overwrite the entire body's recurrent state with an old snapshot.

源码说明：Structural native-cache locality: an inlet is not permission to overwrite the entire body's recurrent state with an old snapshot.

形式陈述：

```lean
theorem untouched_cache (i j : Fin 24) (hne : j ≠ i) (placement : Bool)
    (native : ℝ × (A × C) → A × C) (ε : ℝ)
    (z : Frame A C R) (increment : A) :
    (layerStep i placement native ε z increment).2.1 j = z.2.1 j
```

## CGB.GDNJointStep.read_stage_preserves_reservoir

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L88)

解释：Reading does not itself overwrite stored continuous records. Recording and consolidation must be represented as later explicit transitions.

源码说明：Reading does not itself overwrite stored continuous records. Recording and consolidation must be represented as later explicit transitions.

形式陈述：

```lean
theorem read_stage_preserves_reservoir (i : Fin 24) (placement : Bool)
    (native : ℝ × (A × C) → A × C) (ε : ℝ)
    (z : Frame A C R) (increment : A) :
    (layerStep i placement native ε z increment).2.2 = z.2.2
```

## CGB.GDNJointStep.updated_caches_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L94)

解释：Exact cache replacement derivative, including all untouched layers.

源码说明：Exact cache replacement derivative, including all untouched layers.

形式陈述：

```lean
theorem updated_caches_deriv (cache : ℝ → Fin 24 → C) (e : Fin 24 → C)
    (i : Fin 24) (fresh : ℝ → C) (de : C)
    (hc : HasDerivAt cache e 0) (hf : HasDerivAt fresh de 0) :
    HasDerivAt (fun t => Function.update (cache t) i (fresh t))
      (Function.update e i de) 0
```

## CGB.GDNJointStep.layer_step_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L109)

解释：A complete native block plus its actual recall increment, as a joint activation/cache/reservoir stage. Native derivatives are LOCAL to this block, and the increment derivative is separately supplied by the recall contract.

源码说明：A complete native block plus its actual recall increment, as a joint activation/cache/reservoir stage. Native derivatives are LOCAL to this block, and the increment derivative is separately supplied by the recall contract.

形式陈述：

```lean
theorem layer_step_deriv (i : Fin 24) (placement : Bool)
    (native : ℝ × (A × C) → A × C)
    (D : ℝ × (A × C) →L[ℝ] A × C)
    (z : ℝ → Frame A C R) (e : Frame A C R)
    (increment : ℝ → A) (di : A)
    (hz : HasDerivAt z e 0) (hi : HasDerivAt increment di 0)
    (hn : HasFDerivAt native D
      (0, ((if placement then (z 0).1 else (z 0).1 + increment 0), (z 0).2.1 i))) :
    HasDerivAt (fun t => layerStep i placement native t (z t) (increment t))
      (layerDirection i placement D e di) 0
```

## CGB.GDNJointStep.hook_increment_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L157)

解释：The same hook computes an increment from current pre-block activation and reservoir fields; the parameter source also includes current encoder, replayed native-block and inlet-projection parameters.

源码说明：The same hook computes an increment from current pre-block activation and reservoir fields; the parameter source also includes current encoder, replayed native-block and inlet-projection parameters.

形式陈述：

```lean
theorem hook_increment_deriv (hook : ℝ × (A × R) → A)
    (D : ℝ × (A × R) →L[ℝ] A)
    (z : ℝ → Frame A C R) (e : Frame A C R)
    (hz : HasDerivAt z e 0)
    (hh : HasFDerivAt hook D (0, ((z 0).1, (z 0).2.2))) :
    HasDerivAt (fun t => hook (t, ((z t).1, (z t).2.2)))
      (D (1, (e.1, e.2.2))) 0
```

## CGB.GDNJointStep.outside_ports_no_increment

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L175)

解释：Formal contract for outside_ports_no_increment; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem outside_ports_no_increment (i : Fin 24) (h : ¬ isPort i)
    (hook : ℝ × (A × R) → A) (ε : ℝ) (z : Frame A C R) :
    portIncrement i hook ε z = 0
```

## CGB.GDNJointStep.port_increment_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L181)

解释：Derivative of the original two-port schedule; no extra inlets are hidden inside the coupled directional state.

源码说明：Derivative of the original two-port schedule; no extra inlets are hidden inside the coupled directional state.

形式陈述：

```lean
theorem port_increment_deriv (i : Fin 24) (hook : ℝ × (A × R) → A)
    (D : ℝ × (A × R) →L[ℝ] A)
    (z : ℝ → Frame A C R) (e : Frame A C R)
    (hz : HasDerivAt z e 0)
    (hh : HasFDerivAt hook D (0, ((z 0).1, (z 0).2.2))) :
    HasDerivAt (fun t => portIncrement i hook t (z t))
      (if isPort i then D (1, (e.1, e.2.2)) else 0) 0
```

## CGB.GDNJointStep.hooked_step_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L224)

解释：Formal contract for hooked_step_deriv; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem hooked_step_deriv (i : Fin 24) (z : ℝ → Frame A C R) (e : Frame A C R)
    (hz : HasDerivAt z e 0)
    (hadmit : localAdmission native hook placement nativeD hookD i (z 0)) :
    HasDerivAt (fun t => hookedStep native hook placement i t (z t))
      (hookedDirection placement nativeD hookD i (z 0) e) 0
```

## CGB.GDNJointStep.run_layers_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L260)

解释：Native, hook and joint-cache derivatives assemble along the SAME forward program, including input- and output-placement variants of the old inlet.

源码说明：Native, hook and joint-cache derivatives assemble along the SAME forward program, including input- and output-placement variants of the old inlet.

形式陈述：

```lean
theorem run_layers_deriv (layers : List (Fin 24)) (z : ℝ → Frame A C R)
    (e : Frame A C R) (hz : HasDerivAt z e 0)
    (ha : runAdmission native hook placement nativeD hookD layers (z 0)) :
    HasDerivAt (fun t => runLayers native hook placement layers t (z t))
      (runDirections native hook placement nativeD hookD layers (z 0) e) 0
```

## CGB.GDNJointStep.all_24_layers_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L274)

解释：Specialization to all 24 original layers in order, without introducing new inlets or changing the original LM head.

源码说明：Specialization to all 24 original layers in order, without introducing new inlets or changing the original LM head.

形式陈述：

```lean
theorem all_24_layers_deriv (z : ℝ → Frame A C R) (e : Frame A C R)
    (hz : HasDerivAt z e 0)
    (ha : runAdmission native hook placement nativeD hookD (List.finRange 24) (z 0)) :
    HasDerivAt (fun t => runLayers native hook placement (List.finRange 24) t (z t))
      (runDirections native hook placement nativeD hookD (List.finRange 24) (z 0) e) 0
```

## CGB.GDNJointStep.record_write_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L292)

解释：A record capture has its own causal source. Detaching it in runtime autograd does not prove that a historical intervention could not change the captured contents. Overwrite also replaces the previous record's direction.

源码说明：A record capture has its own causal source. Detaching it in runtime autograd does not prove that a historical intervention could not change the captured contents. Overwrite also replaces the previous record's direction.

形式陈述：

```lean
theorem record_write_deriv (bank : ℝ → Slot → R) (e : Slot → R) (slot : Slot)
    (captured : ℝ → R) (dc : R) (hb : HasDerivAt bank e 0)
    (hc : HasDerivAt captured dc 0) :
    HasDerivAt (fun t => Function.update (bank t) slot (captured t))
      (Function.update e slot dc) 0
```

## CGB.GDNJointStep.record_merge_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L307)

解释：The source/action feature tables also merge observed vectors by their fixed current count. That discrete count is held for this one event, while the old and arriving continuous fields both retain their directions.

源码说明：The source/action feature tables also merge observed vectors by their fixed current count. That discrete count is held for this one event, while the old and arriving continuous fields both retain their directions.

形式陈述：

```lean
theorem record_merge_deriv (old arriving : ℝ → R) (de da : R) (n : ℝ)
    (ho : HasDerivAt old de 0) (ha : HasDerivAt arriving da 0) :
    HasDerivAt (fun t => (1 / (n + 1)) • (n • old t + arriving t))
      ((1 / (n + 1)) • (n • de + da)) 0
```

## CGB.GDNJointStep.complete_tag_coordinate_count

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L315)

解释：Finite continuous record fields also cost K times their coordinate count when K exact coupled tags are retained, independently of life duration.

源码说明：Finite continuous record fields also cost K times their coordinate count when K exact coupled tags are retained, independently of life duration.

形式陈述：

```lean
theorem complete_tag_coordinate_count (K native records : ℕ) :
    K * (native + records) = K * native + K * records
```

## CGB.GDNJointStep.softmax_weight_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L327)

解释：All current selected logits, including a null candidate, can vary. This is the ordinary finite softmax for a fixed occupied support. Invalid slots and the clamp branch must be excluded/handled before using it.

源码说明：All current selected logits, including a null candidate, can vary. This is the ordinary finite softmax for a fixed occupied support. Invalid slots and the clamp branch must be excluded/handled before using it.

形式陈述：

```lean
theorem softmax_weight_deriv [Nonempty I] (score : I → ℝ → ℝ) (ds : I → ℝ)
    (hs : ∀ i, HasDerivAt (score i) (ds i) 0) (i : I) :
    HasDerivAt (fun t => Real.exp (score i t) / ∑ j, Real.exp (score j t))
      ((Real.exp (score i 0) * ds i * (∑ j, Real.exp (score j 0)) -
        Real.exp (score i 0) * (∑ j, Real.exp (score j 0) * ds j)) /
        (∑ j, Real.exp (score j 0)) ^ 2) 0
```

## CGB.GDNJointStep.selected_read_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L341)

解释：Fixed selected identities include the zero-valued null candidate, if present. Raw normalization, score, encoder and replay effects enter dweight or dvalue. A support-changing top-k is not assumed differentiable here.

源码说明：Fixed selected identities include the zero-valued null candidate, if present. Raw normalization, score, encoder and replay effects enter dweight or dvalue. A support-changing top-k is not assumed differentiable here.

形式陈述：

```lean
theorem selected_read_deriv (weight : I → ℝ → ℝ) (value : I → V → ℝ → ℝ)
    (dw : I → ℝ) (dv : I → V → ℝ)
    (hw : ∀ i, HasDerivAt (weight i) (dw i) 0)
    (hv : ∀ i j, HasDerivAt (value i j) (dv i j) 0) (j : V) :
    HasDerivAt (fun t => ∑ i, weight i t * value i j t)
      (∑ i, (dw i * value i j 0 + weight i 0 * dv i j)) 0
```

## CGB.GDNJointStep.held_replay_leg_difference

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L353)

解释：Holding old native replay derivatives does not erase current query, weight or encoder derivatives. Their common terms cancel in this difference; the removed term is the selected OLD-native value leg only.

源码说明：Holding old native replay derivatives does not erase current query, weight or encoder derivatives. Their common terms cancel in this difference; the removed term is the selected OLD-native value leg only.

形式陈述：

```lean
theorem held_replay_leg_difference (w dw value current old : I → ℝ) :
    (∑ i, (dw i * value i + w i * (current i + old i))) -
      (∑ i, (dw i * value i + w i * current i)) = ∑ i, w i * old i
```

## CGB.GDNJointStep.strict_support_locally_stable

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L364)

解释：A strict finite top-k separation persists in a neighborhood of the current point for continuous scores. Tied cutoffs do not satisfy this premise; slot ordering or discrete credit there needs its own convention.

源码说明：A strict finite top-k separation persists in a neighborhood of the current point for continuous scores. Tied cutoffs do not satisfy this premise; slot ordering or discrete credit there needs its own convention.

形式陈述：

```lean
theorem strict_support_locally_stable (score : I → ℝ → ℝ) (selected : I → Bool)
    (hcont : ∀ i, ContinuousAt (score i) 0)
    (hgap : ∀ i j, selected i = true → selected j = false → score j 0 < score i 0) :
    ∀ᶠ t in nhds (0 : ℝ), ∀ i j, selected i = true → selected j = false →
      score j t < score i t
```

## CGB.GDNJointStep.layer_direction_add_transport

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L401)

解释：Existing and incoming directions share one native evaluation law; only the first receives the direct parameter source.

源码说明：Existing and incoming directions share one native evaluation law; only the first receives the direct parameter source.

形式陈述：

```lean
theorem layer_direction_add_transport (i : Fin 24) (placement : Bool)
    (D : ℝ × (A × C) →L[ℝ] A × C) (e f : Frame A C R) (inc extra : A) :
    layerDirection i placement D (e + f) (inc + extra) =
      layerDirection i placement D e inc + layerTransport i placement D f extra
```

## CGB.GDNJointStep.native_partial_gain

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L427)

解释：Local two-by-two gain from the native differential's input/cache legs. No full 24-layer Jacobian or source term is multiplied into each basis column.

源码说明：Local two-by-two gain from the native differential's input/cache legs. No full 24-layer Jacobian or source term is multiplied into each basis column.

形式陈述：

```lean
theorem native_partial_gain
    (D : ℝ × (A × C) →L[ℝ] A × C) (a b c d : ℝ)
    (hInput : ∀ x, ‖(D (0, (x, 0))).1‖ ≤ a * ‖x‖ ∧
      ‖(D (0, (x, 0))).2‖ ≤ c * ‖x‖)
    (hCache : ∀ x, ‖(D (0, (0, x))).1‖ ≤ b * ‖x‖ ∧
      ‖(D (0, (0, x))).2‖ ≤ d * ‖x‖) (x : A) (cache : C) :
    ‖(D (0, (x, cache))).1‖ ≤ a * ‖x‖ + b * ‖cache‖ ∧
      ‖(D (0, (x, cache))).2‖ ≤ c * ‖x‖ + d * ‖cache‖
```

## CGB.GDNJointStep.layer_transport_budget

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L446)

解释：Actual inlet placement changes BOTH output and cache gain. Input recall affects cache writes; output recall changes only the outgoing activation here.

源码说明：Actual inlet placement changes BOTH output and cache gain. Input recall affects cache writes; output recall changes only the outgoing activation here.

形式陈述：

```lean
theorem layer_transport_budget (i : Fin 24) (placement : Bool)
    (D : ℝ × (A × C) →L[ℝ] A × C) (e : Frame A C R) (inc : A)
    (a b c d inputError cacheError incError : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hNative : ∀ x cache, ‖(D (0, (x, cache))).1‖ ≤ a * ‖x‖ + b * ‖cache‖ ∧
      ‖(D (0, (x, cache))).2‖ ≤ c * ‖x‖ + d * ‖cache‖)
    (hInput : ‖e.1‖ ≤ inputError) (hCache : ‖e.2.1 i‖ ≤ cacheError) (hInc : ‖inc‖ ≤ incError) :
    ‖(layerTransport i placement D e inc).1‖ ≤
      a * (if placement then inputError else inputError + incError) + b * cacheError +
        (if placement then incError else 0) ∧
    ‖(layerTransport i placement D e inc).2.1 i‖ ≤
      c * (if placement then inputError else inputError + incError) + d * cacheError
```

## CGB.GDNJointStep.hooked_direction_add_transport

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L484)

解释：Source separation includes the replay/encoder/inlet hook, not merely native S. This is the homogeneous map required by factor transport.

源码说明：Source separation includes the replay/encoder/inlet hook, not merely native S. This is the homogeneous map required by factor transport.

形式陈述：

```lean
theorem hooked_direction_add_transport (i : Fin 24) (z e f : Frame A C R) :
    hookedDirection placement nativeD hookD i z (e + f) =
      hookedDirection placement nativeD hookD i z e + hookedTransport placement nativeD hookD i z f
```

## CGB.GDNJointStep.run_directions_add_transport

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L512)

解释：Assemble source-free transport through the SAME 24-stage program. Subtracting the zero-initial tag isolates transported initial participation.

源码说明：Assemble source-free transport through the SAME 24-stage program. Subtracting the zero-initial tag isolates transported initial participation.

形式陈述：

```lean
theorem run_directions_add_transport (layers : List (Fin 24)) (z e f : Frame A C R) :
    runDirections native hook placement nativeD hookD layers z (e + f) =
      runDirections native hook placement nativeD hookD layers z e +
        runTransports placement nativeD hookD native hook layers z f
```

## CGB.GDNJointStep.joint_source_transport_split

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L525)

解释：The joint parameter source is generated once from zero initial tag; factor columns use only the homogeneous term. Applies in particular to finRange24.

源码说明：The joint parameter source is generated once from zero initial tag; factor columns use only the homogeneous term. Applies in particular to finRange24.

形式陈述：

```lean
theorem joint_source_transport_split (layers : List (Fin 24)) (z e : Frame A C R) :
    runDirections native hook placement nativeD hookD layers z e =
      runDirections native hook placement nativeD hookD layers z 0 +
        runTransports placement nativeD hookD native hook layers z e
```

## CGB.GDNJointStep.run_transport_operator_apply

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L577)

解释：This operator executes exactly the already-separated homogeneous program.

源码说明：This operator executes exactly the already-separated homogeneous program.

形式陈述：

```lean
theorem run_transport_operator_apply (layers : List (Fin 24)) (z e : Frame A C R) :
    runTransportCLM placement nativeD hookD native hook layers z e =
      runTransports placement nativeD hookD native hook layers z e
```

## CGB.GDNJointStep.transport_gain_certificate

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L596)

解释：A finite scalar fallback gain derived from stage composition, rather than assumed as a whole-system constant. Sparse field budgets above can be tighter; the local norms/primitive branches still need runtime admission.

源码说明：A finite scalar fallback gain derived from stage composition, rather than assumed as a whole-system constant. Sparse field budgets above can be tighter; the local norms/primitive branches still need runtime admission.

形式陈述：

```lean
theorem transport_gain_certificate (layers : List (Fin 24)) (z : Frame A C R) :
    ‖runTransportCLM placement nativeD hookD native hook layers z‖ ≤
      transportGain placement nativeD hookD native hook layers z
```

## CGB.GDNJointStep.packed_transport_exec

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L615)

解释：In code coordinates the transport executes precisely the joint program.

源码说明：In code coordinates the transport executes precisely the joint program.

形式陈述：

```lean
theorem packed_transport_exec (pack : E ≃L[ℝ] Frame A C R)
    (layers : List (Fin 24)) (z : Frame A C R) (e : E) :
    pack (packedTransport placement nativeD hookD native hook pack layers z e) =
      runTransports placement nativeD hookD native hook layers z (pack e)
```

## CGB.GDNJointStep.packed_transport_gain

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L625)

解释：A scalar fallback includes coordinate-norm conversion costs. Fieldwise budgets can avoid repeatedly multiplying this coarse whole-state bound.

源码说明：A scalar fallback includes coordinate-norm conversion costs. Fieldwise budgets can avoid repeatedly multiplying this coarse whole-state bound.

形式陈述：

```lean
theorem packed_transport_gain (pack : E ≃L[ℝ] Frame A C R)
    (layers : List (Fin 24)) (z : Frame A C R) :
    ‖packedTransport placement nativeD hookD native hook pack layers z‖ ≤
      ‖pack.symm.toContinuousLinearMap‖ *
        (transportGain placement nativeD hookD native hook layers z * ‖pack.toContinuousLinearMap‖)
```

## CGB.GDNJointStep.commit_fields_deriv

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L661)

解释：The same count/slot and the captured OPEN-boundary derivative are used on the side-state. Runtime detach is not a reason to erase these directions.

源码说明：The same count/slot and the captured OPEN-boundary derivative are used on the side-state. Runtime detach is not a reason to erase these directions.

形式陈述：

```lean
theorem commit_fields_deriv (bank : ℝ → Slot → CapsuleFields Q X B)
    (dbank : Slot → CapsuleFields Q X B) (slot : Slot) (n : ℕ)
    (freshKey : ℝ → Q) (input : ℝ → X) (openBoundary : ℝ → B)
    (dkey : Q) (dinput : X) (dopen : B)
    (hb : HasDerivAt bank dbank 0) (hk : HasDerivAt freshKey dkey 0)
    (hi : HasDerivAt input dinput 0) (ho : HasDerivAt openBoundary dopen 0) :
    HasDerivAt (fun t => commitFields (bank t) slot n (freshKey t) (input t) (openBoundary t))
      (commitFields dbank slot n dkey dinput dopen) 0
```

## CGB.GDNJointStep.replacement_zero_count

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L678)

解释：Replacement n=0 releases the old address AND old capsule direction.

源码说明：Replacement n=0 releases the old address AND old capsule direction.

形式陈述：

```lean
theorem replacement_zero_count (bank : Slot → CapsuleFields Q X B) (slot : Slot)
    (freshKey : Q) (input : X) (openBoundary : B) :
    commitFields bank slot 0 freshKey input openBoundary slot = (freshKey, (input, openBoundary))
```

## CGB.GDNJointStep.commit_untouched_slot

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L686)

解释：All untouched records retain their values/directions with no global copy.

源码说明：All untouched records retain their values/directions with no global copy.

形式陈述：

```lean
theorem commit_untouched_slot (bank : Slot → CapsuleFields Q X B) (slot other : Slot)
    (h : other ≠ slot) (n : ℕ) (freshKey : Q) (input : X) (openBoundary : B) :
    commitFields bank slot n freshKey input openBoundary other = bank other
```

## CGB.GDNJointStep.open_boundary_cannot_be_replaced_by_end

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L695)

解释：Taking the live end-boundary direction instead of the open snapshot is observably a different record whenever those two directions differ.

源码说明：Taking the live end-boundary direction instead of the open snapshot is observably a different record whenever those two directions differ.

形式陈述：

```lean
theorem open_boundary_cannot_be_replaced_by_end (bank : Slot → CapsuleFields Q X B)
    (slot : Slot) (n : ℕ) (freshKey : Q) (input : X) (atOpen atEnd : B)
    (h : atOpen ≠ atEnd) :
    commitFields bank slot n freshKey input atOpen ≠ commitFields bank slot n freshKey input atEnd
```

## CGB.GDNJointStep.committed_version_rejects_old

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L709)

解释：Both merge and replacement invalidate receipts/tags bound to the old slot version. Reusing the same slot index does not preserve event identity.

源码说明：Both merge and replacement invalidate receipts/tags bound to the old slot version. Reusing the same slot index does not preserve event identity.

形式陈述：

```lean
theorem committed_version_rejects_old (versions : Slot → ℕ) (slot : Slot) :
    nextVersions versions slot slot ≠ versions slot
```

## CGB.GDNJointStep.source_mean_error_bound

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L715)

解释：The source mean has a concrete gain: old n/(n+1), arriving 1/(n+1). No all-history or whole-body operator norm is a premise here.

源码说明：The source mean has a concrete gain: old n/(n+1), arriving 1/(n+1). No all-history or whole-body operator norm is a premise here.

形式陈述：

```lean
theorem source_mean_error_bound (n : ℕ) (old oldApprox fresh freshApprox : Q)
    (oldError freshError : ℝ) (ho : ‖old - oldApprox‖ ≤ oldError)
    (hf : ‖fresh - freshApprox‖ ≤ freshError) :
    ‖sourceMean n old fresh - sourceMean n oldApprox freshApprox‖ ≤
      (1 / ((n : ℝ) + 1)) * ((n : ℝ) * oldError + freshError)
```

## CGB.GDNJointStep.source_mean_preserves_budget

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L732)

解释：Averaging does not amplify a common fieldwise error budget. A generic gain >1 per copied record would throw away this important structure.

源码说明：Averaging does not amplify a common fieldwise error budget. A generic gain >1 per copied record would throw away this important structure.

形式陈述：

```lean
theorem source_mean_preserves_budget (n : ℕ) (old oldApprox fresh freshApprox : Q)
    (budget : ℝ) (ho : ‖old - oldApprox‖ ≤ budget) (hf : ‖fresh - freshApprox‖ ≤ budget) :
    ‖sourceMean n old fresh - sourceMean n oldApprox freshApprox‖ ≤ budget
```

## CGB.GDNJointStep.commit_budget_certificate

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L758)

解释：Sparse concrete error handoff: key budgets merge, input/open-boundary budgets overwrite, and all other slots retain their own budgets.

源码说明：Sparse concrete error handoff: key budgets merge, input/open-boundary budgets overwrite, and all other slots retain their own budgets.

形式陈述：

```lean
theorem commit_budget_certificate (full approximate : Slot → CapsuleFields Q X B)
    (r : RecordBudget Slot) (hr : RecordsWithin full approximate r) (slot : Slot) (n : ℕ)
    (key keyApprox : Q) (input inputApprox : X) (openBoundary openApprox : B)
    (keyError inputError openError : ℝ)
    (hk : ‖key - keyApprox‖ ≤ keyError) (hi : ‖input - inputApprox‖ ≤ inputError)
    (ho : ‖openBoundary - openApprox‖ ≤ openError) :
    RecordsWithin (commitFields full slot n key input openBoundary)
      (commitFields approximate slot n keyApprox inputApprox openApprox)
      (commitBudget r slot n keyError inputError openError)
```

## CGB.GDNJointStep.read_direction_centered

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L789)

解释：Actual softmax score credit is centered on the read value; common value shifts cancel. The null candidate must remain in this same sum.

源码说明：Actual softmax score credit is centered on the read value; common value shifts cancel. The null candidate must remain in this same sum.

形式陈述：

```lean
theorem read_direction_centered (w ds : I → ℝ) (value dvalue : I → E) :
    readDirection w ds value dvalue = (∑ i, w i • dvalue i) +
      ∑ i, (w i * ds i) • (value i - weightedRead w value)
```

## CGB.GDNJointStep.read_direction_gain

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L808)

解释：Concrete value/score gain for fixed selected identities. Weights are the real forward weights, NOT a renormalized history-only distribution.

源码说明：Concrete value/score gain for fixed selected identities. Weights are the real forward weights, NOT a renormalized history-only distribution.

形式陈述：

```lean
theorem read_direction_gain (w ds : I → ℝ) (value dvalue : I → E)
    (hw : ∀ i, 0 ≤ w i) :
    ‖readDirection w ds value dvalue‖ ≤
      (∑ i, w i * ‖dvalue i‖) +
        ∑ i, w i * |ds i| * ‖value i - weightedRead w value‖
```

## CGB.GDNJointStep.read_gain_from_local_bounds

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L824)

解释：A checkable gain row from local value and score bounds. No factor equal to the whole reservoir capacity is introduced by summing a selected read.

源码说明：A checkable gain row from local value and score bounds. No factor equal to the whole reservoir capacity is introduced by summing a selected read.

形式陈述：

```lean
theorem read_gain_from_local_bounds (w ds valueGain scoreGain : I → ℝ)
    (value dvalue : I → E) (hw : ∀ i, 0 ≤ w i)
    (hv : ∀ i, ‖dvalue i‖ ≤ valueGain i) (hs : ∀ i, |ds i| ≤ scoreGain i) :
    ‖readDirection w ds value dvalue‖ ≤
      (∑ i, w i * valueGain i) +
        ∑ i, w i * scoreGain i * ‖value i - weightedRead w value‖
```

## CGB.GDNJointStep.read_record_budget_row

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L841)

解释：Actual record-budget-to-read row: source keys determine weights, local input/open-boundary errors determine replay values, and query error is shared. The replay/encoder coefficients are local operator obligations, not a global whole-history gain assumed as one unexplained number.

源码说明：Actual record-budget-to-read row: source keys determine weights, local input/open-boundary errors determine replay values, and query error is shared. The replay/encoder coefficients are local operator obligations, not a global whole-history gain assumed as one unexplained number.

形式陈述：

```lean
theorem read_record_budget_row {Slot : Type*} (slots : I → Slot) (r : RecordBudget Slot)
    (w ds : I → ℝ) (value dvalue : I → E) (hw : ∀ i, 0 ≤ w i)
    (inputGain boundaryGain queryGain keyGain : I → ℝ) (queryError : ℝ)
    (hv : ∀ i, ‖dvalue i‖ ≤ inputGain i * r.input (slots i) + boundaryGain i * r.boundary (slots i))
    (hs : ∀ i, |ds i| ≤ queryGain i * queryError + keyGain i * r.key (slots i)) :
    ‖readDirection w ds value dvalue‖ ≤
      (∑ i, w i * ‖value i - weightedRead w value‖ * queryGain i) * queryError +
        ∑ i, (w i * (inputGain i * r.input (slots i) + boundaryGain i * r.boundary (slots i)) +
          w i * ‖value i - weightedRead w value‖ * keyGain i * r.key (slots i))
```

## CGB.GDNJointStep.address_score_gain

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L867)

解释：Inner-product address score differential, including both query and key paths and an optional additive preference direction, at positive temperature.

源码说明：Inner-product address score differential, including both query and key paths and an optional additive preference direction, at positive temperature.

形式陈述：

```lean
theorem address_score_gain (q k dq dk : E) (temperature dbias : ℝ)
    (ht : 0 < temperature) :
    |(inner ℝ dq k + inner ℝ q dk) / temperature + dbias| ≤
      (‖dq‖ * ‖k‖ + ‖q‖ * ‖dk‖) / temperature + |dbias|
```

## CGB.GDNJointStep.cache_unpack_pack

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L900)

解释：Formal contract for cache_unpack_pack; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem cache_unpack_pack (c : NativeCache) : cacheUnflatten (cacheFlatten c) = c
```

## CGB.GDNJointStep.cache_pack_unpack

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L904)

解释：Formal contract for cache_pack_unpack; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem cache_pack_unpack (v : CacheCoord → ℝ) : cacheFlatten (cacheUnflatten v) = v
```

## CGB.GDNJointStep.frame_unpack_pack

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L935)

解释：Formal contract for frame_unpack_pack; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem frame_unpack_pack {T C : ℕ} (z : FixedFrame T C) :
    frameUnflatten (frameFlatten z) = z
```

## CGB.GDNJointStep.frame_pack_unpack

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L940)

解释：Formal contract for frame_pack_unpack; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem frame_pack_unpack {T C : ℕ} (v : JointCoord T C → ℝ) :
    frameFlatten (frameUnflatten v) = v
```

## CGB.GDNJointStep.fixed_pack_coordinate

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L977)

解释：Formal contract for fixed_pack_coordinate; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem fixed_pack_coordinate {T C : ℕ} (x : JointHilbert T C) (i : JointCoord T C) :
    frameFlatten (fixedPack T C x) i = x i
```

## CGB.GDNJointStep.coordinate_projection_gain

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L987)

解释：Formal contract for coordinate_projection_gain; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem coordinate_projection_gain {T C : ℕ} (i : JointCoord T C) (x : JointHilbert T C) :
    ‖coordinateProjection T C i x‖ ≤ ‖x‖
```

## CGB.GDNJointStep.field_restriction_gain

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L992)

解释：Restriction to an actual field is an L2 contraction, so its gain does not grow merely because the record library has more slots.

源码说明：Restriction to an actual field is an L2 contraction, so its gain does not grow merely because the record library has more slots.

形式陈述：

```lean
theorem field_restriction_gain {T C : ℕ} {J : Type*} [Fintype J]
    (embed : J → JointCoord T C) (hinj : Function.Injective embed)
    (x : JointHilbert T C) :
    ‖WithLp.toLp 2 (fun j => x (embed j))‖ ≤ ‖x‖
```

## CGB.GDNJointStep.record_field_injective

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L1021)

解释：Formal contract for record_field_injective; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem record_field_injective {T C : ℕ} (port : Fin 2) (slot : Fin C) :
    Function.Injective (recordFieldCoord (T
```

## CGB.GDNJointStep.pending_field_injective

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L1026)

解释：Formal contract for pending_field_injective; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem pending_field_injective {T C : ℕ} (port : Fin 2) :
    Function.Injective (pendingFieldCoord (T
```

## CGB.GDNJointStep.cache_coordinate_count

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L1031)

解释：Formal contract for cache_coordinate_count; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem cache_coordinate_count : Fintype.card CacheCoord = 274432
```

## CGB.GDNJointStep.record_coordinate_count

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L1034)

解释：Formal contract for record_coordinate_count; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem record_coordinate_count : Fintype.card RecordCoord = 342016
```

## CGB.GDNJointStep.joint_coordinate_count

[源码位置](../../../../lean/CGBAxioms/reviews/gdn_joint_step_20261108.lean#L1038)

解释：Slot capacity is per port, not a second copy of a total-capacity option.

源码说明：Slot capacity is per port, not a second copy of a total-capacity option.

形式陈述：

```lean
theorem joint_coordinate_count (T C : ℕ) :
    Fintype.card (JointCoord T C) = 1024 * T + 6586368 + (2 * C + 2) * 342016
```
