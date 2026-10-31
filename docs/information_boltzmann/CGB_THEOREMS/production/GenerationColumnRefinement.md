# GenerationColumnRefinement

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnRefinement.lean) · [返回导航](../README.md)

模块源码说明：Column refinement: one body, several readable projections This file does not construct a column organism.

## CGB.Generation.LocalColumnProjections.activity_unchanged_outside

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnRefinement.lean#L83)

源码说明：P1b makes a nonparticipant's activity readout unchanged under the same physical event. No independent activity update is permitted.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {H : Type u_5} {C : Type u_6} {S : Type u_7}
  {M : CGB.Medium U Λ A I} (P : CGB.Generation.LocalColumnProjections M) {a : A} {s : U → I}
  {x : CGB.Generation.Body U Λ} {u : U}, M.P1b → u ∉ M.parts a → P.activity (M.act a s x) u = P.activity x u
```

## CGB.Generation.LocalColumnProjections.carrier_unchanged_outside

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnRefinement.lean#L89)

源码说明：The same event leaves a nonparticipant's local causal reading unchanged.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {H : Type u_5} {C : Type u_6} {S : Type u_7}
  {M : CGB.Medium U Λ A I} (P : CGB.Generation.LocalColumnProjections M) {a : A} {s : U → I}
  {x : CGB.Generation.Body U Λ} {u : U}, M.P1b → u ∉ M.parts a → P.carrier (M.act a s x) u = P.carrier x u
```

## CGB.Generation.LocalColumnProjections.contact_unchanged_outside

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnRefinement.lean#L95)

源码说明：A contact reading changes only when at least one endpoint participates.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {H : Type u_5} {C : Type u_6} {S : Type u_7}
  {M : CGB.Medium U Λ A I} (P : CGB.Generation.LocalColumnProjections M) {a : A} {s : U → I}
  {x : CGB.Generation.Body U Λ} {u v : U},
  M.P1b → u ∉ M.parts a → v ∉ M.parts a → P.contact (M.act a s x) u v = P.contact x u v
```

## CGB.Generation.BodyCausalCarrier.satisfies_causal_carrier_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnRefinement.lean#L111)

源码说明：The carrier fields meet the existing exact readout-and-successor contract. This is a use of the actual body step, not a second recurrence.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {History : Type u_5} {Arrival : Type u_6}
  {Readout : Type u_7} {Carrier : Type u_8} {M : CGB.Medium U Λ A I} (K : CGB.Generation.BodyCausalCarrier M),
  ∃ read next,
    (∀ (h : History), read (K.state (K.encode h)) = K.required h) ∧
      ∀ (h : History) (a : Arrival), next (K.state (K.encode h)) a = K.state (K.encode (K.advance h a))
```

## CGB.Generation.BodyCausalCarrier.merged_history_preserves_read_and_successor

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnRefinement.lean#L120)

源码说明：Equivalently, any two histories merged by the carrier agree on every declared readout and on the carrier successor for each arrival.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {History : Type u_5} {Arrival : Type u_6}
  {Readout : Type u_7} {Carrier : Type u_8} {M : CGB.Medium U Λ A I} (K : CGB.Generation.BodyCausalCarrier M),
  (∀ (h h' : History), K.state (K.encode h) = K.state (K.encode h') → K.required h = K.required h') ∧
    ∀ (h h' : History),
      K.state (K.encode h) = K.state (K.encode h') →
        ∀ (a : Arrival), K.state (K.encode (K.advance h a)) = K.state (K.encode (K.advance h' a))
```
