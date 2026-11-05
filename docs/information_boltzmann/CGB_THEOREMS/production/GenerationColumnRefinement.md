# GenerationColumnRefinement

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnRefinement.lean) · [返回导航](../README.md)

模块源码说明：Column refinement: one body, several readable projections This file does not construct a column organism.

## CGB.Generation.LocalColumnProjections.activity_unchanged_outside

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnRefinement.lean#L83)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入局部柱投影算子、介质动作与局部性公理P1b，经空间动作支持集分离分析，输出动作作用外部件的柱节点内部活动状态在动作施加后严格保持不变。

源码说明：P1b makes a nonparticipant's activity readout unchanged under the same physical event. No independent activity update is permitted.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {H : Type u_5} {C : Type u_6} {S : Type u_7}
  {M : CGB.Medium U Λ A I} (P : CGB.Generation.LocalColumnProjections M) {a : A} {s : U → I}
  {x : CGB.Generation.Body U Λ} {u : U}, M.P1b → u ∉ M.parts a → P.activity (M.act a s x) u = P.activity x u
```

## CGB.Generation.LocalColumnProjections.carrier_unchanged_outside

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnRefinement.lean#L89)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入局部柱投影算子、动作输入与局部性公理P1b，经局部载体支持分析，输出非动作作用范围外部件的底层物质载体在转移后严格不变。

源码说明：The same event leaves a nonparticipant's local causal reading unchanged.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} {H : Type u_5} {C : Type u_6} {S : Type u_7}
  {M : CGB.Medium U Λ A I} (P : CGB.Generation.LocalColumnProjections M) {a : A} {s : U → I}
  {x : CGB.Generation.Body U Λ} {u : U}, M.P1b → u ∉ M.parts a → P.carrier (M.act a s x) u = P.carrier x u
```

## CGB.Generation.LocalColumnProjections.contact_unchanged_outside

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationColumnRefinement.lean#L95)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入局部柱投影算子、介质动作与公理P1b，经外周接触面隔离判定，输出不在动作作用区域内的柱节点接触面状态在系统动作后保持不变。

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

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入物理多柱因果载体，经载体算子抽象提升，输出存在一致的读出算子与推进算子，使编码物理状态精确复现历史依赖的读出与推进契约。

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

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入多柱物理介质载体对象，经历史合并状态等价性检验，输出在编码状态相同时保证目标读出严格一致，且对任意到达输入其后继推进状态依然严格相等。

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
