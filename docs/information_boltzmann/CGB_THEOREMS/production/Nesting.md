# Nesting

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/Nesting.lean) · [返回导航](../README.md)

模块源码说明：Nesting: an organism is a part of one medium, its boundary induced by contact There is one medium.

## CGB.Medium.boundary_eq_outer_compl

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L44)

解释：子集S与外部补集直接接触的内部边界单元集合严格等价于补集Sᶜ的外部接触邻域outer。

源码说明：`S`'s own boundary: the units of `S` touching the outside. It is `outer` of the complement.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) (S : Set U),
  {u | u ∈ S ∧ M.touches Sᶜ u} = M.outer Sᶜ
```

## CGB.Medium.participant_near

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L49)

解释：在局域性公理P1b下，任何与S中单元共同参与同一跃迁事件的单元必须位于S内或与S边界相邻。

源码说明：A participant of an event that also has a participant in `S` is in `S` or touches it.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I),
  M.P1b → ∀ {S : Set U} {a : A} {u : U}, u ∈ M.parts a → u ∈ S → ∀ {v : U}, v ∈ M.parts a → v ∈ S ∨ v ∈ M.outer S
```

## CGB.Medium.blanket_of_contact

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L57)

解释：只要两个全局状态在机体S及其接触邻域上完全一致，所有与S相关的局域事件发生率及其对S的作用效果均完全相同（定理T-N1马尔可夫接触毯）。

源码说明：**T-N1 (the blanket from contact)**: two states that agree on `S` and on the units touching it give every event with a participant in `S` the same rate and the same effect on `S`.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I),
  M.P1b →
    ∀ (S : Set U) {x y : U → Λ} (s : U → I),
      (∀ (u : U), u ∈ S ∨ u ∈ M.outer S → x u = y u) →
        ∀ {a : A}, (∃ u ∈ M.parts a, u ∈ S) → M.rate a s x = M.rate a s y ∧ ∀ u ∈ S, M.act a s x u = M.act a s y u
```

## CGB.Medium.untouched_unchanged

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L71)

解释：未包含S中任何单元作为参与者的跃迁事件绝不改变子集S内的物理微观状态。

源码说明：Events with no participant in `S` leave `S` unchanged.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I),
  M.P1b → ∀ (S : Set U) {a : A}, (∀ u ∈ M.parts a, u ∉ S) → ∀ (s : U → I) (x : U → Λ), ∀ u ∈ S, M.act a s x u = x u
```

## CGB.Medium.organism_blanket

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L76)

解释：机体区域O内部的状态变化与跃迁速率仅依赖于O内部状态及与其直接相连的边界单元状态。

源码说明：The organism's half: what happens to `O` reads only `O` and the units touching it.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I),
  M.P1b →
    ∀ (O : Set U) {x y : U → Λ} (s : U → I),
      (∀ (u : U), u ∈ O ∨ u ∈ M.outer O → x u = y u) →
        ∀ {a : A}, (∃ u ∈ M.parts a, u ∈ O) → M.rate a s x = M.rate a s y ∧ ∀ u ∈ O, M.act a s x u = M.act a s y u
```

## CGB.Medium.world_blanket

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L83)

解释：外部世界其余部分的状态演化仅依赖于外部自身以及机体O的接触边界，确立了机体对世界的马尔可夫边界公理。

源码说明：The world's half (formerly the assumption `ReadsBoundary`): what happens to the rest reads only the rest and the organism's own boundary units.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I),
  M.P1b →
    ∀ (O : Set U) {x y : U → Λ} (s : U → I),
      (∀ (u : U), u ∉ O ∨ u ∈ O ∧ M.touches Oᶜ u → x u = y u) →
        ∀ {a : A}, (∃ u ∈ M.parts a, u ∉ O) → M.rate a s x = M.rate a s y ∧ ∀ u ∉ O, M.act a s x u = M.act a s y u
```

## CGB.Medium.exchange_conserves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L96)

解释：封闭跃迁事件中机体区域O内的物质增量严格等于外部环境区域Oᶜ中物质增量的相反数（定理T-N2交换守恒律）。

源码说明：**T-N2 (exchange conserves)**: a closed event with positive rate changes the content of `O` by exactly minus the change of the rest.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : DecidableEq U],
  M.P2a →
    ∀ (O : Finset U) {a : A} {s : U → I} {x : U → Λ},
      M.Closed a →
        0 < M.rate a s x →
          ↑(∑ u ∈ O, M.content (M.act a s x u)) - ↑(∑ u ∈ O, M.content (x u)) =
            -(↑(∑ u ∈ Oᶜ, M.content (M.act a s x u)) - ↑(∑ u ∈ Oᶜ, M.content (x u)))
```

## CGB.JumpProcess.freezeOutside_balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L127)

解释：当联合系统满足细致平衡时，在冻结外部世界状态后机体自身内部跃迁在诱导测度下同样保持细致平衡（定理T-N3）。

源码说明：**T-N3 (the organism's own moves are balanced with the world frozen)**: if the joint calm dynamics has detailed-balance weight `π`, the dynamics with the outside of `O` frozen has the same detailed balance. (G13 for the split into organism and world.)

形式陈述：

```lean
∀ {U : Type u_2} {Λ : Type u_3} (K : CGB.JumpProcess (U → Λ)) (O : Set U) {π : (U → Λ) → ℝ},
  (∀ (x y : U → Λ), π x * K.rate x y = π y * K.rate y x) →
    ∀ (x y : U → Λ), π x * (K.freezeOutside O).rate x y = π y * (K.freezeOutside O).rate y x
```
