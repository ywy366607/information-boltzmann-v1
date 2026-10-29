# SubconditionGraph

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean) · [返回导航](../README.md)

模块源码说明：Structural Graph of the 14 Atomic Sub-conditions This file establishes the formal dependency and redundancy relations among the 14 atomic sub-conditions (A01–A14) from the 6 compound CGB axioms.

## CGB.Medium.p1b_iff_atoms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L61)

解释：P1b 等价于参与位置接触、写入局部、速率读取局部和作用读取局部四个条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I),
  M.P1b ↔ M.ParticipantsContact ∧ M.WritesParticipants ∧ M.RateReadsParticipants ∧ M.EffectReadsParticipants
```

## CGB.Medium.onlyBoundarySenses_iff_atoms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L72)

解释：只有边界感知刺激的条件等价于边界速率与边界作用两个子条件同时成立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I),
  M.OnlyBoundarySenses ↔ M.BoundaryRate ∧ M.BoundaryEffect
```

## CGB.Medium.p3_iff_atoms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L80)

解释：P3 等价于边界速率、边界作用和初始状态不变三个条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I),
  M.P3 ↔ M.BoundaryRate ∧ M.BoundaryEffect ∧ M.InitialInvariant
```

## CGB.Medium.a01_of_finite

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L88)

解释：局部状态类型有限时，任意内容子水平集均有限。

源码说明：In any finite local state space, sublevel sets are automatically finite (A01 is redundant).

形式陈述：

```lean
∀ {Λ : Type u_2} [Finite Λ] (content : Λ → ℕ) (n : ℕ), {l | content l ≤ n}.Finite
```

## CGB.Medium.a02_of_finite

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L94)

解释：局部状态类型有限时，任何自然数内容函数都有统一上界。

源码说明：In any finite local state space, content is automatically bounded (A02 is redundant).

形式陈述：

```lean
∀ {Λ : Type u_2} [Finite Λ] (content : Λ → ℕ), ∃ K, ∀ (l : Λ), content l ≤ K
```

## CGB.Medium.a01_and_a02_redundant_of_finite

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L104)

解释：局部状态类型有限时，每个内容子水平集有限且内容存在统一上界。

源码说明：Under `[Finite Λ]`, A01 and A02 hold unconditionally for ANY content function.

形式陈述：

```lean
∀ {Λ : Type u_2} [Finite Λ] (content : Λ → ℕ), (∀ (n : ℕ), {l | content l ≤ n}.Finite) ∧ ∃ K, ∀ (l : Λ), content l ≤ K
```

## CGB.Medium.uniqueVacuum_independent_of_finite

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L110)

解释：三个有限状态内容函数分别见证没有零态、零态不唯一和唯一零态，说明有限性本身不决定唯一零态。

源码说明：A03 (uniqueVacuum) is strictly independent of finiteness: on `Fin 2`, we can have zero vacuums, multiple vacuums, or a unique vacuum.

形式陈述：

```lean
(¬∃! l, (fun x => 1) l = 0) ∧ (¬∃! l, (fun x => 0) l = 0) ∧ ∃! l, (fun l => if l = 0 then 0 else 1) l = 0
```

## CGB.Medium.farFlip_not_rateReads

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L136)

解释：farFlip 不满足事件速率只读取参与位置的条件。

源码说明：`farFlip` violates rateReadsParticipants (A07).

形式陈述：

```lean
¬CGB.Models.farFlip.RateReadsParticipants
```

## CGB.Medium.farFlip_satisfies_effectReads

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L150)

解释：farFlip 的事件作用满足只读取参与位置的条件。

源码说明：`farFlip` SATISFIES effectReadsParticipants (A08).

形式陈述：

```lean
CGB.Models.farFlip.EffectReadsParticipants
```

## CGB.Medium.farFlip_satisfies_writes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L158)

解释：farFlip 的事件只写入参与位置。

源码说明：`farFlip` satisfies writesParticipants (A06).

形式陈述：

```lean
CGB.Models.farFlip.WritesParticipants
```

## CGB.Medium.farFlip_satisfies_partsContact

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L164)

解释：farFlip 的事件参与位置满足接触条件。

源码说明：`farFlip` satisfies participantsContact (A05).

形式陈述：

```lean
CGB.Models.farFlip.ParticipantsContact
```

## CGB.Medium.farFlip_satisfies_homoContact

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L171)

解释：farFlip 的接触关系满足所定义的均匀性。

源码说明：`farFlip` satisfies homogeneousContact (A04).

形式陈述：

```lean
CGB.Models.farFlip.HomogeneousContact
```

## CGB.Medium.farFlip_isolates_a07

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L176)

解释：farFlip 违反速率只读取参与位置，同时满足列出的其他局部性子条件。

源码说明：Decoupling summary for `farFlip`: it isolates and violates A07 while satisfying A08, A06, A05, A04.

形式陈述：

```lean
¬CGB.Models.farFlip.RateReadsParticipants ∧
  CGB.Models.farFlip.EffectReadsParticipants ∧
    CGB.Models.farFlip.WritesParticipants ∧
      CGB.Models.farFlip.ParticipantsContact ∧ CGB.Models.farFlip.HomogeneousContact
```

## CGB.Medium.farAct_not_effectReads

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L199)

解释：farAct 不满足事件作用只读取参与位置的条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.Medium.farAct.EffectReadsParticipants
```

## CGB.Medium.farAct_satisfies_rateReads

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L214)

解释：farAct 的事件速率满足只读取参与位置的条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Medium.farAct.RateReadsParticipants
```

## CGB.Medium.farAct_satisfies_writes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L218)

解释：farAct 的事件只写入参与位置。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Medium.farAct.WritesParticipants
```

## CGB.Medium.farAct_satisfies_partsContact

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L223)

解释：farAct 的事件参与位置满足接触条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Medium.farAct.ParticipantsContact
```

## CGB.Medium.farAct_satisfies_homoContact

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L229)

解释：farAct 的接触关系满足所定义的均匀性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Medium.farAct.HomogeneousContact
```

## CGB.Medium.farAct_isolates_a08

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L234)

解释：farAct 违反作用只读取参与位置，同时满足列出的其他局部性子条件。

源码说明：Decoupling summary for `farAct`: it isolates and violates A08 while satisfying A07, A06, A05, A04.

形式陈述：

```lean
¬CGB.Medium.farAct.EffectReadsParticipants ∧
  CGB.Medium.farAct.RateReadsParticipants ∧
    CGB.Medium.farAct.WritesParticipants ∧ CGB.Medium.farAct.ParticipantsContact ∧ CGB.Medium.farAct.HomogeneousContact
```

## CGB.Medium.innerClock_not_boundaryRate

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L247)

解释：innerClock 不满足外部刺激只经边界影响速率的条件。

源码说明：`innerClock` violates boundaryRate (A12).

形式陈述：

```lean
¬CGB.Models.innerClock.BoundaryRate
```

## CGB.Medium.innerClock_satisfies_boundaryEffect

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L256)

解释：innerClock 满足外部刺激只经边界影响作用的条件。

源码说明：`innerClock` SATISFIES boundaryEffect (A13).

形式陈述：

```lean
CGB.Models.innerClock.BoundaryEffect
```

## CGB.Medium.innerClock_satisfies_initialInvariant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L261)

解释：innerClock 的初始状态满足重标记不变性。

源码说明：`innerClock` SATISFIES initialInvariant (A14).

形式陈述：

```lean
CGB.Models.innerClock.InitialInvariant
```

## CGB.Medium.innerClock_isolates_a12

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L268)

解释：innerClock 违反边界速率条件，但满足边界作用和初始不变性条件。

源码说明：Decoupling summary for `innerClock`: it isolates and violates A12 while satisfying A13 and A14.

形式陈述：

```lean
¬CGB.Models.innerClock.BoundaryRate ∧ CGB.Models.innerClock.BoundaryEffect ∧ CGB.Models.innerClock.InitialInvariant
```

## CGB.Medium.innerActClock_not_boundaryEffect

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L289)

解释：innerActClock 不满足外部刺激只经边界影响作用的条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.Medium.innerActClock.BoundaryEffect
```

## CGB.Medium.innerActClock_satisfies_boundaryRate

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L299)

解释：innerActClock 满足外部刺激只经边界影响速率的条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Medium.innerActClock.BoundaryRate
```

## CGB.Medium.innerActClock_satisfies_initialInvariant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L303)

解释：innerActClock 的初始状态满足重标记不变性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Medium.innerActClock.InitialInvariant
```

## CGB.Medium.innerActClock_isolates_a13

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L310)

解释：innerActClock 违反边界作用条件，但满足边界速率与初始不变性条件。

源码说明：Decoupling summary for `innerActClock`: it isolates and violates A13 while satisfying A12 and A14.

形式陈述：

```lean
¬CGB.Medium.innerActClock.BoundaryEffect ∧
  CGB.Medium.innerActClock.BoundaryRate ∧ CGB.Medium.innerActClock.InitialInvariant
```

## CGB.Medium.pairPreset_isolates_a14

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L319)

解释：pairPreset 违反初始重标记不变性，但满足边界速率和边界作用条件。

源码说明：`pairPreset` isolates A14: satisfies boundaryRate (A12) and boundaryEffect (A13), but violates A14.

形式陈述：

```lean
¬CGB.Models.pairPreset.InitialInvariant ∧ CGB.Models.pairPreset.BoundaryRate ∧ CGB.Models.pairPreset.BoundaryEffect
```

## CGB.Medium.complete_graph_satisfies_a04_a05

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L339)

解释：若所有位置彼此接触，则接触均匀性和事件参与位置接触条件均成立。

源码说明：Under a complete graph (`near u v ↔ True`), homogeneousContact (A04) and participantsContact (A05) are vacuously satisfied.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I),
  (∀ (u v : U), M.near u v) → M.HomogeneousContact ∧ M.ParticipantsContact
```

## CGB.Medium.universal_parts_satisfies_a06_a07_a08

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L348)

解释：若每个事件参与全部位置，则写入局部及速率、作用读取局部条件自动成立。

源码说明：Under universal participants (`∀ a, parts a = univ`), writesParticipants (A06), rateReadsParticipants (A07), and effectReadsParticipants (A08) are vacuously satisfied.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U],
  (∀ (a : A), M.parts a = Finset.univ) → M.WritesParticipants ∧ M.RateReadsParticipants ∧ M.EffectReadsParticipants
```

## CGB.Medium.all_boundary_satisfies_a12_a13

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L361)

解释：所有位置都是边界时，刺激依赖只来自边界的速率与作用条件自动成立。

源码说明：When boundary = univ, boundaryRate (A12) and boundaryEffect (A13) are vacuously satisfied.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U],
  M.boundary = Finset.univ → M.BoundaryRate ∧ M.BoundaryEffect
```

## CGB.Medium.path3_not_homogeneous

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L381)

解释：指定三位置路径的接触自同构不能把任意位置映到任意另一位置。

源码说明：This specific three-site contact relation is not vertex-transitive. No identification of contact with a realized biological wiring graph is assumed.

形式陈述：

```lean
¬∀ (u v : Fin 3), ∃ σ, (∀ (a b : Fin 3), CGB.Medium.path3Near (σ a) (σ b) ↔ CGB.Medium.path3Near a b) ∧ σ u = v
```

## CGB.Medium.atomized_implies_reducedCore

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L426)

解释：满足全部原子化条件的介质必满足指定精简核心条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : Fintype A] [inst_2 : DecidableEq (U → Λ)], M.Atomized → M.ReducedCore
```

## CGB.Medium.reducedCore_implies_atomized_of_regimes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L433)

解释：在局部状态有限且介质刚性的条件下，精简核心推出完整原子化条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : Fintype A] [inst_2 : DecidableEq (U → Λ)] [Finite Λ], M.Rigid → M.ReducedCore → M.Atomized
```

## CGB.Medium.reducedCore_iff_atomized_of_regimes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L447)

解释：在局部状态有限且介质刚性的条件下，精简核心与完整原子化条件等价。

源码说明：Exact recovery under finite local states and rigid structural data (contact homogeneity is no longer part of the axioms since 2026-10-26).

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : Fintype A] [inst_2 : DecidableEq (U → Λ)] [Finite Λ], M.Rigid → (M.ReducedCore ↔ M.Atomized)
```

## CGB.Medium.path3Rigid_reducedCore

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L467)

解释：指定三位置刚性路径模型满足精简核心条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Medium.path3Rigid.ReducedCore
```

## CGB.Medium.path3Rigid_atomized_not_homogeneous

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L485)

解释：指定三位置刚性路径模型满足原子化条件但接触关系不均匀。

源码说明：Since contact homogeneity left P1b (2026-10-26), the rigid three-unit chain, whose contact is not homogeneous (its ends have one contact, its middle two), is a model; before, it was the countermodel showing that finiteness and rigidity do not recover A04.

形式陈述：

```lean
CGB.Medium.path3Rigid.Atomized ∧ ¬CGB.Medium.path3Rigid.HomogeneousContact
```
