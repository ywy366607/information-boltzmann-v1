# ConstraintAudit

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean) · [返回导航](../README.md)

模块源码说明：Joint audit of irredundancy and generative freedom No new physical clause is introduced.

## CGB.Medium.axioms_iff_atomized

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L38)

解释：在指定有限性条件下，介质的 Axioms 结构等价于其原子化条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : Fintype A] [inst_2 : DecidableEq (U → Λ)], M.Axioms ↔ M.Atomized
```

## CGB.Medium.p1a_iff_finite_and_unique

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L58)

解释：P1a 等价于局部状态类型有限且零内容状态唯一。

源码说明：The first two cost clauses jointly say finite local state space in this language. Once it is finite, the numerical cost function remains undetermined.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I),
  M.P1a ↔ Finite Λ ∧ ∃! l, M.content l = 0
```

## CGB.Medium.rigid_of_injective_identity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L71)

解释：若位置身份标签是单射，则介质满足所定义的刚性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I),
  Function.Injective M.ident → M.Rigid
```

## CGB.Medium.p2c_of_rigid

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L77)

解释：介质刚性成立时，P2c 重标记等变性成立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I), M.Rigid → M.P2c
```

## CGB.Medium.initial_invariant_of_rigid

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L82)

解释：若介质刚性成立，则初始状态对每个自同构保持不变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I),
  M.Rigid → ∀ (σ : Equiv.Perm U), M.IsAut σ → M.init ∘ ⇑(Equiv.symm σ) = M.init
```

## CGB.Medium.onlyBoundarySenses_of_all_boundary

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L88)

解释：若每个位置都是边界，则介质自动满足只有边界感知条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U],
  M.boundary = Finset.univ → M.OnlyBoundarySenses
```

## CGB.Medium.p2a_of_every_event_open

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L95)

解释：若所有事件都不是闭事件，则 P2a 的闭事件守恒条件真空成立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U],
  (∀ (a : A), ¬M.Closed a) → M.P2a
```

## CGB.Medium.empty_participants_no_effect

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L98)

解释：若事件只写参与位置且某事件参与集为空，则该事件对任意状态没有作用。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I),
  (∀ (a : A) (s : U → I) (x : U → Λ), ∀ u ∉ M.parts a, M.act a s x u = x u) →
    ∀ (a : A), M.parts a = ∅ → ∀ (s : U → I) (x : U → Λ), M.act a s x = x
```

## CGB.Medium.jump_equivariant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L106)

解释：若 P2c 成立，则在任意介质自同构下同步重标记刺激和状态可保持跳跃速率。

源码说明：A genuine joint consequence: event equivariance descends to the aggregate jump kernel, including the single shared event-permutation witness.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype A]
  [inst_1 : DecidableEq (U → Λ)],
  M.P2c →
    ∀ (σ : Equiv.Perm U),
      M.IsAut σ →
        ∀ (s : U → I) (x y : U → Λ),
          (M.jump (s ∘ ⇑(Equiv.symm σ))).rate (x ∘ ⇑(Equiv.symm σ)) (y ∘ ⇑(Equiv.symm σ)) = (M.jump s).rate x y
```

## CGB.Medium.empty_boundary_every_stimulus_reversible

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L124)

解释：空边界、只有边界感知及 P2b 成立时，任意刺激下的跳跃过程都可逆。

源码说明：P3 with no boundary removes ALL stimulus dependence. Combined with P2b, every stimulus then gives the same reversible aggregate kernel.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [DecidableEq U] [Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A],
  M.boundary = ∅ → M.OnlyBoundarySenses → M.P2b → ∀ (s : U → I), (M.jump s).Reversible
```

## CGB.Medium.rescale_jump

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L142)

解释：将介质速率按非负系数缩放后，每个聚合跳跃速率按同一系数缩放。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype A]
  [inst_1 : DecidableEq (U → Λ)] (c : ℝ) (hc : 0 ≤ c) (s : U → I) (x y : U → Λ),
  ((M.rescale c hc).jump s).rate x y = c * (M.jump s).rate x y
```

## CGB.Medium.rescale_axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L150)

解释：将满足公理的介质所有速率乘任意非负系数后，所得介质仍满足这些公理。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : Fintype A] [inst_2 : DecidableEq (U → Λ)] (c : ℝ) (hc : 0 ≤ c), M.Axioms → (M.rescale c hc).Axioms
```

## CGB.Medium.rescale_changes_nonzero_jump

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L173)

解释：若某跳跃速率为正，将介质速率乘二会严格改变该跳跃速率。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype A]
  [inst_1 : DecidableEq (U → Λ)] (s : U → I) (x y : U → Λ),
  0 < (M.jump s).rate x y → ((M.rescale 2 ⋯).jump s).rate x y ≠ (M.jump s).rate x y
```

## CGB.JumpProcess.fromConductance_reversible

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L192)

解释：由正权重和对称非负电导构造的跳跃过程满足可逆性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} (π : S → ℝ) (hp : ∀ (x : S), 0 < π x) (C : S → S → ℝ) (hc : ∀ (x y : S), 0 ≤ C x y),
  (∀ (x y : S), C x y = C y x) → (CGB.JumpProcess.fromConductance π hp C hc).Reversible
```

## CGB.JumpProcess.reversible_conductance_representation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L200)

解释：任意可逆跳跃过程都可表示为对称非负电导除以正平衡权重。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} (P : CGB.JumpProcess S),
  P.Reversible →
    ∃ π,
      (∀ (x : S), 0 < π x) ∧
        ∃ C, (∀ (x y : S), 0 ≤ C x y) ∧ (∀ (x y : S), C x y = C y x) ∧ ∀ (x y : S), P.rate x y = C x y / π x
```
