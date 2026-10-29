# GenerationLattice

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean) · [返回导航](../README.md)

模块源码说明：Lattice media: cells with neighbour contact (generation chain, M4; nesting T-E1) The joint medium of an organism nested in its world.

## CGB.Medium.p2b_of_conductance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L39)

解释：具有电导对称性与对数微观平衡的反应介质满足微观可逆性公理P2b。

源码说明：**M1a with conductances**: rates `κ a x · exp (-(E y - E x) / 2)` with `κ` equal on an event and its reverse (read at the respective source states) give P2b.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [Fintype U] [inst : Fintype A]
  [inst_1 : DecidableEq (U → Λ)] (M : CGB.Medium U Λ A I) (E : (U → Λ) → ℝ) (κ : A → (U → Λ) → ℝ) (rev : A → A),
  Function.Involutive rev →
    (∀ (a : A) (x : U → Λ), M.act a M.calm x ≠ x → M.act (rev a) M.calm (M.act a M.calm x) = x) →
      (∀ (a : A) (x : U → Λ), M.act a M.calm x ≠ x → κ (rev a) (M.act a M.calm x) = κ a x) →
        (∀ (a : A) (x : U → Λ),
            M.act a M.calm x ≠ x → M.rate a M.calm x = κ a x * Real.exp (-(E (M.act a M.calm x) - E x) / 2)) →
          M.P2b
```

## CGB.LatticeCRN.sum_update

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L151)

解释：更新一个位置后，有限位置函数总和的变化等于该位置新值减旧值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} [inst : Fintype U] [inst_1 : DecidableEq U] (g : U → Λ → ℝ) (u : U) (l : Λ) (x : U → Λ),
  ∑ w, g w (Function.update x u l w) - ∑ w, g w (x w) = g u l - g u (x u)
```

## CGB.LatticeCRN.sum_update2

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L159)

解释：更新两个不同位置后，有限位置函数总和的变化等于两位置新值之和减旧值之和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} [inst : Fintype U] [inst_1 : DecidableEq U] (g : U → Λ → ℝ) {u v : U},
  u ≠ v →
    ∀ (a b : Λ) (x : U → Λ),
      ∑ w, g w (Function.update (Function.update x u a) v b w) - ∑ w, g w (x w) = g u a + g v b - g u (x u) - g v (x v)
```

## CGB.LatticeCRN.adj_ne

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L167)

解释：指定晶格中相邻的两个位置互不相同。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R₁ : Type u_4} {R₂ : Type u_5} [Fintype U] [DecidableEq U]
  (C : CGB.LatticeCRN U Λ I R₁ R₂) {u v : U}, C.adj u v → u ≠ v
```

## CGB.LatticeCRN.p1a

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L171)

解释：若晶格介质内容函数满足有限子水平集、统一上界和唯一零态，则该介质满足 P1a。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R₁ : Type u_4} {R₂ : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] (C : CGB.LatticeCRN U Λ I R₁ R₂),
  ((∀ (n : ℕ), {l | C.content l ≤ n}.Finite) ∧ (∃ K, ∀ (l : Λ), C.content l ≤ K) ∧ ∃! l, C.content l = 0) → C.medium.P1a
```

## CGB.LatticeCRN.p1b

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L174)

解释：指定晶格反应介质满足 P1b，单体事件局限于参与位置且双体事件局限于相邻参与位置。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R₁ : Type u_4} {R₂ : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] (C : CGB.LatticeCRN U Λ I R₁ R₂), C.medium.P1b
```

## CGB.LatticeCRN.p2a

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L219)

解释：若不触及边界的单体事件保持内容、双体事件保持两端内容之和，则晶格介质满足 P2a。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R₁ : Type u_4} {R₂ : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] (C : CGB.LatticeCRN U Λ I R₁ R₂),
  (∀ (r : R₁) (i : ℕ) (l : Λ), i ∉ C.bid → C.content (C.eff1 r i l) = C.content l) →
    (∀ (r : R₂) (i j : ℕ) (p : Λ × Λ),
        i ∉ C.bid →
          j ∉ C.bid → C.content (C.eff2 r i j p).1 + C.content (C.eff2 r i j p).2 = C.content p.1 + C.content p.2) →
      C.medium.P2a
```

## CGB.LatticeCRN.act1_ne

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L263)

解释：若晶格介质的单体作用改变了整体状态，则参与位置上的局部作用也改变该位置状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R₁ : Type u_4} {R₂ : Type u_5} [Fintype U] [inst : DecidableEq U]
  (C : CGB.LatticeCRN U Λ I R₁ R₂) {r : R₁} {u : U} {x : U → Λ}, C.act1 r u x ≠ x → C.eff1 r (C.ident u) (x u) ≠ x u
```

## CGB.LatticeCRN.act2_ne

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L266)

解释：若两位置相邻且晶格双体作用改变了整体状态，则对应的局部状态对也被改变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R₁ : Type u_4} {R₂ : Type u_5} [Fintype U] [inst : DecidableEq U]
  (C : CGB.LatticeCRN U Λ I R₁ R₂) {r : R₂} {u v : U} {x : U → Λ},
  C.adj u v → C.act2 r u v x ≠ x → C.eff2 r (C.ident u) (C.ident v) (x u, x v) ≠ (x u, x v)
```

## CGB.LatticeCRN.p2b

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L278)

解释：若平静储库势为零且单体、双体有效事件均有对合逆事件和匹配电导，则晶格介质满足 P2b。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R₁ : Type u_4} {R₂ : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] (C : CGB.LatticeCRN U Λ I R₁ R₂) [inst_2 : DecidableEq (U → Λ)] [inst_3 : Fintype R₁]
  [inst_4 : Fintype R₂],
  C.μ C.calm = 0 →
    Function.Involutive C.rev1 →
      Function.Involutive C.rev2 →
        (∀ (r : R₁) (i : ℕ) (l : Λ), C.eff1 r i l ≠ l → C.eff1 (C.rev1 r) i (C.eff1 r i l) = l) →
          (∀ (r : R₂) (i j : ℕ) (p : Λ × Λ), C.eff2 r i j p ≠ p → C.eff2 (C.rev2 r) i j (C.eff2 r i j p) = p) →
            (∀ (r : R₁) (i : ℕ) (l : Λ), C.eff1 r i l ≠ l → C.c1 (C.rev1 r) i (C.eff1 r i l) = C.c1 r i l) →
              (∀ (r : R₂) (i j : ℕ) (p : Λ × Λ),
                  C.eff2 r i j p ≠ p → C.c2 (C.rev2 r) i j (C.eff2 r i j p) = C.c2 r i j p) →
                C.medium.P2b
```

## CGB.LatticeCRN.adj_aut

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L353)

解释：晶格介质的任一自同构保持并反映相邻关系。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R₁ : Type u_4} {R₂ : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] (C : CGB.LatticeCRN U Λ I R₁ R₂) {σ : Equiv.Perm U},
  C.medium.IsAut σ → ∀ (u v : U), C.adj (σ u) (σ v) ↔ C.adj u v
```

## CGB.LatticeCRN.p2c

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L362)

解释：指定晶格反应介质满足 P2c 重标记等变性：介质自同构可对应到保持速率与作用的事件重标记。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R₁ : Type u_4} {R₂ : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] (C : CGB.LatticeCRN U Λ I R₁ R₂), C.medium.P2c
```

## CGB.LatticeCRN.p3

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L396)

解释：指定晶格反应介质满足 P3：外部刺激的速率与作用依赖受边界约束，初态对自同构不变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R₁ : Type u_4} {R₂ : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] (C : CGB.LatticeCRN U Λ I R₁ R₂), C.medium.P3
```

## CGB.LatticeCRN.axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L423)

解释：在定理列出的内容条件、局部守恒、逆事件及正反电导匹配前提下，指定晶格反应介质满足 Axioms。

源码说明：**M4 (lattice media are admitted)**: a lattice reaction medium satisfies the six clauses when its rule tables pass local obligations: content levels (P1a), conservation by rules among interior cells (P2a), a zero calm potential, involutive reverses that undo every effective rule, and conductances equal on a rule and its reverse (P2b). P1b, P2c and P3 hold by construction.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R₁ : Type u_4} {R₂ : Type u_5} [inst : Fintype U]
  [inst_1 : DecidableEq U] (C : CGB.LatticeCRN U Λ I R₁ R₂) [inst_2 : DecidableEq (U → Λ)] [inst_3 : Fintype R₁]
  [inst_4 : Fintype R₂],
  ((∀ (n : ℕ), {l | C.content l ≤ n}.Finite) ∧ (∃ K, ∀ (l : Λ), C.content l ≤ K) ∧ ∃! l, C.content l = 0) →
    (∀ (r : R₁) (i : ℕ) (l : Λ), i ∉ C.bid → C.content (C.eff1 r i l) = C.content l) →
      (∀ (r : R₂) (i j : ℕ) (p : Λ × Λ),
          i ∉ C.bid →
            j ∉ C.bid → C.content (C.eff2 r i j p).1 + C.content (C.eff2 r i j p).2 = C.content p.1 + C.content p.2) →
        C.μ C.calm = 0 →
          Function.Involutive C.rev1 →
            Function.Involutive C.rev2 →
              (∀ (r : R₁) (i : ℕ) (l : Λ), C.eff1 r i l ≠ l → C.eff1 (C.rev1 r) i (C.eff1 r i l) = l) →
                (∀ (r : R₂) (i j : ℕ) (p : Λ × Λ), C.eff2 r i j p ≠ p → C.eff2 (C.rev2 r) i j (C.eff2 r i j p) = p) →
                  (∀ (r : R₁) (i : ℕ) (l : Λ), C.eff1 r i l ≠ l → C.c1 (C.rev1 r) i (C.eff1 r i l) = C.c1 r i l) →
                    (∀ (r : R₂) (i j : ℕ) (p : Λ × Λ),
                        C.eff2 r i j p ≠ p → C.c2 (C.rev2 r) i j (C.eff2 r i j p) = C.c2 r i j p) →
                      C.medium.Axioms
```
