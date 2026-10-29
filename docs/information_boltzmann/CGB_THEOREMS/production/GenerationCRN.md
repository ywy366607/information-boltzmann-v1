# GenerationCRN

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean) · [返回导航](../README.md)

模块源码说明：Well-mixed pair reaction media are admitted (generation chain, M2) One compartment (every unit touches every other: a single clique, which by G8 can integrate anything).

## CGB.PairCRN.sum_pairAct

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L96)

解释：成对化学反应事件对单元求和的改变量仅局限于参与反应的两个接触单元。

源码说明：A pair event changes a unit-wise sum only at its two units.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U] [Fintype R]
  [DecidableEq R] (C : CGB.PairCRN U Λ I R) (g : U → Λ → ℝ) (r : R) {u v : U},
  u ≠ v →
    ∀ (x : U → Λ),
      ∑ w, g w (C.pairAct r u v x w) - ∑ w, g w (x w) =
        g u (C.eff r (C.ident u) (C.ident v) (x u, x v)).1 + g v (C.eff r (C.ident u) (C.ident v) (x u, x v)).2 -
            g u (x u) -
          g v (x v)
```

## CGB.PairCRN.totalE_pairAct

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L117)

解释：两位置不同时且相应储库势为零，成对反应前后总能量之差等于该反应定义的能量差 dE。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U] [Fintype R]
  [DecidableEq R] (C : CGB.PairCRN U Λ I R) (r : R) {u v : U},
  u ≠ v → ∀ (s : U → I) (x : U → Λ), C.reservoir s u v = 0 → C.totalE (C.pairAct r u v x) - C.totalE x = C.dE r u v s x
```

## CGB.PairCRN.p1a

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L125)

解释：若成对反应介质内容函数满足有限子水平集、统一上界和唯一零态，则该介质满足 P1a。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U] [Fintype R]
  [DecidableEq R] (C : CGB.PairCRN U Λ I R),
  ((∀ (n : ℕ), {l | C.content l ≤ n}.Finite) ∧ (∃ K, ∀ (l : Λ), C.content l ≤ K) ∧ ∃! l, C.content l = 0) → C.medium.P1a
```

## CGB.PairCRN.p1b

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L128)

解释：指定成对反应介质满足 P1b，事件读取与写入均受参与位置局部性约束。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U] [Fintype R]
  [DecidableEq R] (C : CGB.PairCRN U Λ I R), C.medium.P1b
```

## CGB.PairCRN.p2a

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L152)

解释：若每个不触及边界身份页的成对反应都保持两端内容之和，则该介质满足闭事件内容守恒 P2a。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U] [Fintype R]
  [DecidableEq R] (C : CGB.PairCRN U Λ I R),
  (∀ (r : R) (i j : ℕ) (p : Λ × Λ),
      i ∉ C.bid →
        j ∉ C.bid → C.content (C.eff r i j p).1 + C.content (C.eff r i j p).2 = C.content p.1 + C.content p.2) →
    C.medium.P2a
```

## CGB.PairCRN.p2b

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L176)

解释：若平静储库势为零、逆反应为对合、正反系数匹配且有效反应可逆，则成对反应介质满足 P2b。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype R] [DecidableEq R] (C : CGB.PairCRN U Λ I R) [inst_4 : DecidableEq (U → Λ)],
  C.μ C.calm = 0 →
    Function.Involutive C.rev →
      (∀ (r : R), C.κ (C.rev r) = C.κ r) →
        (∀ (r : R) (i j : ℕ) (p : Λ × Λ), C.eff r i j p ≠ p → C.eff (C.rev r) i j (C.eff r i j p) = p) → C.medium.P2b
```

## CGB.PairCRN.p2c

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L213)

解释：指定成对反应介质满足 P2c：对每个自同构，同时重标记状态、刺激与事件可保持速率和作用的对应关系。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U] [Fintype R]
  [DecidableEq R] (C : CGB.PairCRN U Λ I R), C.medium.P2c
```

## CGB.PairCRN.p3

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L236)

解释：指定成对反应介质满足 P3 的边界刺激依赖和初始状态不变条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U] [Fintype R]
  [DecidableEq R] (C : CGB.PairCRN U Λ I R), C.medium.P3
```

## CGB.PairCRN.homogeneous

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L256)

解释：指定成对反应介质的接触关系满足所定义的齐次性。

源码说明：The single compartment's contact is homogeneous (an optional model property).

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U] [Fintype R]
  [DecidableEq R] (C : CGB.PairCRN U Λ I R), C.medium.HomogeneousContact
```

## CGB.PairCRN.axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L263)

解释：在定理列出的内容条件、平静储库势、内部反应守恒与逆反应条件下，指定成对反应介质满足 Axioms。

源码说明：**M2**: a well-mixed pair reaction medium is a model of the CGB axioms as soon as its finite rule table passes the local obligations: content levels (P1a), conservation of closed rules (P2a), reversible rules with symmetric prefactors and a zero calm reservoir potential (P2b).

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} {R : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype R] [DecidableEq R] (C : CGB.PairCRN U Λ I R) [inst_4 : DecidableEq (U → Λ)],
  C.μ C.calm = 0 →
    ((∀ (n : ℕ), {l | C.content l ≤ n}.Finite) ∧ (∃ K, ∀ (l : Λ), C.content l ≤ K) ∧ ∃! l, C.content l = 0) →
      (∀ (r : R) (i j : ℕ) (p : Λ × Λ),
          i ∉ C.bid →
            j ∉ C.bid → C.content (C.eff r i j p).1 + C.content (C.eff r i j p).2 = C.content p.1 + C.content p.2) →
        Function.Involutive C.rev →
          (∀ (r : R), C.κ (C.rev r) = C.κ r) →
            (∀ (r : R) (i j : ℕ) (p : Λ × Λ), C.eff r i j p ≠ p → C.eff (C.rev r) i j (C.eff r i j p) = p) →
              C.medium.Axioms
```
