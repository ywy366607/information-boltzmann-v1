# GenerationTable

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationTable.lean) · [返回导航](../README.md)

模块源码说明：Rule-table media (generation chain, M3) An organism is a finite table of reversible two-unit transitions: "a pair of units with identity pages `(i, j)` in states `a` may go to states `b`", each with a prefactor `κ`.

## CGB.RuleSpec.apply_back

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationTable.lean#L34)

解释：反应规则的双向应用满足对合性，即沿着正向非平庸改写后再执行反向规则，精确复原原有状态对。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λ : Type u_1} [inst : DecidableEq Λ] (s : CGB.RuleSpec Λ) (dir : Bool) (i j : ℕ) (p : Λ × Λ),
  s.apply dir i j p ≠ p → s.apply (!dir) i j (s.apply dir i j p) = p
```

## CGB.tableCRN.axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationTable.lean#L72)

解释：在静默化学势为零、空位唯一且非边界反应守恒的条件下，查表化学反应网络导出的物理介质严格满足基础 CGB 六公理。

源码说明：**M3**: a rule-table medium is a model of the CGB axioms as soon as its content has a unique free state over a finite local state type, the calm reservoir potential is zero, and every rule between two interior pages conserves content.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_3} [inst : Fintype U] [inst_1 : DecidableEq U] [inst_2 : DecidableEq Λ]
  {n : ℕ} (spec : Fin n → CGB.RuleSpec Λ) (ident : U → ℕ) (bid : Finset ℕ) (content : Λ → ℕ) (φ : ℕ → Λ → ℝ) (μ : I → ℝ)
  (birth : ℕ → Λ) (calm : I) [Finite Λ] [inst_4 : DecidableEq (U → Λ)],
  μ calm = 0 →
    (∃! l, content l = 0) →
      (∀ (k : Fin n),
          (spec k).i ∉ bid →
            (spec k).j ∉ bid →
              content (spec k).a.1 + content (spec k).a.2 = content (spec k).b.1 + content (spec k).b.2) →
        (CGB.tableCRN spec ident bid content φ μ birth calm).medium.Axioms
```
