# GenerationContextBody

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean) · [返回导航](../README.md)

模块源码说明：The context-cell language body as a Medium `ConstraintFreedom.arbitrary_driven_tables_admitted` shows that one exposed site satisfies every clause with an arbitrary table, so a Medium instance says nothing about a body u

## CGB.Generation.ContextBody.sense_mem

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L142)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for sense_mem; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [Fintype (CGB.Generation.ContextBody.Site Cell)] (c : Ctx),
  CGB.Generation.ContextBody.Site.sense ∈ CGB.Generation.ContextBody.touched M c
```

## CGB.Generation.ContextBody.context_mem

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L143)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for context_mem; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [Fintype (CGB.Generation.ContextBody.Site Cell)] (c : Ctx),
  CGB.Generation.ContextBody.Site.context ∈ CGB.Generation.ContextBody.touched M c
```

## CGB.Generation.ContextBody.pool_mem

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L144)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for pool_mem; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [Fintype (CGB.Generation.ContextBody.Site Cell)] (c : Ctx),
  CGB.Generation.ContextBody.Site.pool ∈ CGB.Generation.ContextBody.touched M c
```

## CGB.Generation.ContextBody.cell_mem

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L145)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for cell_mem; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [Fintype (CGB.Generation.ContextBody.Site Cell)] {c : Ctx} {j : Cell},
  j ∈ M.region c → CGB.Generation.ContextBody.Site.cell j ∈ CGB.Generation.ContextBody.touched M c
```

## CGB.Generation.ContextBody.parts_subset

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L148)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for parts_subset; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [Fintype (CGB.Generation.ContextBody.Site Cell)] (a : Ctx × CGB.Generation.ContextBody.Arrival Tick),
  CGB.Generation.ContextBody.parts M a ⊆ CGB.Generation.ContextBody.touched M a.1
```

## CGB.Generation.ContextBody.sense_part

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L157)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for sense_part; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [Fintype (CGB.Generation.ContextBody.Site Cell)] (a : Ctx × CGB.Generation.ContextBody.Arrival Tick),
  CGB.Generation.ContextBody.Site.sense ∈ CGB.Generation.ContextBody.parts M a
```

## CGB.Generation.ContextBody.context_part

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L162)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for context_part; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [Fintype (CGB.Generation.ContextBody.Site Cell)] (a : Ctx × CGB.Generation.ContextBody.Arrival Tick),
  CGB.Generation.ContextBody.Site.context ∈ CGB.Generation.ContextBody.parts M a
```

## CGB.Generation.ContextBody.body_rigid

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L167)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for body_rigid; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [inst_1 : Fintype (CGB.Generation.ContextBody.Site Cell)], (CGB.Generation.ContextBody.body M).Rigid
```

## CGB.Generation.ContextBody.write_outside

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L171)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for write_outside; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [Fintype (CGB.Generation.ContextBody.Site Cell)] (a : Ctx × CGB.Generation.ContextBody.Arrival Tick)
  (x : CGB.Generation.ContextBody.Site Cell → CGB.Generation.ContextBody.Local Ctx Pool Count)
  {u : CGB.Generation.ContextBody.Site Cell},
  u ∉ CGB.Generation.ContextBody.parts M a → CGB.Generation.ContextBody.write M a x u = x u
```

## CGB.Generation.ContextBody.write_congr

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L190)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for write_congr; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [Fintype (CGB.Generation.ContextBody.Site Cell)] (a : Ctx × CGB.Generation.ContextBody.Arrival Tick)
  {x y : CGB.Generation.ContextBody.Site Cell → CGB.Generation.ContextBody.Local Ctx Pool Count},
  (∀ u ∈ CGB.Generation.ContextBody.parts M a, x u = y u) →
    ∀ {u : CGB.Generation.ContextBody.Site Cell},
      u ∈ CGB.Generation.ContextBody.parts M a →
        CGB.Generation.ContextBody.write M a x u = CGB.Generation.ContextBody.write M a y u
```

## CGB.Generation.ContextBody.body_p1b

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L213)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for body_p1b; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [inst_1 : Fintype (CGB.Generation.ContextBody.Site Cell)], (CGB.Generation.ContextBody.body M).P1b
```

## CGB.Generation.ContextBody.body_p2a

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L237)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for body_p2a; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [inst_1 : Fintype (CGB.Generation.ContextBody.Site Cell)], (CGB.Generation.ContextBody.body M).P2a
```

## CGB.Generation.ContextBody.calm_inactive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L241)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for calm_inactive; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [inst_1 : Fintype (CGB.Generation.ContextBody.Site Cell)] (a : Ctx × CGB.Generation.ContextBody.Arrival Tick)
  (x : CGB.Generation.ContextBody.Site Cell → CGB.Generation.ContextBody.Local Ctx Pool Count),
  ¬CGB.Generation.ContextBody.active a (CGB.Generation.ContextBody.body M).calm x
```

## CGB.Generation.ContextBody.body_p2b

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L246)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for body_p2b; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [inst_1 : Fintype (CGB.Generation.ContextBody.Site Cell)] [inst_2 : Fintype Ctx] [inst_3 : Fintype Tick]
  [inst_4 : DecidableEq (CGB.Generation.ContextBody.Site Cell → CGB.Generation.ContextBody.Local Ctx Pool Count)],
  (CGB.Generation.ContextBody.body M).P2b
```

## CGB.Generation.ContextBody.body_p3

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L259)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for body_p3; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [inst_1 : Fintype (CGB.Generation.ContextBody.Site Cell)], (CGB.Generation.ContextBody.body M).P3
```

## CGB.Generation.ContextBody.body_p1a

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L270)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for body_p1a; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [inst_1 : Fintype (CGB.Generation.ContextBody.Site Cell)] [Finite Ctx] [Finite Pool] [Finite Count],
  (CGB.Generation.ContextBody.body M).P1a
```

## CGB.Generation.ContextBody.body_axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L277)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：**O01.** The body with finite unit states satisfies every clause.

源码说明：**O01.** The body with finite unit states satisfies every clause.

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [inst_1 : Fintype (CGB.Generation.ContextBody.Site Cell)] [Finite Ctx] [Finite Pool] [Finite Count]
  [inst_5 : Fintype Ctx] [inst_6 : Fintype Tick]
  [inst_7 : DecidableEq (CGB.Generation.ContextBody.Site Cell → CGB.Generation.ContextBody.Local Ctx Pool Count)],
  (CGB.Generation.ContextBody.body M).Axioms
```

## CGB.Generation.ContextBody.events_touch_bounded

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L285)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：**O22.** An event touches at most `B + 3` units, whatever the cell count, and with more cells than the address reaches, some unit is left untouched.

源码说明：**O22.** An event touches at most `B + 3` units, whatever the cell count, and with more cells than the address reaches, some unit is left untouched.

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [inst_1 : Fintype (CGB.Generation.ContextBody.Site Cell)] (B : ℕ),
  (∀ (c : Ctx), (M.region c).card ≤ B) →
    B + 3 < Fintype.card (CGB.Generation.ContextBody.Site Cell) →
      ∀ (a : Ctx × CGB.Generation.ContextBody.Arrival Tick),
        ((CGB.Generation.ContextBody.body M).parts a).card ≤ B + 3 ∧
          ∃ u, u ∉ (CGB.Generation.ContextBody.body M).parts a
```

## CGB.Generation.ContextBody.modulation_at_boundary

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L314)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：O25：由 P1b 与 P3 的感知部分推出 G9 对上下文身体成立；对该身体接近空成立。

解释：**O25.** Plasticity that depends on the stimulus sits at the boundary (G9, `modulation_needs_boundary`). The premises are `body_p1b` and the sensing half of `body_p3`, so this follows from the Medium instance; it is stated so that the gate can name it. For this body it is close to vacuous: every event already has the sense unit as a part.

源码说明：**O25.** Plasticity that depends on the stimulus sits at the boundary (G9, `modulation_needs_boundary`). The premises are `body_p1b` and the sensing half of `body_p3`, so this follows from the Medium instance; it is stated so that the gate can name it. For this body it is close to vacuous: every event already has the sense unit as a part.

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [inst_1 : Fintype (CGB.Generation.ContextBody.Site Cell)] [inst_2 : Fintype Ctx] [inst_3 : Fintype Tick]
  [inst_4 : DecidableEq (CGB.Generation.ContextBody.Site Cell → CGB.Generation.ContextBody.Local Ctx Pool Count)]
  {s s' : CGB.Generation.ContextBody.Site Cell → Option (CGB.Generation.ContextBody.Arrival Tick)}
  {x y : CGB.Generation.ContextBody.Site Cell → CGB.Generation.ContextBody.Local Ctx Pool Count}
  (D : Set (CGB.Generation.ContextBody.Site Cell)),
  (∀ (u : CGB.Generation.ContextBody.Site Cell), u ∈ D ↔ x u ≠ y u) →
    ((CGB.Generation.ContextBody.body M).jump s).rate x y ≠ ((CGB.Generation.ContextBody.body M).jump s').rate x y →
      ∃ u, (u ∈ D ∨ u ∈ (CGB.Generation.ContextBody.body M).scope D) ∧ u ∈ (CGB.Generation.ContextBody.body M).boundary
```

## CGB.Generation.ContextBody.step_active

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L346)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：Formal contract for step_active; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [inst_1 : Fintype (CGB.Generation.ContextBody.Site Cell)]
  {x : CGB.Generation.ContextBody.Site Cell → CGB.Generation.ContextBody.Local Ctx Pool Count} {c : Ctx},
  x CGB.Generation.ContextBody.Site.context = some (Sum.inl c) →
    ∀ (b : CGB.Generation.ContextBody.Arrival Tick),
      CGB.Generation.ContextBody.step M x b = CGB.Generation.ContextBody.write M (c, b) x
```

## CGB.Generation.ContextBody.order_separated

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L357)

分类：design；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：上下文单元语言身体的 Medium 实例：局部性、读写结构与顺序分离，不声称语言能力。

解释：**O07.** The actual write holds order. If `a` then `b` reaches a cell that `b` then `a` never reaches, and that tally moves the count, the two states differ. This is `context_write_separates_order` for this body.

源码说明：**O07.** The actual write holds order. If `a` then `b` reaches a cell that `b` then `a` never reaches, and that tally moves the count, the two states differ. This is `context_write_separates_order` for this body.

形式陈述：

```lean
∀ {Ctx : Type u_1} {Pool : Type u_2} {Cell : Type u_3} {Count : Type u_4} {Tick : Type u_5}
  (M : CGB.Generation.ContextBody.Model Ctx Pool Cell Count Tick) [inst : DecidableEq Cell]
  [inst_1 : Fintype (CGB.Generation.ContextBody.Site Cell)]
  {x : CGB.Generation.ContextBody.Site Cell → CGB.Generation.ContextBody.Local Ctx Pool Count} {c : Ctx}
  (a b : CGB.Generation.ContextBody.Byte),
  x CGB.Generation.ContextBody.Site.context = some (Sum.inl c) →
    ∀ (j : Cell) {n : Count},
      x (CGB.Generation.ContextBody.Site.cell j) = some (Sum.inr (Sum.inr n)) →
        j ∉ M.region c →
          ∀ (ha : j ∈ M.region (M.advance c a)),
            j ∉ M.region (M.advance c b) →
              (∀ (p : Pool) (cells : ↥(M.region (M.advance c a)) → Count),
                  M.tally (M.advance c a) b p cells ⟨j, ha⟩ ≠ n) →
                List.foldl (CGB.Generation.ContextBody.step M) x [Sum.inl a, Sum.inl b] ≠
                  List.foldl (CGB.Generation.ContextBody.step M) x [Sum.inl b, Sum.inl a]
```

## CGB.Generation.ContextBody.card_site

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L436)

分类：witness；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：满足义务前提的具体小模型，证明相应定理非空。

解释：Formal contract for card_site; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Cell : Type u_3} [inst : Fintype Cell], Fintype.card (CGB.Generation.ContextBody.Site Cell) = Fintype.card Cell + 3
```

## CGB.Generation.ContextBody.suffixOne_is_local

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L453)

分类：witness；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：满足义务前提的具体小模型，证明相应定理非空。

解释：The premises of `events_touch_bounded` are met by a real instance.

源码说明：The premises of `events_touch_bounded` are met by a real instance.

形式陈述：

```lean
∀ (a : CGB.Generation.ContextBody.Byte × CGB.Generation.ContextBody.Arrival Unit),
  ((CGB.Generation.ContextBody.body CGB.Generation.ContextBody.suffixOne).parts a).card ≤ 4 ∧
    ∃ u, u ∉ (CGB.Generation.ContextBody.body CGB.Generation.ContextBody.suffixOne).parts a
```

## CGB.Generation.ContextBody.suffixOne_holds_order

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L459)

分类：witness；适用范围：candidate；审核：draft；候选：ContextBody

分类说明：满足义务前提的具体小模型，证明相应定理非空。

解释：The premises of `order_separated` are met from birth by a real instance.

源码说明：The premises of `order_separated` are met from birth by a real instance.

形式陈述：

```lean
List.foldl (CGB.Generation.ContextBody.step CGB.Generation.ContextBody.suffixOne)
    (CGB.Generation.ContextBody.body CGB.Generation.ContextBody.suffixOne).init [Sum.inl 1, Sum.inl 2] ≠
  List.foldl (CGB.Generation.ContextBody.step CGB.Generation.ContextBody.suffixOne)
    (CGB.Generation.ContextBody.body CGB.Generation.ContextBody.suffixOne).init [Sum.inl 2, Sum.inl 1]
```
