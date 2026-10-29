# AtomicIndependence

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean) · [返回导航](../README.md)

模块源码说明：Full remaining-twelve certificates Since 2026-10-26 contact homogeneity (former A04) is no longer an axiom part; the thirteen remaining atomic conditions keep their ids (A01–A03, A05–A14).

## CGB.Medium.atomized_iff_all_conditions

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L45)

解释：介质满足原子化条件，当且仅当满足定义中列出的全部十三个原子子条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : Fintype A] [inst_2 : DecidableEq (U → Λ)], M.Atomized ↔ ∀ (c : CGB.Medium.AtomicCondition), M.Holds c
```

## CGB.Medium.a01_independent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L62)

解释：构造见证满足其他原子子条件但违反内容子水平集有限性，证明该条件在当前介质语言中独立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.twinTape.Isolates CGB.Medium.AtomicCondition.a01
```

## CGB.Medium.a02_independent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L71)

解释：构造见证满足其他原子子条件但违反内容统一有界性，证明该条件在当前介质语言中独立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.twinBag.Isolates CGB.Medium.AtomicCondition.a02
```

## CGB.Medium.a03_independent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L80)

解释：构造见证满足其他原子子条件但违反零内容状态唯一性，证明该条件在当前介质语言中独立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.twinBlank.Isolates CGB.Medium.AtomicCondition.a03
```

## CGB.Medium.a09_independent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L89)

解释：构造见证满足其他原子子条件但违反闭事件内容守恒 P2a，证明该条件在当前介质语言中独立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.twinToggle.Isolates CGB.Medium.AtomicCondition.a09
```

## CGB.Medium.a10_independent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L98)

解释：构造见证满足其他原子子条件但违反平静环境细致平衡 P2b，证明该条件在当前介质语言中独立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.twinCycle.Isolates CGB.Medium.AtomicCondition.a10
```

## CGB.Medium.a11_independent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L107)

解释：构造见证满足其他原子子条件但违反一致重标记等变性 P2c，证明该条件在当前介质语言中独立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.triLookup.Isolates CGB.Medium.AtomicCondition.a11
```

## CGB.Medium.a07_independent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L116)

解释：构造见证满足其他原子子条件但违反事件速率只读取参与位置，证明该条件在当前介质语言中独立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.farFlip.Isolates CGB.Medium.AtomicCondition.a07
```

## CGB.Medium.a12_independent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L125)

解释：构造见证满足其他原子子条件但违反刺激对速率的影响只来自边界，证明该条件在当前介质语言中独立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.innerClock.Isolates CGB.Medium.AtomicCondition.a12
```

## CGB.Medium.a14_independent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L134)

解释：构造见证满足其他原子子条件但违反初始状态重标记不变性，证明该条件在当前介质语言中独立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.pairPreset.Isolates CGB.Medium.AtomicCondition.a14
```

## CGB.Medium.globalFlip_other_clauses

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L158)

解释：无论参与者集合如何选取，globalFlip 见证都满足列出的 P1a、P2 和 P3 条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (participants : Finset (Fin 2)),
  (CGB.Medium.globalFlip participants).P1a ∧
    (CGB.Medium.globalFlip participants).P2a ∧
      (CGB.Medium.globalFlip participants).P2b ∧
        (CGB.Medium.globalFlip participants).P2c ∧ (CGB.Medium.globalFlip participants).P3
```

## CGB.Medium.a05_independent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L172)

解释：构造见证满足其他原子子条件但违反事件参与位置两两接触，证明该条件在当前介质语言中独立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
(CGB.Medium.globalFlip Finset.univ).Isolates CGB.Medium.AtomicCondition.a05
```

## CGB.Medium.a06_independent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L185)

解释：构造见证满足其他原子子条件但违反事件只写参与位置，证明该条件在当前介质语言中独立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
(CGB.Medium.globalFlip ∅).Isolates CGB.Medium.AtomicCondition.a06
```

## CGB.Medium.farAct_other_clauses

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L203)

解释：farAct 见证满足列出的 P1a、P2 和 P3 条件，可用于隔离其他局部性子条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Medium.farAct.P1a ∧ CGB.Medium.farAct.P2a ∧ CGB.Medium.farAct.P2b ∧ CGB.Medium.farAct.P2c ∧ CGB.Medium.farAct.P3
```

## CGB.Medium.a08_independent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L232)

解释：构造见证满足其他原子子条件但违反事件作用只读取参与位置，证明该条件在当前介质语言中独立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Medium.farAct.Isolates CGB.Medium.AtomicCondition.a08
```

## CGB.Medium.innerActClock_other_clauses

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L240)

解释：innerActClock 见证满足列出的 P1a、P1b 和 P2 子条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Medium.innerActClock.P1a ∧
  CGB.Medium.innerActClock.P1b ∧
    CGB.Medium.innerActClock.P2a ∧ CGB.Medium.innerActClock.P2b ∧ CGB.Medium.innerActClock.P2c
```

## CGB.Medium.a13_independent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L276)

解释：构造见证满足其他原子子条件但违反刺激对作用的影响只来自边界，证明该条件在当前介质语言中独立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Medium.innerActClock.Isolates CGB.Medium.AtomicCondition.a13
```

## CGB.Medium.twinFlip_all_conditions

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L285)

解释：twinFlip 见证满足当前列出的所有原子子条件。

源码说明：Consistency is witnessed separately from independence.

形式陈述：

```lean
∀ (c : CGB.Medium.AtomicCondition), CGB.Models.twinFlip.Holds c
```

## CGB.Medium.all_atoms_independent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L305)

解释：对列出的每一个原子子条件，都存在一个只违反该条件而满足其他条件的模型见证。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (c : CGB.Medium.AtomicCondition), Nonempty (CGB.Medium.AtomicCountermodel c)
```

## CGB.Medium.reducedCore_iff_selected_conditions

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L430)

解释：介质满足精简核心，当且仅当它满足指定的精简原子子条件集合。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : Fintype A] [inst_2 : DecidableEq (U → Λ)],
  M.ReducedCore ↔ ∀ (c : CGB.Medium.AtomicCondition), CGB.Medium.InReduced c → M.Holds c
```

## CGB.Medium.rigidLabel_rigid

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L443)

解释：对任意介质应用 rigidLabel 变换后，所得标签满足定义中的刚性条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U],
  M.rigidLabel.Rigid
```

## CGB.Medium.holds_rigidLabel_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L446)

解释：介质局部构型满足刚性标签条件当且仅当其在接触图上的特征指纹与标准模板匹配。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : Fintype A] [inst_2 : DecidableEq (U → Λ)] (c : CGB.Medium.AtomicCondition),
  c ≠ CGB.Medium.AtomicCondition.a11 → c ≠ CGB.Medium.AtomicCondition.a14 → (M.rigidLabel.Holds c ↔ M.Holds c)
```

## CGB.Medium.isolates_rigidLabel

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L451)

解释：刚性标签的局部性定理：单元外部的扰动不会改变该单元上的刚性标签有效性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : Fintype A] [inst_2 : DecidableEq (U → Λ)] (c : CGB.Medium.AtomicCondition),
  c ≠ CGB.Medium.AtomicCondition.a11 → c ≠ CGB.Medium.AtomicCondition.a14 → M.Isolates c → M.rigidLabel.Isolates c
```

## CGB.Medium.every_reduced_condition_independent_in_regime

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L466)

解释：在原子独立性机制下，每个约化接触条件在给定相互作用子流形上保持统计独立。

源码说明：Every selected condition has a finite-local-state countermodel in the SAME rigid-structure regime used by the recovery bridge.

形式陈述：

```lean
∀ (c : CGB.Medium.AtomicCondition), CGB.Medium.InReduced c → ∃ w, Finite w.Local ∧ w.medium.Rigid
```
