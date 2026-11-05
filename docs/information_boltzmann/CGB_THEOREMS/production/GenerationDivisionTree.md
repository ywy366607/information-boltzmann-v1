# GenerationDivisionTree

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationDivisionTree.lean) · [返回导航](../README.md)

模块源码说明：What division alone gives a tree (language organism, 2026-11-03) A unit divides into two.

## CGB.DivisionTree.Lineage.leaves_eq_splits_succ

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationDivisionTree.lean#L45)

分类：design；适用范围：candidate；审核：draft；候选：DivisionTree

分类说明：一分为二的谱系组合学（11-03）；不涉及几何，不说明分裂发生在哪里。

解释：Formal contract for leaves_eq_splits_succ; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t : CGB.DivisionTree.Lineage), t.leaves = t.splits + 1
```

## CGB.DivisionTree.Lineage.leaves_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationDivisionTree.lean#L50)

分类：design；适用范围：candidate；审核：draft；候选：DivisionTree

分类说明：一分为二的谱系组合学（11-03）；不涉及几何，不说明分裂发生在哪里。

解释：Formal contract for leaves_pos; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t : CGB.DivisionTree.Lineage), 0 < t.leaves
```

## CGB.DivisionTree.Lineage.leaves_le_two_pow_depth

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationDivisionTree.lean#L53)

分类：design；适用范围：candidate；审核：draft；候选：DivisionTree

分类说明：一分为二的谱系组合学（11-03）；不涉及几何，不说明分裂发生在哪里。

解释：Formal contract for leaves_le_two_pow_depth; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t : CGB.DivisionTree.Lineage), t.leaves ≤ 2 ^ t.depth
```

## CGB.DivisionTree.Lineage.clog_leaves_le_depth

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationDivisionTree.lean#L65)

分类：design；适用范围：candidate；审核：draft；候选：DivisionTree

分类说明：一分为二的谱系组合学（11-03）；不涉及几何，不说明分裂发生在哪里。

解释：Reading `k` parts through two-part units takes at least `⌈log₂ k⌉` levels.

源码说明：Reading `k` parts through two-part units takes at least `⌈log₂ k⌉` levels.

形式陈述：

```lean
∀ (t : CGB.DivisionTree.Lineage), Nat.clog 2 t.leaves ≤ t.depth
```

## CGB.DivisionTree.Lineage.exists_tree_of_clog_depth

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationDivisionTree.lean#L69)

分类：design；适用范围：candidate；审核：draft；候选：DivisionTree

分类说明：一分为二的谱系组合学（11-03）；不涉及几何，不说明分裂发生在哪里。

解释：The depth bound is reached for every number of parts.

源码说明：The depth bound is reached for every number of parts.

形式陈述：

```lean
∀ (k : ℕ), 0 < k → ∃ t, t.leaves = k ∧ t.depth = Nat.clog 2 k
```

## CGB.DivisionTree.sharedPrefix_ultrametric

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationDivisionTree.lean#L94)

分类：design；适用范围：candidate；审核：draft；候选：DivisionTree

分类说明：一分为二的谱系组合学（11-03）；不涉及几何，不说明分裂发生在哪里。

解释：**Nested groups**: sharing `m` divisions is transitive, so each level of the tree partitions the units. No group straddles another.

源码说明：**Nested groups**: sharing `m` divisions is transitive, so each level of the tree partitions the units. No group straddles another.

形式陈述：

```lean
∀ (x y z : List Bool),
  min (CGB.DivisionTree.sharedPrefix x y) (CGB.DivisionTree.sharedPrefix y z) ≤ CGB.DivisionTree.sharedPrefix x z
```

## CGB.DivisionTree.sharedPrefix_comm

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationDivisionTree.lean#L115)

分类：design；适用范围：candidate；审核：draft；候选：DivisionTree

分类说明：一分为二的谱系组合学（11-03）；不涉及几何，不说明分裂发生在哪里。

解释：Formal contract for sharedPrefix_comm; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (x y : List Bool), CGB.DivisionTree.sharedPrefix x y = CGB.DivisionTree.sharedPrefix y x
```
