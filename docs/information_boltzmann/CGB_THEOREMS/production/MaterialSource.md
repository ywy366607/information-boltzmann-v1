# MaterialSource

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean) · [返回导航](../README.md)

模块源码说明：Source-resolved material transport Fixed token identities locate material in the body, accessible source, return sink or exterior.

## CGB.MaterialSource.balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L43)

解释：物料源系统的三室物质守恒律：机体、源与汇以及流动通量在任意允许转移中严格守恒。

源码说明：One transport law yields both body balance and the accessible-source budget. Positive gross turnover at constant body count consumes source stock unless there is explicit supply or recycling. No irreversibility is postulated.

形式陈述：

```lean
∀ {C : Type u_1} [inst : Fintype C] (before after : C → CGB.MaterialSource.Place),
  CGB.MaterialSource.Allowed before after →
    CGB.MaterialSource.count after CGB.MaterialSource.Place.body +
          CGB.MaterialSource.moved before after CGB.MaterialSource.Place.body CGB.MaterialSource.Place.sink =
        CGB.MaterialSource.count before CGB.MaterialSource.Place.body +
          CGB.MaterialSource.moved before after CGB.MaterialSource.Place.source CGB.MaterialSource.Place.body ∧
      CGB.MaterialSource.count after CGB.MaterialSource.Place.source +
              CGB.MaterialSource.moved before after CGB.MaterialSource.Place.source CGB.MaterialSource.Place.body +
            CGB.MaterialSource.moved before after CGB.MaterialSource.Place.source CGB.MaterialSource.Place.outside =
          CGB.MaterialSource.count before CGB.MaterialSource.Place.source +
              CGB.MaterialSource.moved before after CGB.MaterialSource.Place.outside CGB.MaterialSource.Place.source +
            CGB.MaterialSource.moved before after CGB.MaterialSource.Place.sink CGB.MaterialSource.Place.source ∧
        CGB.MaterialSource.count after CGB.MaterialSource.Place.body +
                  CGB.MaterialSource.count after CGB.MaterialSource.Place.source +
                CGB.MaterialSource.count after CGB.MaterialSource.Place.sink +
              CGB.MaterialSource.moved before after CGB.MaterialSource.Place.source CGB.MaterialSource.Place.outside +
            CGB.MaterialSource.moved before after CGB.MaterialSource.Place.sink CGB.MaterialSource.Place.outside =
          CGB.MaterialSource.count before CGB.MaterialSource.Place.body +
                CGB.MaterialSource.count before CGB.MaterialSource.Place.source +
              CGB.MaterialSource.count before CGB.MaterialSource.Place.sink +
            CGB.MaterialSource.moved before after CGB.MaterialSource.Place.outside CGB.MaterialSource.Place.source
```

## CGB.MaterialSource.moved_le_stock

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L61)

解释：单步发生的位置间物料转移量严格不超过源位置当前的有效物料存量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {C : Type u_1} [inst : Fintype C] (before after : C → CGB.MaterialSource.Place) (p q : CGB.MaterialSource.Place),
  CGB.MaterialSource.moved before after p q ≤ CGB.MaterialSource.count before p
```

## CGB.MaterialSource.no_intake_from_empty_source

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L69)

解释：当物料源存量为空时，向机体的净摄入转移量严格为零。

源码说明：An empty accessible source cannot feed the body in this tick. Supply first arrives in the source; there is no unregistered exterior-to-body shortcut.

形式陈述：

```lean
∀ {C : Type u_1} [inst : Fintype C] (before after : C → CGB.MaterialSource.Place),
  CGB.MaterialSource.count before CGB.MaterialSource.Place.source = 0 →
    CGB.MaterialSource.moved before after CGB.MaterialSource.Place.source CGB.MaterialSource.Place.body = 0
```

## CGB.MaterialSource.no_transport_without_relocation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L76)

解释：在两点间若没有物理位置变更事件发生，则对应的宏观物料传输通量为零。

源码说明：With identities fixed, an unchanged ownership state cannot be labelled as positive gross transport. A coarse constant-content state may still refine it.

形式陈述：

```lean
∀ {C : Type u_1} [inst : Fintype C] (owner : C → CGB.MaterialSource.Place),
  CGB.MaterialSource.moved owner owner CGB.MaterialSource.Place.source CGB.MaterialSource.Place.body = 0 ∧
    CGB.MaterialSource.moved owner owner CGB.MaterialSource.Place.body CGB.MaterialSource.Place.sink = 0
```

## CGB.MaterialSource.source_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L86)

解释：物料源存量变化严格等于外界补给流入量与流向机体的摄入量之差。

源码说明：Exact finite-prefix stock law, applicable to each supported sample path. It includes both replenishment and recycling instead of asserting that matter must be consumed. All sums are in the existing external tick count.

形式陈述：

```lean
∀ {C : Type u_1} [inst : Fintype C] (path : ℕ → C → CGB.MaterialSource.Place),
  (∀ (t : ℕ), CGB.MaterialSource.Allowed (path t) (path (t + 1))) →
    ∀ (n : ℕ),
      CGB.MaterialSource.count (path n) CGB.MaterialSource.Place.source +
          ∑ t ∈ Finset.range n,
            (CGB.MaterialSource.moved (path t) (path (t + 1)) CGB.MaterialSource.Place.source
                CGB.MaterialSource.Place.body +
              CGB.MaterialSource.moved (path t) (path (t + 1)) CGB.MaterialSource.Place.source
                CGB.MaterialSource.Place.outside) =
        CGB.MaterialSource.count (path 0) CGB.MaterialSource.Place.source +
          ∑ t ∈ Finset.range n,
            (CGB.MaterialSource.moved (path t) (path (t + 1)) CGB.MaterialSource.Place.outside
                CGB.MaterialSource.Place.source +
              CGB.MaterialSource.moved (path t) (path (t + 1)) CGB.MaterialSource.Place.sink
                CGB.MaterialSource.Place.source)
```

## CGB.MaterialSource.intake_bound_without_refill

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L103)

解释：在无外部补给条件下，从物料源转移至机体的总摄入量不超过源的初始物料储量。

源码说明：A finite unreplenished, unrecycled source bounds cumulative intake; this is a material-throughput conclusion, not a death/free-energy theorem.

形式陈述：

```lean
∀ {C : Type u_1} [inst : Fintype C] (path : ℕ → C → CGB.MaterialSource.Place),
  (∀ (t : ℕ), CGB.MaterialSource.Allowed (path t) (path (t + 1))) →
    (∀ (t : ℕ),
        CGB.MaterialSource.moved (path t) (path (t + 1)) CGB.MaterialSource.Place.outside
              CGB.MaterialSource.Place.source +
            CGB.MaterialSource.moved (path t) (path (t + 1)) CGB.MaterialSource.Place.sink
              CGB.MaterialSource.Place.source =
          0) →
      ∀ (n : ℕ),
        ∑ t ∈ Finset.range n,
            CGB.MaterialSource.moved (path t) (path (t + 1)) CGB.MaterialSource.Place.source
              CGB.MaterialSource.Place.body ≤
          CGB.MaterialSource.count (path 0) CGB.MaterialSource.Place.source
```
