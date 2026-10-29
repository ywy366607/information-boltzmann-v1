# RenewableLearningCarrier

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean) · [返回导航](../README.md)

模块源码说明：Source and recovery certificates for the unchanged renewable-learning process Fixed material identities realize every supported normal/cut transition.

## _private.CGBAxioms.RenewableLearningCarrier.0.CGB.RenewableLearningCarrier.body_supported

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L23)

解释：在可再生学习载体系统中，机体支持状态的微观所有权计数严格等于机体宏观质量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (t : ℕ) (z z' : CGB.RenewableLearning.S) (e : Option Unit),
  0 < (CGB.RenewableLearning.process true true cut).step t z e z' →
    0 <
      (CGB.RenewableLearning.plant true true).eventStep CGB.RenewableLearning.world.clock
        (CGB.RenewableLearning.world.sense z.2) z.1 e z'.1
```

## CGB.RenewableLearningCarrier.realized

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L72)

解释：可再生学习机体微观物理转移完全实现了宏观可再生记账规则的全部公理。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool),
  CGB.MaterialSource.Realization (CGB.RenewableLearningCarrier.physical cut)
    (CGB.RenewableLearning.process true true cut) (CGB.RenewableLearningCarrier.owner cut)
    (CGB.RenewableLearningCarrier.project cut) (fun z => (CGB.RenewableLearning.plant true true).total z.1)
    (CGB.RenewableLearningCarrier.declared (CGB.RenewableLearning.incoming true))
    (CGB.RenewableLearningCarrier.declared (CGB.RenewableLearning.outgoing true true)) CGB.RenewableLearning.world.clock
```

## _private.CGBAxioms.RenewableLearningCarrier.0.CGB.RenewableLearningCarrier.intake_matches

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L87)

解释：微观物质载体向机体的迁移量与宏观接口声称的输入通量完全相等。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (t : ℕ) (r r' : CGB.RenewableLearningCarrier.Carrier cut) (e : Option Unit),
  0 < (CGB.RenewableLearningCarrier.physical cut).step t r e r' →
    CGB.RenewableLearningCarrier.intake cut t r e r' =
      CGB.Exchange.intake (CGB.RenewableLearning.plant true true) CGB.RenewableLearning.world
        (CGB.RenewableLearning.ledger true true) (fun x => CGB.RenewableLearning.world.sense) t
        (CGB.RenewableLearningCarrier.project cut r) e (CGB.RenewableLearningCarrier.project cut r')
```

## CGB.RenewableLearningCarrier.recovery_preserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L97)

解释：微观载体物理过程的首次恢复事件概率与宏观可再生系统的恢复度量精确相等。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (t k : ℕ) (r : CGB.RenewableLearningCarrier.Carrier cut),
  CGB.EventChannels.Kernel.firstHit (CGB.Energetics.events (CGB.RenewableLearningCarrier.physical cut))
      (CGB.RenewableLearningCarrier.intake cut) t k r =
    CGB.Exchange.recovery (CGB.RenewableLearning.plant true true) CGB.RenewableLearning.world ⋯
      (CGB.RenewableLearning.ledger true true) cut (fun x => CGB.RenewableLearning.world.sense) t k
      (CGB.RenewableLearningCarrier.project cut r)
```

## CGB.RenewableLearningCarrier.recovery_by_certificate

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L104)

解释：由局域恢复证书保证：只要机体位于安全域内，在有限时间视界内的微观物质恢复度严格为1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t : ℕ) (r : CGB.RenewableLearningCarrier.Carrier false),
  CGB.RenewableRecoveryCertificate.certificate.region (CGB.RenewableLearningCarrier.project false r) →
    CGB.EventChannels.Kernel.firstHit (CGB.Energetics.events (CGB.RenewableLearningCarrier.physical false))
        (CGB.RenewableLearningCarrier.intake false) t CGB.RenewableLearning.world.horizon r =
      1
```

## CGB.RenewableLearningCarrier.score_preserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L119)

解释：微观载体物理系统的全时存活得分与宏观可再生学习植物的生存概率精确一致。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (n : ℕ),
  CGB.RenewableLearningCarrier.score cut n =
    CGB.Exchange.survival (CGB.RenewableLearning.plant true true) CGB.RenewableLearning.world ⋯
      (CGB.RenewableLearning.ledger true true) cut (fun x => CGB.RenewableLearning.world.sense) n
```

## CGB.RenewableLearningCarrier.maintenance_preserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L144)

解释：正常闭环系统的全时生存概率恒为1，且开环截断在6步后存活率至少下降1/4（维持必要性）。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
(∀ (n : ℕ), CGB.RenewableLearningCarrier.score false n = 1) ∧
  ∀ (n : ℕ), 6 ≤ n → CGB.RenewableLearningCarrier.score true n + 1 / 4 ≤ CGB.RenewableLearningCarrier.score false n
```

## CGB.RenewableLearningCarrier.stock_and_recharge

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L186)

解释：在微观转移中源头恒定维持4单位物质存量，且外源补充通量与机体声明的摄入通量严格相等。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (t : ℕ) (r r' : CGB.RenewableLearningCarrier.Carrier cut) (e : Option Unit),
  0 < (CGB.RenewableLearningCarrier.physical cut).step t r e r' →
    CGB.MaterialSource.count (CGB.RenewableLearningCarrier.owner cut r) CGB.MaterialSource.Place.source = 4 ∧
      (CGB.RenewableLearningCarrier.sourceAccount cut).drive t r e r' =
        ↑(CGB.RenewableLearningCarrier.declared (CGB.RenewableLearning.incoming true) t
            (CGB.RenewableLearningCarrier.project cut r) e (CGB.RenewableLearningCarrier.project cut r'))
```

## CGB.RenewableLearningCarrier.initial_energy

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L197)

解释：在系统初始创生时，外源物质库中包含6单位独立于机体的初始有效能量储备。

源码说明：The extra source stock is external to the old two-unit body at birth.

形式陈述：

```lean
∀ (cut : Bool),
  (CGB.RenewableLearningCarrier.sourceAccount cut).stored 0 (CGB.RenewableLearningCarrier.initialCarrier cut) = 6
```

## CGB.RenewableLearningCarrier.source_energy_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L207)

解释：沿可再生物理演化全过程，外源储能增量与排入汇的物质流之和由持续驱动通量严格平衡。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (n : ℕ),
  CGB.Energetics.mean
        (CGB.Energetics.law (CGB.RenewableLearningCarrier.physical cut)
          (CGB.RenewableLearningCarrier.initialCarrier cut) n)
        ((CGB.RenewableLearningCarrier.sourceAccount cut).stored n) +
      ∑ t ∈ Finset.range n,
        CGB.Energetics.flux (CGB.RenewableLearningCarrier.physical cut)
          (CGB.RenewableLearningCarrier.initialCarrier cut) (CGB.RenewableLearningCarrier.release cut) t =
    6 +
      ∑ t ∈ Finset.range n,
        CGB.Energetics.flux (CGB.RenewableLearningCarrier.physical cut)
          (CGB.RenewableLearningCarrier.initialCarrier cut) (CGB.RenewableLearningCarrier.sourceAccount cut).drive t
```
