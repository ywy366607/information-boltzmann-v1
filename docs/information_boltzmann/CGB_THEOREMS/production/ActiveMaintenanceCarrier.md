# ActiveMaintenanceCarrier

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean) · [返回导航](../README.md)

模块源码说明：Three-token realization of the maintenance process One token is in the body, one in the accessible source, one in the return sink.

## _private.CGBAxioms.ActiveMaintenanceCarrier.0.CGB.ActiveMaintenanceCarrier.event_delivery

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L53)

解释：主动维护载体中的每个离散事件交付均对应于确定性的微观状态跳转。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : Unit → CGB.ActiveMaintenance.Stim) (x y : Unit → CGB.ActiveMaintenance.State) (e : Option Unit),
  0 < CGB.ActiveMaintenance.body.eventStep 1 s x e y →
    y () = CGB.ActiveMaintenance.next (s ()) (x ()) ∧ CGB.ActiveMaintenanceCarrier.Enabled✝ (s ()) (x ()) e
```

## _private.CGBAxioms.ActiveMaintenanceCarrier.0.CGB.ActiveMaintenanceCarrier.physical_delivery

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L74)

解释：物理层物质包的交付与宏观感官维护时钟在离散步长下严格步调一致。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (t : ℕ) (r r' : CGB.ActiveMaintenanceCarrier.R) (e : Option Unit),
  0 < (CGB.ActiveMaintenanceCarrier.physical cut).step t r e r' →
    r'.1.1 () = CGB.ActiveMaintenance.next (CGB.ActiveMaintenance.world.sense r.1.2 ()) (r.1.1 ()) ∧
      r'.2 = CGB.ActiveMaintenanceCarrier.phaseNext (CGB.ActiveMaintenance.world.sense r.1.2 ()) (r.1.1 ()) e r.2 ∧
        CGB.ActiveMaintenanceCarrier.Enabled✝ (CGB.ActiveMaintenance.world.sense r.1.2 ()) (r.1.1 ()) e
```

## _private.CGBAxioms.ActiveMaintenanceCarrier.0.CGB.ActiveMaintenanceCarrier.token_count

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L91)

解释：维护载体中流通的令牌总数在内部传输过程中保持代数守恒。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (a : CGB.ActiveMaintenance.State) (p : CGB.ActiveMaintenanceCarrier.Phase),
  CGB.MaterialSource.count (CGB.ActiveMaintenanceCarrier.owners a p) CGB.MaterialSource.Place.body =
    CGB.ActiveMaintenance.cost a
```

## _private.CGBAxioms.ActiveMaintenanceCarrier.0.CGB.ActiveMaintenanceCarrier.token_transport

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L94)

解释：令牌在不同位点间的空间传输受到通道开闭状态和局域排他律的完全约束。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : CGB.ActiveMaintenance.Stim) (a : CGB.ActiveMaintenance.State) (p : CGB.ActiveMaintenanceCarrier.Phase)
  (e : Option Unit),
  CGB.ActiveMaintenanceCarrier.Enabled✝ s a e →
    CGB.MaterialSource.Allowed (CGB.ActiveMaintenanceCarrier.owners a p)
        (CGB.ActiveMaintenanceCarrier.owners (CGB.ActiveMaintenance.next s a)
          (CGB.ActiveMaintenanceCarrier.phaseNext s a e p)) ∧
      (CGB.MaterialSource.moved (CGB.ActiveMaintenanceCarrier.owners a p)
            (CGB.ActiveMaintenanceCarrier.owners (CGB.ActiveMaintenance.next s a)
              (CGB.ActiveMaintenanceCarrier.phaseNext s a e p))
            CGB.MaterialSource.Place.source CGB.MaterialSource.Place.body =
          e.elim 0 fun x => CGB.ActiveMaintenance.incoming s a) ∧
        CGB.MaterialSource.moved (CGB.ActiveMaintenanceCarrier.owners a p)
            (CGB.ActiveMaintenanceCarrier.owners (CGB.ActiveMaintenance.next s a)
              (CGB.ActiveMaintenanceCarrier.phaseNext s a e p))
            CGB.MaterialSource.Place.body CGB.MaterialSource.Place.sink =
          e.elim 0 fun x => CGB.ActiveMaintenance.outgoing s a
```

## CGB.ActiveMaintenanceCarrier.realized

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L106)

解释：主动维护载体在环境驱动下实现了物理底层介质到粗粒化事件过程的状态映射与进出物料通量守恒。

源码说明：Full process and clock certificate, for both normal and marginal-ghost cut. Carrier identities cannot alter any future observed event/state cylinder.

形式陈述：

```lean
∀ (cut : Bool),
  CGB.MaterialSource.Realization (CGB.ActiveMaintenanceCarrier.physical cut) (CGB.ActiveMaintenanceEnergy.process cut)
    CGB.ActiveMaintenanceCarrier.owner CGB.ActiveMaintenanceCarrier.project
    (fun z => CGB.ActiveMaintenance.body.total z.1)
    (CGB.ActiveMaintenanceCarrier.declared CGB.ActiveMaintenance.incoming)
    (CGB.ActiveMaintenanceCarrier.declared CGB.ActiveMaintenance.outgoing) CGB.ActiveMaintenance.world.clock
```

## _private.CGBAxioms.ActiveMaintenanceCarrier.0.CGB.ActiveMaintenanceCarrier.token_energy

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L121)

解释：每个流动物质令牌所携带的有效势能与当前微观位点静电势严格匹配。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : CGB.ActiveMaintenance.Stim) (a : CGB.ActiveMaintenance.State) (p : CGB.ActiveMaintenanceCarrier.Phase)
  (e : Option Unit),
  CGB.ActiveMaintenanceCarrier.Enabled✝ s a e →
    (CGB.MaterialSource.count
            (CGB.ActiveMaintenanceCarrier.owners (CGB.ActiveMaintenance.next s a)
              (CGB.ActiveMaintenanceCarrier.phaseNext s a e p))
            CGB.MaterialSource.Place.body +
          CGB.MaterialSource.count
            (CGB.ActiveMaintenanceCarrier.owners (CGB.ActiveMaintenance.next s a)
              (CGB.ActiveMaintenanceCarrier.phaseNext s a e p))
            CGB.MaterialSource.Place.source +
        e.elim 0 fun x => CGB.ActiveMaintenance.outgoing s a) =
      CGB.MaterialSource.count (CGB.ActiveMaintenanceCarrier.owners a p) CGB.MaterialSource.Place.body +
          CGB.MaterialSource.count (CGB.ActiveMaintenanceCarrier.owners a p) CGB.MaterialSource.Place.source +
        CGB.MaterialSource.moved (CGB.ActiveMaintenanceCarrier.owners a p)
          (CGB.ActiveMaintenanceCarrier.owners (CGB.ActiveMaintenance.next s a)
            (CGB.ActiveMaintenanceCarrier.phaseNext s a e p))
          CGB.MaterialSource.Place.sink CGB.MaterialSource.Place.source
```
