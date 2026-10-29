# ActiveMaintenanceEnergy

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceEnergy.lean) · [返回导航](../README.md)

模块源码说明：Reserve, explicit consumption and source-resolved replacement The same registered normal/cut event process pays one reserve unit on a live feeding or scarcity event.

## CGB.ActiveMaintenanceEnergy.actual_law

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceEnergy.lean#L28)

解释：主动维护能量过程在任意步长下的概率分布律，与机体在环境感官驱动下的实际事件转移律（包含通道切断条件）完全一致。

源码说明：The energy law is exactly the state law used in the viability comparison, including the cut's own distribution-dependent future, not a second process.

形式陈述：

```lean
∀ (cut : Bool) (n : ℕ) (z : CGB.ActiveMaintenanceEnergy.S),
  CGB.Energetics.law (CGB.ActiveMaintenanceEnergy.process cut) CGB.ActiveMaintenanceEnergy.initial n z =
    if cut = true then
      CGB.ActiveMaintenance.body.lawCut CGB.ActiveMaintenance.world (fun x => CGB.ActiveMaintenance.world.sense) n z.1
        z.2
    else
      CGB.ActiveMaintenance.body.law CGB.ActiveMaintenance.world (fun x => CGB.ActiveMaintenance.world.sense) n z.1 z.2
```

## _private.CGBAxioms.ActiveMaintenanceEnergy.0.CGB.ActiveMaintenanceEnergy.body_supported

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceEnergy.lean#L45)

解释：主动维护能量过程的正概率转移必然具有底层物理机体事件步的正概率支持。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (t : ℕ) (z z' : CGB.ActiveMaintenanceEnergy.S) (e : Option Unit),
  0 < (CGB.ActiveMaintenanceEnergy.process cut).step t z e z' →
    0 <
      CGB.ActiveMaintenance.body.eventStep CGB.ActiveMaintenance.world.clock (CGB.ActiveMaintenance.world.sense z.2) z.1
        e z'.1
```

## CGB.ActiveMaintenanceEnergy.isothermal

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceEnergy.lean#L86)

解释：主动维护能量过程在零外加驱动功与耗散条件下满足等温账户能量平衡。

源码说明：Conditional mean second-law admission for the explicitly declared internal entropy and bath units. Energy expenditure is fixed by outgoing material use.

形式陈述：

```lean
∀ (cut : Bool),
  CGB.Energetics.Isothermal (CGB.ActiveMaintenanceEnergy.process cut) (CGB.ActiveMaintenanceEnergy.account cut)
    CGB.ActiveMaintenanceEnergy.initial 1 (fun x x_1 => 0) fun x x_1 x_2 x_3 => 0
```

## CGB.ActiveMaintenanceEnergy.availability_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceEnergy.lean#L108)

解释：主动维护过程的平均储能增量与流出通量之和，严格守恒等于初始基准能量加累计流入通量。

源码说明：Every finite prefix uses the SAME event law and tick. At T=1, zero declared internal/carried entropy, imported energy pays actual consumption and reserve. The reservoir that recharges recycled tokens is registered separately below.

形式陈述：

```lean
∀ (cut : Bool) (n : ℕ),
  CGB.Energetics.mean
        (CGB.Energetics.law (CGB.ActiveMaintenanceEnergy.process cut) CGB.ActiveMaintenanceEnergy.initial n)
        ((CGB.ActiveMaintenanceEnergy.account cut).stored n) +
      ∑ t ∈ Finset.range n,
        CGB.Energetics.flux (CGB.ActiveMaintenanceEnergy.process cut) CGB.ActiveMaintenanceEnergy.initial
          (CGB.ActiveMaintenanceEnergy.port CGB.ActiveMaintenance.outgoing) t =
    1 +
      ∑ t ∈ Finset.range n,
        CGB.Energetics.flux (CGB.ActiveMaintenanceEnergy.process cut) CGB.ActiveMaintenanceEnergy.initial
          (CGB.ActiveMaintenanceEnergy.port CGB.ActiveMaintenance.incoming) t
```
