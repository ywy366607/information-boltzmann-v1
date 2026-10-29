# EnergyControls

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/EnergyControls.lean) · [返回导航](../README.md)

模块源码说明：Energy-interpretation controls on the unchanged material process The same material transport admits different energy calibrations.

## CGB.EnergyControls.total_partition

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EnergyControls.lean#L24)

解释：在该有限物料见证中，内部、外部、功接收与热接收四部分之和始终等于 2a。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (a : ℝ) (p : CGB.MaterialSourceControls.Phase),
  CGB.EnergyControls.internal a p + CGB.EnergyControls.exterior a p + CGB.EnergyControls.workReceiver a p +
      CGB.EnergyControls.heatReceiver a p =
    2 * a
```

## CGB.EnergyControls.material_energy_binding

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EnergyControls.lean#L43)

解释：该物料见证的每个受支持事件中，物料能量端口由同一外部物料转移量确定。

源码说明：On every supported event, the material-energy port is carried by the same exterior token identified in the material-source certificate.

形式陈述：

```lean
∀ (a : ℝ) (t : ℕ) (p p' : CGB.MaterialSourceControls.Phase) (e : Option Unit),
  0 < CGB.MaterialSourceControls.physical.step t p e p' →
    (CGB.EnergyControls.account a).matter t p e p' =
      2 * a *
        ↑(CGB.MaterialSource.moved (CGB.MaterialSourceControls.owner p) (CGB.MaterialSourceControls.owner p')
            CGB.MaterialSource.Place.outside CGB.MaterialSource.Place.source)
```

## CGB.EnergyControls.isothermal_account

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EnergyControls.lean#L54)

解释：当 a 非负时，该见证的指定 T=1、零系统熵账本满足所定义的等温条件。

源码说明：Nonnegative mean production for the declared T=1, zero-system-entropy accounting interpretation along the actual process; not a derivation of the microscopic entropy function or of local detailed balance.

形式陈述：

```lean
∀ (a : ℝ),
  0 ≤ a →
    CGB.Energetics.Isothermal CGB.MaterialSourceControls.physical (CGB.EnergyControls.account a) 0 1 (fun x x_1 => 0)
      fun x x_1 x_2 x_3 => 0
```

## CGB.EnergyControls.same_material_different_energy

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EnergyControls.lean#L75)

解释：该见证中相同的物料转移可对应不同的能量账本，说明单凭物料计数不能唯一确定功或化学能。

源码说明：Fixed identities and identical counts do not select chemical energy or work. The positive interpretation has a work receiver AND a heat receiver.

形式陈述：

```lean
∀ (t : ℕ),
  CGB.MaterialSource.moved (CGB.MaterialSourceControls.owner 1) (CGB.MaterialSourceControls.owner 2)
        CGB.MaterialSource.Place.source CGB.MaterialSource.Place.body =
      1 ∧
    (CGB.EnergyControls.account 0).load t 1 (some ()) 2 = 0 ∧
      (CGB.EnergyControls.account 1).load t 1 (some ()) 2 = 1 ∧
        (CGB.EnergyControls.account 1).heat t 1 (some ()) 2 = -1 ∧ (CGB.EnergyControls.account 1).matter t 0 none 1 = 2
```
