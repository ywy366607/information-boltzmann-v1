# SensoryMaintenanceCarrier

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean) · [返回导航](../README.md)

模块源码说明：Two existing finite material buffers realize structural fuel and the new reserve.

## _private.CGBAxioms.SensoryMaintenanceCarrier.0.CGB.SensoryMaintenanceCarrier.component_rules

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L28)

解释：双隔室感知维持系统的各个局部组件在转移前后严格满足各自独立的存量与通量守恒规则。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (b : Bool) (s : CGB.SensoryMaintenance.Stim) (x : CGB.SensoryMaintenance.State),
  CGB.SensoryMaintenanceCarrier.partContent b x ≤ 4 ∧
    CGB.SensoryMaintenanceCarrier.partIn b s x ≤ 4 ∧
      CGB.SensoryMaintenanceCarrier.partOut b s x ≤ CGB.SensoryMaintenanceCarrier.partContent b x ∧
        CGB.SensoryMaintenanceCarrier.partContent b (CGB.SensoryMaintenance.next s x) +
            CGB.SensoryMaintenanceCarrier.partOut b s x =
          CGB.SensoryMaintenanceCarrier.partContent b x + CGB.SensoryMaintenanceCarrier.partIn b s x
```

## CGB.SensoryMaintenanceCarrier.component_sums

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L36)

解释：两个子组件的微观物质存量、输入通量与输出通量求和精确还原全局感知维持系统的宏观对应量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : CGB.SensoryMaintenance.Stim) (x : CGB.SensoryMaintenance.State),
  CGB.SensoryMaintenanceCarrier.partContent false x + CGB.SensoryMaintenanceCarrier.partContent true x =
      CGB.SensoryMaintenance.cost x ∧
    CGB.SensoryMaintenanceCarrier.partIn false s x + CGB.SensoryMaintenanceCarrier.partIn true s x =
        CGB.SensoryMaintenance.incoming s x ∧
      CGB.SensoryMaintenanceCarrier.partOut false s x + CGB.SensoryMaintenanceCarrier.partOut true s x =
        CGB.SensoryMaintenance.outgoing s x
```

## _private.CGBAxioms.SensoryMaintenanceCarrier.0.CGB.SensoryMaintenanceCarrier.coarse_positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L78)

解释：微观缓冲载体在具有正演化概率时，其粗粒化投影后的宏观交互机跃迁概率同样严格为正。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {X : Type u_1} {E : Type u_2} [inst : Fintype X] [inst_1 : DecidableEq X] [inst_2 : Fintype E]
  {P : CGB.Interaction.Machine X ℕ E} (B : CGB.BufferedMaterial.Rules P) (t : ℕ) (r r' : CGB.BufferedMaterial.State B)
  (e : E),
  0 < (CGB.BufferedMaterial.physical B).step t r e r' →
    0 < P.step t (CGB.BufferedMaterial.project B r) e (CGB.BufferedMaterial.project B r')
```

## _private.CGBAxioms.SensoryMaintenanceCarrier.0.CGB.SensoryMaintenanceCarrier.count_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L117)

解释：复合感知维护载体在各个空间位置的总物质计数等于其分解子载体在该位置计数之和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.SensoryMaintenance.Stim)
  (r : CGB.SensoryMaintenanceCarrier.Carrier cut sched) (p : CGB.MaterialSource.Place),
  CGB.MaterialSource.count (CGB.SensoryMaintenanceCarrier.owner cut sched r) p =
    CGB.MaterialSource.count
        (CGB.BufferedMaterial.owner (CGB.SensoryMaintenanceCarrier.firstRules cut sched)
          (CGB.BufferedMaterial.project (CGB.SensoryMaintenanceCarrier.secondRules cut sched) r))
        p +
      CGB.MaterialSource.count (CGB.BufferedMaterial.owner (CGB.SensoryMaintenanceCarrier.secondRules cut sched) r) p
```

## _private.CGBAxioms.SensoryMaintenanceCarrier.0.CGB.SensoryMaintenanceCarrier.moved_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L125)

解释：微观空间各位置间迁移的总物质数量等于第一与第二子系统微观迁移量的代数和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.SensoryMaintenance.Stim)
  (r r' : CGB.SensoryMaintenanceCarrier.Carrier cut sched) (p q : CGB.MaterialSource.Place),
  CGB.MaterialSource.moved (CGB.SensoryMaintenanceCarrier.owner cut sched r)
      (CGB.SensoryMaintenanceCarrier.owner cut sched r') p q =
    CGB.MaterialSource.moved
        (CGB.BufferedMaterial.owner (CGB.SensoryMaintenanceCarrier.firstRules cut sched)
          (CGB.BufferedMaterial.project (CGB.SensoryMaintenanceCarrier.secondRules cut sched) r))
        (CGB.BufferedMaterial.owner (CGB.SensoryMaintenanceCarrier.firstRules cut sched)
          (CGB.BufferedMaterial.project (CGB.SensoryMaintenanceCarrier.secondRules cut sched) r'))
        p q +
      CGB.MaterialSource.moved (CGB.BufferedMaterial.owner (CGB.SensoryMaintenanceCarrier.secondRules cut sched) r)
        (CGB.BufferedMaterial.owner (CGB.SensoryMaintenanceCarrier.secondRules cut sched) r') p q
```

## CGB.SensoryMaintenanceCarrier.realized

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L136)

解释：双隔室感知维持系统的微观载体模型完全实现了宏观感知维持动力学规则。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.SensoryMaintenance.Stim),
  CGB.MaterialSource.Realization (CGB.SensoryMaintenanceCarrier.physical cut sched)
    (CGB.SensoryMaintenance.process cut sched) (CGB.SensoryMaintenanceCarrier.owner cut sched)
    (CGB.SensoryMaintenanceCarrier.project cut sched) (fun z => CGB.SensoryMaintenance.body.total z.1)
    (CGB.SensoryMaintenanceCarrier.declared sched CGB.SensoryMaintenance.incoming)
    (CGB.SensoryMaintenanceCarrier.declared sched CGB.SensoryMaintenance.outgoing) 1
```

## CGB.SensoryMaintenanceCarrier.source_stock

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L167)

解释：感知维持系统中外源物质库的微观储备量在演化中恒定维持在4单位。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.SensoryMaintenance.Stim)
  (r : CGB.SensoryMaintenanceCarrier.Carrier cut sched),
  CGB.MaterialSource.count (CGB.SensoryMaintenanceCarrier.owner cut sched r) CGB.MaterialSource.Place.source = 8
```

## CGB.SensoryMaintenanceCarrier.recharge_paid

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L176)

解释：外源向机体输送的物质通量与从汇回流充电的物质流在微观上严格平衡。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.SensoryMaintenance.Stim) (t : ℕ)
  (r r' : CGB.SensoryMaintenanceCarrier.Carrier cut sched) (e : Option Unit),
  0 < (CGB.SensoryMaintenanceCarrier.physical cut sched).step t r e r' →
    CGB.MaterialSource.moved (CGB.SensoryMaintenanceCarrier.owner cut sched r)
        (CGB.SensoryMaintenanceCarrier.owner cut sched r') CGB.MaterialSource.Place.sink
        CGB.MaterialSource.Place.source =
      CGB.MaterialSource.moved (CGB.SensoryMaintenanceCarrier.owner cut sched r)
        (CGB.SensoryMaintenanceCarrier.owner cut sched r') CGB.MaterialSource.Place.source CGB.MaterialSource.Place.body
```

## CGB.SensoryMaintenanceCarrier.recovery_preserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L195)

解释：物理微观感知载体过程的首次恢复命中概率与宏观感知维持系统的恢复度量严格一致。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.SensoryMaintenance.Stim) (t k : ℕ)
  (r : CGB.SensoryMaintenanceCarrier.Carrier cut sched),
  CGB.EventChannels.Kernel.firstHit (CGB.Energetics.events (CGB.SensoryMaintenanceCarrier.physical cut sched))
      (CGB.SensoryMaintenanceCarrier.intake cut sched) t k r =
    CGB.Exchange.recovery CGB.SensoryMaintenance.body CGB.SensoryMaintenance.world CGB.SensoryMaintenance.ec
      CGB.SensoryMaintenance.material cut sched t k (CGB.SensoryMaintenanceCarrier.project cut sched r)
```

## CGB.SensoryMaintenanceCarrier.score_preserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L212)

解释：物理感知维持系统的全时存活评分与宏观感知机体的生存概率指标完全相等。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.SensoryMaintenance.Stim) (n : ℕ),
  CGB.SensoryMaintenanceCarrier.score cut sched n =
    CGB.Exchange.survival CGB.SensoryMaintenance.body CGB.SensoryMaintenance.world CGB.SensoryMaintenance.ec
      CGB.SensoryMaintenance.material cut sched n
```

## CGB.SensoryMaintenanceCarrier.initial_energy

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L260)

解释：感知维持物理系统在初态赋予外源物质库的可用能量储备严格等于12单位。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.SensoryMaintenance.Stim),
  (CGB.SensoryMaintenanceCarrier.sourceAccount cut sched).stored 0
      (CGB.SensoryMaintenanceCarrier.initialCarrier cut sched) =
    12
```

## CGB.SensoryMaintenanceCarrier.source_energy_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L269)

解释：沿感知维持过程外源储能变化与物质释放流严格由外界驱动输入能量精确维系。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.SensoryMaintenance.Stim) (n : ℕ),
  CGB.Energetics.mean
        (CGB.Energetics.law (CGB.SensoryMaintenanceCarrier.physical cut sched)
          (CGB.SensoryMaintenanceCarrier.initialCarrier cut sched) n)
        ((CGB.SensoryMaintenanceCarrier.sourceAccount cut sched).stored n) +
      ∑ t ∈ Finset.range n,
        CGB.Energetics.flux (CGB.SensoryMaintenanceCarrier.physical cut sched)
          (CGB.SensoryMaintenanceCarrier.initialCarrier cut sched) (CGB.SensoryMaintenanceCarrier.release cut sched) t =
    12 +
      ∑ t ∈ Finset.range n,
        CGB.Energetics.flux (CGB.SensoryMaintenanceCarrier.physical cut sched)
          (CGB.SensoryMaintenanceCarrier.initialCarrier cut sched)
          (CGB.SensoryMaintenanceCarrier.sourceAccount cut sched).drive t
```
