# PortSeparation

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean) · [返回导航](../README.md)

模块源码说明：Independent sensory and resource ports on the same finite body Absent sensing with resources supplied is a noninformative maintenance inlet.

## CGB.PortSeparation.local_budgets

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L54)

解释：两分离端口各自持有局部能量与物质预算，彼此独立核算且总和守恒。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (u : CGB.PortSeparation.Input) (x : CGB.PortSeparation.State),
  CGB.PortSeparation.cost (CGB.PortSeparation.next u x) + CGB.PortSeparation.outgoing u x =
      CGB.PortSeparation.cost x + CGB.PortSeparation.incoming u x ∧
    CGB.RenewableLearning.credit (CGB.PortSeparation.next u x).1 + CGB.PortSeparation.writes u x +
          CGB.PortSeparation.discarded u x =
        CGB.RenewableLearning.credit x.1 + CGB.PortSeparation.topups u x ∧
      CGB.PortSeparation.writes u x ≤ CGB.PortSeparation.outgoing u x
```

## CGB.PortSeparation.starvation_progress

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L65)

解释：饥饿推进定理：在持续匮乏条件下，系统状态单调流向确定的休眠或重启吸引子。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : CGB.SensoryMaintenance.Stim) (x : CGB.PortSeparation.State),
  CGB.PortSeparation.rank x ≤ 3 ∧
    (CGB.PortSeparation.rank x = 0 ↔ x = CGB.PortSeparation.empty) ∧
      (x ≠ CGB.PortSeparation.empty →
          CGB.PortSeparation.rank (CGB.PortSeparation.next (s, false) x) < CGB.PortSeparation.rank x) ∧
        (x ≠ CGB.PortSeparation.empty → 0 < CGB.PortSeparation.outgoing (s, false) x)
```

## CGB.PortSeparation.next_empty

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L74)

解释：当两端资源均告空时，分离端口下一步系统的联合状态保持为空。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (u : CGB.PortSeparation.Input), CGB.PortSeparation.next u CGB.PortSeparation.empty = CGB.PortSeparation.empty
```

## CGB.PortSeparation.admission

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L92)

解释：端口分离结构满足广义系统准入准则，两端无意图隐蔽信道。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.PortSeparation.body.Axioms ∧ CGB.PortSeparation.body.EventClocked 1
```

## CGB.PortSeparation.legacy_rules

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L129)

解释：端口分离体系在退化情形下与传统无隔离交互系统的动力学规则保持向后兼容。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : CGB.SensoryMaintenance.Stim) (x : CGB.PortSeparation.State),
  CGB.PortSeparation.next (CGB.PortSeparation.legacy s) x = CGB.SensoryMaintenance.next s x ∧
    CGB.PortSeparation.incoming (CGB.PortSeparation.legacy s) x = CGB.SensoryMaintenance.incoming s x ∧
      CGB.PortSeparation.outgoing (CGB.PortSeparation.legacy s) x = CGB.SensoryMaintenance.outgoing s x
```

## CGB.PortSeparation.actual_transition

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L148)

解释：端口分离机制下的复合状态实际转移与两独立子系统局部转移的乘积精确重合。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.PortSeparation.Input) (t : ℕ)
  (z z' : CGB.PortSeparation.S) (e : Option Unit),
  0 < (CGB.PortSeparation.process cut sched).step t z e z' → z'.1 () = CGB.PortSeparation.next (sched t z.2 ()) (z.1 ())
```
