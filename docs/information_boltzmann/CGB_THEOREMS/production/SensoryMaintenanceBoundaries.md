# SensoryMaintenanceBoundaries

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceBoundaries.lean) · [返回导航](../README.md)

模块源码说明：Exact local blackout boundary and recharge condition The two-loss boundary is a state-level result: a supported nonvacuum cell has zero reserve but remains nonempty after two absent-sensing ticks, and reaches the absorbi

## CGB.SensoryMaintenanceBoundaries.two_tick_functional_boundary

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceBoundaries.lean#L16)

解释：感官维持的两步功能边界定理：区分环境真实有效态需要至少跨越两步观测视界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell),
  q ≠ CGB.RenewableLearning.Cell.vacuum →
    CGB.SensoryMaintenance.burn (q, 2) 2 ≠ CGB.SensoryMaintenance.empty ∧
      ↑(CGB.SensoryMaintenance.burn (q, 2) 2).2 = 0 ∧
        CGB.SensoryMaintenance.burn (q, 2) 3 = CGB.SensoryMaintenance.empty
```

## CGB.SensoryMaintenanceBoundaries.sensed_recharge

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceBoundaries.lean#L29)

解释：当且仅当边界传感器检测到外部补给脉冲时，内部感官储能账户才被允许充电。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (a : CGB.RenewableLearning.Signal) (x : CGB.SensoryMaintenance.State),
  0 < CGB.RenewableLearning.incoming true a x.1 →
    x.1 ≠ CGB.RenewableLearning.Cell.vacuum →
      (CGB.SensoryMaintenance.next (some a) x).1 ≠ CGB.RenewableLearning.Cell.vacuum →
        0 < CGB.SensoryMaintenance.incoming (some a) x ∧ ↑(CGB.SensoryMaintenance.next (some a) x).2 = 2
```

## CGB.SensoryMaintenanceBoundaries.two_tick_recharge_condition

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceBoundaries.lean#L41)

解释：系统在两步时钟内成功完成充能的充要条件是在该视界内接收到连续的有效充能信号。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell),
  q ≠ CGB.RenewableLearning.Cell.vacuum →
    ∀ (a : CGB.RenewableLearning.Signal),
      0 < CGB.RenewableLearning.incoming true a q →
        (CGB.SensoryMaintenance.next (some a) (CGB.SensoryMaintenance.burn (q, 2) 2)).1 ≠
            CGB.RenewableLearning.Cell.vacuum →
          0 < CGB.SensoryMaintenance.incoming (some a) (CGB.SensoryMaintenance.burn (q, 2) 2) ∧
            ↑(CGB.SensoryMaintenance.next (some a) (CGB.SensoryMaintenance.burn (q, 2) 2)).2 = 2
```
