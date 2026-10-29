# SensoryMaintenanceRepairAudit

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairAudit.lean) · [返回导航](../README.md)

模块源码说明：Audit of state-only repairs after the two-tick refutation The existing material ports make the `aSpent`/`miss` failure structurally forced.

## _private.CGBAxioms.SensoryMaintenanceRepairAudit.0.CGB.SensoryMaintenanceRepairAudit.cost_zero_iff_empty

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairAudit.lean#L16)

解释：感官维护内部成本为零当且仅当系统处于没有任何历史残留的纯粹空初始态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (x : CGB.SensoryMaintenance.State), CGB.SensoryMaintenance.cost x = 0 ↔ x = CGB.SensoryMaintenance.empty
```

## CGB.SensoryMaintenanceRepairAudit.spent_miss_budget_forces_empty

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairAudit.lean#L23)

解释：当容许的错过错误预算耗尽时，感官维护状态机被不可逆地强制置为空状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (y : CGB.SensoryMaintenance.State),
  CGB.SensoryMaintenance.cost y +
        CGB.SensoryMaintenance.outgoing (some CGB.RenewableLearning.Signal.miss)
          (CGB.RenewableLearning.Cell.aSpent, 0) =
      CGB.SensoryMaintenance.cost (CGB.RenewableLearning.Cell.aSpent, 0) +
        CGB.SensoryMaintenance.incoming (some CGB.RenewableLearning.Signal.miss)
          (CGB.RenewableLearning.Cell.aSpent, 0) →
    y = CGB.SensoryMaintenance.empty
```

## CGB.SensoryMaintenanceRepairAudit.no_state_only_two_tick_repair

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairAudit.lean#L42)

解释：审计反例证明：仅依靠状态记忆而不利用两步环境关联，无法完成受损感官协议的修复。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (repair : CGB.SensoryMaintenance.Stim → CGB.SensoryMaintenance.State → CGB.SensoryMaintenance.State),
  (∀ (s : CGB.SensoryMaintenance.Stim) (x : CGB.SensoryMaintenance.State),
      CGB.SensoryMaintenance.cost (repair s x) + CGB.SensoryMaintenance.outgoing s x =
        CGB.SensoryMaintenance.cost x + CGB.SensoryMaintenance.incoming s x) →
    repair (some CGB.RenewableLearning.Signal.miss) (CGB.RenewableLearning.Cell.aSpent, 0) ≠
        CGB.SensoryMaintenance.empty →
      False
```
