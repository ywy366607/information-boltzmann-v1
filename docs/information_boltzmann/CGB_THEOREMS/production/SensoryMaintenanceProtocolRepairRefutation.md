# SensoryMaintenanceProtocolRepairRefutation

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceProtocolRepairRefutation.lean) · [返回导航](../README.md)

## _private.CGBAxioms.SensoryMaintenanceProtocolRepairRefutation.0.CGB.SensoryMaintenanceProtocolRepairRefutation.policy_table

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceProtocolRepairRefutation.lean#L7)

解释：感官维护协议修复反驳模型中细胞状态到离散策略编号的显式对照查表。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell),
  CGB.RenewableLearning.policy q =
    match q with
    | CGB.RenewableLearning.Cell.aSpent => 0
    | CGB.RenewableLearning.Cell.aReady => 0
    | CGB.RenewableLearning.Cell.aHeld => 0
    | CGB.RenewableLearning.Cell.bSpent => 1
    | CGB.RenewableLearning.Cell.bReady => 1
    | CGB.RenewableLearning.Cell.bHeld => 1
    | x => 2
```

## _private.CGBAxioms.SensoryMaintenanceProtocolRepairRefutation.0.CGB.SensoryMaintenanceProtocolRepairRefutation.path_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceProtocolRepairRefutation.lean#L29)

解释：五步有限路径上的每一步转移概率在对抗环境下均严格为正。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ t < 5,
  0 <
    (CGB.SensoryMaintenanceProtocolRepairRefutation.K✝ t).step (CGB.SensoryMaintenanceProtocolRepairRefutation.path✝ t)
      (CGB.SensoryMaintenanceProtocolRepairRefutation.event✝ t)
      (CGB.SensoryMaintenanceProtocolRepairRefutation.path✝ (t + 1))
```

## CGB.SensoryMaintenanceProtocolRepairRefutation.protected_protocol_empty_mass_positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceProtocolRepairRefutation.lean#L44)

解释：受保护协议在有限时间内出现空质量态的概率严格为正，无法实现绝对保护。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
0 <
  CGB.SensoryMaintenance.body.law CGB.SensoryMaintenance.world
    (CGB.SensoryMaintenanceRepairCandidates.protectedTwoTickProtocol 2) 5 (fun x => CGB.SensoryMaintenance.empty)
    CGB.RenewableLearning.Env.probeB
```

## CGB.SensoryMaintenanceProtocolRepairRefutation.protected_protocol_survival_lt_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceProtocolRepairRefutation.lean#L63)

解释：反驳定理：声称受保护的感官维护协议在真实环境扰动下的长期存活率严格小于一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Exchange.survival CGB.SensoryMaintenance.body CGB.SensoryMaintenance.world CGB.SensoryMaintenance.ec
    CGB.SensoryMaintenance.material false (CGB.SensoryMaintenanceRepairCandidates.protectedTwoTickProtocol 2) 5 <
  1
```
