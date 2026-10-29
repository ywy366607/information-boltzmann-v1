# SensoryMaintenanceTwoTickObstruction

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickObstruction.lean) · [返回导航](../README.md)

## _private.CGBAxioms.SensoryMaintenanceTwoTickObstruction.0.CGB.SensoryMaintenanceTwoTickObstruction.policy_table

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickObstruction.lean#L7)

解释：两步感知阻塞测试模型中各细胞类型到控制输出的查表映射。

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

## CGB.SensoryMaintenanceTwoTickObstruction.two_tick_world_branch

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickObstruction.lean#L14)

解释：在两步分支世界树中，存在环境使得局部感知代理无法区分真假恢复分支。

源码说明：A concrete world branch that remains physically possible through two lost ticks while the projected cell is `aSpent`.

形式陈述：

```lean
CGB.GraceRecovery.moves CGB.RenewableLearning.Env.feedA (fun x => CGB.RenewableLearning.Cell.aSpent)
      CGB.RenewableLearning.Env.feedB =
    1 / 2 ∧
  CGB.GraceRecovery.moves CGB.RenewableLearning.Env.feedB (fun x => CGB.RenewableLearning.Cell.aSpent)
      CGB.RenewableLearning.Env.missB =
    1
```

## CGB.SensoryMaintenanceTwoTickObstruction.two_tick_local_failure

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickObstruction.lean#L22)

解释：两步局部阻塞定理：任何局限于局部视野的两步前瞻策略均存在对抗环境导致维持失败。

源码说明：The same branch drains the two-unit reserve and the next sensed miss sends the cell to vacuum. This is a local obstruction to lifting the one-tick certificate unchanged to a two-tick first-hit certificate.

形式陈述：

```lean
CGB.SensoryMaintenance.next (some CGB.RenewableLearning.Signal.miss)
    (CGB.SensoryMaintenance.burn (CGB.RenewableLearning.Cell.aSpent, 2) 2) =
  CGB.SensoryMaintenance.empty
```
