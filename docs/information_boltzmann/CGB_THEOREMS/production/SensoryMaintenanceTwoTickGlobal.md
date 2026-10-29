# SensoryMaintenanceTwoTickGlobal

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean) · [返回导航](../README.md)

## _private.CGBAxioms.SensoryMaintenanceTwoTickGlobal.0.CGB.SensoryMaintenanceTwoTickGlobal.policy_table

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L9)

解释：给出全局两拍感知维护模型中神经单元状态到离散输出动作的映射表。

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

## _private.CGBAxioms.SensoryMaintenanceTwoTickGlobal.0.CGB.SensoryMaintenanceTwoTickGlobal.stateLaw_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L14)

解释：马尔可夫核在任意时刻和状态下的诱导状态转移分布值恒非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype E]
  (K : ℕ → CGB.EventChannels.Kernel S E) (initial : S) (n : ℕ) (s : S),
  0 ≤ CGB.EventChannels.Kernel.stateLaw K (fun s => if s = initial then 1 else 0) n s
```

## CGB.SensoryMaintenanceTwoTickGlobal.stateLaw_pos_of_path

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L28)

解释：只要存在一条各步转移核均为正的演化路径，终点状态在对应时刻的概率质量严格大于零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype E]
  (K : ℕ → CGB.EventChannels.Kernel S E) (initial : S) (path : ℕ → S) (event : ℕ → E),
  path 0 = initial →
    ∀ (n : ℕ),
      (∀ t < n, 0 < (K t).step (path t) (event t) (path (t + 1))) →
        0 < CGB.EventChannels.Kernel.stateLaw K (fun s => if s = initial then 1 else 0) n (path n)
```

## _private.CGBAxioms.SensoryMaintenanceTwoTickGlobal.0.CGB.SensoryMaintenanceTwoTickGlobal.path_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L79)

解释：验证特定测试路径在五拍演化过程中的每一步局部转移概率均严格为正。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ t < 5,
  0 <
    (CGB.SensoryMaintenanceTwoTickGlobal.K✝ t).step (CGB.SensoryMaintenanceTwoTickGlobal.path✝ t)
      (CGB.SensoryMaintenanceTwoTickGlobal.event✝ t) (CGB.SensoryMaintenanceTwoTickGlobal.path✝ (t + 1))
```

## CGB.SensoryMaintenanceTwoTickGlobal.empty_mass_positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L92)

解释：经历两拍环境扰动后，机体在第5拍坍缩为空状态（死态）的概率质量严格为正。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
0 <
  CGB.SensoryMaintenance.body.law CGB.SensoryMaintenance.world CGB.SensoryMaintenanceTwoTickGlobal.protocol✝ 5
    (fun x => CGB.SensoryMaintenance.empty) CGB.RenewableLearning.Env.probeB
```

## CGB.SensoryMaintenanceTwoTickGlobal.two_tick_survival_lt_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L108)

解释：在两拍感知缺失协议下，机体在第5拍的综合存活概率严格小于1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Exchange.survival CGB.SensoryMaintenance.body CGB.SensoryMaintenance.world CGB.SensoryMaintenance.ec
    CGB.SensoryMaintenance.material false CGB.SensoryMaintenanceTwoTickGlobal.protocol✝ 5 <
  1
```

## CGB.SensoryMaintenanceTwoTickGlobal.not_two_tick_tolerance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L169)

解释：形式否定机体具备对两拍感知缺失的容忍度，确立感知维护的严苛时限壁垒。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 2
```
