# SensoryMaintenanceRepairCandidates

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean) · [返回导航](../README.md)

模块源码说明：Minimal repair candidates after the two-tick refutation The registered model is unchanged.

## _private.CGBAxioms.SensoryMaintenanceRepairCandidates.0.CGB.SensoryMaintenanceRepairCandidates.policy_table

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L16)

解释：给出指定可再生学习单元各 Cell 状态对应的确定性策略动作表。

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

## CGB.SensoryMaintenanceRepairCandidates.protected_second_tick

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L26)

解释：指定 protectedTwoTickProtocol 在中断起始后的第二拍输出 some probe。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ : ℕ) (w : CGB.RenewableLearning.Env) (u : Unit),
  CGB.SensoryMaintenanceRepairCandidates.protectedTwoTickProtocol n₀ (n₀ + 1) w u =
    some CGB.RenewableLearning.Signal.probe
```

## CGB.SensoryMaintenanceRepairCandidates.protected_first_tick

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L30)

解释：指定 protectedTwoTickProtocol 在中断起始拍输出 none。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ : ℕ) (w : CGB.RenewableLearning.Env) (u : Unit),
  CGB.SensoryMaintenanceRepairCandidates.protectedTwoTickProtocol n₀ n₀ w u = none
```

## CGB.SensoryMaintenanceRepairCandidates.protocol_repair_preserves_one_tick

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L34)

解释：复用已有定理，证明原感知维护见证满足误差参数零的一拍感官丢失耐受条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 1
```

## CGB.SensoryMaintenanceRepairCandidates.probe_recharges_after_one_loss

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L38)

解释：若初态单元非 vacuum 且储备至少一，一拍 none 后接收 probe 具有正输入、保持单元非 vacuum 且储备变为二。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (x : CGB.SensoryMaintenance.State),
  x.1 ≠ CGB.RenewableLearning.Cell.vacuum →
    1 ≤ ↑x.2 →
      0 <
          CGB.SensoryMaintenance.incoming (some CGB.RenewableLearning.Signal.probe)
            (CGB.SensoryMaintenance.next none x) ∧
        (CGB.SensoryMaintenance.next (some CGB.RenewableLearning.Signal.probe) (CGB.SensoryMaintenance.next none x)).1 ≠
            CGB.RenewableLearning.Cell.vacuum ∧
          ↑(CGB.SensoryMaintenance.next (some CGB.RenewableLearning.Signal.probe)
                  (CGB.SensoryMaintenance.next none x)).2 =
            2
```

## CGB.SensoryMaintenanceRepairCandidates.protected_probe_path_outgoing

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L53)

解释：若初态单元非 vacuum 且储备至少一，一拍 none 与随后 probe 两步的登记输出量之和为二。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (x : CGB.SensoryMaintenance.State),
  x.1 ≠ CGB.RenewableLearning.Cell.vacuum →
    1 ≤ ↑x.2 →
      CGB.SensoryMaintenance.outgoing none x +
          CGB.SensoryMaintenance.outgoing (some CGB.RenewableLearning.Signal.probe)
            (CGB.SensoryMaintenance.next none x) =
        2
```

## CGB.SensoryMaintenanceRepairCandidates.protected_probe_path_incoming

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L66)

解释：若初态单元非 vacuum 且储备至少二，一拍 none 后接收 probe 的登记输入量为二。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (x : CGB.SensoryMaintenance.State),
  x.1 ≠ CGB.RenewableLearning.Cell.vacuum →
    2 ≤ ↑x.2 →
      CGB.SensoryMaintenance.incoming (some CGB.RenewableLearning.Signal.probe) (CGB.SensoryMaintenance.next none x) = 2
```

## CGB.SensoryMaintenanceRepairCandidates.protected_branch_excludes_local_blackout

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L78)

解释：若初态单元非 vacuum 且储备至少一，一拍 none 再接收 probe 后，状态不等于 empty。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (x : CGB.SensoryMaintenance.State),
  x.1 ≠ CGB.RenewableLearning.Cell.vacuum →
    1 ≤ ↑x.2 →
      CGB.SensoryMaintenance.next (some CGB.RenewableLearning.Signal.probe) (CGB.SensoryMaintenance.next none x) ≠
        CGB.SensoryMaintenance.empty
```

## CGB.SensoryMaintenanceRepairCandidates.initial_guarded_two_tick_start

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L102)

解释：指定初始 Cell.search 和 Env.probeA 的组合满足 guardedTwoTickStart。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.SensoryMaintenanceRepairCandidates.guardedTwoTickStart CGB.RenewableLearning.Cell.search
  CGB.RenewableLearning.Env.probeA
```

## CGB.SensoryMaintenanceRepairCandidates.guarded_third_tick_recharges

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L116)

解释：若起始状态满足守卫条件且两次世界转移均有正概率，满储备身体 burn 两拍后接收第二次世界感知会有正输入并恢复储备二。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell) (w : CGB.RenewableLearning.Env),
  CGB.SensoryMaintenanceRepairCandidates.guardedTwoTickStart q w →
    ∀ (v₁ v₂ : CGB.RenewableLearning.Env),
      0 < CGB.GraceRecovery.moves w (fun x => q) v₁ →
        0 < CGB.GraceRecovery.moves v₁ (fun x => q) v₂ →
          0 <
              CGB.SensoryMaintenance.incoming (some (CGB.RenewableLearning.sense v₂ ()))
                (CGB.SensoryMaintenance.burn (q, 2) 2) ∧
            ↑(CGB.SensoryMaintenance.next (some (CGB.RenewableLearning.sense v₂ ()))
                    (CGB.SensoryMaintenance.burn (q, 2) 2)).2 =
              2
```

## CGB.SensoryMaintenanceRepairCandidates.guarded_repair_preserves_one_tick

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L149)

解释：复用已有定理，证明原感知维护见证满足误差参数零的一拍感官丢失耐受条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 1
```

## CGB.SensoryMaintenanceRepairCandidates.firstHit_three_tick_layered

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L153)

解释：若所有正概率一步、二步后继依次进入给定层且未命中，而第三步均命中，则从初态三拍内首次命中概率为一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : Fintype E] (K : ℕ → CGB.EventChannels.Kernel S E)
  (mark : ℕ → S → E → S → Bool) (t : ℕ) (s₀ : S) (layer₁ layer₂ : S → Prop),
  (∀ (e : E) (s₁ : S), 0 < (K t).step s₀ e s₁ → layer₁ s₁ ∧ mark t s₀ e s₁ = false) →
    (∀ (s₁ : S),
        layer₁ s₁ → ∀ (e : E) (s₂ : S), 0 < (K (t + 1)).step s₁ e s₂ → layer₂ s₂ ∧ mark (t + 1) s₁ e s₂ = false) →
      (∀ (s₁ : S),
          layer₁ s₁ →
            ∀ (s₂ : S), layer₂ s₂ → ∀ (e : E) (s₃ : S), 0 < (K (t + 2)).step s₂ e s₃ → mark (t + 2) s₂ e s₃ = true) →
        CGB.EventChannels.Kernel.firstHit K mark t 3 s₀ = 1
```

## CGB.SensoryMaintenanceRepairCandidates.guarded_firstHit

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L216)

解释：若 q 与 w 满足 guardedTwoTickStart，则从身体状态 (q,2) 出发，指定守卫协议下三拍内首次摄入概率为一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ : ℕ) (q : CGB.RenewableLearning.Cell) (w : CGB.RenewableLearning.Env),
  CGB.SensoryMaintenanceRepairCandidates.guardedTwoTickStart q w →
    have sched := CGB.SensoryMaintenanceRepairCandidates.guardedLossProtocol n₀;
    have K := fun t =>
      CGB.SensoryMaintenance.body.jointEventKernel CGB.SensoryMaintenance.world CGB.SensoryMaintenance.ec false sched t;
    have mark :=
      CGB.Exchange.intake CGB.SensoryMaintenance.body CGB.SensoryMaintenance.world CGB.SensoryMaintenance.material
        sched;
    CGB.EventChannels.Kernel.firstHit K mark n₀ 3 (fun x => (q, 2), w) = 1
```

## CGB.SensoryMaintenanceRepairCandidates.guarded_alive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L313)

解释：若 q 与 w 满足 guardedTwoTickStart，则从身体状态 (q,2) 出发，在指定守卫协议的中断时刻交换生命指标为一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ : ℕ) (q : CGB.RenewableLearning.Cell) (w : CGB.RenewableLearning.Env),
  CGB.SensoryMaintenanceRepairCandidates.guardedTwoTickStart q w →
    CGB.Exchange.alive CGB.SensoryMaintenance.body CGB.SensoryMaintenance.world CGB.SensoryMaintenance.ec
        CGB.SensoryMaintenance.material false (CGB.SensoryMaintenanceRepairCandidates.guardedLossProtocol n₀) n₀
        (fun x => (q, 2), w) =
      1
```

## CGB.SensoryMaintenanceRepairCandidates.guarded_two_tick_local_tolerance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L333)

解释：指定守卫协议满足从时刻零出发的 GuardedTwoTickLocalTolerance；该结论本身不是全局两拍耐受。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.SensoryMaintenanceRepairCandidates.GuardedTwoTickLocalTolerance 0
```

## CGB.SensoryMaintenanceRepairCandidates.guarded_survival_at

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L344)

解释：若中断时刻满足 GuardedTwoTickSupport，则指定守卫协议在该时刻的交换生存评分为一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ : ℕ),
  CGB.SensoryMaintenanceRepairCandidates.GuardedTwoTickSupport n₀ →
    CGB.Exchange.survival CGB.SensoryMaintenance.body CGB.SensoryMaintenance.world CGB.SensoryMaintenance.ec
        CGB.SensoryMaintenance.material false (CGB.SensoryMaintenanceRepairCandidates.guardedLossProtocol n₀) n₀ =
      1
```

## CGB.SensoryMaintenanceRepairCandidates.guarded_support_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L396)

解释：指定守卫协议在起始时刻零满足 GuardedTwoTickSupport 条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.SensoryMaintenanceRepairCandidates.GuardedTwoTickSupport 0
```

## _private.CGBAxioms.SensoryMaintenanceRepairCandidates.0.CGB.SensoryMaintenanceRepairCandidates.guarded_support_path_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L421)

解释：验证特定测试路径在前两步演化中的局部转移概率均严格大于零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ t < 2,
  0 <
    (CGB.SensoryMaintenanceRepairCandidates.guardedSupportK✝ t).step
      (CGB.SensoryMaintenanceRepairCandidates.guardedSupportPath✝ t)
      (CGB.SensoryMaintenanceRepairCandidates.guardedSupportEvent✝ t)
      (CGB.SensoryMaintenanceRepairCandidates.guardedSupportPath✝ (t + 1))
```

## CGB.SensoryMaintenanceRepairCandidates.not_guarded_support_two

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L436)

解释：指定守卫协议在时刻二不满足 GuardedTwoTickSupport。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.SensoryMaintenanceRepairCandidates.GuardedTwoTickSupport 2
```

## CGB.SensoryMaintenanceRepairCandidates.guarded_initial_survival

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L460)

解释：从指定初始状态运行在零拍开始的守卫协议时，零时刻交换生存评分为一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Exchange.survival CGB.SensoryMaintenance.body CGB.SensoryMaintenance.world CGB.SensoryMaintenance.ec
    CGB.SensoryMaintenance.material false (CGB.SensoryMaintenanceRepairCandidates.guardedLossProtocol 0) 0 =
  1
```

## CGB.SensoryMaintenanceRepairCandidates.eligible_excludes_spent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L466)

解释：满足指定两拍合格初态条件的状态，其主体单元既不是 aSpent 也不是 bSpent。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (x : CGB.SensoryMaintenance.State),
  CGB.SensoryMaintenanceRepairCandidates.eligibleTwoTickStart x →
    x.1 ≠ CGB.RenewableLearning.Cell.aSpent ∧ x.1 ≠ CGB.RenewableLearning.Cell.bSpent
```

## CGB.SensoryMaintenanceRepairCandidates.eligible_excludes_verified_dangerous_branch

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L472)

解释：满足指定两拍合格初态条件时，主体为 aSpent 且世界为 feedA 的组合不可能成立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (x : CGB.SensoryMaintenance.State) (w : CGB.RenewableLearning.Env),
  CGB.SensoryMaintenanceRepairCandidates.eligibleTwoTickStart x →
    ¬(x.1 = CGB.RenewableLearning.Cell.aSpent ∧ w = CGB.RenewableLearning.Env.feedA)
```

## CGB.SensoryMaintenanceRepairCandidates.eligible_two_loss_preserves_cell

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L479)

解释：对指定合格初态且初始储备至少为二，连续两拍无感知后主体单元不变且储备为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (x : CGB.SensoryMaintenance.State),
  CGB.SensoryMaintenanceRepairCandidates.eligibleTwoTickStart x →
    2 ≤ ↑x.2 →
      (CGB.SensoryMaintenance.next none (CGB.SensoryMaintenance.next none x)).1 = x.1 ∧
        ↑(CGB.SensoryMaintenance.next none (CGB.SensoryMaintenance.next none x)).2 = 0
```

## CGB.SensoryMaintenanceRepairCandidates.eligible_keeps_existing_budget_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L491)

解释：对任意刺激和状态，原感知维护更新满足更新后内容成本加流出等于更新前成本加流入。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : CGB.SensoryMaintenance.Stim) (x : CGB.SensoryMaintenance.State),
  CGB.SensoryMaintenance.cost (CGB.SensoryMaintenance.next s x) + CGB.SensoryMaintenance.outgoing s x =
    CGB.SensoryMaintenance.cost x + CGB.SensoryMaintenance.incoming s x
```

## CGB.SensoryMaintenanceRepairCandidates.region_repair_preserves_one_tick

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L498)

解释：复用已有定理，证明原感知维护见证满足误差参数零的一拍感官丢失耐受条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 1
```
