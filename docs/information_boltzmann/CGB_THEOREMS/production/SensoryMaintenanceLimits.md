# SensoryMaintenanceLimits

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean) · [返回导航](../README.md)

模块源码说明：Finite blackout loss and actual-event energy accounting.

## _private.CGBAxioms.SensoryMaintenanceLimits.0.CGB.SensoryMaintenanceLimits.support_closed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L10)

解释：若时变性质在初态成立且沿核的每个正概率转移保持，则任意时刻违反该性质的状态概率为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {X : Type u_1} {E : Type u_2} [inst : Fintype X] [inst_1 : DecidableEq X] [inst_2 : Fintype E]
  (K : ℕ → CGB.EventChannels.Kernel X E) (x₀ : X) (G : ℕ → X → Prop),
  G 0 x₀ →
    (∀ (t : ℕ) (x : X) (e : E) (y : X), G t x → 0 < (K t).step x e y → G (t + 1) y) →
      ∀ (n : ℕ) (x : X), ¬G n x → CGB.EventChannels.Kernel.stateLaw K (fun x => if x = x₀ then 1 else 0) n x = 0
```

## _private.CGBAxioms.SensoryMaintenanceLimits.0.CGB.SensoryMaintenanceLimits.loss_closed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L36)

解释：处于感知缺失损伤态的机体集合在持续无输入环境下保持前向不变封闭性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ T : ℕ),
  3 ≤ T →
    ∀ (t : ℕ) (z z' : CGB.SensoryMaintenance.S) (e : Option Unit),
      CGB.SensoryMaintenanceLimits.lossRegion✝ n₀ t z →
        0 < (CGB.SensoryMaintenance.process false (CGB.SensoryMaintenance.sensoryProtocol n₀ T)).step t z e z' →
          CGB.SensoryMaintenanceLimits.lossRegion✝ n₀ (t + 1) z'
```

## CGB.SensoryMaintenanceLimits.blackout_body_empty

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L63)

解释：在指定连续感官中断协议中，若中断长度至少三拍，则从中断开始第三拍起所有正概率支持的身体状态为 empty。

源码说明：After three consecutive absent-sensing ticks, every supported body is empty. The world still advances at clock one throughout the interval.

形式陈述：

```lean
∀ (n₀ T n : ℕ),
  3 ≤ T →
    n₀ + 3 ≤ n →
      ∀ (z : CGB.SensoryMaintenance.S),
        0 <
            CGB.SensoryMaintenance.body.law CGB.SensoryMaintenance.world (CGB.SensoryMaintenance.sensoryProtocol n₀ T) n
              z.1 z.2 →
          z.1 () = CGB.SensoryMaintenance.empty
```

## CGB.SensoryMaintenanceLimits.blackout_survival_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L81)

解释：对指定连续感官中断协议，中断长度至少三拍且时刻不早于开始后三拍时，交换生存评分为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ T n : ℕ),
  3 ≤ T →
    n₀ + 3 ≤ n →
      CGB.Exchange.survival CGB.SensoryMaintenance.body CGB.SensoryMaintenance.world CGB.SensoryMaintenance.ec
          CGB.SensoryMaintenance.material false (CGB.SensoryMaintenance.sensoryProtocol n₀ T) n =
        0
```

## CGB.SensoryMaintenanceLimits.long_loss_not_tolerated

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L98)

解释：对指定有限感知维护见证，任何 T≥3 都不满足误差参数为零的 T 拍感官丢失耐受定义。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (T : ℕ), 3 ≤ T → ¬CGB.SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 T
```

## CGB.SensoryMaintenanceLimits.no_input_active_duration

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L108)

解释：沿指定 next 轨迹，若前 n 拍均非 calm、内容成本为正且输入为零，则 n 不超过初始内容成本。

源码说明：A declared positive loss at every active tick yields a finite duration bound from the actual first law. Clock advance alone is not the premise.

形式陈述：

```lean
∀ (stim : ℕ → CGB.SensoryMaintenance.Stim) (p : ℕ → CGB.SensoryMaintenance.State),
  (∀ (t : ℕ), p (t + 1) = CGB.SensoryMaintenance.next (stim t) (p t)) →
    ∀ (n : ℕ),
      (∀ t < n, stim t ≠ some CGB.RenewableLearning.Signal.calm ∧ 0 < CGB.SensoryMaintenance.cost (p t)) →
        (∀ t < n, CGB.SensoryMaintenance.incoming (stim t) (p t) = 0) → n ≤ CGB.SensoryMaintenance.cost (p 0)
```

## CGB.SensoryMaintenanceLimits.fuel_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L155)

解释：对指定联合过程和账本，任意有限期限的平均剩余内容加累计输出等于初始内容四加累计输入。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.SensoryMaintenance.Stim) (n : ℕ),
  CGB.Energetics.mean (CGB.Energetics.law (CGB.SensoryMaintenance.process cut sched) CGB.SensoryMaintenance.initial n)
        ((CGB.SensoryMaintenanceLimits.fuelAccount cut sched).stored n) +
      ∑ t ∈ Finset.range n,
        CGB.Energetics.flux (CGB.SensoryMaintenance.process cut sched) CGB.SensoryMaintenance.initial
          (CGB.SensoryMaintenanceLimits.port sched CGB.SensoryMaintenance.outgoing) t =
    4 +
      ∑ t ∈ Finset.range n,
        CGB.Energetics.flux (CGB.SensoryMaintenance.process cut sched) CGB.SensoryMaintenance.initial
          (CGB.SensoryMaintenanceLimits.port sched CGB.SensoryMaintenance.incoming) t
```
