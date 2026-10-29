# SensoryMaintenanceRecovery

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean) · [返回导航](../README.md)

## _private.CGBAxioms.SensoryMaintenanceRecovery.0.CGB.SensoryMaintenanceRecovery.intake_same

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L14)

解释：感知维护系统在受到刺激时的正摄入判定与原可再生学习模型在相应信号下的正摄入严格等价。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : CGB.SensoryMaintenance.Stim) (x : CGB.SensoryMaintenance.State),
  decide (0 < CGB.SensoryMaintenance.incoming s x) =
    decide (0 < CGB.RenewableLearning.incoming true (Option.getD s CGB.RenewableLearning.Signal.calm) x.1)
```

## _private.CGBAxioms.SensoryMaintenanceRecovery.0.CGB.SensoryMaintenanceRecovery.old_event

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L22)

解释：在非真空且具有正储备的机体状态下，感知系统的单步演化核与原宽限恢复事件核转移概率精确为1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : CGB.SensoryMaintenance.Stim) (x : CGB.SensoryMaintenance.State),
  x.1 ≠ CGB.RenewableLearning.Cell.vacuum →
    0 < ↑x.2 →
      (CGB.GraceRecovery.body.eventStep 1 (fun x => Option.getD s CGB.RenewableLearning.Signal.calm) (fun x_1 => x.1)
          (CGB.SensoryMaintenanceRecovery.oldEvent s) fun x_1 => (CGB.SensoryMaintenance.next s x).1) =
        1
```

## _private.CGBAxioms.SensoryMaintenanceRecovery.0.CGB.SensoryMaintenanceRecovery.supported

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L34)

解释：在非静默刺激与非真空支持下，感知机体的后继状态、事件跃迁与物质摄入通量严格保真投影至宽限恢复系统。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.SensoryMaintenance.Stim) (t : ℕ)
  (z z' : CGB.SensoryMaintenanceRecovery.S) (e : Option Unit),
  sched t z.2 () ≠ some CGB.RenewableLearning.Signal.calm →
    (z.1 ()).1 ≠ CGB.RenewableLearning.Cell.vacuum →
      0 < ↑(z.1 ()).2 →
        0 <
            (CGB.SensoryMaintenance.body.jointEventKernel CGB.SensoryMaintenance.world CGB.SensoryMaintenance.ec false
                  sched t).step
              z e z' →
          z'.1 () = CGB.SensoryMaintenance.next (sched t z.2 ()) (z.1 ()) ∧
            0 <
                (CGB.GraceRecovery.body.jointEventKernel CGB.GraceRecovery.world CGB.GraceRecovery.ec false
                      (CGB.SensoryMaintenanceRecovery.decode sched) t).step
                  (CGB.SensoryMaintenanceRecovery.project z) (CGB.SensoryMaintenanceRecovery.oldEvent (sched t z.2 ()))
                  (CGB.SensoryMaintenanceRecovery.project z') ∧
              CGB.Exchange.intake CGB.SensoryMaintenance.body CGB.SensoryMaintenance.world
                  CGB.SensoryMaintenance.material sched t z e z' =
                CGB.Exchange.intake CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.material
                  (CGB.SensoryMaintenanceRecovery.decode sched) t (CGB.SensoryMaintenanceRecovery.project z)
                  (CGB.SensoryMaintenanceRecovery.oldEvent (sched t z.2 ())) (CGB.SensoryMaintenanceRecovery.project z')
```

## _private.CGBAxioms.SensoryMaintenanceRecovery.0.CGB.SensoryMaintenanceRecovery.sensed_not_calm

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L87)

解释：任意环境状态在感知函数映射下的传感器输出信号均不等于静默信号。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (w : CGB.RenewableLearning.Env), some (CGB.RenewableLearning.sense w ()) ≠ some CGB.RenewableLearning.Signal.calm
```

## CGB.SensoryMaintenanceRecovery.decode_protocol

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L90)

解释：长度为1的感知中断协议经解码后精确还原为宽限恢复框架中的对应单步中断环境。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ : ℕ),
  CGB.SensoryMaintenanceRecovery.decode (CGB.SensoryMaintenance.sensoryProtocol n₀ 1) = CGB.GraceRecovery.interrupted n₀
```

## _private.CGBAxioms.SensoryMaintenanceRecovery.0.CGB.SensoryMaintenanceRecovery.positive_region

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L96)

解释：凡是落入宽限恢复安全区域内的系统状态，其内部机体单元构型必非真空死态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ t : ℕ) (z : CGB.SensoryMaintenanceRecovery.S),
  (CGB.GraceRecovery.certificate n₀).region t (CGB.SensoryMaintenanceRecovery.project z) →
    (z.1 ()).1 ≠ CGB.RenewableLearning.Cell.vacuum
```

## _private.CGBAxioms.SensoryMaintenanceRecovery.0.CGB.SensoryMaintenanceRecovery.reserve_sensed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L104)

解释：在有效感知信号输入下，具有正储备的非真空状态在转移后持续保持正储备且在满储备时不变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (a : CGB.RenewableLearning.Signal) (x : CGB.SensoryMaintenance.State),
  1 ≤ ↑x.2 →
    (CGB.SensoryMaintenance.next (some a) x).1 ≠ CGB.RenewableLearning.Cell.vacuum →
      1 ≤ ↑(CGB.SensoryMaintenance.next (some a) x).2 ∧ (↑x.2 = 2 → ↑(CGB.SensoryMaintenance.next (some a) x).2 = 2)
```

## CGB.SensoryMaintenanceRecovery.one_tick_survival

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L154)

解释：在经历单拍感知丢失后实施恢复协议，系统在任意未来时刻的生存概率恒等于1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ n : ℕ),
  CGB.Exchange.survival CGB.SensoryMaintenance.body CGB.SensoryMaintenance.world CGB.SensoryMaintenance.ec
      CGB.SensoryMaintenance.material false (CGB.SensoryMaintenance.sensoryProtocol n₀ 1) n =
    1
```

## CGB.SensoryMaintenanceRecovery.natural_survival

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L187)

解释：在自然平稳无损的感知输入流驱动下，机体的全时期生存概率恒等于1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n : ℕ),
  CGB.Exchange.survival CGB.SensoryMaintenance.body CGB.SensoryMaintenance.world CGB.SensoryMaintenance.ec
      CGB.SensoryMaintenance.material false (fun x => CGB.SensoryMaintenance.world.sense) n =
    1
```

## CGB.SensoryMaintenanceRecovery.one_tick_tolerance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L205)

解释：机体具备对单拍感知丢失的完全容忍耐受性（单拍耐受定理成立）。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 1
```
