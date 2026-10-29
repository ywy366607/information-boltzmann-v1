# GraceRecoveryControls

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GraceRecoveryControls.lean) · [返回导航](../README.md)

模块源码说明：Separate controls for diagnostic grace and recovery lookahead Changing the lookahead alone does not repair the old absorbing failure.

## _private.CGBAxioms.GraceRecoveryControls.0.CGB.GraceRecoveryControls.point_loss

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecoveryControls.lean#L23)

解释：在单点损失条件下，优雅恢复的步进转移具有确定性的点损失界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (μ a : CGB.RenewableLearning.S → ℝ) (bad : CGB.RenewableLearning.S),
  (∀ (z : CGB.RenewableLearning.S), 0 ≤ μ z) →
    ∑ z, μ z = 1 → (∀ (z : CGB.RenewableLearning.S), a z ≤ 1) → a bad = 0 → ∑ z, a z * μ z ≤ 1 - μ bad
```

## _private.CGBAxioms.GraceRecoveryControls.0.CGB.GraceRecoveryControls.off_trap_recovery

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecoveryControls.lean#L45)

解释：在脱离陷阱控制下，系统脱离不可恢复陷阱态的转移概率有明确上界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t k : ℕ) (x : Unit → CGB.RenewableLearning.Cell),
  CGB.Exchange.recovery CGB.GraceRecoveryControls.body CGB.GraceRecoveryControls.offWorld
      CGB.GraceRecoveryControls.offClock CGB.GraceRecoveryControls.material false CGB.GraceRecoveryControls.offProtocol
      t k (x, CGB.RenewableLearning.Env.trap) =
    0
```

## CGB.GraceRecoveryControls.horizon_change_alone_fails

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecoveryControls.lean#L67)

解释：在优雅恢复控制中，仅改变决策视界长度无法挽救受损存活率，其在离线环境下的存活概率不超过二分之一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Exchange.survival CGB.GraceRecoveryControls.body CGB.GraceRecoveryControls.offWorld
    CGB.GraceRecoveryControls.offClock CGB.GraceRecoveryControls.material false CGB.GraceRecoveryControls.offProtocol
    6 ≤
  1 / 2
```

## CGB.GraceRecoveryControls.two_tick_lookahead_insufficient

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecoveryControls.lean#L89)

解释：两步前瞻机制不足以实现优雅恢复，中断状态下恢复至就绪态的概率精确为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Exchange.recovery CGB.GraceRecoveryControls.body CGB.GraceRecovery.world CGB.GraceRecovery.ec
    CGB.GraceRecoveryControls.material false (CGB.GraceRecovery.interrupted 5) 4 2
    (fun x => CGB.RenewableLearning.Cell.aReady, CGB.RenewableLearning.Env.missB) =
  0
```
