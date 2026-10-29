# RenewableCandidateAudit

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean) · [返回导航](../README.md)

模块源码说明：A concrete obstruction to the missing sensory-interruption obligation Keep the original body, world, clock and interruption protocol.

## _private.CGBAxioms.RenewableCandidateAudit.0.CGB.RenewableCandidateAudit.kernel_at_next

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L30)

解释：在可再生候选机体中，由确定性后继状态映射诱导的转移核概率精确为1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : Unit → CGB.RenewableLearning.Signal) (q : CGB.RenewableLearning.Cell),
  (CGB.RenewableCandidateAudit.body.kernel 1 s (fun x => q) fun x => CGB.RenewableLearning.next true true (s ()) q) = 1
```

## _private.CGBAxioms.RenewableCandidateAudit.0.CGB.RenewableCandidateAudit.policy_values

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L37)

解释：给出可再生候选机体在搜索态与A通道就绪态下的决策输出动作常数值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.RenewableLearning.policy CGB.RenewableLearning.Cell.search = 2 ∧
  CGB.RenewableLearning.policy CGB.RenewableLearning.Cell.aSpent = 0 ∧
    CGB.RenewableLearning.policy CGB.RenewableLearning.Cell.aReady = 0
```

## _private.CGBAxioms.RenewableCandidateAudit.0.CGB.RenewableCandidateAudit.edge

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L40)

解释：给出在环境时序扰动下候选机体在特定路径边上的联合转移概率值（关键分支处为1/2，其余为1）。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (T : ℕ),
  0 < T →
    ∀ (i : Fin 6),
      CGB.RenewableLearning.moves (CGB.RenewableCandidateAudit.path✝ ↑i).2 (CGB.RenewableCandidateAudit.path✝ ↑i).1
            (CGB.RenewableCandidateAudit.path✝ (↑i + 1)).2 *
          CGB.RenewableCandidateAudit.body.kernel 1
            (CGB.RenewableCandidateAudit.schedule T (↑i) (CGB.RenewableCandidateAudit.path✝ ↑i).2)
            (CGB.RenewableCandidateAudit.path✝ ↑i).1 (CGB.RenewableCandidateAudit.path✝ (↑i + 1)).1 =
        if ↑i = 2 then 1 / 2 else 1
```

## _private.CGBAxioms.RenewableCandidateAudit.0.CGB.RenewableCandidateAudit.step_lower

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L51)

解释：建立候选审计机体在一步转移过程中的概率非负性与正下界估计。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.RenewableLearning.Signal) (n : ℕ) (x y : CGB.RenewableLearning.S),
  CGB.RenewableCandidateAudit.body.law CGB.RenewableLearning.world sched n x.1 x.2 *
      (CGB.RenewableLearning.moves x.2 x.1 y.2 * CGB.RenewableCandidateAudit.body.kernel 1 (sched n x.2) x.1 y.1) ≤
    CGB.RenewableCandidateAudit.body.law CGB.RenewableLearning.world sched (n + 1) y.1 y.2
```

## _private.CGBAxioms.RenewableCandidateAudit.0.CGB.RenewableCandidateAudit.path_lower

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L78)

解释：沿着特定干扰演化轨迹，系统在各个时刻的概率质量始终保持严格正下界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.RenewableLearning.Signal) (n : ℕ),
  ∏ t ∈ Finset.range n,
      CGB.RenewableLearning.moves (CGB.RenewableCandidateAudit.path✝ t).2 (CGB.RenewableCandidateAudit.path✝ t).1
          (CGB.RenewableCandidateAudit.path✝ (t + 1)).2 *
        CGB.RenewableCandidateAudit.body.kernel 1 (sched t (CGB.RenewableCandidateAudit.path✝ t).2)
          (CGB.RenewableCandidateAudit.path✝ t).1 (CGB.RenewableCandidateAudit.path✝ (t + 1)).1 ≤
    CGB.RenewableCandidateAudit.body.law CGB.RenewableLearning.world sched n (CGB.RenewableCandidateAudit.path✝ n).1
      (CGB.RenewableCandidateAudit.path✝ n).2
```

## CGB.RenewableCandidateAudit.interrupted_trap_mass

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L90)

解释：在环境间歇性中断下，机体陷落于不可恢复耗尽死态的概率质量随时间累积。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (T : ℕ),
  0 < T →
    1 / 2 ≤
      CGB.RenewableCandidateAudit.body.law CGB.RenewableLearning.world (CGB.RenewableCandidateAudit.schedule T) 6
        (fun x => CGB.RenewableLearning.Cell.aReady) CGB.RenewableLearning.Env.trap
```

## _private.CGBAxioms.RenewableCandidateAudit.0.CGB.RenewableCandidateAudit.trap_recovery

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L103)

解释：一旦系统状态跌入中断陷阱，其物质恢复通量与恢复度量均不可逆地归零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (T t k : ℕ) (x : Unit → CGB.RenewableLearning.Cell),
  CGB.Exchange.recovery CGB.RenewableCandidateAudit.body CGB.RenewableLearning.world ⋯
      (CGB.RenewableLearning.ledger true true) false (CGB.RenewableCandidateAudit.schedule T) t k
      (x, CGB.RenewableLearning.Env.trap) =
    0
```

## CGB.RenewableCandidateAudit.interruption_failure

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L127)

解释：受审候选机体在有限时段的环境扰动中断下必然发生恢复失败且存活率严格小于1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (T : ℕ),
  0 < T →
    CGB.Exchange.survival CGB.RenewableCandidateAudit.body CGB.RenewableLearning.world ⋯
        (CGB.RenewableLearning.ledger true true) false (CGB.RenewableCandidateAudit.schedule T) 6 ≤
      1 / 2
```

## CGB.RenewableCandidateAudit.no_positive_interruption_tolerance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L165)

解释：该受审候选机制无法耐受任何正时长的环境中断，从而否定了其作为鲁棒可再生架构的合格性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (T : ℕ),
  ¬CGB.Exchange.ToleratesSensoryInterruption CGB.RenewableCandidateAudit.body CGB.RenewableLearning.world ⋯
      (CGB.RenewableLearning.ledger true true) 0 T
```
