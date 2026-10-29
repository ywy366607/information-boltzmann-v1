# ActiveMaintenance

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean) · [返回导航](../README.md)

模块源码说明：Resource-dependent maintenance witness A separate successor to the refuted ActiveLoop candidate.

## CGB.ActiveMaintenance.admission

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L42)

解释：主动维持机体完全满足物理公理系统且受速率为1的事件时钟准入。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.ActiveMaintenance.body.Axioms ∧ CGB.ActiveMaintenance.body.EventClocked 1
```

## CGB.ActiveMaintenance.reads_boundary

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L107)

解释：主动维持机体内部状态演化仅读取环境边界接触信息。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.ActiveMaintenance.body.ReadsBoundary CGB.ActiveMaintenance.world
```

## _private.CGBAxioms.ActiveMaintenance.0.CGB.ActiveMaintenance.sum_configs

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L130)

解释：三状态机体构型空间实值函数的求和展开等价性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (f : (Unit → CGB.ActiveMaintenance.State) → ℝ), ∑ x, f x = ((f fun x => 0) + f fun x => 1) + f fun x => 2
```

## _private.CGBAxioms.ActiveMaintenance.0.CGB.ActiveMaintenance.kernel_table

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L139)

解释：给出主动维持机体转移核在确定性后继状态处取1的查找表结构。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : Unit → CGB.ActiveMaintenance.Stim) (x y : CGB.ActiveMaintenance.State),
  (CGB.ActiveMaintenance.body.kernel 1 s (fun x_1 => x) fun x => y) =
    if y = CGB.ActiveMaintenance.next (s ()) x then 1 else 0
```

## _private.CGBAxioms.ActiveMaintenance.0.CGB.ActiveMaintenance.natural_law

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L156)

解释：给出在自然感知流下机体各步联合演化概率密度的显式解析表达式。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n : ℕ) (a : CGB.ActiveMaintenance.State) (w : CGB.ActiveMaintenance.W),
  CGB.ActiveMaintenance.body.law CGB.ActiveMaintenance.world (fun x => CGB.ActiveMaintenance.world.sense) n (fun x => a)
      w =
    CGB.ActiveMaintenance.density✝ n a w
```

## _private.CGBAxioms.ActiveMaintenance.0.CGB.ActiveMaintenance.recovery_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L179)

解释：给出机体恢复度函数在k+1步向后动态规划递推展开式。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t k : ℕ) (a : CGB.ActiveMaintenance.State) (w : CGB.ActiveMaintenance.W),
  CGB.ActiveMaintenance.recover✝ false t (k + 1) a w =
    ∑ v,
      CGB.ActiveMaintenance.moves w (fun x => a) v *
        if 0 < CGB.ActiveMaintenance.incoming (CGB.ActiveMaintenance.sense w ()) a then 1
        else
          CGB.ActiveMaintenance.recover✝ false (t + 1) k
            (CGB.ActiveMaintenance.next (CGB.ActiveMaintenance.sense w ()) a) v
```

## _private.CGBAxioms.ActiveMaintenance.0.CGB.ActiveMaintenance.normal_recovery

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L205)

解释：正常环境交互下机体的多步恢复度量与预设前瞻成功概率完全一致。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t k : ℕ) (a : CGB.ActiveMaintenance.State) (w : CGB.ActiveMaintenance.W),
  CGB.ActiveMaintenance.recover✝ false t k a w = CGB.ActiveMaintenance.prospect✝ k a w
```

## _private.CGBAxioms.ActiveMaintenance.0.CGB.ActiveMaintenance.natural_alive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L211)

解释：给出自然演化下机体处于存活状态的充要判定条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t : ℕ) (a : CGB.ActiveMaintenance.State) (w : CGB.ActiveMaintenance.W),
  CGB.Exchange.alive CGB.ActiveMaintenance.body CGB.ActiveMaintenance.world CGB.ActiveMaintenance.he
      CGB.ActiveMaintenance.ledger false (fun x => CGB.ActiveMaintenance.world.sense) t (fun x => a, w) =
    if a ≠ 0 ∧ 1 / 2 ≤ CGB.ActiveMaintenance.prospect✝ 3 a w then 1 else 0
```

## CGB.ActiveMaintenance.natural_survival

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L227)

解释：在自然平稳无损的环境感知驱动下，机体的全时期存活概率恒等于1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n : ℕ),
  CGB.Exchange.survival CGB.ActiveMaintenance.body CGB.ActiveMaintenance.world CGB.ActiveMaintenance.he
      CGB.ActiveMaintenance.ledger false (fun x => CGB.ActiveMaintenance.world.sense) n =
    1
```

## _private.CGBAxioms.ActiveMaintenance.0.CGB.ActiveMaintenance.starve_world_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L246)

解释：在饥饿吸收状态下环境以概率1停留在死态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (t : ℕ) (x : Unit → CGB.ActiveMaintenance.State) (v : CGB.ActiveMaintenance.W),
  CGB.ActiveMaintenance.body.eventWorldStep CGB.ActiveMaintenance.world cut (fun x => CGB.ActiveMaintenance.world.sense)
      t (x, 5) v =
    if v = 5 then 1 else 0
```

## _private.CGBAxioms.ActiveMaintenance.0.CGB.ActiveMaintenance.starve_recovery

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L258)

解释：处于饥饿状态下的机体在任意有限视界内的恢复度量恒为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (t k : ℕ) (a : CGB.ActiveMaintenance.State), CGB.ActiveMaintenance.recover✝ cut t k a 5 = 0
```

## _private.CGBAxioms.ActiveMaintenance.0.CGB.ActiveMaintenance.cut_move_early

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L278)

解释：在前3个环境状态内，施加截断后的环境转移算子与无截断动力学完全一致。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t : ℕ) (x : Unit → CGB.ActiveMaintenance.State) (w v : CGB.ActiveMaintenance.W),
  ↑w < 3 →
    CGB.Medium.cutMove CGB.ActiveMaintenance.world
        (CGB.ActiveMaintenance.body.lawCut CGB.ActiveMaintenance.world (fun x => CGB.ActiveMaintenance.world.sense) t) x
        w v =
      CGB.ActiveMaintenance.moves w x v
```

## _private.CGBAxioms.ActiveMaintenance.0.CGB.ActiveMaintenance.cut_early_law

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L292)

解释：在演化前2步内，截断演化下的联合状态概率分布与无截断自然分布精确相同。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ n ≤ 2,
  ∀ (a : CGB.ActiveMaintenance.State) (w : CGB.ActiveMaintenance.W),
    CGB.ActiveMaintenance.body.lawCut CGB.ActiveMaintenance.world (fun x => CGB.ActiveMaintenance.world.sense) n
        (fun x => a) w =
      CGB.ActiveMaintenance.density✝ n a w
```

## _private.CGBAxioms.ActiveMaintenance.0.CGB.ActiveMaintenance.starve_mass_three

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L312)

解释：在演化第3步落入饥饿死态的概率质量精确等于1/2。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.ActiveMaintenance.starveMass✝ 3 = 1 / 2
```

## _private.CGBAxioms.ActiveMaintenance.0.CGB.ActiveMaintenance.starve_mass_monotone

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L322)

解释：系统处于饥饿死态的累积概率质量随时间步单调递增。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n : ℕ), CGB.ActiveMaintenance.starveMass✝ n ≤ CGB.ActiveMaintenance.starveMass✝ (n + 1)
```

## _private.CGBAxioms.ActiveMaintenance.0.CGB.ActiveMaintenance.cut_alive_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L348)

解释：截断演化下的存活指示测度在进入饥饿环境死态时严格归零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n : ℕ) (a : CGB.ActiveMaintenance.State) (w : CGB.ActiveMaintenance.W),
  CGB.Exchange.alive CGB.ActiveMaintenance.body CGB.ActiveMaintenance.world CGB.ActiveMaintenance.he
      CGB.ActiveMaintenance.ledger true (fun x => CGB.ActiveMaintenance.world.sense) n (fun x => a, w) ≤
    if w = 5 then 0 else 1
```

## _private.CGBAxioms.ActiveMaintenance.0.CGB.ActiveMaintenance.cut_survival_mass_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L363)

解释：导出系统截断存活概率受限于1减去累积饥饿概率质量的解析上界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n : ℕ),
  CGB.Exchange.survival CGB.ActiveMaintenance.body CGB.ActiveMaintenance.world CGB.ActiveMaintenance.he
      CGB.ActiveMaintenance.ledger true (fun x => CGB.ActiveMaintenance.world.sense) n ≤
    1 - CGB.ActiveMaintenance.starveMass✝ n
```

## CGB.ActiveMaintenance.cut_survival

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L390)

解释：施加环路截断后，机体在第4步的综合存活概率严格低于1/2（确立闭环维持必要性）。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n : ℕ),
  3 ≤ n →
    CGB.Exchange.survival CGB.ActiveMaintenance.body CGB.ActiveMaintenance.world CGB.ActiveMaintenance.he
        CGB.ActiveMaintenance.ledger true (fun x => CGB.ActiveMaintenance.world.sense) n ≤
      1 / 2
```

## CGB.ActiveMaintenance.maintenance_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L403)

解释：构造出主动维持机体同时满足持续生存与自我维持公理的显式见证。

源码说明：The existing gross-exchange targets, for ALL future times. This is a maintenance subtarget witness, not an encoded or cognitive life candidate.

形式陈述：

```lean
CGB.Exchange.Persists CGB.ActiveMaintenance.body CGB.ActiveMaintenance.world CGB.ActiveMaintenance.he
    CGB.ActiveMaintenance.ledger 0 ∧
  CGB.Exchange.SelfMaintains CGB.ActiveMaintenance.body CGB.ActiveMaintenance.world CGB.ActiveMaintenance.he
    CGB.ActiveMaintenance.ledger
```

## CGB.ActiveMaintenance.starvation_depletes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L416)

解释：在饥饿环境下机体体内有效物质储备不可逆地耗散清空。

源码说明：Scarcity really changes the body reserve, under either process; vacuum then has no action in any stimulus. The event clock and ghost future are kept.

形式陈述：

```lean
∀ (cut : Bool) (t : ℕ) (a : CGB.ActiveMaintenance.State),
  (CGB.ActiveMaintenance.body.jointEventKernel CGB.ActiveMaintenance.world CGB.ActiveMaintenance.he cut
            (fun x => CGB.ActiveMaintenance.world.sense) t).step
        (fun x => a, 5) (if a = 0 then none else some ()) (fun x => 0, 5) =
      1 ∧
    CGB.ActiveMaintenance.next 4 a = 0 ∧
      CGB.ActiveMaintenance.cost (CGB.ActiveMaintenance.next 4 a) = 0 ∧
        ∀ (s : CGB.ActiveMaintenance.Stim), CGB.ActiveMaintenance.next s 0 = 0
```

## CGB.ActiveMaintenance.sustained_exchange

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L432)

解释：在正常闭环交互中机体持续与环境维持非零且平衡的物质交换。

源码说明：Normal maintenance consumes and imports a positive unit EVERY active tick; this demand is a property of this mechanism, not a universal life axiom.

形式陈述：

```lean
∀ (n : ℕ),
  2 ≤ n →
    ∑ x,
          ∑ w,
            CGB.ActiveMaintenance.body.law CGB.ActiveMaintenance.world (fun x => CGB.ActiveMaintenance.world.sense) n x
                w *
              ↑(CGB.ActiveMaintenance.incoming (CGB.ActiveMaintenance.sense w ()) (x ())) =
        1 ∧
      ∑ x,
          ∑ w,
            CGB.ActiveMaintenance.body.law CGB.ActiveMaintenance.world (fun x => CGB.ActiveMaintenance.world.sense) n x
                w *
              ↑(CGB.ActiveMaintenance.outgoing (CGB.ActiveMaintenance.sense w ()) (x ())) =
        1
```
