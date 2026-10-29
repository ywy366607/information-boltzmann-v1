# GraceRecovery

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean) · [返回导航](../README.md)

模块源码说明：A recoverable diagnostic opportunity after missed sensing Keep the original body, learning payments, food amounts and clock.

## _private.CGBAxioms.GraceRecovery.0.CGB.GraceRecovery.policy_table

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L50)

解释：给出宽限恢复模型中各细胞构型对应策略动作的显式分段查找表。

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

## CGB.GraceRecovery.moves_unchanged_on_good

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L68)

解释：在良性状态下，宽限恢复环境的转移概率与标准可再生学习环境的转移完全相同。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell) (w v : CGB.RenewableLearning.Env),
  CGB.GraceRecovery.Good q w → CGB.GraceRecovery.moves w (fun x => q) v = CGB.RenewableLearning.moves w (fun x => q) v
```

## _private.CGBAxioms.GraceRecovery.0.CGB.GraceRecovery.good_safe

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L73)

解释：所有处于良性集合的状态必然属于安全集合。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell) (w : CGB.RenewableLearning.Env),
  CGB.GraceRecovery.Good q w → CGB.GraceRecovery.Safe q w
```

## _private.CGBAxioms.GraceRecovery.0.CGB.GraceRecovery.safe_nonvacuum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L76)

解释：处于安全集合的机体构型必非真空空虚死态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell) (w : CGB.RenewableLearning.Env),
  CGB.GraceRecovery.Safe q w → q ≠ CGB.RenewableLearning.Cell.vacuum
```

## _private.CGBAxioms.GraceRecovery.0.CGB.GraceRecovery.normal_good

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L79)

解释：在正常环境转移下，安全状态在一步演化后依然保持在良性集合内。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell) (w v : CGB.RenewableLearning.Env),
  CGB.GraceRecovery.Good q w →
    0 < CGB.GraceRecovery.moves w (fun x => q) v →
      CGB.GraceRecovery.Good (CGB.RenewableLearning.next true true (CGB.RenewableLearning.sense w ()) q) v
```

## _private.CGBAxioms.GraceRecovery.0.CGB.GraceRecovery.normal_safe

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L86)

解释：在正常环境转移下，安全状态集合在一步转移后保持前向不变封闭性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell) (w v : CGB.RenewableLearning.Env),
  CGB.GraceRecovery.Safe q w →
    0 < CGB.GraceRecovery.moves w (fun x => q) v →
      CGB.GraceRecovery.Safe (CGB.RenewableLearning.next true true (CGB.RenewableLearning.sense w ()) q) v
```

## _private.CGBAxioms.GraceRecovery.0.CGB.GraceRecovery.idle_safe

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L92)

解释：处于良性状态的机体在经历空闲等待转移后依然维持在安全状态集内。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell) (w v : CGB.RenewableLearning.Env),
  CGB.GraceRecovery.Good q w → 0 < CGB.GraceRecovery.moves w (fun x => q) v → CGB.GraceRecovery.Safe q v
```

## _private.CGBAxioms.GraceRecovery.0.CGB.GraceRecovery.normal_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L99)

解释：安全机体若在某步未获得摄入，则环境必发生未命中转移并进入探测态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell) (w v : CGB.RenewableLearning.Env),
  CGB.GraceRecovery.Safe q w →
    0 < CGB.GraceRecovery.moves w (fun x => q) v →
      CGB.RenewableLearning.incoming true (CGB.RenewableLearning.sense w ()) q = 0 →
        CGB.GraceRecovery.missed w = true ∧
          (v = CGB.RenewableLearning.Env.probeA ∨ v = CGB.RenewableLearning.Env.probeB)
```

## _private.CGBAxioms.GraceRecovery.0.CGB.GraceRecovery.idle_progress

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L107)

解释：空闲等待步在势能度量上产生严格进展，排除系统发生无限死锁循环。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell) (w v : CGB.RenewableLearning.Env),
  CGB.GraceRecovery.Good q w →
    0 < CGB.GraceRecovery.moves w (fun x => q) v →
      (if CGB.GraceRecovery.missed v = true then 1 else 0) < if CGB.GraceRecovery.lateFeed q w = true then 2 else 1
```

## CGB.GraceRecovery.supported

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L117)

解释：所有受支持的机体事件均精确执行原局部更新，且空闲状态摄入通量严格为零。

源码说明：Supported body events execute exactly the old local update; an idle slot has zero incoming material. This lemma does not assume a favourable world.

形式陈述：

```lean
∀ (cut : Bool) (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.RenewableLearning.Signal) (t : ℕ)
  (z z' : CGB.RenewableLearning.S) (e : Option Unit),
  0 < (CGB.GraceRecovery.process cut sched).step t z e z' →
    0 < CGB.GraceRecovery.body.eventWorldStep CGB.GraceRecovery.world cut sched t z z'.2 ∧
      z'.1 () = CGB.RenewableLearning.next true true (sched t z.2 ()) (z.1 ()) ∧
        CGB.Exchange.intake CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.material sched t z e z' =
          decide (0 < CGB.RenewableLearning.incoming true (sched t z.2 ()) (z.1 ()))
```

## _private.CGBAxioms.GraceRecovery.0.CGB.GraceRecovery.probe_potential

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L157)

解释：在探测环境A或B中，系统的恢复势能函数值严格不超过1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ t : ℕ) (q : CGB.RenewableLearning.Cell) (v : CGB.RenewableLearning.Env),
  v = CGB.RenewableLearning.Env.probeA ∨ v = CGB.RenewableLearning.Env.probeB → CGB.GraceRecovery.potential n₀ t q v ≤ 1
```

## _private.CGBAxioms.GraceRecovery.0.CGB.GraceRecovery.stimulus_before

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L162)

解释：在单步中断时刻之前，施加给机体的刺激信号严格等于环境真实传感器感知信号。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ t : ℕ) (w : CGB.RenewableLearning.Env),
  t < n₀ → CGB.GraceRecovery.interrupted n₀ t w () = CGB.RenewableLearning.sense w ()
```

## _private.CGBAxioms.GraceRecovery.0.CGB.GraceRecovery.stimulus_idle

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L166)

解释：在设定的中断时刻n₀，环境向机体注入的刺激被严格置换为空闲静默信号。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ : ℕ) (w : CGB.RenewableLearning.Env), CGB.GraceRecovery.interrupted n₀ n₀ w () = CGB.RenewableLearning.Signal.calm
```

## _private.CGBAxioms.GraceRecovery.0.CGB.GraceRecovery.stimulus_after

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L169)

解释：在中断时刻n₀之后，输入刺激信号瞬时恢复为环境真实的实时传感器观测。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ t : ℕ) (w : CGB.RenewableLearning.Env),
  n₀ < t → CGB.GraceRecovery.interrupted n₀ t w () = CGB.RenewableLearning.sense w ()
```

## _private.CGBAxioms.GraceRecovery.0.CGB.GraceRecovery.scheduled_closed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L173)

解释：安全区域在时序中断协议驱动下依然保持步步前向不变封闭性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ t : ℕ) (z z' : CGB.RenewableLearning.S) (e : Option Unit),
  CGB.GraceRecovery.region n₀ t z →
    0 < (CGB.GraceRecovery.process false (CGB.GraceRecovery.interrupted n₀)).step t z e z' →
      CGB.GraceRecovery.region n₀ (t + 1) z'
```

## _private.CGBAxioms.GraceRecovery.0.CGB.GraceRecovery.scheduled_progress

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L200)

解释：在指定中断协议、区域条件和正概率一步下，若该步没有摄入，则定义的势函数严格下降。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ t : ℕ) (z z' : CGB.RenewableLearning.S) (e : Option Unit),
  CGB.GraceRecovery.region n₀ t z →
    0 < (CGB.GraceRecovery.process false (CGB.GraceRecovery.interrupted n₀)).step t z e z' →
      CGB.Exchange.intake CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.material
            (CGB.GraceRecovery.interrupted n₀) t z e z' =
          false →
        CGB.GraceRecovery.potential n₀ (t + 1) (z'.1 ()) z'.2 < CGB.GraceRecovery.potential n₀ t (z.1 ()) z.2
```

## _private.CGBAxioms.GraceRecovery.0.CGB.GraceRecovery.positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L240)

解释：位于宽限恢复安全区域内的所有机体构型其所含有效物质总量均严格大于零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ t : ℕ) (z : CGB.RenewableLearning.S), CGB.GraceRecovery.region n₀ t z → 0 < CGB.GraceRecovery.body.total z.1
```

## CGB.GraceRecovery.one_tick_survival

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L251)

解释：在遭受单拍感知中断扰动后，机体在宽限恢复协议下的全时生存概率恒等于1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n₀ n : ℕ),
  CGB.Exchange.survival CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.ec CGB.GraceRecovery.material
      false (CGB.GraceRecovery.interrupted n₀) n =
    1
```

## CGB.GraceRecovery.one_tick_tolerance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L258)

解释：机体具备对单拍环境感知中断的完全容忍耐受性（单拍耐受定理成立）。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Exchange.ToleratesSensoryInterruption CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.ec
  CGB.GraceRecovery.material 0 1
```

## CGB.GraceRecovery.natural_survival

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L291)

解释：在自然无干扰的环境感知流驱动下，机体的全时段综合生存概率恒等于1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n : ℕ),
  CGB.Exchange.survival CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.ec CGB.GraceRecovery.material
      false CGB.GraceRecovery.natural n =
    1
```
