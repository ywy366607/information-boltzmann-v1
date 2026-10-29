# RenewableLearningClosure

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean) · [返回导航](../README.md)

模块源码说明：Close the ORIGINAL distribution-dependent marginal-ghost comparison.

## _private.CGBAxioms.RenewableLearningClosure.0.CGB.RenewableLearningClosure.policy_table

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L16)

解释：给出可再生学习闭包中细胞状态到策略决策动作的显式映射表。

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

## _private.CGBAxioms.RenewableLearningClosure.0.CGB.RenewableLearningClosure.sum_cells

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L22)

解释：可再生学习机体细胞构型全空间的实值函数求和展开等价性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (f : CGB.RenewableLearning.Cell → ℝ),
  ∑ q, f q =
    f CGB.RenewableLearning.Cell.vacuum + f CGB.RenewableLearning.Cell.search + f CGB.RenewableLearning.Cell.aSpent +
              f CGB.RenewableLearning.Cell.bSpent +
            f CGB.RenewableLearning.Cell.aReady +
          f CGB.RenewableLearning.Cell.bReady +
        f CGB.RenewableLearning.Cell.aHeld +
      f CGB.RenewableLearning.Cell.bHeld
```

## _private.CGBAxioms.RenewableLearningClosure.0.CGB.RenewableLearningClosure.sum_configs

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L29)

解释：单单元构型空间全状态求和的完整展开式。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (f : (Unit → CGB.RenewableLearning.Cell) → ℝ),
  ∑ x, f x =
    (((((((f fun x => CGB.RenewableLearning.Cell.vacuum) + f fun x => CGB.RenewableLearning.Cell.search) +
                f fun x => CGB.RenewableLearning.Cell.aSpent) +
              f fun x => CGB.RenewableLearning.Cell.bSpent) +
            f fun x => CGB.RenewableLearning.Cell.aReady) +
          f fun x => CGB.RenewableLearning.Cell.bReady) +
        f fun x => CGB.RenewableLearning.Cell.aHeld) +
      f fun x => CGB.RenewableLearning.Cell.bHeld
```

## _private.CGBAxioms.RenewableLearningClosure.0.CGB.RenewableLearningClosure.sum_env

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L37)

解释：九状态环境空间中全状态实值函数的逐项求和展开式。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (f : CGB.RenewableLearning.Env → ℝ),
  ∑ w, f w =
    f CGB.RenewableLearning.Env.feedA + f CGB.RenewableLearning.Env.feedB + f CGB.RenewableLearning.Env.missA +
                f CGB.RenewableLearning.Env.missB +
              f CGB.RenewableLearning.Env.probeA +
            f CGB.RenewableLearning.Env.probeB +
          f CGB.RenewableLearning.Env.answerA +
        f CGB.RenewableLearning.Env.answerB +
      f CGB.RenewableLearning.Env.trap
```

## _private.CGBAxioms.RenewableLearningClosure.0.CGB.RenewableLearningClosure.kernel_table

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L46)

解释：给出可再生学习机体转移核在后继神经状态处的离散Dirac分布结构。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : Unit → CGB.RenewableLearning.Signal) (q r : CGB.RenewableLearning.Cell),
  (CGB.RenewableLearningClosure.body.kernel 1 s (fun x => q) fun x => r) =
    if r = CGB.RenewableLearning.next true true (s ()) q then 1 else 0
```

## _private.CGBAxioms.RenewableLearningClosure.0.CGB.RenewableLearningClosure.early_move

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L70)

解释：在系统演化前4步内，截断演化下的环境状态跃迁与标准动力学完全重合。

源码说明：Until tick four the body marginal is a point mass, so the ORIGINAL cut transition is derived from that marginal, not assigned a convenient law.

形式陈述：

```lean
∀ n ≤ 4,
  (∀ (q : CGB.RenewableLearning.Cell) (w : CGB.RenewableLearning.world.W),
      CGB.RenewableLearningClosure.body.lawCut CGB.RenewableLearning.world (fun x => CGB.RenewableLearning.world.sense)
          n (fun x => q) w =
        CGB.RenewableLearningClosure.early✝ n q w) →
    ∀ (x : Unit → CGB.RenewableLearning.Cell) (w v : CGB.RenewableLearning.Env),
      CGB.Medium.cutMove CGB.RenewableLearning.world
          (CGB.RenewableLearningClosure.body.lawCut CGB.RenewableLearning.world
            (fun x => CGB.RenewableLearning.world.sense) n)
          x w v =
        CGB.RenewableLearning.moves w (fun x => CGB.RenewableLearningClosure.ghost✝ n) v
```

## _private.CGBAxioms.RenewableLearningClosure.0.CGB.RenewableLearningClosure.cut_early

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L83)

解释：可再生学习闭包系统在演化前5步内截断演化律与自然演化律严格相同。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ n ≤ 5,
  ∀ (q : CGB.RenewableLearning.Cell) (w : CGB.RenewableLearning.Env),
    CGB.RenewableLearningClosure.body.lawCut CGB.RenewableLearning.world (fun x => CGB.RenewableLearning.world.sense) n
        (fun x => q) w =
      CGB.RenewableLearningClosure.early✝ n q w
```

## _private.CGBAxioms.RenewableLearningClosure.0.CGB.RenewableLearningClosure.move_five

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L100)

解释：可再生学习系统在第5步的环境转移算子在截断下保持不变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (x : Unit → CGB.RenewableLearning.Cell) (w v : CGB.RenewableLearning.Env),
  CGB.Medium.cutMove CGB.RenewableLearning.world
      (CGB.RenewableLearningClosure.body.lawCut CGB.RenewableLearning.world (fun x => CGB.RenewableLearning.world.sense)
        5)
      x w v =
    1 / 2 * CGB.RenewableLearning.moves w (fun x => CGB.RenewableLearning.Cell.search) v +
      1 / 2 * CGB.RenewableLearning.moves w (fun x => CGB.RenewableLearning.Cell.aReady) v
```

## _private.CGBAxioms.RenewableLearningClosure.0.CGB.RenewableLearningClosure.trapped_next

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L113)

解释：处于陷阱环境状态的系统在一步演化后必然保持在陷阱集合内。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n : ℕ),
  CGB.RenewableLearningClosure.trapped✝ (n + 1) =
    ∑ x,
      ∑ w,
        CGB.RenewableLearningClosure.body.lawCut CGB.RenewableLearning.world
            (fun x => CGB.RenewableLearning.world.sense) n x w *
          CGB.Medium.cutMove CGB.RenewableLearning.world
            (CGB.RenewableLearningClosure.body.lawCut CGB.RenewableLearning.world
              (fun x => CGB.RenewableLearning.world.sense) n)
            x w CGB.RenewableLearning.Env.trap
```

## _private.CGBAxioms.RenewableLearningClosure.0.CGB.RenewableLearningClosure.trapped_six

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L124)

解释：系统在截断演化第6步落入陷阱吸收态的概率严格等于1/4。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.RenewableLearningClosure.trapped✝ 6 = 1 / 4
```

## _private.CGBAxioms.RenewableLearningClosure.0.CGB.RenewableLearningClosure.trap_moves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L130)

解释：一旦环境落入陷阱状态，在任意机体动作下环境均以概率1保持在陷阱态（吸收态）。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (x : Unit → CGB.RenewableLearning.Cell) (v : CGB.RenewableLearning.Env),
  CGB.RenewableLearning.moves CGB.RenewableLearning.Env.trap x v = if v = CGB.RenewableLearning.Env.trap then 1 else 0
```

## _private.CGBAxioms.RenewableLearningClosure.0.CGB.RenewableLearningClosure.trap_world_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L134)

解释：陷阱环境在任意机体动作驱动下以概率1转移回陷阱自身。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (t : ℕ) (x : Unit → CGB.RenewableLearning.Cell) (v : CGB.RenewableLearning.Env),
  CGB.RenewableLearningClosure.body.eventWorldStep CGB.RenewableLearning.world cut
      (fun x => CGB.RenewableLearning.world.sense) t (x, CGB.RenewableLearning.Env.trap) v =
    if v = CGB.RenewableLearning.Env.trap then 1 else 0
```

## _private.CGBAxioms.RenewableLearningClosure.0.CGB.RenewableLearningClosure.trapped_monotone

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L148)

解释：截断动力学下陷入陷阱状态的累积概率质量随时间推进单调不减。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n : ℕ), CGB.RenewableLearningClosure.trapped✝ n ≤ CGB.RenewableLearningClosure.trapped✝ (n + 1)
```

## _private.CGBAxioms.RenewableLearningClosure.0.CGB.RenewableLearningClosure.trap_recovery

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L163)

解释：一旦系统陷入陷阱状态，其未来任意有限时域内的恢复度量恒为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t k : ℕ) (x : Unit → CGB.RenewableLearning.Cell),
  CGB.Exchange.recovery CGB.RenewableLearningClosure.body CGB.RenewableLearning.world CGB.RenewableLearningClosure.ec
      CGB.RenewableLearningClosure.material true (fun x => CGB.RenewableLearning.world.sense) t k
      (x, CGB.RenewableLearning.Env.trap) =
    0
```

## _private.CGBAxioms.RenewableLearningClosure.0.CGB.RenewableLearningClosure.cut_alive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L180)

解释：在截断演化中，系统存活指示函数在陷入陷阱态时直接归零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n : ℕ) (x : Unit → CGB.RenewableLearning.Cell) (w : CGB.RenewableLearning.Env),
  CGB.Exchange.alive CGB.RenewableLearningClosure.body CGB.RenewableLearning.world CGB.RenewableLearningClosure.ec
      CGB.RenewableLearningClosure.material true (fun x => CGB.RenewableLearning.world.sense) n (x, w) ≤
    if w = CGB.RenewableLearning.Env.trap then 0 else 1
```

## _private.CGBAxioms.RenewableLearningClosure.0.CGB.RenewableLearningClosure.cut_mass_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L193)

解释：导出系统截断存活概率由1减去陷入陷阱死态的概率质量所构成的严格上界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n : ℕ),
  CGB.Exchange.survival CGB.RenewableLearningClosure.body CGB.RenewableLearning.world CGB.RenewableLearningClosure.ec
      CGB.RenewableLearningClosure.material true (fun x => CGB.RenewableLearning.world.sense) n ≤
    1 - CGB.RenewableLearningClosure.trapped✝ n
```

## CGB.RenewableLearningClosure.cut_survival

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L216)

解释：在截断演化第6步起，系统存活概率至少损失1/4并持续保持在3/4以下（闭环维持必要性）。

源码说明：The original cut loses at least a quarter of viable mass by tick six; absorption extends the bound to ALL later times, including future cut marginals.

形式陈述：

```lean
∀ (n : ℕ),
  6 ≤ n →
    CGB.Exchange.survival CGB.RenewableLearningClosure.body CGB.RenewableLearning.world CGB.RenewableLearningClosure.ec
        CGB.RenewableLearningClosure.material true (fun x => CGB.RenewableLearning.world.sense) n ≤
      3 / 4
```

## CGB.RenewableLearningClosure.maintenance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L225)

解释：可再生学习机体完全满足持续生存与主动自我维持公理。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Exchange.Persists CGB.RenewableLearningClosure.body CGB.RenewableLearning.world CGB.RenewableLearningClosure.ec
    CGB.RenewableLearningClosure.material 0 ∧
  CGB.Exchange.SelfMaintains CGB.RenewableLearningClosure.body CGB.RenewableLearning.world
    CGB.RenewableLearningClosure.ec CGB.RenewableLearningClosure.material
```

## CGB.RenewableLearningClosure.cut_seed_mass

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L233)

解释：精确计算出在截断演化第6步系统落入吸收陷阱死态的初始种子概率质量严格等于1/4。

源码说明：Exact seed mass for the absorbing failure region under the original cut.

形式陈述：

```lean
∑ x,
    CGB.RenewableLearningClosure.body.lawCut CGB.RenewableLearning.world (fun x => CGB.RenewableLearning.world.sense) 6
      x CGB.RenewableLearning.Env.trap =
  1 / 4
```
