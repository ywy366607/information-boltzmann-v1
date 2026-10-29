# SensoryMaintenance

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean) · [返回导航](../README.md)

模块源码说明：Sensory loss does not suspend physical maintenance `some calm` remains the equilibrium stimulus.

## CGB.SensoryMaintenance.local_budgets

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L60)

解释：指定单步更新对任意刺激和状态同时满足内容输入输出账本与信用写入、损失和补充账本。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : CGB.SensoryMaintenance.Stim) (x : CGB.SensoryMaintenance.State),
  CGB.SensoryMaintenance.cost (CGB.SensoryMaintenance.next s x) + CGB.SensoryMaintenance.outgoing s x =
      CGB.SensoryMaintenance.cost x + CGB.SensoryMaintenance.incoming s x ∧
    CGB.RenewableLearning.credit (CGB.SensoryMaintenance.next s x).1 + CGB.SensoryMaintenance.writes s x +
        CGB.SensoryMaintenance.lostCredit s x =
      CGB.RenewableLearning.credit x.1 + CGB.SensoryMaintenance.topups s x
```

## CGB.SensoryMaintenance.transfer_bounds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L68)

解释：在指定模型中，内容成本与输入量均不超过六，输出量不超过当前内容成本。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : CGB.SensoryMaintenance.Stim) (x : CGB.SensoryMaintenance.State),
  CGB.SensoryMaintenance.cost x ≤ 6 ∧
    CGB.SensoryMaintenance.incoming s x ≤ 6 ∧ CGB.SensoryMaintenance.outgoing s x ≤ CGB.SensoryMaintenance.cost x
```

## CGB.SensoryMaintenance.active_loss

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L75)

解释：在指定感知维护模型中，刺激不是 calm 且当前内容成本为正时，对应输出量严格为正。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : CGB.SensoryMaintenance.Stim) (x : CGB.SensoryMaintenance.State),
  s ≠ some CGB.RenewableLearning.Signal.calm →
    0 < CGB.SensoryMaintenance.cost x → 0 < CGB.SensoryMaintenance.outgoing s x
```

## CGB.SensoryMaintenance.missing_no_intake

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L84)

解释：在指定模型中，刺激为 none 时登记的内容输入量为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (x : CGB.SensoryMaintenance.State), CGB.SensoryMaintenance.incoming none x = 0
```

## CGB.SensoryMaintenance.next_empty

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L87)

解释：在指定模型中，empty 状态对任意刺激的 next 更新保持为 empty。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : CGB.SensoryMaintenance.Stim),
  CGB.SensoryMaintenance.next s CGB.SensoryMaintenance.empty = CGB.SensoryMaintenance.empty
```

## CGB.SensoryMaintenance.admission

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L102)

解释：指定感知维护身体满足 Axioms 和速率一的逐事件时钟条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.SensoryMaintenance.body.Axioms ∧ CGB.SensoryMaintenance.body.EventClocked 1
```

## CGB.SensoryMaintenance.equilibrium_distinct_from_loss

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L154)

解释：该身体的 calm 刺激与该世界的 quiet 刺激在定义上不同。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.SensoryMaintenance.body.calm () ≠ CGB.SensoryMaintenance.world.quiet ()
```

## CGB.SensoryMaintenance.actual_transition

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L161)

解释：指定联合过程的正概率一步中，后继身体单元等于对当前单元和该拍刺激应用 next。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.SensoryMaintenance.Stim) (t : ℕ)
  (z z' : CGB.SensoryMaintenance.S) (e : Option Unit),
  0 < (CGB.SensoryMaintenance.process cut sched).step t z e z' →
    z'.1 () = CGB.SensoryMaintenance.next (sched t z.2 ()) (z.1 ())
```

## CGB.SensoryMaintenance.actual_law

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L183)

解释：指定联合过程在闭环或断环配置下的状态分布，分别等于身体—世界的 law 或 lawCut。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (cut : Bool) (sched : ℕ → CGB.RenewableLearning.Env → Unit → CGB.SensoryMaintenance.Stim) (n : ℕ)
  (z : CGB.SensoryMaintenance.S),
  CGB.Energetics.law (CGB.SensoryMaintenance.process cut sched) CGB.SensoryMaintenance.initial n z =
    if cut = true then CGB.SensoryMaintenance.body.lawCut CGB.SensoryMaintenance.world sched n z.1 z.2
    else CGB.SensoryMaintenance.body.law CGB.SensoryMaintenance.world sched n z.1 z.2
```

## CGB.SensoryMaintenance.kernel_table

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L194)

解释：指定身体的速率一转移核在后继等于 next 时为一，否则为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : Unit → CGB.SensoryMaintenance.Stim) (x y : CGB.SensoryMaintenance.State),
  (CGB.SensoryMaintenance.body.kernel 1 s (fun x_1 => x) fun x => y) =
    if y = CGB.SensoryMaintenance.next (s ()) x then 1 else 0
```

## CGB.SensoryMaintenance.burn_empty

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L210)

解释：按指定 burn 递推，empty 状态在任意步数后仍为 empty。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n : ℕ), CGB.SensoryMaintenance.burn CGB.SensoryMaintenance.empty n = CGB.SensoryMaintenance.empty
```

## CGB.SensoryMaintenance.burn_three

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L218)

解释：按指定 burn 递推，任意初态经过三拍后都等于 empty。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (x : CGB.SensoryMaintenance.State), CGB.SensoryMaintenance.burn x 3 = CGB.SensoryMaintenance.empty
```

## CGB.SensoryMaintenance.burn_all_later

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L222)

解释：按指定 burn 递推，任意初态经过至少三拍后都等于 empty。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (x : CGB.SensoryMaintenance.State) (n : ℕ), 3 ≤ n → CGB.SensoryMaintenance.burn x n = CGB.SensoryMaintenance.empty
```

## CGB.SensoryMaintenance.funded_two_tick_bridge

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L228)

解释：在该模型中，非 vacuum 单元从储备二出发，burn 一、二、三拍后分别为储备一、储备零和 empty。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell),
  q ≠ CGB.RenewableLearning.Cell.vacuum →
    CGB.SensoryMaintenance.burn (q, 2) 1 = (q, 1) ∧
      CGB.SensoryMaintenance.burn (q, 2) 2 = (q, 0) ∧
        CGB.SensoryMaintenance.burn (q, 2) 3 = CGB.SensoryMaintenance.empty
```

## CGB.SensoryMaintenance.path_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L234)

解释：沿满足 next 递推的任意有限轨迹，末态内容成本加累计输出等于初态成本加累计输入。

源码说明：Exact actual-channel fuel balance for every registered input path. Physical time does not stop during absent sensing.

形式陈述：

```lean
∀ (stim : ℕ → CGB.SensoryMaintenance.Stim) (p : ℕ → CGB.SensoryMaintenance.State),
  (∀ (t : ℕ), p (t + 1) = CGB.SensoryMaintenance.next (stim t) (p t)) →
    ∀ (n : ℕ),
      CGB.SensoryMaintenance.cost (p n) + ∑ t ∈ Finset.range n, CGB.SensoryMaintenance.outgoing (stim t) (p t) =
        CGB.SensoryMaintenance.cost (p 0) + ∑ t ∈ Finset.range n, CGB.SensoryMaintenance.incoming (stim t) (p t)
```

## CGB.SensoryMaintenance.path_credit_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L246)

解释：沿任意感知轨迹演化，机体信用余额加上累积写入与损失之和精确等于初始信用加上累积充能量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (stim : ℕ → CGB.SensoryMaintenance.Stim) (p : ℕ → CGB.SensoryMaintenance.State),
  (∀ (t : ℕ), p (t + 1) = CGB.SensoryMaintenance.next (stim t) (p t)) →
    ∀ (n : ℕ),
      CGB.RenewableLearning.credit (p n).1 +
          ∑ t ∈ Finset.range n,
            (CGB.SensoryMaintenance.writes (stim t) (p t) + CGB.SensoryMaintenance.lostCredit (stim t) (p t)) =
        CGB.RenewableLearning.credit (p 0).1 + ∑ t ∈ Finset.range n, CGB.SensoryMaintenance.topups (stim t) (p t)
```
