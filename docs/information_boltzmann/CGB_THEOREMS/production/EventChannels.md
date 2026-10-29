# EventChannels

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/EventChannels.lean) · [返回导航](../README.md)

模块源码说明：Event-resolved observations with unchanged body/world marginals State-preserving events can carry gross exchange.

## CGB.FirstHit.Kernel.eq_of_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L17)

解释：若两个首次命中马尔可夫核的单步转移概率算子完全相同，则这两个核对象数学上完全等价。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (K K' : CGB.FirstHit.Kernel S), K.step = K'.step → K = K'
```

## CGB.EventChannels.Kernel.firstHit_bounds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L50)

解释：带标记事件通道核在任意时域与状态下的首次命中概率度量严格介于0和1之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : Fintype E] (K : ℕ → CGB.EventChannels.Kernel S E)
  (mark : ℕ → S → E → S → Bool) (t k : ℕ) (s : S),
  0 ≤ CGB.EventChannels.Kernel.firstHit K mark t k s ∧ CGB.EventChannels.Kernel.firstHit K mark t k s ≤ 1
```

## CGB.EventChannels.Kernel.state_mark_firstHit

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L79)

解释：仅依赖状态的标记在事件通道下的首次命中概率严格等价于遗忘事件标签后的无标签马尔可夫核命中概率。

源码说明：A state-only mark is the specialization of the single channel observer that ignores event labels. This also covers the legacy net-growth question.

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : Fintype E] (K : ℕ → CGB.EventChannels.Kernel S E)
  (mark : S → S → Bool) (t k : ℕ) (s : S),
  CGB.EventChannels.Kernel.firstHit K (fun x x_1 x_2 y => mark x_1 y) t k s =
    CGB.FirstHit.during (fun n => (K n).forget) mark t k s
```

## CGB.EventChannels.Kernel.firstHit_unmarked

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L92)

解释：在恒假判决条件下，系统的首次命中概率在任意时域内严格为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : Fintype E] (K : ℕ → CGB.EventChannels.Kernel S E) (t k : ℕ)
  (s : S), CGB.EventChannels.Kernel.firstHit K (fun x x_1 x_2 x_3 => false) t k s = 0
```

## CGB.EventChannels.Kernel.labelledLaw_marginal

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L110)

解释：对带标签联合状态分布在事件标签维度求和，精确还原为系统的边缘状态概率分布。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : Fintype E] (K : ℕ → CGB.EventChannels.Kernel S E)
  (initial : S → ℝ) (previous : E) (n : ℕ) (s : S),
  ∑ e, CGB.EventChannels.Kernel.labelledLaw K initial previous n s e = CGB.EventChannels.Kernel.stateLaw K initial n s
```

## CGB.Medium.exit_rate_events

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L138)

解释：介质在某状态下的总逃逸率等于所有能引起有效状态改变的离散事件跃迁率之和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (s : U → I) (x : U → Λ),
  ∑ y, (M.jump s).rate x y = ∑ a, if M.act a s x = x then 0 else M.rate a s x
```

## CGB.Medium.eventClocked_implies_clocked

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L153)

解释：介质若满足带事件标签的时钟准入条件，则必然满足标准状态时钟准入条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) {R : ℝ},
  M.EventClocked R → M.Clocked R
```

## CGB.Medium.eventStep_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L168)

解释：在事件时钟准入条件下，单步事件转移概率在所有状态与事件标签下均严格非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq Λ]
  [inst_2 : Fintype A] (M : CGB.Medium U Λ A I) {R : ℝ},
  M.EventClocked R → ∀ (s : U → I) (x : U → Λ) (e : Option A) (y : U → Λ), 0 ≤ M.eventStep R s x e y
```

## CGB.Medium.eventStep_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L183)

解释：单步事件转移概率在所有可能事件标签与后继状态上的联合求和精确归一为1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (R : ℝ) (s : U → I)
  (x : U → Λ), ∑ e, ∑ y, M.eventStep R s x e y = 1
```

## CGB.Medium.eventStep_forget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L191)

解释：在单步事件转移概率中对事件标签求边缘和，精确还原为原始无标签离散转移核概率。

源码说明：Forgetting the event labels recovers the ORIGINAL kernel, even when some positive-rate events leave the body state unchanged.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (R : ℝ) (s : U → I)
  (x y : U → Λ), ∑ e, M.eventStep R s x e y = M.kernel R s x y
```

## CGB.Medium.eventWorldStep_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L229)

解释：机体与世界在事件通道耦合下的环境单步转移矩阵元素恒非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I),
  M.EventClocked Wd.clock →
    ∀ (cut : Bool) (sched : ℕ → Wd.W → U → I) (t : ℕ) (z : (U → Λ) × Wd.W) (w' : Wd.W),
      0 ≤ M.eventWorldStep Wd cut sched t z w'
```

## CGB.Medium.eventWorldStep_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L239)

解释：环境在事件通道驱动下的后继状态转移概率之和精确归一为1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (cut : Bool) (sched : ℕ → Wd.W → U → I) (t : ℕ) (z : (U → Λ) × Wd.W), ∑ w', M.eventWorldStep Wd cut sched t z w' = 1
```

## CGB.Medium.jointEventKernel_forget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L257)

解释：联合事件马尔可夫核在遗忘事件标签后，其单步转移算子精确还原为对应调度的标准或截断状态转移核。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (he : M.EventClocked Wd.clock) (cut : Bool) (sched : ℕ → Wd.W → U → I) (t : ℕ) (z z' : (U → Λ) × Wd.W),
  (M.jointEventKernel Wd he cut sched t).forget.step z z' =
    (if cut = true then M.scheduledCutKernel Wd ⋯ sched t else M.scheduledKernel Wd ⋯ sched t).step z z'
```

## CGB.Medium.event_state_mark_matches_operational

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L267)

解释：仅依赖机体状态的事件通道标记在首次命中概率上与原宏观恢复度量计算完全等价。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (he : M.EventClocked Wd.clock) (cut : Bool) (sched : ℕ → Wd.W → U → I) (t k : ℕ) (z : (U → Λ) × Wd.W),
  CGB.EventChannels.Kernel.firstHit (M.jointEventKernel Wd he cut sched) (fun x x_1 x_2 y => M.materialIntake Wd x_1 y)
      t k z =
    CGB.Operational.recovery M Wd ⋯ cut sched t k z
```

## CGB.Medium.jointEventKernel_stateLaw

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L287)

解释：由联合事件核诱导的边际状态演化分布精确重合于宏观机体-世界系统的标准状态演化律。

源码说明：The entire observed forward process has the original normal/cut state law, not just an equal one-step kernel. The observer is not an extra body state.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (he : M.EventClocked Wd.clock) (cut : Bool) (sched : ℕ → Wd.W → U → I) (n : ℕ) (z : (U → Λ) × Wd.W),
  CGB.EventChannels.Kernel.stateLaw (M.jointEventKernel Wd he cut sched)
      (fun z => if z.1 = M.init ∧ z.2 = Wd.start then 1 else 0) n z =
    if cut = true then M.lawCut Wd sched n z.1 z.2 else M.law Wd sched n z.1 z.2
```
