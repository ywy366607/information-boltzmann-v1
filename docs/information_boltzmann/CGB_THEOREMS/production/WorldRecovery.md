# WorldRecovery

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean) · [返回导航](../README.md)

模块源码说明：First-hit probabilities in the actual body--world chain No best-stimulus oracle is used.

## CGB.FirstHit.bounds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L31)

解释：有限随机核上，在任意有限期限内首次发生指定事件的概率介于零和一之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (K : CGB.FirstHit.Kernel S) (mark : S → S → Bool) (k : ℕ) (z : S),
  0 ≤ CGB.FirstHit.probability K mark k z ∧ CGB.FirstHit.probability K mark k z ≤ 1
```

## CGB.FirstHit.monotone_horizon

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L44)

解释：在同一随机核下，把首次命中的观察期限增加一步不会降低命中概率。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (K : CGB.FirstHit.Kernel S) (mark : S → S → Bool) (k : ℕ) (z : S),
  CGB.FirstHit.probability K mark k z ≤ CGB.FirstHit.probability K mark (k + 1) z
```

## CGB.FirstHit.flagged_true

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L62)

解释：已经发生标记事件的观察标志为真时，任意后续期限上的标志评分恒为一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (K : CGB.FirstHit.Kernel S) (mark : S → S → Bool) (k : ℕ) (z : S),
  CGB.FirstHit.flagged K mark k z true = 1
```

## CGB.FirstHit.flagged_false

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L68)

解释：初始标志为假时，带吸收标志的观察器与首次命中概率递推给出相同结果。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (K : CGB.FirstHit.Kernel S) (mark : S → S → Bool) (k : ℕ) (z : S),
  CGB.FirstHit.flagged K mark k z false = CGB.FirstHit.probability K mark k z
```

## CGB.FirstHit.zero_on_closed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L82)

解释：若一个区域对正概率转移封闭且内部不存在被标记的正概率转移，则从该区域出发任意有限期限的命中概率为零。

源码说明：A closed region with no marked positive-probability transition has zero success probability at every finite horizon, even if success exists elsewhere.

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (K : CGB.FirstHit.Kernel S) (mark : S → S → Bool) (C : S → Prop),
  (∀ (z z' : S), C z → 0 < K.step z z' → C z') →
    (∀ (z z' : S), C z → 0 < K.step z z' → mark z z' = false) →
      ∀ (k : ℕ) (z : S), C z → CGB.FirstHit.probability K mark k z = 0
```

## CGB.FirstHit.impossible_absorbing

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L101)

解释：若从当前状态沿实际核不可能发生标记事件，则任何一步正概率可达的后继状态也不可能发生该事件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (K : CGB.FirstHit.Kernel S) (mark : S → S → Bool) {z z' : S},
  ¬CGB.FirstHit.Possible K mark z → 0 < K.step z z' → ¬CGB.FirstHit.Possible K mark z'
```

## CGB.FirstHit.zero_of_impossible

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L106)

解释：如果实际随机核中不存在通向标记事件的正概率路径，则每个有限期限的首次命中概率都为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (K : CGB.FirstHit.Kernel S) (mark : S → S → Bool) {z : S},
  ¬CGB.FirstHit.Possible K mark z → ∀ (k : ℕ), CGB.FirstHit.probability K mark k z = 0
```

## CGB.FirstHit.during_bounds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L129)

解释：对于随时间变化的随机核，任意起始时刻和有限期限的首次命中概率都在零和一之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (K : ℕ → CGB.FirstHit.Kernel S) (mark : S → S → Bool) (t k : ℕ) (z : S),
  0 ≤ CGB.FirstHit.during K mark t k z ∧ CGB.FirstHit.during K mark t k z ≤ 1
```

## CGB.FirstHit.during_constant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L143)

解释：当时变随机核每个时刻都等于同一个核时，时变首次命中概率等于普通首次命中概率。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (K : CGB.FirstHit.Kernel S) (mark : S → S → Bool) (t k : ℕ) (z : S),
  CGB.FirstHit.during (fun x => K) mark t k z = CGB.FirstHit.probability K mark k z
```

## CGB.Medium.jointStep_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L162)

解释：身体与世界按定义联合更新时，对所有后继联合状态求转移权重之和等于一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (sched : ℕ → Wd.W → U → I) (t : ℕ) (z : (U → Λ) × Wd.W), ∑ z', M.jointStep Wd sched t z z' = 1
```

## CGB.Medium.law_uses_jointStep

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L168)

解释：联合身体—世界分布的下一拍递推可直接写成当前分布与联合转移核的有限求和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (sched : ℕ → Wd.W → U → I) (n : ℕ) (y : U → Λ) (w' : Wd.W),
  M.law Wd sched (n + 1) y w' = ∑ z, M.law Wd sched n z.1 z.2 * M.jointStep Wd sched n z (y, w')
```

## CGB.Medium.lawCut_uses_scheduledCutKernel

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L202)

解释：断环对照的下一拍分布由断环后的时变转移核递推，因此评估不会暗中把未来闭环接回去。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock) (sched : ℕ → Wd.W → U → I) (n : ℕ) (y : U → Λ) (w' : Wd.W),
  M.lawCut Wd sched (n + 1) y w' =
    ∑ z, M.lawCut Wd sched n z.1 z.2 * (M.scheduledCutKernel Wd hc sched n).step z (y, w')
```

## CGB.Medium.actualRecover_bounds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L216)

解释：在满足时钟条件时，真实身体—世界链中有限期限内摄入材料的概率介于零和一之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock) (k : ℕ) (x : U → Λ) (w : Wd.W),
  0 ≤ M.actualRecover Wd hc k x w ∧ M.actualRecover Wd hc k x w ≤ 1
```

## CGB.Medium.actualRecover_mono

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L220)

解释：在真实身体—世界链中，增加一次允许的恢复时间不会降低材料摄入概率。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock) (k : ℕ) (x : U → Λ) (w : Wd.W),
  M.actualRecover Wd hc k x w ≤ M.actualRecover Wd hc (k + 1) x w
```

## CGB.Medium.actualRecover_zero_without_intake

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L226)

解释：若所有身体状态都不可能摄入内容，则真实世界协议下任意有限期限的材料恢复概率为零。

源码说明：No possible material intake implies zero actual recovery, without assuming that a favourable stimulus can be summoned from some other world state.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock),
  (∀ (x : U → Λ), ¬M.CanTakeIn x) → ∀ (k : ℕ) (x : U → Λ) (w : Wd.W), M.actualRecover Wd hc k x w = 0
```
