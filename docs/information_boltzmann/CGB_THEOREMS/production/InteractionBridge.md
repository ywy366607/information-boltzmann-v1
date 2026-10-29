# InteractionBridge

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean) · [返回导航](../README.md)

模块源码说明：The existing medium is a probabilistic interaction coalgebra The observation wrapper preserves the full joint transition and its world tick.

## CGB.Interaction.Machine.ofKernels_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L27)

解释：交互机单步联合转移概率等于与当前观测匹配的分量核转移概率值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {I : Type u_2} {O : Type u_3} [inst : Fintype S] [inst_1 : Fintype O] [inst_2 : DecidableEq O]
  (K : I → CGB.FirstHit.Kernel S) (observe : I → S → S → O) (i : I) (s : S) (o : O) (s' : S),
  (CGB.Interaction.Machine.ofKernels K observe).step i s o s' = if observe i s s' = o then (K i).step s s' else 0
```

## CGB.Interaction.Machine.ofKernels_marginal

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L33)

解释：交互机的输出观测值在全观测空间边际求和后，精确还原为底层状态转移核。

源码说明：Forgetting the observer's label recovers EXACTLY the original transition.

形式陈述：

```lean
∀ {S : Type u_1} {I : Type u_2} {O : Type u_3} [inst : Fintype S] [inst_1 : Fintype O] [inst_2 : DecidableEq O]
  (K : I → CGB.FirstHit.Kernel S) (observe : I → S → S → O) (i : I) (s s' : S),
  ∑ o, (CGB.Interaction.Machine.ofKernels K observe).step i s o s' = (K i).step s s'
```

## CGB.Interaction.Machine.ofKernels_firstHit

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L40)

解释：由马尔可夫核族构成的交互机其标记首达时间与底层分量核的持续时间分布等价。

源码说明：Observe a time protocol in the existing external world clock. A list history supplies the tick number to the test context, not extra internal body storage.

形式陈述：

```lean
∀ {S : Type u_1} {O : Type u_3} [inst : Fintype S] [inst_1 : Fintype O] [inst_2 : DecidableEq O]
  (K : ℕ → CGB.FirstHit.Kernel S) (observe : S → S → O) (mark : O → Bool) (history : List O) (k : ℕ) (s : S),
  (CGB.Interaction.Machine.ofKernels K fun x => observe).firstHit List.length mark history k s =
    CGB.FirstHit.during K (fun z z' => mark (observe z z')) history.length k s
```

## CGB.Medium.intakeCoalgebra_firstHit

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L91)

解释：摄入余代数下观测序列的首达时间分布与底层物理介质的击中时钟完全相容。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock) (cut : Bool) (sched : ℕ → Wd.W → U → I) (history : List Bool) (k : ℕ) (z : (U → Λ) × Wd.W),
  (M.intakeCoalgebra Wd hc cut sched).firstHit List.length id history k z =
    CGB.FirstHit.during (if cut = true then M.scheduledCutKernel Wd hc sched else M.scheduledKernel Wd hc sched)
      (M.materialIntake Wd) history.length k z
```

## CGB.Medium.intakeCoalgebra_actualRecover

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L101)

解释：摄入余代数结构确保了系统在具有正摄入流时能实现实际状态恢复。

源码说明：This normal-world observation yields the existing actualRecover, not a different coalgebraic notion of success or a favourable-stimulus menu.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock) (history : List Bool) (k : ℕ) (x : U → Λ) (w : Wd.W),
  (M.intakeCoalgebra Wd hc false fun x => Wd.sense).firstHit List.length id history k (x, w) =
    M.actualRecover Wd hc k x w
```

## CGB.Medium.refinement_preserves_recovery_and_time

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L112)

解释：系统精细化映射保结构定理：粗粒化到精细介质的状态提升严格保持恢复概率与特征耗时。

源码说明：Actual recovery AND elapsed time transfer along a certified implementation. This is conditional on a universal refinement proof, not inferred from samples.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock) {V : Type u_5} {Θ : Type u_6} {B : Type u_7} {J : Type u_8} [inst_5 : Fintype V]
  [inst_6 : DecidableEq V] [inst_7 : Fintype Θ] [inst_8 : DecidableEq Θ] [inst_9 : Fintype B] (N : CGB.Medium V Θ B J)
  (Vd : CGB.World V Θ J) (hcN : N.Clocked Vd.clock) (f : (U → Λ) × Wd.W → (V → Θ) × Vd.W),
  CGB.Interaction.TimedRefinement (M.intakeCoalgebra Wd hc false fun x => Wd.sense)
      (N.intakeCoalgebra Vd hcN false fun x => Vd.sense) f Wd.clock Vd.clock →
    ∀ (n k : ℕ) (z : (U → Λ) × Wd.W),
      M.actualRecover Wd hc k z.1 z.2 = N.actualRecover Vd hcN k (f z).1 (f z).2 ∧ ↑n / Wd.clock = ↑n / Vd.clock
```
