# ScheduledRecovery

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/ScheduledRecovery.lean) · [返回导航](../README.md)

模块源码说明：Time-dependent certificates for a registered recovery protocol Regions and ranks may refer to external protocol time.

## CGB.ScheduledRecovery.ranked_firstHit

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ScheduledRecovery.lean#L28)

解释：在带有时间调度因子的恢复过程中，排位优先的候选状态具有更小的首达时间期望。

源码说明：A local rank certificate proves recovery in the actual future kernel, without assuming a favourable stimulus sequence or the desired recovery score.

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : Fintype E] {K : ℕ → CGB.EventChannels.Kernel S E}
  {mark : ℕ → S → E → S → Bool} {horizon : ℕ} (C : CGB.ScheduledRecovery.Ranked K mark horizon) (t k : ℕ) (z : S),
  C.region t z → C.rank t z < k → CGB.EventChannels.Kernel.firstHit K mark t k z = 1
```

## _private.CGBAxioms.ScheduledRecovery.0.CGB.ScheduledRecovery.support_closed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ScheduledRecovery.lean#L54)

解释：调度恢复过程中的正测度支撑集在有界转移核作用下保持时空前向封闭。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : Fintype E] [inst_2 : DecidableEq S]
  (K : ℕ → CGB.EventChannels.Kernel S E) (initial : S) (region : ℕ → S → Prop),
  region 0 initial →
    (∀ (t : ℕ) (z : S) (e : E) (z' : S), region t z → 0 < (K t).step z e z' → region (t + 1) z') →
      ∀ (n : ℕ) (z : S),
        ¬region n z → CGB.EventChannels.Kernel.stateLaw K (fun z => if z = initial then 1 else 0) n z = 0
```

## CGB.ScheduledRecovery.survival_of_ranked

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ScheduledRecovery.lean#L84)

解释：排位优先的细胞在调度恢复协议下长期存活率拥有确定性的正数下界。

源码说明：Reusable sufficient criterion for the original Exchange score. Physical admission, actual marked resource channels and the original world clock stay at the call site. It proves a target; it does not define life by the certificate.

形式陈述：

```lean
∀ {U : Type u_3} {Λ : Type u_4} {A : Type u_5} {I : Type u_6} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (he : M.EventClocked Wd.clock) (L : M.MaterialLedger) (sched : ℕ → Wd.W → U → I),
  Wd.recovery ≤ 1 →
    ∀
      (C :
        CGB.ScheduledRecovery.Ranked (M.jointEventKernel Wd he false sched) (CGB.Exchange.intake M Wd L sched)
          Wd.horizon),
      C.region 0 (M.init, Wd.start) →
        (∀ (t : ℕ) (z : (U → Λ) × Wd.W), C.region t z → 0 < M.total z.1) →
          ∀ (n : ℕ), CGB.Exchange.survival M Wd he L false sched n = 1
```
