# RecoveryCriteria

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/RecoveryCriteria.lean) · [返回导航](../README.md)

模块源码说明：Local sufficient criteria for actual resource recovery Regions and ranks are proof objects, not new organism state, policy inputs or life axioms.

## CGB.RecoveryCriteria.ranked_firstHit

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RecoveryCriteria.lean#L28)

解释：依据局部排位序给出的首达恢复时间具有单调随机占优性质。

源码说明：A local rank certificate proves recovery in the actual future kernel, without assuming a favourable stimulus sequence or the desired recovery score.

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : Fintype E] {K : ℕ → CGB.EventChannels.Kernel S E}
  {mark : ℕ → S → E → S → Bool} {horizon : ℕ} (C : CGB.RecoveryCriteria.Ranked K mark horizon) (t k : ℕ) (z : S),
  C.region z → C.rank z < k → CGB.EventChannels.Kernel.firstHit K mark t k z = 1
```

## CGB.RecoveryCriteria.firstHit_zero_of_closed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RecoveryCriteria.lean#L54)

解释：在无代谢摄入的封闭系统中，从任意状态出发首次命中有效恢复区的概率恒为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : Fintype E] (K : ℕ → CGB.EventChannels.Kernel S E)
  (mark : ℕ → S → E → S → Bool) (region : S → Prop),
  (∀ (t : ℕ) (z : S) (e : E) (z' : S), region z → 0 < (K t).step z e z' → region z') →
    (∀ (t : ℕ) (z : S) (e : E) (z' : S), region z → 0 < (K t).step z e z' → mark t z e z' = false) →
      ∀ (t k : ℕ) (z : S), region z → CGB.EventChannels.Kernel.firstHit K mark t k z = 0
```

## _private.CGBAxioms.RecoveryCriteria.0.CGB.RecoveryCriteria.support_closed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RecoveryCriteria.lean#L74)

解释：从初始区域出发的正概率转移在步进封闭条件下始终保持在初始区域内。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : Fintype E] [inst_2 : DecidableEq S]
  (K : ℕ → CGB.EventChannels.Kernel S E) (initial : S) (region : S → Prop),
  region initial →
    (∀ (t : ℕ) (z : S) (e : E) (z' : S), region z → 0 < (K t).step z e z' → region z') →
      ∀ (n : ℕ) (z : S), ¬region z → CGB.EventChannels.Kernel.stateLaw K (fun z => if z = initial then 1 else 0) n z = 0
```

## CGB.RecoveryCriteria.survival_of_ranked

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RecoveryCriteria.lean#L104)

解释：具有良好排位的候选态在长期演化中存活概率有正下界。

源码说明：Reusable sufficient criterion for the original Exchange score. Physical admission, actual marked resource channels and the original world clock stay at the call site. It proves a target; it does not define life by the certificate.

形式陈述：

```lean
∀ {U : Type u_3} {Λ : Type u_4} {A : Type u_5} {I : Type u_6} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (he : M.EventClocked Wd.clock) (L : M.MaterialLedger),
  Wd.recovery ≤ 1 →
    ∀
      (C :
        CGB.RecoveryCriteria.Ranked (M.jointEventKernel Wd he false fun x => Wd.sense)
          (CGB.Exchange.intake M Wd L fun x => Wd.sense) Wd.horizon),
      C.region (M.init, Wd.start) →
        (∀ (z : (U → Λ) × Wd.W), C.region z → 0 < M.total z.1) →
          ∀ (n : ℕ), CGB.Exchange.survival M Wd he L false (fun x => Wd.sense) n = 1
```
