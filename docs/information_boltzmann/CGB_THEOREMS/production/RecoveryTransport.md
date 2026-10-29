# RecoveryTransport

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/RecoveryTransport.lean) · [返回导航](../README.md)

模块源码说明：Recovery and state-score transport through existing exact refinements State-dependent intake must match on supported transitions, in addition to the joint event/state law and body content.

## _private.CGBAxioms.RecoveryTransport.0.CGB.RecoveryTransport.mean_next

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RecoveryTransport.lean#L15)

解释：恢复传输单步迭代后的后继均值满足递推鞅关系。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {T : Type u_1} {E : Type u_2} [inst : Fintype T] [inst_1 : DecidableEq T] [inst_2 : Fintype E]
  (M : CGB.Interaction.Machine T ℕ E) (z : T) (n : ℕ) (v : T → ℝ),
  CGB.Energetics.mean (CGB.Energetics.law M z (n + 1)) v =
    CGB.Energetics.mean (CGB.Energetics.law M z n) fun x => ∑ e, ∑ y, M.step n x e y * v y
```

## CGB.RecoveryTransport.firstHit_eq

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RecoveryTransport.lean#L32)

解释：恢复传输算子作用下，两个分布的首次命中时间概率分布完全相等。

源码说明：Supported actual-intake correspondence preserves every recovery horizon, including in a distribution-dependent cut supplied as a separate kernel.

形式陈述：

```lean
∀ {R : Type u_1} {S : Type u_2} {E : Type u_3} [inst : Fintype R] [DecidableEq R] [inst_2 : Fintype S]
  [inst_3 : DecidableEq S] [inst_4 : Fintype E] {P : CGB.Interaction.Machine R ℕ E} {Q : CGB.Interaction.Machine S ℕ E}
  {f : R → S},
  CGB.Interaction.Refines P Q f →
    ∀ (a : ℕ → R → E → R → Bool) (b : ℕ → S → E → S → Bool),
      (∀ (t : ℕ) (r : R) (e : E) (r' : R), 0 < P.step t r e r' → a t r e r' = b t (f r) e (f r')) →
        ∀ (t k : ℕ) (r : R),
          CGB.EventChannels.Kernel.firstHit (CGB.Energetics.events P) a t k r =
            CGB.EventChannels.Kernel.firstHit (CGB.Energetics.events Q) b t k (f r)
```

## CGB.RecoveryTransport.mean_eq

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RecoveryTransport.lean#L61)

解释：在保测度传输下，任意可观测量的均值在步进前后保持守恒。

源码说明：Exact state-score expectations at every time, without assuming that the score is a probability or replacing physical state by output histories.

形式陈述：

```lean
∀ {R : Type u_1} {S : Type u_2} {E : Type u_3} [inst : Fintype R] [inst_1 : DecidableEq R] [inst_2 : Fintype S]
  [inst_3 : DecidableEq S] [inst_4 : Fintype E] {P : CGB.Interaction.Machine R ℕ E} {Q : CGB.Interaction.Machine S ℕ E}
  {f : R → S},
  CGB.Interaction.Refines P Q f →
    ∀ (initial : R) (n : ℕ) (g : S → ℝ),
      (CGB.Energetics.mean (CGB.Energetics.law P initial n) fun r => g (f r)) =
        CGB.Energetics.mean (CGB.Energetics.law Q (f initial) n) g
```
