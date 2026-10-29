# MaterialExchange

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean) · [返回导航](../README.md)

模块源码说明：Versioned gross material exchange observer The old net-growth observer remains intact.

## CGB.Medium.ledger_p2a

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L26)

解释：任意带有物质进出账本的介质必满足总物质封闭守恒公理P2a。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] (M : CGB.Medium U Λ A I)
  (L : M.MaterialLedger), M.P2a
```

## CGB.Medium.net_increase_implies_incoming

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L34)

解释：若介质局域事件导致总物质净增加，则该事件必包含来自外部环境的正输入通量。

源码说明：A positive net increase necessarily includes incoming material. The reverse need not hold because incoming and outgoing can be equal and positive.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] (M : CGB.Medium U Λ A I)
  (L : M.MaterialLedger) (a : A) (s : U → I) (x : U → Λ),
  0 < M.rate a s x → M.total x < M.total (M.act a s x) → 0 < L.incoming a s x
```

## CGB.Medium.equal_content_iff_balanced_exchange

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L40)

解释：在正跃迁率事件下，局部物质存量保持不变当且仅当该事件的输入物质通量与输出通量严格平衡。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] (M : CGB.Medium U Λ A I)
  (L : M.MaterialLedger) (a : A) (s : U → I) (x : U → Λ),
  0 < M.rate a s x → (M.total (M.act a s x) = M.total x ↔ L.incoming a s x = L.outgoing a s x)
```

## CGB.Exchange.recovery_bounds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L66)

解释：在任意调度策略与环境交互下，物质恢复度测度严格有界在区间[0, 1]内。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (he : M.EventClocked Wd.clock) (L : M.MaterialLedger) (cut : Bool) (sched : ℕ → Wd.W → U → I) (t k : ℕ)
  (z : (U → Λ) × Wd.W),
  0 ≤ CGB.Exchange.recovery M Wd he L cut sched t k z ∧ CGB.Exchange.recovery M Wd he L cut sched t k z ≤ 1
```

## CGB.Exchange.recovery_zero_without_incoming

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L71)

解释：若外部环境对机体完全关闭物质输入通量，则机体任意时刻的恢复度均恒为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (he : M.EventClocked Wd.clock) (L : M.MaterialLedger),
  (∀ (a : A) (s : U → I) (x : U → Λ), L.incoming a s x = 0) →
    ∀ (cut : Bool) (sched : ℕ → Wd.W → U → I) (t k : ℕ) (z : (U → Λ) × Wd.W),
      CGB.Exchange.recovery M Wd he L cut sched t k z = 0
```

## CGB.Exchange.survival_bounds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L90)

解释：机体在任意截断与时序交互过程下的生存概率度量严格介于0与1之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (he : M.EventClocked Wd.clock) (L : M.MaterialLedger) (cut : Bool) (sched : ℕ → Wd.W → U → I) (n : ℕ),
  0 ≤ CGB.Exchange.survival M Wd he L cut sched n ∧ CGB.Exchange.survival M Wd he L cut sched n ≤ 1
```

## CGB.Exchange.not_persists_without_incoming

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L112)

解释：在缺乏外部物质补给的封闭耗散环境下，机体不可能在任意正阈值δ下实现持久生存。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (he : M.EventClocked Wd.clock) (L : M.MaterialLedger),
  (∀ (a : A) (s : U → I) (x : U → Λ), L.incoming a s x = 0) → ∀ (δ : ℝ), ¬CGB.Exchange.Persists M Wd he L δ
```

## CGB.Description.ExchangeBook.one_percent_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L162)

解释：当且仅当基因组编码比特数不超过几何点数平方的百分之一且位点数非零时，该交换账本满足1%基因组开销预算。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (b : CGB.Description.ExchangeBook) (graph : CGB.Description.WiringReference b.base.sites),
  b.GenomeBudget graph 1 100 ↔
    0 < b.base.sites * b.base.sites ∧ 100 * CGB.Description.bits b ≤ b.base.sites * b.base.sites
```
