# RenewableLearningEnergy

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningEnergy.lean) · [返回导航](../README.md)

模块源码说明：The total fuel account contains both buffer and write credit.

## _private.CGBAxioms.RenewableLearningEnergy.0.CGB.RenewableLearningEnergy.supported

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningEnergy.lean#L16)

解释：可再生学习状态转移的正概率事件必须由底层植物反应网络的对应事件支撑。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (reopen refill cut : Bool) (t : ℕ) (z z' : CGB.RenewableLearning.S) (e : Option Unit),
  0 < (CGB.RenewableLearning.process reopen refill cut).step t z e z' →
    0 <
      (CGB.RenewableLearning.plant reopen refill).eventStep CGB.RenewableLearning.world.clock
        (CGB.RenewableLearning.world.sense z.2) z.1 e z'.1
```

## CGB.RenewableLearningEnergy.allocation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningEnergy.lean#L81)

解释：可再生学习过程的能量分配矩阵行和与列和满足守恒双预算。

源码说明：Allocating credit uses part of the specified food input. Effective writes and destroyed credit use part of total specified consumption.

形式陈述：

```lean
∀ (reopen refill : Bool) (s : CGB.RenewableLearning.Signal) (q : CGB.RenewableLearning.Cell),
  CGB.RenewableLearning.topup refill s q ≤ CGB.RenewableLearning.incoming refill s q ∧
    CGB.RenewableLearning.writeCost reopen s q + CGB.RenewableLearning.discarded s q ≤
      CGB.RenewableLearning.outgoing reopen refill s q
```

## _private.CGBAxioms.RenewableLearningEnergy.0.CGB.RenewableLearningEnergy.telescope

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningEnergy.lean#L86)

解释：能量账户的伸缩求和恒等式：终端期望储能加累计耗能等于初始储量加累计补给。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (P : CGB.Interaction.Machine CGB.RenewableLearning.S ℕ (Option Unit)) (a : CGB.Energetics.Account P)
  (supply spend : ℕ → CGB.RenewableLearning.S → Option Unit → CGB.RenewableLearning.S → ℝ),
  (a.drive = fun x x_1 x_2 x_3 => 0) →
    (a.load = fun x x_1 x_2 x_3 => 0) →
      a.matter = supply →
        (a.heat = fun t z e z' => -spend t z e z') →
          ∀ (n : ℕ),
            CGB.Energetics.mean (CGB.Energetics.law P CGB.RenewableLearning.initial n) (a.stored n) +
                ∑ t ∈ Finset.range n, CGB.Energetics.flux P CGB.RenewableLearning.initial spend t =
              a.stored 0 CGB.RenewableLearning.initial +
                ∑ t ∈ Finset.range n, CGB.Energetics.flux P CGB.RenewableLearning.initial supply t
```

## CGB.RenewableLearningEnergy.fuel_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningEnergy.lean#L107)

解释：可再生学习的燃料消耗与输入通量之差受储存燃料状态的单调下界保护。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (reopen refill cut : Bool) (n : ℕ),
  CGB.Energetics.mean
        (CGB.Energetics.law (CGB.RenewableLearning.process reopen refill cut) CGB.RenewableLearning.initial n)
        ((CGB.RenewableLearningEnergy.fuel_account reopen refill cut).stored n) +
      ∑ t ∈ Finset.range n,
        CGB.Energetics.flux (CGB.RenewableLearning.process reopen refill cut) CGB.RenewableLearning.initial
          (CGB.RenewableLearningEnergy.port (CGB.RenewableLearning.outgoing reopen refill)) t =
    2 +
      ∑ t ∈ Finset.range n,
        CGB.Energetics.flux (CGB.RenewableLearning.process reopen refill cut) CGB.RenewableLearning.initial
          (CGB.RenewableLearningEnergy.port (CGB.RenewableLearning.incoming refill)) t
```

## CGB.RenewableLearningEnergy.credit_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableLearningEnergy.lean#L114)

解释：信用分配通量受可再生学习账户的可用余额与补给上限联合约束。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (reopen refill cut : Bool) (n : ℕ),
  CGB.Energetics.mean
        (CGB.Energetics.law (CGB.RenewableLearning.process reopen refill cut) CGB.RenewableLearning.initial n)
        ((CGB.RenewableLearningEnergy.credit_account reopen refill cut).stored n) +
      ∑ t ∈ Finset.range n,
        CGB.Energetics.flux (CGB.RenewableLearning.process reopen refill cut) CGB.RenewableLearning.initial
          (CGB.RenewableLearningEnergy.port fun s q =>
            CGB.RenewableLearning.writeCost reopen s q + CGB.RenewableLearning.discarded s q)
          t =
    1 +
      ∑ t ∈ Finset.range n,
        CGB.Energetics.flux (CGB.RenewableLearning.process reopen refill cut) CGB.RenewableLearning.initial
          (CGB.RenewableLearningEnergy.port (CGB.RenewableLearning.topup refill)) t
```
