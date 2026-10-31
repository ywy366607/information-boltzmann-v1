# GenerationFutureValue

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationFutureValue.lean) · [返回导航](../README.md)

模块源码说明：Future-paid language distinctions A finite body cannot retain every distinction that *might* be queried.

## CGB.GDNLife.Process.future_cross_entropy_congr_on_positive_support

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationFutureValue.lean#L52)

源码说明：A continuation of zero world-side recurrence has no effect on the future code-length audit. Therefore a distinction useful only under such a continuation creates no retention obligation.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) {m : ℕ} (Q : CGB.GDNLife.Process.WeightedExternalFutureLaw m)
  (s s' : CGB.GDNLife.State k d),
  (∀ (i : Fin m),
      0 < Q.weight i →
        CGB.PredictiveHarvest.crossEntropy (Q.law i)
            (CGB.GDNLife.Process.futureLanguageSignature M Q.toExternalFutureProbes s i) =
          CGB.PredictiveHarvest.crossEntropy (Q.law i)
            (CGB.GDNLife.Process.futureLanguageSignature M Q.toExternalFutureProbes s' i)) →
    CGB.GDNLife.Process.futureLanguageCrossEntropy M Q s = CGB.GDNLife.Process.futureLanguageCrossEntropy M Q s'
```

## CGB.GDNLife.Process.positive_continuation_improvement_is_future_paid

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationFutureValue.lean#L71)

源码说明：If a body change never worsens any declared future continuation and strictly improves one continuation that recurs with positive mass, it strictly improves expected future language code length. This makes the preservation value an information-theoretic quantity, rather than a hand-assigned reward.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) {m : ℕ} (Q : CGB.GDNLife.Process.WeightedExternalFutureLaw m)
  (before after : CGB.GDNLife.State k d),
  (∀ (i : Fin m),
      CGB.PredictiveHarvest.crossEntropy (Q.law i)
          (CGB.GDNLife.Process.futureLanguageSignature M Q.toExternalFutureProbes after i) ≤
        CGB.PredictiveHarvest.crossEntropy (Q.law i)
          (CGB.GDNLife.Process.futureLanguageSignature M Q.toExternalFutureProbes before i)) →
    ∀ (i : Fin m),
      0 < Q.weight i →
        CGB.PredictiveHarvest.crossEntropy (Q.law i)
              (CGB.GDNLife.Process.futureLanguageSignature M Q.toExternalFutureProbes after i) <
            CGB.PredictiveHarvest.crossEntropy (Q.law i)
              (CGB.GDNLife.Process.futureLanguageSignature M Q.toExternalFutureProbes before i) →
          CGB.GDNLife.Process.futureLanguagePaid M Q before after
```
