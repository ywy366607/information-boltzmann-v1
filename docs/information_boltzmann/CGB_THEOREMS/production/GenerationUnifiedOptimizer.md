# GenerationUnifiedOptimizer

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean) · [返回导航](../README.md)

模块源码说明：Unified CGB optimizer contract This module is a composition layer, not a new axiom and not a claim about a trained language model.

## CGB.Unified.build_is_cgb_candidate

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L58)

源码说明：The composed relation and gain satisfy the existing CGB admission contract.

形式陈述：

```lean
∀ (κ base kp km p₀ γ R ρ : ℝ) (carrier : CGB.LanguageLifeKernel.ConservedCarrierPath),
  κ ≠ 0 →
    0 < base →
      CGB.Generation.CGBOptimizerCandidate (CGB.Unified.build κ base kp km p₀ γ R ρ carrier).relation
        (CGB.Unified.build κ base kp km p₀ γ R ρ carrier).gain
```

## CGB.Unified.balanced_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L70)

源码说明：A bounded exchange fraction gives a nonnegative allocation.

形式陈述：

```lean
∀ {p : ℝ}, 0 ≤ p → p ≤ 1 → CGB.Continual.Nonneg (CGB.Unified.balancedAllocation p)
```

## CGB.Unified.balanced_col_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L76)

源码说明：A balanced allocation preserves every column budget.

形式陈述：

```lean
∀ (p : ℝ), CGB.Continual.ColBudget (CGB.Unified.balancedAllocation p)
```

## CGB.Unified.balanced_row_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L81)

源码说明：A balanced allocation preserves every row budget.

形式陈述：

```lean
∀ (p : ℝ), CGB.Continual.RowBudget (CGB.Unified.balancedAllocation p)
```

## CGB.Unified.balanced_allocation_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L86)

源码说明：The double-budget contract bounds the positive-feedback payoff.

形式陈述：

```lean
∀ {p : ℝ},
  0 ≤ p →
    p ≤ 1 →
      CGB.Continual.Nonneg (CGB.Unified.balancedAllocation p) ∧
        CGB.Continual.ColBudget (CGB.Unified.balancedAllocation p) ∧
          CGB.Continual.RowBudget (CGB.Unified.balancedAllocation p) ∧
            CGB.Continual.payoff (CGB.Unified.balancedAllocation p) ≤ 2
```

## CGB.Unified.balanced_max_differentiates

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L97)

源码说明：At the doubly-budgeted maximum, the two rows own different directions.

形式陈述：

```lean
∀ {p : ℝ},
  0 ≤ p →
    p ≤ 1 →
      CGB.Continual.payoff (CGB.Unified.balancedAllocation p) = 2 →
        CGB.Unified.balancedAllocation p 0 0 ≠ CGB.Unified.balancedAllocation p 1 0
```

## CGB.Unified.uniform_allocation_is_budgeted

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L106)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Continual.Nonneg CGB.Unified.uniformAllocation ∧
  CGB.Continual.ColBudget CGB.Unified.uniformAllocation ∧
    CGB.Continual.RowBudget CGB.Unified.uniformAllocation ∧
      ¬CGB.Generation.PairwiseRelationCarrier CGB.Unified.uniformAllocation
```

## CGB.Unified.balanced_allocation_interaction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L119)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p : ℝ), CGB.Generation.interaction (CGB.Unified.balancedAllocation p) = 4 * p - 2
```

## CGB.Unified.balanced_allocation_has_relation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L125)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {p : ℝ}, p ≠ 1 / 2 → CGB.Generation.PairwiseRelationCarrier (CGB.Unified.balancedAllocation p)
```

## CGB.Unified.phase_relation_interaction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L139)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (κ h : ℝ), CGB.Generation.interaction (CGB.Unified.phaseRelation κ h) = κ * (2 * h - 1)
```

## CGB.Unified.phase_relation_has_relation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L144)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {κ h : ℝ}, κ ≠ 0 → h ≠ 1 / 2 → CGB.Generation.PairwiseRelationCarrier (CGB.Unified.phaseRelation κ h)
```

## CGB.Unified.phase_history_is_generated

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L150)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (κ h kp km : ℝ),
  kp + km ≠ 0 →
    ∀ (t : ℕ), CGB.Generation.twoStateIter kp km h t - kp / (kp + km) = (1 - kp - km) ^ t * (h - kp / (kp + km))
```

## CGB.Unified.phase_step_stays_in_unit_interval

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L156)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {kp km h : ℝ},
  0 ≤ kp →
    kp ≤ 1 →
      0 ≤ km →
        km ≤ 1 → 0 ≤ h → h ≤ 1 → 0 ≤ CGB.Generation.twoStateStep kp km h ∧ CGB.Generation.twoStateStep kp km h ≤ 1
```

## CGB.Unified.phase_history_stays_in_unit_interval

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L167)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {kp km h : ℝ},
  0 ≤ kp →
    kp ≤ 1 →
      0 ≤ km →
        km ≤ 1 →
          0 ≤ h →
            h ≤ 1 → ∀ (t : ℕ), 0 ≤ CGB.Generation.twoStateIter kp km h t ∧ CGB.Generation.twoStateIter kp km h t ≤ 1
```

## CGB.Unified.real_transfer_total

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L189)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t : CGB.Unified.RealConservedTransfer), t.sourceAfter + t.targetAfter = t.sourceBefore + t.targetBefore
```

## CGB.Unified.real_transfer_target_readout

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L193)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t : CGB.Unified.RealConservedTransfer), t.targetAfter - t.targetBefore = t.amount
```

## CGB.Unified.phase_exchange_transfer_is_conserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L207)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (kp km h : ℝ),
  (CGB.Unified.phaseExchangeTransfer kp km h).sourceAfter + (CGB.Unified.phaseExchangeTransfer kp km h).targetAfter = 1
```

## CGB.Unified.phase_exchange_transfer_reads_phase

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L213)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (kp km h : ℝ),
  (CGB.Unified.phaseExchangeTransfer kp km h).targetAfter - (CGB.Unified.phaseExchangeTransfer kp km h).targetBefore =
    (CGB.Unified.phaseExchangeTransfer kp km h).amount
```

## CGB.Unified.fresh_outcome_measurement_unbiased

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L228)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (h e u ξ : ℝ),
  (h * e ^ 2 / 2 + ξ * e - (h * (e - u) ^ 2 / 2 + ξ * (e - u)) +
        (h * e ^ 2 / 2 + -ξ * e - (h * (e - u) ^ 2 / 2 + -ξ * (e - u)))) /
      2 =
    h * e ^ 2 / 2 - h * (e - u) ^ 2 / 2
```

## CGB.Unified.fresh_outcome_trust_stays_in_band

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L234)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (γ R ρ predicted realized deadband : ℝ),
  1 < γ →
    0 < R →
      R / γ ≤ ρ →
        ρ ≤ R * γ →
          R / γ ≤ CGB.Unified.freshOutcomeTrust γ R ρ predicted realized deadband ∧
            CGB.Unified.freshOutcomeTrust γ R ρ predicted realized deadband ≤ R * γ
```

## CGB.Unified.bernoulli_expected_brier_gap

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L254)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p q : ℝ), CGB.Unified.bernoulliExpectedBrier p q = CGB.Unified.bernoulliExpectedBrier q q - (p - q) ^ 2
```

## CGB.Unified.bernoulli_expected_brier_calibrated

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L260)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p q : ℝ), CGB.Unified.bernoulliExpectedBrier p q ≤ CGB.Unified.bernoulliExpectedBrier q q
```

## CGB.Unified.centered_bernoulli_evidence_zero_mean

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L268)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p q : ℝ),
  q * CGB.Unified.centeredBernoulliEvidence p 1 (CGB.Unified.bernoulliExpectedBrier p q) +
      (1 - q) * CGB.Unified.centeredBernoulliEvidence p 0 (CGB.Unified.bernoulliExpectedBrier p q) =
    0
```

## CGB.Unified.calibrated_evidence_penalty_is_square

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L274)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p q : ℝ), CGB.Unified.bernoulliExpectedBrier q q - CGB.Unified.bernoulliExpectedBrier p q = (p - q) ^ 2
```

## CGB.Unified.ema_history_collision

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L290)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
0 ≠ 2 ∧
  CGB.Unified.ema3 (1 / 2) 2 0 0 = CGB.Unified.ema3 (1 / 2) (-2 / 3) (4 / 3) 0 ∧
    CGB.Unified.emaSquare3 (1 / 2) 2 0 0 = CGB.Unified.emaSquare3 (1 / 2) (-2 / 3) (4 / 3) 0
```

## CGB.Unified.build_has_competition_and_differentiation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L297)

源码说明：The built architecture exposes both row and column competition budgets.

形式陈述：

```lean
∀ (κ base kp km p₀ γ R ρ : ℝ) (carrier : CGB.LanguageLifeKernel.ConservedCarrierPath) {p : ℝ},
  0 ≤ p →
    p ≤ 1 →
      CGB.Continual.Nonneg ((CGB.Unified.build κ base kp km p₀ γ R ρ carrier).allocation p) ∧
        CGB.Continual.ColBudget ((CGB.Unified.build κ base kp km p₀ γ R ρ carrier).allocation p) ∧
          CGB.Continual.RowBudget ((CGB.Unified.build κ base kp km p₀ γ R ρ carrier).allocation p)
```

## CGB.Unified.build_history_is_geometric

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L309)

源码说明：The retained history is the geometric trace generated by two-state exchange.

形式陈述：

```lean
∀ (κ base kp km p₀ γ R ρ : ℝ) (carrier : CGB.LanguageLifeKernel.ConservedCarrierPath),
  kp + km ≠ 0 →
    ∀ (t : ℕ),
      (CGB.Unified.build κ base kp km p₀ γ R ρ carrier).history t - kp / (kp + km) =
        (1 - kp - km) ^ t * (p₀ - kp / (kp + km))
```

## CGB.Unified.build_history_balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L317)

源码说明：The supplied typed carrier exposes its telescoping conserved history.

形式陈述：

```lean
∀ (κ base kp km p₀ γ R ρ : ℝ) (carrier : CGB.LanguageLifeKernel.ConservedCarrierPath) (k : ℕ),
  (CGB.Unified.build κ base kp km p₀ γ R ρ carrier).carrier.value k -
      (CGB.Unified.build κ base kp km p₀ γ R ρ carrier).carrier.value 0 =
    ∑ n ∈ Finset.range k,
      ((CGB.Unified.build κ base kp km p₀ γ R ρ carrier).carrier.input n -
        (CGB.Unified.build κ base kp km p₀ γ R ρ carrier).carrier.output n)
```

## CGB.Unified.build_modulation_stays_in_band

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L329)

源码说明：The outcome gate leaves a dead band and stays inside the trust interval.

形式陈述：

```lean
∀ (κ base kp km p₀ γ R ρ : ℝ) (carrier : CGB.LanguageLifeKernel.ConservedCarrierPath),
  1 < γ →
    0 < R →
      R / γ ≤ ρ →
        ρ ≤ R * γ →
          ∀ (evidence : Bool),
            R / γ ≤ (CGB.Unified.build κ base kp km p₀ γ R ρ carrier).modulation evidence ∧
              (CGB.Unified.build κ base kp km p₀ γ R ρ carrier).modulation evidence ≤ R * γ
```

## CGB.Unified.build_latch_holds_and_releases

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L338)

源码说明：The latch both holds under short contrary evidence and releases at its threshold.

形式陈述：

```lean
∀ (c runHold runRelease : ℕ) (v d : ℝ),
  runHold < c →
    c ≤ runRelease → CGB.Continual.deferLatch c runHold v d = v ∧ CGB.Continual.deferLatch c runRelease v d = d
```

## CGB.Unified.build_mechanical_admission

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L349)

源码说明：Combined mechanical admission: relation, competition, history and modulation.

形式陈述：

```lean
∀ (κ base kp km p₀ γ R ρ : ℝ) (carrier : CGB.LanguageLifeKernel.ConservedCarrierPath),
  κ ≠ 0 →
    0 < base →
      ∀ {p : ℝ},
        0 ≤ p →
          p ≤ 1 →
            1 < γ →
              0 < R →
                R / γ ≤ ρ →
                  ρ ≤ R * γ →
                    kp + km ≠ 0 →
                      ∀ (t : ℕ),
                        CGB.Generation.CGBOptimizerCandidate (CGB.Unified.build κ base kp km p₀ γ R ρ carrier).relation
                            (CGB.Unified.build κ base kp km p₀ γ R ρ carrier).gain ∧
                          CGB.Continual.Nonneg ((CGB.Unified.build κ base kp km p₀ γ R ρ carrier).allocation p) ∧
                            CGB.Continual.ColBudget ((CGB.Unified.build κ base kp km p₀ γ R ρ carrier).allocation p) ∧
                              CGB.Continual.RowBudget ((CGB.Unified.build κ base kp km p₀ γ R ρ carrier).allocation p) ∧
                                (CGB.Unified.build κ base kp km p₀ γ R ρ carrier).history t - kp / (kp + km) =
                                    (1 - kp - km) ^ t * (p₀ - kp / (kp + km)) ∧
                                  R / γ ≤ (CGB.Unified.build κ base kp km p₀ γ R ρ carrier).modulation false ∧
                                    (CGB.Unified.build κ base kp km p₀ γ R ρ carrier).modulation false ≤ R * γ
```
