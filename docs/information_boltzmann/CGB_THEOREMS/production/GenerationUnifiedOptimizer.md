# GenerationUnifiedOptimizer

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean) · [返回导航](../README.md)

模块源码说明：Unified CGB optimizer contract This module is a composition layer, not a new axiom and not a claim about a trained language model.

## CGB.Unified.build_is_cgb_candidate

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L58)

解释：输入复合关系矩阵与增益调节算子，经准入谓词核算，输出构建的优化器候选满足CGB准入公理。

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

解释：输入落在单位区间[0, 1]内的交换参数p，经元素符号检验，输出平衡分配矩阵各元素非负。

源码说明：A bounded exchange fraction gives a nonnegative allocation.

形式陈述：

```lean
∀ {p : ℝ}, 0 ≤ p → p ≤ 1 → CGB.Continual.Nonneg (CGB.Unified.balancedAllocation p)
```

## CGB.Unified.balanced_col_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L76)

解释：输入平衡分配矩阵，经列向求和计算，输出每列分配总和精确守恒为1。

源码说明：A balanced allocation preserves every column budget.

形式陈述：

```lean
∀ (p : ℝ), CGB.Continual.ColBudget (CGB.Unified.balancedAllocation p)
```

## CGB.Unified.balanced_row_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L81)

解释：输入平衡分配矩阵，经行向求和计算，输出每行分配总和精确守恒为1。

源码说明：A balanced allocation preserves every row budget.

形式陈述：

```lean
∀ (p : ℝ), CGB.Continual.RowBudget (CGB.Unified.balancedAllocation p)
```

## CGB.Unified.balanced_allocation_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L86)

解释：输入交换比例参数p∈[0, 1]，经双预算可行集检验，输出平衡分配矩阵满足非负性、列预算与行预算，且集中收益不超过2。

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

解释：输入达到双预算极值收益2的分配矩阵，经两行极值比较，输出两行分别完全独占不同方向实现完全分化。

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

解释：输入均匀分配矩阵，经行列求和检验与反对称分析，输出均匀矩阵满足行列双预算但无法提供非平凡的两两特化关系。

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

解释：输入参数p，经矩阵相互作用泛函求值，输出相互作用量精确等于4*p - 2。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p : ℝ), CGB.Generation.interaction (CGB.Unified.balancedAllocation p) = 4 * p - 2
```

## CGB.Unified.balanced_allocation_has_relation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L125)

解释：输入非对称分配参数p≠1/2，经双向差分检验，输出平衡分配矩阵满足非平凡两两关系载体条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {p : ℝ}, p ≠ 1 / 2 → CGB.Generation.PairwiseRelationCarrier (CGB.Unified.balancedAllocation p)
```

## CGB.Unified.phase_relation_interaction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L139)

解释：输入增益κ与相位h，经相互作用泛函求值，输出相位相互作用量精确等于κ * (2 * h - 1)。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (κ h : ℝ), CGB.Generation.interaction (CGB.Unified.phaseRelation κ h) = κ * (2 * h - 1)
```

## CGB.Unified.phase_relation_has_relation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L144)

解释：输入非零增益κ与非中心相位h≠1/2，经差分检验，输出相位关系矩阵满足两两反对称非平凡关系载体条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {κ h : ℝ}, κ ≠ 0 → h ≠ 1 / 2 → CGB.Generation.PairwiseRelationCarrier (CGB.Unified.phaseRelation κ h)
```

## CGB.Unified.phase_history_is_generated

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L150)

解释：输入非零速率和与时间步t，经一阶常系数差分方程解析求解，输出相位偏差随时间按(1 - kp - km)^t几何幂次指数收敛。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (κ h kp km : ℝ),
  kp + km ≠ 0 →
    ∀ (t : ℕ), CGB.Generation.twoStateIter kp km h t - kp / (kp + km) = (1 - kp - km) ^ t * (h - kp / (kp + km))
```

## CGB.Unified.phase_step_stays_in_unit_interval

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L156)

解释：输入各合法参数，经单步凸组合检验，输出单步两状态步进后的相位严格落在单位区间[0, 1]内。

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

解释：输入各合法速率参数与初态，经数学归纳法推演，输出相位在任意离散时间步均严格保持在单位区间[0, 1]内。

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

解释：输入实数守恒转移算子，经全量代数求和，输出转移后两端总量与转移前两端总量严格守恒相等。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t : CGB.Unified.RealConservedTransfer), t.sourceAfter + t.targetAfter = t.sourceBefore + t.targetBefore
```

## CGB.Unified.real_transfer_target_readout

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L193)

解释：输入实数守恒转移记录，经目标增量提取，输出转移后目标值减去转移前目标值严格等于转移量amount。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t : CGB.Unified.RealConservedTransfer), t.targetAfter - t.targetBefore = t.amount
```

## CGB.Unified.phase_exchange_transfer_is_conserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L207)

解释：输入相位交换转移算子，经前后状态求和，输出转移前后的总相位质量严格守恒为1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (kp km h : ℝ),
  (CGB.Unified.phaseExchangeTransfer kp km h).sourceAfter + (CGB.Unified.phaseExchangeTransfer kp km h).targetAfter = 1
```

## CGB.Unified.phase_exchange_transfer_reads_phase

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L213)

解释：输入相位交换转移结构，经后继与前序做差，输出目标节点相位净增量精确等于单步交换转移量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (kp km h : ℝ),
  (CGB.Unified.phaseExchangeTransfer kp km h).targetAfter - (CGB.Unified.phaseExchangeTransfer kp km h).targetBefore =
    (CGB.Unified.phaseExchangeTransfer kp km h).amount
```

## CGB.Unified.fresh_outcome_measurement_unbiased

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L228)

解释：输入扰动项与对称零均值噪声，经两极扰动期望抵消，输出新鲜测量差分对真实目标导数无偏。

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

解释：输入置信带宽因子γ、基准R及当前置信度，经门控限幅截断，输出后继置信度参数严格被夹在闭区间[R/γ, R*γ]内。

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

解释：输入预测概率p与目标概率q，经平方误差分解，输出期望得分与校准得分之差严格等于平方差距-(p - q)²。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p q : ℝ), CGB.Unified.bernoulliExpectedBrier p q = CGB.Unified.bernoulliExpectedBrier q q - (p - q) ^ 2
```

## CGB.Unified.bernoulli_expected_brier_calibrated

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L260)

解释：输入预测概率p与目标概率q，经Brier得分期望展开，输出在自校准预测点p=q处期望Brier得分取得全局最大值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p q : ℝ), CGB.Unified.bernoulliExpectedBrier p q ≤ CGB.Unified.bernoulliExpectedBrier q q
```

## CGB.Unified.bernoulli_prediction_error_expectation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L271)

解释：输入预测概率p与目标分布q，经伯努利线性期望展开，输出单步预测误差的无偏期望精确等于真实概率与预测概率之差q - p。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p q : ℝ), q * CGB.Unified.bernoulliPredictionError p 1 + (1 - q) * CGB.Unified.bernoulliPredictionError p 0 = q - p
```

## CGB.Unified.brier_score_is_not_outcome_prediction_error

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L276)

解释：输入中性预测p=1/2与结果0，经代数比较，输出Brier得分计算与线性预测误差严格不等（证明二者非等价）。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Unified.bernoulliBrierScore (1 / 2) 0 ≠ CGB.Unified.bernoulliPredictionError (1 / 2) 0
```

## CGB.Unified.centered_bernoulli_evidence_zero_mean

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L280)

解释：输入均值中心化的伯努利证据变量，经期望积分计算，输出中心化证据在真实分布下的数学期望恒等于0。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p q : ℝ),
  q * CGB.Unified.centeredBernoulliEvidence p 1 (CGB.Unified.bernoulliExpectedBrier p q) +
      (1 - q) * CGB.Unified.centeredBernoulliEvidence p 0 (CGB.Unified.bernoulliExpectedBrier p q) =
    0
```

## CGB.Unified.calibrated_evidence_penalty_is_square

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L286)

解释：输入预测概率与真实概率，经期望差分展开，输出校准惩罚量精确等于二者概率偏差的二次方(p - q)²。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p q : ℝ), CGB.Unified.bernoulliExpectedBrier q q - CGB.Unified.bernoulliExpectedBrier p q = (p - q) ^ 2
```

## CGB.Unified.eligibility_gated_phase_stays_in_unit_interval

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L299)

解释：输入在[0, 1]内的两状态速率、初始相位与门控值，经凸组合放缩，输出门控后继相位依然严格保持在单位区间[0, 1]内。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {kp km h m : ℝ},
  0 ≤ kp →
    kp ≤ 1 →
      0 ≤ km →
        km ≤ 1 →
          0 ≤ h →
            h ≤ 1 →
              0 ≤ m →
                m ≤ 1 →
                  0 ≤ CGB.Unified.eligibilityGatedPhaseStep kp km h m ∧
                    CGB.Unified.eligibilityGatedPhaseStep kp km h m ≤ 1
```

## CGB.Unified.eligibility_gate_zero_holds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L312)

解释：输入两状态速率与门控值0，经步进计算，输出当资格门控关闭时状态严格保持不变（零门控锁存）。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (kp km h : ℝ), CGB.Unified.eligibilityGatedPhaseStep kp km h 0 = h
```

## CGB.Unified.eligibility_gate_one_recovers

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L316)

解释：输入两状态速率与门控值1，经门控步进计算，输出当资格门控全开时状态更新精确复现标准两状态步进。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (kp km h : ℝ), CGB.Unified.eligibilityGatedPhaseStep kp km h 1 = CGB.Generation.twoStateStep kp km h
```

## CGB.Unified.eligibility_gated_signed_phase_stays_bounded

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L323)

解释：输入在[-1, 1]内的前序有符号相位、原始值与有效门控，经插值缩放，输出门控后的有符号相位严格有界于[-1, 1]。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {previous raw m : ℝ},
  -1 ≤ previous →
    previous ≤ 1 →
      -1 ≤ raw →
        raw ≤ 1 →
          0 ≤ m →
            m ≤ 1 →
              -1 ≤ CGB.Unified.eligibilityGatedSignedPhase previous raw m ∧
                CGB.Unified.eligibilityGatedSignedPhase previous raw m ≤ 1
```

## CGB.Unified.eligibility_gated_signed_phase_zero_holds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L336)

解释：输入前序相位与零门控，经插值求值，输出在门控为0时有符号相位严格等于前序状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (previous raw : ℝ), CGB.Unified.eligibilityGatedSignedPhase previous raw 0 = previous
```

## CGB.Unified.latch_phase_holds_before_release

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L343)

解释：输入释放阈值c与当前运行计数run < c，经资格门控求值，输出未达阈值前锁存相位严格保持前序值不变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (c run : ℕ) (previous raw m : ℝ),
  run < c → CGB.Unified.eligibilityGatedSignedPhase previous raw (CGB.Unified.latchEligibilityGate c run m) = previous
```

## CGB.Unified.latch_phase_releases_at_threshold

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L349)

解释：输入运行步数达到阈值run ≥ c，经门控激活判定，输出锁存相位在达到阈值后正常释放并按输入信号更新。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (c run : ℕ) (previous raw m : ℝ),
  c ≤ run →
    CGB.Unified.eligibilityGatedSignedPhase previous raw (CGB.Unified.latchEligibilityGate c run m) =
      CGB.Unified.eligibilityGatedSignedPhase previous raw m
```

## CGB.Unified.eligibility_gated_transfer_is_conserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L365)

解释：输入源与目标相位转移量，经代数求和，输出门控转移后的源与目标总量精确守恒为1。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (kp km h m : ℝ),
  (CGB.Unified.eligibilityGatedPhaseTransfer kp km h m).sourceAfter +
      (CGB.Unified.eligibilityGatedPhaseTransfer kp km h m).targetAfter =
    1
```

## CGB.Unified.eligibility_gated_transfer_reads_phase

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L371)

解释：输入门控相位转移结构，经目标差分提取，输出目标增量严格精确等于声明的转移量amount。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (kp km h m : ℝ),
  (CGB.Unified.eligibilityGatedPhaseTransfer kp km h m).targetAfter -
      (CGB.Unified.eligibilityGatedPhaseTransfer kp km h m).targetBefore =
    (CGB.Unified.eligibilityGatedPhaseTransfer kp km h m).amount
```

## CGB.Unified.ema_history_collision

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L388)

解释：输入EMA平滑系数与两组不同历史输入，经二阶多项式求值，输出两组非零不同历史序列生成完全相同的EMA一阶与二阶矩（证明有限阶EMA无法唯一区分高阶历史）。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
0 ≠ 2 ∧
  CGB.Unified.ema3 (1 / 2) 2 0 0 = CGB.Unified.ema3 (1 / 2) (-2 / 3) (4 / 3) 0 ∧
    CGB.Unified.emaSquare3 (1 / 2) 2 0 0 = CGB.Unified.emaSquare3 (1 / 2) (-2 / 3) (4 / 3) 0
```

## CGB.Unified.build_has_competition_and_differentiation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L395)

解释：输入构建的多柱优化器架构，经资源与方向分配分析，输出系统同时满足行向与列向竞争与特化分化条件。

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

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L407)

解释：输入两状态转移率与初态，经差分方程求解，输出保留的历史状态精确等于由两状态参数生成的几何衰减轨迹。

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

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L415)

解释：输入带类型的载体状态，经裂项级数平衡分析，输出历史痕迹满足守恒平衡方程。

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

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L427)

解释：输入结果门控与死区范围，经非线性饱和截断分析，输出调制信号在死区外保持且全局稳定在置信带宽之内。

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

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L436)

解释：输入阈值判定逻辑与反向证据序列，经双稳态动力学求解，输出锁存在短程反向证据下保持，并在累积证据突破阈值后快速释放。

源码说明：The latch both holds under short contrary evidence and releases at its threshold.

形式陈述：

```lean
∀ (c runHold runRelease : ℕ) (v d : ℝ),
  runHold < c →
    c ≤ runRelease → CGB.Continual.deferLatch c runHold v d = v ∧ CGB.Continual.deferLatch c runRelease v d = d
```

## CGB.Unified.build_mechanical_admission

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L447)

解释：输入关系载体、竞争结构与历史追踪，经复合机械准入规约，输出系统整体通过机械物理准入审计。

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
