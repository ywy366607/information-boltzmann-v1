# optimizer_v6_handoff

探索区：仅按源码盘点；不在生产区全量审计覆盖内，不能据此声称已通过当前构建或信任审计。

[源文件](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean) · [返回导航](../README.md)

模块源码说明：Scope checks for the ongoing CLO6 branch.

## CGB.OptimizerHandoffReview.orthogonal_columns_exclude_one_row_wta

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L14)

解释：二维矩阵若两列均为单位向量且彼此正交，不可能两列都把全部平方质量集中在同一行。

源码说明：The column-budget WTA witness cannot be the squared entries of a 2x2 orthogonal matrix. Dropping orthogonality before constructing a counterexample does not refute Muon itself.

形式陈述：

```lean
theorem orthogonal_columns_exclude_one_row_wta (a b c d : ℝ)
    (h0 : a ^ 2 + c ^ 2 = 1) (h1 : b ^ 2 + d ^ 2 = 1)
    (hcross : a * b + c * d = 0) (hc : c = 0) (hd : d = 0) : False
```

## CGB.OptimizerHandoffReview.posterior_mean_subtracted_power

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L29)

解释：指定高斯后验公式中，后验均值平方期望减后验方差等于 S(S−R)/(S+R)，并非一般地等于信号方差。

源码说明：Algebraic specialization of a scalar conjugate Gaussian calculation: prior variance S, observation noise R, posterior mean variance S^2/(S+R), posterior variance S*R/(S+R). Subtracting the posterior variance does NOT recover S. The probability-model interpretation is external to this lemma.

形式陈述：

```lean
theorem posterior_mean_subtracted_power (S R : ℝ) :
    S ^ 2 / (S + R) - S * R / (S + R) = S * (S - R) / (S + R)
```

## CGB.OptimizerHandoffReview.unit_posterior_power_counterexample

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L37)

解释：信号和噪声方差均为一时，错误的后验平方减方差为零，而相加得到一。

源码说明：At unit signal/noise variance this proposed subtraction gives zero, although the signal variance is one; adding posterior variance recovers it.

形式陈述：

```lean
theorem unit_posterior_power_counterexample :
    (1 / 2 : ℝ) - 1 / 2 = 0 ∧ (1 / 2 : ℝ) + 1 / 2 = 1
```

## CGB.OptimizerHandoffReview.mixture_total_variance

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L42)

解释：两分支二阶矩减混合均值平方，等于分支方差加权和加分支均值差的平方项。

源码说明：Total variance of a two-component mixture. Probability interpretation requires 0 <= w <= 1 and nonnegative component variances.

形式陈述：

```lean
theorem mixture_total_variance (w mo mj po pj : ℝ) :
    w * (po + mo ^ 2) + (1 - w) * (pj + mj ^ 2) -
      (w * mo + (1 - w) * mj) ^ 2 =
    w * po + (1 - w) * pj + w * (1 - w) * (mo - mj) ^ 2
```

## CGB.OptimizerHandoffReview.soft_hold_variance_gap

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L51)

解释：指定软保持混合模型的真实方差相对简化递推多出 w(1−w)(k·innovation)²。

源码说明：If soft hold mixes an ordinary Kalman write with an unchanged branch, its mean matches CLO5/6, but its variance includes a between-branch term. This identifies the missing term under that interpretation, not a complete generative model for jump acceptance or correlated observations.

形式陈述：

```lean
theorem soft_hold_variance_gap (w k p mu innovation : ℝ) :
    w * ((1 - k) * p + (mu + k * innovation) ^ 2) +
      (1 - w) * (p + mu ^ 2) - (mu + w * k * innovation) ^ 2 -
      (p - w * k * p) = w * (1 - w) * (k * innovation) ^ 2
```

## CGB.OptimizerHandoffReview.disturbed_error_stays_in_tube

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L59)

解释：若误差递推满足给定收缩与噪声界且 q·radius+noise≤radius，则初始误差在界内时以后始终在界内。

源码说明：Conditional noisy-error contract: once the actual update satisfies this recurrence, a forward-invariant error tube follows. Noise need not contract on every tick. This does not assert that CLO6 satisfies the recurrence.

形式陈述：

```lean
theorem disturbed_error_stays_in_tube (err : ℕ → ℝ) (q noise radius : ℝ)
    (hq : 0 ≤ q) (hbudget : q * radius + noise ≤ radius)
    (hinit : err 0 ≤ radius)
    (hstep : ∀ n, err (n + 1) ≤ q * err n + noise) (n : ℕ) : err n ≤ radius
```

## CGB.OptimizerAdmissionReview.complete_step_certificate

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L84)

解释：若完整步长的模型误差包络成立且最终目标不为正，则新损失加明确成本不超过旧损失。

源码说明：The final executed step is certified only with a valid whole-step model, valid error envelopes, and a nonpositive final objective. This neither assumes nor proves that the Python solver or online estimates meet these premises.

形式陈述：

```lean
theorem complete_step_certificate {n : ℕ} (g actual h err u : Fin n → ℝ)
    (cost oldLoss newLoss : ℝ)
    (herr : ∀ i, |actual i - g i| ≤ err i)
    (hmodel : newLoss - oldLoss ≤
      -(∑ i, actual i * u i) + (∑ i, h i * u i ^ 2) / 2)
    (hchosen : objective g h err u cost ≤ 0) : newLoss + cost ≤ oldLoss
```

## CGB.OptimizerAdmissionReview.threshold_hold_and_plasticity

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L114)

解释：在非负阈值、正分母和正可用半径下，该门控规则对小信号保持零更新，对超过阈值的正负信号给出相应方向。

源码说明：This proposal permits exact holding and both signs of plastic response under positive available radius/denominator. This is a gate property, not yet a healthy or beneficial historical latch.

形式陈述：

```lean
theorem threshold_hold_and_plasticity (g threshold denom radius : ℝ)
    (ht : 0 ≤ threshold) (hd : 0 < denom) (hr : 0 < radius) :
    (|g| ≤ threshold → thresholdMove g threshold denom radius = 0) ∧
    (threshold < g → 0 < thresholdMove g threshold denom radius) ∧
    (g < -threshold → thresholdMove g threshold denom radius < 0)
```

## CGB.OptimizerAdmissionReview.shared_budget_has_no_uniform_tail_share

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L138)

解释：在给定二次更新目标和共享预算下，可构造正信号方向使弱方向的最优平方更新份额小于任意正数。

源码说明：With the same quadratic update objective and one upper budget, both directions can have positive signal, yet the weaker one's squared update share can be arbitrarily small at the exact optimum. Thus a shared budget alone gives no uniform anti-concentration margin. This concerns update shares, and is not a claim about functional diversity.

形式陈述：

```lean
theorem shared_budget_has_no_uniform_tail_share (ε : ℝ) (hε : 0 < ε) :
    ∃ t : ℝ, 0 < t ∧ t < 1 ∧ t ^ 2 / (1 + t ^ 2) < ε ∧
      ∀ u v : ℝ,
        (1 + t ^ 2) / 2 - 1 - t * t ≤ (u ^ 2 + v ^ 2) / 2 - u - t * v
```

## CGB.OptimizerAdmissionReview.identical_prefix_retention_adaptation_bound

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L165)

解释：同一标量决策若面对目标值相差 δ 的两个不可区分世界，则至少一个世界的平方误差不小于 δ²/4。

源码说明：If an identical visible prefix must support the same scalar decision a in two worlds with incompatible desired values 0 and delta, at least one squared error is delta^2/4. Future return/nonreturn cannot be inferred from that identical prefix. Extra delivered evidence changes this premise.

形式陈述：

```lean
theorem identical_prefix_retention_adaptation_bound (a delta : ℝ) :
    delta ^ 2 / 4 ≤ max (a ^ 2) ((a - delta) ^ 2)
```

## CGB.OptimizerAdmissionReview.replenished_pool_temporal_service

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L177)

解释：在给定正补充量及连续两次需求下界等前提下，资源池提供定理所述的跨两次机会的服务下界。

源码说明：A temporal resource condition, using the existing G20 exchange law. A single use can be zero, but replenishment and two persistent eligible demands give a positive two-opportunity service bound. For a stochastic two-state pool this is a conditional-mean statement, not a pathwise deadline. Nothing here proves that demand represents useful learning or that opportunities are supplied by the actual organism.

形式陈述：

```lean
theorem replenished_pool_temporal_service (supply demand nextDemand available η : ℝ)
    (hs0 : 0 < supply) (hs1 : supply ≤ 1)
    (ha0 : 0 ≤ available) (ha1 : available ≤ 1)
    (hη : 0 < η) (hd : η ≤ demand) (hd1 : demand ≤ 1)
    (hn : η ≤ nextDemand) :
    let next
```

## CGB.ConservedWriteReview.conserved_write_spec

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L219)

解释：正阈值整数载体将总请求分成有界待处理量与量子倍数已提交量，并精确保持两者之和。

源码说明：One bounded carrier retains uncommitted requests exactly; committed writes are multiples of the declared quantum. This neither stores a new history tape nor establishes the usefulness of the requests themselves.

形式陈述：

```lean
theorem conserved_write_spec (k old incoming : ℤ) (hk : 0 < k) :
    pending k old incoming + committed k old incoming = old + incoming ∧
    (-k ≤ pending k old incoming ∧ pending k old incoming < k) ∧
    ∃ packets : ℤ, committed k old incoming = (2 * k) * packets
```

## CGB.ConservedWriteReview.accumulated_hold_and_release

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L238)

解释：指定整数载体在累积请求位于阈值内时不提交，越过正或负阈值时提交对应符号更新。

源码说明：Holding depends on accumulated evidence, not solely on the newest input; crossing either boundary releases a signed write using the SAME carrier.

形式陈述：

```lean
theorem accumulated_hold_and_release (k old incoming : ℤ) (hk : 0 < k) :
    ((-k ≤ old + incoming ∧ old + incoming < k) → committed k old incoming = 0) ∧
    (k ≤ old + incoming → 0 < committed k old incoming) ∧
    (old + incoming < -k → committed k old incoming < 0)
```

## CGB.ConservedWriteReview.persistent_request_cannot_freeze

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L257)

解释：有上界的守恒待处理载体若每步收到至少一单位输入，则某个时刻必有非零输出。

源码说明：Reuse the existing carrier-path theorem: bounded pending state cannot discard indefinitely a persistent integer request. Scheduling, resource payment and input quantization must still be proved for an actual learner.

形式陈述：

```lean
theorem persistent_request_cannot_freeze
    (p : CGB.LanguageLifeKernel.ConservedCarrierPath) (ceiling : ℤ)
    (hbound : ∀ n, p.value n < ceiling) (hdemand : ∀ n, 1 ≤ p.input n) :
    ∃ n, p.output n ≠ 0
```

## CGB.OptimizerIntrinsicReview.ema_history_gap

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L326)

解释：相同未来驱动下，两个 EMA 状态的历史差在 t 步后恰为初始差乘 β 的 t 次幂。

源码说明：With the same future drive, each moment's old-history difference decays exactly by beta^t, independently of how its output will be normalized.

形式陈述：

```lean
theorem ema_history_gap (a b drive : ℕ → ℝ) (β : ℝ)
    (ha : ∀ t, a (t + 1) = β * a t + drive t)
    (hb : ∀ t, b (t + 1) = β * b t + drive t) (t : ℕ) :
    a t - b t = β ^ t * (a 0 - b 0)
```

## CGB.OptimizerIntrinsicReview.ema_fixed_point_unique

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L339)

解释：固定驱动下，若 β 不等于一，则 EMA 递推不能拥有两个不同的不动点。

源码说明：A moment with beta != 1 has only one fixed value under a fixed drive; there are no two self-maintaining moment codes selected by past history.

形式陈述：

```lean
theorem ema_fixed_point_unique (β drive a b : ℝ) (hβ : β ≠ 1)
    (ha : a = β * a + drive) (hb : b = β * b + drive) : a = b
```

## CGB.OptimizerIntrinsicReview.supplied_path_preserves_offset

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L363)

解释：两次运行共享优化器状态及未来外给梯度时，参数初始偏移在该抽象更新路径中逐步原样保留。

源码说明：Nonlinear output processing cannot repair a parameter displacement when the moment state and supplied future gradients agree: the displacement is preserved exactly at every future step. No isolated restorative parameter latch is provided by this optimizer interface alone.

形式陈述：

```lean
theorem supplied_path_preserves_offset {P S G : Type*} [AddCommGroup P]
    (advance : ℕ → S → G → S) (read : ℕ → S → G → P)
    (gradient : ℕ → G) (parameter offset : P) (state : S) (t : ℕ) :
    (suppliedPath advance read gradient (parameter + offset) state t).1 =
      (suppliedPath advance read gradient parameter state t).1 + offset ∧
    (suppliedPath advance read gradient (parameter + offset) state t).2 =
      (suppliedPath advance read gradient parameter state t).2
```

## CGB.OptimizerIntrinsicReview.maintained_gap_requires_regeneration

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L411)

解释：若历史差始终不小于 ε 且每步至多按 q 收缩并加再生项，则有限时段累计再生须满足给定下界。

源码说明：Regeneration has a cumulative obligation: preserving a positive history gap against uniform passive contraction requires continuing regenerative contribution, up to the initial/final stored gap. The units are gap units; converting this bound to energy requires a separate material/cost contract.

形式陈述：

```lean
theorem maintained_gap_requires_regeneration (gap boost : ℕ → ℝ) (q ε : ℝ)
    (hq : q ≤ 1) (hkeep : ∀ t, ε ≤ gap t)
    (hstep : ∀ t, gap (t + 1) ≤ q * gap t + boost t) (T : ℕ) :
    (1 - q) * ((T : ℝ) * ε) ≤
      gap 0 - gap T + ∑ t ∈ Finset.range T, boost t
```

## CGB.OptimizerAdmissionReview.normalized_observation_clock

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L482)

解释：在有效观测率及质量处于给定范围时，归一化观测质量导出相应的有界增益和均值递推。

源码说明：A bounded exposure mass replaces a growing update counter. The effective gain is derived from normalized accumulated observations, including startup. The old mean is already normalized. No beta^t or stationary-prior shortcut.

形式陈述：

```lean
theorem normalized_observation_clock (rate available mass old observed : ℝ)
    (hα : 0 < rate * available) (hα1 : rate * available ≤ 1)
    (hm0 : 0 ≤ mass) (hm1 : mass ≤ 1) :
    let α
```

## CGB.OptimizerAdmissionReview.pooled_gain_risk

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L505)

解释：对指定逐坐标先验和噪声风险，共享增益的总代价可分解为异质性损失与偏离群体最优增益的损失。

源码说明：Exact cost of sharing one gain: heterogeneity loss plus the chosen gain's mismatch to the pooled optimum. P and R are declared prior/noise risks with zero within-coordinate cross term, NOT quantities inferred by an EMA identity. Correlation across coordinates is allowed for this additive risk. A resource-clock gain is admissible, but not asserted Bayes optimal.

形式陈述：

```lean
theorem pooled_gain_risk {n : ℕ} (P R : Fin n → ℝ) (gain : ℝ)
    (hlocal : ∀ i, 0 < P i + R i)
    (htotal : 0 < (∑ i, P i) + ∑ i, R i) :
    let total
```

## CGB.OptimizerAdmissionReview.observer_energy_and_step

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_v6_handoff.lean#L547)

解释：在增益、迁移率和旧状态能量满足给定界时，指定观测更新和归一化步长满足相应能量约束。

源码说明：Same-weight direction and observed energy generate a bounded normalized update. This is empirical energy, not a Hessian/noise estimate. No artificial epsilon or hard clipping is needed: zero energy gives zero update. The caller can absorb sqrt(compartment size) into eta to use RMS instead of sum energy. All allocation/reserve mobilities remain inside the actual final update.

形式陈述：

```lean
theorem observer_energy_and_step {n : ℕ} (mean observed mobility : Fin n → ℝ)
    (gain energy η : ℝ) (hg0 : 0 ≤ gain) (hg1 : gain ≤ 1)
    (he : (∑ i, (mean i)^2) ≤ energy)
    (ha : ∀ i, 0 ≤ mobility i ∧ mobility i ≤ 1) :
    let nextMean
```
