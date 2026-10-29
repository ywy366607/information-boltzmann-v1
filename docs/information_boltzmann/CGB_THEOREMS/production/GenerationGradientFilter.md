# GenerationGradientFilter

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean) · [返回导航](../README.md)

模块源码说明：An optimizer as a learner of its gradient stream (side branch, 2026-10-27) The outer loop's world is the stream of noisy gradients.

## CGB.GradientFilter.memoryless_gain_excess

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L34)

解释：在该随机游走估计模型中，采用增益 1 的无记忆估计器，其误差超出模型最优误差的量恰为 r²/(P+r)。这个结论依赖给定噪声模型，不能直接宣称所有 SGD 不如所有有记忆优化器。

源码说明：A learner without memory (gain 1) exceeds the optimal error `P` by `r^2 / (P + r)`.

形式陈述：

```lean
∀ (r P : ℝ), 0 < r → 0 < P → (P ^ 2 / (P + r) + 1 ^ 2 * r) / (1 * (2 - 1)) - P = r ^ 2 / (P + r)
```

## CGB.GradientFilter.memoryless_strictly_worse

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L42)

解释：当 r 和 P 均为正时，前述增益 1 的估计器严格劣于此模型的最优误差；严格性来自非零观测噪声。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (r P : ℝ), 0 < r → 0 < P → P < (P ^ 2 / (P + r) + 1 ^ 2 * r) / (1 * (2 - 1))
```

## CGB.GradientFilter.shrink_error_square

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L52)

解释：当信号与噪声功率之和为正时，缩放估计的均方误差可以拆成最优误差加上与缩放系数偏差的平方成正比的非负项。概率含义依赖信号与噪声不相关等定义所述假设。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (S N w : ℝ), 0 < S + N → CGB.GradientFilter.shrinkError S N w = S * N / (S + N) + (S + N) * (w - S / (S + N)) ^ 2
```

## CGB.GradientFilter.shrink_optimum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L58)

解释：对所定义的二次误差，只要 S+N 为正，缩放系数 S/(S+N) 的误差不大于任何其他实数系数。这是该单步估计问题的最优解，不直接给出连续训练的最优优化器。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (S N w : ℝ), 0 < S + N → CGB.GradientFilter.shrinkError S N (S / (S + N)) ≤ CGB.GradientFilter.shrinkError S N w
```

## CGB.GradientFilter.unshrunk_excess

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L64)

解释：在给定估计误差模型中，直接使用未缩放观测相较最优缩放的超额误差恰为 N²/(S+N)，前提是分母为正。

源码说明：Taking the noisy estimate at face value costs exactly `N^2 / (S + N)` more.

形式陈述：

```lean
∀ (S N : ℝ),
  0 < S + N → CGB.GradientFilter.shrinkError S N 1 - CGB.GradientFilter.shrinkError S N (S / (S + N)) = N ^ 2 / (S + N)
```

## CGB.GradientFilter.no_shared_shrink_for_two_snr

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L72)

解释：如果两个方向的 S/(S+N) 不同且各自分母为正，那么任何单一共享缩放系数至少在一个方向上严格劣于该方向的最优系数。这个结论针对所定义的单步二次误差。

源码说明：One shared factor is strictly suboptimal in at least one of two directions whose signal fractions differ.

形式陈述：

```lean
∀ (S₁ N₁ S₂ N₂ w : ℝ),
  0 < S₁ + N₁ →
    0 < S₂ + N₂ →
      S₁ / (S₁ + N₁) ≠ S₂ / (S₂ + N₂) →
        CGB.GradientFilter.shrinkError S₁ N₁ (S₁ / (S₁ + N₁)) < CGB.GradientFilter.shrinkError S₁ N₁ w ∨
          CGB.GradientFilter.shrinkError S₂ N₂ (S₂ / (S₂ + N₂)) < CGB.GradientFilter.shrinkError S₂ N₂ w
```

## CGB.GradientFilter.vanishing_gain_loses_under_drift

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L108)

解释：对于定理写出的跟踪方差递推，若环境漂移 q 持续为正，所有增益长期都不超过 q/(4M)，那么方差最终超过任意给定正阈值 M。结论说明此模型中的过小增益无法无限期维持给定误差界。

源码说明：**W2 (drift forbids a vanishing gain)**: the tracking error variance obeys `V (t+1) = (1 - K t)^2 V t + q + K t^2 r`. With drift `q > 0`, if every gain is at most `q / (4 M)` the variance exceeds `M`: a gain that decays to zero (1/t, AdaGrad) loses under drift, whatever bound `M` is asked for.

形式陈述：

```lean
∀ (V K : ℕ → ℝ) (q r M : ℝ),
  0 < q →
    0 ≤ r →
      0 < M →
        0 ≤ V 0 →
          (∀ (t : ℕ), 0 ≤ K t) →
            (∀ (t : ℕ), K t ≤ q / (4 * M)) →
              (∀ (t : ℕ), V (t + 1) = (1 - K t) ^ 2 * V t + q + K t ^ 2 * r) → ∃ t, M < V t
```

## CGB.GradientFilter.emaVar_fixed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L153)

解释：对于所定义的 emaVar，只要平均系数 x 非负，它满足递推 v = x²v+(1−x)²r 的不动点等式；该恒等式不单独证明任意初始化下的收敛。

源码说明：It is the fixed point of `v = x^2 v + (1 - x)^2 r`.

形式陈述：

```lean
∀ (r x : ℝ), 0 ≤ x → x ^ 2 * CGB.GradientFilter.emaVar r x + (1 - x) ^ 2 * r = CGB.GradientFilter.emaVar r x
```

## CGB.GradientFilter.occurrence_clock_beats_time_clock

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L163)

解释：在恒定噪声方差、0<β<1 和每 k≥2 步才出现一次的设定下，以每次出现为时钟的 EMA 稳态方差小于等效因子 β^k 的方案。比较依赖该两种特定更新方式，不说明所有稀疏参数都必须跳过未出现步骤。

源码说明：**W3 (an occurrence clock)**: a direction observed once every `k ≥ 2` steps, averaged with the per-step factor `β` of the frequent directions, effectively uses `β^k` per observation and has a strictly larger stationary variance than averaging per occurrence.

形式陈述：

```lean
∀ (r β : ℝ) (k : ℕ), 0 < r → 0 < β → β < 1 → 2 ≤ k → CGB.GradientFilter.emaVar r β < CGB.GradientFilter.emaVar r (β ^ k)
```

## CGB.GradientFilter.diagonal_cannot_whiten

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L177)

解释：若二维曲率矩阵的非对角元不为零，逐坐标对角缩放不能把它变成非零倍的单位矩阵。这里讨论的是一次精确白化，不排除对角方法通过多步迭代收敛。

源码说明：**W4 (a relation basis)**: if the curvature `[[a, b], [b, d]]` has a nonzero off-diagonal term, no diagonal preconditioner `diag(d₁, d₂)` turns it into a nonzero multiple of the identity (one-step whitening); a basis-aware preconditioner does.

形式陈述：

```lean
∀ (a b d d₁ d₂ c : ℝ), b ≠ 0 → c ≠ 0 → ¬(d₁ * a = c ∧ d₁ * b = 0 ∧ d₂ * b = 0 ∧ d₂ * d = c)
```

## CGB.GradientFilter.usage_normalized_gain_anti_dominant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L187)

解释：当 s>0、c≥0 且两次使用量为正并严格增加时，s/√(c+m) 严格下降。这证明这个函数具有局部使用越多、系数越小的性质，不等于全系统不会赢家通吃。

源码说明：**W5 is implied by W3**: a gain normalized by that direction's accumulated use (`s / sqrt (c + m)`, the AdaGrad / Adam form) strictly falls when the direction's own demand grows: it is locally anti-dominant without a separate anti-winner-take-all component.

形式陈述：

```lean
∀ (s c m₁ m₂ : ℝ), 0 < s → 0 ≤ c → 0 < m₁ → m₁ < m₂ → s / √(c + m₂) < s / √(c + m₁)
```

## CGB.GradientFilter.step_error_square

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L215)

解释：在指定的一维含噪二次模型中，当 h²e²+σ² 为正，一步后的均方误差等于最小项加上步长偏离特定值的平方惩罚项。实际使用仍须估计未知曲率、误差与噪声。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (h e σ2 η : ℝ),
  0 < h ^ 2 * e ^ 2 + σ2 →
    CGB.GradientFilter.stepErr h e σ2 η =
      e ^ 2 * σ2 / (h ^ 2 * e ^ 2 + σ2) + (h ^ 2 * e ^ 2 + σ2) * (η - h * e ^ 2 / (h ^ 2 * e ^ 2 + σ2)) ^ 2
```

## CGB.GradientFilter.optimal_step_minimizes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L222)

解释：对给定的一维一步均方误差，并在 h²e²+σ² 为正时，步长 h·e²/(h²e²+σ²) 的误差不大于任意其他实数步长。它是一维单步最优，不是跨时间或神经网络整体最优。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (h e σ2 η : ℝ),
  0 < h ^ 2 * e ^ 2 + σ2 →
    CGB.GradientFilter.stepErr h e σ2 (h * e ^ 2 / (h ^ 2 * e ^ 2 + σ2)) ≤ CGB.GradientFilter.stepErr h e σ2 η
```

## CGB.GradientFilter.optimal_step_newton_times_signal

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L227)

解释：当曲率 h 不为零时，上述步长代数上等于逆曲率乘以梯度信号功率占信号加噪声功率的比例。此式只是恒等式，不能保证现实中的估计量准确。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (h e σ2 : ℝ), h ≠ 0 → h * e ^ 2 / (h ^ 2 * e ^ 2 + σ2) = 1 / h * ((h * e) ^ 2 / ((h * e) ^ 2 + σ2))
```

## CGB.GradientFilter.no_fixed_step_two_distances

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L233)

解释：在正曲率和正噪声、两个不同非负误差平方下，单步最优系数不同，因此同一个固定系数不能同时等于这两个位置的单步最优。

源码说明：With noise, the optimal step differs at two different distances to the optimum.

形式陈述：

```lean
∀ (h σ2 x y : ℝ), 0 < h → 0 < σ2 → 0 ≤ x → 0 ≤ y → x ≠ y → h * x / (h ^ 2 * x + σ2) ≠ h * y / (h ^ 2 * y + σ2)
```

## CGB.GradientFilter.no_optimal_step_negative_curvature

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L247)

解释：对给定负曲率的一维二次式和非零梯度方向，沿该方向可以取足够大的实数步长使值低于任意给定 M，即无有限全局最小值；有限步长的选择还需其他条件。

源码说明：**W6 (nonconvex data)**: with negative curvature the loss along the step is unbounded below, so no optimal quadratic step exists and the step must be bounded.

形式陈述：

```lean
∀ (h e g M : ℝ), h < 0 → g ≠ 0 → ∃ η, h * (e - η * g) ^ 2 / 2 < M
```

## CGB.GradientFilter.no_fixed_bound_two_radii

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L292)

解释：对于定义的理想进展函数，两个不同正有效半径无法由同一个固定步长上限同时获得各自的全部进展。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (ρ R₁ R₂ : ℝ), 0 < R₁ → R₁ < R₂ → CGB.GradientFilter.progress ρ R₁ < R₁ ∨ CGB.GradientFilter.progress ρ R₂ < R₂
```

## CGB.GradientFilter.bound_stays_in_band

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L306)

解释：若实际有效半径 R 已知且乘除因子 γ>1，理想信任步长规则会把原处于 [R/γ,Rγ] 的上限保持在该区间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (γ R ρ : ℝ),
  1 < γ →
    0 < R →
      R / γ ≤ ρ → ρ ≤ R * γ → R / γ ≤ CGB.GradientFilter.trustStep γ R ρ ∧ CGB.GradientFilter.trustStep γ R ρ ≤ R * γ
```

## CGB.GradientFilter.newton_step_exact

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L324)

解释：在一维二次方向上，用非零精确曲率的倒数乘精确梯度更新一次，误差恰为零。

源码说明：The exact curvature step reaches the optimum of a quadratic direction.

形式陈述：

```lean
∀ (h e : ℝ), h ≠ 0 → e - 1 / h * (h * e) = 0
```

## CGB.GradientFilter.momentum_after_exact_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L330)

解释：一维精确牛顿步已到达最优点时，非零重球动量会把原本非零的误差重新带离零点。

源码说明：After an exact step, momentum `β ≠ 0` (heavy ball: next error `β (e₁ - e₀)` with `e₁ = 0`) moves a direction that started away from the optimum off it again.

形式陈述：

```lean
∀ (h e₀ β : ℝ),
  h ≠ 0 →
    e₀ ≠ 0 →
      β ≠ 0 → e₀ - 1 / h * (h * e₀) - 1 / h * (h * (e₀ - 1 / h * (h * e₀))) + β * (e₀ - 1 / h * (h * e₀) - e₀) ≠ 0
```

## CGB.GradientFilter.gain_error_square

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L354)

解释：当 P+r>0 时，单次增益估计误差可分解为最小误差和偏离最优增益 P/(P+r) 的非负平方项。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (P r K : ℝ), 0 < P + r → CGB.GradientFilter.gainErr P r K = P * r / (P + r) + (P + r) * (K - P / (P + r)) ^ 2
```

## CGB.GradientFilter.steady_gain_worse_for_fresh

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L362)

解释：在正噪声和两个不同正先验方差下，用另一个先验的稳态增益会比该方向自己的最优增益产生严格更大的单步估计误差。

源码说明：The steady gain `P* / (P* + r)` is strictly worse than the evidence-dependent gain for a direction whose prior variance is different (larger: a fresh or rare direction).

形式陈述：

```lean
∀ (P Ps r : ℝ),
  0 < r →
    0 < P →
      0 < Ps → P ≠ Ps → CGB.GradientFilter.gainErr P r (P / (P + r)) < CGB.GradientFilter.gainErr P r (Ps / (Ps + r))
```

## CGB.GradientFilter.in_sample_decrease_bias

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L400)

解释：在指定的一维带噪二次模型里，同一批次评估出的下降量减去无噪真实下降量等于 ηξ(he+ξ)。

源码说明：Decrease measured on the batch that made the step minus the decrease on fresh data.

形式陈述：

```lean
∀ (h e ξ η : ℝ),
  h * e ^ 2 / 2 + ξ * e - (h * (e - η * (h * e + ξ)) ^ 2 / 2 + ξ * (e - η * (h * e + ξ))) -
      (h * e ^ 2 / 2 - h * (e - η * (h * e + ξ)) ^ 2 / 2) =
    η * ξ * (h * e + ξ)
```

## CGB.GradientFilter.in_sample_decrease_bias_mean

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L405)

解释：当步长为正且噪声非零时，对正负噪声配对平均，同批次下降量的偏差严格为正。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (h e ξ η : ℝ), 0 < η → ξ ≠ 0 → 0 < (η * ξ * (h * e + ξ) + η * -ξ * (h * e + -ξ)) / 2
```

## CGB.GradientFilter.own_step_biases_noise_estimate

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L412)

解释：在指定的对称噪声配对下，直接用两次梯度差估计滞后噪声时会混入自身更新造成的 −h²ab 项。

源码说明：Lag-one noise estimate averaged over the noise pair `(x, y)`, `(-x, -y)`.

形式陈述：

```lean
∀ (h a b x y : ℝ), (-((h * a + x) * (h * b + y)) + -((h * a - x) * (h * b - y))) / 2 = -(x * y) - h ^ 2 * (a * b)
```

## CGB.GradientFilter.residual_removes_own_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L416)

解释：在给定线性模型中，从两个梯度差中减去各自精确的自身步长效应后，剩余乘积恰为噪声项 −xy。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (h a b x y : ℝ), -((h * a + x - h * a) * (h * b + y - h * b)) = -(x * y)
```

## CGB.GradientFilter.secant_needs_diagonal

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L420)

解释：在有跨方向曲率 c 的二维模型中，单坐标割线值包含另一个坐标步长比造成的 c·a₂/a₁ 偏差。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (h₁ c a₁ a₂ : ℝ), a₁ ≠ 0 → (h₁ * a₁ + c * a₂) / a₁ = h₁ + c * (a₂ / a₁)
```

## CGB.GradientFilter.secant_exact_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L424)

解释：二维模型的单坐标割线对所有非零第一坐标步长都等于该坐标曲率，当且仅当跨方向曲率为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (h₁ c : ℝ), (∀ (a₁ a₂ : ℝ), a₁ ≠ 0 → (h₁ * a₁ + c * a₂) / a₁ = h₁) ↔ c = 0
```

## CGB.GradientFilter.fresh_decrease_unbiased

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L452)

解释：在所给一维模型中，对独立评估噪声的正负配对求平均，评估批次测得的下降量等于无噪下降量。

源码说明：Fresh-batch decrease averaged over the unseen noise pair `±ξ'`.

形式陈述：

```lean
∀ (h e u ξ' : ℝ),
  (h * e ^ 2 / 2 + ξ' * e - (h * (e - u) ^ 2 / 2 + ξ' * (e - u)) +
        (h * e ^ 2 / 2 + -ξ' * e - (h * (e - u) ^ 2 / 2 + -ξ' * (e - u)))) /
      2 =
    h * e ^ 2 / 2 - h * (e - u) ^ 2 / 2
```

## CGB.GradientFilter.estimated_curvature_residual_bias

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L458)

解释：用估计曲率扣除自身步长后，对称噪声配对的残余偏差为 −(h−ĥ)²ab，反映曲率估计误差的影响。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (h ĥ a b x y : ℝ),
  (-((h * a + x - ĥ * a) * (h * b + y - ĥ * b)) + -((h * a - x - ĥ * a) * (h * b - y - ĥ * b))) / 2 =
    -(x * y) - (h - ĥ) ^ 2 * (a * b)
```

## CGB.GradientFilter.eigenbasis_decouples

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L463)

解释：若 w 是矩阵 S 的特征向量且 v 与 w 正交，则 v 与 S·w 的内积为零，给出两个方向交叉项消失的条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {n : ℕ} (S : Matrix (Fin n) (Fin n) ℝ) (v w : Fin n → ℝ) (μ : ℝ),
  S.mulVec w = μ • w → v ⬝ᵥ w = 0 → v ⬝ᵥ S.mulVec w = 0
```

## CGB.GradientFilter.trust_with_deadband_stays_in_band

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L471)

解释：理想信任上限规则加入无证据时保持不变的死区后，原先的 [R/γ,Rγ] 区间仍保持不变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (γ R ρ : ℝ) (ev : Bool),
  1 < γ →
    0 < R →
      R / γ ≤ ρ →
        ρ ≤ R * γ → R / γ ≤ CGB.GradientFilter.trustStep3 γ R ρ ev ∧ CGB.GradientFilter.trustStep3 γ R ρ ev ≤ R * γ
```

## CGB.GradientFilter.linGain_closed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L495)

解释：固定增益反复接收相同值 d 时，估计值等于 d 加上初始偏差乘以 (1−K)^n。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (K v d : ℝ) (n : ℕ), CGB.GradientFilter.linGain K v d n = d + (1 - K) ^ n * (v - d)
```

## CGB.GradientFilter.linear_damage_eq_adoption

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L500)

解释：固定增益对相同长度恒定输入的归一化响应恰为 1−(1−K)^n，未来是否回转不会改变该前缀响应。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (K v d : ℝ) (n : ℕ), v ≠ d → (CGB.GradientFilter.linGain K v d n - v) / (d - v) = 1 - (1 - K) ^ n
```

## CGB.GradientFilter.no_linear_gain_separates

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L507)

解释：对于同一响应量和 ε<1/2，不可能同时要求响应不大于 ε 又不小于 1−ε，说明该前缀上保护和采纳目标冲突。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (K ε : ℝ) (n : ℕ), ε < 1 / 2 → ¬(1 - (1 - K) ^ n ≤ ε ∧ 1 - ε ≤ 1 - (1 - K) ^ n)
```
