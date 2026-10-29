# GenerationContinualOptimizer

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean) · [返回导航](../README.md)

模块源码说明：A continual optimizer: competition, no winner-take-all, a healthy latch (side branch, 2026-10-27) The optimizer of a stream learner must (i) make units compete so that they differentiate, (ii) keep one unit from taking e

## CGB.Continual.polar_conserves_direction_mass

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L54)

解释：若矩阵列正交且各列范数为一，则每个输入方向对应列的元素平方和恰好为一。这给出正交矩阵上的按方向质量约束；它本身并不证明整个 Muon 更新满足健康锁存或避免赢家通吃。

源码说明：Orthonormal columns carry unit mass along every input direction.

形式陈述：

```lean
∀ {m n : ℕ} (O : Matrix (Fin m) (Fin n) ℝ), O.transpose * O = 1 → ∀ (j : Fin n), ∑ i, O i j ^ 2 = 1
```

## CGB.Continual.payoff_le_two_of_col

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L71)

解释：在 2×2 非负分配矩阵中，每列预算恰好为一时，分配平方和作为集中收益最多为二。这是所指定收益和可行集的界，不是任意规模或任意优化器的性能定理。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Fin 2 → Fin 2 → ℝ}, CGB.Continual.Nonneg A → CGB.Continual.ColBudget A → CGB.Continual.payoff A ≤ 2
```

## CGB.Continual.payoff_le_two_of_row

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L80)

解释：在 2×2 非负分配矩阵中，每行预算恰好为一时，分配平方和最多为二。单独的行约束并未要求不同的行选择不同方向。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Fin 2 → Fin 2 → ℝ}, CGB.Continual.Nonneg A → CGB.Continual.RowBudget A → CGB.Continual.payoff A ≤ 2
```

## CGB.Continual.column_budget_allows_wta

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L89)

解释：构造的 2×2 分配把两个方向都交给同一行，同时满足非负和每列单位预算，集中收益达到二，却不满足行预算。这说明单列预算的可行集合允许赢家占有全部方向；该见证不自动满足 Muon 的正交约束。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Continual.Nonneg CGB.Continual.wta ∧
  CGB.Continual.ColBudget CGB.Continual.wta ∧
    CGB.Continual.payoff CGB.Continual.wta = 2 ∧ ¬CGB.Continual.RowBudget CGB.Continual.wta
```

## CGB.Continual.row_budget_allows_copies

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L97)

解释：构造的 2×2 分配让两行都选择同一个方向，同时满足非负和每行单位预算、达到收益二，却违反列预算。因而单靠行预算不能排除重复分配。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Continual.Nonneg CGB.Continual.copies ∧
  CGB.Continual.RowBudget CGB.Continual.copies ∧
    CGB.Continual.payoff CGB.Continual.copies = 2 ∧ ¬CGB.Continual.ColBudget CGB.Continual.copies
```

## CGB.Continual.doubly_budgeted_form

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L105)

解释：对 2×2 矩阵，如果每行和每列之和都为一，那么四个元素由一个数 p 决定，形式为 [[p,1-p],[1-p,p]]。这个代数形式本身不需要非负条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Fin 2 → Fin 2 → ℝ},
  CGB.Continual.ColBudget A → CGB.Continual.RowBudget A → A 1 1 = A 0 0 ∧ A 0 1 = 1 - A 0 0 ∧ A 1 0 = 1 - A 0 0
```

## CGB.Continual.doubly_budgeted_max_iff_permutation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L110)

解释：在该 2×2 行列双预算问题中，平方和收益达到二当且仅当左上角为零或一，即分配处在两个置换形式之一。这是静态极值条件，不是动态收敛或通用防垄断保证。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Fin 2 → Fin 2 → ℝ},
  CGB.Continual.Nonneg A →
    CGB.Continual.ColBudget A → CGB.Continual.RowBudget A → (CGB.Continual.payoff A = 2 ↔ A 0 0 = 1 ∨ A 0 0 = 0)
```

## CGB.Continual.doubly_budgeted_max_differentiates

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L125)

解释：在 2×2 非负、行列均有单位预算且平方和收益达到二的前提下，两行在同一方向上的分配不同。它证明该特定极值模型里的分化，不保证训练中一定到达极值。

源码说明：At a doubly budgeted maximum the two rows own different directions.

形式陈述：

```lean
∀ {A : Fin 2 → Fin 2 → ℝ},
  CGB.Continual.Nonneg A →
    CGB.Continual.ColBudget A → CGB.Continual.RowBudget A → CGB.Continual.payoff A = 2 → A 0 0 ≠ A 1 0
```

## CGB.Continual.linear_protect_implies_slow

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L133)

解释：在定理给出的固定增益与冲击保护条件下，长度 c 的变化若最多造成 ε 的响应，则持续 m 个同样长度后的累计采纳量有相应上界。这刻画该线性递推中的保护与采纳权衡，并非所有 Adam 或 Muon 轨迹的完整分类。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (K ε : ℝ) (c m : ℕ), ε ≤ 1 → 1 - (1 - K) ^ c ≤ ε → 1 - (1 - K) ^ (m * c) ≤ 1 - (1 - ε) ^ m
```

## CGB.Continual.deferLatch_holds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L145)

解释：对所定义的连续反证计数规则，只要反证连续长度小于阈值 c，输出就保持旧值 v。该结论针对给定规则，并未证明旧值有用或存在物理恢复盆地。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (c run : ℕ) (v d : ℝ), run < c → CGB.Continual.deferLatch c run v d = v
```

## CGB.Continual.deferLatch_releases

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L148)

解释：对同一计数规则，当连续反证长度达到或超过阈值 c 时，输出切换到新值 d；这是在给定证据序列下的释放条件，不说明环境必然会提供该序列。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (c run : ℕ) (v d : ℝ), c ≤ run → CGB.Continual.deferLatch c run v d = d
```

## CGB.Continual.deferLatch_senses

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L153)

解释：当旧值和新值不同且阈值 c 非零时，反证长度为零和达到 c 的两种输入产生不同输出；若 c 为零则落入定理给出的退化分支。这说明规则可对这类证据作区分，不等于通用感知能力。

源码说明：The latch is not blind: the same held state gives different outputs for different evidence runs (H2 is avoided), and every hold ends after `c` contrary steps (H1).

形式陈述：

```lean
∀ (c : ℕ) (v d : ℝ), v ≠ d → CGB.Continual.deferLatch c 0 v d ≠ CGB.Continual.deferLatch c c v d ∨ c = 0
```

## CGB.Continual.held_budget_starves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L161)

解释：在预算 B 中每次永久占用份额 s 的这个模型里，若累计占用 n·s 至少达到 B，那么剩余量 B−n·s 不大于零。它说明在这些假设下无限期占用会耗尽自由预算，而非证明现有代码采用了同样的守恒账本。

源码说明：Consolidated entries that keep a share `s` of a conserved budget `B` and are never released leave `B - n s` free after `n` consolidations; it is gone once `n s ≥ B`.

形式陈述：

```lean
∀ (B s : ℝ) (n : ℕ), B ≤ ↑n * s → B - ↑n * s ≤ 0
```

## CGB.Continual.secant_own_noise_bias

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L174)

解释：在所列四个对称噪声组合中，依赖旧噪声生成的步长使割线乘积平均值比理想曲率项多出 −c·y² 的偏差。它是明确有限求和的代数恒等式，统计解释要额外满足这些噪声和步长关系。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (h s c x y : ℝ),
  ((h * (s + c * y) + x - y) * (s + c * y) + (h * (s + c * y) - x - y) * (s + c * y) +
          (h * (s - c * y) + x + y) * (s - c * y) +
        (h * (s - c * y) - x + y) * (s - c * y)) /
      4 =
    h * (((s + c * y) ^ 2 + (s - c * y) ^ 2) / 2) - c * y ^ 2
```

## CGB.Continual.secant_same_batch_exact

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L180)

解释：在给定线性梯度模型中，同一个噪声 ξ 在新旧参数梯度差里抵消，得到 h·a²。这为同批次测量自身步长造成的梯度变化提供代数依据，但并不保证任意非线性模型都精确。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (h a ξ : ℝ), (h * a + ξ - ξ) * a = h * a ^ 2
```

## CGB.Continual.fresh_decrease_mixture

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L204)

解释：出现与未出现两种条件下的单步下降量按出现率混合后，等于背景下降量加出现率乘以两者之差。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p d₁ d₀ : ℝ), p * d₁ + (1 - p) * d₀ = d₀ + p * (d₁ - d₀)
```

## CGB.Continual.conditional_prediction_excess

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L207)

解释：仅用出现时的下降量预测混合平均下降量，其差额恰为未出现概率乘以两种条件下降量的差。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p d₁ d₀ : ℝ), d₁ - (p * d₁ + (1 - p) * d₀) = (1 - p) * (d₁ - d₀)
```

## CGB.Continual.shared_reference_flips

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L211)

解释：在满足给定正性条件时，仅增大另一行的幅度就能使当前行从通过共享相对阈值变为不通过。

源码说明：Shared reference: row `x` occurs when `x ≥ f · max x y`. Growing the other row erases it.

形式陈述：

```lean
∀ (f x : ℝ), 0 < f → 0 < x → f * x ≤ x → x ≥ f * max x x ∧ ¬x ≥ f * max x (2 * x / f)
```

## CGB.Continual.expected_gradient_is_mixture

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L241)

解释：事件和背景两种梯度按出现率混合，可写为背景梯度加出现率乘以事件与背景的差。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (p g₁ g₀ : ℝ), p * g₁ + (1 - p) * g₀ = g₀ + p * (g₁ - g₀)
```

## CGB.Continual.positive_part_bias

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L246)

解释：在指定的四点零信号构造中，先取正部再平均得到 P/2，虽然未截断的估计平均为零，展示逐次截断带来的偏差。

源码说明：Zero signal, calibrated variance: `m` is `0, 0, c, -c` with `c^2 = 2P`, so `E[m^2] = P`. The average of `m^2 - P` is `0`; the average of the per-step plug-in `max(m^2 - P, 0)` is `P / 2`.

形式陈述：

```lean
∀ (P : ℝ),
  0 < P →
    (0 - P + (0 - P) + (2 * P - P) + (2 * P - P)) / 4 = 0 ∧
      (max (0 - P) 0 + max (0 - P) 0 + max (2 * P - P) 0 + max (2 * P - P) 0) / 4 = P / 2
```

## CGB.Continual.consistent_prediction_excess

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L253)

解释：当收缩系数不大于一且 m·u 非负时，用未收缩均值预测下降会比用收缩均值多出非负项 (1−f)mu。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (m u f h : ℝ),
  f ≤ 1 → 0 ≤ m * u → m * u - h * u ^ 2 / 2 - (f * m * u - h * u ^ 2 / 2) = (1 - f) * (m * u) ∧ 0 ≤ (1 - f) * (m * u)
```

## CGB.Continual.softmax_gradients_sum_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L259)

解释：二分类 softmax 的两个类别梯度按定义相加恰为零。

源码说明：Two-class softmax: the class-row gradients `(p - y)` and `((1 - p) - (1 - y))` sum to zero.

形式陈述：

```lean
∀ (p y : ℝ), p - y + (1 - p - (1 - y)) = 0
```

## CGB.Continual.softmax_rows_coupled

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L263)

解释：二分类概率严格位于零和一之间时，softmax 曲率的跨类别项 −p(1−p) 非零。

源码说明：The output factor of the softmax curvature, `diag p - p pᵀ` for two classes, has the off-diagonal entry `-p (1 - p)`, nonzero for `0 < p < 1`.

形式陈述：

```lean
∀ (p : ℝ), 0 < p → p < 1 → -(p * (1 - p)) ≠ 0
```

## CGB.Continual.ema_forgets_rare_spikes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L290)

解释：衰减因子 0.95 经过 200 个时间步后的残留比例小于 1/200，说明这个具体 EMA 窗口难以保留稀疏尖峰。

源码说明：The registered window (factor 0.95) with spikes 200 steps apart: the held share 0.95^200 is below the spike's share 1/200 of the long-run power.

形式陈述：

```lean
(19 / 20) ^ 200 < 1 / 200
```

## CGB.Continual.partition_is_unbiased

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L292)

解释：有限加权和按任意布尔事件拆成事件项与非事件项后，两部分之和严格等于原和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {ι : Type u_1} (s : Finset ι) (w g : ι → ℝ) (ev : ι → Bool),
  ∑ i ∈ s, w i * g i = (∑ i ∈ s, w i * if ev i = true then g i else 0) + ∑ i ∈ s, w i * if ev i = true then 0 else g i
```

## CGB.Continual.event_part_is_rate_times_size

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L299)

解释：当批量 B 非零时，事件数量 n 乘事件平均幅度 μ 再除以 B，等于事件率 n/B 乘幅度。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n B μ : ℝ), B ≠ 0 → n * μ / B = n / B * μ
```

## CGB.Continual.frozen_hold_loses_on_drift

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L327)

解释：若每步误差增长量 q 为正，则冻结估计的线性增长误差最终会超过任何有限阈值 M。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (e₀ q M : ℝ), 0 < q → ∃ n, M < e₀ + ↑n * q
```

## CGB.Continual.mixture_update_bounded

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L333)

解释：当 c、a 和输入 y 均为正时，所定义的混合更新 y/(1+c·exp(ay²)) 小于 1/(cay)，从而限制大正输入的影响。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (c a y : ℝ), 0 < c → 0 < a → 0 < y → y / (1 + c * Real.exp (a * y ^ 2)) < 1 / (c * a * y)
```

## CGB.Continual.mixture_weight_at_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L343)

解释：所定义的混合权重在输入为零时化简为 1/(1+c)。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (c a : ℝ), 1 / (1 + c * Real.exp (a * 0 ^ 2)) = 1 / (1 + c)
```

## CGB.Continual.release_threshold_monotone_in_age

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L346)

解释：若先验赔率随年龄不增且证据因子非负，较老状态在 k 步达到释放条件时，较年轻状态也已达到该条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (o : ℕ → ℝ), Antitone o → ∀ (L : ℝ) (k T₁ T₂ : ℕ), 0 ≤ L → T₁ ≤ T₂ → 1 ≤ o T₂ * L ^ k → 1 ≤ o T₁ * L ^ k
```

## CGB.Continual.succession_odds_decreasing

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L352)

解释：自然数年龄 T 增加时，赔率 1/(T+1) 不增，提供一个随年龄递减的具体先验例子。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
Antitone fun T => 1 / (↑T + 1)
```

## CGB.Continual.linear_burst_damage_integral

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L374)

解释：若线性系统的阶跃响应从 N 时刻起恒为一，则长度 L 的突发响应在指定有限时段内的有符号总和恰为 L。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (a : ℕ → ℝ) (N L : ℕ),
  (∀ (n : ℕ), N ≤ n → a n = 1) → ∑ n ∈ Finset.range (N + L), CGB.Continual.burstResponse a L n = ↑L
```

## CGB.Continual.beta_hazard_decreasing

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L410)

解释：对于正 α、非负 β，函数 α/(α+β+T) 随自然数年龄 T 不增。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (α β : ℝ), 0 < α → 0 ≤ β → Antitone fun T => α / (α + β + ↑T)
```

## CGB.Continual.beta_hazard_succession

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L417)

解释：当两个 Beta 参数均取一时，给定风险函数化简为 1/(T+2)。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (T : ℕ), 1 / (1 + 1 + ↑T) = 1 / (↑T + 2)
```

## CGB.Continual.accepted_run_variance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L420)

解释：在 k>0 时，k 个相同方差项之和除以 k² 等于 s/k，可用于独立同方差均值的方差计算。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : ℝ) (k : ℕ), 0 < k → (∑ _i ∈ Finset.range k, s) / ↑k ^ 2 = s / ↑k
```

## CGB.Continual.posterior_shrink_error

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L440)

解释：对于均值 m、由两个对称点构成的方差 P 的后验，动作 fm 的平均平方误差恰为 P+(1−f)²m²。

源码说明：Posterior of the gradient as the two points `m ± √P` (mean `m`, variance `P`): the mean squared error of the action `f m` is `P + (1 - f)^2 m^2`.

形式陈述：

```lean
∀ (m P f : ℝ), 0 ≤ P → ((m + √P - f * m) ^ 2 + (m - √P - f * m) ^ 2) / 2 = P + (1 - f) ^ 2 * m ^ 2
```

## CGB.Continual.posterior_mean_is_bayes_action

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L445)

解释：若 m 非零且 f 不等于一，指定二次损失下直接采用后验均值的误差严格小于再次乘 f 的误差。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (m P f : ℝ), m ≠ 0 → f ≠ 1 → P + (1 - 1) ^ 2 * m ^ 2 < P + (1 - f) ^ 2 * m ^ 2
```

## CGB.Continual.newton_contracts_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L469)

解释：正曲率一维二次问题中，用正估计曲率进行精确梯度牛顿步时，误差严格收缩当且仅当估计曲率大于真实曲率的一半。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (h ĥ : ℝ), 0 < h → 0 < ĥ → (|1 - h / ĥ| < 1 ↔ h / 2 < ĥ)
```

## CGB.Continual.overestimate_keeps_contraction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L484)

解释：真实曲率为正且估计曲率位于 [h,κh] 时，单步误差系数位于零与 1−1/κ 之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (h ĥ κ : ℝ), 0 < h → h ≤ ĥ → ĥ ≤ κ * h → 0 ≤ 1 - h / ĥ ∧ 1 - h / ĥ ≤ 1 - 1 / κ
```

## CGB.Continual.saturating_step_bounded

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L499)

解释：只要 ρ>0，饱和步长 ρ·tanh(u/ρ) 的绝对值不超过 ρ。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (ρ u : ℝ), 0 < ρ → |ρ * Real.tanh (u / ρ)| ≤ ρ
```

## CGB.Continual.pooled_signal_unbiased

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L518)

解释：若每个元素都已满足 y²−P=S 的恒等式，则对这些元素求平均后该等式仍成立；它不证明实际统计量满足前提。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {N : ℕ},
  0 < N → ∀ (y2 P S : Fin N → ℝ), (∀ (i : Fin N), y2 i - P i = S i) → (∑ i, (y2 i - P i)) / ↑N = (∑ i, S i) / ↑N
```

## CGB.Continual.next_value_variance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L523)

解释：两种分支方差 q 和 q+J 按权重 1−h、h 线性混合时，结果代数上等于 q+hJ。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q h J : ℝ), (1 - h) * q + h * (q + J) = q + h * J
```
