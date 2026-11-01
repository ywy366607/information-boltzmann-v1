# GenerationClosedLoopCredit

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean) · [返回导航](../README.md)

模块源码说明：Closed-loop credit with a shared policy and predictor G15 (`Medium.expected_reward_deriv`) differentiates a fixed reward table.

## CGB.ClosedLoopCredit.objective_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L43)

解释：输入参数化策略、环境转移核与奖赏损失函数，经乘积求导法则，输出闭环目标函数的全导数严格分解为策略探索导数项与价值直接导数项之和。

源码说明：Product rule for the shared policy/predictor, including direct score credit. No derivative of the objective is assumed: only component total derivatives.

形式陈述：

```lean
∀ {A : Type u_1} {O : Type u_2} [inst : Fintype A] [inst_1 : Fintype O] (π : ℝ → A → ℝ) (K : A → O → ℝ)
  (R : ℝ → A → O → ℝ) (θ₀ : ℝ) (π' : A → ℝ) (R' : A → O → ℝ),
  (∀ (a : A), HasDerivAt (fun θ => π θ a) (π' a) θ₀) →
    (∀ (a : A) (o : O), HasDerivAt (fun θ => R θ a o) (R' a o) θ₀) →
      HasDerivAt (CGB.ClosedLoopCredit.objective π K R) (∑ a, ∑ o, K a o * (π' a * R θ₀ a o + π θ₀ a * R' a o)) θ₀
```

## CGB.ClosedLoopCredit.policy_deriv_sum_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L61)

解释：输入概率归一化的参数化策略，经对全动作求和并交换求导次序，输出策略导数在全动作空间上的积分严格为零。

源码说明：Differentiating normalization forces the policy tangents to sum to zero.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (π : ℝ → A → ℝ) (θ₀ : ℝ) (π' : A → ℝ),
  (∀ (a : A), HasDerivAt (fun θ => π θ a) (π' a) θ₀) → (∀ (θ : ℝ), ∑ a, π θ a = 1) → ∑ a, π' a = 0
```

## CGB.ClosedLoopCredit.scalar_baseline_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L72)

解释：输入条件概率核、全空间积分为零的策略导数与常数基线，经双重求和展开，输出常数基线与策略导数的乘积在全空间期望值恒等于零。

源码说明：A scalar baseline cancels against a normalized conditional outcome law. An action/outcome-dependent baseline cannot use this lemma: its conditional mean must instead be shown identical across actions (or cancellation proved).

形式陈述：

```lean
∀ {A : Type u_1} {O : Type u_2} [inst : Fintype A] [inst_1 : Fintype O] (K : A → O → ℝ) (π' : A → ℝ) (b : ℝ),
  (∀ (a : A), ∑ o, K a o = 1) → ∑ a, π' a = 0 → ∑ a, ∑ o, K a o * (π' a * b) = 0
```

## CGB.ClosedLoopCredit.objective_deriv_advantage

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L82)

解释：输入参数化策略、环境核、目标值与任意标量基线函数，经优势重整化变换，输出闭环导数在标量基线减免下保持无偏，精确表示为优势加权的得分三因子项与直接得分导数项。

源码说明：Three-factor advantage/score credit PLUS the complete direct score derivative. `baseline θ₀` may depend on θ, the conditioned body and the past. Only its current scalar value is used; this is not differentiation of a baseline-subtracted loss. Consequently no derivative premise for `baseline` is required.

形式陈述：

```lean
∀ {A : Type u_1} {O : Type u_2} [inst : Fintype A] [inst_1 : Fintype O] (π : ℝ → A → ℝ) (K : A → O → ℝ)
  (R : ℝ → A → O → ℝ) (θ₀ : ℝ) (π' : A → ℝ) (R' : A → O → ℝ) (baseline : ℝ → ℝ),
  (∀ (a : A), HasDerivAt (fun θ => π θ a) (π' a) θ₀) →
    (∀ (a : A) (o : O), HasDerivAt (fun θ => R θ a o) (R' a o) θ₀) →
      (∀ (θ : ℝ), ∑ a, π θ a = 1) →
        (∀ (a : A), ∑ o, K a o = 1) →
          (∀ (a : A), 0 < π θ₀ a) →
            HasDerivAt (CGB.ClosedLoopCredit.objective π K R)
              (∑ a, ∑ o, π θ₀ a * K a o * ((R θ₀ a o - baseline θ₀) * (π' a / π θ₀ a)) +
                ∑ a, ∑ o, π θ₀ a * K a o * R' a o)
              θ₀
```

## CGB.ClosedLoopCredit.log_policy_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L106)

解释：输入可微的正值策略分布，经复合函数对数求导法则，输出策略对数概率对参数的导数严格等于策略导数与当前策略概率的比值（得分函数标准形式）。

源码说明：For a positive policy, the ratio appearing above really is the log score.

形式陈述：

```lean
∀ {A : Type u_1} (π : ℝ → A → ℝ) (θ₀ : ℝ) (π' : A → ℝ),
  (∀ (a : A), HasDerivAt (fun θ => π θ a) (π' a) θ₀) →
    (∀ (a : A), 0 < π θ₀ a) → ∀ (a : A), HasDerivAt (fun θ => Real.log (π θ a)) (π' a / π θ₀ a) θ₀
```

## CGB.ClosedLoopCredit.missing_score_derivative_counterexample

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L114)

解释：输入参数基线反例，经闭环目标导数计算，输出若仅计算策略得分项而忽略直接得分导数，则策略估计梯度为零而实际目标导数为一，证实必须引入三因子闭环直接梯度。

源码说明：Singleton action/outcome, π=K=1, Rθ=θ: both laws are fixed and normalized; the policy-only estimator is zero while the actual objective derivative is one.

形式陈述：

```lean
∀ (θ₀ b : ℝ),
  (∀ (_θ : ℝ), ∑ _a, 1 = 1) ∧
    (∀ (_a : Unit), ∑ _o, 1 = 1) ∧
      HasDerivAt (CGB.ClosedLoopCredit.objective (fun x x_1 => 1) (fun x x_1 => 1) fun θ x x_1 => θ) 1 θ₀ ∧
        ∑ _a, ∑ _o, 1 * 1 * ((θ₀ - b) * (0 / 1)) = 0
```

## CGB.ClosedLoopCredit.constant_likelihood_posterior

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L135)

解释：输入归一化先验分布与严格正的常数似然函数，经贝叶斯后验概率归一化变换，输出后验分布严格等于原先验分布（证明常数似然不提供关于未知状态的新证据）。

源码说明：Constant positive likelihood leaves a normalized prior unchanged. This is conditional on the current body/action: a pure command copy has no new evidence about an unknown world if its likelihood is independent of that unknown state. It does NOT exclude learning from reafference whose likelihood depends on an unknown body/world, or training a predictor to reproduce a known command.

形式陈述：

```lean
∀ {W : Type u_3} [inst : Fintype W] (prior : W → ℝ),
  ∑ w, prior w = 1 → ∀ (c : ℝ), 0 < c → ∀ (w : W), prior w * c / ∑ v, prior v * c = prior w
```
