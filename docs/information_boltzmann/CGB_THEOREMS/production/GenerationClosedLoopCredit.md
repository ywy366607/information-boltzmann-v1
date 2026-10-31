# GenerationClosedLoopCredit

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean) · [返回导航](../README.md)

模块源码说明：Closed-loop credit with a shared policy and predictor G15 (`Medium.expected_reward_deriv`) differentiates a fixed reward table.

## CGB.ClosedLoopCredit.objective_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L43)

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

源码说明：Differentiating normalization forces the policy tangents to sum to zero.

形式陈述：

```lean
∀ {A : Type u_1} [inst : Fintype A] (π : ℝ → A → ℝ) (θ₀ : ℝ) (π' : A → ℝ),
  (∀ (a : A), HasDerivAt (fun θ => π θ a) (π' a) θ₀) → (∀ (θ : ℝ), ∑ a, π θ a = 1) → ∑ a, π' a = 0
```

## CGB.ClosedLoopCredit.scalar_baseline_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L72)

源码说明：A scalar baseline cancels against a normalized conditional outcome law. An action/outcome-dependent baseline cannot use this lemma: its conditional mean must instead be shown identical across actions (or cancellation proved).

形式陈述：

```lean
∀ {A : Type u_1} {O : Type u_2} [inst : Fintype A] [inst_1 : Fintype O] (K : A → O → ℝ) (π' : A → ℝ) (b : ℝ),
  (∀ (a : A), ∑ o, K a o = 1) → ∑ a, π' a = 0 → ∑ a, ∑ o, K a o * (π' a * b) = 0
```

## CGB.ClosedLoopCredit.objective_deriv_advantage

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L82)

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

源码说明：For a positive policy, the ratio appearing above really is the log score.

形式陈述：

```lean
∀ {A : Type u_1} (π : ℝ → A → ℝ) (θ₀ : ℝ) (π' : A → ℝ),
  (∀ (a : A), HasDerivAt (fun θ => π θ a) (π' a) θ₀) →
    (∀ (a : A), 0 < π θ₀ a) → ∀ (a : A), HasDerivAt (fun θ => Real.log (π θ a)) (π' a / π θ₀ a) θ₀
```

## CGB.ClosedLoopCredit.missing_score_derivative_counterexample

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L114)

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

源码说明：Constant positive likelihood leaves a normalized prior unchanged. This is conditional on the current body/action: a pure command copy has no new evidence about an unknown world if its likelihood is independent of that unknown state. It does NOT exclude learning from reafference whose likelihood depends on an unknown body/world, or training a predictor to reproduce a known command.

形式陈述：

```lean
∀ {W : Type u_3} [inst : Fintype W] (prior : W → ℝ),
  ∑ w, prior w = 1 → ∀ (c : ℝ), 0 < c → ∀ (w : W), prior w * c / ∑ v, prior v * c = prior w
```
