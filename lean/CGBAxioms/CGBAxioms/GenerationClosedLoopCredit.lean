import CGBAxioms.Generation

/-!
# Closed-loop credit with a shared policy and predictor

G15 (`Medium.expected_reward_deriv`) differentiates a fixed reward table. Here
`π θ a` and the COMPLETE score `R θ a o` share one parameter coordinate `θ`.
Condition on the current body and past; `a` is an intervention record and `o`
contains the scored observations, including self-hearing and scoring samples.
The derivative premises are TOTAL derivatives of these supplied functions:
learning, state recursion, sample sensitivity and encoder/query dependence inside
`R` must not be silently frozen. Differentiability is a substantive premise.
Discrete sampling effects belong in the joint law, not a frozen sample derivative.

A fixed `K a o` is justified only if `a` is a sufficiently complete causal boundary
record: given that record, the conditional world law has no further θ-dependence.
If it still has `Kθ`, the derivative also contains `Σ π K' R`; the fixed-kernel
results do not apply. A boundary refinement must retain its actual joint law.
These are immediate/finite-trajectory interventional objectives, not a claim that
changing today's parameter retroactively replaces all parameters in the past.
A finite trajectory may be encoded by `a,o` only if this factorization holds.

The identities do not supply exact local eligibility for every GDN encoder/query,
nor remove the time/storage cost of its state Jacobian. A runtime must establish
the total-derivative premises for its actual learning and scoring procedure.
The algebra also holds for signed weights; probability interpretations require
nonnegative π and K in addition to their stated normalization conditions.
-/

open Finset

namespace CGB.ClosedLoopCredit

variable {A O : Type*} [Fintype A] [Fintype O]

/-- Expected complete score in the same causal model as the action law. -/
def objective (π : ℝ → A → ℝ) (K : A → O → ℝ)
    (R : ℝ → A → O → ℝ) (θ : ℝ) : ℝ :=
  ∑ a, π θ a * ∑ o, K a o * R θ a o

/-- Product rule for the shared policy/predictor, including direct score credit.
No derivative of the objective is assumed: only component total derivatives. -/
theorem objective_deriv (π : ℝ → A → ℝ) (K : A → O → ℝ)
    (R : ℝ → A → O → ℝ) (θ₀ : ℝ) (π' : A → ℝ) (R' : A → O → ℝ)
    (hπ : ∀ a, HasDerivAt (fun θ => π θ a) (π' a) θ₀)
    (hR : ∀ a o, HasDerivAt (fun θ => R θ a o) (R' a o) θ₀) :
    HasDerivAt (objective π K R)
      (∑ a, ∑ o, K a o * (π' a * R θ₀ a o + π θ₀ a * R' a o)) θ₀ := by
  have h := HasDerivAt.fun_sum (u := univ) (fun a _ =>
    (hπ a).mul (HasDerivAt.fun_sum (u := univ) (fun o _ =>
      (hR a o).const_mul (K a o))))
  apply h.congr_deriv
  apply sum_congr rfl
  intro a _
  rw [mul_sum, mul_sum, ← sum_add_distrib]
  apply sum_congr rfl
  intro o _
  ring

/-- Differentiating normalization forces the policy tangents to sum to zero. -/
theorem policy_deriv_sum_zero (π : ℝ → A → ℝ) (θ₀ : ℝ) (π' : A → ℝ)
    (hπ : ∀ a, HasDerivAt (fun θ => π θ a) (π' a) θ₀)
    (hnorm : ∀ θ, ∑ a, π θ a = 1) : ∑ a, π' a = 0 := by
  have h := HasDerivAt.fun_sum (u := univ) (fun a _ => hπ a)
  have heq : (fun θ => ∑ a, π θ a) = (fun _ : ℝ => 1) := funext hnorm
  rw [heq] at h
  exact h.unique (hasDerivAt_const θ₀ 1)

/-- A scalar baseline cancels against a normalized conditional outcome law.
An action/outcome-dependent baseline cannot use this lemma: its conditional
mean must instead be shown identical across actions (or cancellation proved). -/
theorem scalar_baseline_zero (K : A → O → ℝ) (π' : A → ℝ) (b : ℝ)
    (hK : ∀ a, ∑ o, K a o = 1) (hzero : ∑ a, π' a = 0) :
    (∑ a, ∑ o, K a o * (π' a * b)) = 0 := by
  simp_rw [← sum_mul, hK, one_mul]
  rw [← sum_mul, hzero, zero_mul]

/-- Three-factor advantage/score credit PLUS the complete direct score derivative.
`baseline θ₀` may depend on θ, the conditioned body and the past. Only its current
scalar value is used; this is not differentiation of a baseline-subtracted loss.
Consequently no derivative premise for `baseline` is required. -/
theorem objective_deriv_advantage (π : ℝ → A → ℝ) (K : A → O → ℝ)
    (R : ℝ → A → O → ℝ) (θ₀ : ℝ) (π' : A → ℝ) (R' : A → O → ℝ)
    (baseline : ℝ → ℝ)
    (hπ : ∀ a, HasDerivAt (fun θ => π θ a) (π' a) θ₀)
    (hR : ∀ a o, HasDerivAt (fun θ => R θ a o) (R' a o) θ₀)
    (hnorm : ∀ θ, ∑ a, π θ a = 1)
    (hK : ∀ a, ∑ o, K a o = 1) (hpos : ∀ a, 0 < π θ₀ a) :
    HasDerivAt (objective π K R)
      ((∑ a, ∑ o, π θ₀ a * K a o *
          ((R θ₀ a o - baseline θ₀) * (π' a / π θ₀ a))) +
        ∑ a, ∑ o, π θ₀ a * K a o * R' a o) θ₀ := by
  apply (objective_deriv π K R θ₀ π' R' hπ hR).congr_deriv
  have hz := scalar_baseline_zero K π' (baseline θ₀) hK
    (policy_deriv_sum_zero π θ₀ π' hπ hnorm)
  have heq : ∀ a o, K a o * (π' a * R θ₀ a o + π θ₀ a * R' a o) =
      π θ₀ a * K a o * ((R θ₀ a o - baseline θ₀) * (π' a / π θ₀ a)) +
        π θ₀ a * K a o * R' a o + K a o * (π' a * baseline θ₀) := by
    intro a o
    field_simp [(hpos a).ne']; ring
  simp_rw [heq, sum_add_distrib]
  rw [hz, add_zero]

omit [Fintype A] in
/-- For a positive policy, the ratio appearing above really is the log score. -/
theorem log_policy_deriv (π : ℝ → A → ℝ) (θ₀ : ℝ) (π' : A → ℝ)
    (hπ : ∀ a, HasDerivAt (fun θ => π θ a) (π' a) θ₀)
    (hpos : ∀ a, 0 < π θ₀ a) (a : A) :
    HasDerivAt (fun θ => Real.log (π θ a)) (π' a / π θ₀ a) θ₀ :=
  (hπ a).log (hpos a).ne'

/-- Singleton action/outcome, π=K=1, Rθ=θ: both laws are fixed and normalized;
the policy-only estimator is zero while the actual objective derivative is one. -/
theorem missing_score_derivative_counterexample (θ₀ b : ℝ) :
    (∀ _θ : ℝ, ∑ _a : Unit, (1 : ℝ) = 1) ∧
    (∀ _a : Unit, ∑ _o : Unit, (1 : ℝ) = 1) ∧
    HasDerivAt
      (objective (fun _ (_ : Unit) => 1) (fun _ (_ : Unit) => 1)
        (fun θ _ _ => θ)) 1 θ₀ ∧
    (∑ _a : Unit, ∑ _o : Unit,
      (1 : ℝ) * 1 * ((θ₀ - b) * (0 / 1))) = 0 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro θ; simp
  · intro a; simp
  · convert hasDerivAt_id θ₀ using 1
    funext θ
    simp [objective]
  · simp

/-- Constant positive likelihood leaves a normalized prior unchanged. This is
conditional on the current body/action: a pure command copy has no new evidence
about an unknown world if its likelihood is independent of that unknown state.
It does NOT exclude learning from reafference whose likelihood depends on an
unknown body/world, or training a predictor to reproduce a known command. -/
theorem constant_likelihood_posterior {W : Type*} [Fintype W]
    (prior : W → ℝ) (hnorm : ∑ w, prior w = 1)
    (c : ℝ) (hc : 0 < c) (w : W) :
    prior w * c / (∑ v, prior v * c) = prior w := by
  rw [← sum_mul, hnorm, one_mul]
  exact mul_div_cancel_right₀ (prior w) hc.ne'

end CGB.ClosedLoopCredit
