import Mathlib.Tactic

/-! Body binding (identified_event_credit.py):
H = mutable hot experience prototype, K = finite identified causal support.
e = actual raw (source, action, mother response, stable event identity).
compress = commit_episode (including averaging); U = existing revise;
score = the original held-out finite-U score with current common coordinates.
Legacy merged events apply identity instead of U to K. The lifted event
transaction applies U to K independently of compression of H. Real-body
tests bind persistence, exact payment, next-boundary labels and FK references.
-/
namespace CGB.NativeEventCreditTransition
noncomputable section

def transition {H K E : Type*} (compress : H → E → H) (revise : K → E → K)
    (s : H × K) (e : E) : H × K := (compress s.1 e, revise s.2 e)

theorem credit_commutes {H K E : Type*} (compress : H → E → H)
    (revise : K → E → K) (s : H × K) (e : E) :
    (transition compress revise s e).2 = revise s.2 e := rfl

theorem compression_independent {H K E : Type*} (c₁ c₂ : H → E → H)
    (u : K → E → K) (s : H × K) (e : E) :
    (transition c₁ u s e).2 = (transition c₂ u s e).2 := rfl

theorem finite_path {H K E : Type*} (c : H → E → H) (u : K → E → K)
    (s : H × K) (es : List E) :
    (es.foldl (transition c u) s).2 = es.foldl u s.2 := by
  induction es generalizing s with
  | nil => rfl
  | cons e es ih => simpa only [List.foldl_cons, ih, credit_commutes]

def gain {K E : Type*} (score : K → ℝ) (u : K → E → K) (k : K) (e : E) :=
  score (u k e) - score k

theorem same_payment {H K E : Type*} (c : H → E → H) (u : K → E → K)
    (score : K → ℝ) (s : H × K) (e : E) :
    score (transition c u s e).2 - score s.2 = gain score u s.2 e := rfl

/-- Existing merge gate pays zero. Its exact discrepancy is the U gain,
including either sign; no positive reward is manufactured by this repair. -/
theorem merge_gate_discrepancy {K E : Type*} (score : K → ℝ)
    (u : K → E → K) (k : K) (e : E) :
    gain score u k e - (score k - score k) = gain score u k e := by ring

/-- Exact finite expectation uses the SAME real event transaction as payment.
The actual actor probabilities may be arbitrary; they are not recall weights. -/
theorem expected_same_transaction {A E K : Type*} [Fintype A] [Fintype E]
    (pi : A → ℝ) (world : A → E → ℝ) (score : K → ℝ)
    (u : K → E → K) (k : K) :
    (∑ a, pi a * ∑ e, world a e * (score (u k e) - score k)) =
    ∑ a, pi a * ∑ e, world a e * gain score u k e := rfl

/-- Logs bind these scalars at the actual next decision. Separates transport,
actor sampling, merge admission and observed-response error without fitting. -/
theorem future_error_decomposition (preview onPolicy selected admitted actual : ℝ) :
    preview - actual = (preview-onPolicy) + (onPolicy-selected) +
      (selected-admitted) + (admitted-actual) := by ring

end
end CGB.NativeEventCreditTransition
