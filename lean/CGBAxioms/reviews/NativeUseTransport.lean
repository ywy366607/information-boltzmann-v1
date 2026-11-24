import Mathlib.Tactic

/-! Actual causal_use_transport.py mapping: a query identity is fixed before
action. At the end of the event, `old`, `append`, `committed` are log_score
of mechanism with the old support, old+arrived evidence, and the SQL-committed
support, respectively. Each fit excludes the query fact; all three share the
current encoder/target. `initial` is separately scored before action.
The real U calls retain_constraints and commit with a finite fact budget.
These identities apply to arbitrary real body scores and arbitrary query sets.
-/
namespace CGB.NativeUseTransport
noncomputable section

structure QueryScores where
  initial : ℝ
  old : ℝ
  append : ℝ
  committed : ℝ

def appendGain (s : QueryScores) := s.append - s.old
def releaseEffect (s : QueryScores) := s.committed - s.append
def actualGain (s : QueryScores) := s.committed - s.old
def representationShift (s : QueryScores) := s.old - s.initial

theorem actual_U_decomposition (s : QueryScores) :
    actualGain s = appendGain s + releaseEffect s := by
  simp only [actualGain, appendGain, releaseEffect]; ring

theorem action_to_committed_decomposition (s : QueryScores) :
    s.committed - s.initial = representationShift s + actualGain s := by
  simp only [representationShift, actualGain]; ring

theorem budget_cost_reverses_append (s : QueryScores)
    (h : appendGain s < -releaseEffect s) : actualGain s < 0 := by
  rw [actual_U_decomposition]; linarith

theorem same_U_ranking {A : Type*} (s : A → QueryScores) (a b : A) :
    actualGain (s b) < actualGain (s a) ↔
      appendGain (s b) + releaseEffect (s b) <
      appendGain (s a) + releaseEffect (s a) := by
  simp only [actual_U_decomposition]

theorem weighted_queries {Q : Type*} [Fintype Q]
    (weight : Q → ℝ) (s : Q → QueryScores) :
    (∑ q, weight q * actualGain (s q)) =
      (∑ q, weight q * appendGain (s q)) +
      (∑ q, weight q * releaseEffect (s q)) := by
  simp_rw [actual_U_decomposition, mul_add, Finset.sum_add_distrib]

end
end CGB.NativeUseTransport
