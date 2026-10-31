import CGBAxioms.GenerationGDNLife

/-!
# Future-paid language distinctions

A finite body cannot retain every distinction that *might* be queried.  The
existing `ExternalFutureProbes` contract is deliberately conservative: it
requires preservation for every registered continuation.  This module adds a
world-side distribution and a language prediction law, so a distinction only
has a retention claim when it has positive realised continuation mass and
improves the same language cross-entropy used by predictive harvest.

This is not a reward channel, food account, or persistent reserve field.  The
weights and future laws are an evaluation contract for a language world.  The
body still carries only its ordinary GDN state.  Finite capacity and the
selective-export theorems decide that not every paid distinction can coexist;
this module supplies the missing criterion for comparing them.
-/

namespace CGB.GDNLife.Process

open Finset

/-- A finite family of external language continuations, weighted by their
actual world-side recurrence and paired with their conditional next-byte laws.
The weights describe which future contexts can pay for a retained distinction;
they do not live in the organism. -/
structure WeightedExternalFutureLaw (m : ℕ) extends ExternalFutureProbes m where
  weight : Fin m → ℝ
  weight_nonneg : ∀ i, 0 ≤ weight i
  weight_sum : ∑ i, weight i = 1
  law : Fin m → Byte → ℝ
  law_nonneg : ∀ i b, 0 ≤ law i b
  law_sum : ∀ i, ∑ b, law i b = 1

/-- Expected future language code length of a body state under the declared
continuation law.  It is an audit functional, not an added runtime scalar. -/
noncomputable def futureLanguageCrossEntropy (M : Model k d)
    {m : ℕ} (Q : WeightedExternalFutureLaw m) (s : State k d) : ℝ :=
  ∑ i, Q.weight i * CGB.PredictiveHarvest.crossEntropy (Q.law i)
    (futureLanguageSignature M Q.toExternalFutureProbes s i)

/-- A state change is paid by the language world precisely when it lowers the
expected code length of its actual future continuations. -/
def futureLanguagePaid (M : Model k d)
    {m : ℕ} (Q : WeightedExternalFutureLaw m) (before after : State k d) : Prop :=
  futureLanguageCrossEntropy M Q after < futureLanguageCrossEntropy M Q before

/-- A continuation of zero world-side recurrence has no effect on the future
code-length audit.  Therefore a distinction useful only under such a
continuation creates no retention obligation. -/
theorem future_cross_entropy_congr_on_positive_support (M : Model k d)
    {m : ℕ} (Q : WeightedExternalFutureLaw m) (s s' : State k d)
    (h : ∀ i, 0 < Q.weight i →
      CGB.PredictiveHarvest.crossEntropy (Q.law i)
          (futureLanguageSignature M Q.toExternalFutureProbes s i) =
        CGB.PredictiveHarvest.crossEntropy (Q.law i)
          (futureLanguageSignature M Q.toExternalFutureProbes s' i)) :
    futureLanguageCrossEntropy M Q s = futureLanguageCrossEntropy M Q s' := by
  unfold futureLanguageCrossEntropy
  apply sum_congr rfl
  intro i _
  rcases (Q.weight_nonneg i).eq_or_lt with hzero | hpos
  · simp [hzero]
  · rw [h i hpos]

/-- If a body change never worsens any declared future continuation and
strictly improves one continuation that recurs with positive mass, it strictly
improves expected future language code length.  This makes the preservation
value an information-theoretic quantity, rather than a hand-assigned reward. -/
theorem positive_continuation_improvement_is_future_paid (M : Model k d)
    {m : ℕ} (Q : WeightedExternalFutureLaw m) (before after : State k d)
    (hle : ∀ i,
      CGB.PredictiveHarvest.crossEntropy (Q.law i)
          (futureLanguageSignature M Q.toExternalFutureProbes after i) ≤
        CGB.PredictiveHarvest.crossEntropy (Q.law i)
          (futureLanguageSignature M Q.toExternalFutureProbes before i))
    (i : Fin m) (hweight : 0 < Q.weight i)
    (hlt : CGB.PredictiveHarvest.crossEntropy (Q.law i)
          (futureLanguageSignature M Q.toExternalFutureProbes after i) <
        CGB.PredictiveHarvest.crossEntropy (Q.law i)
          (futureLanguageSignature M Q.toExternalFutureProbes before i)) :
    futureLanguagePaid M Q before after := by
  unfold futureLanguagePaid futureLanguageCrossEntropy
  apply sum_lt_sum
  · intro j _
    exact mul_le_mul_of_nonneg_left (hle j) (Q.weight_nonneg j)
  · exact ⟨i, mem_univ i, mul_lt_mul_of_pos_left hlt hweight⟩

end CGB.GDNLife.Process
