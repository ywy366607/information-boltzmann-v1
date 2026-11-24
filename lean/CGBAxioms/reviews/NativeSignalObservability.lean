import Mathlib.Tactic

/-!
Measurement postconditions for actual relation vectors, not a new body model.
`B1` is the certified boundary of NativeDowkerActual2401. Every latent
coordinate is handled by the same identity. Missing signals are not zeros.
-/
namespace CGB.NativeSignalObservability

variable {V E : Type*} [Fintype V] [Fintype E]

def gradient (B1 : V → E → ℝ) (x : V → ℝ) (edge : E) : ℝ :=
  ∑ vertex, B1 vertex edge * x vertex

def divergence (B1 : V → E → ℝ) (h : E → ℝ) (vertex : V) : ℝ :=
  ∑ edge, B1 vertex edge * h edge

/-- Node prediction differences cannot expose a divergence-free harmonic mode,
even if every node was measured. This holds for the full certified complex;
the additional B2^T h = 0 condition is unnecessary for this obstruction. -/
theorem node_difference_orthogonal_to_harmonic
    (B1 : V → E → ℝ) (x : V → ℝ) (h : E → ℝ)
    (closed : ∀ vertex, divergence B1 h vertex = 0) :
    ∑ edge, gradient B1 x edge * h edge = 0 := by
  simp only [gradient, Finset.sum_mul]
  rw [Finset.sum_comm]
  have each : ∀ vertex, (∑ edge, B1 vertex edge * x vertex * h edge) = 0 := by
    intro vertex
    calc
      _ = x vertex * ∑ edge, B1 vertex edge * h edge := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro edge _
        ring
      _ = 0 := by rw [← divergence, closed vertex, mul_zero]
  simp only [each, Finset.sum_const_zero]

variable {A W H : Type*} [AddCommGroup A] [Module ℝ A]
  [AddCommGroup W] [Module ℝ W] [AddCommGroup H] [Module ℝ H]

/-- A hidden harmonic component changes the answer without changing any
recorded samples. R is actual sampling; Q is the Hodge harmonic readout. -/
theorem invisible_mode_produces_ambiguous_readout
    (R : A →ₗ[ℝ] W) (Q : A →ₗ[ℝ] H) (signal hidden : A)
    (unseen : R hidden = 0) (nonzero : Q hidden ≠ 0) :
    R (signal + hidden) = R signal ∧ Q (signal + hidden) ≠ Q signal := by
  constructor
  · simp [map_add, unseen]
  · rw [map_add]
    intro equal
    apply nonzero
    exact add_left_cancel (equal.trans (add_zero (Q signal)).symm)

/-- Exact adequacy condition: equal recorded samples must imply equal
harmonic readout. Equivalent to excluding every invisible readout mode. -/
theorem readout_identifiable_iff
    (R : A →ₗ[ℝ] W) (Q : A →ₗ[ℝ] H) :
    (∀ a b, R a = R b → Q a = Q b) ↔
      (∀ hidden, R hidden = 0 → Q hidden = 0) := by
  constructor
  · intro adequate hidden unseen
    simpa using adequate hidden 0 (by simpa using unseen)
  · intro no_hidden a b same
    have h := no_hidden (a - b) (by simp [map_sub, same])
    exact sub_eq_zero.mp (by simpa using h)

/-- Native posterior: (query * design^T) * coefficient. Each evidence-row
message is recoverable without fitting another predictor or subset replay. -/
theorem evidence_messages_recover_correction
    {I F : Type*} [Fintype I] [Fintype F]
    (query : F → ℝ) (design : I → F → ℝ) (coefficient : I → ℝ) :
    (∑ i, (∑ f, query f * design i f) * coefficient i) =
      ∑ f, query f * (∑ i, design i f * coefficient i) := by
  simp only [Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro f _
  apply Finset.sum_congr rfl
  intro i _
  ring

end CGB.NativeSignalObservability
