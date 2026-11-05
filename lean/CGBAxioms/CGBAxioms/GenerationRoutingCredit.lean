import CGBAxioms.Generation

/-!
# Finite sparse-route credit

A fixed sparse support is a finite boundary alphabet.  This file proves the
only new arithmetic needed to train a route *after its support has been
selected*: a softmax route's local read has an exact forward directional
response.  Selection changes themselves are deliberately outside this result;
the executable route must keep its support fixed while issuing the receipt.

This is a one-event local identity.  It neither retains a reverse-time graph
nor supplies delayed credit for H/C/S transitions.
-/

namespace CGB.Generation

open Finset

/-- A finite route reads a value attached to each selected site. -/
noncomputable def softmaxRead {B : Type*} [Fintype B]
    (logits value : B → ℝ) : ℝ :=
  ∑ b, CGB.Medium.softmax logits b * value b

/-- The log-softmax derivative for an arbitrary finite direction.  G14b is
the one-coordinate instance; a selected route may change all of its fixed
support logits together. -/
theorem log_softmax_direction_deriv {B : Type*} [Fintype B] [Nonempty B]
    (logits direction : B → ℝ) (target : B) :
    HasDerivAt
      (fun θ => Real.log (CGB.Medium.softmax
        (fun b => logits b + θ * direction b) target))
      (direction target - ∑ b, CGB.Medium.softmax logits b * direction b) 0 := by
  have hden : ∀ θ : ℝ, 0 < ∑ b, Real.exp (logits b + θ * direction b) := fun θ =>
    Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
  have hform :
      (fun θ => Real.log (CGB.Medium.softmax
        (fun b => logits b + θ * direction b) target)) =
      fun θ => logits target + θ * direction target -
        Real.log (∑ b, Real.exp (logits b + θ * direction b)) := by
    funext θ
    simp only [CGB.Medium.softmax]
    rw [Real.log_div (Real.exp_pos _).ne' (hden θ).ne', Real.log_exp]
  rw [hform]
  have hlin : ∀ b, HasDerivAt (fun θ : ℝ => logits b + θ * direction b)
      (direction b) 0 := fun b => by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const (direction b)).const_add (logits b)
  have hsum : HasDerivAt (fun θ => ∑ b, Real.exp (logits b + θ * direction b))
      (∑ b, Real.exp (logits b) * direction b) 0 := by
    simpa using HasDerivAt.fun_sum (u := Finset.univ) (fun b _ => (hlin b).exp)
  have h := (hlin target).sub (hsum.log (hden 0).ne')
  simp only [zero_mul, add_zero] at h
  have hsoft :
      (∑ b, CGB.Medium.softmax logits b * direction b) =
        (∑ b, Real.exp (logits b) * direction b) /
          ∑ b, Real.exp (logits b) := by
    simp only [CGB.Medium.softmax]
    calc
      (∑ b, Real.exp (logits b) / (∑ c, Real.exp (logits c)) * direction b) =
          ∑ b, (Real.exp (logits b) * direction b) /
            (∑ c, Real.exp (logits c)) := by
        apply Finset.sum_congr rfl
        intro b _
        ring
      _ = _ := by
        simp only [div_eq_mul_inv]
        rw [Finset.sum_mul]
  rw [hsoft]
  exact h

/-- Each selected route probability has the expected softmax response. -/
theorem softmax_direction_deriv {B : Type*} [Fintype B] [Nonempty B]
    (logits direction : B → ℝ) (target : B) :
    HasDerivAt
      (fun θ => CGB.Medium.softmax
        (fun b => logits b + θ * direction b) target)
      (CGB.Medium.softmax logits target *
        (direction target - ∑ b, CGB.Medium.softmax logits b * direction b)) 0 := by
  have hlog := log_softmax_direction_deriv logits direction target
  have hexp := hlog.exp
  have hpos : 0 < CGB.Medium.softmax logits target := by
    unfold CGB.Medium.softmax
    exact div_pos (Real.exp_pos _) (Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty)
  have heq :
      (fun θ => Real.exp (Real.log (CGB.Medium.softmax
        (fun b => logits b + θ * direction b) target))) =
      (fun θ => CGB.Medium.softmax (fun b => logits b + θ * direction b) target) := by
    funext θ
    exact Real.exp_log (by
      unfold CGB.Medium.softmax
      exact div_pos (Real.exp_pos _) (Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty))
  rw [heq] at hexp
  simpa only [zero_mul, mul_zero, add_zero, Real.exp_log hpos,
    mul_comm, mul_left_comm, mul_assoc] using hexp

/-- A route read changes by the sum of its local, centered softmax responses.
This is the exact receipt used by an immediately scored fixed-support route. -/
theorem softmax_read_direction_deriv {B : Type*} [Fintype B] [Nonempty B]
    (logits direction value : B → ℝ) :
    HasDerivAt
      (fun θ => softmaxRead (fun b => logits b + θ * direction b) value)
      (∑ b, CGB.Medium.softmax logits b *
        (direction b - ∑ c, CGB.Medium.softmax logits c * direction c) * value b) 0 := by
  unfold softmaxRead
  have hterm : ∀ b, HasDerivAt
      (fun θ => CGB.Medium.softmax (fun c => logits c + θ * direction c) b * value b)
      (CGB.Medium.softmax logits b *
        (direction b - ∑ c, CGB.Medium.softmax logits c * direction c) * value b) 0 := by
    intro b
    simpa [mul_assoc] using (softmax_direction_deriv logits direction b).mul_const (value b)
  simpa using HasDerivAt.fun_sum (u := Finset.univ) (fun b _ => hterm b)

end CGB.Generation
