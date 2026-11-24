import Mathlib.Algebra.Module.Basic

/-!
# Grounded self-hearing, rather than an ungrounded fixed point

Review proof, not a production organism/capability theorem. The variables map
to `world_learning_cycle.py`: `physical` includes the ACTUALLY delivered
utterance, self-hearing and received response; `encode t` is the coordinate
map at the corresponding parameter version; `predict t` is the action-
conditioned forecast. Neither one-step agreement nor a faithful encoder is
assumed to have been learned by the current body.

The time-indexed graph theorem permits plasticity and continuous motion.
The affine update is the single-observation predictive marginal of conjugate
linear conditioning, NOT an assertion that the body's whole Adam update is
an EMA or contraction. See `causal_relation_program.posterior`; a finite,
regularized, conflicting evidence set need not fit every observation.

Brouwer existence alone does not provide the evidence anchoring used below.
No fixed point of the organism, truth of self-beliefs, or privileged accuracy
over external-world predictions is asserted.
-/

namespace CGB.SelfHearingGrounding

section MovingGraph

variable {Body Latent Action : Type*}

/-- The very same executed action indexes actual and predicted transitions.
The representation is allowed to change at every step. -/
theorem moving_graph_preserved
    (physical : ℕ → Body → Action → Body)
    (encode : ℕ → Body → Latent)
    (predict : ℕ → Latent → Action → Latent)
    (body : ℕ → Body) (model : ℕ → Latent) (action : ℕ → Action)
    (hbody : ∀ t, body (t + 1) = physical t (body t) (action t))
    (hmodel : ∀ t, model (t + 1) = predict t (model t) (action t))
    (hinitial : model 0 = encode 0 (body 0))
    (hground : ∀ t, predict t (encode t (body t)) (action t) =
      encode (t + 1) (physical t (body t) (action t))) :
    ∀ t, model t = encode t (body t) := by
  intro t
  induction t with
  | zero => exact hinitial
  | succ t ih => rw [hmodel, ih, hground, hbody]

/-- A stationary physical state induces a latent fixed point only when the
actual self-hearing transition is represented correctly. -/
theorem grounded_rest_gives_latent_fixed_point
    (physical : Body → Body) (encode : Body → Latent)
    (predict : Latent → Latent) (s : Body)
    (hground : predict (encode s) = encode (physical s))
    (hrest : physical s = s) : predict (encode s) = encode s := by
  rw [hground, hrest]

end MovingGraph

section AnchoredRevision

variable {K E : Type*} [Field K] [AddCommGroup E] [Module K E]

/-- A grounded predictive marginal update; y is observed evidence. -/
def revise (η : K) (y p : E) : E := p + η • (y - p)

/-- Positive/nonzero effective evidence gain rules out an incorrect fixed
point of this marginal update. It is not enough merely to be self-consistent. -/
theorem grounded_fixed_iff (η : K) (hη : η ≠ 0) (y p : E) :
    revise η y p = p ↔ p = y := by
  constructor
  · intro h
    have hz : η • (y - p) = 0 := by
      apply add_left_cancel (a := p)
      simpa [revise] using h
    have hy : y - p = 0 := by
      calc
        y - p = (η⁻¹ * η) • (y - p) := by rw [inv_mul_cancel₀ hη, one_smul]
        _ = η⁻¹ • (η • (y - p)) := by rw [smul_smul]
        _ = 0 := by rw [hz, smul_zero]
    exact (sub_eq_zero.mp hy).symm
  · rintro rfl
    simp [revise]

/-- Exact predictive-error transfer, valid in every latent dimension. -/
theorem revision_error (η : K) (y p : E) :
    revise η y p - y = (1 - η) • (p - y) := by
  calc
    revise η y p - y = (p - y) + η • (y - p) := by
      exact add_sub_right_comm p (η • (y - p)) y
    _ = (p - y) - η • (p - y) := by
      rw [← neg_sub p y, smul_neg, ← sub_eq_add_neg]
    _ = (1 - η) • (p - y) := by rw [sub_smul, one_smul]

/-- The exact residual after any number of grounded reactivations needs no
derivative through the old trajectory. A fixed target and gain are explicit
premises; the live finite evidence bank need not have either property. -/
theorem repeated_revision_error (η : K) (y : E) (p : ℕ → E)
    (hstep : ∀ t, p (t + 1) = revise η y (p t)) :
    ∀ t, p t - y = (1 - η) ^ t • (p 0 - y) := by
  intro t
  induction t with
  | zero => simp
  | succ t ih => rw [hstep, revision_error, ih, smul_smul, pow_succ']

/-- With no effective evidence gain every belief is fixed: self-consistency
without feedback has no truth-selecting force. -/
theorem zero_gain_keeps_every_prediction (y p : E) : revise (0 : K) y p = p := by
  simp [revise]

end AnchoredRevision

end CGB.SelfHearingGrounding

#print axioms CGB.SelfHearingGrounding.moving_graph_preserved
#print axioms CGB.SelfHearingGrounding.grounded_fixed_iff
#print axioms CGB.SelfHearingGrounding.repeated_revision_error
