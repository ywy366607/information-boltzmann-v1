import Mathlib.Tactic

/-! Body bindings:
`latent_action_planner.packet` changes raw role 1 (nominal action), keeping
pre-source role 0 and momentum-of-pre-source role 6 fixed. The local-action
encoder therefore has its own kernel, even when the pre-source joint readout
is injective. `plan` predicts one common post-intention state, separate raw
word endpoints and q, then calls compose at those states and endpoints.
The finite four-corner identity below splits that actual total-score error.
The actor's existing common-mode theorem is inherited, rather than reproved.
-/
namespace CGB.NativeActionPortTransport
noncomputable section

abbrev H := Fin 256 → ℝ
abbrev Z := Fin 128 → ℝ
abbrev RawPacket := Fin 13 → H

/-- The exact nominal role updated by the production packet constructor. -/
def withNominal (p : RawPacket) (a : H) : RawPacket := Function.update p 1 a

/-- Even with a complementary source/P view, a nominal-only encoder-null
change gives identical encoded role blocks. Other packet fields stay fixed. -/
theorem nominal_kernel_preserves_encoded_packet
    (E : H →ₗ[ℝ] Z) (p : RawPacket) (a d : H) (hd : E d = 0) :
    (fun i => E (withNominal p (a+d) i)) =
      (fun i => E (withNominal p a i)) := by
  funext i
  by_cases hi : i = 1
  · subst i; simp [withNominal, hd]
  · simp [withNominal, hi]

/-- Exact terminal contrast error: q error, endpoint correction at the
predicted common state, then common-state correction at true endpoints.
SS/ST/TT stand for student/student, student/teacher and teacher/teacher.
Each F includes the actual momentum re-read at its post-intention state. -/
theorem contrast_error_four_corners
    (qa qb ta tb gamma ssa ssb sta stb tta ttb : ℝ) :
    ((qa+gamma*ssa)-(qb+gamma*ssb)) -
      ((ta+gamma*tta)-(tb+gamma*ttb)) =
    ((qa-ta)-(qb-tb)) +
      gamma*((ssa-sta)-(ssb-stb)) +
      gamma*((sta-tta)-(stb-ttb)) := by ring

/-- Changing correction order reallocates the mixed interaction, with exact
accounting. TS is the true state with student endpoints. -/
theorem correction_order_interaction
    (ss st ts tt : ℝ) :
    (ss-st)-(ts-tt) = ss-st-ts+tt := by ring

/-- Shared-state changes are a common score offset precisely when their
action contrasts agree across the two states. This is the actual condition
needed to apply the inherited actor common-mode result. -/
theorem shared_state_common_offset_iff
    (fa fb ga gb : ℝ) :
    fa-fb = ga-gb ↔ fa-ga = fb-gb := by constructor <;> intro h <;> linarith

/-- ActionEvidence.confidence, after its positive denominator guard.
The finite-use channel uses this value on the raw content residual. -/
def contentConfidence (rawError guardedBaseError : ℝ) : ℝ :=
  max 0 (min 1 (1-rawError/guardedBaseError))

theorem content_gate_zero_of_error
    (rawError guardedBaseError : ℝ) (hb : 0 < guardedBaseError)
    (hr : guardedBaseError ≤ rawError) :
    contentConfidence rawError guardedBaseError = 0 := by
  have ratio : 1 ≤ rawError/guardedBaseError :=
    (le_div_iff₀ hb).mpr (by simpa using hr)
  unfold contentConfidence
  exact max_eq_left ((min_le_right _ _).trans (by linarith))

/-- With this actual gate closed, arbitrary content forecasts map to the
same option baseline. Conditional information can then arrive via the bank. -/
theorem gated_content_uses_option_base
    (base residual rawError guardedBaseError : ℝ) (hb : 0 < guardedBaseError)
    (hr : guardedBaseError ≤ rawError) :
    base+contentConfidence rawError guardedBaseError*residual = base := by
  rw [content_gate_zero_of_error _ _ hb hr]
  simp

end
end CGB.NativeActionPortTransport
