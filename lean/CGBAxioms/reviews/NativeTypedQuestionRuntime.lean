import Mathlib.Tactic

/-! Executed-port binding (2026-11-23):
`CausalProgramEngine._condition_mother_question` binds the real relation and
returns `base + question_reactivation legacy`. `GenerativeCycleMixin` then
subtracts that SAME legacy activation and adds the sampled typed filler.
The real bind/reference receipt is retained by `typed_replacement=True`.
`native_amortized_decision.capture_actual` resets `_latent_teacher_queue`
before the actual typed question utilities are evaluated. Thus legacy queue
contents have no consumer. Rhythm read-cache work and old-menu diagnostics
are separately audited against the real 2813 continuation in Python.
The algebra is over exact vectors; floating cancellation is checked at the
actual choice, update, held-out score and checkpoint ports.
-/
namespace CGB.NativeTypedQuestionRuntime
variable {V : Type*} [AddCommGroup V]

theorem selected_context (base legacy selected : V) :
    (base + legacy) - legacy + selected = base + selected := by abel

def resetTeachers {K T : Type*} (_old : K → Option T) : K → Option T := fun _ => none

theorem discarded_teachers (old₁ old₂ : K → Option T) :
    resetTeachers old₁ = resetTeachers old₂ := rfl

end CGB.NativeTypedQuestionRuntime
