import Mathlib.Tactic

/-! Actual bindings: the anchor is the current input-free S/P, pending byte,
fast table and model. read0 is _read_source without an intention, readQ is
the same pending byte with the native question context; endpoint is known_film.
The future raw boundary remains the DECISION label before question selection.
Only the shared executor increment is transported, using CURRENT parameters.
-/
namespace CGB.NativeSelfExecution
variable {V : Type*} [AddCommGroup V]
def rebase (future anchor output : V) := future + (output-anchor)

theorem anchor_execution_exact (anchor output : V) :
    rebase anchor anchor output = output := by simp [rebase]

theorem source_and_action_share_intention (future anchor post endpoint : V) :
    rebase future anchor endpoint - rebase future anchor post = endpoint-post := by
  simp only [rebase]; abel

theorem omitted_intention_is_source_error (anchor post : V) :
    anchor - rebase anchor anchor post = anchor-post := by
  rw [anchor_execution_exact]
end CGB.NativeSelfExecution
