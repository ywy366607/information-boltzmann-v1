import Mathlib.Tactic

/-! Body binding: native_word_batch.endpoints executes independent original
GDN trajectories on the B axis, at fixed weights, question, S and coevolving
workset. LatentNavigationBody's private request returns its ORIGINAL
experience-layer input port. Later layers do not feed back to that port;
their next recurrent state is not committed by prospective_self_execution.
Padding is after each trajectory's own endpoint. Every original operator is
causal and operates separately on B; actual body equivalence is checked with
ragged payloads, nonzero recurrent state, intention and saved tables.
-/
namespace CGB.NativeWordBatch
variable {B S : Type*}

def run (layers : List (S → S)) (s : S) : S :=
  layers.foldl (fun x f => f x) s

theorem original_layers_batch_pointwise (layers : List (S → S)) (s : B → S) (b : B) :
    (run (layers.map (fun f x b => f (x b))) s) b = run layers (s b) := by
  induction layers generalizing s with
  | nil => rfl
  | cons f tail ih =>
    simpa [run, List.foldl_cons] using ih (fun b => f (s b))

theorem original_port_before_unused_tail (front rest : List (S → S)) (s : S) :
    run (front ++ rest) s = run rest (run front s) := by
  simp [run, List.foldl_append]

theorem causal_prefix_composes (cut f g : S → S)
    (hf : ∀ s, cut (f s) = cut (f (cut s)))
    (hg : ∀ s, cut (g s) = cut (g (cut s))) (s : S) :
    cut (g (f s)) = cut (g (f (cut s))) := by
  calc
    cut (g (f s)) = cut (g (cut (f s))) := hg _
    _ = cut (g (cut (f (cut s)))) := by rw [hf]
    _ = cut (g (f (cut s))) := (hg _).symm

end CGB.NativeWordBatch
