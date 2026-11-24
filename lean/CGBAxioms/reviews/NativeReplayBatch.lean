import Init

/-! Instance of the original pointwise-layer contract with branch-local controls.
  C = (excluded slot, version, resulting coevolution and deep-read masks).
  S = all language/conv/workspace state and current hidden values.
  Each coevolution cell, GDN layer, and deep read acts on its own B row.
  RMSNorm/softmax reduce feature or slot axes; causal_scan reduces time.
  Only logits/next-state/read identities are the targets; scalar diagnostics
  can aggregate B, and no private trajectory is committed.
-/
namespace CGB.NativeReplayBatch
variable {B C S : Type}

def serial (layers : List (C → S → S)) (control : C) (state : S) : S :=
  layers.foldl (fun s f => f control s) state

def batched (layers : List (C → S → S)) (control : B → C) (state : B → S) : B → S :=
  layers.foldl (fun s f b => f (control b) (s b)) state

theorem branch_local_masks_preserve_original_trajectory
    (layers : List (C → S → S)) (control : B → C) (state : B → S) (b : B) :
    batched layers control state b = serial layers (control b) (state b) := by
  induction layers generalizing state with
  | nil => rfl
  | cons f tail ih =>
    exact ih (fun b => f (control b) (state b))
end CGB.NativeReplayBatch
