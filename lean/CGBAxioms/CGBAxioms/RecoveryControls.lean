import CGBAxioms.OperationalLife

/-!
# Recovery semantic controls (not organism capability experiments)

Identical body and stimulus menus; only world accessibility differs. The oracle
cannot distinguish these worlds; actual first-hit probability must distinguish
them. These are mathematical countermodels of a definition, not trained agents.
-/

open Finset

namespace CGB.RecoveryControls

def body : Medium Unit Bool Unit Bool where
  near _ _ := True
  ident _ := 0
  boundary := univ
  content b := if b then 1 else 0
  parts _ := univ
  act _ s x := if s () then fun _ => true else x
  rate _ s x := if s () && !x () then 1 else 0
  rate_nonneg _ _ _ := by split_ifs <;> norm_num
  init _ := false
  calm _ := false

noncomputable abbrev world (opens : Bool) : World Unit Bool Bool where
  W := Bool
  sense w _ := w
  moves w _ w' := if w' = (w || opens) then 1 else 0
  moves_nonneg _ _ _ := by split_ifs <;> norm_num
  moves_sum w _ := by simp
  start := false
  quiet _ := false
  clock := 1
  clock_pos := by norm_num
  horizon := 2
  recovery := 1 / 2
  recovery_pos := by norm_num

theorem sum_functions (f : (Unit → Bool) → ℝ) :
    ∑ x, f x = f (fun _ => false) + f (fun _ => true) := by
  calc
    _ = ∑ b : Bool, f (fun _ => b) :=
      Fintype.sum_equiv (Equiv.funUnique Unit Bool) _ _ (by
        intro x
        congr 1)
    _ = _ := by simp; ring

theorem body_kernel (s : Unit → Bool) (x y : Unit → Bool) :
    body.kernel 1 s x y = if y () = (x () || s ()) then 1 else 0 := by
  classical
  have hx : x = fun _ => x () := by funext u; cases u; rfl
  have hy : y = fun _ => y () := by funext u; cases u; rfl
  have hs : s = fun _ => s () := by funext u; cases u; rfl
  rw [hx, hy, hs]
  simp only [Medium.kernel, sum_functions, Medium.jump, body, Fintype.sum_unique]
  cases hxb : x () <;> cases hsb : s () <;> cases hyb : y () <;> norm_num [funext_iff]

theorem body_clock : body.Clocked 1 := by
  constructor
  · norm_num
  · intro s x
    classical
    simp only [Medium.jump, body, Fintype.sum_unique]
    rw [Finset.sum_eq_single (if s () then (fun _ : Unit => true) else x)]
    · split_ifs <;> norm_num
    · intro y _ hy
      simp [Ne.symm hy]
    · simp

theorem oracle_ignores_access (k : ℕ) (x : Unit → Bool) :
    body.recover (world false) k x = body.recover (world true) k x := by
  induction k generalizing x with
  | zero => rfl
  | succ k ih => simp only [Medium.recover, ih]

theorem actual_recovery_distinguishes_world_moves :
    body.actualRecover (world false) body_clock 2 (fun _ => false) false = 0 ∧
    body.actualRecover (world true) body_clock 2 (fun _ => false) false = 1 := by
  simp only [Medium.actualRecover, FirstHit.probability, Medium.actualKernel, Medium.jointStep]
  dsimp only [world]
  simp only [Fintype.sum_prod_type, sum_functions, Fintype.sum_bool]
  simp_rw [body_kernel]
  norm_num [Medium.materialIntake, Medium.total, body]

end CGB.RecoveryControls
