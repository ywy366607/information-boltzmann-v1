import CGBAxioms.ActiveLoop

/- Review checks against the unchanged proposed dynamics; not production claims.
   Run with: lake env lean reviews/active_loop_20261024.lean -/

open Finset CGB CGB.ActiveLoop

namespace ActiveLoopReview20261024

theorem sum_configs (f : (Unit → State) → ℝ) :
    ∑ x, f x = f (fun _ => 0) + f (fun _ => 1) +
      f (fun _ => 2) + f (fun _ => 3) := by
  calc
    _ = ∑ a : State, f (fun _ => a) :=
      Fintype.sum_equiv (Equiv.funUnique Unit State) _ _ (by intro x; congr 1)
    _ = _ := by
      have h : (univ : Finset State) = {0, 1, 2, 3} := by decide
      rw [h]
      simp
      ring

set_option maxHeartbeats 800000 in
theorem body_kernel_table (s : Stim) (x y : State) :
    body.kernel 1 (fun _ => s) (fun _ => x) (fun _ => y) =
      if y = advance () s x then 1 else 0 := by
  simp only [Medium.kernel, Medium.jump, body, Fintype.sum_unique]
  rw [sum_configs]
  fin_cases s <;> fin_cases x <;> fin_cases y <;>
    norm_num [advance, funext_iff]

-- Every entry in the exact rational replay uses this normal-process table.
theorem natural_step_table (t : ℕ) (x y : State) (w v : WorldState) :
    (jointMachine false).step t ((fun _ => x), w) () ((fun _ => y), v) =
      worldMoves w (fun _ => x) v *
        (if y = advance () (worldSense w ()) x then 1 else 0) := by
  simp only [jointMachine, Bool.false_eq_true, ↓reduceIte, Medium.scheduledKernel,
    Medium.jointStep, world]
  exact congrArg (fun q => worldMoves w (fun _ => x) v * q)
    (body_kernel_table (worldSense w ()) x y)

theorem cut_step_table (t : ℕ) (x y : State) (w v : WorldState) :
    (jointMachine true).step t ((fun _ => x), w) () ((fun _ => y), v) =
      (∑ g : State, (∑ u : WorldState,
        body.lawCut world (fun _ => world.sense) t (fun _ => g) u) *
          worldMoves w (fun _ => g) v) *
        (if y = advance () (worldSense w ()) x then 1 else 0) := by
  change (∑ g : Unit → State,
    (∑ u : WorldState, body.lawCut world (fun _ => world.sense) t g u) *
      worldMoves w g v) * body.kernel 1 (fun _ => worldSense w ())
        (fun _ => x) (fun _ => y) = _
  rw [body_kernel_table]
  congr 1

-- Feeding erases the B action: the very next arena-B check rejects action A.
example : advance () 3 3 = 1 ∧ worldSense 4 () = 3 ∧
    worldMoves 4 (fun _ => 1) 5 = 1 := by
  norm_num [advance, worldSense, worldMoves]

-- A closed-loop path of probability 1/2 reaches starvation at time 5.
example :
    (jointMachine false).step 0 ((fun _ => 1), (0 : WorldState)) ()
      ((fun _ => 1), (2 : WorldState)) *
    (jointMachine false).step 1 ((fun _ => 1), (2 : WorldState)) ()
      ((fun _ => 2), (4 : WorldState)) *
    (jointMachine false).step 2 ((fun _ => 2), (4 : WorldState)) ()
      ((fun _ => 3), (4 : WorldState)) *
    (jointMachine false).step 3 ((fun _ => 3), (4 : WorldState)) ()
      ((fun _ => 1), (4 : WorldState)) *
    (jointMachine false).step 4 ((fun _ => 1), (4 : WorldState)) ()
      ((fun _ => 3), (5 : WorldState)) = 1 / 2 := by
  simp only [natural_step_table]
  norm_num [worldMoves, worldSense, advance]

-- The account permits positive heat input to pay work in the fed -> action step.
example : (activeAccount 1 2 false).heat 0 ((fun _ => 3), (3 : WorldState)) ()
    ((fun _ => 1), (3 : WorldState)) = 1 := by
  norm_num [activeAccount, storedEnergy, actionWork, matterIntakeEnergy, cost]

-- Starvation freezes the body, without spending its stored energy or paying work.
example (a w : ℝ) :
    (jointMachine false).step 0 ((fun _ => 3), (5 : WorldState)) ()
      ((fun _ => 3), (5 : WorldState)) = 1 ∧
    (activeAccount a w false).stored 1 ((fun _ => 3), (5 : WorldState)) = 2 * a ∧
    (activeAccount a w false).load 0 ((fun _ => 3), (5 : WorldState)) ()
      ((fun _ => 3), (5 : WorldState)) = 0 ∧
    (activeAccount a w false).matter 0 ((fun _ => 3), (5 : WorldState)) ()
      ((fun _ => 3), (5 : WorldState)) = 0 ∧
    (activeAccount a w false).heat 0 ((fun _ => 3), (5 : WorldState)) ()
      ((fun _ => 3), (5 : WorldState)) = 0 := by
  rw [natural_step_table]
  norm_num [worldMoves, worldSense, advance, activeAccount, storedEnergy,
    actionWork, matterIntakeEnergy, cost]

end ActiveLoopReview20261024
