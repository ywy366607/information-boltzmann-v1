import CGBAxioms.RenewableLearningLife
import CGBAxioms.RecoveryCriteria

/-! Instantiate the reusable local recovery criterion on the existing model.
The body, environment, resource ports and world tick are unchanged. -/

open Finset

namespace CGB.RenewableRecoveryCertificate
open RenewableLearning RenewableLearningLife

private theorem policy_table (q : Cell) : policy q =
    match q with
    | .aSpent | .aReady | .aHeld => 0
    | .bSpent | .bReady | .bHeld => 1
    | _ => 2 := by
  cases q <;> decide

private theorem nonvacuum (q : Cell) (w : Env) (hg : Good q w) : q ≠ .vacuum := by
  cases w <;> cases q <;> simp_all [Good]

private theorem active (w : Env) : sense w () ≠ .calm := by cases w <;> decide

set_option maxHeartbeats 1800000 in
-- Check only the finite supported rows of the proof region.
private theorem next_good (q : Cell) (w v : Env) (hg : Good q w)
    (hp : 0 < moves w (fun _ => q) v) : Good (next true true (sense w ()) q) v := by
  cases w <;> cases q <;> simp [Good] at hg
  all_goals cases v <;> norm_num [moves, policy_table] at hp
  all_goals norm_num [next, sense, Good]

private theorem supported_good (t : ℕ) (z z' : S) (e : Option Unit)
    (hg : Good (z.1 ()) z.2) (hp : 0 < (process true true false).step t z e z') :
    Good (z'.1 ()) z'.2 := by
  have hq := nonvacuum (z.1 ()) z.2 hg
  have hs := active z.2
  change 0 < moves z.2 z.1 z'.2 *
    (plant true true).eventStep 1 (sense z.2) z.1 e z'.1 at hp
  cases e with
  | none => simp [Medium.eventStep, plant, hq, hs] at hp
  | some u =>
    simp only [Medium.eventStep, plant, hs, hq, false_or, ↓reduceIte, div_one] at hp
    change 0 < moves z.2 z.1 z'.2 *
      (if (fun _ => next true true (sense z.2 ()) (z.1 ())) = z'.1 then 1 else 0) at hp
    split_ifs at hp with hn
    · have hw : 0 < moves z.2 z.1 z'.2 := by simpa using hp
      have hx : z.1 = fun _ => z.1 () := by funext u; cases u; rfl
      rw [hx] at hw
      have he := congrFun hn ()
      rw [← he]
      exact next_good (z.1 ()) z.2 z'.2 hg hw
    · norm_num at hp


private theorem unmarked_progress (q : Cell) (w v : Env) (hg : Good q w)
    (hw : 0 < moves w (fun _ => q) v) (hn : ¬0 < incoming true (sense w ()) q) :
    (if v = .missA ∨ v = .missB then (1 : ℕ) else 0) <
      (if w = .missA ∨ w = .missB then (1 : ℕ) else 0) := by
  cases w <;> cases q <;> simp [Good] at hg
  all_goals norm_num [incoming, sense] at hn
  all_goals cases v <;> norm_num [moves, policy_table] at hw
  all_goals norm_num

noncomputable def certificate : RecoveryCriteria.Ranked
    ((plant true true).jointEventKernel world (he true true) false (fun _ => world.sense))
    (Exchange.intake (plant true true) world (ledger true true) (fun _ => world.sense))
    world.horizon where
  region z := Good (z.1 ()) z.2
  rank z := if z.2 = .missA ∨ z.2 = .missB then 1 else 0
  bounded z _ := by split_ifs <;> norm_num [world]
  closed t z e z' hg hp := supported_good t z z' e hg hp
  progresses t z e z' hg hp hm := by
    have hq := nonvacuum (z.1 ()) z.2 hg
    have hs := active z.2
    change 0 < moves z.2 z.1 z'.2 *
      (plant true true).eventStep 1 (sense z.2) z.1 e z'.1 at hp
    cases e with
    | none => simp [Medium.eventStep, plant, hq, hs] at hp
    | some u =>
      have hw : 0 < moves z.2 z.1 z'.2 := by
        by_contra hn
        have hb := (plant true true).eventStep_nonneg (he true true) (sense z.2) z.1
          (some u) z'.1
        exact (not_lt_of_ge (mul_nonpos_of_nonpos_of_nonneg (le_of_not_gt hn) hb)) hp
      have hn : ¬0 < incoming true (sense z.2 ()) (z.1 ()) := by
        intro hi
        simp [Exchange.intake, ledger, hi] at hm
      have hx : z.1 = fun _ => z.1 () := by funext u; cases u; rfl
      rw [hx] at hw
      exact unmarked_progress (z.1 ()) z.2 z'.2 hg hw hn

/-- The original all-time score now also follows from the shared local
criterion; it is not inserted as a premise of that criterion. -/
theorem survival_by_criterion (n : ℕ) :
    Exchange.survival (plant true true) world (he true true) (ledger true true)
      false (fun _ => world.sense) n = 1 := by
  apply RecoveryCriteria.survival_of_ranked (plant true true) world (he true true)
    (ledger true true) (by norm_num [world]) certificate (by rfl) _ n
  intro z hg
  have hn := nonvacuum (z.1 ()) z.2 hg
  change 0 < cost (z.1 ())
  generalize hh : z.1 () = q at *
  cases q <;> norm_num [cost, buffer, credit] at *

end CGB.RenewableRecoveryCertificate
