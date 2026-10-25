import CGBAxioms.RenewableLearning

/-! All-time viability in the same changing body/world process. The invariant
is a proof region, not a state flag, reset, or extra policy input. -/

open Finset

namespace CGB.RenewableLearningLife
open RenewableLearning

def Good (q : Cell) : Env → Prop
  | .feedA => q = .aSpent ∨ q = .aReady ∨ q = .bReady
  | .feedB => q = .bSpent ∨ q = .aReady ∨ q = .bReady
  | .missA => q = .bReady
  | .missB => q = .aReady
  | .probeA | .probeB | .answerA | .answerB => q = .search
  | .trap => False

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

private theorem law_invariant {X E : Type*} [Fintype X] [DecidableEq X] [Fintype E]
    (Q : Interaction.Machine X ℕ E) (initial : X) (Region : X → Prop) (hi : Region initial)
    (hs : ∀ t x e y, Region x → 0 < Q.step t x e y → Region y)
    (n : ℕ) (x : X) (hx : ¬ Region x) : Energetics.law Q initial n x = 0 := by
  classical
  induction n generalizing x with
  | zero =>
    have hn : x ≠ initial := by intro he; subst x; exact hx hi
    simp [Energetics.law, EventChannels.Kernel.stateLaw, hn]
  | succ n ih =>
    change (∑ y : X, Energetics.law Q initial n y * (∑ e : E, Q.step n y e x)) = 0
    apply Finset.sum_eq_zero
    intro y _
    by_cases hy : Region y
    · have hz (e : E) : Q.step n y e x = 0 := by
        by_cases hp : 0 < Q.step n y e x
        · exact False.elim (hx (hs n y e x hy hp))
        · exact le_antisymm (le_of_not_gt hp) (Q.nonneg n y e x)
      simp [hz]
    · rw [ih y hy, zero_mul]

theorem reachable_good (n : ℕ) (z : S)
    (hp : 0 < Energetics.law (process true true false) initial n z) : Good (z.1 ()) z.2 := by
  classical
  by_contra hn
  have hz := law_invariant (process true true false) initial
    (fun z => Good (z.1 ()) z.2) (by rfl)
      (fun t x e y hg hp => supported_good t x y e hg hp) n z hn
  linarith

private theorem sum_configs (f : (Unit → Cell) → ℝ) :
    ∑ x, f x = ∑ q : Cell, f (fun _ => q) :=
  Fintype.sum_equiv (Equiv.funUnique Unit Cell) _ _ (by intro x; congr 1)

private theorem sum_env (f : Env → ℝ) : ∑ w, f w =
    f .feedA + f .feedB + f .missA + f .missB + f .probeA + f .probeB +
      f .answerA + f .answerB + f .trap := by
  have h : (univ : Finset Env) =
      {.feedA, .feedB, .missA, .missB, .probeA, .probeB, .answerA, .answerB, .trap} := by decide
  rw [h]; simp; ring

private noncomputable abbrev recover (t k : ℕ) (q : Cell) (w : Env) :=
  Exchange.recovery (plant true true) world (he true true) (ledger true true)
    false (fun _ => world.sense) t k ((fun _ => q), w)

private theorem recovery_step (t k : ℕ) (q : Cell) (w : Env) (hq : q ≠ .vacuum) :
    recover t (k + 1) q w = ∑ v, moves w (fun _ => q) v *
      (if 0 < incoming true (sense w ()) q then 1 else
        recover (t + 1) k (next true true (sense w ()) q) v) := by
  have hs := active w
  by_cases hi : 0 < incoming true (sense w ()) q
  all_goals simp only [recover, Exchange.recovery, EventChannels.Kernel.firstHit,
    Fintype.sum_prod_type, Fintype.sum_option, Fintype.sum_unique, sum_configs]
  all_goals simp [Medium.jointEventKernel, Medium.eventWorldStep, Medium.eventStep,
    plant, hq, hs, hi, Exchange.intake, ledger, funext_iff, ite_mul, zero_mul]

private theorem immediate_recovery (t k : ℕ) (q : Cell) (w : Env)
    (hq : q ≠ .vacuum) (hi : 0 < incoming true (sense w ()) q) :
    recover t (k + 1) q w = 1 := by
  rw [recovery_step t k q w hq]
  simpa only [hi, ↓reduceIte, mul_one] using world.moves_sum w (fun _ => q)

theorem recovery_good (t : ℕ) (q : Cell) (w : Env) (hg : Good q w) :
    recover t 2 q w = 1 := by
  by_cases hm : w = .missA ∨ w = .missB
  · rcases hm with rfl | rfl
    · have hq : q = .bReady := hg
      subst q
      rw [recovery_step t 1 .bReady .missA (by decide)]
      simp only [sum_env]
      norm_num [moves, policy_table, incoming, next, sense]
      exact immediate_recovery (t + 1) 0 .search .probeA (by decide) (by decide)
    · have hq : q = .aReady := hg
      subst q
      rw [recovery_step t 1 .aReady .missB (by decide)]
      simp only [sum_env]
      norm_num [moves, policy_table, incoming, next, sense]
      exact immediate_recovery (t + 1) 0 .search .probeB (by decide) (by decide)
  · have hi : 0 < incoming true (sense w ()) q := by
      cases w <;> cases q <;> simp_all [Good, incoming, sense]
    exact immediate_recovery t 1 q w (nonvacuum q w hg) hi

private theorem alive_good (t : ℕ) (q : Cell) (w : Env) (hg : Good q w) :
    Exchange.alive (plant true true) world (he true true) (ledger true true)
      false (fun _ => world.sense) t ((fun _ => q), w) = 1 := by
  classical
  have hr := recovery_good t q w hg
  have hq : 0 < cost q := by
    have hn := nonvacuum q w hg
    cases q <;> norm_num [cost, buffer, credit] at *
  have ht : 0 < (plant true true).total (fun _ => q) := by
    simpa only [Medium.total, plant, Fintype.sum_unique] using hq
  unfold Exchange.alive Viability.alive
  apply ite_eq_left
  refine ⟨ht, ?_⟩
  change (1 : ℝ) / 2 ≤ recover t 2 q w
  rw [hr]
  norm_num

/-- Every supported continuation remains viable through missed food, paid
reopening, probing, identification and actual resource-funded renewal. -/
theorem natural_survival (n : ℕ) :
    Exchange.survival (plant true true) world (he true true) (ledger true true)
      false (fun _ => world.sense) n = 1 := by
  classical
  have hb (x : Unit → Cell) (w : Env) :
      Exchange.alive (plant true true) world (he true true) (ledger true true)
          false (fun _ => world.sense) n (x, w) *
        (plant true true).law world (fun _ => world.sense) n x w =
        (plant true true).law world (fun _ => world.sense) n x w := by
    have hx : x = fun _ => x () := by funext u; cases u; rfl
    by_cases hg : Good (x ()) w
    · have ha := alive_good n (x ()) w hg
      rw [← hx] at ha
      rw [ha, one_mul]
    · have hz := law_invariant (process true true false) initial
        (fun z => Good (z.1 ()) z.2) (by rfl)
        (fun t x e y hg hp => supported_good t x y e hg hp) n (x, w) hg
      rw [actual_law] at hz
      simp only [Bool.false_eq_true, ↓reduceIte] at hz
      rw [hz, mul_zero]
  change (∑ x : Unit → Cell, ∑ w : Env,
    Exchange.alive (plant true true) world (he true true) (ledger true true)
      false (fun _ => world.sense) n (x, w) *
      (plant true true).law world (fun _ => world.sense) n x w) = 1
  simp only [hb]
  exact (plant true true).law_sum world (fun _ => world.sense) n

theorem persistence : Exchange.Persists (plant true true) world (he true true)
    (ledger true true) 0 := by
  refine ⟨by norm_num, fun n => ?_⟩
  rw [natural_survival]
  norm_num

end CGB.RenewableLearningLife
