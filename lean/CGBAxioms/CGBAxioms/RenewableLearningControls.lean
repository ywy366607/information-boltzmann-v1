import CGBAxioms.RenewableLearning

/-! Finite-path probabilities are products of the actual registered step law.
Paths and cumulative counts are observer evidence, never body state or inputs.
The repeated A/B witness proves arbitrarily long positive-probability relearning
prefixes, not an almost-sure frequency or real-data capability estimate. -/

open Finset

namespace CGB.RenewableLearningControls
open RenewableLearning

private theorem policy_table (q : Cell) : policy q =
    match q with
    | .aSpent | .aReady | .aHeld => 0
    | .bSpent | .bReady | .bHeld => 1
    | _ => 2 := by
  cases q <;> decide

def index (n : ℕ) : Fin 10 := ⟨n % 10, by omega⟩

def cycle (i : Fin 10) : S :=
  match i.val with
  | 0 => ((fun _ => .aReady), .feedB)
  | 1 => ((fun _ => .aReady), .missB)
  | 2 => ((fun _ => .search), .probeB)
  | 3 => ((fun _ => .search), .answerB)
  | 4 => ((fun _ => .bSpent), .feedB)
  | 5 => ((fun _ => .bReady), .feedA)
  | 6 => ((fun _ => .bReady), .missA)
  | 7 => ((fun _ => .search), .probeA)
  | 8 => ((fun _ => .search), .answerA)
  | _ => ((fun _ => .aSpent), .feedA)

private theorem cycle_step (t : ℕ) (i : Fin 10) :
    (process true true false).step t (cycle i) (some ())
      (cycle (index (i.val + 1))) = if i = 4 ∨ i = 9 then 1 / 2 else 1 := by
  fin_cases i <;> norm_num [cycle, index, process, MaterialSource.machine,
    Medium.jointEventKernel, Medium.eventWorldStep, Medium.eventStep, plant, world,
    sense, moves, policy_table, next, funext_iff]

private theorem cycle_payment :
    (∑ i ∈ range 10, writeCost true (sense (cycle (index i)).2 ())
      ((cycle (index i)).1 ())) = 4 ∧
    (∑ i ∈ range 10, topup true (sense (cycle (index i)).2 ())
      ((cycle (index i)).1 ())) = 4 := by decide

def path : ℕ → S
  | 0 => ((fun _ => .search), .probeA)
  | 1 => ((fun _ => .search), .answerA)
  | 2 => ((fun _ => .aSpent), .feedA)
  | n + 3 => cycle (index n)

private theorem path_after (n : ℕ) : path (n + 3) = cycle (index n) := rfl

private theorem path_step (n : ℕ) :
    0 < (process true true false).step n (path n) (some ()) (path (n + 1)) := by
  rcases n with _ | n
  · norm_num [path, process, MaterialSource.machine, Medium.jointEventKernel,
      Medium.eventWorldStep, Medium.eventStep, plant, world, sense, moves, policy_table, next,
      funext_iff]
  · rcases n with _ | n
    · norm_num [path, process, MaterialSource.machine, Medium.jointEventKernel,
        Medium.eventWorldStep, Medium.eventStep, plant, world, sense, moves, policy_table, next,
        funext_iff]
    · rcases n with _ | n
      · norm_num [path, cycle, index, process, MaterialSource.machine, Medium.jointEventKernel,
          Medium.eventWorldStep, Medium.eventStep, plant, world, sense, moves, policy_table, next,
          funext_iff]
      · rw [path_after, show n + 3 + 1 = (n + 1) + 3 by omega, path_after]
        have he : (index ((index n).val + 1)) = (index (n + 1)) := by
          apply Fin.ext
          change (n % 10 + 1) % 10 = (n + 1) % 10
          omega
        rw [← he, cycle_step]
        split_ifs <;> norm_num

def tag (z : S) : Option Unit := if z.1 () = .vacuum then none else some ()

private theorem path_active (n : ℕ) : (path n).1 () ≠ .vacuum := by
  rcases n with _ | n
  · decide
  · rcases n with _ | n
    · decide
    · rcases n with _ | n
      · decide
      · rw [path_after]
        generalize index n = i
        fin_cases i <;> decide

noncomputable def weight (P : Interaction.Machine S ℕ (Option Unit)) (p : ℕ → S) (n : ℕ) : ℝ :=
  ∏ t ∈ range n, P.step t (p t) (tag (p t)) (p (t + 1))

private theorem write_blocks (k : ℕ) :
    (∑ t ∈ range (3 + 10 * k), writeCost true (sense (path t).2 ()) ((path t).1 ())) =
        1 + 4 * k ∧
    (∑ t ∈ range (3 + 10 * k), topup true (sense (path t).2 ()) ((path t).1 ())) =
        2 + 4 * k := by
  have he (k i : ℕ) : path (3 + 10 * k + i) = cycle (index i) := by
    rw [show 3 + 10 * k + i = (10 * k + i) + 3 by omega, path_after]
    apply congrArg cycle
    apply Fin.ext
    change (10 * k + i) % 10 = i % 10
    omega
  have hw (k : ℕ) :
      (∑ i ∈ range 10, writeCost true (sense (path (3 + 10 * k + i)).2 ())
        ((path (3 + 10 * k + i)).1 ())) = 4 := by
    simpa only [he] using cycle_payment.1
  have hr (k : ℕ) :
      (∑ i ∈ range 10, topup true (sense (path (3 + 10 * k + i)).2 ())
        ((path (3 + 10 * k + i)).1 ())) = 4 := by
    simpa only [he] using cycle_payment.2
  induction k with
  | zero => norm_num [Finset.sum_range_succ, path, writeCost, topup, sense]
  | succ k ih =>
    rw [show 3 + 10 * (k + 1) = (3 + 10 * k) + 10 by omega]
    constructor
    · rw [Finset.sum_range_add, ih.1, hw]
      omega
    · rw [Finset.sum_range_add, ih.2, hr]
      omega

/-- Every requested number of paid A-to-B-to-A cycles has a positive-probability
finite prefix of this one process, with explicit equal credit replenishment.
No body or world reset, and no clock substitution, occurs between cycles. -/
theorem arbitrary_relearning (k : ℕ) :
    0 < weight (process true true false) path (3 + 10 * k) ∧
    (∑ t ∈ range (3 + 10 * k), writeCost true (sense (path t).2 ()) ((path t).1 ())) =
        1 + 4 * k ∧
    (∑ t ∈ range (3 + 10 * k), topup true (sense (path t).2 ()) ((path t).1 ())) =
        2 + 4 * k := by
  refine ⟨?_, write_blocks k⟩
  exact Finset.prod_pos (fun t _ => by simpa only [tag, path_active t, ↓reduceIte] using path_step t)

private theorem law_step (P : Interaction.Machine S ℕ (Option Unit)) (n : ℕ) (x y : S) (e : Option Unit) :
    Energetics.law P initial n x * P.step n x e y ≤
      Energetics.law P initial (n + 1) y := by
  have hn := (Energetics.law_probability P initial n).1
  calc
    _ ≤ Energetics.law P initial n x * (∑ e, P.step n x e y) := by
      apply mul_le_mul_of_nonneg_left _ (hn x)
      exact Finset.single_le_sum (fun e _ => P.nonneg n x e y) (Finset.mem_univ _)
    _ ≤ _ := by
      apply Finset.single_le_sum _ (Finset.mem_univ x)
      intro z _
      exact mul_nonneg (hn z) (Finset.sum_nonneg (fun e _ => P.nonneg n z e y))

private theorem path_law_bound (P : Interaction.Machine S ℕ (Option Unit)) (p : ℕ → S)
    (hstart : p 0 = initial) (n : ℕ) : weight P p n ≤ Energetics.law P initial n (p n) := by
  induction n with
  | zero => simp [weight, Energetics.law, EventChannels.Kernel.stateLaw, hstart]
  | succ n ih =>
    calc
      _ = weight P p n * P.step n (p n) (tag (p n)) (p (n + 1)) := by
        simp only [weight, Finset.prod_range_succ]
      _ ≤ Energetics.law P initial n (p n) * P.step n (p n) (tag (p n)) (p (n + 1)) :=
        mul_le_mul_of_nonneg_right ih (P.nonneg _ _ _ _)
      _ ≤ _ := law_step P n (p n) (p (n + 1)) (tag (p n))

def failingPath (refill : Bool) : ℕ → S
  | 0 => ((fun _ => .search), .probeA)
  | 1 => ((fun _ => .search), .answerA)
  | 2 => ((fun _ => .aSpent), .feedA)
  | 3 => ((fun _ => if refill then .aReady else .aSpent), .feedB)
  | 4 => ((fun _ => if refill then .aReady else .aSpent), .missB)
  | 5 => ((fun _ => if refill then .aHeld else .vacuum), .probeB)
  | _ => ((fun _ => if refill then .aHeld else .vacuum), .trap)

set_option maxHeartbeats 1800000 in
-- Only six explicitly specified steps are needed for each paired control.
private theorem failure_weight (reopen refill : Bool) (hb : reopen = false ∨ refill = false) :
    weight (process reopen refill false) (failingPath refill) 6 = 1 / 2 := by
  cases reopen <;> cases refill <;> simp at hb
  all_goals norm_num [weight, tag, Finset.prod_range_succ, failingPath, process, MaterialSource.machine,
    Medium.jointEventKernel, Medium.eventWorldStep, Medium.eventStep, plant, world,
    sense, moves, policy_table, next, funext_iff]

private theorem trap_recovery (reopen refill : Bool) (t k : ℕ) (x : Unit → Cell) :
    Exchange.recovery (plant reopen refill) world (he reopen refill) (ledger reopen refill)
      false (fun _ => world.sense) t k (x, Env.trap) = 0 := by
  induction k generalizing t x with
  | zero => rfl
  | succ k ih =>
    simp only [Exchange.recovery, ledger, incoming, sense] at ih
    have hw (v : Env) : moves .trap x v = if v = .trap then 1 else 0 := by
      by_cases hq : x () = .vacuum <;> simp [moves, hq]
    simp only [Exchange.recovery, EventChannels.Kernel.firstHit, Fintype.sum_prod_type]
    simp [Medium.jointEventKernel, Medium.eventWorldStep, hw, Exchange.intake,
      ledger, sense, incoming, ih, ite_mul, zero_mul]

private theorem trap_alive (reopen refill : Bool) (n : ℕ) (x : Unit → Cell) :
    Exchange.alive (plant reopen refill) world (he reopen refill) (ledger reopen refill)
      false (fun _ => world.sense) n (x, Env.trap) = 0 := by
  classical
  unfold Exchange.alive Viability.alive
  rw [show world.horizon = 2 from rfl, trap_recovery]
  norm_num [world]

private theorem survival_mass_bound (reopen refill : Bool) (n : ℕ) (bad : S)
    (ha : Exchange.alive (plant reopen refill) world (he reopen refill) (ledger reopen refill)
      false (fun _ => world.sense) n bad = 0) :
    Exchange.survival (plant reopen refill) world (he reopen refill) (ledger reopen refill)
      false (fun _ => world.sense) n ≤
        1 - Energetics.law (process reopen refill false) initial n bad := by
  classical
  let P := process reopen refill false
  let alive := Exchange.alive (plant reopen refill) world (he reopen refill) (ledger reopen refill)
    false (fun _ => world.sense) n
  have hp := Energetics.law_probability P initial n
  have hb (z : S) : alive z ≤ 1 - (if z = bad then 1 else 0) := by
    by_cases hz : z = bad
    · subst z
      simpa [alive] using le_of_eq ha
    · simp only [hz, ↓reduceIte, sub_zero]
      exact (Viability.alive_bounds _ _ _).2
  have he : Exchange.survival (plant reopen refill) world (he reopen refill) (ledger reopen refill)
      false (fun _ => world.sense) n = ∑ z : S, alive z * Energetics.law P initial n z := by
    simp only [Fintype.sum_prod_type, P, actual_law, Bool.false_eq_true, ↓reduceIte]
    rfl
  rw [he]
  calc
    _ ≤ ∑ z : S, (1 - (if z = bad then 1 else 0)) * Energetics.law P initial n z :=
      Finset.sum_le_sum (fun z _ => mul_le_mul_of_nonneg_right (hb z) (hp.1 z))
    _ = _ := by
      simp only [sub_mul, one_mul, ite_mul, one_mul, zero_mul, Finset.sum_sub_distrib,
        Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte, hp.2]
      all_goals rfl

/-- Blocking either paid reopening or storage renewal creates a half-probability
loss of viability by tick six, under the same world, initial state and food rule. -/
theorem paired_failure (reopen refill : Bool) (hb : reopen = false ∨ refill = false) :
    Exchange.survival (plant reopen refill) world (he reopen refill) (ledger reopen refill)
      false (fun _ => world.sense) 6 ≤ 1 / 2 := by
  have hm := path_law_bound (process reopen refill false) (failingPath refill) rfl 6
  rw [failure_weight reopen refill hb] at hm
  have ha := trap_alive reopen refill 6 (fun _ => if refill then .aHeld else .vacuum)
  have hs := survival_mass_bound reopen refill 6 (failingPath refill 6) ha
  linarith

end CGB.RenewableLearningControls
