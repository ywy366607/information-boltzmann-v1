import CGBAxioms.GenerationSelfProduction

/-!
# Historical memory in the unchanged grace process

The natural law at tick four has one body state and three world states.
A single actual observation distinguishes feed from miss with one bit of
conditional information. Quiet freezes that bit; this is `Remembers`, not
`Lives` or historical persistence, and does not identify this body with GDN.
-/

open Finset

namespace CGB.GraceMemory
open RenewableLearning

private theorem sum_cells (f : Cell → ℝ) : ∑ q, f q =
    f .vacuum + f .search + f .aSpent + f .bSpent +
      f .aReady + f .bReady + f .aHeld + f .bHeld := by
  have h : (univ : Finset Cell) =
      {.vacuum, .search, .aSpent, .bSpent, .aReady, .bReady, .aHeld, .bHeld} := by decide
  rw [h]; simp; ring

private theorem sum_configs (f : (Unit → Cell) → ℝ) :
    ∑ x, f x = f (fun _ => .vacuum) + f (fun _ => .search) +
      f (fun _ => .aSpent) + f (fun _ => .bSpent) + f (fun _ => .aReady) +
      f (fun _ => .bReady) + f (fun _ => .aHeld) + f (fun _ => .bHeld) := by
  rw [Fintype.sum_equiv (Equiv.funUnique Unit Cell) _ (fun q => f (fun _ => q))
    (by intro x; congr 1)]
  exact sum_cells _

private theorem sum_env (f : Env → ℝ) : ∑ w, f w =
    f .feedA + f .feedB + f .missA + f .missB + f .probeA + f .probeB +
      f .answerA + f .answerB + f .trap := by
  have h : (univ : Finset Env) =
      {.feedA, .feedB, .missA, .missB, .probeA, .probeB, .answerA, .answerB, .trap} := by decide
  rw [h]; simp; ring

/-- The original body kernel, obtained by forgetting its actual event label. -/
theorem kernel_table (s : Unit → Signal) (q r : Cell) :
    GraceRecovery.body.kernel 1 s (fun _ => q) (fun _ => r) =
      if r = next true true (s ()) q then 1 else 0 := by
  have hs : s = fun _ => s () := by funext u; cases u; rfl
  rw [hs, ← GraceRecovery.body.eventStep_forget, Fintype.sum_option, Fintype.sum_unique]
  by_cases hc : s () = .calm ∨ q = .vacuum
  · have hn : next true true (s ()) q = q := by
      rcases hc with h | h
      · by_cases hq : q = .vacuum <;> simp [next, h, hq]
      · simp [next, h]
    simp [Medium.eventStep, GraceRecovery.body, plant, hc, hn, funext_iff]
    simp only [eq_comm]
  · simp [Medium.eventStep, GraceRecovery.body, plant, hc, funext_iff]
    simp only [eq_comm]

private theorem policy_table (q : Cell) : policy q =
    match q with
    | .aSpent | .aReady | .aHeld => 0
    | .bSpent | .bReady | .bHeld => 1
    | _ => 2 := by cases q <;> decide

/-- Sparse rows of the actual natural law, not probabilities assigned to paths. -/
noncomputable def early : ℕ → Cell → Env → ℝ
  | 0, q, w => if q = .search ∧ w = .probeA then 1 else 0
  | 1, q, w => if q = .search ∧ w = .answerA then 1 else 0
  | 2, q, w => if q = .aSpent ∧ w = .feedA then 1 else 0
  | 3, q, w => if q = .aReady ∧ (w = .feedA ∨ w = .feedB) then 1 / 2 else 0
  | 4, q, w => if q = .aReady then
      (if w = .missB then 1 / 2 else if w = .feedA ∨ w = .feedB then 1 / 4 else 0) else 0
  | _, _, _ => 0

set_option maxHeartbeats 2600000 in
-- Check only the four finite sparse transitions, not complete histories.
/-- Each step rewrites the preceding sparse row before any finite summation;
there is no expansion of a Cartesian product of histories. -/
theorem natural_law_early (n : ℕ) (hn : n ≤ 4) (q : Cell) (w : Env) :
    GraceRecovery.body.law GraceRecovery.world (fun _ => GraceRecovery.world.sense)
      n (fun _ => q) w = early n q w := by
  induction n generalizing q w with
  | zero =>
    simp [Medium.law, early, GraceRecovery.body, plant, GraceRecovery.world,
      GraceRecovery.worldWith, RenewableLearning.world, funext_iff]
  | succ n ih =>
    have hi := ih (by omega)
    change (∑ x : Unit → Cell, ∑ v : Env,
      GraceRecovery.body.law GraceRecovery.world (fun _ => GraceRecovery.world.sense) n x v *
        GraceRecovery.moves v x w *
          GraceRecovery.body.kernel 1 (sense v) x (fun _ => q)) = _
    simp only [sum_configs, sum_env, hi]
    have hc : n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 := by omega
    rcases hc with rfl | rfl | rfl | rfl <;> simp only [early, ↓reduceIte,
      zero_mul, zero_add, add_zero, one_mul] <;> cases q <;> cases w <;>
      norm_num [GraceRecovery.moves, RenewableLearning.moves, policy_table,
        kernel_table, sense, next]

/-- The same row is the law of the registered event process. -/
theorem actual_event_law_four (q : Cell) (w : Env) :
    Energetics.law (GraceRecovery.process false GraceRecovery.natural) initial 4
      ((fun _ => q), w) = early 4 q w := by
  rw [GraceRecoveryAccounting.actual_law]
  exact natural_law_early 4 (by omega) q w

private noncomputable def writeJoint (x₀ : Unit → Cell) (w₀ : Env) (x : Unit → Cell) : ℝ :=
  if x₀ = (fun _ => .aReady) then
    if w₀ = .missB ∧ x = (fun _ => .search) then 1 / 2
    else if (w₀ = .feedA ∨ w₀ = .feedB) ∧ x = (fun _ => .aReady) then 1 / 4
    else 0
  else 0

/-- Marginalize the actual world transition, whose row sums to one. -/
private theorem one_write_marginal (n : ℕ) (x₀ x : Unit → Cell) (w₀ : Env) :
    (∑ w, GraceRecovery.body.memLaw GraceRecovery.world n 1 x w x₀ w₀) =
      GraceRecovery.body.law GraceRecovery.world (fun _ => GraceRecovery.world.sense) n x₀ w₀ *
        GraceRecovery.body.kernel 1 (sense w₀) x₀ x := by
  simp only [Medium.memLaw, ite_true, ite_and, mul_ite, ite_mul, zero_mul, mul_zero]
  simp only [Finset.sum_ite_irrel, Finset.sum_const_zero, Finset.sum_ite_eq',
    Finset.mem_univ, ↓reduceIte]
  rw [← Finset.sum_mul, ← Finset.mul_sum, GraceRecovery.world.moves_sum, mul_one]

set_option maxHeartbeats 800000 in
-- Normalize the finite feed/miss observation row.
/-- The three nonzero tagged outcomes have masses 1/4, 1/4 and 1/2. -/
theorem one_write_joint (x₀ x : Unit → Cell) (w₀ : Env) :
    (∑ w, GraceRecovery.body.memLaw GraceRecovery.world 4 1 x w x₀ w₀) =
      writeJoint x₀ w₀ x := by
  have hx₀ : x₀ = fun _ => x₀ () := by funext u; cases u; rfl
  have hx : x = fun _ => x () := by funext u; cases u; rfl
  rw [hx₀, hx, one_write_marginal, natural_law_early 4 (by omega), kernel_table]
  generalize x₀ () = q₀
  generalize x () = q
  by_cases hq₀ : q₀ = .aReady
  · subst q₀
    cases w₀ <;> cases q <;>
      norm_num [early, writeJoint, sense, next, funext_iff]
  · simp [early, writeJoint, funext_iff, hq₀]

set_option maxHeartbeats 800000 in
-- Evaluate just the three nonzero summands and their exact marginals.
/-- Exact conditional mutual information in nats: one bit, not a support argument. -/
theorem one_write_information :
    Medium.condMutualInfo (fun x₀ w₀ x => ∑ w,
      GraceRecovery.body.memLaw GraceRecovery.world 4 1 x w x₀ w₀) = Real.log 2 := by
  have hp : (fun x₀ w₀ x => ∑ w,
      GraceRecovery.body.memLaw GraceRecovery.world 4 1 x w x₀ w₀) = writeJoint := by
    funext x₀ w₀ x
    exact one_write_joint x₀ x w₀
  rw [hp]
  unfold Medium.condMutualInfo
  rw [sum_configs]
  simp only [writeJoint, funext_iff, forall_const, ↓reduceIte]
  simp only [sum_env, sum_configs]
  norm_num
  ring

/-- The original `Life.Remembers` predicate holds for the unchanged grace body/world. -/
theorem remembers : GraceRecovery.body.Remembers GraceRecovery.world := by
  apply SelfProduction.grace_historical_memory_iff_one_write.mpr
  refine ⟨4, ?_⟩
  rw [one_write_information]
  exact Real.log_pos (by norm_num)

/-- Quiet retains exactly the same bit at every later delay. -/
theorem retained_information (m : ℕ) :
    Medium.condMutualInfo (fun x₀ w₀ x => ∑ w,
      GraceRecovery.body.memLaw GraceRecovery.world 4 (m + 1) x w x₀ w₀) = Real.log 2 := by
  have hid : ∀ x y, GraceRecovery.body.kernel GraceRecovery.world.clock
      GraceRecovery.world.quiet x y = if x = y then 1 else 0 := by
    intro x y
    simp [Medium.kernel, Medium.jump, GraceRecovery.body, plant]
  have h (x₀ : Unit → Cell) (w₀ : Env) (x : Unit → Cell) :=
    SelfProduction.quiet_identity_memory_marginal GraceRecovery.body GraceRecovery.world
      hid 4 x x₀ w₀ m
  simp_rw [h]
  exact one_write_information

/-- Combine the numerical memory witness with the already-proved supported
histories whose stored states select different actions. -/
theorem remembers_with_reachable_action_change :
    GraceRecovery.body.Remembers GraceRecovery.world ∧
      ∃ p q : ℕ → S,
        0 < RenewableLearningControls.weight
          (GraceRecovery.process false GraceRecovery.natural) p 6 ∧
        0 < RenewableLearningControls.weight
          (GraceRecovery.process false GraceRecovery.natural) q 6 ∧
        (p 5).1 = (q 5).1 ∧
        policy ((p 6).1 ()) ≠ policy ((q 6).1 ()) := by
  refine ⟨remembers,
    SelfProduction.renewableObservableFromBirth (fun _ => false),
    SelfProduction.renewableObservableFromBirth (fun _ => true), ?_⟩
  have h := SelfProduction.grace_reachable_memory_changes_action
  exact ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.2.2⟩

end CGB.GraceMemory
