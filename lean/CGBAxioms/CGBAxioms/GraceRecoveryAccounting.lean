import CGBAxioms.GraceRecovery
import CGBAxioms.RenewableLearningCarrier

/-!
# Unchanged paid learning and actual source accounting through the grace world

All ledgers use the actual registered protocol, including the interrupted tick.
No food or write credit is added to the calm slot. The same finite identity
realization is instantiated for both normal/cut and interrupted protocols.
-/

open Finset

namespace CGB.GraceRecoveryAccounting
open RenewableLearning MaterialSource Energetics

noncomputable abbrev world := GraceRecovery.world
noncomputable abbrev body := GraceRecovery.body
abbrev ec := GraceRecovery.ec
noncomputable abbrev process := GraceRecovery.process

noncomputable def declared (sched : ℕ → Env → Unit → Signal) (f : Signal → Cell → ℕ)
    (t : ℕ) (z : S) (e : Option Unit) (_z' : S) : ℕ :=
  e.elim 0 (fun _ => f (sched t z.2 ()) (z.1 ()))

private theorem body_supported (cut : Bool) (sched : ℕ → Env → Unit → Signal)
    (t : ℕ) (z z' : S) (e : Option Unit)
    (hp : 0 < (process cut sched).step t z e z') :
    0 < body.eventStep 1 (sched t z.2) z.1 e z'.1 := by
  by_contra hn
  exact (not_lt_of_ge (mul_nonpos_of_nonneg_of_nonpos
    (body.eventWorldStep_nonneg world ec cut sched t z z'.2) (le_of_not_gt hn))) hp

noncomputable def rules (cut : Bool) (sched : ℕ → Env → Unit → Signal) :
    BufferedMaterial.Rules (process cut sched) where
  content z := cost (z.1 ())
  incoming := declared sched (incoming true)
  outgoing := declared sched (outgoing true true)
  bounded z := by cases z.1 () <;> decide
  intake_bound t z e z' _ := by
    cases e with
    | none => simp [declared]
    | some u =>
      change incoming true (sched t z.2 ()) (z.1 ()) ≤ 4
      generalize sched t z.2 () = s; generalize z.1 () = q
      cases s <;> cases q <;> decide
  release_bound t z e z' _ := by
    cases e with
    | none => simp [declared]
    | some u =>
      change outgoing true true (sched t z.2 ()) (z.1 ()) ≤ cost (z.1 ())
      generalize sched t z.2 () = s; generalize z.1 () = q
      cases s <;> cases q <;> decide
  balance t z e z' hp := by
    cases e with
    | none =>
      have hb := body_supported cut sched t z z' none hp
      have hx : z.1 = z'.1 := by by_contra hn; simp [Medium.eventStep, hn] at hb
      simp [declared, ← hx]
    | some u =>
      have hx := (GraceRecovery.supported cut sched t z z' (some u) hp).2.1
      have hb := (local_budget true true (sched t z.2 ()) (z.1 ())).1
      rw [← hx] at hb
      exact hb

private theorem credit_step (cut : Bool) (sched : ℕ → Env → Unit → Signal)
    (t : ℕ) (z z' : S) (e : Option Unit)
    (hp : 0 < (process cut sched).step t z e z') :
    credit (z'.1 ()) + declared sched (fun s q => writeCost true s q + discarded s q) t z e z' =
      credit (z.1 ()) + declared sched (topup true) t z e z' := by
  cases e with
  | none =>
    have hb := body_supported cut sched t z z' none hp
    have hx : z.1 = z'.1 := by by_contra hn; simp [Medium.eventStep, hn] at hb
    simp [declared, ← hx]
  | some u =>
    have hx := (GraceRecovery.supported cut sched t z z' (some u) hp).2.1
    have hb := (local_budget true true (sched t z.2 ()) (z.1 ())).2
    rw [← hx] at hb
    simpa only [declared, Option.elim_some, Nat.add_assoc] using hb

/-- Exact budgets on EVERY supported finite path, including interrupted and
cut paths. Destroyed credit is separate from completed candidate writes. -/
theorem path_budgets (cut : Bool) (sched : ℕ → Env → Unit → Signal)
    (p : ℕ → S) (events : ℕ → Option Unit)
    (hp : ∀ t, 0 < (process cut sched).step t (p t) (events t) (p (t + 1))) (n : ℕ) :
    (cost ((p n).1 ()) + ∑ t ∈ range n,
      declared sched (outgoing true true) t (p t) (events t) (p (t + 1)) =
    cost ((p 0).1 ()) + ∑ t ∈ range n,
      declared sched (incoming true) t (p t) (events t) (p (t + 1))) ∧
    (credit ((p n).1 ()) + ∑ t ∈ range n,
      declared sched (fun s q => writeCost true s q + discarded s q) t (p t) (events t) (p (t + 1)) =
    credit ((p 0).1 ()) + ∑ t ∈ range n,
      declared sched (topup true) t (p t) (events t) (p (t + 1))) := by
  induction n with
  | zero => simp
  | succ n ih =>
    have hb := (rules cut sched).balance n (p n) (events n) (p (n + 1)) (hp n)
    have hc := credit_step cut sched n (p n) (p (n + 1)) (events n) (hp n)
    change cost ((p (n + 1)).1 ()) +
      declared sched (outgoing true true) n (p n) (events n) (p (n + 1)) =
      cost ((p n).1 ()) + declared sched (incoming true) n (p n) (events n) (p (n + 1)) at hb
    simp only [Finset.sum_range_succ]
    constructor <;> omega

abbrev Carrier (cut : Bool) (sched : ℕ → Env → Unit → Signal) :=
  BufferedMaterial.State (rules cut sched)
noncomputable abbrev physical (cut : Bool) (sched : ℕ → Env → Unit → Signal) :=
  BufferedMaterial.physical (rules cut sched)
noncomputable abbrev project (cut : Bool) (sched : ℕ → Env → Unit → Signal) :=
  BufferedMaterial.project (rules cut sched)
noncomputable abbrev owner (cut : Bool) (sched : ℕ → Env → Unit → Signal) :=
  BufferedMaterial.owner (rules cut sched)
noncomputable def initialCarrier (cut : Bool) (sched : ℕ → Env → Unit → Signal) : Carrier cut sched :=
  BufferedMaterial.seed (rules cut sched) initial

theorem realized (cut : Bool) (sched : ℕ → Env → Unit → Signal) :
    Realization (physical cut sched) (process cut sched) (owner cut sched) (project cut sched)
      (fun z => body.total z.1) (declared sched (incoming true))
      (declared sched (outgoing true true)) world.clock := by
  have hc : (fun z : S => body.total z.1) = (rules cut sched).content := by
    funext z; simp [Medium.total, body, plant, rules]
  rw [hc]
  exact BufferedMaterial.realized (rules cut sched) world.clock world.clock_pos

noncomputable def release (cut : Bool) (sched : ℕ → Env → Unit → Signal)
    (_t : ℕ) (r : Carrier cut sched) (_e : Option Unit) (r' : Carrier cut sched) : ℝ :=
  (moved (owner cut sched r) (owner cut sched r') .body .sink : ℝ)

noncomputable def sourceAccount (cut : Bool) (sched : ℕ → Env → Unit → Signal) :
    Account (physical cut sched) where
  stored _ r := (count (owner cut sched r) .body : ℝ) + (count (owner cut sched r) .source : ℝ)
  drive _ r _ r' := (moved (owner cut sched r) (owner cut sched r') .sink .source : ℝ)
  load _ _ _ _ := 0
  matter _ _ _ _ := 0
  heat t r e r' := -release cut sched t r e r'
  balance t r e r' hp := by
    have h := BufferedMaterial.delivery (rules cut sched) t r r' e hp
    have hb := (MaterialSource.balance (owner cut sched r) (owner cut sched r') h.1).1
    have hs : count (owner cut sched r) .source = 4 := r.property.2.1
    have hs' : count (owner cut sched r') .source = 4 := r'.property.2.1
    have hr : moved (owner cut sched r) (owner cut sched r') .sink .source =
        moved (owner cut sched r) (owner cut sched r') .source .body := h.2.2.2.trans h.2.1.symm
    have hn : count (owner cut sched r') .body + count (owner cut sched r') .source +
        moved (owner cut sched r) (owner cut sched r') .body .sink =
      count (owner cut sched r) .body + count (owner cut sched r) .source +
        moved (owner cut sched r) (owner cut sched r') .sink .source := by omega
    have he : (count (owner cut sched r') .body : ℝ) + (count (owner cut sched r') .source : ℝ) +
        (moved (owner cut sched r) (owner cut sched r') .body .sink : ℝ) =
      (count (owner cut sched r) .body : ℝ) + (count (owner cut sched r) .source : ℝ) +
        (moved (owner cut sched r) (owner cut sched r') .sink .source : ℝ) := by exact_mod_cast hn
    unfold release
    linarith

theorem initial_energy (cut : Bool) (sched : ℕ → Env → Unit → Signal) :
    (sourceAccount cut sched).stored 0 (initialCarrier cut sched) = 6 := by
  change (count (owner cut sched (initialCarrier cut sched)) .body : ℝ) +
    (count (owner cut sched (initialCarrier cut sched)) .source : ℝ) = 6
  rw [(realized cut sched).body_count]
  have hs : count (owner cut sched (initialCarrier cut sched)) .source = 4 :=
    (initialCarrier cut sched).property.2.1
  rw [hs]
  norm_num [initialCarrier, BufferedMaterial.seed, project, BufferedMaterial.project,
    Medium.total, body, plant, initial, cost, buffer, credit]

theorem source_energy_budget (cut : Bool) (sched : ℕ → Env → Unit → Signal) (n : ℕ) :
    mean (law (physical cut sched) (initialCarrier cut sched) n) ((sourceAccount cut sched).stored n) +
      ∑ t ∈ range n, flux (physical cut sched) (initialCarrier cut sched) (release cut sched) t =
    6 + ∑ t ∈ range n,
      flux (physical cut sched) (initialCarrier cut sched) (sourceAccount cut sched).drive t := by
  induction n with
  | zero =>
    simp only [Finset.range_zero, Finset.sum_empty, add_zero]
    simpa only [mean, law, EventChannels.Kernel.stateLaw, ite_mul, one_mul, zero_mul,
      Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte] using initial_energy cut sched
  | succ n ih =>
    have hb := expected_first_law (physical cut sched) (sourceAccount cut sched) (initialCarrier cut sched) n
    have hz : flux (physical cut sched) (initialCarrier cut sched) (fun _ _ _ _ => 0) n = 0 := by
      simp [flux]
    have hh : flux (physical cut sched) (initialCarrier cut sched)
        (fun t r e r' => -release cut sched t r e r') n =
      -flux (physical cut sched) (initialCarrier cut sched) (release cut sched) n := by
      simp only [flux, mul_neg, Finset.sum_neg_distrib]
    change _ + flux (physical cut sched) (initialCarrier cut sched) (fun _ _ _ _ => 0) n =
      _ + _ + flux (physical cut sched) (initialCarrier cut sched) (fun _ _ _ _ => 0) n +
        flux (physical cut sched) (initialCarrier cut sched) (fun t r e r' => -release cut sched t r e r') n at hb
    rw [hz, hh] at hb
    simp only [Finset.sum_range_succ]
    linarith

private theorem old_path_good (n : ℕ) :
    GraceRecovery.Good ((RenewableLearningControls.path n).1 ()) (RenewableLearningControls.path n).2 := by
  rcases n with _ | n
  · simp [GraceRecovery.Good, RenewableLearningLife.Good, RenewableLearningControls.path]
  · rcases n with _ | n
    · simp [GraceRecovery.Good, RenewableLearningLife.Good, RenewableLearningControls.path]
    · rcases n with _ | n
      · simp [GraceRecovery.Good, RenewableLearningLife.Good, RenewableLearningControls.path]
      · change GraceRecovery.Good
          ((RenewableLearningControls.cycle (RenewableLearningControls.index n)).1 ())
          (RenewableLearningControls.cycle (RenewableLearningControls.index n)).2
        generalize RenewableLearningControls.index n = i
        fin_cases i <;> simp [GraceRecovery.Good, RenewableLearningLife.Good,
          RenewableLearningControls.cycle]

theorem normal_step_unchanged (t : ℕ) (z z' : S) (e : Option Unit)
    (hg : GraceRecovery.Good (z.1 ()) z.2) :
    (process false GraceRecovery.natural).step t z e z' =
      (RenewableLearning.process true true false).step t z e z' := by
  change GraceRecovery.moves z.2 z.1 z'.2 * body.eventStep 1 (sense z.2) z.1 e z'.1 = _
  have he : z.1 = fun _ => z.1 () := by funext u; cases u; rfl
  have hw := GraceRecovery.moves_unchanged_on_good (z.1 ()) z.2 z'.2 hg
  rw [← he] at hw
  rw [hw]
  rfl

theorem arbitrary_relearning (k : ℕ) :
    0 < RenewableLearningControls.weight (process false GraceRecovery.natural)
      RenewableLearningControls.path (3 + 10 * k) ∧
    (∑ t ∈ range (3 + 10 * k), writeCost true (sense (RenewableLearningControls.path t).2 ())
      ((RenewableLearningControls.path t).1 ())) = 1 + 4 * k ∧
    (∑ t ∈ range (3 + 10 * k), topup true (sense (RenewableLearningControls.path t).2 ())
      ((RenewableLearningControls.path t).1 ())) = 2 + 4 * k := by
  have h := RenewableLearningControls.arbitrary_relearning k
  have hw : RenewableLearningControls.weight (process false GraceRecovery.natural)
      RenewableLearningControls.path (3 + 10 * k) =
      RenewableLearningControls.weight (RenewableLearning.process true true false)
        RenewableLearningControls.path (3 + 10 * k) := by
    unfold RenewableLearningControls.weight
    apply Finset.prod_congr rfl
    intro t _
    exact normal_step_unchanged _ _ _ _ (old_path_good t)
  exact ⟨by rw [hw]; exact h.1, h.2⟩

noncomputable def intake (cut : Bool) (sched : ℕ → Env → Unit → Signal)
    (_t : ℕ) (r : Carrier cut sched) (_e : Option Unit) (r' : Carrier cut sched) : Bool :=
  decide (0 < moved (owner cut sched r) (owner cut sched r') .source .body)

theorem recovery_preserved (cut : Bool) (sched : ℕ → Env → Unit → Signal)
    (t k : ℕ) (r : Carrier cut sched) :
    EventChannels.Kernel.firstHit (events (physical cut sched)) (intake cut sched) t k r =
      Exchange.recovery body world ec (ledger true true) cut sched t k (project cut sched r) := by
  apply RecoveryTransport.firstHit_eq (realized cut sched).process.representation
  intro t r e r' hp
  have h := (realized cut sched).transport t r e r' hp
  simp only [intake, h.2.1, declared, Exchange.intake, ledger]
  cases e with
  | none => rfl
  | some u => cases u; rfl

theorem actual_law (cut : Bool) (sched : ℕ → Env → Unit → Signal) (n : ℕ) (z : S) :
    law (process cut sched) initial n z = if cut then
      body.lawCut world sched n z.1 z.2 else body.law world sched n z.1 z.2 := by
  have hi : (fun z : S => if z = initial then (1 : ℝ) else 0) =
      (fun z => if z.1 = body.init ∧ z.2 = world.start then 1 else 0) := by
    funext z
    simp [initial, body, plant, world, GraceRecovery.world, GraceRecovery.worldWith, Prod.ext_iff]
  change EventChannels.Kernel.stateLaw (body.jointEventKernel world ec cut sched)
    (fun z : S => if z = initial then (1 : ℝ) else 0) n z = _
  rw [hi]
  exact body.jointEventKernel_stateLaw world ec cut sched n z

noncomputable def score (cut : Bool) (sched : ℕ → Env → Unit → Signal) (n : ℕ) : ℝ :=
  mean (law (physical cut sched) (initialCarrier cut sched) n) (fun r =>
    if 0 < count (owner cut sched r) .body ∧ world.recovery ≤
      EventChannels.Kernel.firstHit (events (physical cut sched)) (intake cut sched)
        n world.horizon r then 1 else 0)

theorem score_preserved (cut : Bool) (sched : ℕ → Env → Unit → Signal) (n : ℕ) :
    score cut sched n = Exchange.survival body world ec (ledger true true) cut sched n := by
  have point (r : Carrier cut sched) :
      (if 0 < count (owner cut sched r) .body ∧ world.recovery ≤
        EventChannels.Kernel.firstHit (events (physical cut sched)) (intake cut sched)
          n world.horizon r then (1 : ℝ) else 0) =
      Exchange.alive body world ec (ledger true true) cut sched n (project cut sched r) := by
    rw [(realized cut sched).body_count, recovery_preserved]
    unfold Exchange.alive Viability.alive
    split_ifs <;> rfl
  unfold score
  simp_rw [point]
  rw [RecoveryTransport.mean_eq (realized cut sched).process.representation]
  change (∑ z : S, law (process cut sched) initial n z *
    Exchange.alive body world ec (ledger true true) cut sched n z) = _
  simp only [Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro x _
  apply Finset.sum_congr rfl
  intro w _
  rw [GraceRecoveryAccounting.actual_law cut sched n (x,w)]
  exact mul_comm _ _

theorem physical_one_tick_survival (n₀ n : ℕ) :
    score false (GraceRecovery.interrupted n₀) n = 1 := by
  rw [score_preserved]
  exact GraceRecovery.one_tick_survival n₀ n

theorem calm_has_no_intake (n₀ : ℕ) (z z' : S) (e : Option Unit) :
    declared (GraceRecovery.interrupted n₀) (incoming true) n₀ z e z' = 0 := by
  cases e <;> simp [declared, GraceRecovery.interrupted, Medium.interrupt, body, plant, incoming]

end CGB.GraceRecoveryAccounting
