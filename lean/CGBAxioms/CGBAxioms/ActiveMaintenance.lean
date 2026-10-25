import CGBAxioms.EnergyBudget
import CGBAxioms.ActiveLoop

/-!
# Resource-dependent maintenance witness

A separate successor to the refuted ActiveLoop candidate. A/B are physical
states with one unit of reserve. A feeding event replaces one unit and spends
one unit; scarcity spends the remaining reserve and reaches absorbing vacuum.
These are declared carrier rules, not consequences of the six general clauses.
The existing gross-exchange viability observer and marginal-ghost cut are used.
-/

open Finset

namespace CGB.ActiveMaintenance

abbrev State := Fin 3
abbrev Stim := Fin 5
abbrev W := Fin 6

def cost (l : State) : ℕ := if l = 0 then 0 else 1

/-- Calm freezes; cues preserve reserve; feeding preserves the remembered
action; scarcity consumes reserve. Vacuum cannot act or be resurrected. -/
def next (s : Stim) (l : State) : State :=
  if l = 0 then 0 else if s = 1 then 1 else if s = 2 then 2
  else if s = 4 then 0 else l

noncomputable def body : Medium Unit State Unit Stim where
  near _ _ := True
  ident _ := 0
  boundary := univ
  content := cost
  parts _ := univ
  act _ s x := fun _ => next (s ()) (x ())
  rate _ s x := if s () = 0 ∨ x () = 0 then 0 else 1
  rate_nonneg _ _ _ := by split_ifs <;> norm_num
  init _ := 1
  calm _ := 0

theorem admission : body.Axioms ∧ body.EventClocked 1 := by
  have hr : body.Rigid := by
    intro σ _
    apply Equiv.ext
    intro u
    exact Subsingleton.elim _ _
  constructor
  · refine ⟨(body.p1a_iff_unique_vacuum).2 ?_, ?_, ?_, ?_, body.p2c_of_rigid hr, ?_⟩
    · refine ⟨0, rfl, ?_⟩
      intro l hl
      fin_cases l <;> norm_num [body, cost] at *
    · refine ⟨?_, by intros; trivial, ?_, ?_⟩
      · intro u v
        exact ⟨Equiv.refl Unit, by intros; rfl, Subsingleton.elim _ _⟩
      · intro e s x u hu
        simp [body] at hu
      · intro e s s' x y hh
        have hx : x = y := funext fun u => (hh u (by simp [body])).1
        have hs : s = s' := funext fun u => (hh u (by simp [body])).2
        subst y; subst s'
        exact ⟨rfl, fun _ _ => rfl⟩
    · apply body.p2a_of_every_event_open
      intro e
      change ¬ Disjoint (univ : Finset Unit) univ
      decide
    · refine ⟨fun _ => 1, fun _ => one_pos, fun x y => ?_⟩
      simp [Medium.jump, body]
    · exact ⟨body.onlyBoundarySenses_of_all_boundary rfl, body.initial_invariant_of_rigid hr⟩
  · refine ⟨by norm_num, ?_⟩
    intro s x
    simp only [body, Fintype.sum_unique]
    split_ifs <;> norm_num

def sense (w : W) (_ : Unit) : Stim :=
  if w = 1 then 1 else if w = 2 then 2 else if w = 3 ∨ w = 4 then 3
  else if w = 5 then 4 else 0

noncomputable def moves (w : W) (x : Unit → State) (v : W) : ℝ :=
  if w = 0 then (if v = 1 then 1 / 2 else if v = 2 then 1 / 2 else 0)
  else if w = 1 then (if v = 3 then 1 else 0)
  else if w = 2 then (if v = 4 then 1 else 0)
  else if w = 3 then (if v = (if x () = 1 then 3 else 5) then 1 else 0)
  else if w = 4 then (if v = (if x () = 2 then 4 else 5) then 1 else 0)
  else (if v = 5 then 1 else 0)

noncomputable abbrev world : World Unit State Stim where
  W := W
  sense := sense
  moves := moves
  moves_nonneg w x v := by unfold moves; split_ifs <;> norm_num
  moves_sum w x := by
    rw [ActiveLoop.fin6_sum]
    generalize hx : x () = a
    fin_cases w <;> fin_cases a <;> norm_num [moves, hx]
  start := 0
  quiet _ := 0
  clock := 1
  clock_pos := by norm_num
  horizon := 3
  recovery := 1 / 2
  recovery_pos := by norm_num

abbrev he : body.EventClocked world.clock := by
  change body.EventClocked 1
  exact admission.2
abbrev hc : body.Clocked world.clock := body.eventClocked_implies_clocked he

theorem reads_boundary : body.ReadsBoundary world := by
  intro w x y v h
  have hxy : x = y := funext fun u => h u (by simp [body])
  rw [hxy]

def incoming (s : Stim) (l : State) : ℕ := if s = 3 ∧ l ≠ 0 then 1 else 0
def outgoing (s : Stim) (l : State) : ℕ :=
  if (s = 3 ∨ s = 4) ∧ l ≠ 0 then 1 else 0

noncomputable def ledger : body.MaterialLedger where
  incoming _ s x := incoming (s ()) (x ())
  outgoing _ s x := outgoing (s ()) (x ())
  balance _ s x _ := by
    change cost (next (s ()) (x ())) + outgoing (s ()) (x ()) =
      cost (x ()) + incoming (s ()) (x ())
    generalize hs : s () = a
    generalize hx : x () = b
    fin_cases a <;> fin_cases b <;>
      norm_num [next, incoming, outgoing, cost]
  closed_zero a s x h _ := by
    have hn : ¬ body.Closed a := by simp [Medium.Closed, body]
    exact False.elim (hn h)

private theorem sum_configs (f : (Unit → State) → ℝ) :
    ∑ x, f x = f (fun _ => 0) + f (fun _ => 1) + f (fun _ => 2) := by
  calc
    _ = ∑ a : State, f (fun _ => a) :=
      Fintype.sum_equiv (Equiv.funUnique Unit State) _ _ (by intro x; congr 1)
    _ = _ := by
      have h : (univ : Finset State) = {0, 1, 2} := by decide
      rw [h]; simp; ring

private theorem kernel_table (s : Unit → Stim) (x y : State) :
    body.kernel 1 s (fun _ => x) (fun _ => y) =
      if y = next (s ()) x then 1 else 0 := by
  have hs : s = fun _ => s () := by funext u; cases u; rfl
  rw [hs]
  rw [← body.eventStep_forget, Fintype.sum_option]
  generalize h : s () = a
  fin_cases a <;> fin_cases x <;> fin_cases y <;>
    norm_num [Medium.eventStep, body, next, funext_iff, h]

private noncomputable def density : ℕ → State → W → ℝ
  | 0, a, w => if a = 1 ∧ w = 0 then 1 else 0
  | 1, a, w => if a = 1 ∧ (w = 1 ∨ w = 2) then 1 / 2 else 0
  | _ + 2, a, w => if (a = 1 ∧ w = 3) ∨ (a = 2 ∧ w = 4) then 1 / 2 else 0

set_option maxHeartbeats 1600000 in
-- Expand the finite transition rows in the all-time induction.
private theorem natural_law (n : ℕ) (a : State) (w : W) :
    body.law world (fun _ => world.sense) n (fun _ => a) w = density n a w := by
  induction n generalizing a w with
  | zero => simp [Medium.law, density, body, world, funext_iff]
  | succ n ih =>
    change (∑ x : Unit → State, ∑ v : W,
      body.law world (fun _ => world.sense) n x v * moves v x w *
        body.kernel 1 (sense v) x (fun _ => a)) = _
    simp only [sum_configs, ActiveLoop.fin6_sum, ih]
    rcases n with _ | n
    · fin_cases a <;> fin_cases w <;>
        norm_num [density, sense, moves, kernel_table, next]
    · rcases n with _ | n
      · fin_cases a <;> fin_cases w <;>
          norm_num [density, sense, moves, kernel_table, next]
      · fin_cases a <;> fin_cases w <;>
          norm_num [density, sense, moves, kernel_table, next]

private noncomputable abbrev recover (cut : Bool) (t k : ℕ) (a : State) (w : W) :=
  Exchange.recovery body world he ledger cut (fun _ => world.sense) t k ((fun _ => a), w)

set_option maxHeartbeats 1800000 in
-- Resolve labelled self-events against each finite body/world row.
private theorem recovery_step (t k : ℕ) (a : State) (w : W) :
    recover false t (k + 1) a w =
      ∑ v : W, moves w (fun _ => a) v *
        (if 0 < incoming (sense w ()) a then 1 else
          recover false (t + 1) k (next (sense w ()) a) v) := by
  simp only [recover, Exchange.recovery, EventChannels.Kernel.firstHit,
    Fintype.sum_prod_type]
  change (∑ e : Option Unit, ∑ y : Unit → State, ∑ v : W,
    (body.jointEventKernel world he false (fun _ => world.sense) t).step
      ((fun _ => a), w) e (y, v) *
      (if Exchange.intake body world ledger (fun _ => world.sense) t
        ((fun _ => a), w) e (y, v) then 1 else
          Exchange.recovery body world he ledger false (fun _ => world.sense)
            (t + 1) k (y, v))) = _
  simp only [Fintype.sum_option, Fintype.sum_unique, sum_configs, ActiveLoop.fin6_sum]
  fin_cases a <;> fin_cases w <;>
    norm_num [Medium.jointEventKernel, Medium.eventWorldStep, Medium.eventStep,
      Exchange.intake, ledger, incoming, body, world, sense, moves, next, funext_iff,
      recover]
  all_goals rfl

private noncomputable def prospect : ℕ → State → W → ℝ
  | 0, _, _ => 0
  | k + 1, a, w => ∑ v : W, moves w (fun _ => a) v *
      (if 0 < incoming (sense w ()) a then 1 else prospect k (next (sense w ()) a) v)

private theorem normal_recovery (t k : ℕ) (a : State) (w : W) :
    recover false t k a w = prospect k a w := by
  induction k generalizing t a w with
  | zero => rfl
  | succ k ih => simp only [recovery_step, prospect, ih]

private theorem natural_alive (t : ℕ) (a : State) (w : W) :
    Exchange.alive body world he ledger false (fun _ => world.sense) t
      ((fun _ => a), w) =
        if a ≠ 0 ∧ (1 : ℝ) / 2 ≤ prospect 3 a w then 1 else 0 := by
  classical
  unfold Exchange.alive Viability.alive
  have hp : (0 < body.total (fun _ => a)) ↔ a ≠ 0 := by
    fin_cases a <;> norm_num [Medium.total, body, cost]
  have hr := normal_recovery t 3 a w
  change Exchange.recovery body world he ledger false (fun _ => world.sense)
    t 3 ((fun _ => a), w) = prospect 3 a w at hr
  rw [show world.horizon = 3 from rfl, hr, hp]
  split_ifs <;> rfl

set_option maxHeartbeats 1800000 in
-- Evaluate the three-tick prospect after reducing the exact support.
theorem natural_survival (n : ℕ) :
    Exchange.survival body world he ledger false (fun _ => world.sense) n = 1 := by
  classical
  change (∑ x : Unit → State, ∑ w : W,
    Exchange.alive body world he ledger false (fun _ => world.sense) n (x, w) *
      body.law world (fun _ => world.sense) n x w) = 1
  simp only [sum_configs, ActiveLoop.fin6_sum, natural_law]
  rcases n with _ | n
  · norm_num [density]
    rw [natural_alive 0 1 0]
    norm_num [prospect, ActiveLoop.fin6_sum, moves, sense, next, incoming]
  · rcases n with _ | n
    · norm_num [density]
      rw [natural_alive 1 1 1, natural_alive 1 1 2]
      norm_num [prospect, ActiveLoop.fin6_sum, moves, sense, next, incoming]
    · norm_num [density]
      rw [natural_alive (n + 2) 1 3, natural_alive (n + 2) 2 4]
      norm_num [prospect, ActiveLoop.fin6_sum, moves, sense, next, incoming]

private theorem starve_world_step (cut : Bool) (t : ℕ) (x : Unit → State) (v : W) :
    body.eventWorldStep world cut (fun _ => world.sense) t (x, (5 : W)) v =
      if v = 5 then 1 else 0 := by
  cases cut with
  | false => simp [Medium.eventWorldStep, moves]
  | true =>
    change (∑ x, (∑ u, body.lawCut world (fun _ => world.sense) t x u) *
      (if v = 5 then 1 else 0)) = _
    rw [← Finset.sum_mul, body.lawCut_sum, one_mul]

set_option maxHeartbeats 1200000 in
-- Expand the finite event rows once per horizon induction step.
private theorem starve_recovery (cut : Bool) (t k : ℕ) (a : State) :
    recover cut t k a (5 : W) = 0 := by
  induction k generalizing t a with
  | zero => rfl
  | succ k ih =>
    simp only [recover, Exchange.recovery, EventChannels.Kernel.firstHit,
      Fintype.sum_prod_type]
    change (∑ e : Option Unit, ∑ y : Unit → State, ∑ v : W,
      (body.eventWorldStep world cut (fun _ => world.sense) t
        ((fun _ => a), (5 : W)) v * body.eventStep 1 (sense (5 : W)) (fun _ => a) e y) *
        (if Exchange.intake body world ledger (fun _ => world.sense) t
          ((fun _ => a), (5 : W)) e (y, v) then 1 else
          recover cut (t + 1) k (y ()) v)) = 0
    have hw (v : W) := starve_world_step cut t (fun _ => a) v
    simp_rw [hw]
    simp only [Fintype.sum_option,
      Fintype.sum_unique, sum_configs, ActiveLoop.fin6_sum]
    fin_cases a <;> norm_num [Medium.eventStep, body, world, sense, next, Exchange.intake,
      ledger, incoming, funext_iff, ih]

private theorem cut_move_early (t : ℕ) (x : Unit → State) (w v : W) (hw : w.val < 3) :
    Medium.cutMove world (body.lawCut world (fun _ => world.sense) t) x w v = moves w x v := by
  have hm (g : Unit → State) : world.moves w g v = moves w x v := by
    fin_cases w <;> norm_num at hw <;> simp [moves]
  change (∑ g, (∑ u, body.lawCut world (fun _ => world.sense) t g u) * world.moves w g v) = _
  calc
    _ = ∑ g, (∑ u, body.lawCut world (fun _ => world.sense) t g u) * moves w x v := by
      apply Finset.sum_congr rfl
      intro g _
      exact congrArg (fun q => (∑ u, body.lawCut world (fun _ => world.sense) t g u) * q) (hm g)
    _ = _ := by rw [← Finset.sum_mul, body.lawCut_sum, one_mul]

set_option maxHeartbeats 1800000 in
-- Check the first two cut-law rows before using the absorbing-mass bound.
private theorem cut_early_law (n : ℕ) (hn : n ≤ 2) (a : State) (w : W) :
    body.lawCut world (fun _ => world.sense) n (fun _ => a) w = density n a w := by
  induction n generalizing a w with
  | zero => simp [Medium.lawCut, density, body, world, funext_iff]
  | succ n ih =>
    have hi := ih (by omega)
    change (∑ x : Unit → State, ∑ v : W,
      body.lawCut world (fun _ => world.sense) n x v *
        Medium.cutMove world (body.lawCut world (fun _ => world.sense) n) x v w *
          body.kernel 1 (sense v) x (fun _ => a)) = _
    simp only [sum_configs, ActiveLoop.fin6_sum, hi]
    have hc : n = 0 ∨ n = 1 := by omega
    rcases hc with rfl | rfl <;> fin_cases a <;> fin_cases w <;>
      norm_num [density, cut_move_early, sense, moves, kernel_table, next]

private noncomputable def starveMass (n : ℕ) : ℝ :=
  ∑ x : Unit → State, body.lawCut world (fun _ => world.sense) n x (5 : W)

set_option maxHeartbeats 1800000 in
-- Evaluate the exact third-step ghost marginal, without sampling.
private theorem starve_mass_three : starveMass 3 = 1 / 2 := by
  change (∑ y : Unit → State, ∑ x : Unit → State, ∑ w : W,
    body.lawCut world (fun _ => world.sense) 2 x w *
      (∑ g : Unit → State, (∑ v : W,
        body.lawCut world (fun _ => world.sense) 2 g v) * moves w g 5) *
        body.kernel 1 (sense w) x y) = _
  have h2 (a : State) (w : W) := cut_early_law 2 (by omega) a w
  simp only [sum_configs, ActiveLoop.fin6_sum, h2]
  norm_num [density, moves, kernel_table, sense, next]

private theorem starve_mass_monotone (n : ℕ) : starveMass n ≤ starveMass (n + 1) := by
  have hnext : starveMass (n + 1) =
      ∑ x, ∑ w, body.lawCut world (fun _ => world.sense) n x w *
        Medium.cutMove world (body.lawCut world (fun _ => world.sense) n) x w (5 : W) := by
    unfold starveMass
    simp only [Medium.lawCut]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro x _
    rw [Finset.sum_comm]
    simp only [← Finset.mul_sum, Medium.kernel_sum, mul_one]
  rw [hnext]
  unfold starveMass
  apply Finset.sum_le_sum
  intro x _
  have h5 : Medium.cutMove world (body.lawCut world (fun _ => world.sense) n) x
      (5 : W) (5 : W) = 1 := by
    exact starve_world_step true n x 5
  calc
    _ = body.lawCut world (fun _ => world.sense) n x (5 : W) *
        Medium.cutMove world (body.lawCut world (fun _ => world.sense) n) x 5 5 := by
      rw [h5, mul_one]
    _ ≤ _ := Finset.single_le_sum (fun w _ => mul_nonneg
      (body.lawCut_nonneg_of_clocked world hc _ n x w)
      (body.eventWorldStep_nonneg world he true _ n (x, w) 5)) (Finset.mem_univ _)

private theorem cut_alive_bound (n : ℕ) (a : State) (w : W) :
    Exchange.alive body world he ledger true (fun _ => world.sense) n
      ((fun _ => a), w) ≤ if w = 5 then 0 else 1 := by
  classical
  by_cases hw : w = 5
  · subst w
    have hr := starve_recovery true n 3 a
    change Exchange.recovery body world he ledger true (fun _ => world.sense)
      n 3 ((fun _ => a), (5 : W)) = 0 at hr
    unfold Exchange.alive Viability.alive
    rw [show world.horizon = 3 from rfl, hr]
    norm_num [world]
  · simp only [hw, ↓reduceIte]
    exact (Viability.alive_bounds _ _ _).2

private theorem cut_survival_mass_bound (n : ℕ) :
    Exchange.survival body world he ledger true (fun _ => world.sense) n ≤ 1 - starveMass n := by
  classical
  have hb (x : Unit → State) (w : W) :
      Exchange.alive body world he ledger true (fun _ => world.sense) n (x, w) ≤
        if w = 5 then 0 else 1 := by
    have hx : x = fun _ => x () := by funext u; cases u; rfl
    rw [hx]
    exact cut_alive_bound n (x ()) w
  calc
    _ ≤ ∑ x : Unit → State, ∑ w : W, (if w = 5 then 0 else 1) *
        body.lawCut world (fun _ => world.sense) n x w := by
      apply Finset.sum_le_sum
      intro x _
      apply Finset.sum_le_sum
      intro w _
      exact mul_le_mul_of_nonneg_right (hb x w)
        (body.lawCut_nonneg_of_clocked world hc _ n x w)
    _ = _ := by
      have hs := body.lawCut_sum world (fun _ => world.sense) n
      change (∑ x : Unit → State, ∑ w : W,
        body.lawCut world (fun _ => world.sense) n x w) = 1 at hs
      unfold starveMass
      simp only [sum_configs, ActiveLoop.fin6_sum] at hs ⊢
      norm_num at hs ⊢
      linarith

theorem cut_survival (n : ℕ) (hn : 3 ≤ n) :
    Exchange.survival body world he ledger true (fun _ => world.sense) n ≤ 1 / 2 := by
  have hm : (1 : ℝ) / 2 ≤ starveMass n := by
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hn
    induction k with
    | zero => simp [starve_mass_three]
    | succ k ih =>
      have hh := starve_mass_monotone (3 + k)
      exact le_trans (ih (by omega)) hh
  linarith [cut_survival_mass_bound n]

/-- The existing gross-exchange targets, for ALL future times. This is a
maintenance subtarget witness, not an encoded or cognitive life candidate. -/
theorem maintenance_witness : Exchange.Persists body world he ledger 0 ∧
    Exchange.SelfMaintains body world he ledger := by
  constructor
  · refine ⟨by norm_num, fun n => ?_⟩
    rw [natural_survival]
    norm_num
  · refine ⟨3, (1 : ℝ) / 2, by norm_num, ?_⟩
    intro n hn
    rw [natural_survival]
    linarith [cut_survival n hn]

/-- Scarcity really changes the body reserve, under either process; vacuum
then has no action in any stimulus. The event clock and ghost future are kept. -/
theorem starvation_depletes (cut : Bool) (t : ℕ) (a : State) :
    (body.jointEventKernel world he cut (fun _ => world.sense) t).step
      ((fun _ => a), (5 : W)) (if a = 0 then none else some ())
        ((fun _ => 0), (5 : W)) = 1 ∧
    next 4 a = 0 ∧ cost (next 4 a) = 0 ∧ (∀ s, next s 0 = 0) := by
  constructor
  · change body.eventWorldStep world cut (fun _ => world.sense) t
      ((fun _ => a), (5 : W)) 5 *
        body.eventStep 1 (sense (5 : W)) (fun _ => a)
          (if a = 0 then none else some ()) (fun _ => 0) = 1
    rw [starve_world_step]
    fin_cases a <;> norm_num [Medium.eventStep, body, sense, next, funext_iff]
  · fin_cases a <;> simp [next, cost]

/-- Normal maintenance consumes and imports a positive unit EVERY active tick;
this demand is a property of this mechanism, not a universal life axiom. -/
theorem sustained_exchange (n : ℕ) (hn : 2 ≤ n) :
    (∑ x : Unit → State, ∑ w : W,
      body.law world (fun _ => world.sense) n x w * (incoming (sense w ()) (x ()) : ℝ)) = 1 ∧
    (∑ x : Unit → State, ∑ w : W,
      body.law world (fun _ => world.sense) n x w * (outgoing (sense w ()) (x ()) : ℝ)) = 1 := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hn
  simp only [sum_configs, ActiveLoop.fin6_sum, natural_law]
  constructor <;> norm_num [Nat.add_comm 2 k, density, incoming, outgoing, sense]

end CGB.ActiveMaintenance
