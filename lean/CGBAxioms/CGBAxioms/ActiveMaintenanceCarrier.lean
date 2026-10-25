import CGBAxioms.ActiveMaintenanceEnergy

/-!
# Three-token realization of the maintenance process

One token is in the body, one in the accessible source, one in the return sink.
Feeding rotates these fixed identities; scarcity releases the body token.
Recycling sink -> source needs one explicit unit of external recharge energy.
The extra phase is carrier bookkeeping: exact timed refinement proves it cannot
change the body's normal/cut event law or provide hidden action memory.
-/

open Finset

namespace CGB.ActiveMaintenanceCarrier

open ActiveMaintenance ActiveMaintenanceEnergy MaterialSource Energetics

abbrev Phase := Fin 3
abbrev R := S × Phase

def rotate (p : Phase) : Phase := if p = 0 then 1 else if p = 1 then 2 else 0

def phaseNext (s : Stim) (a : State) (e : Option Unit) (p : Phase) : Phase :=
  if e.isSome ∧ 0 < incoming s a then rotate p else p

def owners (a : State) (p c : Phase) : Place :=
  if c = p then (if a = 0 then .sink else .body)
  else if c = rotate p then .source else .sink

def owner (r : R) : Phase → Place := owners (r.1.1 ()) r.2
def project (r : R) : S := r.1

noncomputable def declared (f : Stim → State → ℕ) (_t : ℕ) (z : S)
    (e : Option Unit) (_z' : S) : ℕ :=
  e.elim 0 (fun _ => f (world.sense z.2 ()) (z.1 ()))

noncomputable def physical (cut : Bool) : Interaction.Machine R ℕ (Option Unit) where
  step t r e r' := (process cut).step t r.1 e r'.1 *
    if r'.2 = phaseNext (world.sense r.1.2 ()) (r.1.1 ()) e r.2 then 1 else 0
  nonneg t r e r' := mul_nonneg ((process cut).nonneg t r.1 e r'.1)
    (by split_ifs <;> norm_num)
  sum_one t r := by
    simp only [Fintype.sum_prod_type, mul_ite, mul_one, mul_zero,
      Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
    simpa only [Fintype.sum_prod_type] using (process cut).sum_one t r.1

private def Enabled (s : Stim) (a : State) (e : Option Unit) : Prop :=
  (e = none ∧ (s = 0 ∨ a = 0)) ∨ (e = some () ∧ s ≠ 0 ∧ a ≠ 0)

set_option maxHeartbeats 800000 in
-- Check all supported finite channel rows before lifting to carrier transport.
private theorem event_delivery (s : Unit → Stim) (x y : Unit → State) (e : Option Unit)
    (hp : 0 < body.eventStep 1 s x e y) :
    y () = next (s ()) (x ()) ∧ Enabled (s ()) (x ()) e := by
  have hx : x = fun _ => x () := by funext u; cases u; rfl
  have hy : y = fun _ => y () := by funext u; cases u; rfl
  have hs : s = fun _ => s () := by funext u; cases u; rfl
  rw [hx, hy, hs] at hp
  generalize ha : s () = a at hp ⊢
  generalize hb : x () = b at hp ⊢
  generalize hc : y () = c at hp ⊢
  cases e with
  | none =>
    fin_cases a <;> fin_cases b <;> fin_cases c <;>
      norm_num [Medium.eventStep, body, next, Enabled, funext_iff] at hp
    all_goals norm_num [next, Enabled]
  | some e =>
    cases e
    fin_cases a <;> fin_cases b <;> fin_cases c <;>
      norm_num [Medium.eventStep, body, next, Enabled, funext_iff] at hp
    all_goals norm_num [next, Enabled]

private theorem physical_delivery (cut : Bool) (t : ℕ) (r r' : R) (e : Option Unit)
    (hp : 0 < (physical cut).step t r e r') :
    r'.1.1 () = next (world.sense r.1.2 ()) (r.1.1 ()) ∧
    r'.2 = phaseNext (world.sense r.1.2 ()) (r.1.1 ()) e r.2 ∧
    Enabled (world.sense r.1.2 ()) (r.1.1 ()) e := by
  have hphase : r'.2 = phaseNext (world.sense r.1.2 ()) (r.1.1 ()) e r.2 := by
    by_contra h
    simp [physical, h] at hp
  have hq : 0 < (process cut).step t r.1 e r'.1 := by
    simpa only [physical, hphase, ↓reduceIte, mul_one] using hp
  have hb : 0 < body.eventStep 1 (world.sense r.1.2) r.1.1 e r'.1.1 := by
    by_contra h
    have hw := body.eventWorldStep_nonneg world he cut (fun _ => world.sense) t r.1 r'.1.2
    exact (not_lt_of_ge (mul_nonpos_of_nonneg_of_nonpos hw (le_of_not_gt h))) hq
  have hd := event_delivery _ _ _ e hb
  exact ⟨hd.1, hphase, hd.2⟩

private theorem token_count : ∀ (a : State) (p : Phase),
    count (owners a p) .body = cost a := by decide

private theorem token_transport : ∀ (s : Stim) (a : State) (p : Phase) (e : Option Unit),
    Enabled s a e →
    Allowed (owners a p) (owners (next s a) (phaseNext s a e p)) ∧
    moved (owners a p) (owners (next s a) (phaseNext s a e p)) .source .body =
      e.elim 0 (fun _ => incoming s a) ∧
    moved (owners a p) (owners (next s a) (phaseNext s a e p)) .body .sink =
      e.elim 0 (fun _ => outgoing s a) := by
  unfold Enabled Allowed
  decide

/-- Full process and clock certificate, for both normal and marginal-ghost cut.
Carrier identities cannot alter any future observed event/state cylinder. -/
theorem realized (cut : Bool) :
    Realization (physical cut) (process cut) owner project
      (fun z => body.total z.1) (declared incoming) (declared outgoing) world.clock := by
  classical
  refine ⟨⟨by norm_num [world], rfl, ?_⟩, ?_, ?_⟩
  · intro t r e z
    rcases z with ⟨x, w⟩
    simp [physical, project, Fintype.sum_prod_type]
  · intro r
    simpa [owner, project, Medium.total, body] using token_count (r.1.1 ()) r.2
  · intro t r e r' hp
    obtain ⟨hx, hphase, hen⟩ := physical_delivery cut t r r' e hp
    have ht := token_transport (world.sense r.1.2 ()) (r.1.1 ()) r.2 e hen
    simpa only [owner, project, declared, hx, hphase] using ht

private theorem token_energy : ∀ (s : Stim) (a : State) (p : Phase) (e : Option Unit),
    Enabled s a e →
    count (owners (next s a) (phaseNext s a e p)) .body +
        count (owners (next s a) (phaseNext s a e p)) .source +
        e.elim 0 (fun _ => outgoing s a) =
      count (owners a p) .body + count (owners a p) .source +
        moved (owners a p) (owners (next s a) (phaseNext s a e p)) .sink .source := by
  unfold Enabled
  decide

/-- Total body+source+sink energy: source/body tokens carry one unit, sink
tokens carry zero. The external work reservoir explicitly recharges recycling.
It must supply indefinitely if normal feeding is to continue indefinitely. -/
noncomputable def sourceAccount (cut : Bool) : Account (physical cut) where
  stored _ r := (count (owner r) .body : ℝ) + (count (owner r) .source : ℝ)
  drive _ r _ r' := (moved (owner r) (owner r') .sink .source : ℝ)
  load _ _ _ _ := 0
  matter _ _ _ _ := 0
  heat t r e r' := -port outgoing t (project r) e (project r')
  balance t r e r' hp := by
    obtain ⟨hx, hphase, hen⟩ := physical_delivery cut t r r' e hp
    have hb := token_energy (world.sense r.1.2 ()) (r.1.1 ()) r.2 e hen
    have hr : (count (owner r') .body : ℝ) + (count (owner r') .source : ℝ) +
        (declared outgoing t (project r) e (project r') : ℝ) =
      (count (owner r) .body : ℝ) + (count (owner r) .source : ℝ) +
        (moved (owner r) (owner r') .sink .source : ℝ) := by
      have hnat : count (owner r') .body + count (owner r') .source +
          declared outgoing t (project r) e (project r') =
        count (owner r) .body + count (owner r) .source +
          moved (owner r) (owner r') .sink .source := by
        simpa only [owner, declared, project, hx, hphase] using hb
      exact_mod_cast hnat
    have he : port outgoing t (project r) e (project r') =
        (declared outgoing t (project r) e (project r') : ℝ) := by
      cases e <;> simp [port, declared]
    rw [he]
    linarith

end CGB.ActiveMaintenanceCarrier
