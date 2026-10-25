import CGBAxioms.SensoryMaintenanceLimits

/-! Two existing finite material buffers realize structural fuel and the new
reserve. All 24 identities move from pre-event stocks, on the same clock. -/
open Finset

namespace CGB.SensoryMaintenanceCarrier
open RenewableLearning (Cell Signal Env)
open SensoryMaintenance MaterialSource Energetics

def partContent (reserve : Bool) (x : State) : ℕ :=
  if reserve then x.2.val else RenewableLearning.cost x.1

def structureIn (s : Stim) (x : State) : ℕ :=
  RenewableLearning.incoming true (s.getD .calm) x.1

def structureOut (s : Stim) (x : State) : ℕ :=
  match s with
  | none => if x.2.val = 0 then RenewableLearning.cost x.1 else 0
  | some a => RenewableLearning.outgoing true true a x.1

def partIn (reserve : Bool) (s : Stim) (x : State) : ℕ :=
  if reserve then incoming s x - structureIn s x else structureIn s x

def partOut (reserve : Bool) (s : Stim) (x : State) : ℕ :=
  if reserve then outgoing s x - structureOut s x else structureOut s x

private theorem component_rules (b : Bool) (s : Stim) (x : State) :
    partContent b x ≤ 4 ∧ partIn b s x ≤ 4 ∧ partOut b s x ≤ partContent b x ∧
      partContent b (next s x) + partOut b s x = partContent b x + partIn b s x := by
  rcases x with ⟨q,r⟩
  cases b <;> fin_cases r <;> cases s with
  | none => cases q <;> decide
  | some a => cases a <;> cases q <;> decide

theorem component_sums (s : Stim) (x : State) :
    partContent false x + partContent true x = cost x ∧
    partIn false s x + partIn true s x = incoming s x ∧
    partOut false s x + partOut true s x = outgoing s x := by
  rcases x with ⟨q,r⟩
  fin_cases r <;> cases s with
  | none => cases q <;> decide
  | some a => cases a <;> cases q <;> decide

def declared (sched : ℕ → Env → Unit → Stim) (f : Stim → State → ℕ)
    (t : ℕ) (z : SensoryMaintenance.S) (e : Option Unit) (_z' : SensoryMaintenance.S) : ℕ :=
  e.elim 0 (fun _ => f (sched t z.2 ()) (z.1 ()))

noncomputable def rules (cut b : Bool) (sched : ℕ → Env → Unit → Stim) :
    BufferedMaterial.Rules (process cut sched) where
  content z := partContent b (z.1 ())
  incoming := declared sched (partIn b)
  outgoing := declared sched (partOut b)
  bounded z := (component_rules b none (z.1 ())).1
  intake_bound t z e z' _ := by
    cases e
    · simp [declared]
    · exact (component_rules b (sched t z.2 ()) (z.1 ())).2.1
  release_bound t z e z' _ := by
    cases e
    · simp [declared]
    · exact (component_rules b (sched t z.2 ()) (z.1 ())).2.2.1
  balance t z e z' hp := by
    cases e with
    | none =>
      have hb : 0 < body.eventStep 1 (sched t z.2) z.1 none z'.1 := by
        by_contra hn
        exact (not_lt_of_ge (mul_nonpos_of_nonneg_of_nonpos
          (body.eventWorldStep_nonneg world ec cut sched t z z'.2) (le_of_not_gt hn))) hp
      have he : z.1 = z'.1 := by by_contra hn; simp [Medium.eventStep, hn] at hb
      simp [declared, ← he]
    | some u =>
      have hx := actual_transition cut sched t z z' (some u) hp
      have h := (component_rules b (sched t z.2 ()) (z.1 ())).2.2.2
      rw [← hx] at h
      exact h

private theorem coarse_positive {X E : Type*} [Fintype X] [DecidableEq X] [Fintype E]
    {P : Interaction.Machine X ℕ E} (B : BufferedMaterial.Rules P)
    (t : ℕ) (r r' : BufferedMaterial.State B) (e : E)
    (hp : 0 < (BufferedMaterial.physical B).step t r e r') :
    0 < P.step t (BufferedMaterial.project B r) e (BufferedMaterial.project B r') := by
  by_contra hn
  change 0 < P.step t (BufferedMaterial.project B r) e (BufferedMaterial.project B r') *
    (if r' = BufferedMaterial.advance B t r e (BufferedMaterial.project B r') then 1 else 0) at hp
  exact (not_lt_of_ge (mul_nonpos_of_nonpos_of_nonneg (le_of_not_gt hn)
    (by split_ifs <;> norm_num))) hp

noncomputable def liftRules {X E : Type*} [Fintype X] [DecidableEq X] [Fintype E]
    {P : Interaction.Machine X ℕ E} (B C : BufferedMaterial.Rules P) :
    BufferedMaterial.Rules (BufferedMaterial.physical B) where
  content r := C.content (BufferedMaterial.project B r)
  incoming t r e r' := C.incoming t (BufferedMaterial.project B r) e (BufferedMaterial.project B r')
  outgoing t r e r' := C.outgoing t (BufferedMaterial.project B r) e (BufferedMaterial.project B r')
  bounded r := C.bounded _
  intake_bound t r e r' hp := C.intake_bound _ _ _ _ (coarse_positive B t r r' e hp)
  release_bound t r e r' hp := C.release_bound _ _ _ _ (coarse_positive B t r r' e hp)
  balance t r e r' hp := C.balance _ _ _ _ (coarse_positive B t r r' e hp)

noncomputable abbrev firstRules (cut : Bool) (sched : ℕ → Env → Unit → Stim) := rules cut false sched
noncomputable abbrev secondRules (cut : Bool) (sched : ℕ → Env → Unit → Stim) :=
  liftRules (firstRules cut sched) (rules cut true sched)
abbrev Carrier (cut : Bool) (sched : ℕ → Env → Unit → Stim) :=
  BufferedMaterial.State (secondRules cut sched)
noncomputable abbrev physical (cut : Bool) (sched : ℕ → Env → Unit → Stim) :=
  BufferedMaterial.physical (secondRules cut sched)
noncomputable def project (cut : Bool) (sched : ℕ → Env → Unit → Stim) (r : Carrier cut sched) :=
  BufferedMaterial.project (firstRules cut sched) (BufferedMaterial.project (secondRules cut sched) r)

abbrev Token := Sum (Fin 12) (Fin 12)
noncomputable def owner (cut : Bool) (sched : ℕ → Env → Unit → Stim)
    (r : Carrier cut sched) : Token → Place
  | .inl c => BufferedMaterial.owner (firstRules cut sched)
      (BufferedMaterial.project (secondRules cut sched) r) c
  | .inr c => BufferedMaterial.owner (secondRules cut sched) r c

private theorem count_sum (cut : Bool) (sched : ℕ → Env → Unit → Stim)
    (r : Carrier cut sched) (p : Place) : count (owner cut sched r) p =
    count (BufferedMaterial.owner (firstRules cut sched)
      (BufferedMaterial.project (secondRules cut sched) r)) p +
    count (BufferedMaterial.owner (secondRules cut sched) r) p := by
  simp only [count, owner, Fintype.sum_sum_type]
  rfl

private theorem moved_sum (cut : Bool) (sched : ℕ → Env → Unit → Stim)
    (r r' : Carrier cut sched) (p q : Place) : moved (owner cut sched r) (owner cut sched r') p q =
    moved (BufferedMaterial.owner (firstRules cut sched)
      (BufferedMaterial.project (secondRules cut sched) r))
      (BufferedMaterial.owner (firstRules cut sched)
        (BufferedMaterial.project (secondRules cut sched) r')) p q +
    moved (BufferedMaterial.owner (secondRules cut sched) r)
      (BufferedMaterial.owner (secondRules cut sched) r') p q := by
  simp only [moved, owner, Fintype.sum_sum_type]
  rfl

theorem realized (cut : Bool) (sched : ℕ → Env → Unit → Stim) :
    Realization (physical cut sched) (process cut sched) (owner cut sched) (project cut sched)
      (fun z => body.total z.1) (declared sched incoming) (declared sched outgoing) 1 := by
  refine ⟨⟨by norm_num, rfl, ?_⟩, ?_, ?_⟩
  · exact Interaction.Refines.comp (BufferedMaterial.refines (secondRules cut sched))
      (BufferedMaterial.refines (firstRules cut sched))
  · intro r
    rw [count_sum, (BufferedMaterial.realized (firstRules cut sched) 1 (by norm_num)).body_count,
      (BufferedMaterial.realized (secondRules cut sched) 1 (by norm_num)).body_count]
    exact (component_sums none ((project cut sched r).1 ())).1
  · intro t r e r' hp
    have h2 := BufferedMaterial.delivery (secondRules cut sched) t r r' e hp
    have hc := coarse_positive (secondRules cut sched) t r r' e hp
    have h1 := BufferedMaterial.delivery (firstRules cut sched) t
      (BufferedMaterial.project (secondRules cut sched) r)
      (BufferedMaterial.project (secondRules cut sched) r') e hc
    refine ⟨?_, ?_, ?_⟩
    · intro c; cases c with
      | inl c => exact h1.1 c
      | inr c => exact h2.1 c
    · rw [moved_sum, h1.2.1, h2.2.1]
      cases e with
      | none => rfl
      | some u => exact (component_sums (sched t (project cut sched r).2 ())
          ((project cut sched r).1 ())).2.1
    · rw [moved_sum, h1.2.2.1, h2.2.2.1]
      cases e with
      | none => rfl
      | some u => exact (component_sums (sched t (project cut sched r).2 ())
          ((project cut sched r).1 ())).2.2

theorem source_stock (cut : Bool) (sched : ℕ → Env → Unit → Stim) (r : Carrier cut sched) :
    count (owner cut sched r) .source = 8 := by
  rw [count_sum]
  have h1 : count (BufferedMaterial.owner (firstRules cut sched)
      (BufferedMaterial.project (secondRules cut sched) r)) .source = 4 :=
    (BufferedMaterial.project (secondRules cut sched) r).property.2.1
  have h2 : count (BufferedMaterial.owner (secondRules cut sched) r) .source = 4 := r.property.2.1
  rw [h1, h2]

theorem recharge_paid (cut : Bool) (sched : ℕ → Env → Unit → Stim)
    (t : ℕ) (r r' : Carrier cut sched) (e : Option Unit)
    (hp : 0 < (physical cut sched).step t r e r') :
    moved (owner cut sched r) (owner cut sched r') .sink .source =
      moved (owner cut sched r) (owner cut sched r') .source .body := by
  have h2 := BufferedMaterial.delivery (secondRules cut sched) t r r' e hp
  have hc := coarse_positive (secondRules cut sched) t r r' e hp
  have h1 := BufferedMaterial.delivery (firstRules cut sched) t
    (BufferedMaterial.project (secondRules cut sched) r)
    (BufferedMaterial.project (secondRules cut sched) r') e hc
  rw [moved_sum, moved_sum, h1.2.2.2, h2.2.2.2, h1.2.1, h2.2.1]

noncomputable def initialCarrier (cut : Bool) (sched : ℕ → Env → Unit → Stim) : Carrier cut sched :=
  BufferedMaterial.seed (secondRules cut sched) (BufferedMaterial.seed (firstRules cut sched) initial)

noncomputable def intake (cut : Bool) (sched : ℕ → Env → Unit → Stim)
    (_t : ℕ) (r : Carrier cut sched) (_e : Option Unit) (r' : Carrier cut sched) : Bool :=
  decide (0 < moved (owner cut sched r) (owner cut sched r') .source .body)

theorem recovery_preserved (cut : Bool) (sched : ℕ → Env → Unit → Stim)
    (t k : ℕ) (r : Carrier cut sched) :
    EventChannels.Kernel.firstHit (events (physical cut sched)) (intake cut sched) t k r =
      Exchange.recovery body world ec material cut sched t k (project cut sched r) := by
  apply RecoveryTransport.firstHit_eq (realized cut sched).process.representation
  intro t r e r' hp
  rw [show intake cut sched t r e r' =
      decide (0 < declared sched incoming t (project cut sched r) e (project cut sched r')) from
    congrArg (fun n : ℕ => decide (0 < n)) ((realized cut sched).transport t r e r' hp).2.1]
  cases e <;> rfl

noncomputable def score (cut : Bool) (sched : ℕ → Env → Unit → Stim) (n : ℕ) : ℝ :=
  mean (law (physical cut sched) (initialCarrier cut sched) n) (fun r =>
    if 0 < count (owner cut sched r) .body ∧ world.recovery ≤
      EventChannels.Kernel.firstHit (events (physical cut sched)) (intake cut sched) n world.horizon r
    then 1 else 0)

theorem score_preserved (cut : Bool) (sched : ℕ → Env → Unit → Stim) (n : ℕ) :
    score cut sched n = Exchange.survival body world ec material cut sched n := by
  have point (r : Carrier cut sched) :
      (if 0 < count (owner cut sched r) .body ∧ world.recovery ≤
        EventChannels.Kernel.firstHit (events (physical cut sched)) (intake cut sched) n world.horizon r
      then (1 : ℝ) else 0) = Exchange.alive body world ec material cut sched n (project cut sched r) := by
    rw [(realized cut sched).body_count, recovery_preserved]
    unfold Exchange.alive Viability.alive
    split_ifs <;> rfl
  unfold score
  simp_rw [point]
  rw [RecoveryTransport.mean_eq (realized cut sched).process.representation]
  change (∑ z : SensoryMaintenance.S, law (process cut sched) initial n z *
    Exchange.alive body world ec material cut sched n z) = _
  simp only [Fintype.sum_prod_type]
  apply Finset.sum_congr rfl; intro x _
  apply Finset.sum_congr rfl; intro w _
  rw [SensoryMaintenance.actual_law cut sched n (x,w)]
  exact mul_comm _ _

noncomputable def release (cut : Bool) (sched : ℕ → Env → Unit → Stim)
    (_t : ℕ) (r : Carrier cut sched) (_e : Option Unit) (r' : Carrier cut sched) : ℝ :=
  (moved (owner cut sched r) (owner cut sched r') .body .sink : ℝ)

noncomputable def sourceAccount (cut : Bool) (sched : ℕ → Env → Unit → Stim) :
    Account (physical cut sched) where
  stored _ r := (count (owner cut sched r) .body : ℝ) + (count (owner cut sched r) .source : ℝ)
  drive _ r _ r' := (moved (owner cut sched r) (owner cut sched r') .sink .source : ℝ)
  load _ _ _ _ := 0
  matter _ _ _ _ := 0
  heat t r e r' := -release cut sched t r e r'
  balance t r e r' hp := by
    have h := (realized cut sched).transport t r e r' hp
    have hb := (MaterialSource.balance (owner cut sched r) (owner cut sched r') h.1).1
    have hs := source_stock cut sched r
    have hs' := source_stock cut sched r'
    have hr := recharge_paid cut sched t r r' e hp
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

theorem initial_energy (cut : Bool) (sched : ℕ → Env → Unit → Stim) :
    (sourceAccount cut sched).stored 0 (initialCarrier cut sched) = 12 := by
  change (count (owner cut sched (initialCarrier cut sched)) .body : ℝ) +
    (count (owner cut sched (initialCarrier cut sched)) .source : ℝ) = 12
  rw [(realized cut sched).body_count, source_stock]
  norm_num [initialCarrier, project, BufferedMaterial.project, BufferedMaterial.seed,
    initial, body, Medium.total, cost, RenewableLearning.cost, RenewableLearning.buffer,
    RenewableLearning.credit]

theorem source_energy_budget (cut : Bool) (sched : ℕ → Env → Unit → Stim) (n : ℕ) :
    mean (law (physical cut sched) (initialCarrier cut sched) n) ((sourceAccount cut sched).stored n) +
      ∑ t ∈ range n, flux (physical cut sched) (initialCarrier cut sched) (release cut sched) t =
    12 + ∑ t ∈ range n,
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
        flux (physical cut sched) (initialCarrier cut sched)
          (fun t r e r' => -release cut sched t r e r') n at hb
    rw [hz, hh] at hb
    simp only [Finset.sum_range_succ]
    linarith

end CGB.SensoryMaintenanceCarrier
