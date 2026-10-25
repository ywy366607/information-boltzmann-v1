import CGBAxioms.BufferedMaterial
import CGBAxioms.RecoveryTransport
import CGBAxioms.RenewableLearningClosure
import CGBAxioms.RenewableRecoveryCertificate
import CGBAxioms.RenewableLearningEnergy

/-!
# Source and recovery certificates for the unchanged renewable-learning process

Fixed material identities realize every supported normal/cut transition.
Returned carriers are recharged by an explicit external work port. Body fuel
and write credit keep their old interpretation; credit is not added twice.
-/

open Finset

namespace CGB.RenewableLearningCarrier
open RenewableLearning MaterialSource Energetics

noncomputable def declared (f : Signal → Cell → ℕ) (_t : ℕ) (z : S)
    (e : Option Unit) (_z' : S) : ℕ := e.elim 0 (fun _ => f (sense z.2 ()) (z.1 ()))

private theorem body_supported (cut : Bool) (t : ℕ) (z z' : S) (e : Option Unit)
    (hp : 0 < (process true true cut).step t z e z') :
    0 < (plant true true).eventStep world.clock (world.sense z.2) z.1 e z'.1 := by
  by_contra hn
  have hw := (plant true true).eventWorldStep_nonneg world (he true true)
    cut (fun _ => world.sense) t z z'.2
  exact (not_lt_of_ge (mul_nonpos_of_nonneg_of_nonpos hw (le_of_not_gt hn))) hp

noncomputable def rules (cut : Bool) : BufferedMaterial.Rules (process true true cut) where
  content z := cost (z.1 ())
  incoming := declared (incoming true)
  outgoing := declared (outgoing true true)
  bounded z := by cases z.1 () <;> decide
  intake_bound t z e z' _ := by
    cases e with
    | none => simp [declared]
    | some u =>
      change incoming true (sense z.2 ()) (z.1 ()) ≤ 4
      generalize sense z.2 () = s; generalize z.1 () = q
      cases s <;> cases q <;> decide
  release_bound t z e z' _ := by
    cases e with
    | none => simp [declared]
    | some u =>
      change outgoing true true (sense z.2 ()) (z.1 ()) ≤ cost (z.1 ())
      generalize sense z.2 () = s; generalize z.1 () = q
      cases s <;> cases q <;> decide
  balance t z e z' hp := by
    have hb := body_supported cut t z z' e hp
    cases e with
    | none =>
      have hx : z.1 = z'.1 := by
        by_contra hn
        simp [Medium.eventStep, hn] at hb
      simp [declared, ← hx]
    | some u =>
      have hx : (plant true true).act u (sense z.2) z.1 = z'.1 := by
        by_contra hn
        simp [Medium.eventStep, hn] at hb
      have hn : next true true (sense z.2 ()) (z.1 ()) = z'.1 () := congrFun hx ()
      simpa only [declared, Option.elim_some, hn] using
        (local_budget true true (sense z.2 ()) (z.1 ())).1

abbrev Carrier (cut : Bool) := BufferedMaterial.State (rules cut)
noncomputable abbrev physical (cut : Bool) := BufferedMaterial.physical (rules cut)
noncomputable abbrev project (cut : Bool) := BufferedMaterial.project (rules cut)
noncomputable abbrev owner (cut : Bool) := BufferedMaterial.owner (rules cut)
noncomputable def initialCarrier (cut : Bool) : Carrier cut := BufferedMaterial.seed (rules cut) initial

theorem realized (cut : Bool) :
    Realization (physical cut) (process true true cut) (owner cut) (project cut)
      (fun z => (plant true true).total z.1)
      (declared (incoming true)) (declared (outgoing true true)) world.clock := by
  have hc : (fun z : S => (plant true true).total z.1) = (rules cut).content := by
    funext z
    simp [Medium.total, plant, rules]
  rw [hc]
  exact BufferedMaterial.realized (rules cut) world.clock world.clock_pos

/-- An actual identity relocation, not merely a positive declared input. -/
noncomputable def intake (cut : Bool) (_t : ℕ) (r : Carrier cut)
    (_e : Option Unit) (r' : Carrier cut) : Bool :=
  decide (0 < moved (owner cut r) (owner cut r') .source .body)

private theorem intake_matches (cut : Bool) (t : ℕ) (r r' : Carrier cut) (e : Option Unit)
    (hp : 0 < (physical cut).step t r e r') :
    intake cut t r e r' = Exchange.intake (plant true true) world (ledger true true)
      (fun _ => world.sense) t (project cut r) e (project cut r') := by
  have h := (realized cut).transport t r e r' hp
  simp only [intake, h.2.1, declared, Exchange.intake, ledger]
  cases e with
  | none => rfl
  | some u => cases u; rfl

theorem recovery_preserved (cut : Bool) (t k : ℕ) (r : Carrier cut) :
    EventChannels.Kernel.firstHit (events (physical cut)) (intake cut) t k r =
      Exchange.recovery (plant true true) world (he true true) (ledger true true)
        cut (fun _ => world.sense) t k (project cut r) :=
  RecoveryTransport.firstHit_eq (realized cut).process.representation _ _
    (fun t r e r' hp => intake_matches cut t r r' e hp) t k r

theorem recovery_by_certificate (t : ℕ) (r : Carrier false)
    (hg : RenewableRecoveryCertificate.certificate.region (project false r)) :
    EventChannels.Kernel.firstHit (events (physical false)) (intake false)
      t world.horizon r = 1 := by
  rw [recovery_preserved]
  exact RecoveryCriteria.ranked_firstHit RenewableRecoveryCertificate.certificate
    t world.horizon (project false r) hg
    (RenewableRecoveryCertificate.certificate.bounded (project false r) hg)

noncomputable def score (cut : Bool) (n : ℕ) : ℝ :=
  mean (law (physical cut) (initialCarrier cut) n) (fun r =>
    if 0 < count (owner cut r) .body ∧ world.recovery ≤
      EventChannels.Kernel.firstHit (events (physical cut)) (intake cut) n world.horizon r
    then 1 else 0)

theorem score_preserved (cut : Bool) (n : ℕ) : score cut n =
    Exchange.survival (plant true true) world (he true true) (ledger true true)
      cut (fun _ => world.sense) n := by
  have point (r : Carrier cut) :
      (if 0 < count (owner cut r) .body ∧ world.recovery ≤
        EventChannels.Kernel.firstHit (events (physical cut)) (intake cut) n world.horizon r
      then (1 : ℝ) else 0) =
        Exchange.alive (plant true true) world (he true true) (ledger true true)
          cut (fun _ => world.sense) n (project cut r) := by
    rw [(realized cut).body_count, recovery_preserved]
    unfold Exchange.alive Viability.alive
    split_ifs <;> rfl
  unfold score
  simp_rw [point]
  rw [RecoveryTransport.mean_eq (realized cut).process.representation]
  change (∑ z : S, law (process true true cut) initial n z *
    Exchange.alive (plant true true) world (he true true) (ledger true true)
      cut (fun _ => world.sense) n z) = _
  simp only [actual_law, Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro x _
  apply Finset.sum_congr rfl
  intro w _
  exact mul_comm _ _

theorem maintenance_preserved : (∀ n, score false n = 1) ∧
    (∀ n, 6 ≤ n → score true n + 1 / 4 ≤ score false n) := by
  constructor
  · intro n; rw [score_preserved, RenewableLearningLife.natural_survival]
  · intro n hn
    rw [score_preserved, score_preserved, RenewableLearningLife.natural_survival]
    linarith [RenewableLearningClosure.cut_survival n hn]

noncomputable def release (cut : Bool) (_t : ℕ) (r : Carrier cut)
    (_e : Option Unit) (r' : Carrier cut) : ℝ :=
  (moved (owner cut r) (owner cut r') .body .sink : ℝ)

/-- Body and accessible-source units carry one declared energy unit; returned
units carry zero. Recycling requires that much external work. Internal source
to body transfer is NOT charged as extra external material energy. -/
noncomputable def sourceAccount (cut : Bool) : Account (physical cut) where
  stored _ r := (count (owner cut r) .body : ℝ) + (count (owner cut r) .source : ℝ)
  drive _ r _ r' := (moved (owner cut r) (owner cut r') .sink .source : ℝ)
  load _ _ _ _ := 0
  matter _ _ _ _ := 0
  heat t r e r' := -release cut t r e r'
  balance t r e r' hp := by
    have h := BufferedMaterial.delivery (rules cut) t r r' e hp
    have hb := (MaterialSource.balance (owner cut r) (owner cut r') h.1).1
    have hs := r.property.2.1
    have hs' := r'.property.2.1
    have hn : count (owner cut r') .body + count (owner cut r') .source +
        moved (owner cut r) (owner cut r') .body .sink =
      count (owner cut r) .body + count (owner cut r) .source +
        moved (owner cut r) (owner cut r') .sink .source := by
      change count (owner cut r) .source = 4 at hs
      change count (owner cut r') .source = 4 at hs'
      have hr : moved (owner cut r) (owner cut r') .sink .source =
          moved (owner cut r) (owner cut r') .source .body := h.2.2.2.trans h.2.1.symm
      omega
    have he : (count (owner cut r') .body : ℝ) + (count (owner cut r') .source : ℝ) +
        (moved (owner cut r) (owner cut r') .body .sink : ℝ) =
      (count (owner cut r) .body : ℝ) + (count (owner cut r) .source : ℝ) +
        (moved (owner cut r) (owner cut r') .sink .source : ℝ) := by exact_mod_cast hn
    unfold release
    linarith

theorem stock_and_recharge (cut : Bool) (t : ℕ) (r r' : Carrier cut) (e : Option Unit)
    (hp : 0 < (physical cut).step t r e r') :
    count (owner cut r) .source = 4 ∧
      (sourceAccount cut).drive t r e r' =
        (declared (incoming true) t (project cut r) e (project cut r') : ℝ) := by
  refine ⟨r.property.2.1, ?_⟩
  have h := (BufferedMaterial.delivery (rules cut) t r r' e hp).2.2.2
  change (moved (owner cut r) (owner cut r') .sink .source : ℝ) = _
  exact_mod_cast h

/-- The extra source stock is external to the old two-unit body at birth. -/
theorem initial_energy (cut : Bool) : (sourceAccount cut).stored 0 (initialCarrier cut) = 6 := by
  change (count (owner cut (initialCarrier cut)) .body : ℝ) +
    (count (owner cut (initialCarrier cut)) .source : ℝ) = 6
  rw [(realized cut).body_count]
  have hs := (initialCarrier cut).property.2.1
  change count (owner cut (initialCarrier cut)) .source = 4 at hs
  rw [hs]
  norm_num [initialCarrier, BufferedMaterial.seed, project, BufferedMaterial.project,
    Medium.total, plant, initial, cost, buffer, credit]

theorem source_energy_budget (cut : Bool) (n : ℕ) :
    mean (law (physical cut) (initialCarrier cut) n) ((sourceAccount cut).stored n) +
      ∑ t ∈ range n, flux (physical cut) (initialCarrier cut) (release cut) t =
    6 + ∑ t ∈ range n, flux (physical cut) (initialCarrier cut) (sourceAccount cut).drive t := by
  induction n with
  | zero =>
    simp only [Finset.range_zero, Finset.sum_empty, add_zero]
    simpa only [mean, law, EventChannels.Kernel.stateLaw, ite_mul, one_mul, zero_mul,
      Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte] using initial_energy cut
  | succ n ih =>
    have hb := expected_first_law (physical cut) (sourceAccount cut) (initialCarrier cut) n
    have hz : flux (physical cut) (initialCarrier cut) (fun _ _ _ _ => 0) n = 0 := by
      simp [flux]
    have hh : flux (physical cut) (initialCarrier cut)
        (fun t r e r' => -release cut t r e r') n =
      -flux (physical cut) (initialCarrier cut) (release cut) n := by
      simp only [flux, mul_neg, Finset.sum_neg_distrib]
    change _ + flux (physical cut) (initialCarrier cut) (fun _ _ _ _ => 0) n =
      _ + _ + flux (physical cut) (initialCarrier cut) (fun _ _ _ _ => 0) n +
        flux (physical cut) (initialCarrier cut) (fun t r e r' => -release cut t r e r') n at hb
    rw [hz, hh] at hb
    simp only [Finset.sum_range_succ]
    linarith

end CGB.RenewableLearningCarrier
