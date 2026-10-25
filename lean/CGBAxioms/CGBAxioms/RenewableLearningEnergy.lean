import CGBAxioms.RenewableLearning

/-! The total fuel account contains both buffer and write credit. The credit
account is a subaccount, not additional energy to add a second time. All ports
are specified event counts before their supported balances are checked. -/

open Finset

namespace CGB.RenewableLearningEnergy
open RenewableLearning Energetics

noncomputable def port (f : Signal → Cell → ℕ) (_t : ℕ) (z : S)
    (e : Option Unit) (_z' : S) : ℝ :=
  e.elim 0 (fun _ => (f (sense z.2 ()) (z.1 ()) : ℝ))

private theorem supported (reopen refill cut : Bool) (t : ℕ) (z z' : S) (e : Option Unit)
    (hp : 0 < (process reopen refill cut).step t z e z') :
    0 < (plant reopen refill).eventStep world.clock (world.sense z.2) z.1 e z'.1 := by
  by_contra hn
  have hw := (plant reopen refill).eventWorldStep_nonneg world (he reopen refill)
    cut (fun _ => world.sense) t z z'.2
  exact (not_lt_of_ge (mul_nonpos_of_nonneg_of_nonpos hw (le_of_not_gt hn))) hp

noncomputable def fuel_account (reopen refill cut : Bool) : Account (process reopen refill cut) where
  stored _ z := (cost (z.1 ()) : ℝ)
  drive _ _ _ _ := 0
  load _ _ _ _ := 0
  matter := port (incoming refill)
  heat t z e z' := -port (outgoing reopen refill) t z e z'
  balance t z e z' hp := by
    have hb := supported reopen refill cut t z z' e hp
    cases e with
    | none =>
      have hx : z.1 = z'.1 := by
        by_contra hn
        simp [Medium.eventStep, hn] at hb
      simp [port, ← hx]
    | some u =>
      have hx : (plant reopen refill).act u (sense z.2) z.1 = z'.1 := by
        by_contra hn
        simp [Medium.eventStep, hn] at hb
      have hn : next reopen refill (sense z.2 ()) (z.1 ()) = z'.1 () := congrFun hx ()
      have hc := (local_budget reopen refill (sense z.2 ()) (z.1 ())).1
      rw [hn] at hc
      have hr : (cost (z'.1 ()) : ℝ) + (outgoing reopen refill (sense z.2 ()) (z.1 ()) : ℝ) =
          (cost (z.1 ()) : ℝ) + (incoming refill (sense z.2 ()) (z.1 ()) : ℝ) := by
        exact_mod_cast hc
      simp only [port, Option.elim_some, add_zero]
      linarith

noncomputable def credit_account (reopen refill cut : Bool) : Account (process reopen refill cut) where
  stored _ z := (credit (z.1 ()) : ℝ)
  drive _ _ _ _ := 0
  load _ _ _ _ := 0
  matter := port (topup refill)
  heat t z e z' := -port (fun s q => writeCost reopen s q + discarded s q) t z e z'
  balance t z e z' hp := by
    have hb := supported reopen refill cut t z z' e hp
    cases e with
    | none =>
      have hx : z.1 = z'.1 := by
        by_contra hn
        simp [Medium.eventStep, hn] at hb
      simp [port, ← hx]
    | some u =>
      have hx : (plant reopen refill).act u (sense z.2) z.1 = z'.1 := by
        by_contra hn
        simp [Medium.eventStep, hn] at hb
      have hn : next reopen refill (sense z.2 ()) (z.1 ()) = z'.1 () := congrFun hx ()
      have hc := (local_budget reopen refill (sense z.2 ()) (z.1 ())).2
      rw [hn] at hc
      have hr : (credit (z'.1 ()) : ℝ) +
          ((writeCost reopen (sense z.2 ()) (z.1 ()) + discarded (sense z.2 ()) (z.1 ())) : ℝ) =
          (credit (z.1 ()) : ℝ) + (topup refill (sense z.2 ()) (z.1 ()) : ℝ) := by
        exact_mod_cast (by simpa only [Nat.add_assoc] using hc)
      simp only [port, Option.elim_some, add_zero, Nat.cast_add]
      linarith

/-- Allocating credit uses part of the specified food input. Effective writes
and destroyed credit use part of total specified consumption. -/
theorem allocation (reopen refill : Bool) (s : Signal) (q : Cell) :
    topup refill s q ≤ incoming refill s q ∧
    writeCost reopen s q + discarded s q ≤ outgoing reopen refill s q := by
  cases reopen <;> cases refill <;> cases s <;> cases q <;> decide

private theorem telescope (P : Interaction.Machine S ℕ (Option Unit)) (a : Account P)
    (supply spend : ℕ → S → Option Unit → S → ℝ)
    (hd : a.drive = fun _ _ _ _ => 0) (hl : a.load = fun _ _ _ _ => 0)
    (hm : a.matter = supply) (hh : a.heat = fun t z e z' => -spend t z e z') (n : ℕ) :
    mean (law P initial n) (a.stored n) + ∑ t ∈ range n, flux P initial spend t =
      a.stored 0 initial + ∑ t ∈ range n, flux P initial supply t := by
  induction n with
  | zero =>
    simp only [Finset.range_zero, Finset.sum_empty, add_zero]
    change (∑ z : S, (if z = initial then (1 : ℝ) else 0) * a.stored 0 z) = _
    simp only [ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
  | succ n ih =>
    have hb := expected_first_law P a initial n
    rw [hd, hl, hm, hh] at hb
    have hz : flux P initial (fun _ _ _ _ => 0) n = 0 := by simp [flux]
    have hn : flux P initial (fun t z e z' => -spend t z e z') n =
        -flux P initial spend n := by simp only [flux, mul_neg, Finset.sum_neg_distrib]
    rw [hz, hn] at hb
    simp only [Finset.sum_range_succ]
    linarith

theorem fuel_budget (reopen refill cut : Bool) (n : ℕ) :
    mean (law (process reopen refill cut) initial n) ((fuel_account reopen refill cut).stored n) +
        ∑ t ∈ range n, flux (process reopen refill cut) initial (port (outgoing reopen refill)) t =
      2 + ∑ t ∈ range n, flux (process reopen refill cut) initial (port (incoming refill)) t := by
  exact telescope (process reopen refill cut) (fuel_account reopen refill cut)
    (port (incoming refill)) (port (outgoing reopen refill)) rfl rfl rfl rfl n

theorem credit_budget (reopen refill cut : Bool) (n : ℕ) :
    mean (law (process reopen refill cut) initial n) ((credit_account reopen refill cut).stored n) +
        ∑ t ∈ range n, flux (process reopen refill cut) initial
          (port (fun s q => writeCost reopen s q + discarded s q)) t =
      1 + ∑ t ∈ range n, flux (process reopen refill cut) initial (port (topup refill)) t := by
  simpa only [credit_account, initial, credit, Nat.cast_one] using
    telescope (process reopen refill cut) (credit_account reopen refill cut)
    (port (topup refill)) (port (fun s q => writeCost reopen s q + discarded s q))
    rfl rfl rfl rfl n

end CGB.RenewableLearningEnergy
