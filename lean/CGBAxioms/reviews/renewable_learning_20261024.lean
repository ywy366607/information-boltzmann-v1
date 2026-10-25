import CGBAxioms.RenewableLearningLife
import CGBAxioms.RenewableLearningEnergy
import CGBAxioms.RenewableLearningControls

/- Exact proof/interface review, with no fitted parameters, sampled learning
runs or capability comparison. Counts and paths are observer certificates. -/

open CGB CGB.RenewableLearning

namespace RenewableLearningReview20261024

example (reopen refill : Bool) :
    (plant reopen refill).Axioms ∧ (plant reopen refill).EventClocked 1 :=
  admission reopen refill

example : Fintype.card Cell = 8 ∧ Fintype.card Env = 9 ∧
    ∀ q : Cell, (memory q).reserve.val = credit q := by decide

-- The world does not announce its new branch in ordinary food/miss signals.
example : sense .feedA = sense .feedB ∧ sense .missA = sense .missB ∧
    sense .probeA = sense .probeB := by decide

-- The storage intervention preserves the input rule and initial body.
example (s : Signal) (q : Cell) : incoming false s q = incoming true s q ∧
    (plant true false).init = (plant true true).init := by exact ⟨rfl, rfl⟩

example : cost (next false true .miss .aReady) = 3 ∧
    cost (next true true .miss .aReady) = 2 ∧
    incoming false .feed .aSpent = 4 ∧ outgoing true false .feed .aSpent = 4 := by decide

example (n : ℕ) :
    Exchange.survival (plant true true) world (he true true) (ledger true true)
      false (fun _ => world.sense) n = 1 := RenewableLearningLife.natural_survival n

example (t : ℕ) (q : Cell) (w : Env) (hg : RenewableLearningLife.Good q w) :
    Exchange.recovery (plant true true) world (he true true) (ledger true true)
      false (fun _ => world.sense) t 2 ((fun _ => q), w) = 1 :=
  RenewableLearningLife.recovery_good t q w hg

example (reopen refill cut : Bool) (n : ℕ) :
    Energetics.mean (Energetics.law (process reopen refill cut) initial n)
        ((RenewableLearningEnergy.fuel_account reopen refill cut).stored n) +
      ∑ t ∈ Finset.range n, Energetics.flux (process reopen refill cut) initial
        (RenewableLearningEnergy.port (outgoing reopen refill)) t =
      2 + ∑ t ∈ Finset.range n, Energetics.flux (process reopen refill cut) initial
        (RenewableLearningEnergy.port (incoming refill)) t :=
  RenewableLearningEnergy.fuel_budget reopen refill cut n

example (reopen refill cut : Bool) (n : ℕ) :
    Energetics.mean (Energetics.law (process reopen refill cut) initial n)
        ((RenewableLearningEnergy.credit_account reopen refill cut).stored n) +
      ∑ t ∈ Finset.range n, Energetics.flux (process reopen refill cut) initial
        (RenewableLearningEnergy.port (fun s q => writeCost reopen s q + discarded s q)) t =
      1 + ∑ t ∈ Finset.range n, Energetics.flux (process reopen refill cut) initial
        (RenewableLearningEnergy.port (topup refill)) t :=
  RenewableLearningEnergy.credit_budget reopen refill cut n

example (reopen refill : Bool) (s : Signal) (q : Cell) :
    topup refill s q ≤ incoming refill s q ∧
      writeCost reopen s q + discarded s q ≤ outgoing reopen refill s q :=
  RenewableLearningEnergy.allocation reopen refill s q

example (k : ℕ) :
    0 < RenewableLearningControls.weight (process true true false)
      RenewableLearningControls.path (3 + 10 * k) ∧
    (∑ t ∈ Finset.range (3 + 10 * k), writeCost true
      (sense (RenewableLearningControls.path t).2 ()) ((RenewableLearningControls.path t).1 ())) =
        1 + 4 * k ∧
    (∑ t ∈ Finset.range (3 + 10 * k), topup true
      (sense (RenewableLearningControls.path t).2 ()) ((RenewableLearningControls.path t).1 ())) =
        2 + 4 * k := RenewableLearningControls.arbitrary_relearning k

example :
    Exchange.survival (plant false true) world (he false true) (ledger false true)
      false (fun _ => world.sense) 6 ≤ 1 / 2 ∧
    Exchange.survival (plant true false) world (he true false) (ledger true false)
      false (fun _ => world.sense) 6 ≤ 1 / 2 :=
  ⟨RenewableLearningControls.paired_failure false true (Or.inl rfl),
    RenewableLearningControls.paired_failure true false (Or.inr rfl)⟩

example (reopen refill : Bool) (s : Signal) :
    (plant reopen refill).rate () (fun _ => s) (fun _ => .vacuum) = 0 ∧
      next reopen refill s .vacuum = .vacuum := by simp [plant, next]

end RenewableLearningReview20261024
