import CGBAxioms.RenewableRecoveryCertificate
import CGBAxioms.RenewableLearningClosure
import CGBAxioms.RenewableLearningControls
import CGBAxioms.RenewableLearningEnergy

/- Exact review of the shared criterion and the ORIGINAL cut. No sampled
trajectories, training, fitted parameters or capability estimates are used. -/

open CGB CGB.RenewableLearning

namespace RecoveryClosureReview20261024

example (z : S) (hg : RenewableRecoveryCertificate.certificate.region z) :
    RenewableRecoveryCertificate.certificate.rank z < world.horizon :=
  RenewableRecoveryCertificate.certificate.bounded z hg

-- The analysis rank is not an extra body state; it reads the fixed joint state.
example :
    RenewableRecoveryCertificate.certificate.rank ((fun _ => Cell.aReady), Env.missB) = 1 ∧
    RenewableRecoveryCertificate.certificate.rank ((fun _ => Cell.search), Env.probeB) = 0 ∧
    ¬ RenewableRecoveryCertificate.certificate.region ((fun _ => Cell.aHeld), Env.probeB) := by
  simp [RenewableRecoveryCertificate.certificate, RenewableLearningLife.Good]

example (t : ℕ) (z : S) (hg : RenewableRecoveryCertificate.certificate.region z) :
    Exchange.recovery (plant true true) world (he true true) (ledger true true)
      false (fun _ => world.sense) t world.horizon z = 1 :=
  RecoveryCriteria.ranked_firstHit RenewableRecoveryCertificate.certificate t world.horizon z hg
    (RenewableRecoveryCertificate.certificate.bounded z hg)

example (n : ℕ) :
    Exchange.survival (plant true true) world (he true true) (ledger true true)
      false (fun _ => world.sense) n = 1 :=
  RenewableRecoveryCertificate.survival_by_criterion n

-- Both observations are wrappers around the same registered body/world law.
example (cut : Bool) (t : ℕ) (z z' : S) (e : Option Unit) :
    (RenewableLearningClosure.body.jointEventKernel world RenewableLearningClosure.ec cut
      (fun _ => world.sense) t).step z e z' = (process true true cut).step t z e z' := rfl

example : (∑ x : Unit → Cell,
    (plant true true).lawCut world (fun _ => world.sense) 6 x Env.trap) = 1 / 4 :=
  RenewableLearningClosure.cut_seed_mass

example (n : ℕ) (hn : 6 ≤ n) :
    Exchange.survival (plant true true) world (he true true) (ledger true true)
      true (fun _ => world.sense) n ≤ 3 / 4 :=
  RenewableLearningClosure.cut_survival n hn

example (n : ℕ) (hn : 6 ≤ n) :
    Exchange.survival (plant true true) world (he true true) (ledger true true)
        true (fun _ => world.sense) n + 1 / 4 ≤
      Exchange.survival (plant true true) world (he true true) (ledger true true)
        false (fun _ => world.sense) n := by
  rw [RenewableLearningLife.natural_survival]
  linarith [RenewableLearningClosure.cut_survival n hn]

example : Exchange.Persists (plant true true) world (he true true) (ledger true true) 0 ∧
    Exchange.SelfMaintains (plant true true) world (he true true) (ledger true true) :=
  RenewableLearningClosure.maintenance

-- The older interventions remain distinct comparisons with their own bounds.
example :
    Exchange.survival (plant false true) world (he false true) (ledger false true)
      false (fun _ => world.sense) 6 ≤ 1 / 2 ∧
    Exchange.survival (plant true false) world (he true false) (ledger true false)
      false (fun _ => world.sense) 6 ≤ 1 / 2 :=
  ⟨RenewableLearningControls.paired_failure false true (Or.inl rfl),
    RenewableLearningControls.paired_failure true false (Or.inr rfl)⟩

example (n : ℕ) :
    Energetics.mean (Energetics.law (process true true true) initial n)
        ((RenewableLearningEnergy.fuel_account true true true).stored n) +
      ∑ t ∈ Finset.range n, Energetics.flux (process true true true) initial
        (RenewableLearningEnergy.port (outgoing true true)) t =
      2 + ∑ t ∈ Finset.range n, Energetics.flux (process true true true) initial
        (RenewableLearningEnergy.port (incoming true)) t :=
  RenewableLearningEnergy.fuel_budget true true true n

end RecoveryClosureReview20261024
