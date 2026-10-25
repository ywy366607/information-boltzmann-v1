import CGBAxioms.RenewableLearningCarrier

/- Independent checks of exact source, recovery, clock and work certificates. -/

open CGB CGB.RenewableLearning CGB.RenewableLearningCarrier
open CGB.MaterialSource CGB.Energetics

namespace RenewableCarrierReview20261024

example (cut : Bool) :
    Realization (physical cut) (process true true cut) (owner cut) (project cut)
      (fun z => (plant true true).total z.1)
      (declared (incoming true)) (declared (outgoing true true)) world.clock := realized cut

example (cut : Bool) (r : Carrier cut) :
    count (owner cut r) .body = (plant true true).total (project cut r).1 :=
  (realized cut).body_count r

example (cut : Bool) (t : ℕ) (r r' : Carrier cut) (e : Option Unit)
    (hp : 0 < (physical cut).step t r e r') :
    count (owner cut r) .source = 4 ∧ (sourceAccount cut).drive t r e r' =
      (declared (incoming true) t (project cut r) e (project cut r') : ℝ) :=
  stock_and_recharge cut t r r' e hp

example (cut : Bool) (n : ℕ) :
    (n : ℝ) / world.clock = (n : ℝ) / world.clock :=
  (realized cut).process.elapsed_eq n

example (cut : Bool) (r : Carrier cut) (p : Interaction.Policy ℕ (Option Unit))
    (h u : List (Option Unit)) :
    (physical cut).trace p h u r = (process true true cut).trace p h u (project cut r) :=
  (realized cut).process.representation.sameBehavior r p h u

example (cut : Bool) (t k : ℕ) (r : Carrier cut) :
    EventChannels.Kernel.firstHit (events (physical cut)) (intake cut) t k r =
      Exchange.recovery (plant true true) world (he true true) (ledger true true)
        cut (fun _ => world.sense) t k (project cut r) := recovery_preserved cut t k r

example (t : ℕ) (r : Carrier false)
    (hg : RenewableRecoveryCertificate.certificate.region (project false r)) :
    EventChannels.Kernel.firstHit (events (physical false)) (intake false)
      t world.horizon r = 1 := recovery_by_certificate t r hg

example (cut : Bool) (n : ℕ) : score cut n =
    Exchange.survival (plant true true) world (he true true) (ledger true true)
      cut (fun _ => world.sense) n := score_preserved cut n

example : (∀ n, score false n = 1) ∧
    (∀ n, 6 ≤ n → score true n + 1 / 4 ≤ score false n) := maintenance_preserved

example (cut : Bool) (t : ℕ) (r r' : Carrier cut) (e : Option Unit)
    (hp : 0 < (physical cut).step t r e r') :
    (sourceAccount cut).stored (t + 1) r' + (sourceAccount cut).load t r e r' =
      (sourceAccount cut).stored t r + (sourceAccount cut).drive t r e r' +
        (sourceAccount cut).matter t r e r' + (sourceAccount cut).heat t r e r' :=
  (sourceAccount cut).balance t r e r' hp

example (cut : Bool) : (sourceAccount cut).stored 0 (initialCarrier cut) = 6 := initial_energy cut

example (cut : Bool) (n : ℕ) :
    mean (law (physical cut) (initialCarrier cut) n) ((sourceAccount cut).stored n) +
      ∑ t ∈ Finset.range n, flux (physical cut) (initialCarrier cut) (release cut) t =
    6 + ∑ t ∈ Finset.range n,
      flux (physical cut) (initialCarrier cut) (sourceAccount cut).drive t :=
  source_energy_budget cut n

end RenewableCarrierReview20261024
