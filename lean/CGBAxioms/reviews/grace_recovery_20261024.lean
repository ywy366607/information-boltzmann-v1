import CGBAxioms.GraceRecoveryClosure
import CGBAxioms.GraceRecoveryAccounting
import CGBAxioms.GraceRecoveryControls

/- Independent interface checks: actual protocols, original cut, and source ledgers. -/
open CGB CGB.RenewableLearning

namespace GraceRecoveryReview20261024

example : GraceRecovery.body = plant true true ∧ GraceRecovery.world.clock = 1 ∧
    GraceRecovery.world.horizon = 3 := ⟨rfl, rfl, rfl⟩

example (q : Cell) (w v : Env) (h : GraceRecovery.Good q w) :
    GraceRecovery.moves w (fun _ => q) v = RenewableLearning.moves w (fun _ => q) v :=
  GraceRecovery.moves_unchanged_on_good q w v h

example (n₀ n : ℕ) : Exchange.survival GraceRecovery.body GraceRecovery.world
    GraceRecovery.ec GraceRecovery.material false (GraceRecovery.interrupted n₀) n = 1 :=
  GraceRecovery.one_tick_survival n₀ n

example : Exchange.ToleratesSensoryInterruption GraceRecovery.body GraceRecovery.world
    GraceRecovery.ec GraceRecovery.material 0 1 := GraceRecovery.one_tick_tolerance

example (n : ℕ) : Exchange.survival GraceRecovery.body GraceRecovery.world
    GraceRecovery.ec GraceRecovery.material false GraceRecovery.natural n = 1 :=
  GraceRecovery.natural_survival n

example : Exchange.survival GraceRecovery.body GraceRecoveryControls.offWorld
    GraceRecoveryControls.offClock GraceRecovery.material false
    GraceRecoveryControls.offProtocol 6 ≤ 1 / 2 :=
  GraceRecoveryControls.horizon_change_alone_fails

example : Exchange.recovery GraceRecovery.body GraceRecovery.world GraceRecovery.ec
    GraceRecovery.material false (GraceRecovery.interrupted 5) 4 2
      ((fun _ => .aReady), .missB) = 0 :=
  GraceRecoveryControls.two_tick_lookahead_insufficient

example : Exchange.recovery GraceRecovery.body GraceRecovery.world GraceRecovery.ec
    GraceRecovery.material false (GraceRecovery.interrupted 5) 4 3
      ((fun _ => .aReady), .missB) = 1 := by
  let C := GraceRecovery.certificate 5
  have h : C.region 4 ((fun _ => .aReady), .missB) := by
    simp [C, GraceRecovery.certificate, GraceRecovery.region, GraceRecovery.Good,
      RenewableLearningLife.Good]
  exact ScheduledRecovery.ranked_firstHit C 4 3 _ h (C.bounded 4 _ h)

example (n : ℕ) (hn : 7 ≤ n) : Exchange.survival GraceRecovery.body GraceRecovery.world
    GraceRecovery.ec GraceRecovery.material true GraceRecovery.natural n ≤ 3 / 4 :=
  GraceRecoveryClosure.cut_survival n hn

example : Exchange.Persists GraceRecovery.body GraceRecovery.world GraceRecovery.ec
    GraceRecovery.material 0 ∧ Exchange.ToleratesSensoryInterruption GraceRecovery.body
    GraceRecovery.world GraceRecovery.ec GraceRecovery.material 0 1 ∧
    Exchange.SelfMaintains GraceRecovery.body GraceRecovery.world GraceRecovery.ec
      GraceRecovery.material := GraceRecoveryClosure.maintenance

example (cut : Bool) (sched : ℕ → Env → Unit → Signal) (p : ℕ → S)
    (events : ℕ → Option Unit)
    (hp : ∀ t, 0 < (GraceRecovery.process cut sched).step t (p t) (events t) (p (t + 1)))
    (n : ℕ) := GraceRecoveryAccounting.path_budgets cut sched p events hp n

example (k : ℕ) := GraceRecoveryAccounting.arbitrary_relearning k

example (cut : Bool) (sched : ℕ → Env → Unit → Signal) :=
  GraceRecoveryAccounting.realized cut sched

example (cut : Bool) (sched : ℕ → Env → Unit → Signal) (n : ℕ) :=
  GraceRecoveryAccounting.source_energy_budget cut sched n

example (cut : Bool) (sched : ℕ → Env → Unit → Signal) (t k : ℕ)
    (r : GraceRecoveryAccounting.Carrier cut sched) :=
  GraceRecoveryAccounting.recovery_preserved cut sched t k r

example (cut : Bool) (sched : ℕ → Env → Unit → Signal) (n : ℕ) :
    GraceRecoveryAccounting.score cut sched n = Exchange.survival GraceRecovery.body
      GraceRecovery.world GraceRecovery.ec GraceRecovery.material cut sched n :=
  GraceRecoveryAccounting.score_preserved cut sched n

example (n₀ n : ℕ) : GraceRecoveryAccounting.score false
    (GraceRecovery.interrupted n₀) n = 1 :=
  GraceRecoveryAccounting.physical_one_tick_survival n₀ n

example (n₀ : ℕ) (z z' : S) (e : Option Unit) :
    GraceRecoveryAccounting.declared (GraceRecovery.interrupted n₀) (incoming true)
      n₀ z e z' = 0 := GraceRecoveryAccounting.calm_has_no_intake n₀ z z' e

end GraceRecoveryReview20261024
