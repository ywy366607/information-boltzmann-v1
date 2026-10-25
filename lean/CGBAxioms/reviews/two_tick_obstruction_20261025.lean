import CGBAxioms.SensoryMaintenanceTwoTickObstruction
import CGBAxioms.InformationThermodynamics

open CGB CGB.RenewableLearning CGB.SensoryMaintenance

namespace TwoTickObstructionReview20261025

example :
    GraceRecovery.moves .feedA (fun _ => .aSpent) .feedB = (1 : ℝ) / 2 ∧
    GraceRecovery.moves .feedB (fun _ => .aSpent) .missB = 1 :=
  SensoryMaintenanceTwoTickObstruction.two_tick_world_branch

example :
    SensoryMaintenance.next (some (.miss : Signal))
        (SensoryMaintenance.burn (.aSpent, 2) 2) = SensoryMaintenance.empty :=
  SensoryMaintenanceTwoTickObstruction.two_tick_local_failure

example (r : InformationThermodynamics.RecorderContract)
    (hc : 0 < r.erasureCost) (hb : 0 < r.erasedBits) :
    0 < r.dissipation :=
  InformationThermodynamics.positive_erasure_requires_positive_dissipation r hc hb

end TwoTickObstructionReview20261025
