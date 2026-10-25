import CGBAxioms.SensoryMaintenanceTwoTickGlobal

open CGB

namespace TwoTickGlobalReview20261025

example :
    0 < SensoryMaintenance.body.law SensoryMaintenance.world
      (SensoryMaintenance.sensoryProtocol 2 2) 5
      (fun _ => SensoryMaintenance.empty) .probeB :=
  SensoryMaintenanceTwoTickGlobal.empty_mass_positive

example :
    Exchange.survival SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material false
      (SensoryMaintenance.sensoryProtocol 2 2) 5 < 1 :=
  SensoryMaintenanceTwoTickGlobal.two_tick_survival_lt_one

example : ¬ SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 2 :=
  SensoryMaintenanceTwoTickGlobal.not_two_tick_tolerance

end TwoTickGlobalReview20261025
