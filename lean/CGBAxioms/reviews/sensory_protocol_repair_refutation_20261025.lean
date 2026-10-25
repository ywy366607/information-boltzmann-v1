import CGBAxioms.SensoryMaintenanceProtocolRepairRefutation

open CGB

namespace SensoryProtocolRepairRefutationReview20261025

example :
    0 < SensoryMaintenance.body.law SensoryMaintenance.world
      (SensoryMaintenanceRepairCandidates.protectedTwoTickProtocol 2) 5
      (fun _ => SensoryMaintenance.empty) .probeB :=
  SensoryMaintenanceProtocolRepairRefutation.protected_protocol_empty_mass_positive

example :
    Exchange.survival SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material false
      (SensoryMaintenanceRepairCandidates.protectedTwoTickProtocol 2) 5 < 1 :=
  SensoryMaintenanceProtocolRepairRefutation.protected_protocol_survival_lt_one

end SensoryProtocolRepairRefutationReview20261025
