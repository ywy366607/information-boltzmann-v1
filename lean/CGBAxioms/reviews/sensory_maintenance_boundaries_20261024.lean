import CGBAxioms.SensoryMaintenanceBoundaries

open CGB.RenewableLearning
open CGB.SensoryMaintenanceBoundaries

namespace SensoryMaintenanceBoundariesReview20261024

example (q : Cell) (hq : q ≠ .vacuum) :
    CGB.SensoryMaintenance.burn (q, 2) 2 ≠ CGB.SensoryMaintenance.empty ∧
    (CGB.SensoryMaintenance.burn (q, 2) 2).2.val = 0 ∧
    CGB.SensoryMaintenance.burn (q, 2) 3 = CGB.SensoryMaintenance.empty :=
  two_tick_functional_boundary q hq

example (a : CGB.RenewableLearning.Signal) (x : CGB.SensoryMaintenance.State)
    (ha : 0 < CGB.RenewableLearning.incoming true a x.1)
    (hx : x.1 ≠ CGB.RenewableLearning.Cell.vacuum)
    (hy : (CGB.SensoryMaintenance.next (some a) x).1 ≠ CGB.RenewableLearning.Cell.vacuum) :
    0 < CGB.SensoryMaintenance.incoming (some a) x ∧
      (CGB.SensoryMaintenance.next (some a) x).2.val = 2 :=
  sensed_recharge a x ha hx hy

example (q : CGB.RenewableLearning.Cell) (hq : q ≠ CGB.RenewableLearning.Cell.vacuum)
    (a : CGB.RenewableLearning.Signal)
    (ha : 0 < CGB.RenewableLearning.incoming true a q)
    (hy : (CGB.SensoryMaintenance.next (some a)
      (CGB.SensoryMaintenance.burn (q, 2) 2)).1 ≠ CGB.RenewableLearning.Cell.vacuum) :
    0 < CGB.SensoryMaintenance.incoming (some a)
        (CGB.SensoryMaintenance.burn (q, 2) 2) ∧
      (CGB.SensoryMaintenance.next (some a)
        (CGB.SensoryMaintenance.burn (q, 2) 2)).2.val = 2 :=
  two_tick_recharge_condition q hq a ha hy

end SensoryMaintenanceBoundariesReview20261024
