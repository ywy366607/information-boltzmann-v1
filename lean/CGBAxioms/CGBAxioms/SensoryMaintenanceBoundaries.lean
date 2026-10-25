import CGBAxioms.SensoryMaintenanceLimits

/-!
# Exact local blackout boundary and recharge condition

The two-loss boundary is a state-level result: a supported nonvacuum cell has
zero reserve but remains nonempty after two absent-sensing ticks, and reaches
the absorbing empty state on the third. A normal signal can refill the reserve
only when the resulting structural cell remains nonvacuum. These statements do
not by themselves assert a full recovery score for every future protocol.
-/

namespace CGB.SensoryMaintenanceBoundaries
open RenewableLearning

theorem two_tick_functional_boundary (q : Cell) (hq : q ≠ .vacuum) :
    SensoryMaintenance.burn (q, 2) 2 ≠ SensoryMaintenance.empty ∧
    (SensoryMaintenance.burn (q, 2) 2).2.val = 0 ∧
    SensoryMaintenance.burn (q, 2) 3 = SensoryMaintenance.empty := by
  have hb := SensoryMaintenance.funded_two_tick_bridge q hq
  refine ⟨?_, ?_, hb.2.2⟩
  · rw [hb.2.1]
    intro he
    apply hq
    exact congrArg Prod.fst he
  · rw [hb.2.1]
    rfl

theorem sensed_recharge (a : Signal) (x : SensoryMaintenance.State)
    (ha : 0 < RenewableLearning.incoming true a x.1)
    (hx : x.1 ≠ .vacuum)
    (hy : (SensoryMaintenance.next (some a) x).1 ≠ .vacuum) :
    0 < SensoryMaintenance.incoming (some a) x ∧
      (SensoryMaintenance.next (some a) x).2.val = 2 := by
  rcases x with ⟨q, r⟩
  fin_cases r <;> cases a <;> cases q <;>
    norm_num [SensoryMaintenance.incoming, SensoryMaintenance.next,
      SensoryMaintenance.empty, RenewableLearning.next,
      RenewableLearning.incoming] at *

theorem two_tick_recharge_condition (q : Cell) (hq : q ≠ .vacuum) (a : Signal)
    (ha : 0 < RenewableLearning.incoming true a q)
    (hy : (SensoryMaintenance.next (some a)
      (SensoryMaintenance.burn (q, 2) 2)).1 ≠ .vacuum) :
    0 < SensoryMaintenance.incoming (some a)
        (SensoryMaintenance.burn (q, 2) 2) ∧
      (SensoryMaintenance.next (some a)
        (SensoryMaintenance.burn (q, 2) 2)).2.val = 2 := by
  exact sensed_recharge a (SensoryMaintenance.burn (q, 2) 2) (by
    simpa [SensoryMaintenance.funded_two_tick_bridge q hq |>.2.1] using ha) (by
    rw [SensoryMaintenance.funded_two_tick_bridge q hq |>.2.1]
    exact hq) hy

end CGB.SensoryMaintenanceBoundaries
