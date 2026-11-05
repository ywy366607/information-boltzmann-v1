"""V4q closed loop with witness scoring on current parameters and fast table.

As in the inherited witness protocol, scoring uses a cold recurrent state.
"""
from scripts.ib_local.language_navigation_kanli_v4q import SharedContextWoodEngine
from scripts.ib_local.navigation_live_witness import current_body_witness_nll


class CurrentWitnessEngine(SharedContextWoodEngine):
    def witness_nll(self):
        # The inherited cycle calls this exactly before and after learning,
        # before adding the next witness. Both readings see the same text.
        old = super().witness_nll()
        current = current_body_witness_nll(self)
        self._witness_readings.append((old, current))
        return current

    def step(self, directions, age_ablate=False):
        self._witness_readings = []
        out = super().step(directions, age_ablate)
        if len(self._witness_readings) != 2:
            raise RuntimeError('unexpected witness scoring cadence')
        (old_before, cur_before), (old_after, cur_after) = self._witness_readings
        out.update(stored_table_g=old_before-old_after,
                   current_table_g=cur_before-cur_after,
                   witness_current_pre=cur_before, witness_current_post=cur_after)
        return out
