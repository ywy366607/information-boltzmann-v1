"""Optional source-age refinement of the SAME approximate native-S code.

Only surviving, actually captured sources strictly before incoming T-128 are
identified as old. Unknown inherited sources/FP rounding stay common. This
does not make the local S-only approximation a full-network derivative.
"""
import numpy as np

from scripts.ib_local.bounded_source_ages import BoundedSourceAges
from scripts.ib_local.native_s_eligibility import NativeSEligibility, WINDOW


class AgeResolvedNativeSEligibility(NativeSEligibility):
    def __init__(self, *args, **kwargs):
        self.source_ages = None
        self.strict_old_gradients = {}
        super().__init__(*args, **kwargs)
        self.source_ages = BoundedSourceAges(self.rank, self.owners, WINDOW)
        self._guard(self.nbytes)

    @property
    def nbytes(self):
        return (super().nbytes + (0 if self.source_ages is None else self.source_ages.nbytes)
                + sum(value.nbytes for value in self.strict_old_gradients.values()))

    def _insert_source(self, source, owner):
        receipt = super()._insert_source(source, owner)
        self.source_ages.insert(owner, receipt)
        return receipt

    def advance_window(self):
        if not self.is_complete():
            raise ValueError('cannot advance incomplete source ages')
        start = self.pending['start_seen']
        self.source_ages.begin(start)
        result = super().advance_window()
        self.source_ages.finish(start + WINDOW)
        result.update(source_age_resolved=True, persistent_bytes=self.nbytes,
                      strict_old_source_exclusive_stop=start)
        self._guard(self.nbytes)
        return result

    def raw_credit(self, teaching):
        raw = super().raw_credit(teaching)
        states = self._states(teaching, copy=False)
        contraction = np.zeros(self.rank, dtype=np.float64)
        for (layer, head), name in self.layout.head_fields.items():
            sl = self.layout.slices[name]
            values = states[layer][head].reshape(-1)
            for lo in range(sl.start, sl.stop, self.block_size):
                hi = min(lo + self.block_size, sl.stop)
                actual = values[lo-sl.start:hi-sl.start].astype(np.float64)
                for j in range(self.rank):
                    contraction[j] += float(self.code.U[j,lo:hi].astype(np.float64) @ actual)
        old = self.source_ages.older_credit(contraction)
        self.strict_old_gradients = {
            name: old[sl].astype(np.float32).reshape(self.parameter_shapes[name])
            for name, sl in self.parameter_slices.items()}
        self.last_credit.update(source_age_resolved=True,
            strict_old_source_exclusive_stop=(None if self.clock['seen'] is None else self.clock['seen']-WINDOW),
            strict_old_norm=float(np.linalg.norm(old)),
            common_owner_residual_norm=float(np.linalg.norm(self.source_ages.common_residual(self.code.V))),
            scope='surviving compressed local-S sources; untagged/rounding common, omitted paths unchanged')
        self._guard(self.nbytes)
        return raw

    def state_dict(self):
        state = super().state_dict()
        state['source_ages'] = self.source_ages.state_dict()
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.source_ages = BoundedSourceAges(self.rank, self.owners, WINDOW)
        if state.get('source_ages') is not None:
            self.source_ages.load_state_dict(state['source_ages'])
            unmapped = self.source_ages.boundary is None and not any(
                np.any(getattr(self.source_ages, n)) for n in ('older','previous','current'))
            if self.source_ages.active or (not unmapped and self.source_ages.boundary != self.clock['seen']):
                raise ValueError('source age boundary differs from settled native-S state')
        # Legacy migration does not assign ages to inherited nonzero U/V.
        self.strict_old_gradients = {}
        self._guard(self.nbytes)
