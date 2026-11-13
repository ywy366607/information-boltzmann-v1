"""Fixed-size age decomposition of the EXISTING compressed owner coefficients.

No extra state basis, historical graph or Jacobian. All cohorts share the
producer's U and its projection/drop decisions. Ages describe surviving
compressed sources, not an exact biological/all-path gradient. FP rounding
and inherited untagged sources remain in the common residual, never invented
as old sources. At a settled boundary T, `older` ends no later than T-W-1.
"""
import numpy as np


class BoundedSourceAges:
    def __init__(self, rank, owners, window=128):
        if any(type(x) is not int or x <= 0 for x in (rank, owners, window)):
            raise ValueError('positive rank/owners/window required')
        self.older = np.zeros((rank, owners), dtype=np.float64)
        self.previous = np.zeros_like(self.older)
        self.current = np.zeros_like(self.older)
        self.window = window
        self.boundary = None
        self.active = False

    @property
    def nbytes(self):
        return self.older.nbytes + self.previous.nbytes + self.current.nbytes

    def begin(self, start):
        if type(start) is not int or start < 0 or self.active:
            raise ValueError('invalid/overlapping source age window')
        if self.boundary is not None and start != self.boundary:
            raise ValueError('source age windows must be contiguous')
        self.boundary, self.active = start, True

    def insert(self, owner, receipt):
        """Mirror the producer's actual projection coefficients and row drop."""
        if not self.active:
            raise ValueError('source age insertion outside an actual window')
        owner = np.asarray(owner, dtype=np.float64)
        coefficients = np.asarray(receipt['coefficients'], dtype=np.float64)
        if (owner.shape != (self.current.shape[1],) or coefficients.shape != (self.current.shape[0],)
                or not np.isfinite(owner).all() or not np.isfinite(coefficients).all()):
            raise ValueError('invalid age owner/projection coefficients')
        drop = receipt['drop']
        if drop is not None and (not isinstance(drop, (int, np.integer)) or not 0 <= drop < len(coefficients)):
            raise ValueError('invalid shared basis drop')
        # Rowwise update bounds temporary storage by one owner row.
        for j, coefficient in enumerate(coefficients):
            if j == drop:
                self.older[j].fill(0.)
                self.previous[j].fill(0.)
                self.current[j] = owner
            else:
                self.current[j] += coefficient * owner
            if not np.isfinite(self.current[j]).all():
                raise FloatingPointError('nonfinite source age coefficients')

    def finish(self, stop):
        if not self.active or stop != self.boundary + self.window:
            raise ValueError('source age window ended at the wrong boundary')
        self.older += self.previous
        if not np.isfinite(self.older).all():
            raise FloatingPointError('nonfinite old-source accumulation')
        self.previous[:] = self.current
        self.current.fill(0.)
        self.boundary, self.active = stop, False

    def older_credit(self, contraction):
        if self.active:
            raise ValueError('contract old sources before advancing the captured window')
        contraction = np.asarray(contraction, dtype=np.float64)
        if contraction.shape != (len(self.older),) or not np.isfinite(contraction).all():
            raise ValueError('invalid shared-U contraction')
        return contraction @ self.older

    def common_residual(self, stored_v):
        return np.asarray(stored_v, dtype=np.float64) - self.older - self.previous - self.current

    def state_dict(self):
        return dict(schema=1, window=self.window, boundary=self.boundary, active=self.active,
                    **{name: getattr(self, name).copy() for name in ('older','previous','current')})

    def load_state_dict(self, state):
        if state['schema'] != 1 or state['window'] != self.window:
            raise ValueError('age schema/window mismatch')
        if state['boundary'] is not None and (type(state['boundary']) is not int or state['boundary'] < 0):
            raise ValueError('invalid age boundary')
        if type(state['active']) is not bool or (state['active'] and state['boundary'] is None):
            raise ValueError('invalid age activity')
        loaded = {}
        for name in ('older','previous','current'):
            value = state[name]
            if (not isinstance(value, np.ndarray) or value.shape != self.older.shape
                    or value.dtype != np.float64 or not np.isfinite(value).all()):
                raise ValueError('invalid stored age coefficients')
            loaded[name] = value.copy()
        if state['boundary'] is None and any(np.any(value) for value in loaded.values()):
            raise ValueError('age coefficients without an observed boundary')
        if not state['active'] and np.any(loaded['current']):
            raise ValueError('unsettled current sources in a settled age state')
        for name, value in loaded.items():
            setattr(self, name, value)
        self.boundary, self.active = state['boundary'], state['active']
