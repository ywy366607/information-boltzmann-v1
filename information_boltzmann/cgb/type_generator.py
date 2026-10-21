"""Small CPU deployment interface for fitted CGB type-matching rules.

Predictions are log synapses per possible cell pair, CONDITIONAL on belonging
to the connected-type-pair population used in V3d. Wiring occurrence is a
separate model; this module never turns a strength fit into a wiring law.
"""
from __future__ import annotations

import numpy as np


class TypeStrengthGenerator:
    def __init__(self, archive):
        self.values = {name: np.asarray(value).copy() for name, value in archive.items()}
        required = {'code', 'affinity', 'out_bias', 'in_bias', 'bias', 'geometry_weight',
                    'feature_mean', 'feature_scale', 'types', 'symmetric'}
        missing = required-set(self.values)
        if missing:
            raise ValueError('Incomplete type-generator archive: '+','.join(sorted(missing)))
        self.types = self.values['types'].astype(str)
        self.index = {name: i for i, name in enumerate(self.types)}
        affinity = self.values['affinity']
        self.affinity = (affinity+affinity.T)*.5 if bool(self.values['symmetric']) else affinity
        self._pre_code = self.values['code'] @ self.affinity

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            return cls({name: archive[name] for name in archive.files})

    def log_strength(self, pre, post, raw_features, *, block_size=4096):
        """Predict requested type pairs with bounded E x rank workspace."""
        pre, post = np.asarray(pre, dtype=np.int64), np.asarray(post, dtype=np.int64)
        raw_features = np.asarray(raw_features)
        if pre.ndim != 1 or pre.shape != post.shape or raw_features.shape != (len(pre), len(self.values['feature_mean'])):
            raise ValueError('Expected one index and one full geometry feature vector per pair')
        if (pre < 0).any() or (post < 0).any() or (pre >= len(self.types)).any() or (post >= len(self.types)).any():
            raise IndexError('Unknown type index; no invented code for unseen cell types')
        if block_size < 1:
            raise ValueError('Positive deployment block size required')
        result = np.empty(len(pre), dtype=np.float64)
        for start in range(0, len(pre), block_size):
            end = start+block_size
            a, b = pre[start:end], post[start:end]
            features = (raw_features[start:end]-self.values['feature_mean'])/self.values['feature_scale']
            result[start:end] = (self.values['bias']+self.values['out_bias'][a]+self.values['in_bias'][b]
                +features @ self.values['geometry_weight']
                +np.sum(self._pre_code[a]*self.values['code'][b], axis=1))
        return result

    def expected_synapses(self, pre, post, raw_features, source_cells, target_cells):
        """Convert the conditional per-cell-pair intensity to a type-pair total.

        exp(log prediction) is a geometric-scale estimate; residual variance
        would be needed to claim the arithmetic mean of a lognormal count.
        """
        log_value = self.log_strength(pre, post, raw_features)
        return np.exp(log_value)*np.asarray(source_cells)*np.asarray(target_cells)
