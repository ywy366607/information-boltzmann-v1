"""Bridge a qualified contact population and fitted strength prior into V2.

The caller supplies morphology contacts, occurrence probabilities and cell
physiology explicitly. V3d estimates conditional strength; it does NOT supply
those other three laws. This bridge creates no dense cell-by-cell matrix and
does not read/copy a measured adjacency table.

Bernoulli independence and the strength-to-slot sigmoid are explicit interface
assumptions. A bounded occupancy shape does not identify binding rates or
derive this conversion from connectome statistics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from .medium import Medium, SynapseParams
from .structural_budget import MaterialParams


@dataclass(frozen=True)
class BirthParams:
    conductance_scale: float   # explicit, unfitted conversion from strength prior to slot-binding odds
    seed: int = 0
    dt: float = 0.25

    def __post_init__(self):
        if not np.isfinite(self.conductance_scale) or self.conductance_scale <= 0:
            raise ValueError('A positive finite strength-to-physiology calibration is required')
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('A positive finite integration step is required')


def seed_from_contacts(generator, cell_type, classes, cell_class, pre, post,
                       raw_features, connection_probability, *, params,
                       syn=None, material=None, initial_free=0.0):
    """Sample supplied opportunities, initialize slots, and pay the birth cost.

    One entry per unique directed cell-pair opportunity, not per branch contact.
    Independent Bernoulli occurrence is a caller-supplied interface assumption.
    A structural slot occupancy gmax * odds/(1+odds) converts the fitted strength
    to conductance; its scale is not established by the connectome regression.
    With material enabled, even the initial seed must fit local available stock.
    """
    cell_type = np.asarray(cell_type, dtype=np.int64)
    cell_class = np.asarray(cell_class, dtype=np.int64)
    pre, post = np.asarray(pre, dtype=np.int64), np.asarray(post, dtype=np.int64)
    raw_features = np.asarray(raw_features)
    probability = np.asarray(connection_probability, dtype=float)
    if cell_type.ndim != 1 or cell_class.shape != cell_type.shape:
        raise ValueError('One type and physiology class per cell required')
    if pre.ndim != 1 or post.shape != pre.shape or probability.shape != pre.shape:
        raise ValueError('One occurrence probability per directed opportunity required')
    n = len(cell_type)
    if ((pre < 0).any() or (post < 0).any() or (pre >= n).any() or (post >= n).any()):
        raise ValueError('Contact contains an unknown cell')
    if ((cell_type < 0).any() or (cell_type >= len(generator.types)).any() or
            (cell_class < 0).any() or (cell_class >= len(classes)).any()):
        raise ValueError('Unknown type or physiology class')
    if (pre == post).any() or len(np.unique(pre * np.int64(n) + post)) != len(pre):
        raise ValueError('Provide unique non-self cell-pair opportunities')
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Occurrence probabilities must be supplied in [0, 1]')
    if raw_features.shape != (len(pre), len(generator.values['feature_mean'])) or not np.isfinite(raw_features).all():
        raise ValueError('A full finite morphology feature vector per opportunity is required')
    syn = syn or SynapseParams()
    if not np.isfinite(syn.gmax) or syn.gmax <= 0:
        raise ValueError('Positive finite slot capacity required')
    rng = np.random.default_rng(params.seed)
    keep = rng.random(len(pre)) < probability
    a, b = pre[keep], post[keep]
    logs = generator.log_strength(cell_type[a], cell_type[b], raw_features[keep])
    if not np.isfinite(logs).all():
        raise ValueError('Matching rule produced nonfinite strength')
    desired = syn.gmax * expit(logs + np.log(params.conductance_scale))
    medium = Medium(classes, cell_class, a, b, np.zeros(len(a)), syn=syn,
                    dt=params.dt, seed=params.seed)
    if material is not None:
        if not isinstance(material, MaterialParams):
            raise TypeError('Expected explicit MaterialParams')
        medium.enable_material(material, initial_free)
    medium._set_structure(desired)
    # Preserve the occurrence RNG after birth for exact future continuation.
    medium.rng.bit_generator.state = rng.bit_generator.state
    report = {'candidate_opportunities': len(pre), 'sampled_edges': len(a),
              'desired_structure': float(desired.sum()), 'funded_structure': float(medium.g.sum()),
              'seed': params.seed, 'conductance_scale': params.conductance_scale,
              'material_enabled': material is not None,
              'scope': 'Strength-prior-to-local-medium interface; morphology, occurrence and physiology supplied',
              'unverified': ['developmental morphology generation', 'occurrence calibration',
                             'strength-to-conductance calibration', 'world learning and agency']}
    return medium, report
