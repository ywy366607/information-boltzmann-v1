"""Evidence-scoped graded release in the existing early visual circuit.

No added visual state, feedback edge, semantic input or private visual readout.
Membrane, conductance, adaptation and delays remain the shared physical engine.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd
import torch


VISUAL_PARAMETER_NAMES = ('log_visual_graded_gain',)
# Graded photoreceptor/lamina communication: Zheng et al. 2006.
# Direct in-vivo recordings of Mi1/Tm3/Tm1/Tm2: Behnia et al. 2014.
# Other optic types are deliberately left to the existing engine.
VISUAL_GRADED_CORE_TYPES = ('R1-R6', 'R7*', 'R8*', 'L1', 'L2', 'L3',
                           'Mi1', 'Tm3', 'Tm1', 'Tm2')


def attach_visual_physiology(model, graph_npz: str | Path, annotations: str | Path,
                             *, gain: float = 0.025):
    if getattr(model, 'visual_graded_enabled', False):
        return dict(model.visual_graded_counts)
    graph = np.load(graph_npz, allow_pickle=False)
    ann = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])
    typ = ann['type'].fillna('').astype(str).to_numpy()
    selected = np.array([any(fnmatch.fnmatchcase(t, pattern)
                            for pattern in VISUAL_GRADED_CORE_TYPES) for t in typ])
    original = getattr(model, 'graded_index', None)
    if original is not None:
        selected[original.detach().cpu().numpy()] = False
    index = np.flatnonzero(selected)
    if not index.size:
        raise ValueError('No additional evidence-scoped visual graded cells in this graph')
    model.attach_visual_graded(index, gain=gain, floor=float(model.E_I))
    model.visual_graded_counts = {t: int(np.sum(typ[index] == t)) for t in sorted(set(typ[index]))}
    model.visual_graded_enabled = True
    return dict(model.visual_graded_counts)
