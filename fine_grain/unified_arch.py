"""One Native MoT graph for every I/O port.

Write is always Transolver residual (increment). Text is a condition, not a
canvas replacement. Ports differ by initial X and the loss head only.

Do not add port-only forks here (no s_lang_topk, no write A for VQA).
Old Champion B / F2 write-A checkpoints stay on their own knobs.
"""
from __future__ import annotations

from typing import Any, Dict

# Shared by DualStreamOmni and the triad DualStreamVQAModel.
UNIFIED_KNOBS = {
    "surprise_mode": "v1_bayes",
    "s_update": "rms_dir",
    "prior_loss_coef": 0.1,
    "use_stiefel": True,
    "deslice_topk": 2,
    "gate_on": "u",
    "deslice_write": "increment",
    "gate_h_local": False,
    "vfe_coef": 0.1,
    "s_lang_topk": 0,
    "s_kalman_update": False,
    "prior_write": 0.0,
    "use_null_slice": False,
    "pack_by_surprise": False,
    "hard_admit": False,
    "use_yield_read": True,
}

# Omni scale used for the six I/O ports (matches prior dedicated omni runs).
UNIFIED_OMNI_SIZE = {
    "d_model": 256,
    "n_slices": 64,
    "n_heads": 8,
    "n_layers": 4,
    "res": 32,
}

# Recognition triad scale (matches F2 / Champion B).
UNIFIED_TRIAD_SIZE = {
    "d_model": 128,
    "n_slices": 32,
    "n_heads": 4,
    "n_layers": 4,
    "res": 32,
}

OMNI_PORTS = ("t2t", "i2t", "it2t", "recon", "i2i", "t2i")


def unified_kwargs(**overrides: Any) -> Dict[str, Any]:
    kw = dict(UNIFIED_KNOBS)
    kw.update(overrides)
    return kw
