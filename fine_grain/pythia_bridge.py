"""Phase B helpers: load a generation champion into a frozen-Pythia Omni.

Pythia supplies embeddings and (later) token NLL. Visual Slice–MoT, RGB, and
segmentation stay the published generation graph. Toy embeddings, the class
head, and width-mismatched text_in/text_out/proj are never copied.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

LANGUAGE_INTERFACE_PREFIXES: tuple[str, ...] = (
    "embed.",
    "head.",
    "gaze_head.",
    "mot_stack.text_in.",
    "mot_stack.text_out.",
    "mot_stack.proj.",
    "lm.",
)

INTERFACE_TRAINABLE_PREFIXES: tuple[str, ...] = (
    "mot_stack.text_in.",
    "mot_stack.text_out.",
)

TOKEN_INTERFACE_PREFIXES: tuple[str, ...] = (
    "mot_stack.text_out.",
    "mot_stack.proj.",
    "mot_stack.text_out_gate",
    "mot_stack.proj_gate",
)

# Language readers: maps plus MoT text experts and the F2 language prior.
# Visual stem / SliceRead / Deslice / pix_head stay frozen in this phase.
LANGUAGE_READER_MARKERS: tuple[str, ...] = INTERFACE_TRAINABLE_PREFIXES + (
    ".mot.Wq_t",
    ".mot.Wk_t",
    ".mot.Wv_t",
    ".mot.Wo_t",
    ".mot.rms_t.",
    ".mot.rms_t_ffn.",
    ".mot.ffn_t.",
    ".mot.res_t",
    ".h_in_norm.",
    ".surprise_gate.prior_head.",
    ".surprise_gate.slice_queries",
    ".surprise_gate.q_norm.",
    ".surprise_gate.h_norm.",
)

# Stage-2 named edit: read geometry without opening the global RGB shortcut.
# Query/output only — not visual K/V or FFN. Stem and pix_head stay frozen.
EDIT_READ_MARKERS: tuple[str, ...] = (
    ".read.",
    ".deslice.",
    ".mot.Wq_v",
    ".mot.Wo_v",
)

# Text-to-image adaptation moves only the language-to-visual messages and the
# shared Slice/Deslice RGB write.  It deliberately leaves the visual stem and
# terminal language reader frozen, so a finite natural-image capacity probe is
# not implemented by replacing the perception or language path.
GENERATION_WRITE_MARKERS: tuple[str, ...] = (
    "mot_stack.text_in.",
    ".mot.Wk_t",
    ".mot.Wv_t",
    ".surprise_gate.prior_head.",
    ".surprise_gate.slice_queries",
    ".surprise_gate.q_norm.",
    ".surprise_gate.h_norm.",
)

GENERATION_CAPACITY_MARKERS: tuple[str, ...] = (
    "mot_stack.stem.",
    "mot_stack.stem_local.",
    ".mot.Wk_v",
    ".mot.Wv_v",
    ".mot.ffn_v.",
    ".local.",
)

FROZEN_EVEN_WHEN_JOINT_PREFIXES: tuple[str, ...] = (
    "lm.",
    "embed.",
    "head.",
    "gaze_head.",
)

GENERATION_CHAMPION_PATH = Path(
    "checkpoints/omni_d64_northstar_omni_active_f2_grid_best.pt"
)
CAPABILITY_CHAMPION_PATH = Path(
    "checkpoints/northstar_slice_capability_best.pt"
)

# Matches scripts/probe_northstar_generation.py --recipe active_f2.
GENERATION_CHAMPION_KNOBS: Dict = {
    "d_model": 64,
    "n_slices": 16,
    "n_heads": 4,
    "n_layers": 4,
    "res": 16,
    "surprise_mode": "v1_bayes",
    "s_update": "raw",
    "prior_loss_coef": 0.1,
    "sigreg_coef": 0.0,
    "use_stiefel": False,
    "deslice_topk": 0,
    "deslice_write": "increment",
    "gate_h_local": False,
    "vfe_coef": 0.1,
    "use_null_slice": False,
    "use_residual_read": False,
    "fm_pred": "x",
    "fm_signed": False,
    "prior_write": 1.0,
    "prior_write_by_t": False,
    "pixel_loss_mode": "balanced_bce",
    "spatial_prompt_vocab": True,
    "s0_acc_coef": 0.0,
}

# Exact static-capability graph used by northstar_slice_capability_best.pt.
# Unlike the generation-only bridge, this graph already has evidence for
# reconstruction, segmentation, and next-color editing in one Slice chart.
CAPABILITY_CHAMPION_KNOBS: Dict = {
    **GENERATION_CHAMPION_KNOBS,
    "gate_on": "u",
    "capability_vocab": True,
    "use_modal_precision": True,
    "use_target_time": True,
    "seg_classes": 2,
    "seg_loss_coef": 1.0,
}


def is_language_interface_key(name: str) -> bool:
    return any(name == p[:-1] or name.startswith(p) for p in LANGUAGE_INTERFACE_PREFIXES)


def is_interface_trainable(name: str) -> bool:
    return any(name.startswith(p) for p in INTERFACE_TRAINABLE_PREFIXES)


def is_language_reader(name: str) -> bool:
    return any(marker in name or name.startswith(marker) for marker in LANGUAGE_READER_MARKERS)


def is_token_interface(name: str) -> bool:
    return (
        name.startswith("mot_stack.text_out")
        or name.startswith("mot_stack.proj")
        or name.startswith("mot_stack.terminal_atlas_")
    )


def is_terminal_token_reader(model: nn.Module, name: str) -> bool:
    """Read-only terminal token adapters; none can alter the final X field."""
    if name.startswith("mot_stack.readout."):
        return True
    layers = getattr(getattr(model, "mot_stack", None), "layers", ())
    if not layers:
        return False
    prefix = f"mot_stack.layers.{len(layers) - 1}.mot."
    if not name.startswith(prefix):
        return False
    suffix = name[len(prefix):]
    return suffix.startswith((
        "Wq_t.", "Wo_t.", "rms_t_ffn.", "ffn_t.", "res_t",
    ))


def is_edit_read_visual(name: str) -> bool:
    return any(marker in name for marker in EDIT_READ_MARKERS)


def is_generation_write(name: str) -> bool:
    return (
        any(marker in name or name.startswith(marker)
            for marker in GENERATION_WRITE_MARKERS)
        or is_edit_read_visual(name)
        or "precision_coord" in name
        or name.startswith("pix_head.")
        or name.startswith("pix_log")
    )


def is_generation_capacity(name: str) -> bool:
    """Existing spatial content evolution needed for dense-image capacity."""
    return is_generation_write(name) or any(
        marker in name or name.startswith(marker)
        for marker in GENERATION_CAPACITY_MARKERS
    )


def generation_champion_kwargs(**overrides) -> Dict:
    kw = dict(GENERATION_CHAMPION_KNOBS)
    kw.update(overrides)
    return kw


def capability_champion_kwargs(**overrides) -> Dict:
    """Return the proven static capability graph with optional LM binding."""
    kw = dict(CAPABILITY_CHAMPION_KNOBS)
    kw.update(overrides)
    return kw


def _state_from_ckpt(raw) -> Dict[str, torch.Tensor]:
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        return raw["state_dict"]
    if isinstance(raw, dict) and all(torch.is_tensor(v) for v in raw.values()):
        return raw
    if isinstance(raw, dict) and any(k.startswith("mot_stack.") for k in raw):
        return {k: v for k, v in raw.items() if torch.is_tensor(v)}
    raise TypeError("checkpoint does not contain a parameter state_dict")


def load_visual_champion(
    model: nn.Module,
    path: str | Path,
    skip_language_interface: bool = True,
) -> Dict:
    """Copy shape-compatible visual / Slice–MoT / RGB / seg weights.

    Toy embedding, class head, and Pythia-width text_in/text_out/proj are
    excluded even when a leftover tensor happens to match shape.
    """
    raw = torch.load(path, map_location="cpu")
    state = _state_from_ckpt(raw)
    current = model.state_dict()
    loaded: List[str] = []
    skipped: List[str] = []
    for key, value in state.items():
        if skip_language_interface and is_language_interface_key(key):
            skipped.append(key)
            continue
        if key not in current:
            skipped.append(key)
            continue
        if not torch.is_tensor(value) or current[key].shape != value.shape:
            skipped.append(key)
            continue
        current[key] = value
        loaded.append(key)
    model.load_state_dict(current)
    freeze_lm = getattr(model, "_freeze_lm", None)
    if callable(freeze_lm):
        freeze_lm()
    return {
        "path": str(path),
        "loaded": len(loaded),
        "skipped": skipped,
        "n_skipped": len(skipped),
    }


def set_optimization_phase(model: nn.Module, phase: str) -> List[str]:
    """Select which Omni parameters may move.

    ``interface``: only text_in/text_out.
    ``token_interface``: text_out, proj, and their zero-init residual gates.
    ``token_reader``: token_interface plus terminal SliceRead(X) and only the
    final MoT layer's H-query, H-output, and H-FFN. Text K/V and every visual
    write parameter stay fixed, so the final visual field cannot change.
    ``language``: those maps plus MoT text experts and the F2 language prior.
    ``rgb_likelihood``: only the shared RGB mean/log-variance likelihood head.
    ``language_rgb``: language readers plus the RGB head (not stem/Deslice).
    ``edit_spatial``: language readers plus image/text precision coords;
    pix_head, stem, SliceRead, and Deslice stay frozen.
    ``edit_read``: edit_spatial plus SliceRead, visual MoT query/output,
    and Deslice. Stem and pix_head stay frozen. Use visual_lr=1e-5.
    ``generation_write``: language-to-visual K/V, F2 prior, Slice/Deslice,
    visual query/output, modality precision, and the shared RGB likelihood.
    The visual stem and terminal language reader stay frozen.
    ``generation_capacity``: generation_write plus the existing visual stem,
    visual K/V+FFN, and local full-resolution evolution.  This is an explicit
    capacity diagnostic, not a safe continual-learning phase.
    ``joint``: all non-Pythia weights except the leftover toy embed/class head.
    """
    phase = str(phase).lower()
    if phase not in (
        "interface", "token_interface", "token_reader", "language",
        "rgb_likelihood", "language_rgb", "edit_spatial", "edit_read",
        "generation_write", "generation_capacity", "joint",
    ):
        raise ValueError(f"unknown optimization phase {phase!r}")
    freeze_lm = getattr(model, "_freeze_lm", None)
    if callable(freeze_lm):
        freeze_lm()
    trainable: List[str] = []
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in FROZEN_EVEN_WHEN_JOINT_PREFIXES):
            param.requires_grad_(False)
            continue
        if phase == "interface":
            allow = is_interface_trainable(name)
        elif phase == "token_interface":
            allow = is_token_interface(name)
        elif phase == "token_reader":
            allow = is_token_interface(name) or is_terminal_token_reader(model, name)
        elif phase == "language":
            allow = is_language_reader(name)
        elif phase == "rgb_likelihood":
            allow = name.startswith("pix_head.") or name.startswith("pix_log")
        elif phase == "language_rgb":
            allow = is_language_reader(name) or name.startswith("pix_head.") or name.startswith("pix_log")
        elif phase == "edit_spatial":
            allow = is_language_reader(name) or "precision_coord" in name
        elif phase == "edit_read":
            allow = (
                is_language_reader(name)
                or "precision_coord" in name
                or is_edit_read_visual(name)
            )
        elif phase == "generation_write":
            allow = is_generation_write(name)
        elif phase == "generation_capacity":
            allow = is_generation_capacity(name)
        else:
            allow = True
        param.requires_grad_(allow)
        if allow:
            trainable.append(name)
    return trainable


def param_groups(
    model: nn.Module,
    *,
    interface_lr: float,
    visual_lr: float,
) -> List[Dict]:
    """Language readers at ``interface_lr``; any unfrozen visual at ``visual_lr``."""
    language, visual = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            is_language_reader(name)
            or is_token_interface(name)
            or is_terminal_token_reader(model, name)
            or "precision_coord" in name
        ):
            language.append(param)
        else:
            visual.append(param)
    groups = []
    if language:
        groups.append({"params": language, "lr": float(interface_lr)})
    if visual:
        groups.append({"params": visual, "lr": float(visual_lr)})
    if not groups:
        raise RuntimeError("no trainable parameters; check set_optimization_phase")
    return groups
