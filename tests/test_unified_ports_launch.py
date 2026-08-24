"""One unified Native MoT instance serves recon / i2t / t2t / t2i / i2i."""
from __future__ import annotations

import numpy as np
import torch

from fine_grain.omni_model import DualStreamOmni
from fine_grain.omni_tasks import one_sample


PORTS = ("recon", "i2t", "t2t", "t2i", "i2i")


def _run_ports(seed: int = 0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = DualStreamOmni.unified(
        d_model=32, n_slices=8, n_heads=4, n_layers=1, res=8,
        fm_pred="x", fm_signed=True,
    )
    stack_id = id(m.mot_stack)
    rec = {}
    for kind in PORTS:
        s = one_sample(rng, 8, kind, t2i_canvas="black", t2i_place="center")
        img = s["image"]
        if img.dim() == 3:
            img = img.unsqueeze(0)
        prompt = s["prompt"]
        if isinstance(prompt, str):
            prompt = [prompt]
        need_pix = [bool(s["need_pix"])]
        need_text = [bool(s["need_text"])]
        t = torch.ones(img.shape[0]) if need_pix[0] else None
        out = m(img, prompt, need_pix=need_pix, t=t)
        tgt = s["target_rgb"]
        if tgt.dim() == 3:
            tgt = tgt.unsqueeze(0)
        stroke = s["stroke"]
        if stroke.dim() == 2:
            stroke = stroke.unsqueeze(0)
        loss, meta = m.omni_loss(out, {
            "need_text": need_text,
            "need_pix": need_pix,
            "answer": [s["answer"]],
            "target_rgb": tgt,
            "stroke": stroke,
            "t": t,
        }, img.device)
        val = float(loss.detach())
        assert np.isfinite(val), (kind, val, meta)
        rec[kind] = {"loss": val, "meta": meta, "stack_id": id(m.mot_stack)}
        assert rec[kind]["stack_id"] == stack_id
    return rec, stack_id


def test_unified_ports_one_stack():
    rec, sid = _run_ports(0)
    assert set(rec) == set(PORTS)
    assert all(r["stack_id"] == sid for r in rec.values())
    for kind, r in rec.items():
        assert np.isfinite(r["loss"]), kind


def test_unified_ports_second_launch():
    rec, sid = _run_ports(1)
    assert set(rec) == set(PORTS)
    assert all(np.isfinite(r["loss"]) for r in rec.values())
    assert all(r["stack_id"] == sid for r in rec.values())
