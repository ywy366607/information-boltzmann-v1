"""Optional diagnostics for this 1px microbench.

JiT / FM papers do not report IoU. FID/CLIP need a real image distribution.
Here: paired PSNR is the product score only when the ODE starts from the
given paper. color_acc is a cheap 'did it paint the named ink' check.
digit_iou / digit_top1 slide our 1px templates over the ink mask — a
homemade recognizer, not a generation metric. Do not optimize them.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from fine_grain.ocr_1px import OCR_DIGITS, render_digit_mask
from fine_grain.tasks import SIGNAL

PALETTE_ORDER = ("red", "green", "blue", "yellow")
PALETTE = {name: SIGNAL[i] for i, name in enumerate(PALETTE_ORDER)}
_TEMPLATE_CACHE: Dict[Tuple[int, str, int], torch.Tensor] = {}


def parse_draw_prompt(prompt: str) -> Tuple[str, str]:
    """'Draw digit 6 with a thin yellow stroke' → ('6', 'yellow')."""
    words = prompt.replace(".", " ").split()
    digit = "0"
    color = "red"
    for i, w in enumerate(words):
        if w == "digit" and i + 1 < len(words) and words[i + 1] in OCR_DIGITS:
            digit = words[i + 1]
        if w in PALETTE:
            color = w
    return digit, color


def _palette_tensor(device, dtype) -> torch.Tensor:
    arr = torch.as_tensor(SIGNAL, device=device, dtype=dtype)
    return arr


def nearest_palette(pred: torch.Tensor) -> torch.Tensor:
    """pred [B,3,H,W] → palette index [B,H,W]."""
    cols = _palette_tensor(pred.device, pred.dtype).view(1, 4, 3, 1, 1)
    dist = (pred.unsqueeze(1) - cols).pow(2).sum(dim=2)
    return dist.argmin(dim=1)


def chroma(pred: torch.Tensor) -> torch.Tensor:
    return pred.max(dim=1).values - pred.min(dim=1).values


def free_ink_mask(
    pred: torch.Tensor,
    color: str,
    chroma_min: float = 0.20,
    lum_min: float = 0.15,
) -> torch.Tensor:
    """Pixels that look like the requested ink, anywhere on the canvas."""
    idx = PALETTE_ORDER.index(color)
    ink = (nearest_palette(pred) == idx)
    bright = (chroma(pred) >= chroma_min) & (pred.max(dim=1).values >= lum_min)
    return ink & bright


def free_color_acc(pred: torch.Tensor, color: str) -> float:
    """Of chromatic pixels, fraction whose nearest palette color matches."""
    chroma_px = (chroma(pred) >= 0.20) & (pred.max(dim=1).values >= 0.15)
    if float(chroma_px.sum()) < 1:
        return 0.0
    hit = nearest_palette(pred) == PALETTE_ORDER.index(color)
    return float((hit & chroma_px).sum() / chroma_px.sum().clamp_min(1))


def background_flood_rate(
    pred: torch.Tensor,
    target: torch.Tensor,
    stroke: torch.Tensor,
    color_distance: float = 0.35,
) -> float:
    """Fraction of ground that is visibly painted like the observed ink.

    Color distance alone is insufficient: black is only 1/3 mean-L1 away
    from pure red/green/blue and the historical 0.35 threshold therefore
    labeled a perfect black background as 100% flood.  Requiring visible
    chroma and luminance makes the metric agree with ``free_ink_mask``.
    """
    mask = stroke.unsqueeze(1) if stroke.dim() == 3 else stroke
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    bg = 1.0 - mask
    count = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    ink = (target * mask).sum(dim=(2, 3), keepdim=True) / count
    close = (pred - ink).abs().mean(dim=1, keepdim=True) < float(color_distance)
    chromatic = (
        pred.max(dim=1, keepdim=True).values
        - pred.min(dim=1, keepdim=True).values
    ) >= 0.20
    visible = pred.max(dim=1, keepdim=True).values >= 0.15
    hit = close & chromatic & visible
    return float((hit.to(pred.dtype) * bg).sum() / bg.sum().clamp_min(1.0))


def paired_ink_iou(pred: torch.Tensor, stroke: torch.Tensor, color: str) -> float:
    """IoU at the requested point addresses; unlike digit_iou, no shifting."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    ink = free_ink_mask(pred, color)
    target = stroke.to(device=pred.device)
    if target.dim() == 2:
        target = target.unsqueeze(0)
    if target.dim() == 4:
        target = target[:, 0]
    target = target > 0.5
    inter = (ink & target).flatten(1).sum(dim=1).float()
    union = (ink | target).flatten(1).sum(dim=1).float()
    return float((inter / union.clamp_min(1.0)).mean())


def ink_centroid_error(pred: torch.Tensor, stroke: torch.Tensor, color: str) -> float:
    """Requested-vs-generated ink centroid distance, normalized by image diagonal."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    ink = free_ink_mask(pred, color)
    target = stroke.to(device=pred.device)
    if target.dim() == 2:
        target = target.unsqueeze(0)
    if target.dim() == 4:
        target = target[:, 0]
    target = target > 0.5
    h, w = ink.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=pred.device, dtype=torch.float32),
        torch.arange(w, device=pred.device, dtype=torch.float32),
        indexing="ij",
    )
    xy = torch.stack((yy, xx), dim=-1).view(1, h, w, 2)

    def centroid(mask: torch.Tensor):
        mass = mask.flatten(1).sum(dim=1).float()
        point = (mask.float().unsqueeze(-1) * xy).sum(dim=(1, 2))
        return point / mass.clamp_min(1.0).unsqueeze(-1), mass

    pred_c, pred_mass = centroid(ink)
    target_c, target_mass = centroid(target)
    diag = max(((h - 1) ** 2 + (w - 1) ** 2) ** 0.5, 1.0)
    error = (pred_c - target_c).norm(dim=-1) / diag
    missing = (pred_mass < 1) | (target_mass < 1)
    error = torch.where(missing, torch.ones_like(error), error)
    return float(error.mean())


def _digit_template(digit: str, res: int, box: int) -> torch.Tensor:
    key = (res, str(digit), int(box))
    cached = _TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    y0 = max(0, (res - box) // 2)
    x0 = max(0, (res - box) // 2)
    m = render_digit_mask(str(digit), res, box, y0, x0)
    ys, xs = m.nonzero()
    if len(ys) == 0:
        tmpl = torch.zeros(1, 1, 3, 3)
    else:
        crop = m[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
        tmpl = torch.from_numpy(crop.astype("float32"))[None, None]
    _TEMPLATE_CACHE[key] = tmpl
    return tmpl


def _max_shift_iou(pred_m: torch.Tensor, kernel: torch.Tensor) -> float:
    """Global IoU of a shifted template vs the full ink mask.

    Window-local IoU lets a thin '1' sit on any stem and score ~1.
    """
    if float(pred_m.sum()) < 1 or float(kernel.sum()) < 1:
        return 0.0
    h, w = int(kernel.shape[-2]), int(kernel.shape[-1])
    if pred_m.shape[-2] < h or pred_m.shape[-1] < w:
        return 0.0
    p = pred_m.float().view(1, 1, *pred_m.shape[-2:])
    k = kernel.to(device=p.device, dtype=p.dtype)
    overlap = F.conv2d(p, k)
    pred_total = p.sum()
    tsum = k.sum()
    iou = overlap / (pred_total + tsum - overlap).clamp_min(1e-6)
    return float(iou.max())


def digit_shift_scores(
    pred: torch.Tensor,
    digit: str,
    color: str,
    boxes: Tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22),
) -> Dict[str, float]:
    """Translation-invariant digit match. pred is [1,3,H,W] or [3,H,W]."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    ink = free_ink_mask(pred, color)[0].float()
    res = int(pred.shape[-1])
    per: List[Tuple[str, float]] = []
    for d in OCR_DIGITS:
        best = 0.0
        for box in boxes:
            box = min(int(box), res)
            best = max(best, _max_shift_iou(ink, _digit_template(d, res, box)))
        per.append((d, best))
    by_d = {d: s for d, s in per}
    best_d = max(per, key=lambda kv: kv[1])[0]
    return {
        "digit_iou": float(by_d.get(str(digit), 0.0)),
        "digit_top1": float(best_d == str(digit) and by_d[best_d] > 0.05),
        "digit_best": best_d,
        "digit_best_iou": float(by_d[best_d]),
        "ink_frac": float(ink.mean()),
    }


def gen_free_scores(pred: torch.Tensor, digit: str, color: str) -> Dict[str, float]:
    rec = digit_shift_scores(pred, digit, color)
    rec["color_acc"] = free_color_acc(pred, color)
    return rec
