"""1px stroke synthetic OCR (digits 0–9).

Each digit is a set of unit-square polylines rendered with Bresenham 1px ink
on a noisy canvas — a fine-grain proxy for thin-character OCR (not real fonts).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from fine_grain.tasks import SIGNAL, _bresenham, _canvas, _done

# Closed alphabet for constrained LLM readout
OCR_DIGITS = tuple(str(d) for d in range(10))
N_OCR = len(OCR_DIGITS)

# Unit-square stroke templates: list of polylines, each polyline = list of (x,y) in [0,1]
# y increases downward (image coords). Designed as 1px-friendly stick digits.
_DIGIT_STROKES: Dict[str, List[List[Tuple[float, float]]]] = {
    "0": [
        [(0.2, 0.1), (0.8, 0.1), (0.8, 0.9), (0.2, 0.9), (0.2, 0.1)],
    ],
    "1": [
        [(0.5, 0.1), (0.5, 0.9)],
        [(0.35, 0.25), (0.5, 0.1)],
    ],
    "2": [
        [(0.2, 0.25), (0.2, 0.15), (0.8, 0.15), (0.8, 0.45), (0.2, 0.45), (0.2, 0.9), (0.8, 0.9)],
    ],
    "3": [
        [(0.2, 0.15), (0.8, 0.15), (0.8, 0.5), (0.35, 0.5)],
        [(0.8, 0.5), (0.8, 0.9), (0.2, 0.9)],
    ],
    "4": [
        [(0.25, 0.1), (0.25, 0.55), (0.85, 0.55)],
        [(0.7, 0.1), (0.7, 0.9)],
    ],
    "5": [
        [(0.8, 0.1), (0.2, 0.1), (0.2, 0.5), (0.75, 0.5), (0.75, 0.9), (0.2, 0.9)],
    ],
    "6": [
        [(0.75, 0.15), (0.25, 0.15), (0.25, 0.9), (0.75, 0.9), (0.75, 0.5), (0.25, 0.5)],
    ],
    "7": [
        [(0.2, 0.15), (0.8, 0.15), (0.35, 0.9)],
    ],
    "8": [
        [(0.25, 0.5), (0.25, 0.15), (0.75, 0.15), (0.75, 0.5), (0.25, 0.5),
         (0.25, 0.9), (0.75, 0.9), (0.75, 0.5)],
    ],
    "9": [
        [(0.25, 0.85), (0.75, 0.85), (0.75, 0.15), (0.25, 0.15), (0.25, 0.5), (0.75, 0.5)],
    ],
}


def _draw_polyline_1px(msk: np.ndarray, pts: Sequence[Tuple[int, int]], R: int) -> None:
    for i in range(len(pts) - 1):
        y0, x0 = pts[i]
        y1, x1 = pts[i + 1]
        for y, x in _bresenham(y0, x0, y1, x1):
            if 0 <= y < R and 0 <= x < R:
                msk[y, x] = True


def render_digit_mask(
    digit: str,
    R: int,
    box: int,
    y0: int,
    x0: int,
    *,
    jitter: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Rasterize one digit as 1px strokes into a boolean [R,R] mask."""
    digit = str(digit)
    if digit not in _DIGIT_STROKES:
        raise ValueError(f"unknown digit {digit!r}")
    msk = np.zeros((R, R), dtype=bool)
    strokes = _DIGIT_STROKES[digit]
    for poly in strokes:
        pts = []
        for ux, uy in poly:
            if jitter > 0 and rng is not None:
                ux = float(np.clip(ux + rng.normal(0, jitter), 0.0, 1.0))
                uy = float(np.clip(uy + rng.normal(0, jitter), 0.0, 1.0))
            x = int(round(x0 + ux * (box - 1)))
            y = int(round(y0 + uy * (box - 1)))
            pts.append((y, x))
        if len(pts) >= 2:
            _draw_polyline_1px(msk, pts, R)
    return msk


def make_ocr_1px(
    rng: np.random.Generator,
    labels: Optional[Union[np.ndarray, Sequence[int]]] = None,
    res: int = 32,
    *,
    box_min: int = 12,
    box_max: int = 22,
    hard_frac: float = 0.35,
    hard_box: int = 14,
    ink_color: str = "red",
    n: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch of single-digit 1px OCR images.

    Args:
        labels: class indices 0..9; if None, sample ``n`` uniform labels.
        box_min/max: digit bounding box size in pixels.
        hard_frac: probability of using smaller hard_box (patch-hostile).
        ink_color: red|green|blue|yellow (SIGNAL palette).

    Returns:
        img [n,3,R,R], lab [n] int64 (0..9), mask [n,R*R] float/bool via _done
    """
    if labels is None:
        assert n is not None
        labels = rng.integers(0, N_OCR, int(n))
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = len(labels)
    R = int(res)
    color_idx = {"red": 0, "green": 1, "blue": 2, "yellow": 3}.get(ink_color, 0)
    ink = SIGNAL[color_idx]
    img = _canvas(rng, n, R, n_blobs=4 if R <= 32 else 6)
    msk = np.zeros((n, R, R), dtype=bool)
    for i in range(n):
        d = str(int(labels[i]) % N_OCR)
        use_hard = float(rng.random()) < float(hard_frac)
        box = int(hard_box) if use_hard else int(rng.integers(box_min, box_max + 1))
        box = max(8, min(box, R - 2))
        y0 = int(rng.integers(0, R - box + 1))
        x0 = int(rng.integers(0, R - box + 1))
        pm = render_digit_mask(d, R, box, y0, x0, jitter=0.02, rng=rng)
        # ensure at least a few ink pixels (degenerate guard)
        if pm.sum() < 3:
            pm = render_digit_mask(d, R, box, y0, x0, jitter=0.0, rng=None)
        img[i] = np.clip(img[i], 0, 1)
        for c in range(3):
            img[i, c][pm] = ink[c]
        msk[i] = pm
    return _done(img, labels, msk)


def make_ocr_1px_batch_dicts(
    rng: np.random.Generator,
    batch: int,
    res: int = 32,
    **kwargs,
) -> List[dict]:
    """Per-sample dicts with image tensor [1,3,R,R] and digit label."""
    labs = rng.integers(0, N_OCR, batch)
    img, lab, msk = make_ocr_1px(rng, labs, res=res, **kwargs)
    out = []
    for i in range(batch):
        out.append({
            "image": img[i : i + 1],
            "label": int(lab[i].item()),
            "digit": OCR_DIGITS[int(lab[i].item())],
            "mask": msk[i : i + 1],
        })
    return out


def make_ocr_string_1px(
    rng: np.random.Generator,
    batch: int = 1,
    res: int = 32,
    *,
    min_len: int = 2,
    max_len: int = 4,
    char_box: int = 10,
    gap: int = 2,
    hard_frac: float = 0.3,
    ink_color: str = "red",
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """Batch of multi-digit 1px strings (true short OCR sequences).

    Digits are laid out left-to-right as 1px stick figures. Length in
    ``[min_len, max_len]``. Returns images, gold strings, ink masks.
    """
    R = int(res)
    color_idx = {"red": 0, "green": 1, "blue": 2, "yellow": 3}.get(ink_color, 0)
    ink = SIGNAL[color_idx]
    imgs = []
    masks = []
    texts: List[str] = []
    for _ in range(batch):
        n_char = int(rng.integers(min_len, max_len + 1))
        digits = [str(int(rng.integers(0, N_OCR))) for _ in range(n_char)]
        text = "".join(digits)
        # hard: smaller glyphs
        box = max(7, char_box - 2) if float(rng.random()) < hard_frac else char_box
        box = min(box, R - 2)
        total_w = n_char * box + max(0, n_char - 1) * gap
        if total_w >= R - 1:
            # shrink box to fit
            box = max(6, (R - 2 - (n_char - 1) * gap) // n_char)
            total_w = n_char * box + max(0, n_char - 1) * gap
        x0 = int(rng.integers(0, max(1, R - total_w)))
        y0 = int(rng.integers(0, max(1, R - box)))
        canvas = _canvas(rng, 1, R, n_blobs=3 if R <= 32 else 5)[0]
        msk = np.zeros((R, R), dtype=bool)
        x = x0
        for d in digits:
            pm = render_digit_mask(d, R, box, y0, x, jitter=0.015, rng=rng)
            if pm.sum() < 3:
                pm = render_digit_mask(d, R, box, y0, x, jitter=0.0, rng=None)
            msk |= pm
            x += box + gap
        canvas = np.clip(canvas, 0, 1)
        for c in range(3):
            canvas[c][msk] = ink[c]
        imgs.append(canvas)
        masks.append(msk)
        texts.append(text)
    img_t = torch.from_numpy(np.stack(imgs, 0).astype(np.float32))
    msk_t = torch.from_numpy(np.stack(masks, 0).reshape(batch, -1))
    return img_t, texts, msk_t
