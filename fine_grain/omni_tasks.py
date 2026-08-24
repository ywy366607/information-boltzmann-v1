"""Six I/O ports on the same DualStream field loop.

  t2t   text → text          (language-only)
  i2t   image → text         (caption / 图生文)
  it2t  image+text → text    (VLM OCR / 图文生文)
  recon image → image        (rebuild / 图生图)
  i2i   image+text → image   (instructed edit / 图文生图)
  t2i   text → image         (text-to-image / 文生图)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from fine_grain.ocr_1px import OCR_DIGITS, make_ocr_1px, render_digit_mask
from fine_grain.tasks import SIGNAL, _canvas
from fine_grain.vlm_data import COLORS, KINK_KS, format_prompt

TASKS = ("t2t", "i2t", "it2t", "recon", "i2i", "t2i")

GRID_PLACES = (
    "top_left", "top_center", "top_right",
    "middle_left", "middle_center", "middle_right",
    "bottom_left", "bottom_center", "bottom_right",
)

_GRID_WORDS = {
    place: place.replace("_", " ")
    for place in GRID_PLACES
}

_COLOR_IDX = {"red": 0, "green": 1, "blue": 2, "yellow": 3}

# L2-unit inks. SIGNAL yellow is [1,1,0] (2× energy vs R/G/B) so MSE
# and SliceRead amplitude preferred it. Recognition tasks keep SIGNAL.
SIGNAL_UNIT = SIGNAL / np.clip(
    np.linalg.norm(SIGNAL, axis=1, keepdims=True), 1e-8, None,
)


def equal_energy_ink(color: str) -> np.ndarray:
    """Unit-L2 palette color. Yellow no longer twice as loud as green."""
    return SIGNAL_UNIT[_COLOR_IDX[str(color)]].astype(np.float32).copy()


def _stroke_mask(mask, res: int) -> torch.Tensor:
    m = mask.float()
    if m.dim() == 2:
        m = m.view(-1, res, res)
    return m


def thicken_stroke(stroke: torch.Tensor, px: int) -> torch.Tensor:
    """Dilate a 1px mask to an odd-width brush. px=1 is identity."""
    px = max(1, int(px))
    if px <= 1:
        return stroke
    m = stroke.float()
    while m.dim() < 4:
        m = m.unsqueeze(0)
    k = px if px % 2 == 1 else px + 1
    out = F.max_pool2d(m, kernel_size=k, stride=1, padding=k // 2)
    while out.dim() > stroke.dim():
        out = out.squeeze(0)
    return (out > 0.5).to(dtype=stroke.dtype)


def _paint(img: torch.Tensor, mask: torch.Tensor, rgb: np.ndarray) -> torch.Tensor:
    m = mask.to(img.dtype).unsqueeze(1)
    col = torch.tensor(rgb, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    return img * (1.0 - m) + col * m


def grid_digit_mask(digit: str, res: int, place: str) -> torch.Tensor:
    """Render a deterministic small digit at one named full-resolution address."""
    place = str(place).lower()
    if place not in GRID_PLACES:
        raise ValueError(f"unknown grid placement {place!r}")
    row, col = place.split("_")
    box = max(4, min(16, (int(res) + 2) // 3, int(res) - 2))
    starts = {
        "top": 1,
        "middle": (int(res) - box) // 2,
        "bottom": int(res) - box - 1,
        "left": 1,
        "center": (int(res) - box) // 2,
        "right": int(res) - box - 1,
    }
    pm = render_digit_mask(
        str(digit), int(res), box, starts[row], starts[col], jitter=0.0,
    )
    return torch.from_numpy(pm.astype(np.float32)).view(1, int(res), int(res))


def one_sample(
    rng: np.random.Generator, res: int, kind: str,
    t2i_canvas: str = "paper", t2i_stroke_px: int = 1,
    t2i_place: str = "random",
    t2i_digit=None,
    t2i_color: str | None = None,
) -> Dict:
    color = str(t2i_color) if t2i_color else str(rng.choice(list(COLORS)))
    ink = equal_energy_ink(color)
    d = int(t2i_digit) if t2i_digit is not None else int(rng.integers(0, 10))
    img, lab, msk = make_ocr_1px(rng, np.array([d]), res=res, ink_color=color)
    stroke = _stroke_mask(msk, res)
    extra = {
        "digit": OCR_DIGITS[d],
        "color": color,
        "t2i_canvas": str(t2i_canvas),
        "placement": str(t2i_place).lower(),
    }

    if kind == "it2t":
        q = "What digit is drawn with the thin stroke?"
        return {
            "kind": kind, "image": img, "target_rgb": img, "stroke": stroke,
            "prompt": format_prompt(q), "answer": OCR_DIGITS[int(lab[0].item())],
            "need_text": True, "need_pix": False, **extra,
        }
    if kind == "i2t":
        # image-only recognition: generic prompt, no task wording
        return {
            "kind": kind, "image": img, "target_rgb": img, "stroke": stroke,
            "prompt": format_prompt("What is this"),
            "answer": OCR_DIGITS[int(lab[0].item())],
            "need_text": True, "need_pix": False, **extra,
        }
    if kind == "t2t":
        k = int(rng.choice(list(KINK_KS)))
        q = f"How many corners does a polyline with {k} corners have?"
        return {
            "kind": kind,
            "image": torch.zeros_like(img),
            "target_rgb": torch.zeros_like(img),
            "stroke": torch.zeros_like(stroke),
            "prompt": format_prompt(q), "answer": str(k),
            "need_text": True, "need_pix": False, **extra,
        }
    if kind == "recon":
        return {
            "kind": kind, "image": img, "target_rgb": img, "stroke": stroke,
            "prompt": "Reconstruct the image", "answer": OCR_DIGITS[d],
            "need_text": False, "need_pix": True, **extra,
        }
    if kind == "t2i":
        # paper = product edit: write the digit onto THIS haystack (ODE starts
        # from the paper). black = from-noise generation: prompt names a blank
        # canvas so PSNR vs a random paper is no longer the score.
        canvas = str(t2i_canvas).lower()
        if canvas == "black":
            paper = torch.zeros(1, 3, res, res, dtype=img.dtype)
            # existing vocab only: do not grow embed (old ckpts stay loadable)
            prompt = f"Draw digit {d} with a thin {color} stroke blank image"
        else:
            paper = torch.from_numpy(np.clip(_canvas(rng, 1, res, n_blobs=3), 0, 1))
            # product edit: image is the paper. Same wording as published t2i.
            prompt = f"Draw digit {d} with a thin {color} stroke"
        place = str(t2i_place).lower()
        if place == "grid":
            place = str(rng.choice(GRID_PLACES))
        if place in GRID_PLACES:
            stroke = grid_digit_mask(str(d), res, place)
            prompt = f"{prompt} at {_GRID_WORDS[place]}"
        elif place == "center":
            box = min(16, res - 2)
            y0 = x0 = (res - box) // 2
            pm = render_digit_mask(str(d), res, box, y0, x0, jitter=0.0)
            stroke = torch.from_numpy(pm.astype(np.float32)).view(1, res, res)
        elif place != "random":
            choices = ", ".join(("random", "center", "grid", *GRID_PLACES))
            raise ValueError(f"unknown t2i_place={place!r}; choose one of {choices}")
        extra["placement"] = place
        if int(t2i_stroke_px) > 1:
            stroke = thicken_stroke(stroke, t2i_stroke_px)
        tgt = _paint(paper, stroke, ink)
        return {
            "kind": kind, "image": paper, "target_rgb": tgt, "stroke": stroke,
            "prompt": prompt,
            "answer": OCR_DIGITS[d],
            "need_text": False, "need_pix": True, **extra,
        }
    if kind == "i2i":
        if rng.random() < 0.5:
            new_c = str(rng.choice(list(COLORS)))
            new_ink = equal_energy_ink(new_c)
            src = _paint(img, stroke, np.array([0.85, 0.85, 0.85], dtype=np.float32))
            tgt = _paint(img, stroke, new_ink)
            prompt = f"Paint the stroke {new_c}"
        else:
            src = (img + torch.from_numpy(rng.normal(0, 0.25, img.shape).astype(np.float32))).clamp(0, 1)
            tgt = img
            prompt = "Restore the thin stroke"
        return {
            "kind": kind, "image": src, "target_rgb": tgt, "stroke": stroke,
            "prompt": prompt, "answer": OCR_DIGITS[d],
            "need_text": False, "need_pix": True,
        }
    raise ValueError(kind)


def apply_t2i_gray_hint(
    image: torch.Tensor,
    stroke: torch.Tensor,
    kinds,
    frac: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Train-only: with probability ``frac``, put a gray stroke on t2i paper.

    Eval stays hint-free. Same geometry as the target digit so the model
    first learns to colour an existing line, then to invent one.
    """
    if frac <= 0.0:
        return image
    out = image.clone()
    gray = torch.tensor([0.75, 0.75, 0.75], dtype=out.dtype, device=out.device).view(1, 3, 1, 1)
    st = stroke.to(device=out.device, dtype=out.dtype)
    if st.dim() == 3:
        st = st.unsqueeze(1)
    for i, k in enumerate(kinds):
        if k == "t2i" and float(rng.random()) < float(frac):
            m = st[i : i + 1]
            out[i : i + 1] = out[i : i + 1] * (1.0 - m) + gray * m
    return out


def make_omni_batch(
    rng: np.random.Generator, batch: int, res: int, mix=None,
    t2i_canvas: str = "paper", t2i_stroke_px: int = 1,
    t2i_place: str = "random",
    t2i_digit=None,
    t2i_color: str | None = None,
) -> Dict:
    mix = list(mix or TASKS)
    kinds = [str(rng.choice(mix)) for _ in range(batch)]
    samples = [
        one_sample(
            rng, res, k, t2i_canvas=t2i_canvas,
            t2i_stroke_px=t2i_stroke_px, t2i_place=t2i_place,
            t2i_digit=t2i_digit, t2i_color=t2i_color,
        )
        for k in kinds
    ]
    return {
        "image": torch.cat([s["image"] for s in samples], 0),
        "target_rgb": torch.cat([s["target_rgb"] for s in samples], 0),
        "stroke": torch.cat([s["stroke"] for s in samples], 0),
        "prompt": [s["prompt"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "kind": kinds,
        "need_text": [s["need_text"] for s in samples],
        "need_pix": [s["need_pix"] for s in samples],
        "digit": [s.get("digit", "") for s in samples],
        "color": [s.get("color", "") for s in samples],
        "placement": [s.get("placement", "") for s in samples],
    }
