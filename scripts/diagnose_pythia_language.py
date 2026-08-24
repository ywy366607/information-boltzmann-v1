#!/usr/bin/env python3
"""Why did Pythia language fail to select digit identity?

Compares the published toy-embedding champion against the same visual weights
driven by frozen Pythia embeddings + a fresh text_in.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.omni_model import DualStreamOmni
from fine_grain.pythia_bridge import (
    GENERATION_CHAMPION_PATH,
    generation_champion_kwargs,
)


PROMPT = "Draw digit {d} with a thin {c} stroke blank image at {place}"
PAIRS = {
    "digit": (
        PROMPT.format(d=7, c="green", place="top left"),
        PROMPT.format(d=1, c="green", place="top left"),
    ),
    "color": (
        PROMPT.format(d=7, c="green", place="top left"),
        PROMPT.format(d=7, c="red", place="top left"),
    ),
    "place": (
        PROMPT.format(d=7, c="green", place="top left"),
        PROMPT.format(d=7, c="green", place="bottom right"),
    ),
}


def rms(a, b):
    return float((a - b).float().pow(2).mean().sqrt())


def cosine(a, b):
    x = F.normalize(a.float().reshape(1, -1), dim=-1)
    y = F.normalize(b.float().reshape(1, -1), dim=-1)
    return float((x * y).sum())


def tokens_toy(model, prompt):
    ids, mask = model.tokenize([prompt], torch.device("cpu"))
    words = prompt.replace("?", " ?").split()
    inv = {i: w for w, i in model.vocab.items()}
    return {
        "n": int(mask.sum()),
        "words": words,
        "ids": [int(x) for x in ids[0, : int(mask.sum())]],
        "decoded": [inv.get(int(i), f"#{int(i)}") for i in ids[0, : int(mask.sum())]],
        "unk": [w for w in words if w not in model.vocab],
    }


def tokens_pythia(model, prompt):
    enc = model.lm_tok(prompt, return_tensors="pt")
    ids = enc["input_ids"][0]
    return {
        "n": int(ids.numel()),
        "ids": [int(x) for x in ids],
        "decoded": model.lm_tok.convert_ids_to_tokens(ids.tolist()),
        "text": model.lm_tok.decode(ids),
    }


def text_token_mass(model):
    rows = []
    for i, layer in enumerate(model.mot_stack.layers):
        m = getattr(layer.mot, "last_text_token_mass", None)
        if m is None:
            continue
        rows.append([float(x) for x in m[0].detach().cpu()])
    if not rows:
        return None
    mean = torch.tensor(rows).mean(dim=0)
    return {
        "per_layer": rows,
        "mean": [float(x) for x in mean],
        "entropy": float((-(mean.clamp_min(1e-8) * mean.clamp_min(1e-8).log()).sum())
                         / max(float(mean.numel() ** 0.0), 1.0)),
    }


def run_pair(model, device, p0, p1):
    model.eval()
    img = torch.zeros(2, 3, model.res, model.res, device=device)
    with torch.no_grad():
        out = model(img, [p0, p1])
        H = model.mot_stack._last_H
        He = model.mot_stack._last_text_evidence
        Hin = model.mot_stack.text_in(He)
    mass = getattr(model.mot_stack.layers[0].mot, "last_text_token_mass", None)
    return {
        "rgb_rms": rms(out["rgb"][0], out["rgb"][1]),
        "rgb_cosine": cosine(out["rgb"][0], out["rgb"][1]),
        "H_rms": rms(H[0], H[1]),
        "H_cosine": cosine(H[0], H[1]),
        "H_in_rms": rms(Hin[0], Hin[1]),
        "H_in_cosine": cosine(Hin[0], Hin[1]),
        "evidence_rms": rms(He[0], He[1]),
        "evidence_cosine": cosine(He[0], He[1]),
        "token_mass_0": None if mass is None else [float(x) for x in mass[0].cpu()],
        "token_mass_1": None if mass is None else [float(x) for x in mass[1].cpu()],
    }


def build_toy(device):
    model = DualStreamOmni(**generation_champion_kwargs()).to(device).eval()
    state = torch.load(GENERATION_CHAMPION_PATH, map_location=device)
    missing = model.load_state_dict(state, strict=False)
    if missing.missing_keys:
        print("toy missing", missing.missing_keys, flush=True)
    return model


def build_pythia(device):
    model = DualStreamOmni(
        **generation_champion_kwargs(language="pythia", lm_device=str(device))
    ).to(device).eval()
    if model.lm is not None:
        model.lm.to(device)
    report = model.load_visual_champion(GENERATION_CHAMPION_PATH)
    return model, report


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    toy = build_toy(device)
    pythia, report = build_pythia(device)
    p7, p1 = PAIRS["digit"]
    rec = {
        "device": str(device),
        "champion": str(GENERATION_CHAMPION_PATH),
        "load": {k: report[k] for k in ("loaded", "n_skipped") if k in report},
        "skipped": report.get("skipped", []),
        "toy_tokens": tokens_toy(toy, p7),
        "pythia_tokens": tokens_pythia(pythia, p7),
        "pythia_tokens_digit1": tokens_pythia(pythia, p1),
        "pairs": {},
    }
    print("TOY:", rec["toy_tokens"])
    print("PYTHIA 7:", rec["pythia_tokens"])
    print("PYTHIA 1:", rec["pythia_tokens_digit1"])
    print("skipped", rec["skipped"])
    for name, (a, b) in PAIRS.items():
        rec["pairs"][name] = {
            "toy": run_pair(toy, device, a, b),
            "pythia": run_pair(pythia, device, a, b),
            "prompts": [a, b],
        }
        t, p = rec["pairs"][name]["toy"], rec["pairs"][name]["pythia"]
        print(
            f"{name:5s}  toy rgb={t['rgb_rms']:.4f} H={t['H_rms']:.4f} Hin={t['H_in_rms']:.4f}"
            f"  | pythia rgb={p['rgb_rms']:.4f} H={p['H_rms']:.4f} Hin={p['H_in_rms']:.4f}"
            f"  Hin_cos={p['H_in_cosine']:.4f} rgb_cos={p['rgb_cosine']:.4f}",
            flush=True,
        )

    # Which Pythia tokens actually differ for digit 7 vs 1?
    t7 = rec["pythia_tokens"]
    t1 = rec["pythia_tokens_digit1"]
    rec["pythia_digit_token_delta"] = [
        {"i": i, "tok7": a, "tok1": b}
        for i, (a, b) in enumerate(zip(t7["decoded"], t1["decoded"]))
        if a != b
    ]
    print("digit token delta", rec["pythia_digit_token_delta"])

    out = ROOT / "results" / "published" / "_pythia_language_fail.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
