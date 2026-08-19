"""Synthetic VQA for vision→LLM (LLaVA-style Q/A). No angles.

Train/val use disjoint RNG seeds. Tasks:
  - color: 4-way color of needle square
  - kinks: kink count on red polyline (5..8)
  - ocr: 1px stroke digit 0–9

Format:
  Question: ... Answer: <short>
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from fine_grain.ocr_1px import OCR_DIGITS, make_ocr_1px, make_ocr_string_1px
from fine_grain.tasks import make_kinks, make_needle

COLORS = ("red", "green", "blue", "yellow")
KINK_KS = (5, 6, 7, 8)


def format_vqa(question: str, answer: str) -> str:
    return f"Question: {question} Answer: {answer}"


def format_prompt(question: str) -> str:
    """Prompt for generation (no gold answer)."""
    return f"Question: {question} Answer:"


def _one_color(rng: np.random.Generator, res: int) -> Dict:
    # label 0..3 = SIGNAL color index in make_needle
    sizes = np.array([int(rng.choice([2, 3, 4]))])
    # make_needle samples label randomly; re-roll until we can read lab
    im, lab, _ = make_needle(rng, sizes, res=res)
    lab_i = int(lab[0].item())
    ans = COLORS[lab_i % 4]
    q = "What color is the small square?"
    return {
        "image": im,
        "question": q,
        "answer": ans,
        "text": format_vqa(q, ans),
        "prompt": format_prompt(q),
        "probe": "color",
        "label": lab_i,
    }


def _one_kinks(rng: np.random.Generator, res: int) -> Dict:
    k = int(rng.choice(list(KINK_KS)))
    im, lab, _ = make_kinks(
        rng, np.array([k]), res=res, hard_frac=0.35, hard_tile=16,
    )
    # lab = k - 5 in make_kinks
    ans = str(k)
    q = "How many corners does the red polyline have?"
    return {
        "image": im,
        "question": q,
        "answer": ans,
        "text": format_vqa(q, ans),
        "prompt": format_prompt(q),
        "probe": "kinks",
        "label": int(lab[0].item()),
    }


def _one_ocr(rng: np.random.Generator, res: int) -> Dict:
    """Single 1px-stroke digit; answer is '0'..'9'."""
    lab = int(rng.integers(0, len(OCR_DIGITS)))
    img, labs, _ = make_ocr_1px(rng, np.array([lab]), res=res)
    ans = OCR_DIGITS[int(labs[0].item())]
    q = "What digit is drawn with the thin stroke?"
    return {
        "image": img,
        "question": q,
        "answer": ans,
        "text": format_vqa(q, ans),
        "prompt": format_prompt(q),
        "probe": "ocr",
        "label": int(labs[0].item()),
    }


_BUILDERS = {
    "color": _one_color,
    "kinks": _one_kinks,
    "ocr": _one_ocr,
}


def make_vqa_batch(
    rng: np.random.Generator,
    batch: int,
    res: int = 32,
    mix: Sequence[str] = ("color", "kinks"),
) -> Dict[str, object]:
    """On-the-fly VQA batch. mix ⊆ {color, kinks, ocr}. No angles."""
    kinds = [k for k in mix if k in _BUILDERS]
    if not kinds:
        kinds = ["color", "kinks"]
    samples = []
    for _ in range(batch):
        kind = kinds[int(rng.integers(0, len(kinds)))]
        samples.append(_BUILDERS[kind](rng, res))
    img = torch.cat([s["image"] for s in samples], 0)
    return {
        "image": img,
        "text": [s["text"] for s in samples],
        "prompt": [s["prompt"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "question": [s["question"] for s in samples],
        "probe": [s["probe"] for s in samples],
        "tags": [
            {"probe": s["probe"], "label": s["label"], "answer": s["answer"]}
            for s in samples
        ],
    }


def make_ocr_vqa_batch(
    rng: np.random.Generator,
    batch: int,
    res: int = 32,
    **ocr_kw,
) -> Dict[str, object]:
    """On-the-fly pure OCR VQA batch (single digits)."""
    labs = rng.integers(0, len(OCR_DIGITS), batch)
    img, lab, _ = make_ocr_1px(rng, labs, res=res, **ocr_kw)
    answers = [OCR_DIGITS[int(lab[i].item())] for i in range(batch)]
    q = "What digit is drawn with the thin stroke?"
    prompts = [format_prompt(q) for _ in range(batch)]
    texts = [format_vqa(q, a) for a in answers]
    return {
        "image": img,
        "text": texts,
        "prompt": prompts,
        "answer": answers,
        "question": [q] * batch,
        "probe": ["ocr"] * batch,
        "tags": [
            {"probe": "ocr", "label": int(lab[i].item()), "answer": answers[i]}
            for i in range(batch)
        ],
    }


def make_ocr_string_vqa_batch(
    rng: np.random.Generator,
    batch: int,
    res: int = 32,
    *,
    min_len: int = 2,
    max_len: int = 4,
    char_box: int = 10,
    gap: int = 2,
    hard_frac: float = 0.3,
) -> Dict[str, object]:
    """Multi-digit string OCR for VLM (short sequences, not single-class labels)."""
    img, answers, _ = make_ocr_string_1px(
        rng, batch, res=res,
        min_len=min_len, max_len=max_len,
        char_box=char_box, gap=gap, hard_frac=hard_frac,
    )
    q = "What number is written with the thin strokes?"
    prompts = [format_prompt(q) for _ in range(batch)]
    texts = [format_vqa(q, a) for a in answers]
    return {
        "image": img,
        "text": texts,
        "prompt": prompts,
        "answer": answers,
        "question": [q] * batch,
        "probe": ["ocr_str"] * batch,
        "tags": [
            {"probe": "ocr_str", "label": -1, "answer": answers[i], "length": len(answers[i])}
            for i in range(batch)
        ],
    }


def char_error_rate(pred: str, gold: str) -> float:
    """Levenshtein CER = edit_distance / max(1, len(gold))."""
    p = normalize_answer(pred).replace(" ", "")
    g = normalize_answer(gold).replace(" ", "")
    # keep digits only for synthetic numeric OCR
    p = "".join(ch for ch in p if ch.isdigit())
    g = "".join(ch for ch in g if ch.isdigit())
    if not g:
        return 0.0 if not p else 1.0
    # classic DP
    n, m = len(g), len(p)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            if g[i - 1] == p[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m] / n


def exact_string_match(pred: str, gold: str) -> bool:
    p = "".join(ch for ch in normalize_answer(pred) if ch.isdigit())
    g = "".join(ch for ch in normalize_answer(gold) if ch.isdigit())
    return bool(g) and p == g



# Back-compat alias used by older call sites
def make_llava_batch(rng, batch, res=32, mix=("color", "kinks")):
    return make_vqa_batch(rng, batch, res=res, mix=mix)


def tokenize_captions(tokenizer, texts: List[str], max_length: int = 48):
    out = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return out["input_ids"], out["attention_mask"]


def answer_only_labels(
    tokenizer,
    prompts: Sequence[str],
    texts: Sequence[str],
    max_length: int = 48,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize full Q+A; CE labels only on the answer span (after prompt).

    Prompt is ``Question: … Answer:``; full text appends the gold answer.
    Positions belonging to the prompt (and pad) are set to -100.
    """
    ids, mask = tokenize_captions(tokenizer, list(texts), max_length=max_length)
    labels = ids.clone()
    for i, prompt in enumerate(prompts):
        p = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        plen = int(len(p["input_ids"]))
        # Guard: never empty-supervise whole row if plen overshoots
        plen = min(plen, int(mask[i].sum().item()))
        labels[i, :plen] = -100
        labels[i, mask[i] == 0] = -100
        # If nothing left to train on, fall back to last non-pad token
        if (labels[i] != -100).sum() == 0:
            last = int(mask[i].sum().item()) - 1
            if last >= 0:
                labels[i, last] = ids[i, last]
    return ids, mask, labels


def answer_candidates(task: str) -> Tuple[str, ...]:
    if task == "color":
        return COLORS
    if task == "kinks":
        return tuple(str(k) for k in KINK_KS)
    if task == "ocr":
        return OCR_DIGITS
    return COLORS + tuple(str(k) for k in KINK_KS) + OCR_DIGITS


def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    # keep first token-ish answer
    for sep in ("\n", ".", ",", ";"):
        if sep in s:
            s = s.split(sep)[0]
    # strip leading junk like "? answer:"
    if "answer:" in s:
        s = s.split("answer:")[-1].strip()
    return s.strip()


def exact_match(pred: str, gold: str) -> bool:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not p:
        return False
    # allow gold as substring of first few chars (e.g. "red" in "red square")
    return p == g or p.startswith(g) or g in p.split()
