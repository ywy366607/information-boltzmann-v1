"""Explicit small-bank coverage for every static language/vision boundary.

The real-data bank remains intact. Added QA is human-authored supervision over
visually checked source images, not mislabelled official ShareGPT conversation.
"""
from __future__ import annotations

import numpy as np
import torch

from fine_grain.ocr_1px import make_ocr_1px, render_digit_mask


def extend_unified_bank(samples, resolution, seed=42):
    """Append 6 language controls and 20 native 1px cases without changing inputs."""
    samples = list(samples)
    r = int(resolution)
    if r < 16:
        raise ValueError("native 1px bank requires at least 16px")

    def row(key, task, **kwargs):
        blank = torch.zeros(3, r, r)
        item = dict(id=key, task=task, dataset="explicit-unified-capacity-controls",
                    image=blank, target_rgb=blank.clone(), image_precision=0.0,
                    prompt="", answer="", text_missing=False, need_pix=False,
                    need_text=True, need_seg=False, target_image_precision=0.0,
                    target_text_precision=1.0, target_seg_precision=0.0,
                    target_seg=torch.zeros(r, r, dtype=torch.long),
                    history_images=torch.zeros(2, 3, r, r), history_precision=torch.zeros(2),
                    target_time=0.0, task_id=0 if task in ("t2t", "t2i") else 1,
                    source_files=[])
        item.update(kwargs)
        return item

    for key, question, answer in (
        ("two-plus-two", "What is two plus two?", "Four."),
        ("three-plus-three", "What is three plus three?", "Six."),
    ):
        samples.append(row("text-" + key, "t2t", prompt=question, answer=answer,
                           derived_supervision="authored elementary arithmetic QA"))

    # Cross the questions with both images. Neither question-only nor image-only
    # lookup can satisfy all four labels. Answers were checked against the images.
    sources = ("freedom-t2i-17808", "freedom-t2i-32449")
    questions = ("Is a cat visible?", "Are hot air balloons visible?")
    for i, source_id in enumerate(sources):
        source = next(s for s in samples if s["id"] == source_id)
        for q, question in enumerate(questions):
            yes = (i == 1 and q == 0) or (i == 0 and q == 1)
            samples.append(row(
                f"qa-{i}-{q}", "it2t", prompt=question, answer="Yes." if yes else "No.",
                image=source["target_rgb"].clone(), target_rgb=source["target_rgb"].clone(),
                image_precision=1.0, source_files=[source["target_file"]],
                qa_image_control_id=f"qa-{1-i}-{q}", qa_text_control_id=f"qa-{i}-{1-q}",
                derived_supervision="authored yes/no QA over visually checked ShareGPT target; not official QA",
            ))

    images, labels, masks = make_ocr_1px(np.random.default_rng(seed + 991), np.arange(10),
                                        res=r, hard_frac=1.0, hard_box=14)
    for i in range(10):
        samples.append(row(
            f"ocr1px-{i}", "i2t", family="ocr1px", text_missing=True,
            answer=str(int(labels[i])), image=images[i], target_rgb=images[i].clone(),
            image_precision=1.0, target_seg=masks[i].reshape(r, r).long(),
            derived_supervision="native Bresenham 1px noisy OCR; seed+991; fixed 14px glyph",
        ))
        mask = torch.from_numpy(render_digit_mask(str(i), r, 14, (r-14)//2, (r-14)//2)).long()
        target = torch.zeros(3, r, r)
        target[0] = mask.float()
        samples.append(row(
            f"t2i1px-{i}", "t2i", family="generation1px",
            prompt=f"Draw digit {i} with a thin red stroke at middle center",
            answer="", need_text=False, target_text_precision=0.0,
            target_rgb=target, need_pix=True, target_image_precision=1.0,
            target_seg=mask, need_seg=True, target_seg_precision=1.0,
            derived_supervision="native Bresenham 1px generation; fixed centered 14px glyph",
        ))
    if len({s["id"] for s in samples}) != len(samples):
        raise ValueError("unified sample IDs must be unique; do not append twice")
    return samples
