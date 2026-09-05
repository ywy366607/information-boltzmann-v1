"""Small real-image bank for the shared native-resolution capability graph."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from fine_grain.sharegpt4o_data import (
    collate_real_multimodal, letterbox_tensor, materialize_record,
)

TASK_IDS = {"t2i": 0, "reconstruction": 1, "i2t": 1,
            "segmentation": 1, "it2i": 2, "future": 3}


def letterbox_foreground(path, resolution):
    """Use actual annotation IDs, never infer a mask from RGB brightness."""
    with Image.open(path) as source:
        mask = Image.fromarray((np.asarray(source) > 0).astype(np.uint8))
        scale = min(resolution / mask.width, resolution / mask.height)
        size = (max(1, round(mask.width * scale)), max(1, round(mask.height * scale)))
        mask = mask.resize(size, Image.Resampling.NEAREST)
        canvas = Image.new("L", (resolution, resolution), 0)
        canvas.paste(mask, ((resolution - size[0]) // 2, (resolution - size[1]) // 2))
        return torch.from_numpy(np.asarray(canvas).copy()).long()


def build_real_bank(sharegpt_manifest, davis_manifest, resolution=256):
    """Fixed bank with verifiable source files; no masks/video invented from captions."""
    manifest = json.loads(Path(sharegpt_manifest).read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in manifest["records"]}
    samples = []
    for key in ("freedom-t2i-17808", "freedom-t2i-32449"):
        row = materialize_record(rows[key], resolution)
        samples.append(row)
        # Exact identity reconstruction of the same real target; no caption input.
        samples.append({**row, "id": key + "-reconstruct", "task": "reconstruction",
                        "image": row["target_rgb"].clone(), "image_precision": 1.0,
                        "text_missing": True, "prompt": "", "need_text": False,
                        "source_files": [row["target_file"]]})
    for key in ("freedom-it2i-34244", "freedom-it2i-24875"):
        row = materialize_record(rows[key], resolution)
        samples.append(row)
        # Same image + same edit TToken, different instruction/target. This
        # prevents fixed source -> edited-image memorization from passing alone.
        samples.append({**row, "id": key + "-identity", "prompt": "Keep the image unchanged.",
                        "target_rgb": row["image"].clone(), "target_file": row["source_files"][0],
                        "derived_supervision": "identity edit of original source"})
    for key in ("opengv-1", "opengv-4"):
        row = materialize_record(rows[key], resolution)
        row["answer"] = re.split(r"(?<=[.!?])\s+", row["answer"].strip())[0]
        row["answer_scope"] = "first complete sentence of original dataset caption"
        row["need_pix"] = True
        row["target_image_precision"] = 1.0
        samples.append(row)
    temporal = json.loads(Path(davis_manifest).read_text(encoding="utf-8"))
    by_sequence = {}
    for row in temporal["records"]:
        by_sequence.setdefault(row["sequence"], {})[row["frame"]] = row
    for sequence, frames in sorted(by_sequence.items()):
        if not all(i in frames for i in (0, 1, 2)):
            raise ValueError(f"{sequence} requires ordered frames 0, 1, 2")
        current = letterbox_tensor(frames[1]["image"], resolution)
        base = {
            "id": f"davis-{sequence}-1", "dataset": "DAVIS-2017-trainval",
            "source_files": [frames[1]["image"]], "target_file": frames[1]["image"],
            "image": current, "target_rgb": current.clone(),
            "image_precision": 1.0, "text_missing": True, "prompt": "", "answer": "",
            "need_pix": True, "need_text": False, "need_seg": True,
            "target_image_precision": 1.0, "target_text_precision": 0.0,
            "target_seg": letterbox_foreground(frames[1]["mask"], resolution),
            "mask_file": frames[1]["mask"], "sequence": sequence,
        }
        samples.append({**base, "task": "segmentation"})
        samples.append({
            **base, "id": f"davis-{sequence}-0-1-to-2", "task": "future",
            "target_file": frames[2]["image"], "mask_file": frames[2]["mask"],
            "target_rgb": letterbox_tensor(frames[2]["image"], resolution),
            "target_seg": letterbox_foreground(frames[2]["mask"], resolution),
            "history_images": torch.stack([
                letterbox_tensor(frames[0]["image"], resolution), current,
            ]),
            "history_files": [frames[0]["image"], frames[1]["image"]],
            "history_frames": [0, 1], "target_frame": 2,
            "history_precision": torch.ones(2), "target_time": 1.0,
        })
    for sample in samples:
        sample["task_id"] = TASK_IDS[sample["task"]]
        sample.setdefault("target_time", 0.0)
        sample.setdefault("history_images", torch.zeros(2, 3, resolution, resolution))
        sample.setdefault("history_precision", torch.zeros(2))
        sample.setdefault("need_seg", False)
        sample.setdefault("target_seg", torch.zeros(resolution, resolution, dtype=torch.long))
        sample["target_seg_precision"] = float(sample["need_seg"])
    return samples


def collate_real_capacity(tokenizer, samples, device):
    batch = collate_real_multimodal(tokenizer, samples, max_text_tokens=128,
                                    min_answer_tokens=64, append_eos=True)
    # Unlike the older real adapter, propagate semantic intent and temporal
    # boundaries. Missing input precision remains separate from supervision.
    for key in ("task_id", "target_time", "target_seg_precision"):
        batch[key] = torch.tensor([s[key] for s in samples],
                                  dtype=torch.long if key == "task_id" else torch.float32)
    for key in ("history_images", "history_precision", "target_seg"):
        batch[key] = torch.stack([s[key] for s in samples])
    batch["need_seg"] = [s["need_seg"] for s in samples]
    batch["action"] = torch.zeros(len(samples), 2)
    batch["action_precision"] = torch.zeros(len(samples))
    batch["t"] = torch.zeros(len(samples))
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def forward_real_capacity(model, batch, with_posterior=False):
    kwargs = {k: batch[k] for k in (
        "labels", "visual_prompt_mask", "text_precision", "image_precision",
        "target_time", "history_images", "history_precision", "action",
        "action_precision", "task_id", "t",
    )}
    kwargs["score_tokens"] = any(batch["need_text"])
    args = (batch["image"], batch["input_ids"], batch["attention_mask"])
    if with_posterior and bool((batch["target_time"] > 0).any()):
        if not bool((batch["target_time"] > 0).all()):
            raise ValueError("posterior training requires a homogeneous temporal microbatch")
        return model.forward_tokens_with_future_posterior(*args, batch["target_rgb"], **kwargs)
    return model.forward_tokens(*args, **kwargs)


def bank_metadata(samples):
    return [{k: v for k, v in sample.items() if not torch.is_tensor(v)} for sample in samples]


def add_zero_horizon_controls(samples):
    """Same future intention/history, zero elapsed time -> observed current frame.

    This prevents a future TToken alone from identifying the target timestamp.
    No future pixel becomes input evidence and no fake action is introduced.
    """
    controls = []
    for sample in samples:
        if sample["task"] != "future" or sample["target_time"] <= 0:
            continue
        observed = next(s for s in samples if s["task"] == "segmentation"
                        and s["sequence"] == sample["sequence"])
        controls.append({**sample, "id": sample["id"] + "-tau0", "target_time": 0.0,
                         "target_rgb": sample["image"].clone(),
                         "target_seg": observed["target_seg"].clone(),
                         "target_file": sample["source_files"][0],
                         "mask_file": observed["mask_file"], "target_frame": sample["history_frames"][-1],
                         "derived_supervision": "zero-horizon prediction of observed current frame"})
    return list(samples) + controls
