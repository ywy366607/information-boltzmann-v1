"""Capability-closed synthetic boundary conditions for one Omni checkpoint.

Each sample supervises every available readout. Ports are not separate models
or task tokens: observed-modality precision and prediction horizon define the
boundary condition, while the same X/H Slice-MoT-Deslice graph always runs.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from fine_grain.omni_tasks import GRID_PLACES, _paint, equal_energy_ink, grid_digit_mask
from fine_grain.vlm_data import COLORS, OCR_DIGITS


CAPABILITY_CASES = (
    "text_to_both",       # T2T + T2I
    "image_to_current",   # I2T + reconstruction + segmentation
    "image_text_edit",    # IT2T + IT2I editing
    "image_to_future",    # I2I next-frame world prediction
)

WORLD_ACTIONS = (
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, -1.0),
    (0.0, 1.0),
)
WORLD_ACTIONS_WITH_NOOP = WORLD_ACTIONS + ((0.0, 0.0),)
_ROWS = ("top", "middle", "bottom")
_COLS = ("left", "center", "right")


def next_color(color: str) -> str:
    colors = list(COLORS)
    return str(colors[(colors.index(str(color)) + 1) % len(colors)])


def next_place(place: str) -> str:
    """Legacy name for one explicit rightward action on a periodic grid."""
    return action_place(place, (1.0, 0.0))


def action_place(place: str, action) -> str:
    """Move one Eulerian grid cell with periodic boundaries."""
    row, col = str(place).split("_", 1)
    dx, dy = float(action[0]), float(action[1])
    ri = (_ROWS.index(row) + int(round(dy))) % len(_ROWS)
    ci = (_COLS.index(col) + int(round(dx))) % len(_COLS)
    return f"{_ROWS[ri]}_{_COLS[ci]}"


def _scene(digit: str, color: str, place: str, res: int):
    stroke = grid_digit_mask(str(digit), int(res), str(place))
    blank = torch.zeros(1, 3, int(res), int(res), dtype=torch.float32)
    rgb = _paint(blank, stroke, equal_energy_ink(str(color)))
    return rgb, stroke


def capability_sample(
    rng: np.random.Generator,
    res: int,
    case: str,
    digit: str | None = None,
    color: str | None = None,
    place: str | None = None,
    velocity_override=None,
    action_override=None,
) -> Dict:
    """Create one falsifiable boundary case on the shared digit scene family."""
    case = str(case)
    if case not in CAPABILITY_CASES:
        raise ValueError(f"unknown capability case {case!r}")
    digit = str(digit if digit is not None else rng.choice(OCR_DIGITS))
    color = str(color if color is not None else rng.choice(list(COLORS)))
    place = str(place if place is not None else rng.choice(GRID_PLACES))
    source, source_stroke = _scene(digit, color, place, res)
    action = torch.zeros(2, dtype=torch.float32)
    velocity = torch.zeros(2, dtype=torch.float32)
    action_precision = 0.0
    history_precision = torch.ones(2, dtype=torch.float32)
    history_images = torch.stack([source[0], source[0]], dim=0)

    if case == "text_to_both":
        image = torch.zeros_like(source)
        target = source
        stroke = source_stroke
        prompt = f"Draw digit {digit} with a thin {color} stroke at {place.replace('_', ' ')}"
        answer = digit
        image_precision, text_precision, target_time = 0.0, 1.0, 0.0
        history_images = torch.zeros_like(history_images)
        history_precision.zero_()
        target_color, target_place = color, place
    elif case == "image_to_current":
        image = source
        target = source
        stroke = source_stroke
        prompt = "Reconstruct current frame"
        answer = digit
        image_precision, text_precision, target_time = 1.0, 1.0, 0.0
        # Counterfactual control: the proposed action is observed but has not
        # been executed at tau=0, so it must not alter current reconstruction.
        action = torch.tensor(
            WORLD_ACTIONS[int(rng.integers(0, len(WORLD_ACTIONS)))],
            dtype=torch.float32,
        )
        action_precision = 1.0
        target_color, target_place = color, place
    elif case == "image_text_edit":
        target_color = next_color(color)
        target, stroke = _scene(digit, target_color, place, res)
        image = source
        prompt = "Change the stroke to the next color"
        # The answer is not present in the text: it requires source color + rule.
        answer = target_color
        image_precision, text_precision, target_time = 1.0, 1.0, 0.0
        target_place = place
    else:
        # A real three-event history on fixed Eulerian addresses. History
        # identifies inertial velocity; an independent action adds an impulse.
        # Neither the old list order nor a hidden port determines the target.
        velocity_value = (
            velocity_override
            if velocity_override is not None
            else WORLD_ACTIONS[int(rng.integers(0, len(WORLD_ACTIONS)))]
        )
        action_value = (
            action_override
            if action_override is not None
            else WORLD_ACTIONS_WITH_NOOP[
                int(rng.integers(0, len(WORLD_ACTIONS_WITH_NOOP)))
            ]
        )
        velocity = torch.as_tensor(velocity_value, dtype=torch.float32).clone()
        action = torch.as_tensor(action_value, dtype=torch.float32).clone()
        target_place = action_place(place, velocity + action)
        previous_place = action_place(place, -velocity)
        previous, _ = _scene(digit, color, previous_place, res)
        history_images = torch.stack([previous[0], source[0]], dim=0)
        target, stroke = _scene(digit, color, target_place, res)
        image = source
        prompt = "Predict next frame"
        answer = digit
        image_precision, text_precision, target_time = 1.0, 1.0, 1.0
        action_precision = 1.0
        target_color = color

    return {
        "case": case,
        "kind": case,
        "image": image,
        "target_rgb": target,
        "stroke": stroke,
        "target_seg": (stroke[0] > 0.5).long(),
        "prompt": prompt,
        "answer": answer,
        "need_text": True,
        "need_pix": True,
        "need_seg": True,
        "image_precision": image_precision,
        "text_precision": text_precision,
        "target_time": target_time,
        "history_images": history_images,
        "history_precision": history_precision,
        "action": action,
        "velocity": velocity,
        "action_precision": action_precision,
        "target_image_precision": 1.0,
        "target_text_precision": 1.0,
        "target_seg_precision": 1.0,
        "digit": digit,
        "source_color": color,
        "target_color": target_color,
        "source_place": place,
        "target_place": target_place,
    }


def counterfactual_future_group(
    rng: np.random.Generator,
    res: int,
    group_id: int = 0,
    actions: Sequence[Sequence[float]] = WORLD_ACTIONS_WITH_NOOP,
) -> List[Dict]:
    """Return one state/history paired with every registered action target."""
    digit = str(rng.choice(OCR_DIGITS))
    color = str(rng.choice(list(COLORS)))
    place = str(rng.choice(GRID_PLACES))
    velocity = WORLD_ACTIONS[int(rng.integers(0, len(WORLD_ACTIONS)))]
    samples = []
    for action_index, action in enumerate(actions):
        sample = capability_sample(
            rng,
            res,
            "image_to_future",
            digit=digit,
            color=color,
            place=place,
            velocity_override=velocity,
            action_override=action,
        )
        sample["counterfactual_group"] = int(group_id)
        sample["counterfactual_index"] = int(action_index)
        samples.append(sample)
    return samples


def make_counterfactual_future_batch(
    rng: np.random.Generator,
    groups: int,
    res: int,
) -> Dict:
    """Collate complete five-action intervention groups without task labels."""
    if int(groups) < 1:
        raise ValueError("groups must be positive")
    samples = [
        sample
        for group_id in range(int(groups))
        for sample in counterfactual_future_group(rng, res, group_id=group_id)
    ]
    return collate_capability(samples)


def make_capability_batch(
    rng: np.random.Generator,
    batch: int,
    res: int,
    cases: Iterable[str] | None = None,
) -> Dict:
    cases = tuple(cases or CAPABILITY_CASES)
    chosen = [str(rng.choice(cases)) for _ in range(int(batch))]
    samples = [capability_sample(rng, res, case) for case in chosen]
    return collate_capability(samples)


def collate_capability(samples: List[Dict]) -> Dict:
    """Collate explicit samples without introducing a task/port token."""
    chosen = [s["case"] for s in samples]
    batch = {
        "image": torch.cat([s["image"] for s in samples], dim=0),
        "target_rgb": torch.cat([s["target_rgb"] for s in samples], dim=0),
        "stroke": torch.cat([s["stroke"] for s in samples], dim=0),
        "target_seg": torch.stack([s["target_seg"] for s in samples], dim=0),
        "prompt": [s["prompt"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "kind": chosen,
        "case": chosen,
        "need_text": [True] * len(samples),
        "need_pix": [True] * len(samples),
        "need_seg": [True] * len(samples),
        "image_precision": torch.tensor([s["image_precision"] for s in samples]),
        "text_precision": torch.tensor([s["text_precision"] for s in samples]),
        "target_time": torch.tensor([s["target_time"] for s in samples]),
        "history_images": torch.stack([s["history_images"] for s in samples], dim=0),
        "history_precision": torch.stack(
            [s["history_precision"] for s in samples], dim=0,
        ),
        "action": torch.stack([s["action"] for s in samples], dim=0),
        "velocity": torch.stack([s["velocity"] for s in samples], dim=0),
        "action_precision": torch.tensor([s["action_precision"] for s in samples]),
        "target_image_precision": torch.tensor(
            [s["target_image_precision"] for s in samples],
        ),
        "target_text_precision": torch.tensor(
            [s["target_text_precision"] for s in samples],
        ),
        "target_seg_precision": torch.tensor(
            [s["target_seg_precision"] for s in samples],
        ),
        "digit": [s["digit"] for s in samples],
        "source_color": [s["source_color"] for s in samples],
        "target_color": [s["target_color"] for s in samples],
        "source_place": [s["source_place"] for s in samples],
        "target_place": [s["target_place"] for s in samples],
    }
    if all("counterfactual_group" in s for s in samples):
        batch["counterfactual_group"] = torch.tensor(
            [s["counterfactual_group"] for s in samples], dtype=torch.long,
        )
        batch["counterfactual_index"] = torch.tensor(
            [s["counterfactual_index"] for s in samples], dtype=torch.long,
        )
    return batch


def fixed_capability_bank(res: int) -> List[Dict]:
    """All 10 digits at all nine addresses, with colors cycled deterministically."""
    rng = np.random.default_rng(0)
    bank: List[Dict] = []
    colors = list(COLORS)
    for case in CAPABILITY_CASES:
        for di, digit in enumerate(OCR_DIGITS):
            for pi, place in enumerate(GRID_PLACES):
                color = colors[(di + pi) % len(colors)]
                bank.append(
                    capability_sample(
                        rng, res, case, digit=digit, color=color, place=place,
                    )
                )
    return bank
