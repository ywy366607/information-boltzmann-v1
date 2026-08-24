"""Causal token likelihood and graph-native decoding for capability samples.

The answer span is valid causal language context, but never visual evidence.
For image-only I2T, a zero-precision BOS token provides the legal causal start
position; an empty string is not used as a missing-modality convention.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


TOKEN_CASES = ("text_to_both", "image_to_current", "image_text_edit")


def _token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _null_token_id(tokenizer) -> int:
    token_id = tokenizer.bos_token_id
    if token_id is None:
        token_id = tokenizer.eos_token_id
    if token_id is None:
        raise ValueError("tokenizer needs a BOS or EOS token for missing text")
    return int(token_id)


def token_example(tokenizer, sample: dict) -> dict:
    """Turn one static capability sample into prompt/answer causal spans."""
    case = str(sample["case"])
    if case not in TOKEN_CASES:
        raise ValueError(f"token likelihood does not admit {case!r}")
    if case == "image_to_current":
        prefix = [_null_token_id(tokenizer)]
        prefix_precision = [0.0]
        visual_prefix = [False]
    else:
        prefix = _token_ids(tokenizer, str(sample["prompt"]))
        if not prefix:
            raise ValueError("observed text prompt tokenized to an empty span")
        prefix_precision = [1.0] * len(prefix)
        visual_prefix = [True] * len(prefix)
    answer_ids = _token_ids(tokenizer, " " + str(sample["answer"]).strip())
    if not answer_ids:
        raise ValueError("answer tokenized to an empty span")
    ids = prefix + answer_ids
    return {
        "input_ids": ids,
        "labels": [-100] * len(prefix) + answer_ids,
        "visual_prompt_mask": visual_prefix + [False] * len(answer_ids),
        # Teacher-forced answer tokens are causal language context for later
        # answer positions, not cross-modal evidence for X.
        "text_precision": prefix_precision + [1.0] * len(answer_ids),
        "prefix_ids": prefix,
        "prefix_precision": prefix_precision,
        "answer_ids": answer_ids,
        "answer": str(sample["answer"]).strip(),
        "case": case,
    }


def collate_token_capabilities(tokenizer, samples: Sequence[dict]) -> dict:
    """Pad token spans while retaining per-token observation precision."""
    if not samples:
        raise ValueError("cannot collate an empty token batch")
    examples = [token_example(tokenizer, sample) for sample in samples]
    width = max(len(example["input_ids"]) for example in examples)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer needs a pad or EOS token")

    def padded(values, fill):
        return [values + [fill] * (width - len(values)) for values in values]

    ids = padded([x["input_ids"] for x in examples], int(pad_id))
    labels = padded([x["labels"] for x in examples], -100)
    prompt_mask = padded([x["visual_prompt_mask"] for x in examples], False)
    precision = padded([x["text_precision"] for x in examples], 0.0)
    lengths = [len(x["input_ids"]) for x in examples]
    attention = [
        [1] * length + [0] * (width - length) for length in lengths
    ]
    images = torch.cat([
        sample["image"] if sample["image"].dim() == 4
        else sample["image"].unsqueeze(0)
        for sample in samples
    ], dim=0)
    return {
        "image": images,
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "visual_prompt_mask": torch.tensor(prompt_mask, dtype=torch.bool),
        "text_precision": torch.tensor(precision, dtype=torch.float32),
        "image_precision": torch.tensor(
            [float(sample["image_precision"]) for sample in samples],
            dtype=torch.float32,
        ),
        "target_text_precision": torch.ones(len(samples)),
        "answer_ids": [x["answer_ids"] for x in examples],
        "answer": [x["answer"] for x in examples],
        "case": [x["case"] for x in examples],
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


def counterfactual_token_samples(
    samples: Sequence[dict], bank: Sequence[dict] | None = None,
) -> list[dict]:
    """Shuffle only the observation that determines each sample's answer.

    I2T changes digit while preserving color/address. IT2T changes source
    color while preserving digit/address; the original answer remains fixed,
    so the shuffled image contradicts the next-color rule.
    """
    samples = list(samples)
    controls = list(bank) if bank is not None else samples
    out: list[dict] = []
    for sample in samples:
        case = str(sample["case"])
        if case not in ("image_to_current", "image_text_edit"):
            out.append(deepcopy(sample))
            continue
        if case == "image_to_current":
            candidates = [
                other for other in controls
                if other["case"] == case
                and str(other["digit"]) != str(sample["digit"])
                and other["source_color"] == sample["source_color"]
                and other["source_place"] == sample["source_place"]
            ]
        else:
            candidates = [
                other for other in controls
                if other["case"] == case
                and other["source_color"] != sample["source_color"]
                and other["digit"] == sample["digit"]
                and other["source_place"] == sample["source_place"]
            ]
        if not candidates:
            raise ValueError(
                f"token counterfactual bank lacks a matched control for {case}"
            )
        replacement = candidates[0]
        changed = deepcopy(sample)
        changed["image"] = replacement["image"].clone()
        out.append(changed)
    return out


def answer_token_accuracy(out: dict, batch: dict) -> torch.Tensor:
    """Per-sample teacher-forced accuracy over answer positions."""
    logits = out["token_logits"]
    labels = batch["labels"].to(logits.device)
    n_vis = int(out["n_vis_tokens"])
    full_labels = torch.cat([
        labels.new_full((labels.shape[0], n_vis), -100), labels,
    ], dim=1)
    pred = logits[:, :-1].argmax(dim=-1)
    target = full_labels[:, 1:]
    valid = target.ne(-100)
    correct = pred.eq(target) & valid
    return correct.sum(dim=1).float() / valid.sum(dim=1).clamp_min(1)


def answer_class_nll(out: dict, batch: dict) -> torch.Tensor:
    """Identified answer contrast using the same frozen-decoder logits.

    Every row in an identified group has a one-token answer. Restricting the
    denominator to the answer tokens in that group prevents the 50k-word
    vocabulary from hiding a failure to distinguish 0--9 or four colors. This
    is an auxiliary view of the native token likelihood, not a new head.
    """
    logits = out["token_logits"]
    labels = batch["labels"].to(logits.device)
    supervised = labels.ne(-100)
    if not bool((supervised.sum(dim=1) == 1).all().item()):
        raise ValueError("identified answer contrast requires one-token answers")
    answer_pos = supervised.float().argmax(dim=1).long()
    answer_ids = labels.gather(1, answer_pos[:, None]).squeeze(1)
    class_ids = torch.unique(answer_ids, sorted=True)
    if int(class_ids.numel()) != int(answer_ids.numel()):
        raise ValueError("identified group must contain each answer class once")
    pred_pos = int(out["n_vis_tokens"]) + answer_pos - 1
    row = torch.arange(logits.shape[0], device=logits.device)
    class_logits = logits[row, pred_pos][:, class_ids]
    class_target = (answer_ids[:, None] == class_ids[None]).float().argmax(dim=1)
    return F.cross_entropy(class_logits.float(), class_target)


@torch.no_grad()
def graph_greedy_decode(
    model,
    tokenizer,
    sample: dict,
    *,
    max_new_tokens: int | None = None,
) -> dict:
    """Greedy decode by rerunning the complete Slice graph for every token."""
    model.eval()
    example = token_example(tokenizer, sample)
    prefix = list(example["prefix_ids"])
    prefix_precision = list(example["prefix_precision"])
    observed = [bool(x) for x in example["visual_prompt_mask"][: len(prefix)]]
    if max_new_tokens is None:
        max_new_tokens = max(1, len(example["answer_ids"]))
    generated: list[int] = []
    steps = []
    device = next(model.parameters()).device
    image = sample["image"]
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    for _ in range(int(max_new_tokens)):
        ids = prefix + generated
        precision = prefix_precision + [1.0] * len(generated)
        visual_mask = observed + [False] * len(generated)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention = torch.ones_like(input_ids)
        out = model.forward_tokens(
            image,
            input_ids,
            attention,
            labels=None,
            visual_prompt_mask=torch.tensor(
                [visual_mask], dtype=torch.bool, device=device,
            ),
            image_precision=torch.tensor(
                [float(sample["image_precision"])], device=device,
            ),
            text_precision=torch.tensor(
                [precision], dtype=torch.float32, device=device,
            ),
        )
        position = int(out["n_vis_tokens"]) + len(ids) - 1
        token_id = int(out["token_logits"][0, position].argmax().item())
        generated.append(token_id)
        steps.append({"token_id": token_id, "graph_rerun": True})
        if token_id == tokenizer.eos_token_id:
            break
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return {
        "text": text,
        "token_ids": generated,
        "expected": example["answer"],
        "exact": text == example["answer"],
        "steps": steps,
    }


def move_token_batch(batch: dict, device: torch.device) -> dict:
    """Move tensor values without mutating descriptive list fields."""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def select_cases(samples: Iterable[dict], case: str) -> list[dict]:
    return [sample for sample in samples if sample["case"] == case]
