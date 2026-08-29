from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from fine_grain.capability_tasks import capability_sample
from fine_grain.omni_model import DualStreamOmni
from fine_grain.token_tasks import (
    answer_class_nll,
    collate_token_capabilities,
    counterfactual_token_samples,
    graph_greedy_decode,
)


class WordTokenizer:
    bos_token_id = 0
    eos_token_id = 0
    pad_token_id = 1

    def __init__(self):
        self.vocab = {"<eos>": 0, "<pad>": 1}
        self.inverse = {0: "", 1: ""}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        for word in str(text).split():
            if word not in self.vocab:
                index = len(self.vocab)
                self.vocab[word] = index
                self.inverse[index] = word
            ids.append(self.vocab[word])
        return ids

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(self.inverse.get(int(i), f"tok{int(i)}") for i in ids).strip()


def _tiny_lm(vocab=128, d=32):
    from transformers import GPT2Config, GPT2LMHeadModel

    lm = GPT2LMHeadModel(GPT2Config(
        vocab_size=vocab,
        n_positions=64,
        n_embd=d,
        n_layer=1,
        n_head=4,
        n_inner=4 * d,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    ))
    lm.eval()
    for parameter in lm.parameters():
        parameter.requires_grad_(False)
    return lm


def _omni(**overrides):
    kwargs = dict(
        d_model=32,
        n_slices=8,
        n_layers=1,
        n_heads=4,
        res=8,
        surprise_mode="baseline",
        s_update="raw",
        use_stiefel=False,
        deslice_topk=0,
        prior_write=0.0,
        vfe_coef=0.0,
        prior_loss_coef=0.0,
        s0_acc_coef=0.0,
        use_residual_read=False,
        use_modal_precision=True,
        seg_classes=2,
        language="toy",
        lm=_tiny_lm(),
    )
    kwargs.update(overrides)
    return DualStreamOmni(**kwargs)


def _samples(res=8):
    rng = np.random.default_rng(3)
    return [
        capability_sample(rng, res, "text_to_both", "7", "green", "top_left"),
        capability_sample(rng, res, "image_to_current", "2", "red", "middle_center"),
        capability_sample(rng, res, "image_text_edit", "4", "blue", "bottom_right"),
    ]


def test_token_collator_encodes_missing_text_without_empty_string():
    tokenizer = WordTokenizer()
    batch = collate_token_capabilities(tokenizer, _samples())
    i2t = 1
    assert int(batch["input_ids"][i2t, 0]) == tokenizer.bos_token_id
    assert float(batch["text_precision"][i2t, 0]) == 0.0
    assert not bool(batch["visual_prompt_mask"][i2t].any())
    for row in range(3):
        answer = batch["labels"][row].ne(-100)
        assert bool(answer.any())
        assert not bool((answer & batch["visual_prompt_mask"][row]).any())
        assert bool((batch["text_precision"][row][answer] == 1).all())


def test_token_counterfactual_changes_answer_determining_image_only():
    rng = np.random.default_rng(4)
    bank = []
    for digit in ("1", "2"):
        bank.append(capability_sample(
            rng, 8, "image_to_current", digit, "red", "top_left",
        ))
    for color in ("red", "green"):
        bank.append(capability_sample(
            rng, 8, "image_text_edit", "3", color, "top_left",
        ))
    changed = counterfactual_token_samples([bank[0], bank[2]], bank)
    assert changed[0]["answer"] == bank[0]["answer"]
    assert changed[1]["answer"] == bank[2]["answer"]
    assert not torch.equal(changed[0]["image"], bank[0]["image"])
    assert not torch.equal(changed[1]["image"], bank[2]["image"])


def test_inference_generated_suffix_cannot_write_visual_field():
    torch.manual_seed(5)
    model = _omni().eval()
    image = torch.rand(1, 3, 8, 8)
    ids = torch.tensor([[2, 3, 4, 5]])
    attention = torch.ones_like(ids)
    visual = torch.tensor([[True, True, False, False]])
    precision = torch.ones_like(ids, dtype=torch.float32)
    with torch.no_grad():
        first = model.forward_tokens(
            image, ids, attention, visual_prompt_mask=visual,
            text_precision=precision,
        )
        changed = ids.clone()
        changed[:, -1] = 9
        second = model.forward_tokens(
            image, changed, attention, visual_prompt_mask=visual,
            text_precision=precision,
        )
    n_vis = int(first["n_vis_tokens"])
    keep = n_vis + ids.shape[1] - 1
    assert torch.allclose(first["X"], second["X"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        first["token_logits"][:, :keep], second["token_logits"][:, :keep],
        atol=1e-5, rtol=1e-5,
    )


def test_supervised_answer_cannot_be_marked_as_visual_prompt():
    model = _omni()
    ids = torch.tensor([[2, 3, 4]])
    labels = torch.tensor([[-100, -100, 4]])
    with pytest.raises(ValueError, match="answer tokens"):
        model.forward_tokens(
            torch.rand(1, 3, 8, 8), ids, torch.ones_like(ids), labels,
            visual_prompt_mask=torch.ones_like(ids, dtype=torch.bool),
        )


def test_graph_decode_reruns_one_graph_per_token():
    tokenizer = WordTokenizer()
    sample = _samples()[1]
    tokenizer.encode(" " + sample["answer"])
    model = _omni().eval()
    calls = []
    original = model.forward_tokens

    def wrapped(self, *args, **kwargs):
        calls.append(kwargs["visual_prompt_mask"].detach().clone())
        return original(*args, **kwargs)

    model.forward_tokens = types.MethodType(wrapped, model)
    result = graph_greedy_decode(
        model, tokenizer, sample, max_new_tokens=2,
    )
    assert len(calls) == len(result["steps"])
    assert 1 <= len(calls) <= 2
    assert all(not bool(mask.any()) for mask in calls)
    assert all(step["graph_rerun"] for step in result["steps"])


def test_identified_answer_nll_uses_frozen_decoder_logits_without_new_head():
    logits = torch.zeros(2, 5, 16, requires_grad=True)
    # n_vis=2, answer positions in text are 1 and 1, hence predictors are 2.
    with torch.no_grad():
        logits[0, 2, 7] = 4.0
        logits[1, 2, 9] = 4.0
    out = {"token_logits": logits, "n_vis_tokens": 2}
    batch = {"labels": torch.tensor([[-100, 7], [-100, 9]])}
    loss = answer_class_nll(out, batch)
    assert float(loss.detach()) < 0.1
    loss.backward()
    assert logits.grad is not None


def test_terminal_token_atlas_is_fixed_eulerian_slice_of_live_x():
    model = _omni(n_slices=4, terminal_token_atlas=True).eval()
    names = model.set_optimization_phase("token_interface")
    assert model.mot_stack.terminal_atlas_mix.weight.requires_grad
    assert model.mot_stack.terminal_atlas_to_text.weight.requires_grad
    assert "mot_stack.terminal_atlas_mix.weight" in names
    ids = torch.tensor([[2, 3]])
    with torch.no_grad():
        out = model.forward_tokens(
            torch.rand(1, 3, 8, 8), ids, torch.ones_like(ids),
        )
        field = out["X"].transpose(1, 2).reshape(1, 32, 8, 8)
        expected = torch.nn.functional.adaptive_avg_pool2d(
            field, (2, 2),
        ).flatten(2).transpose(1, 2)
    assert expected.shape == (1, 4, 32)
    assert torch.allclose(
        model.mot_stack._last_terminal_token_slices, expected,
        atol=1e-6, rtol=1e-6,
    )


def test_token_reader_update_cannot_change_static_visual_field():
    torch.manual_seed(11)
    model = _omni(n_layers=2)
    model.set_optimization_phase("token_reader")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
    )
    image = torch.rand(2, 3, 8, 8)
    ids = torch.tensor([[2, 3, 4], [5, 6, 7]])
    labels = torch.tensor([[-100, -100, 4], [-100, -100, 7]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        before = model.forward_tokens(image, ids, mask, labels=labels)
        x_before = before["X"].clone()
        rgb_before = before["rgb"].clone()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model.forward_tokens(image, ids, mask, labels=labels)
    out["token_nll"].mean().backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        after = model.forward_tokens(image, ids, mask, labels=labels)
    assert torch.allclose(x_before, after["X"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(rgb_before, after["rgb"], atol=1e-6, rtol=1e-6)


def test_answer_class_nll_restricts_rows_to_identified_group():
    logits = torch.zeros(3, 5, 16, requires_grad=True)
    with torch.no_grad():
        # Identified group rows 0/1: one-token answers 7 and 9. Row 2 is a
        # hard-replay row duplicating class 7, which must not join the contrast.
        logits[0, 2, 7] = 4.0
        logits[1, 2, 9] = 4.0
        logits[2, 2, 7] = 4.0
    out = {"token_logits": logits, "n_vis_tokens": 2}
    batch = {"labels": torch.tensor([
        [-100, 7], [-100, 9], [-100, 7],
    ])}
    loss = answer_class_nll(out, batch, rows=[0, 1])
    assert float(loss.detach()) < 0.1
    loss.backward()
    assert logits.grad is not None
    with pytest.raises(ValueError):
        answer_class_nll(out, batch)


def test_hard_replay_picks_failing_cells_round_robin():
    from scripts.train_pythia_capabilities import make_static_bank
    from scripts.train_pythia_tokens import (
        hard_cells_from_records,
        pick_hard_samples,
    )

    records = [
        {"digit": "6", "place": "top_left", "color": "red",
         "accuracy": 0.0, "matched_nll": 3.0, "gap": 0.5},
        {"digit": "3", "place": "center", "color": "blue",
         "accuracy": 1.0, "matched_nll": 0.1, "gap": 2.0},
        {"digit": "9", "place": "top_left", "color": "red",
         "accuracy": 0.0, "matched_nll": 4.0, "gap": 0.2},
    ]
    cells = hard_cells_from_records(records)
    assert cells == [("6", "top_left", "red"), ("9", "top_left", "red")]
    bank = make_static_bank(16, "image_to_current")
    picked = pick_hard_samples(bank, cells, 3, 0)
    assert [sample["digit"] for sample in picked] == ["6", "9", "6"]
    assert all(
        sample["source_place"] == "top_left"
        and sample["source_color"] == "red"
        for sample in picked
    )
    shifted = pick_hard_samples(bank, cells, 2, 1)
    assert [sample["digit"] for sample in shifted] == ["9", "6"]
    assert pick_hard_samples(bank, cells, 0, 0) == []


def test_decode_summaries_records_cell_metadata_and_confusion():
    from scripts.train_pythia_tokens import decode_summaries

    samples = [
        {"digit": "6", "source_place": "top_left", "source_color": "red"},
        {"digit": "9", "source_place": "top_left", "source_color": "red"},
    ]
    rows = [
        {"expected": "6", "text": "8", "exact": False},
        {"expected": "9", "text": "9", "exact": True},
    ]
    results, confusion = decode_summaries(samples, rows)
    assert results[0]["digit"] == "6"
    assert results[0]["place"] == "top_left"
    assert results[0]["exact"] is False
    assert confusion == {"6": {"8": 1}, "9": {"9": 1}}
