"""Evaluation alignment and batching tests, not capability experiments."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.ib_local.evaluate_gated_deltanet import FrozenScorer, target_positions, paired_summary
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM


def test_revisit_alignment_excludes_changed_boundary_targets():
    from scripts.ib_local.evaluate_gated_deltanet import aligned_revisit_blocks
    config = dict(tokens=2, phases=[
        dict(name="A_first", start_update=0, stop_update=6, source_byte_start=0),
        dict(name="A_revisit", start_update=8, stop_update=12, source_byte_start=4)])
    rows = [dict(kind="train", phase=p, step=s, window_updates=2, window_nll=v)
            for p,s,v in [("A_first",2,9), ("A_first",4,3), ("A_first",6,100),
                          ("A_revisit",10,2), ("A_revisit",12,200)]]
    result = aligned_revisit_blocks(config, rows)
    assert result["matched_target_bytes"] == 4
    assert result["blocks"][0]["source_target_start"] == 5
    assert result["nll_saving"] == 1
    assert result["recovery_time"] is None


def test_context_arms_share_targets_and_windows_do_not_overlap():
    positions = target_positions(1000, 4, 64, 32, 100)
    assert positions[0] - 64 >= 100
    assert positions[-1] + 32 <= 1000
    assert np.all(np.diff(positions) >= 96)
    with pytest.raises(ValueError):
        target_positions(200, 4, 64, 32, 100)
    summary = paired_summary([.1, .2, .3])
    assert summary["mean"] == pytest.approx(.2)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ragged_continuations_match_individual_causal_scores(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from scripts.ib_local.evaluate_lambada import split_target
    assert split_target("hello world") == (b"hello", b" world")
    torch.manual_seed(19)
    model = GatedDeltaNetLM(vocab_size=256, d=16, layers=1, heads=2,
                           recurrence="contact_exchange", conv_kernel=4,
                           content_budget=1., contact_availability=True).to(device).eval()
    prompts = [b"hello", "语言".encode(), b"a much longer context"]
    targets = [b" world", b"!", b" end"]
    # Include a genuine positive match; all-negative labels cannot audit accuracy.
    with torch.no_grad():
        logits, state, _ = model.predict_logits(torch.tensor([list(prompts[0])], device=device))
        generated = []
        for _ in range(6):
            token = int(logits[0, -1].argmax())
            generated.append(token)
            logits, state, _ = model.predict_logits(torch.tensor([[token]], device=device), state)
        targets[0] = bytes(generated)
    expected, exact = [], []
    with torch.no_grad():
        for prompt, target in zip(prompts, targets):
            ids = torch.tensor([list(prompt+target)], device=device)
            logits, _, _ = model.predict_logits(ids[:, :-1])
            selected = logits[:, len(prompt)-1:]
            labels = ids[:, len(prompt):]
            expected.append(float(F.cross_entropy(selected.flatten(0,1), labels.flatten(), reduction="sum")))
            exact.append(bool((selected.argmax(-1)==labels).all()))
    scorer = FrozenScorer(model, batch=4, chunk=4)
    scores, matches = scorer.continuations(prompts, targets)
    assert scores == pytest.approx(expected, abs=1e-5)
    assert matches == exact
    assert matches[0] is True
    # A partial last batch must not inherit state from earlier rows.
    scores, matches = scorer.continuations(prompts[:1], targets[:1])
    assert scores == pytest.approx(expected[:1], abs=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["decay_then_delta", "feedback_delta", "evidence_exchange", "contact_exchange"])
def test_frozen_score_matches_exact_targets_and_full_forward(device, mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(11)
    model = GatedDeltaNetLM(vocab_size=256, d=16, layers=1, heads=2,
                           recurrence=mode, mlp_hidden=32, conv_kernel=4,
                           content_budget=1.0 if mode != "decay_then_delta" else 0,
                           contact_availability=(mode == "contact_exchange")).to(device).eval()
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    data = np.arange(128, dtype=np.int64)
    positions = np.array([24, 80])
    prefix, scored = 8, 16
    indices = positions[:, None] - prefix + np.arange(prefix + scored)[None, :]
    inputs = torch.as_tensor(data[indices], device=device)
    with torch.no_grad():
        logits, _, _ = model.predict_logits(inputs)
        targets = torch.as_tensor(data[positions[:, None] + np.arange(scored)], device=device)
        expected = F.cross_entropy(logits[:, prefix-1:prefix-1+scored].flatten(0, 1),
                                   targets.flatten(), reduction="none").view(2, scored).mean(1)
    scorer = FrozenScorer(model, batch=2, chunk=4)
    result = scorer.score(data, positions, prefix, scored)
    assert result["region_nll"] == pytest.approx(expected.cpu().tolist(), abs=2e-6)
    assert scorer.score(data, positions, prefix, scored) == result
    assert result["scored_bytes"] == 32
    assert all(torch.equal(saved[k], v) for k, v in model.state_dict().items())
