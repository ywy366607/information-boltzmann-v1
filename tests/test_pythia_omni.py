"""Phase A: frozen Pythia token likelihood on the unified Slice–MoT graph."""
from __future__ import annotations

import pytest
import torch

from fine_grain.llm_backend import canonical_pythia_id, load_frozen_pythia
from fine_grain.omni_model import DualStreamOmni, token_nll_per_sample


def _tiny_causal_lm(vocab=64, d=32, n_layer=2, n_head=4, n_pos=48):
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=vocab,
        n_positions=n_pos,
        n_embd=d,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4 * d,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        summary_first_dropout=0.0,
    )
    lm = GPT2LMHeadModel(cfg)
    for p in lm.parameters():
        p.requires_grad_(False)
    lm.eval()
    return lm


def _omni(lm=None, **kw):
    if lm is None:
        lm = _tiny_causal_lm()
    defaults = dict(
        d_model=32,
        n_slices=8,
        n_layers=2,
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
        use_residual_read=True,
        use_modal_precision=True,
        seg_classes=2,
        pixel_loss_mode="balanced_bce",
        language="toy",
        lm=lm,
    )
    defaults.update(kw)
    return DualStreamOmni(**defaults)


def _ids(batch=2, length=8, vocab=64, prompt=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (batch, length), generator=g)
    mask = torch.ones(batch, length, dtype=torch.long)
    labels = ids.clone()
    labels[:, :prompt] = -100
    return ids, mask, labels


def test_token_nll_reduces_fp16_logits_in_fp32():
    logits = torch.zeros(1, 4, 8, dtype=torch.float16, requires_grad=True)
    labels = torch.tensor([[-100, 2, 3, 4]])
    nll = token_nll_per_sample(logits, labels)
    assert nll.dtype == torch.float32
    nll.mean().backward()
    assert logits.grad is not None


def test_canonical_pythia_id_and_reject_gpt2():
    assert canonical_pythia_id("pythia") == "EleutherAI/pythia-70m"
    assert canonical_pythia_id("pythia-160m") == "EleutherAI/pythia-160m"
    with pytest.raises(ValueError, match="Pythia"):
        canonical_pythia_id("gpt2")
    with pytest.raises(ValueError, match="Pythia"):
        load_frozen_pythia("gpt2")


def test_load_frozen_pythia_fail_loud_no_tinylm(monkeypatch):
    from fine_grain import llm_backend as lb

    monkeypatch.setattr(lb, "_local_manual_dirs", lambda: [])

    def boom(*_a, **_k):
        raise RuntimeError("no network")

    monkeypatch.setattr(lb, "snapshot_modelscope", boom)
    with pytest.raises(RuntimeError) as err:
        lb.load_frozen_pythia("EleutherAI/pythia-70m", device="cpu")
    msg = str(err.value)
    assert "no GPT-2/TinyLM fallback" in msg
    assert "pythia-70m" in msg.lower()
    assert "tinylm path=" not in msg.lower()


def test_pythia_params_stay_frozen():
    lm = _tiny_causal_lm()
    model = _omni(lm)
    model.train()
    assert all(not p.requires_grad for p in model.lm.parameters())
    assert not model.lm.training
    ids, mask, labels = _ids()
    images = torch.rand(2, 3, 8, 8)
    out = model.forward_tokens(images, ids, mask, labels=labels)
    out["token_nll"].mean().backward()
    assert all(p.grad is None for p in model.lm.parameters())
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
    )
    names = {id(p) for g in opt.param_groups for p in g["params"]}
    assert all(id(p) not in names for p in model.lm.parameters())
    sd = model.non_lm_state_dict()
    assert all(not k.startswith("lm.") for k in sd)
    assert model.language_meta()["lm_id"] == "injected"


def test_token_nll_grads_reach_interfaces_and_stem():
    torch.manual_seed(0)
    model = _omni()
    model.train()
    with torch.no_grad():
        model.mot_stack.text_out_gate.fill_(1.0)
        model.mot_stack.proj_gate.fill_(1.0)
    ids, mask, labels = _ids()
    images = torch.rand(2, 3, 8, 8)
    out = model.forward_tokens(images, ids, mask, labels=labels)
    assert out["token_nll"].shape == (2,)
    assert out["rgb"].shape == (2, 3, 8, 8)
    assert out["seg_logits"].shape == (2, 2, 8, 8)
    assert out["X"].shape[1] == 64
    out["token_nll"].mean().backward()
    assert float(model.mot_stack.text_in.weight.grad.abs().sum()) > 0
    assert float(model.mot_stack.text_out.weight.grad.abs().sum()) > 0
    assert float(model.mot_stack.stem.weight.grad.abs().sum()) > 0
    assert float(model.mot_stack.layers[0].mot.Wq_v.weight.grad.abs().sum()) > 0
    assert float(model.mot_stack.layers[0].mot.Wq_t.weight.grad.abs().sum()) > 0


def test_labels_none_skips_nll_and_treats_prefix_as_observed():
    torch.manual_seed(2)
    model = _omni().eval()
    ids, mask, _ = _ids(batch=1, seed=2)
    images = torch.rand(1, 3, 8, 8)
    with torch.no_grad():
        a = model.forward_tokens(images, ids, mask, labels=None)
        ids_b = ids.clone()
        ids_b[:, -1] = (ids_b[:, -1] + 7) % 64
        b = model.forward_tokens(images, ids_b, mask, labels=None)
    assert a["token_nll"] is None and b["token_nll"] is None
    assert a["n_loss_tokens"] == 0
    # Observed prefix: the last token may write vision.
    assert not torch.allclose(a["X"], b["X"], atol=1e-5, rtol=1e-5)


def test_visual_only_token_forward_skips_frozen_decoder(monkeypatch):
    model = _omni().eval()
    ids, mask, _ = _ids(batch=1)
    images = torch.zeros(1, 3, 8, 8)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("frozen causal decoder was called")

    monkeypatch.setattr(model.lm, "forward", forbidden)
    with torch.no_grad():
        out = model.forward_tokens(
            images, ids, mask, labels=None, score_tokens=False,
        )
    assert out["token_logits"] is None
    assert out["token_nll"] is None
    assert out["rgb"].shape == images.shape


def test_init_token_gates_keep_lm_on_embeddings():
    torch.manual_seed(8)
    model = _omni().eval()
    assert float(model.mot_stack.text_out_gate.detach()) == 0.0
    assert float(model.mot_stack.proj_gate.detach()) == 0.0
    ids, mask, labels = _ids(batch=2, seed=8)
    images = torch.zeros(2, 3, 8, 8)
    emb = model.lm.get_input_embeddings()(ids).float()
    native = model.lm(
        inputs_embeds=emb,
        attention_mask=mask,
        labels=labels,
        use_cache=False,
    )
    out = model.forward_tokens(images, ids, mask, labels=labels)
    # Zero vis prefix still occupies n_slices slots, so NLL is not identical
    # to a text-only LM call, but H_llm equals the frozen embeddings.
    Hin = model.mot_stack._last_text_evidence
    assert torch.allclose(Hin, emb, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(out["token_nll"]).all()
    assert float(native.loss) > 0


def test_one_forward_emits_rgb_seg_and_token_nll():
    model = _omni().eval()
    ids, mask, labels = _ids(batch=1)
    images = torch.rand(1, 3, 8, 8)
    out = model.forward_tokens(images, ids, mask, labels=labels)
    assert "token_logits" in out and "rgb" in out and "seg_logits" in out
    assert out["n_vis_tokens"] == 8
    assert out["n_loss_tokens"] == 4
    batch = {
        "need_text": [True],
        "need_pix": [True],
        "need_seg": [True],
        "target_rgb": images,
        "target_seg": torch.zeros(1, 8, 8, dtype=torch.long),
        "target_text_precision": torch.ones(1),
        "target_image_precision": torch.ones(1),
        "target_seg_precision": torch.ones(1),
    }
    loss, meta = model.omni_loss(out, batch, torch.device("cpu"))
    assert torch.isfinite(loss)
    assert meta["n_text"] == meta["n_pix"] == meta["n_seg"] == 1
    assert "token_nll" in meta


def test_answer_suffix_does_not_write_vision_or_earlier_logits():
    torch.manual_seed(3)
    model = _omni().eval()
    ids, mask, labels = _ids(batch=1, seed=3)
    images = torch.rand(1, 3, 8, 8)
    with torch.no_grad():
        a = model.forward_tokens(images, ids, mask, labels=labels)
        ids_b = ids.clone()
        ids_b[:, -1] = (ids_b[:, -1] + 7) % 64
        labels_b = labels.clone()
        labels_b[:, -1] = ids_b[:, -1]
        b = model.forward_tokens(images, ids_b, mask, labels=labels_b)
    n_vis = a["n_vis_tokens"]
    # Last answer token is index n_vis + T - 1 in the frozen decoder.
    keep = n_vis + ids.shape[1] - 1
    assert torch.allclose(a["X"], b["X"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(a["rgb"], b["rgb"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        a["token_logits"][:, :keep], b["token_logits"][:, :keep],
        atol=1e-5, rtol=1e-5,
    )
    assert not torch.allclose(
        a["token_logits"][:, keep], b["token_logits"][:, keep], atol=1e-4,
    )


def test_text_precision_zero_drops_lexical_evidence():
    torch.manual_seed(4)
    model = _omni().eval()
    ids_a, mask, labels = _ids(batch=1, seed=4)
    ids_b = (ids_a + 3) % 64
    labels_b = ids_b.clone()
    labels_b[:, :4] = -100
    images = torch.rand(1, 3, 8, 8)
    zero = torch.zeros(1)
    with torch.no_grad():
        a = model.forward_tokens(
            images, ids_a, mask, labels=labels, text_precision=zero,
        )
        b = model.forward_tokens(
            images, ids_b, mask, labels=labels_b, text_precision=zero,
        )
    assert torch.allclose(a["X"], b["X"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(a["rgb"], b["rgb"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(a["token_logits"], b["token_logits"], atol=1e-5, rtol=1e-5)


def test_target_text_precision_zero_drops_token_accuracy():
    model = _omni().eval()
    ids, mask, labels = _ids(batch=2, seed=5)
    images = torch.rand(2, 3, 8, 8)
    out = model.forward_tokens(images, ids, mask, labels=labels)
    assert float(out["token_nll"].mean().detach()) > 0
    batch = {
        "need_text": [True, True],
        "need_pix": [False, False],
        "need_seg": [False, False],
        "target_text_precision": torch.zeros(2),
        "target_image_precision": torch.zeros(2),
        "target_seg_precision": torch.zeros(2),
    }
    loss, meta = model.omni_loss(out, batch, torch.device("cpu"))
    assert float(loss) == 0.0
    assert meta["token_nll"] == 0.0
    assert meta["n_text"] == 2


def test_real_pythia70m_is_the_token_observation_path():
    """Phase A on the exact local Pythia-70m, not TinyCausalLM."""
    model = DualStreamOmni(
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
        pixel_loss_mode="balanced_bce",
        language="pythia",
        lm_device="cpu",
    )
    assert model.lm_id == "EleutherAI/pythia-70m"
    assert int(model.d_llm) == 512
    assert all(not p.requires_grad for p in model.lm.parameters())
    enc = model.lm_tok(
        ["digit 7 green", "digit 1 red"],
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    ids, mask = enc["input_ids"], enc["attention_mask"]
    labels = ids.clone()
    labels[:, :1] = -100
    images = torch.rand(2, 3, 8, 8)
    model.train()
    with torch.no_grad():
        model.mot_stack.text_out_gate.fill_(1.0)
        model.mot_stack.proj_gate.fill_(1.0)
    out = model.forward_tokens(images, ids, mask, labels=labels)
    assert out["rgb"].shape == (2, 3, 8, 8)
    assert out["seg_logits"].shape == (2, 2, 8, 8)
    assert out["token_nll"].shape == (2,)
    assert torch.isfinite(out["token_nll"]).all()
    out["token_nll"].mean().backward()
    assert float(model.mot_stack.text_in.weight.grad.abs().sum()) > 0
    assert float(model.mot_stack.stem.weight.grad.abs().sum()) > 0
    assert all(p.grad is None for p in model.lm.parameters())
    batch = {
        "need_text": [True, True],
        "need_pix": [True, True],
        "need_seg": [True, True],
        "target_rgb": images,
        "target_seg": torch.zeros(2, 8, 8, dtype=torch.long),
        "target_text_precision": torch.ones(2),
        "target_image_precision": torch.ones(2),
        "target_seg_precision": torch.ones(2),
    }
    loss, meta = model.omni_loss(out, batch, torch.device("cpu"))
    assert torch.isfinite(loss)
    assert meta["n_text"] == meta["n_pix"] == meta["n_seg"] == 2
    assert "token_nll" in meta


def test_forward_tokens_requires_lm():
    model = DualStreamOmni(
        d_model=32, n_slices=8, n_layers=1, n_heads=4, res=8,
        surprise_mode="baseline", s_update="raw", use_stiefel=False,
    )
    with pytest.raises(RuntimeError, match="frozen causal LM"):
        model.forward_tokens(
            torch.rand(1, 3, 8, 8),
            torch.zeros(1, 4, dtype=torch.long),
            torch.ones(1, 4, dtype=torch.long),
        )
