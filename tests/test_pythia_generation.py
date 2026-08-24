"""Phase B: load a generation champion into a frozen-LM Omni without language bypass."""
from __future__ import annotations

import torch

from fine_grain.omni_model import DualStreamOmni
from fine_grain.pythia_bridge import (
    is_language_interface_key,
    load_visual_champion,
    param_groups,
)


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
        use_residual_read=False,
        use_modal_precision=False,
        pixel_loss_mode="balanced_bce",
        language="toy",
        lm=lm,
    )
    defaults.update(kw)
    return DualStreamOmni(**defaults)


def _src_omni():
    torch.manual_seed(0)
    return DualStreamOmni(
        d_model=32,
        n_slices=8,
        n_layers=1,
        n_heads=4,
        res=8,
        surprise_mode="baseline",
        s_update="raw",
        use_stiefel=False,
        deslice_topk=0,
        prior_write=1.0,
        vfe_coef=0.0,
        prior_loss_coef=0.0,
        s0_acc_coef=0.0,
        use_residual_read=False,
        pixel_loss_mode="balanced_bce",
        spatial_prompt_vocab=True,
    )


def test_language_interface_keys():
    assert is_language_interface_key("embed.weight")
    assert is_language_interface_key("head.1.weight")
    assert is_language_interface_key("mot_stack.text_in.weight")
    assert is_language_interface_key("mot_stack.text_out.bias")
    assert is_language_interface_key("mot_stack.proj.net.0.weight")
    assert is_language_interface_key("lm.transformer.wte.weight")
    assert not is_language_interface_key("pix_head.1.weight")
    assert not is_language_interface_key("mot_stack.stem.weight")
    assert not is_language_interface_key("mot_stack.layers.0.mot.Wq_v.weight")


def test_load_skips_embed_head_and_text_maps_even_if_shapes_match(tmp_path):
    src = _src_omni()
    with torch.no_grad():
        src.pix_head[-1].bias.fill_(0.42)
        src.mot_stack.stem.bias.fill_(0.17)
        src.mot_stack.text_in.weight.fill_(1.25)
        src.head[1].weight.fill_(2.5)
        src.embed.weight.fill_(-0.5)
    path = tmp_path / "champ.pt"
    torch.save(src.state_dict(), path)

    lm = _tiny_causal_lm(d=32)
    dst = _omni(
        lm,
        n_layers=1,
        use_residual_read=False,
        use_modal_precision=False,
        prior_write=1.0,
        spatial_prompt_vocab=True,
        seg_classes=0,
    )
    report = load_visual_champion(dst, path)
    skipped = set(report["skipped"])
    assert "embed.weight" in skipped
    assert "head.1.weight" in skipped
    assert "mot_stack.text_in.weight" in skipped
    assert "mot_stack.text_out.weight" in skipped
    assert report["loaded"] > 0
    assert torch.allclose(dst.pix_head[-1].bias, src.pix_head[-1].bias)
    assert torch.allclose(dst.mot_stack.stem.bias, src.mot_stack.stem.bias)
    assert not torch.allclose(dst.mot_stack.text_in.weight, src.mot_stack.text_in.weight)
    assert not torch.allclose(dst.head[1].weight, src.head[1].weight)


def test_full_load_preserves_rgb_on_matched_architecture(tmp_path):
    src = _src_omni().eval()
    image = torch.zeros(1, 3, 8, 8)
    prompts = ["Draw digit 7 with a thin green stroke at top left"]
    with torch.no_grad():
        src_rgb = src(image, prompts)["rgb"]
    path = tmp_path / "champ.pt"
    torch.save(src.state_dict(), path)

    dst = _src_omni().eval()
    with torch.no_grad():
        dst.pix_head[-1].bias.zero_()
    load_visual_champion(dst, path, skip_language_interface=False)
    with torch.no_grad():
        dst_rgb = dst(image, prompts)["rgb"]
    assert torch.allclose(src_rgb, dst_rgb, atol=1e-5, rtol=1e-5)


def test_interface_phase_trains_only_text_maps():
    model = _omni()
    names = set(model.set_optimization_phase("interface"))
    assert names
    assert all(
        n.startswith("mot_stack.text_in.") or n.startswith("mot_stack.text_out.")
        for n in names
    )
    for n, p in model.named_parameters():
        if n.startswith("lm."):
            assert not p.requires_grad
        elif n.startswith("mot_stack.text_in.") or n.startswith("mot_stack.text_out."):
            assert p.requires_grad
        else:
            assert not p.requires_grad


def test_language_phase_trains_readers_not_visual_write():
    model = DualStreamOmni(
        d_model=32, n_slices=8, n_layers=1, n_heads=4, res=8,
        surprise_mode="v1_bayes", s_update="raw", use_stiefel=False,
        deslice_topk=0, prior_write=1.0, vfe_coef=0.0, s0_acc_coef=0.0,
        use_residual_read=False, pixel_loss_mode="balanced_bce",
        language="toy", lm=_tiny_causal_lm(),
    )
    names = set(model.set_optimization_phase("language"))
    assert any(n.endswith("mot.Wq_t.weight") for n in names)
    assert any("text_in" in n for n in names)
    assert any("surprise_gate.prior_head" in n for n in names)
    assert model.mot_stack.layers[0].mot.Wq_t.weight.requires_grad
    assert model.mot_stack.text_in.weight.requires_grad
    assert not model.mot_stack.stem.weight.requires_grad
    assert not model.mot_stack.layers[0].mot.Wq_v.weight.requires_grad
    assert not model.pix_head[-1].weight.requires_grad
    assert not model.mot_stack.layers[0].deslice.proj.weight.requires_grad
    assert not model.mot_stack.layers[0].surprise_gate.post_head[0].weight.requires_grad
    tok_names = set(model.set_optimization_phase("token_interface"))
    assert any("text_out" in n for n in tok_names)
    assert any(n.startswith("mot_stack.proj") for n in tok_names)
    assert "mot_stack.text_out_gate" in tok_names
    assert "mot_stack.proj_gate" in tok_names
    assert not model.mot_stack.text_in.weight.requires_grad
    assert not model.mot_stack.stem.weight.requires_grad
    rgb_names = set(model.set_optimization_phase("language_rgb"))
    assert model.pix_head[-1].weight.requires_grad
    assert not model.mot_stack.stem.weight.requires_grad
    assert any(n.startswith("pix_head.") for n in rgb_names)
    likelihood_names = set(model.set_optimization_phase("rgb_likelihood"))
    assert likelihood_names
    assert all(
        name.startswith("pix_head.") or name.startswith("pix_log")
        for name in likelihood_names
    )
    assert not model.mot_stack.text_in.weight.requires_grad
    assert not model.mot_stack.stem.weight.requires_grad
    spatial = DualStreamOmni(
        d_model=32, n_slices=8, n_layers=1, n_heads=4, res=8,
        surprise_mode="v1_bayes", s_update="raw", use_stiefel=False,
        deslice_topk=0, prior_write=1.0, vfe_coef=0.0, s0_acc_coef=0.0,
        use_residual_read=False, use_modal_precision=True,
        pixel_loss_mode="balanced_bce", language="toy", lm=_tiny_causal_lm(),
    )
    names = set(spatial.set_optimization_phase("edit_spatial"))
    assert spatial.mot_stack.image_precision_coord.weight.requires_grad
    assert spatial.mot_stack.layers[0].mot.Wq_t.weight.requires_grad
    assert not spatial.pix_head[-1].weight.requires_grad
    assert not spatial.mot_stack.stem.weight.requires_grad
    assert not spatial.mot_stack.layers[0].deslice.proj.weight.requires_grad
    assert not spatial.mot_stack.layers[0].read.to_logits.weight.requires_grad
    assert any("image_precision_coord" in n for n in names)
    groups = param_groups(spatial, interface_lr=1e-4, visual_lr=1e-5)
    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-4
    read_names = set(spatial.set_optimization_phase("edit_read"))
    assert spatial.mot_stack.layers[0].read.to_logits.weight.requires_grad
    assert spatial.mot_stack.layers[0].mot.Wq_v.weight.requires_grad
    assert spatial.mot_stack.layers[0].mot.Wo_v.weight.requires_grad
    assert spatial.mot_stack.layers[0].deslice.proj.weight.requires_grad
    assert not spatial.mot_stack.layers[0].mot.Wk_v.weight.requires_grad
    assert not spatial.mot_stack.layers[0].mot.Wv_v.weight.requires_grad
    assert not spatial.mot_stack.stem.weight.requires_grad
    assert not spatial.pix_head[-1].weight.requires_grad
    assert any(".read." in n for n in read_names)
    assert any(".mot.Wq_v" in n for n in read_names)
    read_groups = param_groups(spatial, interface_lr=1e-4, visual_lr=1e-5)
    assert len(read_groups) == 2
    assert read_groups[0]["lr"] == 1e-4
    assert read_groups[1]["lr"] == 1e-5
    visual_ids = {id(p) for p in read_groups[1]["params"]}
    assert id(spatial.mot_stack.layers[0].mot.Wq_v.weight) in visual_ids
    assert id(spatial.mot_stack.layers[0].read.to_logits.weight) in visual_ids
    assert id(spatial.mot_stack.image_precision_coord.weight) not in visual_ids
    generation_names = set(spatial.set_optimization_phase("generation_write"))
    assert spatial.mot_stack.text_in.weight.requires_grad
    assert spatial.mot_stack.layers[0].mot.Wk_t.weight.requires_grad
    assert spatial.mot_stack.layers[0].mot.Wv_t.weight.requires_grad
    assert spatial.mot_stack.layers[0].mot.Wq_v.weight.requires_grad
    assert spatial.mot_stack.layers[0].deslice.proj.weight.requires_grad
    assert spatial.pix_head[-1].weight.requires_grad
    assert not spatial.mot_stack.stem.weight.requires_grad
    assert not spatial.mot_stack.layers[0].mot.Wq_t.weight.requires_grad
    assert not spatial.mot_stack.text_out.weight.requires_grad
    assert any(name.startswith("pix_head.") for name in generation_names)


def test_token_reader_opens_only_terminal_h_query_and_output():
    model = _omni(n_layers=2)
    names = set(model.set_optimization_phase("token_reader"))
    first = model.mot_stack.layers[0].mot
    last = model.mot_stack.layers[1].mot
    assert not first.Wq_t.weight.requires_grad
    assert last.Wq_t.weight.requires_grad
    assert last.Wo_t.weight.requires_grad
    assert last.ffn_t.w1.weight.requires_grad
    assert not last.Wk_t.weight.requires_grad
    assert not last.Wv_t.weight.requires_grad
    assert not last.Wq_v.weight.requires_grad
    assert not model.mot_stack.stem.weight.requires_grad
    assert model.mot_stack.readout.to_logits.weight.requires_grad
    assert any("layers.1.mot.Wq_t" in name for name in names)
    assert any(name.startswith("mot_stack.readout.") for name in names)


def test_edit_step_records_source_shuffle_loss():
    import numpy as np
    from scripts.train_pythia_generation import (
        make_edit_bank,
        make_t2i_bank,
        one_step,
        source_shuffle_images,
    )

    torch.manual_seed(0)
    # Toy embeddings: one_step calls model(image, prompts), which needs a
    # tokenizer if an LM is injected. The shuffle/precision contract is
    # independent of Pythia tokenization.
    model = DualStreamOmni(
        d_model=32, n_slices=8, n_layers=1, n_heads=4, res=8,
        surprise_mode="baseline", s_update="raw", use_stiefel=False,
        deslice_topk=0, prior_write=1.0, vfe_coef=0.0, s0_acc_coef=0.0,
        use_residual_read=False, use_modal_precision=True,
        pixel_loss_mode="balanced_bce", capability_vocab=True,
    )
    bank = make_edit_bank(8, n=8, style="named")
    rng = np.random.default_rng(0)
    _, off = one_step(
        model, bank, torch.device("cpu"), 4, rng,
        shuffle_coef=0.0, edit_shuffle_coef=0.0,
    )
    assert "source_shuffle" not in off
    assert off["image_precision"] == 1.0
    assert off["text_precision"] == 1.0
    _, on = one_step(
        model, bank, torch.device("cpu"), 4, np.random.default_rng(0),
        shuffle_coef=0.0, edit_shuffle_coef=0.1,
    )
    assert "source_shuffle" in on
    assert on["shuffle_coef_used"] == 0.1
    assert on["image_precision"] == 1.0
    assert on["text_precision"] == 1.0
    t2i = make_t2i_bank(8, colors=["green"])
    _, tmeta = one_step(
        model, t2i, torch.device("cpu"), 4, np.random.default_rng(1),
        shuffle_coef=1.0, edit_shuffle_coef=0.1,
    )
    assert "digit_shuffle" in tmeta
    assert "source_shuffle" not in tmeta
    assert tmeta["image_precision"] == 0.0
    assert tmeta["text_precision"] == 1.0


def test_source_shuffle_keeps_target_color_changes_geometry():
    import numpy as np
    from scripts.train_pythia_generation import make_edit_bank, source_shuffle_images

    bank = make_edit_bank(8, n=24, style="named")
    batch = bank[:6]
    shuf = source_shuffle_images(batch, bank, np.random.default_rng(3), torch.device("cpu"))
    assert shuf.shape[0] == 6
    changed = 0
    for i, sample in enumerate(batch):
        img = shuf[i].cpu()
        matches = []
        for other in bank:
            oimg = other["image"]
            if oimg.dim() == 4:
                oimg = oimg[0]
            if torch.allclose(img, oimg.cpu().reshape_as(img)):
                matches.append(other)
        assert matches
        want = sample.get("target_color") or sample.get("color")
        assert all((m.get("target_color") or m.get("color")) == want for m in matches)
        src = sample["image"]
        if src.dim() == 4:
            src = src[0]
        if not torch.allclose(img, src.cpu().reshape_as(img)):
            changed += 1
            assert any(
                str(m.get("digit")) != str(sample.get("digit"))
                or str(m.get("source_place") or m.get("placement") or "")
                != str(sample.get("source_place") or sample.get("placement") or "")
                for m in matches
            )
    assert changed >= 1


def test_joint_phase_unfreezes_visual_keeps_lm_frozen():
    model = _omni()
    model.set_optimization_phase("joint")
    assert model.mot_stack.stem.weight.requires_grad
    assert model.pix_head[-1].weight.requires_grad
    assert model.mot_stack.text_in.weight.requires_grad
    assert not model.embed.weight.requires_grad
    assert not model.head[1].weight.requires_grad
    assert all(not p.requires_grad for p in model.lm.parameters())
    groups = param_groups(model, interface_lr=1e-3, visual_lr=1e-4)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-3
    assert groups[1]["lr"] == 1e-4
