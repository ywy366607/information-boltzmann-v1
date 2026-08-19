"""Traditional patch-ViT VQA baseline on the same DualStream task.

A-path: fixed p×p grid → shared transformer with text tokens → answer head.
No slices, no deslice, no surprise gate. This is the 'what a ViT VLM does'.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fine_grain.models import Block, SelfAttn
from fine_grain.native_mot import NativeLayerTrace
from fine_grain.vlm_data import COLORS, KINK_KS, OCR_DIGITS


class PatchViTVQAModel(nn.Module):
    """Patch-4 ViT + joint self-attn over [visual patches | text]."""

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        res: int = 32,
        patch: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        assert res % patch == 0, (res, patch)
        self.d_model = d_model
        self.res = res
        self.patch = patch
        self.n_layers = n_layers
        self.grid = res // patch
        self.n_patches = self.grid * self.grid
        self.prior_loss_coef = 0.0
        self.sigreg_coef = 0.0
        self.surprise_mode = "patch_vit"

        self.vocab = {
            "<pad>": 0, "What": 1, "color": 2, "is": 3, "the": 4, "small": 5, "square": 6,
            "How": 7, "many": 8, "corners": 9, "does": 10, "red": 11, "polyline": 12,
            "have": 13, "digit": 14, "drawn": 15, "with": 16, "thin": 17, "stroke": 18,
            "?": 19, "Answer:": 20,
        }
        for c in COLORS:
            if c not in self.vocab:
                self.vocab[c] = len(self.vocab)
        for k in KINK_KS:
            if str(k) not in self.vocab:
                self.vocab[str(k)] = len(self.vocab)
        for d in OCR_DIGITS:
            if d not in self.vocab:
                self.vocab[d] = len(self.vocab)

        self.embed = nn.Embedding(len(self.vocab) + 10, d_model)
        self.stem = nn.Conv2d(3, d_model, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.text_in = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, SelfAttn(d_model, heads=n_heads)) for _ in range(n_layers)]
        )
        self.answers = list(COLORS) + [str(k) for k in KINK_KS] + list(OCR_DIGITS)
        self.ans_to_idx = {a: i for i, a in enumerate(self.answers)}
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, len(self.answers)),
        )
        self.last_patch_states: List[torch.Tensor] = []

    def tokenize(self, prompts: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens_list = []
        for p in prompts:
            words = p.replace("?", " ?").split()
            tokens_list.append([self.vocab.get(w, 0) for w in words])
        max_len = max(len(t) for t in tokens_list)
        B = len(prompts)
        pad_t = torch.zeros(B, max_len, dtype=torch.long, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        for i, t in enumerate(tokens_list):
            pad_t[i, :len(t)] = torch.tensor(t, device=device)
            mask[i, :len(t)] = True
        return pad_t, mask

    def encode_patches(self, images: torch.Tensor) -> torch.Tensor:
        p = self.stem(images).flatten(2).transpose(1, 2)
        return p + self.pos

    def forward(self, images: torch.Tensor, prompts: List[str]) -> Dict:
        B = images.shape[0]
        tok_ids, mask = self.tokenize(prompts, images.device)
        text = self.text_in(self.embed(tok_ids))
        patches = self.encode_patches(images)
        x = torch.cat([patches, text], dim=1)
        traces = []
        patch_states = [patches]
        prev = patches
        for i, blk in enumerate(self.blocks):
            x, _ = blk(x)
            cur = x[:, : self.n_patches]
            patch_states.append(cur)
            delta = cur - prev
            traces.append(
                NativeLayerTrace(
                    layer=i,
                    x_delta=float(delta.detach().pow(2).mean().sqrt()),
                    h_delta=0.0,
                    M=self.n_patches,
                    T=text.size(1),
                    surprise_u=0.0,
                    surprise_gate=1.0,
                    rms_s=float(cur.detach().pow(2).mean().sqrt()),
                    rms_delta=float(delta.detach().pow(2).mean().sqrt()),
                    rms_ratio=float(
                        delta.detach().pow(2).mean().sqrt()
                        / cur.detach().pow(2).mean().sqrt().clamp_min(1e-6)
                    ),
                )
            )
            prev = cur
        self.last_patch_states = patch_states
        H = x[:, self.n_patches :]
        mask_f = mask.unsqueeze(-1).float()
        h_pooled = (H * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        logits = self.head(h_pooled)
        z = logits.new_zeros(())
        return {
            "logits": logits,
            "traces": traces,
            "pred_loss": z,
            "sigreg_loss": z,
            "X": patch_states[-1],
        }

    def task_loss(self, out: Dict, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out["logits"], targets)


def compute_patch_layer_saliency(
    model: PatchViTVQAModel,
    img: torch.Tensor,
    prompt: str,
    target_ans: str,
    res: int,
) -> List:
    """Per-layer patch-token |grad| upsampled to the pixel grid."""
    model.eval()
    model.zero_grad(set_to_none=True)
    device = img.device
    tok_ids, mask = model.tokenize([prompt], device)
    text = model.text_in(model.embed(tok_ids))
    patches = model.encode_patches(img)
    patches.retain_grad()
    acts = [patches]
    hooks = []

    def _hook(_mod, _inp, out):
        y = out[0] if isinstance(out, tuple) else out
        y.retain_grad()
        acts.append(y)

    for blk in model.blocks:
        hooks.append(blk.register_forward_hook(_hook))
    try:
        x = torch.cat([patches, text], dim=1)
        for blk in model.blocks:
            x, _ = blk(x)
        H = x[:, model.n_patches :]
        mask_f = mask.unsqueeze(-1).float()
        h_pooled = (H * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        logits = model.head(h_pooled)
        ans_idx = model.ans_to_idx[target_ans]
        logits[0, ans_idx].backward()
    finally:
        for h in hooks:
            h.remove()

    maps = []
    g = model.grid
    p = model.patch
    # embed + first 3 block outputs → 4 maps aligned with DualStream X0..X3
    states = acts[:4]
    for st in states:
        grad = st.grad
        if grad is None:
            maps.append(np.zeros((res, res), dtype=np.float32))
            continue
        token_g = grad[0]
        if token_g.dim() == 2 and token_g.shape[0] > model.n_patches:
            token_g = token_g[: model.n_patches]
        gn = token_g.norm(dim=-1).reshape(g, g).detach().cpu()
        up = gn.repeat_interleave(p, 0).repeat_interleave(p, 1)
        maps.append(up.numpy())
    return maps
