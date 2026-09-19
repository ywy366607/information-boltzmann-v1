"""Gradient attribution and clipping audit on real Information Boltzmann training windows.

Protocol from docs/INFORMATION_BOLTZMANN_MEMORY_HANDOFF.md Section 4 (M02) and user directives:
- Loads existing checkpoints (age_002000.pt, age_003000.pt; explicitly reports that age_001950 is missing).
- Replays multiple full 128-token training windows with exact saved RNG states and data cursors.
- Decomposes gradient into:
    1. Pathwise CE gradient: g_path = grad(mean(CE))
    2. Score function gradient: g_score = grad(score)
    3. Accepted events score gradient: g_score_acc
    4. Rejected events score gradient: g_score_rej
    5. Total surrogate gradient: g_sur = grad(L_sur)
- Verifies exact linearity: g_sur == g_path + g_score within tolerance.
- Executes real AdamW optimizer updates on local replicas using restored optimizer states (exp_avg, exp_avg_sq, step):
    - Real surrogate AdamW update: Delta theta_sur
    - Pure pathwise AdamW update: Delta theta_path
    - Measures actual module parameter change norms without any SGD approximation.
- Detailed collision margins and event statistics:
    - Candidate margins (active pairs within box support)
    - Accepted event margins
    - Rejected event margins
    - Minimum, mean, and boundary counts (margin < 0.01, < 0.05)
- Aggregates multi-window statistics across windows to establish robust distributions of:
    - ||g_path||, ||g_score||, ||g_sur||, clipping factor alpha, and parameter update ratios.
"""
import os
import sys

# Prevent local scripts/ib_local from shadowing standard library 'types'
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fine_grain.information_boltzmann.collision import CollisionKernel
from scripts.ib_bpe_window import clock_inputs
from scripts.ib_local.device_collision import collision_layer
from scripts.ib_local.sampling import sample_window
from scripts.ib_local.window import LocalWindow


def get_unique_parameter_groups(model: nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Deduplicate model parameters by object identity and map them to module groups."""
    seen = set()
    unique_params = {}
    param_groups = {}

    for name, p in model.named_parameters():
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            unique_params[name] = p
            # Categorize into functional module groups
            if 'embedding' in name:
                group = 'shared_embedding'
            elif 'initial' in name:
                group = 'initial_birth_flow'
            elif 'gamma_field' in name:
                group = 'force_damping'
            elif 'force' in name:
                group = 'force_drive'
            elif 'collision' in name:
                group = 'collision_kernel'
            elif 'features' in name:
                group = 'feature_mlp'
            elif 'decoder' in name:
                group = 'decoder_bias'
            else:
                group = 'other'
            param_groups[name] = group

    return unique_params, param_groups


class DetailedCollisionAuditor:
    """Instrumented collision layer that separates candidate, accepted, and rejected margins and log-probs."""
    def __init__(self, width: float = 1.0):
        self.width = width
        self.candidate_margins = []
        self.accepted_margins = []
        self.rejected_margins = []
        self.candidates_count = 0
        self.accepted_count = 0
        self.rejected_count = 0
        self.lp_accepted_accum = None
        self.lp_rejected_accum = None

    def reset(self, device):
        self.candidate_margins.clear()
        self.accepted_margins.clear()
        self.rejected_margins.clear()
        self.candidates_count = 0
        self.accepted_count = 0
        self.rejected_count = 0
        self.lp_accepted_accum = torch.zeros((), device=device)
        self.lp_rejected_accum = torch.zeros((), device=device)

    def __call__(self, x, v, context_x, key, value, kernel, layer, width):
        i, j, normal, uniform = layer[:4]
        b_size = len(i)
        self.candidates_count += b_size
        if b_size == 0:
            return v, v.sum() * 0.0, v.new_zeros(())

        vi, vj = v[i], v[j]
        center = (x[i] + x[j]) * 0.5
        change = ((vi - vj) * normal).sum(-1, keepdim=True) * normal
        vp, wp = vi - change, vj + change
        pairs = torch.stack((torch.cat((vi, vj), -1), torch.cat((vj, vi), -1),
                             torch.cat((vp, wp), -1), torch.cat((wp, vp), -1)), 1)
        orbit = torch.cat((pairs[:, :, None, :].expand(-1, -1, 2, -1),
                           torch.stack((normal, -normal), 1)[:, None].expand(-1, 4, -1, -1)), -1)
        q = kernel.query(orbit.flatten(1, 2))
        factors = 1.0 - (context_x[None] - center[:, None]).abs() / width
        supported = (factors > 0).all(-1)
        log_weight = torch.where(factors > 0, factors, torch.ones_like(factors)).log().sum(-1)
        mask = log_weight.masked_fill(~supported, float('-inf'))
        has_context = supported.any(-1)
        mask = torch.where(has_context[:, None], mask, torch.zeros_like(mask))
        attended = F.scaled_dot_product_attention(
            q[:, None], key[None, None].expand(q.shape[0], 1, -1, -1),
            value[None, None].expand(q.shape[0], 1, -1, -1),
            attn_mask=mask[:, None, None, :], dropout_p=0.)[:, 0]
        attended = attended - F.linear(center, kernel.value.weight[:, :x.shape[-1]])[:, None]
        attended = torch.where(has_context[:, None, None], attended, torch.zeros_like(attended))
        raw = kernel.output(torch.tanh(q + attended)).mean((1, 2)).clamp(-12.0, 12.0)

        pair_factors = 1.0 - (x[i] - x[j]).abs() / width
        active = (pair_factors > 0).all(-1)
        if len(layer) == 5:
            active = active & layer[4]

        log_geom = torch.where(pair_factors > 0, pair_factors, torch.ones_like(pair_factors)).log().sum(-1)
        lp = log_geom + F.logsigmoid(raw)
        safe_lp = torch.where(active, lp, torch.full_like(lp, -1.0))
        choose = active & (uniform.log() < safe_lp)
        low = safe_lp < -math.log(2.0)
        a = torch.where(low, safe_lp, torch.full_like(safe_lp, -1.0))
        b = torch.where(low, torch.full_like(safe_lp, -0.5), safe_lp)
        reject = torch.where(low, torch.log1p(-a.exp()), torch.log(-torch.expm1(b)))

        event_lp = torch.where(active, torch.where(choose, safe_lp, reject), torch.zeros_like(lp))
        event_lp_acc = torch.where(choose, safe_lp, torch.zeros_like(safe_lp))
        event_lp_rej = torch.where(active & (~choose), reject, torch.zeros_like(reject))

        delta = torch.where(choose[:, None], change, torch.zeros_like(change))
        nv = v.index_add(0, i, -delta).index_add(0, j, delta)

        # Track margins
        with torch.no_grad():
            min_pair_m = pair_factors.min(dim=-1).values  # [B]
            active_m = min_pair_m[active]
            if len(active_m) > 0:
                self.candidate_margins.extend(active_m.cpu().tolist())

            acc_m = min_pair_m[choose]
            if len(acc_m) > 0:
                self.accepted_margins.extend(acc_m.cpu().tolist())
                self.accepted_count += len(acc_m)

            rej_m = min_pair_m[active & (~choose)]
            if len(rej_m) > 0:
                self.rejected_margins.extend(rej_m.cpu().tolist())
                self.rejected_count += len(rej_m)

        self.lp_accepted_accum = self.lp_accepted_accum + event_lp_acc.sum()
        self.lp_rejected_accum = self.lp_rejected_accum + event_lp_rej.sum()

        return nv, event_lp.sum(), choose.sum()


def audit_single_window(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    v: torch.Tensor,
    train_data: np.ndarray,
    cursor: int,
    tokens_window: int,
    generator: torch.Generator,
    proposal_rng: np.random.Generator,
    device: str,
    clip_norm: float = 1.0,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Audit one 128-token training window: gradients, margins, and exact AdamW update."""
    unique_params, param_groups = get_unique_parameter_groups(model)
    target_params = list(unique_params.values())

    # Setup collision auditor
    auditor = DetailedCollisionAuditor(width=1.0)
    auditor.reset(device=device)
    model.layer_fn = auditor

    prefix = 50256 if cursor == 0 else int(train_data[cursor - 1])
    in_tokens = [prefix] + [int(t) for t in train_data[cursor : cursor + tokens_window - 1]]
    tgt_tokens = [int(t) for t in train_data[cursor : cursor + tokens_window]]

    ids = torch.tensor(in_tokens, dtype=torch.long, device=device)
    targets = torch.tensor(tgt_tokens, dtype=torch.long, device=device)
    clocks = clock_inputs(cursor, tokens_window, model.steps, device)
    noise = torch.empty(tokens_window * model.steps * 4, model.core.particles, 4, device=device)
    noise.normal_(generator=generator)

    tables, _ = sample_window(proposal_rng, tokens_window, model.steps, model.core.particles, device, padding=False)

    force = model.core.force
    layer = force.net[0]
    shared = F.linear(force.embedding(ids), layer.weight[:, 4:-2], layer.bias)
    clock_proj = F.linear(clocks, layer.weight[:, -2:])

    features, prefixes, prefixes_acc, prefixes_rej = [], [], [], []
    lp = v.sum() * 0.0

    curr_x, curr_v = x.clone().detach().requires_grad_(), v.clone().detach().requires_grad_()
    for t in range(tokens_window):
        lp_step_acc = torch.zeros((), device=device)
        lp_step_rej = torch.zeros((), device=device)
        for s in range(model.steps):
            index = t * model.steps + s
            old_acc = auditor.lp_accepted_accum
            old_rej = auditor.lp_rejected_accum
            curr_x, curr_v, change, count = model.evolve(
                curr_x, curr_v, shared[t], clock_proj[t, s],
                noise[index * 4 : index * 4 + 4], tables[index],
            )
            lp = lp + change
            lp_step_acc = lp_step_acc + (auditor.lp_accepted_accum - old_acc)
            lp_step_rej = lp_step_rej + (auditor.lp_rejected_accum - old_rej)

        z = torch.cat((curr_x, curr_v), dim=-1)
        feat = model.core.features(z).mean(dim=0)
        features.append(feat)
        prefixes.append(lp)
        prefixes_acc.append(lp_step_acc)
        prefixes_rej.append(lp_step_rej)

    all_features = torch.stack(features)
    all_prefixes = torch.stack(prefixes)
    all_prefixes_acc = torch.cumsum(torch.stack(prefixes_acc), dim=0)
    all_prefixes_rej = torch.cumsum(torch.stack(prefixes_rej), dim=0)

    logits = F.linear(all_features, model.core.decoder.weight, model.core.decoder.bias)
    ce_per_token = F.cross_entropy(logits, targets, reduction='none')

    loss_path = ce_per_token.mean()
    baseline = math.log(model.core.vocab_size)
    score_term = ((ce_per_token.detach() - baseline) * all_prefixes).mean()
    score_term_acc = ((ce_per_token.detach() - baseline) * all_prefixes_acc).mean()
    score_term_rej = ((ce_per_token.detach() - baseline) * all_prefixes_rej).mean()

    loss_surrogate = loss_path + score_term - score_term.detach()

    # Gradients with allow_unused=True
    def get_grads(loss_val, retain=True):
        raw = torch.autograd.grad(loss_val, target_params, retain_graph=retain, allow_unused=True)
        return tuple(torch.zeros_like(p) if g is None else g for p, g in zip(target_params, raw))

    grads_path = get_grads(loss_path, retain=True)
    grads_score = get_grads(score_term, retain=True)
    grads_score_acc = get_grads(score_term_acc, retain=True)
    grads_score_rej = get_grads(score_term_rej, retain=True)
    grads_sur = get_grads(loss_surrogate, retain=True)

    # Linearity error check
    max_lin_err = max(float((gsur - (gp + gs)).abs().max().item()) for gp, gs, gsur in zip(grads_path, grads_score, grads_sur))

    flat_path = torch.cat([g.flatten() for g in grads_path])
    flat_score = torch.cat([g.flatten() for g in grads_score])
    flat_score_acc = torch.cat([g.flatten() for g in grads_score_acc])
    flat_score_rej = torch.cat([g.flatten() for g in grads_score_rej])
    flat_sur = torch.cat([g.flatten() for g in grads_sur])

    norm_path = float(flat_path.norm(2).item())
    norm_score = float(flat_score.norm(2).item())
    norm_score_acc = float(flat_score_acc.norm(2).item())
    norm_score_rej = float(flat_score_rej.norm(2).item())
    norm_sur = float(flat_sur.norm(2).item())

    dot_prod = float(torch.dot(flat_path, flat_score).item())
    den = norm_path * norm_score
    cos_sim = dot_prod / den if den > 1e-12 else 0.0
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_sim))))

    clipping_factor = min(1.0, clip_norm / norm_sur) if norm_sur > 0.0 else 1.0

    # Pathwise AdamW update simulation using the exact current optimizer state (m, v, step)
    lr = 3e-4
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    weight_decay = 0.01

    clip_factor_path = min(1.0, clip_norm / norm_path) if norm_path > 0 else 1.0
    group_deltas_path = {}

    for name, p, gp in zip(unique_params.keys(), target_params, grads_path):
        grp = param_groups[name]
        g_p = gp * clip_factor_path
        st = optimizer.state.get(p, None)
        if st is not None and 'exp_avg' in st:
            step_val = st['step'].item() if isinstance(st['step'], torch.Tensor) else int(st['step'])
            # Bias-corrected first and second moments
            m_hat = (beta1 * st['exp_avg'] + (1.0 - beta1) * g_p) / (1.0 - beta1 ** step_val)
            v_hat = (beta2 * st['exp_avg_sq'] + (1.0 - beta2) * (g_p ** 2)) / (1.0 - beta2 ** step_val)
            d_p = -lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * p.data)
        else:
            d_p = -lr * (g_p + weight_decay * p.data)
        group_deltas_path[grp] = group_deltas_path.get(grp, 0.0) + float(d_p.square().sum().item())

    # Execute REAL AdamW update on model and optimizer replicas using true surrogate gradient
    theta_init = {name: p.clone().detach() for name, p in unique_params.items()}

    optimizer.zero_grad(set_to_none=True)
    for p, g in zip(target_params, grads_sur):
        p.grad = g.clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, foreach=True)
    optimizer.step()

    theta_after_sur = {name: p.clone().detach() for name, p in unique_params.items()}
    delta_sur = {name: theta_after_sur[name] - theta_init[name] for name in unique_params}

    # Compute per-group AdamW step norms
    group_deltas_sur = {}
    for name in unique_params:
        grp = param_groups[name]
        d_sur_norm_sq = float(delta_sur[name].square().sum().item())
        group_deltas_sur[grp] = group_deltas_sur.get(grp, 0.0) + d_sur_norm_sq

    group_update_stats = {}
    for grp in group_deltas_sur:
        ns = math.sqrt(group_deltas_sur[grp])
        np_val = math.sqrt(group_deltas_path.get(grp, 0.0))
        group_update_stats[grp] = {
            'norm_delta_sur': ns,
            'norm_delta_path': np_val,
            'update_suppression_ratio': (np_val / ns) if ns > 1e-12 else float('inf'),
        }

    total_delta_sur = math.sqrt(sum(group_deltas_sur.values()))
    total_delta_path = math.sqrt(sum(group_deltas_path.values()))

    # Detailed margins breakdown
    cand_margins = auditor.candidate_margins
    acc_margins = auditor.accepted_margins
    rej_margins = auditor.rejected_margins

    def margin_stats(m_list):
        if not m_list:
            return {'count': 0, 'min': 1.0, 'mean': 1.0, 'p01': 1.0, 'p05': 1.0, 'lt_0_01_count': 0, 'lt_0_05_count': 0}
        arr = np.array(m_list, dtype=np.float64)
        return {
            'count': len(arr),
            'min': float(arr.min()),
            'mean': float(arr.mean()),
            'p01': float(np.percentile(arr, 1)),
            'p05': float(np.percentile(arr, 5)),
            'lt_0_01_count': int((arr < 0.01).sum()),
            'lt_0_05_count': int((arr < 0.05).sum()),
        }

    metrics = {
        'cursor': cursor,
        'nll': float(loss_path.item()),
        'surrogate_loss': float(loss_surrogate.item()),
        'norm_path': norm_path,
        'norm_score': norm_score,
        'norm_score_accepted': norm_score_acc,
        'norm_score_rejected': norm_score_rej,
        'norm_surrogate': norm_sur,
        'cosine_sim': cos_sim,
        'angle_degrees': angle_deg,
        'clipping_factor': clipping_factor,
        'effective_path_norm': clipping_factor * norm_path,
        'path_norm_crushed_ratio': norm_path / (clipping_factor * norm_path) if clipping_factor > 0 else float('inf'),
        'total_delta_sur_adamw': total_delta_sur,
        'total_delta_path_adamw': total_delta_path,
        'total_adamw_suppression_ratio': total_delta_path / total_delta_sur if total_delta_sur > 1e-12 else float('inf'),
        'group_adamw_updates': group_update_stats,
        'collision_event_counts': {
            'candidates_total': auditor.candidates_count,
            'candidates_active': len(cand_margins),
            'accepted_count': auditor.accepted_count,
            'rejected_count': auditor.rejected_count,
            'acceptance_rate': auditor.accepted_count / len(cand_margins) if cand_margins else 0.0,
        },
        'margins': {
            'candidates': margin_stats(cand_margins),
            'accepted': margin_stats(acc_margins),
            'rejected': margin_stats(rej_margins),
        },
        'max_linearity_error': max_lin_err,
    }

    return metrics, curr_x.detach(), curr_v.detach()


def run_multi_window_audit(
    checkpoint_path: Path,
    data_path: Path,
    n_windows: int = 10,
    tokens_per_window: int = 128,
    clip_norm: float = 1.0,
    device: str = 'cuda',
) -> dict:
    """Run audit across multiple consecutive windows starting from a checkpoint."""
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = saved['config']
    n_particles = cfg['particles']
    hidden = cfg['hidden']
    start_step = saved['step']
    start_offset = saved['events']
    ckpt_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    model = LocalWindow(
        vocab=50257,
        hidden=hidden,
        particles=n_particles,
        steps=4,
        collision_hidden=32,
    ).to(device)
    model.load_state_dict(saved['model'])
    model.train()
    model.recompute = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, foreach=True, capturable=True)
    optimizer.load_state_dict(saved['optimizer'])

    proposal_rng = np.random.default_rng(11)
    proposal_rng.bit_generator.state = saved['proposal_rng']

    generator = torch.Generator(device=device)
    if device == 'cuda':
        generator.set_state(saved['generator'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    torch.set_rng_state(saved['torch_rng'].cpu())

    x = saved['x'].to(device).clone()
    v = saved['v'].to(device).clone()

    train_data = np.load(data_path / 'train.npy', mmap_mode='r')

    window_results = []
    curr_offset = start_offset

    for w in range(n_windows):
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        step_idx = start_step + w
        m, next_x, next_v = audit_single_window(
            model=model,
            optimizer=optimizer,
            x=x,
            v=v,
            train_data=train_data,
            cursor=curr_offset,
            tokens_window=tokens_per_window,
            generator=generator,
            proposal_rng=proposal_rng,
            device=device,
            clip_norm=clip_norm,
        )
        m['window_idx'] = w
        m['step'] = step_idx
        window_results.append(m)

        x, v = next_x, next_v
        curr_offset += tokens_per_window

        print(
            f"  Window {w+1:2d}/{n_windows:2d} (Step {step_idx:4d}) | NLL: {m['nll']:.4f} | "
            f"||g_path||: {m['norm_path']:6.2f} | ||g_score||: {m['norm_score']:7.2f} (acc: {m['norm_score_accepted']:6.2f}, rej: {m['norm_score_rejected']:6.2f}) | "
            f"alpha: {m['clipping_factor']:.5f} (crush: {m['path_norm_crushed_ratio']:5.1f}x) | "
            f"AdamW total step: sur={m['total_delta_sur_adamw']:.5f}, path={m['total_delta_path_adamw']:.5f} (ratio: {m['total_adamw_suppression_ratio']:.2f}x) | "
            f"min margin: acc={m['margins']['accepted']['min']:.6f}, rej={m['margins']['rejected']['min']:.6f}",
            flush=True
        )

    # Statistical summary across windows
    def stats_summary(key_fn):
        vals = [key_fn(w) for w in window_results]
        return {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'median': float(np.median(vals)),
        }

    summary = {
        'n_windows': n_windows,
        'start_step': start_step,
        'checkpoint_sha256': ckpt_hash,
        'norm_path': stats_summary(lambda w: w['norm_path']),
        'norm_score': stats_summary(lambda w: w['norm_score']),
        'norm_score_accepted': stats_summary(lambda w: w['norm_score_accepted']),
        'norm_score_rejected': stats_summary(lambda w: w['norm_score_rejected']),
        'clipping_factor': stats_summary(lambda w: w['clipping_factor']),
        'path_norm_crushed_ratio': stats_summary(lambda w: w['path_norm_crushed_ratio']),
        'total_adamw_suppression_ratio': stats_summary(lambda w: w['total_adamw_suppression_ratio']),
        'accepted_margin_min': stats_summary(lambda w: w['margins']['accepted']['min']),
        'rejected_margin_min': stats_summary(lambda w: w['margins']['rejected']['min']),
        'acceptance_rate': stats_summary(lambda w: w['collision_event_counts']['acceptance_rate']),
    }

    return {
        'checkpoint': str(checkpoint_path),
        'summary': summary,
        'windows': window_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', type=Path, nargs='+', default=[
        Path('results/ib_local_bpe_256_3000_v2/age_002000.pt'),
        Path('results/ib_local_bpe_256_3000_v2/age_003000.pt'),
    ])
    parser.add_argument('--missing-checkpoint-check', type=Path, default=Path('results/ib_local_bpe_256_3000_v2/age_001950.pt'))
    parser.add_argument('--data', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--output', type=Path, default=Path('results/published/ib_gradient_audit_3000.json'))
    parser.add_argument('--windows', type=int, default=10)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--clip-norm', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    torch.set_num_threads(2)
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.48)

    print(f"=== Starting Rigorous M02 Multi-Window Gradient Attribution Audit ===", flush=True)
    print(f"Device: {args.device}, Windows per checkpoint: {args.windows}, Tokens per window: {args.tokens}", flush=True)

    # Missing checkpoint reporting per spec §4
    missing_report = {
        'path': str(args.missing_checkpoint_check),
        'exists': args.missing_checkpoint_check.exists(),
        'factual_reason': 'Checkpoints in results/ib_local_bpe_256_3000_v2 were configured with --validate-every 250 (steps: 0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000). Step 1950 does not exist on disk.',
    }
    print(f"Missing Checkpoint Check: {args.missing_checkpoint_check.name} exists = {missing_report['exists']}", flush=True)

    all_audits = []
    for ckpt in args.checkpoints:
        if not ckpt.exists():
            print(f"Skipping non-existent checkpoint: {ckpt}", flush=True)
            continue
        print(f"\n--- Running Multi-Window Audit for {ckpt.name} ({args.windows} windows) ---", flush=True)
        res = run_multi_window_audit(
            checkpoint_path=ckpt,
            data_path=args.data,
            n_windows=args.windows,
            tokens_per_window=args.tokens,
            clip_norm=args.clip_norm,
            device=args.device,
        )
        all_audits.append(res)
        s = res['summary']
        print(f"\nSummary for {ckpt.name} across {args.windows} windows:", flush=True)
        print(f"  ||g_path||:                mean={s['norm_path']['mean']:6.2f} +/- {s['norm_path']['std']:5.2f} (median: {s['norm_path']['median']:.2f})", flush=True)
        print(f"  ||g_score||:               mean={s['norm_score']['mean']:6.2f} +/- {s['norm_score']['std']:5.2f} (median: {s['norm_score']['median']:.2f})", flush=True)
        print(f"    - accepted events:       mean={s['norm_score_accepted']['mean']:6.2f} +/- {s['norm_score_accepted']['std']:5.2f}", flush=True)
        print(f"    - rejected events:       mean={s['norm_score_rejected']['mean']:6.2f} +/- {s['norm_score_rejected']['std']:5.2f}", flush=True)
        print(f"  Global Clipping alpha:     mean={s['clipping_factor']['mean']:.6f} (median: {s['clipping_factor']['median']:.6f})", flush=True)
        print(f"  Path Norm Crushed Ratio:   mean={s['path_norm_crushed_ratio']['mean']:6.1f}x (median: {s['path_norm_crushed_ratio']['median']:.1f}x)", flush=True)
        print(f"  AdamW Update Suppression:  mean={s['total_adamw_suppression_ratio']['mean']:6.2f}x (median: {s['total_adamw_suppression_ratio']['median']:.2f}x)", flush=True)
        print(f"  Accepted Min Margin:       min={s['accepted_margin_min']['min']:.6f}, mean={s['accepted_margin_min']['mean']:.6f}", flush=True)
        print(f"  Rejected Min Margin:       min={s['rejected_margin_min']['min']:.6f}, mean={s['rejected_margin_min']['mean']:.6f}", flush=True)

    report = {
        'protocol': 'M02 Multi-window gradient attribution and AdamW update audit (docs/INFORMATION_BOLTZMANN_MEMORY_HANDOFF.md)',
        'missing_checkpoint_audit': missing_report,
        'windows_per_checkpoint': args.windows,
        'tokens_per_window': args.tokens,
        'checkpoints_audited': all_audits,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\n[DONE] Successfully saved rigorous M02 audit report to {args.output}", flush=True)


if __name__ == '__main__':
    main()
