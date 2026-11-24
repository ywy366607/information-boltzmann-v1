"""Stage 1A + 1B zero-training read-only probes on the saved batch2_full_32c (cycle 2845) body.

Zero optimizer updates, zero world calls, zero checkpoint mutations.

Stage 1A (PastPresentField workspace-state counterfactual):
  Separates the utility of a retrieved memory FACT from the utility of its
  carried EVOLVING WORKSPACE state across real sequential +960B continuations
  and topic-jump boundaries:
    - C1: same 8 (slot, version) anchors + KEEP evolved workspace W_water
    - C2: same 8 (slot, version) anchors + ZERO ONLY workspace W_water
    - C3: production refresh_workset (query = experience_layer[-1:])
    - C4: window-mean refresh candidate (query = experience_layer[-32:].mean())
    - C5: PastPresentField disabled (enabled = 0)
  Also measures per-layer survival factor prod(1 - beta_t), early/late byte
  delta-NLL, and per-slot decomposition (zero_ws vs remove_fact vs exchange
  read weight vs bounded_preference).

Stage 1B (Scalar log-response & curvature scan):
  Sweeps log_relation_ridge across [0.01..10.0] on the 64 fixed old facts
  (exact mechanism + log_score + MSE) and memory.log_temperature across
  [0.03..2.0] on real predict_observed CE + short_loss, computes exact 1st/2nd
  derivatives at the 2845 operating point, and maps safe scalar lr candidates
  against the saved 2845 AdamW preconditioned directions.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

PROJECT = Path('D:/information-boltzmann-v1').resolve()
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch
from torch.nn import functional as F

CKPT = Path('E:/mengli-batch2-regression-20261123/batch2_full_32c/body/last.pt.gz').resolve()
MATERIAL = Path('E:/mengli-batch2-regression-20261123/batch2_full_32c/material.pt').resolve()
VAL_BIN = (PROJECT / 'data/cgb_natural_text_20261115/validation.bin').resolve()
SPOOL = Path('E:/mengli-runtime-staging-20261123').resolve()
TMP_WORK = Path('E:/mengli-batch2-regression-20261123/stage1_probe_tmp').resolve()
OUT_PATH = PROJECT / 'results/published/cgb_stage1_workset_and_scales_probe_20261123.json'
STARTS = [0, 95629, 191258, 286887]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pearson(xs, ys):
    a = np.asarray(xs, dtype=np.float64)
    b = np.asarray(ys, dtype=np.float64)
    if len(a) < 2 or a.std() < 1e-15 or b.std() < 1e-15:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(xs, ys):
    a = np.asarray(xs, dtype=np.float64)
    b = np.asarray(ys, dtype=np.float64)
    if len(a) < 2:
        return None
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return pearson(ra, rb)


def load_body():
    if TMP_WORK.exists():
        shutil.rmtree(TMP_WORK, ignore_errors=True)
    os.environ['CGB_CHECKPOINT_SPOOL'] = str(SPOOL)
    from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
    opts = continuation_parser().parse_args([
        '--checkpoint', str(CKPT),
        '--output', str(TMP_WORK),
        '--cycles', '1',
        '--world-source', 'text',
        '--autonomous-events',
        '--optimizer-policy', 'budgeted',
        '--table-storage', str(TMP_WORK / 'table'),
        '--causal-storage', str(TMP_WORK / 'causal'),
    ])
    loaded = prepare_continuation(opts, resume_output=False)
    return loaded


def snapshot_field(field):
    return dict(
        anchors=field.anchors.detach().clone(),
        slots_id=field.slots_id.detach().clone(),
        versions=field.versions.detach().clone(),
        valid=field.valid.detach().clone(),
        preference=field.preference.detach().clone(),
        enabled=field.enabled.detach().clone(),
    )


def restore_field(field, snap):
    field.anchors.copy_(snap['anchors'])
    field.slots_id.copy_(snap['slots_id'])
    field.versions.copy_(snap['versions'])
    field.valid.copy_(snap['valid'])
    field.preference.copy_(snap['preference'])
    field.enabled.copy_(snap['enabled'])


@torch.no_grad()
def apply_refresh_with_query(engine, recurrent, query):
    """Exact read-only copy of past_present_coevolution.refresh_workset using a given query."""
    owner = engine.model
    field, memory = owner.body.coevolution, owner.memory
    read = memory.read(query, owner.encoder)
    slots = read.slots[0, :field.slots]
    versions = memory.read_credit_version[slots]
    valid = memory.occupied[slots]
    same = ((slots[:, None] == field.slots_id[None]) &
            (versions[:, None] == field.versions[None]) & field.valid[None] & valid[:, None])
    index = same.to(torch.int64).argmax(-1)
    retained = same.any(-1)
    physical = owner.base.layers[0].state_size
    old = recurrent[..., physical:].reshape(
        len(owner.base.layers), recurrent.shape[1], field.slots, field.rank)
    work = old[:, :, index] * retained[None, None, :, None]
    new_rec = torch.cat((recurrent[..., :physical], work.flatten(2)), -1)
    field.slots_id.copy_(slots)
    field.versions.copy_(versions)
    field.valid.copy_(valid)
    for i, name in enumerate(('source', 'action', 'successor', 'long_successor')):
        field.anchors[i].copy_(getattr(memory, name)[slots])
    from scripts.ib_local.experience_consolidation import bounded_preference, consolidates_reads
    pref = bounded_preference(memory) if consolidates_reads(memory) else memory.read_credit_mean.tanh()
    field.preference.copy_(pref[slots])
    return new_rec, int(retained.sum()), slots.cpu().tolist()


@torch.no_grad()
def run_window_readonly(engine, raw_bytes: np.ndarray, init_state: torch.Tensor,
                        *, collect_beta_and_weights=False):
    """Run raw_bytes[:-1] -> raw_bytes[1:] read-only through predict_observed (128B chunks).

    Returns per-byte NLL [L], final state, last-32 experience_layer features [32, D],
    and optionally per-layer beta survival factors & per-slot mean read weights.
    """
    from scripts.ib_local.navigation_live_witness import frozen_table_logp
    from scripts.ib_local.quadratic_revision import diagonal_gain
    model = engine.model
    field = model.body.coevolution
    device = engine.device
    L = len(raw_bytes) - 1
    logp = torch.as_tensor(frozen_table_logp(engine.table, raw_bytes[:-1]), device=device)[None]
    ids = torch.as_tensor(raw_bytes[:-1].astype(np.int64), device=device)[None]
    targets = torch.as_tensor(raw_bytes[1:].astype(np.int64), device=device)[None]

    state = init_state.detach().clone()
    nll_chunks = []
    feat_chunks = []

    # Optional hook on field.exchange to collect exact beta & weights without altering math
    orig_exchange = field.exchange
    layer_idx_counter = [0]
    n_layers = len(model.body.base.layers)
    log_surv = torch.zeros(n_layers, field.slots, device=device, dtype=torch.float64) if collect_beta_and_weights else None
    surv_snapshots = {}
    slot_weight_sum = torch.zeros(field.slots, device=device, dtype=torch.float64) if collect_beta_and_weights else None
    slot_weight_count = [0]
    pos_cursor = [0]

    if collect_beta_and_weights:
        def wrapped_exchange(layer, hidden, incoming, previous, keys, bias):
            out = orig_exchange(layer, hidden, incoming, previous, keys, bias)
            l_idx = layer_idx_counter[0] % n_layers
            layer_idx_counter[0] += 1
            r = field.rank
            h = layer.norm(hidden)
            q = F.linear(h, layer.q_proj.weight[:r])
            normed = field.memory_norm(incoming)
            shared_key = F.linear(normed, layer.k_proj.weight[:r, :r])
            score = (F.normalize(q, dim=-1)[:, :, None] * (
                keys[None, None] + F.normalize(shared_key, dim=-1))).sum(-1) / math.sqrt(r)
            score = score + bias[None, None]
            valid = field.valid[None, None]
            score = score.masked_fill(~valid, -1e4)
            weights = torch.cat((score, field.null(q)), -1).softmax(-1)[..., :-1] * valid * field.enabled
            slot_weight_sum.add_(weights.double().sum((0, 1)))
            slot_weight_count[0] += weights.shape[0] * weights.shape[1]
            query = q[:, :, None].expand_as(incoming)
            beta = diagonal_gain(field.memory_gate(torch.cat((query, normed), -1)))
            beta = (beta * valid[..., None] * field.enabled).squeeze(-1)[0].double()  # [T, slots]
            one_minus = (1.0 - beta).clamp_min(1e-12).log()
            # Track cumulative log survival at horizons 8, 32, 128, 960
            t0 = pos_cursor[0]
            for h_step in (8, 32, 128, L):
                if t0 < h_step <= t0 + beta.shape[0]:
                    upto = h_step - t0
                    snap = (log_surv[l_idx] + one_minus[:upto].sum(0)).exp().cpu().tolist()
                    surv_snapshots.setdefault(h_step, {})[l_idx] = snap
            log_surv[l_idx].add_(one_minus.sum(0))
            return out
        field.exchange = wrapped_exchange

    chunk = min(128, model.max_chunk)
    try:
        for start in range(0, L, chunk):
            end = min(start + chunk, L)
            pos_cursor[0] = start
            logits, state, _, h, _ = model.predict_observed(
                ids[:, start:end], state, logp[:, start:end])
            ce = F.cross_entropy(logits[0], targets[0, start:end], reduction='none')
            nll_chunks.append(ce.detach().cpu())
            feat_chunks.append(h.detach())
    finally:
        field.exchange = orig_exchange

    nll = torch.cat(nll_chunks)
    feats = torch.cat(feat_chunks, dim=0)
    extra = {}
    if collect_beta_and_weights:
        extra['slot_mean_read_weight'] = (slot_weight_sum / max(slot_weight_count[0] / n_layers, 1)).cpu().tolist()
        # Average survival across slots per layer, then summarize across layers
        surv_summary = {}
        for h_step, by_layer in sorted(surv_snapshots.items()):
            vals = [sum(by_layer[l]) / len(by_layer[l]) for l in sorted(by_layer)]
            surv_summary[str(h_step)] = dict(
                mean=float(np.mean(vals)), median=float(np.median(vals)),
                min=float(np.min(vals)), max=float(np.max(vals)),
                per_layer=[round(v, 6) for v in vals],
            )
        extra['workspace_survival_prod_1_minus_beta'] = surv_summary
    return nll, state, feats, extra


@torch.no_grad()
def run_stage1a(engine):
    print('[Stage 1A] Running workspace-state counterfactual probe...', flush=True)
    raw_all = np.fromfile(VAL_BIN, dtype=np.uint8)
    segs = [raw_all[s:s + 1922] for s in STARTS]  # W1: [0:961] (960 targets), W2: [960:1921] (960 targets)

    model = engine.model
    field = model.body.coevolution
    physical = model.body.base.layers[0].state_size
    init_rec = engine.recurrent.detach().clone()
    init_field = snapshot_field(field)
    old_qctx = getattr(model, '_question_context', None)
    model._question_context = None

    seq_results = []
    jump_results = []
    slot_records = []
    surv_all = []

    try:
        for mode, pairs in [
            ('sequential_960', [(i, segs[i][:961], segs[i][960:1921]) for i in range(4)]),
            ('topic_jump', [(i, segs[i][:961], segs[(i + 1) % 4][960:1921]) for i in range(4)]),
        ]:
            for idx, w1_bytes, w2_bytes in pairs:
                restore_field(field, init_field)
                # Refresh workset once before W1 from saved pending so W1 starts with active workset
                if model.memory.pending_count:
                    rec_w0, _, _ = apply_refresh_with_query(engine, init_rec, model.memory.pending[-1:])
                else:
                    rec_w0 = init_rec.clone()
                nll_w1, rec_after_w1, feats_w1, extra_w1 = run_window_readonly(
                    engine, w1_bytes, rec_w0, collect_beta_and_weights=(mode == 'sequential_960'))
                field_after_w1 = snapshot_field(field)
                ws_rms_w1 = float(rec_after_w1[..., physical:].square().mean().sqrt())

                # C1: Keep exact same 8 slots + KEEP carried workspace W_water
                restore_field(field, field_after_w1)
                nll_c1, _, _, extra_c1 = run_window_readonly(
                    engine, w2_bytes, rec_after_w1, collect_beta_and_weights=(mode == 'sequential_960'))
                if mode == 'sequential_960':
                    surv_all.append(extra_c1['workspace_survival_prod_1_minus_beta'])

                # C2: Keep exact same 8 slots + ZERO ONLY workspace W_water
                restore_field(field, field_after_w1)
                rec_zero_ws = rec_after_w1.clone()
                rec_zero_ws[..., physical:].zero_()
                nll_c2, _, _, _ = run_window_readonly(engine, w2_bytes, rec_zero_ws)

                # C3: Production refresh_workset (query = feats_w1[-1:])
                restore_field(field, field_after_w1)
                rec_c3, ret_c3, slots_c3 = apply_refresh_with_query(engine, rec_after_w1, feats_w1[-1:])
                nll_c3, _, _, _ = run_window_readonly(engine, w2_bytes, rec_c3)

                # C4: Window-mean refresh candidate (query = feats_w1[-32:].mean(0, keepdim=True))
                restore_field(field, field_after_w1)
                q_mean = feats_w1[-32:].mean(0, keepdim=True)
                rec_c4, ret_c4, slots_c4 = apply_refresh_with_query(engine, rec_after_w1, q_mean)
                nll_c4, _, _, _ = run_window_readonly(engine, w2_bytes, rec_c4)

                # C5: Disable PastPresentField completely (enabled = 0)
                restore_field(field, field_after_w1)
                field.enabled.zero_()
                nll_c5, _, _, _ = run_window_readonly(engine, w2_bytes, rec_after_w1)

                def bin_diff(a, b):
                    d = a - b
                    return dict(
                        b0_16=float(d[:16].mean()),
                        b16_64=float(d[16:64].mean()),
                        b64_256=float(d[64:256].mean()),
                        b256_960=float(d[256:].mean()),
                        full_960=float(d.mean()),
                    )

                rec_entry = dict(
                    segment=idx,
                    w1_mean_nll=float(nll_w1.mean()),
                    w1_end_workspace_rms=ws_rms_w1,
                    c1_keep_same_keep_ws=float(nll_c1.mean()),
                    c2_keep_same_zero_ws=float(nll_c2.mean()),
                    c3_prod_last_refresh=float(nll_c3.mean()),
                    c3_retained=ret_c3,
                    c4_mean32_refresh=float(nll_c4.mean()),
                    c4_retained=ret_c4,
                    c5_field_disabled=float(nll_c5.mean()),
                    # Positive delta means intervention HURT NLL (so what was removed was USEFUL)
                    pure_workspace_utility_c2_minus_c1=bin_diff(nll_c2, nll_c1),
                    prod_refresh_loss_c3_minus_c1=bin_diff(nll_c3, nll_c1),
                    mean32_vs_prod_c4_minus_c3=bin_diff(nll_c4, nll_c3),
                    total_field_utility_c5_minus_c1=bin_diff(nll_c5, nll_c1),
                )
                if mode == 'sequential_960':
                    seq_results.append(rec_entry)
                    # Per-slot decomposition on all 8 slots (32 total observations across 4 segs)
                    w1_weights = extra_w1['slot_mean_read_weight']
                    w2_weights = extra_c1['slot_mean_read_weight']
                    prefs = field_after_w1['preference'].cpu().tolist()
                    sids = field_after_w1['slots_id'].cpu().tolist()
                    for j in range(field.slots):
                        # Zero ONLY slot j's workspace (keep anchor & valid=True)
                        restore_field(field, field_after_w1)
                        rec_zj = rec_after_w1.clone()
                        ws_view = rec_zj[..., physical:].reshape(
                            len(model.body.base.layers), rec_zj.shape[1], field.slots, field.rank)
                        ws_view[:, :, j, :].zero_()
                        nll_zj, _, _, _ = run_window_readonly(engine, w2_bytes, rec_zj)
                        # Remove slot j completely (valid[j]=False & zero workspace)
                        restore_field(field, field_after_w1)
                        field.valid[j] = False
                        nll_rj, _, _, _ = run_window_readonly(engine, w2_bytes, rec_zj)
                        u_ws = float((nll_zj - nll_c1).mean())
                        u_tot = float((nll_rj - nll_c1).mean())
                        slot_records.append(dict(
                            segment=idx, slot_pos=j, memory_slot=int(sids[j]),
                            w1_read_weight=float(w1_weights[j]),
                            w2_read_weight=float(w2_weights[j]),
                            bounded_preference=float(prefs[j]),
                            utility_workspace_only=u_ws,
                            utility_total_fact_plus_ws=u_tot,
                            utility_fact_anchor_only=u_tot - u_ws,
                        ))
                else:
                    jump_results.append(rec_entry)
    finally:
        restore_field(field, init_field)
        model._question_context = old_qctx

    def agg(items, key):
        return dict(
            b0_16=float(np.mean([x[key]['b0_16'] for x in items])),
            b16_64=float(np.mean([x[key]['b16_64'] for x in items])),
            b64_256=float(np.mean([x[key]['b64_256'] for x in items])),
            b256_960=float(np.mean([x[key]['b256_960'] for x in items])),
            full_960=float(np.mean([x[key]['full_960'] for x in items])),
        )

    # Survival factor averaged across the 4 sequential segments
    surv_agg = {}
    for h_key in ('8', '32', '128', '960'):
        means = [s[h_key]['mean'] for s in surv_all]
        medians = [s[h_key]['median'] for s in surv_all]
        maxs = [s[h_key]['max'] for s in surv_all]
        surv_agg[h_key] = dict(mean_across_layers=float(np.mean(means)),
                               median_across_layers=float(np.mean(medians)),
                               max_layer=float(np.mean(maxs)))

    # Correlations across the 32 per-slot measurements
    ws_u = [r['utility_workspace_only'] for r in slot_records]
    tot_u = [r['utility_total_fact_plus_ws'] for r in slot_records]
    fact_u = [r['utility_fact_anchor_only'] for r in slot_records]
    w1_w = [r['w1_read_weight'] for r in slot_records]
    pref = [r['bounded_preference'] for r in slot_records]

    return dict(
        sequential_960_summary=dict(
            c1_keep_same_keep_ws=float(np.mean([x['c1_keep_same_keep_ws'] for x in seq_results])),
            c2_keep_same_zero_ws=float(np.mean([x['c2_keep_same_zero_ws'] for x in seq_results])),
            c3_prod_last_refresh=float(np.mean([x['c3_prod_last_refresh'] for x in seq_results])),
            c3_retained_mean=float(np.mean([x['c3_retained'] for x in seq_results])),
            c4_mean32_refresh=float(np.mean([x['c4_mean32_refresh'] for x in seq_results])),
            c4_retained_mean=float(np.mean([x['c4_retained'] for x in seq_results])),
            c5_field_disabled=float(np.mean([x['c5_field_disabled'] for x in seq_results])),
            pure_workspace_utility_c2_minus_c1=agg(seq_results, 'pure_workspace_utility_c2_minus_c1'),
            prod_refresh_loss_c3_minus_c1=agg(seq_results, 'prod_refresh_loss_c3_minus_c1'),
            mean32_vs_prod_c4_minus_c3=agg(seq_results, 'mean32_vs_prod_c4_minus_c3'),
            total_field_utility_c5_minus_c1=agg(seq_results, 'total_field_utility_c5_minus_c1'),
        ),
        topic_jump_summary=dict(
            c1_keep_same_keep_ws=float(np.mean([x['c1_keep_same_keep_ws'] for x in jump_results])),
            c2_keep_same_zero_ws=float(np.mean([x['c2_keep_same_zero_ws'] for x in jump_results])),
            c3_prod_last_refresh=float(np.mean([x['c3_prod_last_refresh'] for x in jump_results])),
            c3_retained_mean=float(np.mean([x['c3_retained'] for x in jump_results])),
            c4_mean32_refresh=float(np.mean([x['c4_mean32_refresh'] for x in jump_results])),
            c4_retained_mean=float(np.mean([x['c4_retained'] for x in jump_results])),
            c5_field_disabled=float(np.mean([x['c5_field_disabled'] for x in jump_results])),
            pure_workspace_utility_c2_minus_c1=agg(jump_results, 'pure_workspace_utility_c2_minus_c1'),
            prod_refresh_loss_c3_minus_c1=agg(jump_results, 'prod_refresh_loss_c3_minus_c1'),
            mean32_vs_prod_c4_minus_c3=agg(jump_results, 'mean32_vs_prod_c4_minus_c3'),
            total_field_utility_c5_minus_c1=agg(jump_results, 'total_field_utility_c5_minus_c1'),
        ),
        workspace_survival_prod_1_minus_beta=surv_agg,
        per_slot_decomposition_32obs=dict(
            mean_utility_workspace_only=float(np.mean(ws_u)),
            std_utility_workspace_only=float(np.std(ws_u)),
            positive_ws_utility_count=int(sum(v > 0 for v in ws_u)),
            mean_utility_fact_anchor_only=float(np.mean(fact_u)),
            mean_utility_total_fact_plus_ws=float(np.mean(tot_u)),
            positive_total_utility_count=int(sum(v > 0 for v in tot_u)),
            corr_w1_weight_vs_ws_utility=dict(pearson=pearson(w1_w, ws_u), spearman=spearman(w1_w, ws_u)),
            corr_w1_weight_vs_total_utility=dict(pearson=pearson(w1_w, tot_u), spearman=spearman(w1_w, tot_u)),
            corr_pref_vs_ws_utility=dict(pearson=pearson(pref, ws_u), spearman=spearman(pref, ws_u)),
            corr_pref_vs_total_utility=dict(pearson=pearson(pref, tot_u), spearman=spearman(pref, tot_u)),
            corr_fact_vs_ws_utility=dict(pearson=pearson(fact_u, ws_u), spearman=spearman(fact_u, ws_u)),
        ),
        sequential_per_segment=seq_results,
        topic_jump_per_segment=jump_results,
    )


def run_stage1b(engine):
    print('[Stage 1B] Running scalar log-response & curvature scan...', flush=True)
    from scripts.ib_local.causal_relation_program import mechanism, log_score
    from scripts.ib_local.navigation_live_witness import frozen_table_logp

    model = engine.model
    opt = engine.optimizer
    device = engine.device

    # --- 1B-1: relation_ridge on the exact 64 fixed old facts ---
    material = torch.load(MATERIAL, map_location=device, weights_only=False)
    orig_log_ridge = model.log_relation_ridge.detach().clone()

    def ridge_loss_fn(log_r_tensor):
        r = log_r_tensor.exp().clamp(1e-3, 10.0)
        nlls = []
        mses = []
        for state, fact in material:
            d = mechanism(model, state, fact.source, fact.action,
                          exclude_fact=fact.fact_id, ridge=r)
            target = model.encoder(fact.outcome)
            nlls.append(-log_score(d, target).mean())
            mses.append((target - d['mean']).square().mean())
        return torch.stack(nlls).mean(), torch.stack(mses).mean()

    # Temporarily bypass model.log_relation_ridge inside mechanism by passing explicit tensor
    saved_param = model._parameters.pop('log_relation_ridge')
    try:
        # Exact 1st and 2nd derivative w.r.t. log_ridge at current value
        x0 = orig_log_ridge.detach().clone().requires_grad_(True)
        loss0, mse0 = ridge_loss_fn(x0)
        g1, = torch.autograd.grad(loss0, x0, create_graph=True)
        g2, = torch.autograd.grad(g1, x0)
        ridge_grad = float(g1.detach())
        ridge_hess = float(g2.detach())

        ridge_grid = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        ridge_curve = []
        with torch.no_grad():
            for rv in ridge_grid:
                l_val, m_val = ridge_loss_fn(torch.tensor(math.log(rv), device=device, dtype=orig_log_ridge.dtype))
                ridge_curve.append(dict(ridge=rv, log_ridge=round(math.log(rv), 5),
                                        neg_log_score=float(l_val), mean_score=float(-l_val),
                                        mean_mse=float(m_val)))
    finally:
        model._parameters['log_relation_ridge'] = saved_param

    best_ridge_pt = min(ridge_curve, key=lambda d: d['neg_log_score'])
    st_r = opt.state[model.log_relation_ridge]
    b1, b2 = opt.param_groups[-1]['betas']
    eps = opt.param_groups[-1]['eps']
    step_r = int(st_r['step'])
    m_hat_r = float(st_r['exp_avg']) / (1 - b1 ** step_r)
    v_hat_r = float(st_r['exp_avg_sq']) / (1 - b2 ** step_r)
    adam_dir_r = m_hat_r / (math.sqrt(v_hat_r) + eps)
    gate_r = float(st_r['budget_gate'])

    # --- 1B-2: memory.log_temperature on real predict_observed CE + short_loss ---
    raw_all = np.fromfile(VAL_BIN, dtype=np.uint8)
    # Use 4 real 256-byte chunks from the 4 validation segments (1024 real bytes total)
    chunks = [raw_all[s:s + 257] for s in STARTS]
    orig_log_temp = model.memory.log_temperature.detach().clone()
    init_rec = engine.recurrent.detach().clone()
    old_qctx = getattr(model, '_question_context', None)
    model._question_context = None

    def temp_loss_fn(log_t_tensor, need_grad=False):
        ces, shorts, entropies = [], [], []
        # Temporarily bind log_temperature to the candidate tensor (without mutating saved param)
        model.memory._parameters.pop('log_temperature', None)
        model.memory.log_temperature = log_t_tensor
        ctx = torch.enable_grad() if need_grad else torch.no_grad()
        try:
            with ctx:
                for raw in chunks:
                    logp = torch.as_tensor(frozen_table_logp(engine.table, raw[:-1]), device=device)[None]
                    ids = torch.as_tensor(raw[:-1].astype(np.int64), device=device)[None]
                    targets = torch.as_tensor(raw[1:].astype(np.int64), device=device)[None]
                    st = init_rec.detach().clone()
                    # Run 2 chunks of 128 bytes
                    hs = []
                    ce_list = []
                    for start in range(0, len(raw) - 1, 128):
                        end = min(start + 128, len(raw) - 1)
                        logits, st, _, h, read = model.predict_observed(ids[:, start:end], st, logp[:, start:end])
                        ce_list.append(F.cross_entropy(logits[0], targets[0, start:end], reduction='mean'))
                        hs.append(h)
                        w = read.weights.detach().clamp_min(1e-12)
                        entropies.append(float(-(w * w.log()).sum(-1).mean()))
                    h_all = torch.cat(hs, dim=0)
                    short = model.memory.short
                    src = h_all[:-short]
                    rec = model.memory.read(src, model.encoder)
                    pred = model.predictor(torch.cat((model.encoder(src), rec.value), -1))
                    s_loss = F.mse_loss(pred, model.encoder(h_all[short:]))
                    ces.append(torch.stack(ce_list).mean())
                    shorts.append(s_loss)
        finally:
            if 'log_temperature' in model.memory.__dict__:
                del model.memory.__dict__['log_temperature']
            model.memory._parameters['log_temperature'] = saved_t_param
        ce_m = torch.stack(ces).mean()
        sh_m = torch.stack(shorts).mean()
        return ce_m + 0.1 * sh_m, ce_m, sh_m, float(np.mean(entropies))

    saved_t_param = model.memory._parameters['log_temperature']
    try:
        # Exact 1st derivative (and finite-diff 2nd derivative on GPU to avoid double-backward through fused GDN)
        xt0 = orig_log_temp.detach().clone().requires_grad_(True)
        tot0, ce0, sh0, ent0 = temp_loss_fn(xt0, need_grad=True)
        gt1, = torch.autograd.grad(tot0, xt0)
        temp_grad = float(gt1.detach())
        # Central finite difference for 2nd derivative of total loss w.r.t. log_temperature
        eps_fd = 0.05
        with torch.no_grad():
            tp, _, _, _ = temp_loss_fn(orig_log_temp + eps_fd, need_grad=False)
            tm, _, _, _ = temp_loss_fn(orig_log_temp - eps_fd, need_grad=False)
            t0_val, _, _, _ = temp_loss_fn(orig_log_temp, need_grad=False)
        temp_hess_fd = float((tp - 2 * t0_val + tm) / (eps_fd ** 2))

        temp_grid = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.0, 1.5, 2.0]
        temp_curve = []
        for tv in temp_grid:
            lt = torch.tensor(math.log(tv), device=device, dtype=orig_log_temp.dtype)
            tot_v, ce_v, sh_v, ent_v = temp_loss_fn(lt, need_grad=False)
            temp_curve.append(dict(temperature=tv, log_temperature=round(math.log(tv), 5),
                                   total_ce_plus_01short=float(tot_v), ce=float(ce_v),
                                   short_loss=float(sh_v), read_entropy=ent_v))
    finally:
        model._question_context = old_qctx

    best_temp_pt = min(temp_curve, key=lambda d: d['total_ce_plus_01short'])
    st_t = opt.state[model.memory.log_temperature]
    step_t = int(st_t['step'])
    m_hat_t = float(st_t['exp_avg']) / (1 - b1 ** step_t)
    v_hat_t = float(st_t['exp_avg_sq']) / (1 - b2 ** step_t)
    adam_dir_t = m_hat_t / (math.sqrt(v_hat_t) + eps)
    gate_t = float(st_t['budget_gate'])

    def lr_table(adam_dir, gate, cur_log, opt_log, updates_per_cycle):
        gap = opt_log - cur_log
        rows = []
        for lr in (3e-4, 1e-3, 3e-3, 5e-3, 1e-2, 2e-2):
            step_log = -lr * gate * adam_dir
            rel_pct = (math.exp(abs(step_log)) - 1.0) * 100.0
            cycles_est = (abs(gap) / (abs(step_log) * updates_per_cycle)) if abs(step_log) > 1e-12 else None
            rows.append(dict(lr=lr, log_step_per_update=step_log, rel_change_pct_per_update=rel_pct,
                             est_cycles_to_grid_optimum=None if cycles_est is None else round(cycles_est, 1)))
        return rows

    return dict(
        relation_ridge=dict(
            current_ridge=float(orig_log_ridge.exp()),
            current_log_ridge=float(orig_log_ridge),
            grad_wrt_log_ridge=ridge_grad,
            hess_wrt_log_ridge=ridge_hess,
            newton_log_step=(-ridge_grad / ridge_hess) if abs(ridge_hess) > 1e-12 else None,
            saved_adam_updates=step_r,
            saved_adam_dir=adam_dir_r,
            saved_wiener_gate=gate_r,
            grid_optimum=best_ridge_pt,
            delta_log_to_grid_optimum=best_ridge_pt['log_ridge'] - float(orig_log_ridge),
            gain_at_grid_optimum_vs_01=float(next(d['neg_log_score'] for d in ridge_curve if d['ridge'] == 0.10) - best_ridge_pt['neg_log_score']),
            candidate_lr_table=lr_table(adam_dir_r, gate_r, float(orig_log_ridge), best_ridge_pt['log_ridge'], 1.0),
            curve=ridge_curve,
        ),
        memory_temperature=dict(
            current_temperature=float(orig_log_temp.exp()),
            current_log_temperature=float(orig_log_temp),
            grad_wrt_log_temperature=temp_grad,
            hess_fd_wrt_log_temperature=temp_hess_fd,
            newton_log_step=(-temp_grad / temp_hess_fd) if abs(temp_hess_fd) > 1e-12 else None,
            saved_adam_updates=step_t,
            saved_adam_dir=adam_dir_t,
            saved_wiener_gate=gate_t,
            grid_optimum=best_temp_pt,
            delta_log_to_grid_optimum=best_temp_pt['log_temperature'] - float(orig_log_temp),
            gain_at_grid_optimum_vs_02=float(next(d['total_ce_plus_01short'] for d in temp_curve if d['temperature'] == 0.20) - best_temp_pt['total_ce_plus_01short']),
            candidate_lr_table=lr_table(adam_dir_t, gate_t, float(orig_log_temp), best_temp_pt['log_temperature'], 2.0),
            curve=temp_curve,
        ),
    )


def main():
    t0 = time.perf_counter()
    sha_before = sha256_file(CKPT)
    loaded = load_body()
    engine = loaded.engine
    param_versions_before = [p._version for p in engine.model.parameters()]
    try:
        stage1a = run_stage1a(engine)
        stage1b = run_stage1b(engine)
        param_versions_after = [p._version for p in engine.model.parameters()]
        assert param_versions_before == param_versions_after, 'read-only probe mutated parameters!'
    finally:
        if hasattr(engine.model, '_causal_archive') and hasattr(engine.model._causal_archive, 'db'):
            engine.model._causal_archive.db.close()
        if TMP_WORK.exists():
            shutil.rmtree(TMP_WORK, ignore_errors=True)
    sha_after = sha256_file(CKPT)
    assert sha_before == sha_after, 'checkpoint sha256 changed!'

    report = dict(
        schema=1,
        source_checkpoint=str(CKPT),
        source_sha256=sha_before,
        source_cycle=2845,
        optimizer_updates=0,
        world_calls=0,
        parameters_unchanged=True,
        elapsed_seconds=round(time.perf_counter() - t0, 2),
        stage1a_workspace_counterfactual=stage1a,
        stage1b_scalar_response=stage1b,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(f'\n[DONE in {report["elapsed_seconds"]}s] Saved to {OUT_PATH}', flush=True)
    print('\n=== Stage 1A Summary (Sequential +960B) ===')
    print(json.dumps(stage1a['sequential_960_summary'], indent=2))
    print('\n=== Stage 1A Survival prod(1-beta) ===')
    print(json.dumps(stage1a['workspace_survival_prod_1_minus_beta'], indent=2))
    print('\n=== Stage 1A Per-Slot Decomposition (32 obs) ===')
    print(json.dumps(stage1a['per_slot_decomposition_32obs'], indent=2))
    print('\n=== Stage 1B Summary ===')
    for k in ('relation_ridge', 'memory_temperature'):
        d = stage1b[k]
        print(k, {kk: d[kk] for kk in d if kk != 'curve'})


if __name__ == '__main__':
    main()
