"""arXiv:2609.04963 ('Fractal basins trap latent reasoning') on MengLi 2813 — v2.

Fixes the two v1 probe artifacts:
  1. Uses the REAL thought manifold 2D plane spanned by 3 farthest real FineWeb
     validation contexts (Z_0, Z_A, Z_B) instead of orthogonal 262,144D white noise;
  2. Uses MengLi's REAL stable symplectic step dt=0.01 (zero NaN blowup) and
     neutral table_logp=-log(256) so closed-loop GDN + Water Reservoir runs free
     without being pinned to a single frozen fast-table byte.

Outputs 3 high-resolution figures + JSON report in ~2-3 minutes (read-only, 0 updates).
"""
from __future__ import annotations

import gc
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('CGB_CHECKPOINT_SPOOL', 'E:/mengli-runtime-staging-20261123')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F


def basin_entropy(labels: np.ndarray, box: int = 4):
    h, w = labels.shape
    nh, nw = h // box, w // box
    entropies, boundary_entropies = [], []
    for i in range(nh):
        for j in range(nw):
            patch = labels[i*box:(i+1)*box, j*box:(j+1)*box].ravel()
            _, counts = np.unique(patch, return_counts=True)
            probs = counts / counts.sum()
            ent = float(-(probs * np.log(probs + 1e-15)).sum())
            entropies.append(ent)
            if len(counts) > 1:
                boundary_entropies.append(ent)
    return dict(
        S_b=float(np.mean(entropies)) if entropies else 0.0,
        S_bb=float(np.mean(boundary_entropies)) if boundary_entropies else 0.0,
        boundary_box_fraction=float(len(boundary_entropies) / max(1, len(entropies))),
        unique_basins=int(len(np.unique(labels))),
        box_size=box,
    )


def uncertainty_exponent(labels: np.ndarray, epsilons=(1, 2, 4, 6)):
    h, w = labels.shape
    fracs, valid_eps = [], []
    for eps in epsilons:
        if eps >= min(h, w) // 2:
            continue
        dh = (labels[:, :-eps] != labels[:, eps:])
        dv = (labels[:-eps, :] != labels[eps:, :])
        uncertain = 0.5 * (float(dh.mean()) + float(dv.mean()))
        if uncertain > 0:
            valid_eps.append(eps)
            fracs.append(uncertain)
    if len(valid_eps) < 2:
        return dict(alpha=None, fractal_dimension=None, r2=None,
                    epsilons=valid_eps, uncertain_fractions=fracs)
    lx = np.log(np.asarray(valid_eps, dtype=np.float64))
    ly = np.log(np.asarray(fracs, dtype=np.float64))
    slope, _ = np.polyfit(lx, ly, 1)
    alpha = float(np.clip(slope, 0.0, 2.0))
    r2 = float(np.corrcoef(lx, ly)[0, 1] ** 2)
    return dict(
        alpha=alpha,
        fractal_dimension=float(2.0 - alpha),
        r2=r2,
        epsilons=list(valid_eps),
        uncertain_fractions=[float(x) for x in fracs],
    )


def chunked_ham_quiet(s, p, theta, context, dt, frequency_scale, impedance, *, layer_chunk=4):
    from scripts.ib_local.adaptive_hamiltonian import quiet as ham_quiet
    layers = s.shape[0]
    ns_list, np_list = [], []
    for start in range(0, layers, layer_chunk):
        end = min(start + layer_chunk, layers)
        imp = None if impedance is None else impedance[start:end]
        fs = None if frequency_scale is None else frequency_scale[start:end]
        ns, np_ = ham_quiet(s[start:end], p[start:end], theta[start:end], context[start:end], dt, fs, imp)
        ns_list.append(ns.detach())
        np_list.append(np_.detach())
    return torch.cat(ns_list, dim=0), torch.cat(np_list, dim=0)


@torch.no_grad()
def relation_slot_ids(model, h_batch, rel_keys, rel_ids):
    z = F.normalize(model.encoder(h_batch), dim=-1)
    idx = (z @ rel_keys.T).argmax(-1)
    return rel_ids[idx].cpu().numpy()


@torch.no_grad()
def batched_gdn_step(model, ids, state, table_logp):
    from scripts.ib_local.native_replay_batch import replay_scope
    b = ids.shape[0]
    with replay_scope(model, [(-1, -1)] * b, length=ids.shape[1]):
        logits, next_state, _, h, _ = model.predict_observed(ids, state, table_logp)
    return logits, next_state, h


def sweep_gdn_plane(m, base_rec, d1_rec, d2_rec, start_byte, neutral_logp,
                    rel_keys, rel_ids, *, grid=28, steps=16, lo=-0.4, hi=1.4, batch=28,
                    only_col_slice=None):
    device = base_rec.device
    L = base_rec.shape[0]
    gv = np.linspace(lo, hi, grid, dtype=np.float32)
    ug, vg = np.meshgrid(gv, gv)
    coords = np.stack([ug.ravel(), vg.ravel()], axis=-1)
    total = len(coords)

    out_rel = np.zeros(total, dtype=np.int64)
    out_byte = np.zeros(total, dtype=np.int64)
    out_settle = np.zeros(total, dtype=np.int32)
    out_flips = np.zeros(total, dtype=np.int32)
    out_final_h = np.zeros((total, 256), dtype=np.float32)

    for start in range(0, total, batch):
        end = min(start + batch, total)
        b = end - start
        u_t = torch.as_tensor(coords[start:end, 0], device=device).reshape(1, b, 1)
        v_t = torch.as_tensor(coords[start:end, 1], device=device).reshape(1, b, 1)
        rec_b = base_rec.expand(L, b, -1).clone()
        if only_col_slice is None:
            rec_b = rec_b + u_t * d1_rec + v_t * d2_rec
        else:
            c0, c1 = only_col_slice
            rec_b[..., c0:c1] = rec_b[..., c0:c1] + u_t * d1_rec[..., c0:c1] + v_t * d2_rec[..., c0:c1]

        ids_b = torch.full((b, 1), start_byte, device=device, dtype=torch.long)
        lp_b = neutral_logp.expand(b, 1, 256)
        prev_rel, prev_h = None, None
        flips = np.zeros(b, dtype=np.int32)
        last_change = np.zeros(b, dtype=np.int32)

        for step_t in range(1, steps + 1):
            logits_b, rec_b, h_b = batched_gdn_step(m, ids_b, rec_b, lp_b)
            next_ids = logits_b[:, -1].argmax(-1, keepdim=True)
            rel_now = relation_slot_ids(m, h_b, rel_keys, rel_ids)
            if prev_rel is not None:
                changed = (rel_now != prev_rel)
                flips += changed.astype(np.int32)
                last_change = np.where(changed, step_t, last_change)
            if prev_h is not None:
                vel = ((h_b - prev_h).norm(dim=-1) / prev_h.norm(dim=-1).clamp_min(1e-6)).cpu().numpy()
                last_change = np.where(vel > 0.12, step_t, last_change)
            prev_rel = rel_now
            prev_h = h_b.detach()
            ids_b = next_ids

        out_rel[start:end] = prev_rel
        out_byte[start:end] = ids_b[:, 0].cpu().numpy()
        out_settle[start:end] = last_change
        out_flips[start:end] = flips
        out_final_h[start:end] = prev_h.cpu().numpy()

    rel_2d = out_rel.reshape(grid, grid)
    byte_2d = out_byte.reshape(grid, grid)
    settle_2d = out_settle.reshape(grid, grid)
    flips_2d = out_flips.reshape(grid, grid)
    h_2d = out_final_h.reshape(grid, grid, -1)
    dx = np.linalg.norm(np.diff(h_2d, axis=1, append=h_2d[:, -1:]), axis=-1)
    dy = np.linalg.norm(np.diff(h_2d, axis=0, append=h_2d[-1:, :]), axis=-1)
    du = float((hi - lo) / (grid - 1))
    sens_2d = np.log1p(np.sqrt(dx**2 + dy**2) / max(du, 1e-6))

    metrics = dict(
        grid=grid, steps=steps, range=[lo, hi],
        relation_basin_entropy=basin_entropy(rel_2d, box=4),
        relation_uncertainty=uncertainty_exponent(rel_2d, epsilons=(1, 2, 4, 6)),
        byte_basin_entropy=basin_entropy(byte_2d, box=4),
        byte_uncertainty=uncertainty_exponent(byte_2d, epsilons=(1, 2, 4, 6)),
        settle_mean=float(settle_2d.mean()), settle_max=int(settle_2d.max()),
        flips_mean=float(flips_2d.mean()), flips_max=int(flips_2d.max()),
        sensitivity_p50=float(np.median(sens_2d)),
        sensitivity_p95=float(np.percentile(sens_2d, 95)),
        sensitivity_max=float(sens_2d.max()),
    )
    return dict(rel=rel_2d, byte=byte_2d, settle=settle_2d, flips=flips_2d,
                sens=sens_2d, metrics=metrics)


def run_probe():
    t_all = time.perf_counter()
    ckpt = Path('E:/mengli-native-supply-completed-20261122/original/last.pt.gz')
    fig_dir = Path('results/published/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    art_dir = Path('C:/Users/85750/.gemini/antigravity/brain/b52ce350-543f-4727-bcda-4a922c34066a')

    import scripts.ib_local.run_mother_continuation as cont
    from scripts.ib_local.adaptive_hamiltonian import invariants
    from scripts.ib_local.navigation_live_witness import frozen_table_logp

    tmp = Path(tempfile.mkdtemp(prefix='probe2609_v2_', dir='E:/mengli-runtime-staging-20261123'))
    loaded = None
    try:
        opts = cont.continuation_parser().parse_args([
            '--checkpoint', str(ckpt), '--output', str(tmp / 'body'), '--cycles', '1',
            '--world-source', 'text', '--autonomous-events', '--optimizer-policy', 'budgeted',
            '--table-storage', str(tmp / 'table'), '--causal-storage', str(tmp / 'causal'),
        ])
        orig_save, orig_export = cont.save, cont.export_live
        cont.save = lambda *a, **kw: 'probe-only'
        cont.export_live = lambda *a, **kw: None
        try:
            loaded = cont.prepare_continuation(opts, resume_output=False)
        finally:
            cont.save, cont.export_live = orig_save, orig_export

        e = loaded.engine
        m = e.model
        m._question_context = None
        device = e.device
        base = m.body.base
        aq = base._adaptive_quiet
        coev = m.body.coevolution
        mat_size = base.layers[0].matrix_size
        w_size = coev.state_size
        L = len(base.layers)

        bank = m.causal_relations
        live_mask = bank.identity > 0
        with torch.no_grad():
            rel_keys = F.normalize(m.encoder(bank.condition[live_mask]), dim=-1)
            rel_ids = bank.identity[live_mask]

        rec_origin = e.recurrent.detach().clone()
        p0 = torch.stack([t.detach().clone() for t in e.bfss_momenta])
        theta = aq.theta.detach()
        freq_scale = aq.frequency_scale
        impedance = aq.impedance if aq.spectrum_enabled else None
        dt_real = float(e.bfss_dt)  # EXACT production dt=0.01 (zero NaN!)
        neutral_logp = torch.full((1, 1, 256), -math.log(256.0), device=device, dtype=torch.float32)

        # ---------------------------------------------------------------------
        # Collect 16 REAL living context states from FineWeb validation stream
        # ---------------------------------------------------------------------
        manifest = loaded.config['natural_text_corpus']
        val_raw = Path(manifest['validation_path']).read_bytes()
        starts = np.linspace(0, len(val_raw) - 257, 16, dtype=np.int64)
        ctx_recs, ctx_hs, ctx_nlls, ctx_lastb, ctx_rels, ctx_snips = [], [], [], [], [], []
        for st_idx in starts:
            seg = np.frombuffer(val_raw[st_idx:st_idx+257], np.uint8).copy()
            lp = torch.as_tensor(frozen_table_logp(e.table, seg[:-1]), device=device)[None]
            ids = torch.as_tensor(seg[:-1].astype(np.int64), device=device)[None]
            tgt = torch.as_tensor(seg[1:].astype(np.int64), device=device)[None]
            st_cur = rec_origin.clone()
            losses = []
            with torch.no_grad():
                for pos in range(0, 256, 128):
                    lg, st_cur, _, h_chunk, _ = m.predict_observed(ids[:, pos:pos+128], st_cur, lp[:, pos:pos+128])
                    losses.append(float(F.cross_entropy(lg[0], tgt[0, pos:pos+128])))
            h_end = h_chunk[-1:].detach()
            ctx_recs.append(st_cur.detach().clone())
            ctx_hs.append(h_end.cpu().numpy()[0])
            ctx_nlls.append(float(np.mean(losses)))
            ctx_lastb.append(int(seg[-1]))
            ctx_rels.append(int(relation_slot_ids(m, h_end, rel_keys, rel_ids)[0]))
            ctx_snips.append(bytes(seg[-40:]).decode('utf-8', errors='replace'))

        H_mat = np.stack(ctx_hs)
        # Farthest-point selection of 3 maximally distinct real thought states (Z_0, Z_A, Z_B)
        i0 = int(np.argmin(ctx_nlls))  # calm anchor
        d_from_0 = np.linalg.norm(H_mat - H_mat[i0], axis=-1)
        iA = int(np.argmax(d_from_0))
        d_min_0A = np.minimum(d_from_0, np.linalg.norm(H_mat - H_mat[iA], axis=-1))
        iB = int(np.argmax(d_min_0A))
        calm_idx, hard_idx = int(np.argmin(ctx_nlls)), int(np.argmax(ctx_nlls))

        Z0, ZA, ZB = ctx_recs[i0], ctx_recs[iA], ctx_recs[iB]
        D1, D2 = ZA - Z0, ZB - Z0
        start_byte = ord(' ')  # neutral space boundary byte
        print(f'3 Real Thought Anchors: Z0(idx={i0}, rel={ctx_rels[i0]}, nll={ctx_nlls[i0]:.2f}), '
              f'ZA(idx={iA}, rel={ctx_rels[iA]}, nll={ctx_nlls[iA]:.2f}), '
              f'ZB(idx={iB}, rel={ctx_rels[iB]}, nll={ctx_nlls[iB]:.2f})', flush=True)

        # =====================================================================
        # PROBE 1: Quiet Hamiltonian (S, P) on REAL Thought Plane (dt=0.01!)
        # =====================================================================
        t1 = time.perf_counter()
        N_ham, T_ham, lo, hi = 28, 20, -0.4, 1.4
        s0 = Z0[..., :mat_size].reshape(L, 1, 8, 32, 32)
        ds1 = D1[..., :mat_size].reshape(L, 1, 8, 32, 32)
        ds2 = D2[..., :mat_size].reshape(L, 1, 8, 32, 32)
        corr, _ = invariants(s0)
        ctx_ham = corr.mean(-1)[..., None, None]

        gen = torch.Generator(device=device).manual_seed(260904963)
        tan_s = torch.randn(s0.numel(), generator=gen, device=device)
        tan_s = (tan_s / tan_s.norm()).reshape(L, 1, 8, 32, 32)
        eps_fli = 1e-3

        gv = np.linspace(lo, hi, N_ham, dtype=np.float32)
        uu, vv = np.meshgrid(gv, gv)
        coords = np.stack([uu.ravel(), vv.ravel()], axis=-1)
        total_ham = len(coords)

        ham_rel = np.zeros(total_ham, dtype=np.int64)
        ham_byte = np.zeros(total_ham, dtype=np.int64)
        ham_fli = np.zeros(total_ham, dtype=np.float32)
        ham_changed = np.zeros(total_ham, dtype=np.int32)

        B_ham = 14
        ids_one = torch.full((B_ham, 1), start_byte, device=device, dtype=torch.long)
        lp_one = neutral_logp.expand(B_ham, 1, 256)

        for start in range(0, total_ham, B_ham):
            end = min(start + B_ham, total_ham)
            b = end - start
            u_t = torch.as_tensor(coords[start:end, 0], device=device).reshape(1, b, 1)
            v_t = torch.as_tensor(coords[start:end, 1], device=device).reshape(1, b, 1)
            rec_init = Z0.expand(L, b, -1).clone() + u_t * D1 + v_t * D2
            # Read t=0 relation
            _, _, h_init = batched_gdn_step(m, ids_one[:b], rec_init, lp_one[:b])
            rel_0 = relation_slot_ids(m, h_init, rel_keys, rel_ids)

            sb = rec_init[..., :mat_size].reshape(L, b, 8, 32, 32)
            pb = p0.expand(L, b, 8, 32, 32).contiguous()
            s_both = torch.cat([sb, sb + eps_fli * tan_s], dim=1)
            p_both = torch.cat([pb, pb], dim=1)
            max_fli = torch.full((b,), -10.0, device=device)

            for _ in range(T_ham):
                s_both, p_both = chunked_ham_quiet(
                    s_both, p_both, theta, ctx_ham, dt_real, freq_scale, impedance, layer_chunk=4
                )
                ds = s_both[:, b:] - s_both[:, :b]
                dp = p_both[:, b:] - p_both[:, :b]
                dist = torch.sqrt(ds.square().sum(dim=(0, 2, 3, 4)) + dp.square().sum(dim=(0, 2, 3, 4))).clamp_min(1e-12)
                max_fli = torch.maximum(max_fli, torch.log(dist / eps_fli))

            rec_end = rec_init.clone()
            rec_end[..., :mat_size] = s_both[:, :b].flatten(2)
            lg_end, _, h_end = batched_gdn_step(m, ids_one[:b], rec_end, lp_one[:b])
            rel_T = relation_slot_ids(m, h_end, rel_keys, rel_ids)
            ham_rel[start:end] = rel_T
            ham_byte[start:end] = lg_end[:, -1].argmax(-1).cpu().numpy()
            ham_fli[start:end] = max_fli.cpu().numpy()
            ham_changed[start:end] = (rel_T != rel_0).astype(np.int32)

        ham_rel_2d = ham_rel.reshape(N_ham, N_ham)
        ham_byte_2d = ham_byte.reshape(N_ham, N_ham)
        ham_fli_2d = ham_fli.reshape(N_ham, N_ham)
        ham_changed_2d = ham_changed.reshape(N_ham, N_ham)
        ham_metrics = dict(
            grid=N_ham, steps=T_ham, dt=dt_real, range=[lo, hi],
            seconds=round(time.perf_counter() - t1, 2),
            relation_basin_entropy=basin_entropy(ham_rel_2d, box=4),
            relation_uncertainty=uncertainty_exponent(ham_rel_2d),
            byte_basin_entropy=basin_entropy(ham_byte_2d, box=4),
            byte_uncertainty=uncertainty_exponent(ham_byte_2d),
            fli_stats=dict(min=float(ham_fli_2d.min()), median=float(np.median(ham_fli_2d)),
                           mean=float(ham_fli_2d.mean()), p95=float(np.percentile(ham_fli_2d, 95)),
                           max=float(ham_fli_2d.max())),
            quiet_relation_flip_fraction=float(ham_changed_2d.mean()),
        )
        print('PROBE 1 (Symplectic Quiet, dt=0.01) DONE in', ham_metrics['seconds'], 's:',
              'Rel basins=', ham_metrics['relation_basin_entropy']['unique_basins'],
              'Rel S_b=', round(ham_metrics['relation_basin_entropy']['S_b'], 3),
              'alpha=', ham_metrics['relation_uncertainty']['alpha'],
              'Byte basins=', ham_metrics['byte_basin_entropy']['unique_basins'],
              'FLI med/max=', round(ham_metrics['fli_stats']['median'], 3), round(ham_metrics['fli_stats']['max'], 3),
              flush=True)

        # =====================================================================
        # PROBE 2A & 2B: Closed-Loop 32L GDN + Water ON vs OFF (Neutral logp!)
        # =====================================================================
        t2 = time.perf_counter()
        coev.enabled.fill_(1.0)
        gdn_on = sweep_gdn_plane(m, Z0, D1, D2, start_byte, neutral_logp, rel_keys, rel_ids,
                                 grid=28, steps=16, lo=lo, hi=hi, batch=28)
        gdn_on['metrics']['seconds'] = round(time.perf_counter() - t2, 2)
        print('PROBE 2A (GDN + Water ON) DONE in', gdn_on['metrics']['seconds'], 's:',
              'Rel basins=', gdn_on['metrics']['relation_basin_entropy']['unique_basins'],
              'Rel S_b=', round(gdn_on['metrics']['relation_basin_entropy']['S_b'], 3),
              'Rel alpha=', gdn_on['metrics']['relation_uncertainty']['alpha'],
              'Byte basins=', gdn_on['metrics']['byte_basin_entropy']['unique_basins'],
              'Byte S_b=', round(gdn_on['metrics']['byte_basin_entropy']['S_b'], 3),
              'Byte alpha=', gdn_on['metrics']['byte_uncertainty']['alpha'], flush=True)

        t2b = time.perf_counter()
        coev.enabled.fill_(0.0)
        try:
            gdn_off = sweep_gdn_plane(m, Z0, D1, D2, start_byte, neutral_logp, rel_keys, rel_ids,
                                      grid=28, steps=16, lo=lo, hi=hi, batch=28)
        finally:
            coev.enabled.fill_(1.0)
        gdn_off['metrics']['seconds'] = round(time.perf_counter() - t2b, 2)
        water_rel_flip_frac = float((gdn_on['rel'] != gdn_off['rel']).mean())
        water_byte_flip_frac = float((gdn_on['byte'] != gdn_off['byte']).mean())
        print('PROBE 2B (GDN + Water OFF) DONE in', gdn_off['metrics']['seconds'], 's:',
              'Water ablation flipped Rel:', f'{water_rel_flip_frac*100:.1f}%',
              'Byte:', f'{water_byte_flip_frac*100:.1f}%', flush=True)

        # =====================================================================
        # PROBE 2C: Pure Water-Reservoir Slice at Triple Boundary (S FIXED!)
        # =====================================================================
        t2c = time.perf_counter()
        # Anchor S right at the centroid (u=0.33, v=0.33) between all 3 thought attractors!
        Z_mid = Z0 + 0.33 * D1 + 0.33 * D2
        water_only = sweep_gdn_plane(m, Z_mid, D1, D2, start_byte, neutral_logp, rel_keys, rel_ids,
                                     grid=28, steps=16, lo=-2.0, hi=2.0, batch=28,
                                     only_col_slice=(rec_origin.shape[-1] - w_size, rec_origin.shape[-1]))
        water_only['metrics']['seconds'] = round(time.perf_counter() - t2c, 2)
        print('PROBE 2C (PURE Water Reservoir Slice, S FIXED at boundary!) DONE in', water_only['metrics']['seconds'], 's:',
              'Rel basins=', water_only['metrics']['relation_basin_entropy']['unique_basins'],
              'Rel S_b=', round(water_only['metrics']['relation_basin_entropy']['S_b'], 3),
              'Byte basins=', water_only['metrics']['byte_basin_entropy']['unique_basins'],
              'Byte S_b=', round(water_only['metrics']['byte_basin_entropy']['S_b'], 3),
              'Byte alpha=', water_only['metrics']['byte_uncertainty']['alpha'], flush=True)

        # =====================================================================
        # PROBE 3: Live Inference Trajectories (Calm vs Surprising, dt=0.01!)
        # =====================================================================
        T_traj = 48
        trajectories = {}
        for label_name, c_idx in (('calm', calm_idx), ('surprising', hard_idx)):
            st_init = ctx_recs[c_idx]
            last_b = ctx_lastb[c_idx]
            tan_rec = torch.randn_like(st_init)
            tan_rec = tan_rec / tan_rec.norm()
            eps_tr = 1e-3
            rec_pair = torch.cat([st_init.clone(), st_init.clone() + eps_tr * tan_rec], dim=1)
            ids_c = torch.full((2, 1), last_b, device=device, dtype=torch.long)
            lp_c = neutral_logp.expand(2, 1, 256)

            vel_list, wvel_list, ent_list, rel_list, lyap_list, h_list = [], [], [], [], [], []
            prev_h, prev_w = None, None
            for _ in range(T_traj):
                lg, rec_pair, h_pair = batched_gdn_step(m, ids_c, rec_pair, lp_c)
                h0 = h_pair[0]
                w0_cur = rec_pair[:, 0, -w_size:]
                probs = lg[0, -1].softmax(-1)
                ent = float(-(probs * (probs + 1e-12).log()).sum())
                rel_id = int(relation_slot_ids(m, h0[None], rel_keys, rel_ids)[0])
                b_next = int(lg[0, -1].argmax(-1))
                vel = 0.0 if prev_h is None else float((h0 - prev_h).norm() / prev_h.norm().clamp_min(1e-6))
                wvel = 0.0 if prev_w is None else float((w0_cur - prev_w).norm() / prev_w.norm().clamp_min(1e-6))
                prev_h, prev_w = h0.detach().clone(), w0_cur.detach().clone()
                delta_s = rec_pair[:, 1:2] - rec_pair[:, 0:1]
                dn = float(delta_s.norm())
                lyap = math.log(max(dn, 1e-12) / eps_tr)
                rec_pair[:, 1:2] = rec_pair[:, 0:1] + (eps_tr / max(dn, 1e-12)) * delta_s
                ids_c = torch.full((2, 1), b_next, device=device, dtype=torch.long)
                vel_list.append(vel); wvel_list.append(wvel); ent_list.append(ent)
                rel_list.append(rel_id); lyap_list.append(lyap); h_list.append(h0.cpu().numpy())

            # Quiet Hamiltonian at REAL dt=0.01
            s_q = st_init[..., :mat_size].reshape(L, 1, 8, 32, 32)
            corr_q, _ = invariants(s_q)
            ctx_q = corr_q.mean(-1)[..., None, None]
            s_pair = torch.cat([s_q, s_q + eps_fli * tan_s], dim=1)
            p_pair = torch.cat([p0.clone(), p0.clone()], dim=1)
            ham_lyap, ham_rels, ham_vel = [], [], []
            prev_hq = None
            for _ in range(T_traj):
                s_pair, p_pair = chunked_ham_quiet(
                    s_pair, p_pair, theta, ctx_q, dt_real, freq_scale, impedance, layer_chunk=4
                )
                ds = s_pair[:, 1:2] - s_pair[:, 0:1]
                dp = p_pair[:, 1:2] - p_pair[:, 0:1]
                dn = float(torch.sqrt(ds.square().sum() + dp.square().sum()))
                ham_lyap.append(math.log(max(dn, 1e-12) / eps_fli))
                sc = eps_fli / max(dn, 1e-12)
                s_pair[:, 1:2] = s_pair[:, 0:1] + sc * ds
                p_pair[:, 1:2] = p_pair[:, 0:1] + sc * dp
                rec_q = st_init.clone()
                rec_q[..., :mat_size] = s_pair[:, 0:1].flatten(2)
                _, _, hq = batched_gdn_step(m, torch.full((1, 1), last_b, device=device, dtype=torch.long), rec_q, lp_c[:1])
                ham_rels.append(int(relation_slot_ids(m, hq, rel_keys, rel_ids)[0]))
                ham_vel.append(0.0 if prev_hq is None else float((hq[0] - prev_hq).norm() / prev_hq.norm().clamp_min(1e-6)))
                prev_hq = hq[0].detach().clone()

            trajectories[label_name] = dict(
                context_nll=ctx_nlls[c_idx], snippet=ctx_snips[c_idx],
                gdn_velocity=vel_list, water_velocity=wvel_list, gdn_entropy=ent_list,
                gdn_relations=rel_list,
                gdn_relation_flips=int(sum(rel_list[i] != rel_list[i-1] for i in range(1, len(rel_list)))),
                gdn_local_lyapunov=lyap_list, gdn_mean_lyapunov=float(np.mean(lyap_list)),
                gdn_h=np.stack(h_list),
                ham_local_lyapunov=ham_lyap, ham_mean_lyapunov=float(np.mean(ham_lyap)),
                ham_relations=ham_rels,
                ham_relation_flips=int(sum(ham_rels[i] != ham_rels[i-1] for i in range(1, len(ham_rels)))),
                ham_velocity=ham_vel,
            )

        # Grand closed-loop telemetry from strict8 vs phase8
        grand_loop = {}
        for arm in ('strict8_baseline', 'phase8_rhythm'):
            ev_path = Path(f'D:/mengli-throughput-20261123/{arm}/events.jsonl')
            if not ev_path.exists():
                continue
            rows = []
            for line in ev_path.read_text(encoding='utf8').splitlines():
                r = json.loads(line)
                if r.get('autonomous_event', {}).get('kind') == 'external':
                    pp = r.get('past_present', {})
                    rows.append(dict(
                        cycle=r.get('cycle'), choice=r.get('choice'),
                        response_nll=r.get('response_nll'),
                        workspace_rms=pp.get('workspace_rms'),
                        workset_retained=pp.get('retained'),
                        unknown_id=r.get('shared_unknown', {}).get('identity'),
                        world_cursor=r.get('world', {}).get('cursor'),
                    ))
            grand_loop[arm] = rows

        # =====================================================================
        # RENDER ALL 3 PUBLICATION FIGURES
        # =====================================================================
        plt.rcParams.update({
            'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
            'axes.edgecolor': '#30363d', 'axes.labelcolor': '#e6edf3',
            'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
            'text.color': '#e6edf3', 'font.size': 9.5,
        })

        def remap_ids(arr):
            uniq, inv = np.unique(arr, return_inverse=True)
            return inv.reshape(arr.shape), len(uniq)

        ext = [lo, hi, lo, hi]

        # --- FIGURE 1: Symplectic Quiet (dt=0.01) vs Dissipative Closed-Loop GDN+Water ---
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.6), dpi=160)
        fig.suptitle(
            f"MengLi 2813 Thought-Manifold 2D Slice (Z0, ZA, ZB) — arXiv:2609.04963 Basins\n"
            f"Symplectic Quiet (dt=0.01): Rel={ham_metrics['relation_basin_entropy']['unique_basins']} "
            f"(S_b={ham_metrics['relation_basin_entropy']['S_b']:.3f}, D_b={ham_metrics['relation_uncertainty']['fractal_dimension'] or 0:.2f}), "
            f"FLI_med={ham_metrics['fli_stats']['median']:.3f}  |  "
            f"GDN+Water (neutral): Rel={gdn_on['metrics']['relation_basin_entropy']['unique_basins']} "
            f"(S_b={gdn_on['metrics']['relation_basin_entropy']['S_b']:.3f}, D_b={gdn_on['metrics']['relation_uncertainty']['fractal_dimension'] or 0:.2f}), "
            f"Byte={gdn_on['metrics']['byte_basin_entropy']['unique_basins']} "
            f"(S_b={gdn_on['metrics']['byte_basin_entropy']['S_b']:.3f}, D_b={gdn_on['metrics']['byte_uncertainty']['fractal_dimension'] or 0:.2f})",
            fontsize=10.5, fontweight='bold', y=0.98
        )
        r_idx, nr = remap_ids(ham_rel_2d)
        im = axes[0, 0].imshow(r_idx, origin='lower', extent=ext, cmap='turbo', interpolation='nearest')
        axes[0, 0].set_title(f"(1a) Quiet Hamiltonian (dt=0.01): Relation Basins ({nr})")
        fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

        b_idx, nb = remap_ids(ham_byte_2d)
        im = axes[0, 1].imshow(b_idx, origin='lower', extent=ext, cmap='nipy_spectral', interpolation='nearest')
        axes[0, 1].set_title(f"(1b) Quiet Hamiltonian (dt=0.01): Next-Byte Basins ({nb})")
        fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im = axes[0, 2].imshow(ham_fli_2d, origin='lower', extent=ext, cmap='inferno', interpolation='bilinear')
        axes[0, 2].set_title(f"(1c) Quiet Hamiltonian: FLI [med={ham_metrics['fli_stats']['median']:.3f}, max={ham_metrics['fli_stats']['max']:.3f}]")
        fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

        gr_idx, gnr = remap_ids(gdn_on['rel'])
        im = axes[1, 0].imshow(gr_idx, origin='lower', extent=ext, cmap='turbo', interpolation='nearest')
        axes[1, 0].set_title(f"(2a) GDN+Water Closed-Loop: Relation Basins ({gnr})")
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

        gb_idx, gnb = remap_ids(gdn_on['byte'])
        im = axes[1, 1].imshow(gb_idx, origin='lower', extent=ext, cmap='nipy_spectral', interpolation='nearest')
        axes[1, 1].set_title(f"(2b) GDN+Water Closed-Loop: Byte Attractors ({gnb})")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        im = axes[1, 2].imshow(gdn_on['sens'], origin='lower', extent=ext, cmap='inferno', interpolation='bilinear')
        axes[1, 2].set_title(f"(2c) GDN+Water: Boundary Sensitivity log(1+||dh/du||) [p95={gdn_on['metrics']['sensitivity_p95']:.2f}]")
        fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

        for ax in axes.ravel():
            ax.set_xlabel("u (Z0 -> ZA)"); ax.set_ylabel("v (Z0 -> ZB)")
            for px, py, lbl in ((0, 0, 'Z0'), (1, 0, 'ZA'), (0, 1, 'ZB')):
                ax.plot(px, py, 'o', color='white', markersize=5, markeredgecolor='black')
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        p_fig1 = fig_dir / 'cgb_fractal_basins_ham_vs_gdn_2609_v2.png'
        fig.savefig(p_fig1, facecolor=fig.get_facecolor())
        fig.savefig(art_dir / p_fig1.name, facecolor=fig.get_facecolor())
        plt.close(fig)

        # --- FIGURE 2: Water Reservoir Co-evolution Dynamics (Ablation + Pure W_water Slice) ---
        fig, axes = plt.subplots(2, 2, figsize=(12.2, 10.2), dpi=160)
        wm = water_only['metrics']
        fig.suptitle(
            f"MengLi 2813 Water Reservoir Co-evolution Dynamics on Real Thought Manifold\n"
            f"Water Ablation Flips {water_rel_flip_frac*100:.1f}% Rel / {water_byte_flip_frac*100:.1f}% Byte Basins  |  "
            f"Pure W_water Slice (S FIXED at centroid): Rel={wm['relation_basin_entropy']['unique_basins']}, "
            f"Byte={wm['byte_basin_entropy']['unique_basins']} (S_b={wm['byte_basin_entropy']['S_b']:.3f}, "
            f"D_b={wm['byte_uncertainty']['fractal_dimension'] or 0:.2f})",
            fontsize=10.5, fontweight='bold', y=0.98
        )
        diff_map = (gdn_on['rel'] != gdn_off['rel']).astype(np.float32) + 2.0 * (gdn_on['byte'] != gdn_off['byte']).astype(np.float32)
        im = axes[0, 0].imshow(diff_map, origin='lower', extent=ext, cmap='viridis', interpolation='nearest', vmin=0, vmax=3)
        axes[0, 0].set_title("(a) Water ON vs OFF Basin Shift (0=same, 1=rel, 2=byte, 3=both)")
        axes[0, 0].set_xlabel("u (Z0 -> ZA)"); axes[0, 0].set_ylabel("v (Z0 -> ZB)")
        fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im = axes[0, 1].imshow(gdn_on['settle'], origin='lower', extent=ext, cmap='plasma', interpolation='nearest')
        axes[0, 1].set_title(f"(b) GDN+Water Settling/Escape Time (mean={gdn_on['metrics']['settle_mean']:.1f}, max={gdn_on['metrics']['settle_max']})")
        axes[0, 1].set_xlabel("u (Z0 -> ZA)"); axes[0, 1].set_ylabel("v (Z0 -> ZB)")
        fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

        ext_w = [-2.0, 2.0, -2.0, 2.0]
        wb_idx, wnb = remap_ids(water_only['byte'])
        im = axes[1, 0].imshow(wb_idx, origin='lower', extent=ext_w, cmap='nipy_spectral', interpolation='nearest')
        axes[1, 0].set_title(f"(c) PURE Water Slice (S fixed at centroid): Byte Attractors ({wnb})")
        axes[1, 0].set_xlabel("u_W (W0 -> WA)"); axes[1, 0].set_ylabel("v_W (W0 -> WB)")
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im = axes[1, 1].imshow(water_only['sens'], origin='lower', extent=ext_w, cmap='inferno', interpolation='bilinear')
        axes[1, 1].set_title(f"(d) PURE Water Slice (S fixed): Sensitivity [p95={wm['sensitivity_p95']:.2f}]")
        axes[1, 1].set_xlabel("u_W (W0 -> WA)"); axes[1, 1].set_ylabel("v_W (W0 -> WB)")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        p_fig2 = fig_dir / 'cgb_water_reservoir_basins_2609_v2.png'
        fig.savefig(p_fig2, facecolor=fig.get_facecolor())
        fig.savefig(art_dir / p_fig2.name, facecolor=fig.get_facecolor())
        plt.close(fig)

        # --- FIGURE 3: Clean Live Inference Structure (dt=0.01, zero NaN) + Grand Loop ---
        fig, axes = plt.subplots(2, 2, figsize=(13, 9.6), dpi=160)
        c_tr, h_tr = trajectories['calm'], trajectories['surprising']
        fig.suptitle(
            f"MengLi 2813 Inference Structure (Calm NLL={c_tr['context_nll']:.2f} vs Surprising NLL={h_tr['context_nll']:.2f}, dt=0.01)\n"
            f"GDN+Water Mean lambda: calm={c_tr['gdn_mean_lyapunov']:+.4f}, surprising={h_tr['gdn_mean_lyapunov']:+.4f}  |  "
            f"Quiet Ham Mean lambda: calm={c_tr['ham_mean_lyapunov']:+.5f}, surprising={h_tr['ham_mean_lyapunov']:+.5f}",
            fontsize=10.8, fontweight='bold', y=0.98
        )
        sx = np.arange(T_traj)
        axes[0, 0].plot(sx, c_tr['gdn_velocity'], color='#58a6ff', lw=2, label=f"Calm GDN h-vel (flips={c_tr['gdn_relation_flips']})")
        axes[0, 0].plot(sx, h_tr['gdn_velocity'], color='#ff7b72', lw=2, label=f"Surprising GDN h-vel (flips={h_tr['gdn_relation_flips']})")
        axes[0, 0].plot(sx, c_tr['water_velocity'], color='#3fb950', lw=1.5, ls='--', label='Calm Water W-vel')
        axes[0, 0].plot(sx, h_tr['water_velocity'], color='#d29922', lw=1.5, ls='--', label='Surprising Water W-vel')
        axes[0, 0].set_title("(a) Coupled Language h & Slow Water Reservoir W Velocities")
        axes[0, 0].set_xlabel("Step t"); axes[0, 0].set_ylabel("Relative velocity")
        axes[0, 0].legend(facecolor='#0d1117', edgecolor='#30363d', fontsize=8)
        axes[0, 0].grid(alpha=0.15)

        axes[0, 1].axhline(0.0, color='#8b949e', ls=':', lw=1)
        axes[0, 1].plot(sx, c_tr['gdn_local_lyapunov'], color='#58a6ff', lw=2, label='Calm GDN+Water lambda(t)')
        axes[0, 1].plot(sx, h_tr['gdn_local_lyapunov'], color='#ff7b72', lw=2, label='Surprising GDN+Water lambda(t)')
        axes[0, 1].plot(sx, c_tr['ham_local_lyapunov'], color='#3fb950', lw=1.5, ls='--', label='Calm Quiet Ham (dt=0.01)')
        axes[0, 1].plot(sx, h_tr['ham_local_lyapunov'], color='#d29922', lw=1.5, ls='--', label='Surprising Quiet Ham (dt=0.01)')
        axes[0, 1].set_title("(b) Local Lyapunov lambda(t): Edge-of-Chaos GDN vs Symplectic Ham")
        axes[0, 1].set_xlabel("Step t"); axes[0, 1].set_ylabel("log(||dz_{t+1}|| / ||dz_t||)")
        axes[0, 1].legend(facecolor='#0d1117', edgecolor='#30363d', fontsize=8)
        axes[0, 1].grid(alpha=0.15)

        both_h = np.concatenate([c_tr['gdn_h'], h_tr['gdn_h']], axis=0)
        both_c = both_h - both_h.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(both_c, full_matrices=False)
        proj = both_c @ vt[:2].T
        p_calm, p_hard = proj[:T_traj], proj[T_traj:]
        axes[1, 0].plot(p_calm[:, 0], p_calm[:, 1], color='#58a6ff', lw=1.8, alpha=0.9, label='Calm rollout')
        axes[1, 0].plot(p_hard[:, 0], p_hard[:, 1], color='#ff7b72', lw=1.8, alpha=0.9, label='Surprising rollout')
        axes[1, 0].scatter(p_calm[0, 0], p_calm[0, 1], color='#58a6ff', s=55, edgecolor='white', zorder=6)
        axes[1, 0].scatter(p_calm[-1, 0], p_calm[-1, 1], color='#58a6ff', marker='*', s=120, edgecolor='white', zorder=6)
        axes[1, 0].scatter(p_hard[0, 0], p_hard[0, 1], color='#ff7b72', s=55, edgecolor='white', zorder=6)
        axes[1, 0].scatter(p_hard[-1, 0], p_hard[-1, 1], color='#ff7b72', marker='*', s=120, edgecolor='white', zorder=6)
        axes[1, 0].set_title("(c) 2D PCA Latent Phase Portrait (o=start, *=end)")
        axes[1, 0].set_xlabel("PC1"); axes[1, 0].set_ylabel("PC2")
        axes[1, 0].legend(facecolor='#0d1117', edgecolor='#30363d', fontsize=8.5)
        axes[1, 0].grid(alpha=0.15)

        if 'strict8_baseline' in grand_loop and 'phase8_rhythm' in grand_loop:
            sr, pr = grand_loop['strict8_baseline'], grand_loop['phase8_rhythm']
            cs = [r['cycle'] for r in sr]
            ax_d = axes[1, 1]
            ax_d.plot(cs, [r['response_nll'] for r in sr], color='#58a6ff', marker='o', lw=1.8, label='strict8 NLL')
            ax_d.plot(cs, [r['response_nll'] for r in pr], color='#ff7b72', marker='s', lw=1.8, label='phase8 NLL')
            ax_d2 = ax_d.twinx()
            ax_d2.plot(cs, [r['workspace_rms'] or 0 for r in sr], color='#79c0ff', ls='--', lw=1.4, label='strict8 Water RMS')
            ax_d2.plot(cs, [r['workspace_rms'] or 0 for r in pr], color='#ffa657', ls='--', lw=1.4, label='phase8 Water RMS')
            ax_d2.set_ylabel("Water Co-evolution RMS", color='#ffa657')
            ax_d2.tick_params(axis='y', colors='#ffa657')
            ax_d.set_title("(d) Real 8-Cycle Grand Closed-Loop: NLL & Water Reservoir RMS")
            ax_d.set_xlabel("Life cycle"); ax_d.set_ylabel("Response NLL (nats/byte)")
            ax_d.legend(loc='upper left', facecolor='#0d1117', edgecolor='#30363d', fontsize=8)
            ax_d2.legend(loc='upper right', facecolor='#0d1117', edgecolor='#30363d', fontsize=8)
            ax_d.grid(alpha=0.15)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        p_fig3 = fig_dir / 'cgb_inference_and_grandloop_2609_v2.png'
        fig.savefig(p_fig3, facecolor=fig.get_facecolor())
        fig.savefig(art_dir / p_fig3.name, facecolor=fig.get_facecolor())
        plt.close(fig)

        traj_summary = {k: {tk: tv for tk, tv in v.items() if tk != 'gdn_h'} for k, v in trajectories.items()}
        report = dict(
            reference_paper='arXiv:2609.04963 (Fractal basins trap latent reasoning)',
            version='v2_thought_manifold_dt001_neutral_logp',
            checkpoint=str(ckpt), cycle=loaded.cycle, read_only=True, updates=0, world_calls=0,
            total_seconds=round(time.perf_counter() - t_all, 2),
            thought_anchors=dict(
                Z0=dict(idx=i0, rel=ctx_rels[i0], nll=ctx_nlls[i0], snippet=ctx_snips[i0]),
                ZA=dict(idx=iA, rel=ctx_rels[iA], nll=ctx_nlls[iA], snippet=ctx_snips[iA]),
                ZB=dict(idx=iB, rel=ctx_rels[iB], nll=ctx_nlls[iB], snippet=ctx_snips[iB]),
            ),
            probe1_hamiltonian_quiet=ham_metrics,
            probe2a_gdn_water_on=gdn_on['metrics'],
            probe2b_gdn_water_off=gdn_off['metrics'],
            water_ablation_flip_fraction=dict(relation=water_rel_flip_frac, byte=water_byte_flip_frac),
            probe2c_pure_water_slice=water_only['metrics'],
            probe3_inference_trajectories=traj_summary,
            probe4_grand_closed_loop=grand_loop,
            figures=[str(p_fig1), str(p_fig2), str(p_fig3)],
        )
        out_json = Path('results/published/cgb_fractal_basins_2609_04963_v2_20261123.json')
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
        print('ALL V2 DONE in', report['total_seconds'], 's ->', out_json, flush=True)
    finally:
        if loaded is not None:
            arc = getattr(loaded.engine.model, '_causal_archive', None)
            if arc is not None and getattr(arc, 'db', None) is not None:
                arc.db.close()
            del loaded
            gc.collect()
            torch.cuda.empty_cache()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    run_probe()
