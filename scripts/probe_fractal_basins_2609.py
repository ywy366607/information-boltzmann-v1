"""Probe arXiv:2609.04963 ('Fractal basins trap latent reasoning') across MengLi's
four coupled dynamical levels (read-only on original 2813, 0 updates, 0 world calls):

  1. Local Quiet Hamiltonian (S, P) symplectic BFSS map (2D slice: basins, FLI, flips);
  2. Local Closed-Loop 32L GDN + Water Reservoir Co-evolution (2D slice in S: Water ON vs OFF);
  3. Pure Water-Reservoir Co-evolution Field W_water (2D slice ONLY in recurrent[..., -256:]);
  4. Inference Trajectory Anatomy (Calm vs Surprising contexts) + Grand Closed-Loop Anatomy
     from completed multi-cycle real-life trajectories (strict8_baseline & phase8_rhythm).
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
    """Daza et al. (2016) Basin Entropy S_b and Boundary Basin Entropy S_bb."""
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


def uncertainty_exponent(labels: np.ndarray, epsilons=(1, 2, 4, 8)):
    """Estimate uncertainty exponent alpha from f(eps) ~ eps^alpha and D_b = 2 - alpha."""
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
    r2 = float(np.corrcoef(lx, ly)[0, 1] ** 2) if len(valid_eps) >= 2 else None
    return dict(
        alpha=alpha,
        fractal_dimension=float(2.0 - alpha),
        r2=r2,
        epsilons=list(valid_eps),
        uncertain_fractions=[float(x) for x in fracs],
    )


def chunked_ham_quiet(s, p, theta, context, dt, frequency_scale, impedance, *, layer_chunk=4):
    """Exact AdaptiveHamiltonian.quiet chunked across independent layers (<120MB VRAM)."""
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
    """Vectorized relation binding matching RelationBank.binding_address."""
    z = F.normalize(model.encoder(h_batch), dim=-1)
    sim = z @ rel_keys.T
    idx = sim.argmax(-1)
    return rel_ids[idx].cpu().numpy()


@torch.no_grad()
def batched_gdn_step(model, ids, state, table_logp):
    """Run B independent trajectories through the exact full 32L GDN + Water Co-evolution body."""
    from scripts.ib_local.native_replay_batch import replay_scope
    b = ids.shape[0]
    with replay_scope(model, [(-1, -1)] * b, length=ids.shape[1]):
        logits, next_state, _, h, _ = model.predict_observed(ids, state, table_logp)
    return logits, next_state, h


def run_gdn_slice(m, rec0, e1_slice, e2_slice, start_col, end_col, pending_byte, logp0,
                  rel_keys, rel_ids, *, grid=40, steps=28, radius=1.6, batch=32):
    """Sweep a 2D orthonormal slice inside recurrent[..., start_col:end_col] under closed-loop GDN."""
    device = rec0.device
    L = rec0.shape[0]
    gv = np.linspace(-radius, radius, grid, dtype=np.float32)
    ug, vg = np.meshgrid(gv, gv)
    coords = np.stack([ug.ravel(), vg.ravel()], axis=-1)
    total = len(coords)

    out_rel = np.zeros(total, dtype=np.int64)
    out_byte = np.zeros(total, dtype=np.int64)
    out_settle = np.zeros(total, dtype=np.int32)
    out_flips = np.zeros(total, dtype=np.int32)
    out_final_h = np.zeros((total, 256), dtype=np.float32)
    out_water_rms = np.zeros(total, dtype=np.float32)
    w_size = m.body.coevolution.state_size

    for start in range(0, total, batch):
        end = min(start + batch, total)
        b = end - start
        u_t = torch.as_tensor(coords[start:end, 0], device=device).reshape(1, b, 1)
        v_t = torch.as_tensor(coords[start:end, 1], device=device).reshape(1, b, 1)
        rec_b = rec0.expand(L, b, -1).clone()
        rec_b[..., start_col:end_col] = rec_b[..., start_col:end_col] + u_t * e1_slice + v_t * e2_slice

        ids_b = torch.full((b, 1), pending_byte, device=device, dtype=torch.long)
        logp_b = logp0.expand(b, 1, 256)
        prev_rel = None
        prev_h = None
        flips = np.zeros(b, dtype=np.int32)
        last_change = np.zeros(b, dtype=np.int32)

        for step_t in range(1, steps + 1):
            logits_b, rec_b, h_b = batched_gdn_step(m, ids_b, rec_b, logp_b)
            next_ids = logits_b[:, -1].argmax(-1, keepdim=True)
            rel_now = relation_slot_ids(m, h_b, rel_keys, rel_ids)
            if prev_rel is not None:
                changed = (rel_now != prev_rel)
                flips += changed.astype(np.int32)
                last_change = np.where(changed, step_t, last_change)
            if prev_h is not None:
                vel = ((h_b - prev_h).norm(dim=-1) / prev_h.norm(dim=-1).clamp_min(1e-6)).cpu().numpy()
                last_change = np.where(vel > 0.15, step_t, last_change)
            prev_rel = rel_now
            prev_h = h_b.detach()
            ids_b = next_ids

        out_rel[start:end] = prev_rel
        out_byte[start:end] = ids_b[:, 0].cpu().numpy()
        out_settle[start:end] = last_change
        out_flips[start:end] = flips
        out_final_h[start:end] = prev_h.cpu().numpy()
        out_water_rms[start:end] = rec_b[..., -w_size:].square().mean(dim=(0, 2)).sqrt().cpu().numpy()

    rel_2d = out_rel.reshape(grid, grid)
    byte_2d = out_byte.reshape(grid, grid)
    settle_2d = out_settle.reshape(grid, grid)
    flips_2d = out_flips.reshape(grid, grid)
    h_2d = out_final_h.reshape(grid, grid, -1)
    wrms_2d = out_water_rms.reshape(grid, grid)
    dx = np.linalg.norm(np.diff(h_2d, axis=1, append=h_2d[:, -1:]), axis=-1)
    dy = np.linalg.norm(np.diff(h_2d, axis=0, append=h_2d[-1:, :]), axis=-1)
    du = float(2 * radius / (grid - 1))
    sens_2d = np.log1p(np.sqrt(dx**2 + dy**2) / max(du, 1e-6))

    metrics = dict(
        grid=grid, steps=steps, radius=radius,
        relation_basin_entropy=basin_entropy(rel_2d, box=3),
        relation_uncertainty=uncertainty_exponent(rel_2d, epsilons=(1, 2, 4)),
        byte_basin_entropy=basin_entropy(byte_2d, box=3),
        byte_uncertainty=uncertainty_exponent(byte_2d, epsilons=(1, 2, 4)),
        settle_mean=float(settle_2d.mean()), settle_max=int(settle_2d.max()),
        flips_mean=float(flips_2d.mean()), flips_max=int(flips_2d.max()),
        sensitivity_p50=float(np.median(sens_2d)),
        sensitivity_p95=float(np.percentile(sens_2d, 95)),
        sensitivity_max=float(sens_2d.max()),
        water_rms_mean=float(wrms_2d.mean()),
        water_rms_std=float(wrms_2d.std()),
    )
    return dict(rel=rel_2d, byte=byte_2d, settle=settle_2d, flips=flips_2d,
                sens=sens_2d, water_rms=wrms_2d, metrics=metrics)


def run_probe():
    t_all = time.perf_counter()
    ckpt = Path('E:/mengli-native-supply-completed-20261122/original/last.pt.gz')
    fig_dir = Path('results/published/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    art_dir = Path('C:/Users/85750/.gemini/antigravity/brain/b52ce350-543f-4727-bcda-4a922c34066a')
    art_dir.mkdir(parents=True, exist_ok=True)

    import scripts.ib_local.run_mother_continuation as cont
    from scripts.ib_local.adaptive_hamiltonian import invariants
    from scripts.ib_local.language_action import peek
    from scripts.ib_local.navigation_live_witness import frozen_table_logp

    tmp = Path(tempfile.mkdtemp(prefix='probe_2609_', dir='E:/mengli-runtime-staging-20261123'))
    loaded = None
    try:
        opts = cont.continuation_parser().parse_args([
            '--checkpoint', str(ckpt),
            '--output', str(tmp / 'body'),
            '--cycles', '1',
            '--world-source', 'text',
            '--autonomous-events',
            '--optimizer-policy', 'budgeted',
            '--table-storage', str(tmp / 'table'),
            '--causal-storage', str(tmp / 'causal'),
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
        first = base.layers[0]
        mat_size = first.matrix_size
        w_size = coev.state_size
        L = len(base.layers)

        # Precompute relation keys from RelationBank.condition (exact binding_address formula)
        bank = m.causal_relations
        live_mask = bank.identity > 0
        with torch.no_grad():
            rel_keys = F.normalize(m.encoder(bank.condition[live_mask]), dim=-1)
            rel_ids = bank.identity[live_mask]

        rec0 = e.recurrent.detach().clone()           # (32, 1, 10752)
        s0 = rec0[..., :mat_size].reshape(L, 1, 8, 32, 32)
        p0 = torch.stack([t.detach().clone() for t in e.bfss_momenta])
        w0 = rec0[..., -w_size:]                      # (32, 1, 256) Water Reservoir field!
        if not bool(aq.context_valid):
            corr, _ = invariants(s0)
            ctx = corr.mean(-1)[..., None, None]
        else:
            ctx = aq.context.detach().clone()
        theta = aq.theta.detach()
        freq_scale = aq.frequency_scale
        impedance = aq.impedance if aq.spectrum_enabled else None

        s_norm = float(s0.norm())
        p_norm = float(p0.norm())
        w_norm = float(w0.norm())
        pending_byte = int(e.pending)
        table_probs0 = peek(e.table, pending_byte)
        logp0 = torch.as_tensor(np.log(np.clip(table_probs0, 1e-9, 1.0)),
                                device=device, dtype=torch.float32).reshape(1, 1, 256)

        print(f'Loaded 2813: s_norm={s_norm:.3f}, p_norm={p_norm:.3f}, w_norm={w_norm:.3f}, '
              f'live_relations={int(live_mask.sum())}, coev_valid={int(coev.valid.sum())}')

        gen = torch.Generator(device=device).manual_seed(260904963)
        # Orthonormal 2D basis in S, P, and W_water
        def ortho_pair(numel, scale):
            a = torch.randn(numel, generator=gen, device=device)
            b = torch.randn(numel, generator=gen, device=device)
            a = a / a.norm()
            b = b - (b @ a) * a
            b = b / b.norm()
            return a * scale, b * scale

        e1_s_flat, e2_s_flat = ortho_pair(s0.numel(), s_norm)
        e1_p_flat, e2_p_flat = ortho_pair(p0.numel(), p_norm)
        e1_w_flat, e2_w_flat = ortho_pair(w0.numel(), max(w_norm, 1.0))

        e1_s = e1_s_flat.reshape(L, 1, 8, 32, 32)
        e2_s = e2_s_flat.reshape(L, 1, 8, 32, 32)
        e1_p = e1_p_flat.reshape(L, 1, 8, 32, 32)
        e2_p = e2_p_flat.reshape(L, 1, 8, 32, 32)
        e1_s_col = e1_s_flat.reshape(L, 1, mat_size)
        e2_s_col = e2_s_flat.reshape(L, 1, mat_size)
        e1_w_col = e1_w_flat.reshape(L, 1, w_size)
        e2_w_col = e2_w_flat.reshape(L, 1, w_size)

        tan_s = torch.randn(s0.numel(), generator=gen, device=device)
        tan_s = (tan_s / tan_s.norm()).reshape(L, 1, 8, 32, 32)
        eps_fli = 1e-3

        # =====================================================================
        # PROBE 1: Local Quiet Hamiltonian (S, P) Map (24x24, T=20, dt=0.10)
        # =====================================================================
        t1 = time.perf_counter()
        N_ham, T_ham, radius_ham = 24, 20, 1.6
        dt_ham = float(e.bfss_dt) * 10.0  # 2.0 total phase horizon in 20 fast symplectic steps
        gv = np.linspace(-radius_ham, radius_ham, N_ham, dtype=np.float32)
        uu, vv = np.meshgrid(gv, gv)
        coords = np.stack([uu.ravel(), vv.ravel()], axis=-1)
        total_ham = len(coords)

        ham_rel = np.zeros(total_ham, dtype=np.int64)
        ham_byte = np.zeros(total_ham, dtype=np.int64)
        ham_fli = np.zeros(total_ham, dtype=np.float32)
        ham_flips = np.zeros(total_ham, dtype=np.int32)
        ham_last_flip = np.zeros(total_ham, dtype=np.int32)

        B_ham = 8
        ids_one = torch.full((B_ham, 1), pending_byte, device=device, dtype=torch.long)
        logp_one = logp0.expand(B_ham, 1, 256)

        for start in range(0, total_ham, B_ham):
            end = min(start + B_ham, total_ham)
            b = end - start
            u_t = torch.as_tensor(coords[start:end, 0], device=device).reshape(1, b, 1, 1, 1)
            v_t = torch.as_tensor(coords[start:end, 1], device=device).reshape(1, b, 1, 1, 1)
            sb = s0 + u_t * e1_s + v_t * e2_s
            pb = p0 + u_t * e1_p + v_t * e2_p
            s_both = torch.cat([sb, sb + eps_fli * tan_s], dim=1)
            p_both = torch.cat([pb, pb], dim=1)

            max_fli = torch.full((b,), -10.0, device=device)
            prev_rel = None
            flips = np.zeros(b, dtype=np.int32)
            last_flip = np.zeros(b, dtype=np.int32)

            for step_t in range(1, T_ham + 1):
                s_both, p_both = chunked_ham_quiet(
                    s_both, p_both, theta, ctx, dt_ham, freq_scale, impedance, layer_chunk=4
                )
                ds = s_both[:, b:] - s_both[:, :b]
                dp = p_both[:, b:] - p_both[:, :b]
                dist = torch.sqrt(ds.square().sum(dim=(0, 2, 3, 4)) + dp.square().sum(dim=(0, 2, 3, 4))).clamp_min(1e-12)
                max_fli = torch.maximum(max_fli, torch.log(dist / eps_fli))
                if step_t % 4 == 0 or step_t == T_ham:
                    rec_b = rec0.expand(L, b, -1).clone()
                    rec_b[..., :mat_size] = s_both[:, :b].flatten(2)
                    logits_b, _, h_b = batched_gdn_step(m, ids_one[:b], rec_b, logp_one[:b])
                    rel_now = relation_slot_ids(m, h_b, rel_keys, rel_ids)
                    if prev_rel is not None:
                        changed = (rel_now != prev_rel)
                        flips += changed.astype(np.int32)
                        last_flip = np.where(changed, step_t, last_flip)
                    prev_rel = rel_now
                    if step_t == T_ham:
                        ham_rel[start:end] = rel_now
                        ham_byte[start:end] = logits_b[:, -1].argmax(-1).cpu().numpy()

            ham_fli[start:end] = max_fli.cpu().numpy()
            ham_flips[start:end] = flips
            ham_last_flip[start:end] = last_flip

        ham_rel_2d = ham_rel.reshape(N_ham, N_ham)
        ham_byte_2d = ham_byte.reshape(N_ham, N_ham)
        ham_fli_2d = ham_fli.reshape(N_ham, N_ham)
        ham_flips_2d = ham_flips.reshape(N_ham, N_ham)
        ham_last_flip_2d = ham_last_flip.reshape(N_ham, N_ham)
        ham_metrics = dict(
            grid=N_ham, steps=T_ham, dt_effective=dt_ham, radius=radius_ham,
            seconds=round(time.perf_counter() - t1, 2),
            relation_basin_entropy=basin_entropy(ham_rel_2d, box=3),
            relation_uncertainty=uncertainty_exponent(ham_rel_2d, epsilons=(1, 2, 4)),
            byte_basin_entropy=basin_entropy(ham_byte_2d, box=3),
            byte_uncertainty=uncertainty_exponent(ham_byte_2d, epsilons=(1, 2, 4)),
            fli_stats=dict(min=float(ham_fli_2d.min()), median=float(np.median(ham_fli_2d)),
                           mean=float(ham_fli_2d.mean()), p95=float(np.percentile(ham_fli_2d, 95)),
                           max=float(ham_fli_2d.max())),
            flips_mean=float(ham_flips_2d.mean()), flips_max=int(ham_flips_2d.max()),
        )
        print('PROBE 1 (Hamiltonian Quiet) DONE in', ham_metrics['seconds'], 's:',
              'Rel S_b=', round(ham_metrics['relation_basin_entropy']['S_b'], 3),
              'alpha=', ham_metrics['relation_uncertainty']['alpha'],
              'FLI med/max=', round(ham_metrics['fli_stats']['median'], 2), round(ham_metrics['fli_stats']['max'], 2),
              flush=True)

        # =====================================================================
        # PROBE 2A & 2B: Closed-Loop GDN in S (Water Reservoir ON vs OFF)
        # =====================================================================
        t2 = time.perf_counter()
        coev.enabled.fill_(1.0)
        gdn_on = run_gdn_slice(m, rec0, e1_s_col, e2_s_col, 0, mat_size, pending_byte, logp0,
                               rel_keys, rel_ids, grid=24, steps=18, radius=1.6, batch=16)
        gdn_on['metrics']['seconds'] = round(time.perf_counter() - t2, 2)
        print('PROBE 2A (GDN + Water ON) DONE in', gdn_on['metrics']['seconds'], 's:',
              'Rel S_b=', round(gdn_on['metrics']['relation_basin_entropy']['S_b'], 3),
              'alpha=', gdn_on['metrics']['relation_uncertainty']['alpha'],
              'Byte S_b=', round(gdn_on['metrics']['byte_basin_entropy']['S_b'], 3),
              'alpha=', gdn_on['metrics']['byte_uncertainty']['alpha'], flush=True)

        t2b = time.perf_counter()
        coev.enabled.fill_(0.0)
        try:
            gdn_off = run_gdn_slice(m, rec0, e1_s_col, e2_s_col, 0, mat_size, pending_byte, logp0,
                                    rel_keys, rel_ids, grid=24, steps=18, radius=1.6, batch=16)
        finally:
            coev.enabled.fill_(1.0)
        gdn_off['metrics']['seconds'] = round(time.perf_counter() - t2b, 2)
        # Direct pixel-by-pixel basin flip fraction caused purely by removing the Water Reservoir!
        water_rel_flip_frac = float((gdn_on['rel'] != gdn_off['rel']).mean())
        water_byte_flip_frac = float((gdn_on['byte'] != gdn_off['byte']).mean())
        print('PROBE 2B (GDN + Water OFF) DONE in', gdn_off['metrics']['seconds'], 's:',
              'Water ablation flipped Rel basins:', f'{water_rel_flip_frac*100:.1f}%',
              'Byte basins:', f'{water_byte_flip_frac*100:.1f}%', flush=True)

        # =====================================================================
        # PROBE 2C: Pure Water-Reservoir Co-evolution Field Slice (S0 FIXED!)
        # =====================================================================
        t2c = time.perf_counter()
        water_only = run_gdn_slice(m, rec0, e1_w_col, e2_w_col, rec0.shape[-1] - w_size, rec0.shape[-1],
                                   pending_byte, logp0, rel_keys, rel_ids,
                                   grid=24, steps=18, radius=2.5, batch=16)
        water_only['metrics']['seconds'] = round(time.perf_counter() - t2c, 2)
        print('PROBE 2C (PURE Water Reservoir Slice, S0 FIXED!) DONE in', water_only['metrics']['seconds'], 's:',
              'Rel basins=', water_only['metrics']['relation_basin_entropy']['unique_basins'],
              'Rel S_b=', round(water_only['metrics']['relation_basin_entropy']['S_b'], 3),
              'Byte basins=', water_only['metrics']['byte_basin_entropy']['unique_basins'],
              'Byte S_b=', round(water_only['metrics']['byte_basin_entropy']['S_b'], 3),
              'Byte alpha=', water_only['metrics']['byte_uncertainty']['alpha'], flush=True)

        # =====================================================================
        # PROBE 3: Live Inference Trajectory Structure (Calm vs Surprising)
        # =====================================================================
        manifest = loaded.config['natural_text_corpus']
        val_raw = Path(manifest['validation_path']).read_bytes()
        starts = np.linspace(0, len(val_raw) - 257, 16, dtype=np.int64)
        chunk_nlls, chunk_states = [], []
        for st_idx in starts:
            seg = np.frombuffer(val_raw[st_idx:st_idx+257], np.uint8).copy()
            lp = torch.as_tensor(frozen_table_logp(e.table, seg[:-1]), device=device)[None]
            ids = torch.as_tensor(seg[:-1].astype(np.int64), device=device)[None]
            tgt = torch.as_tensor(seg[1:].astype(np.int64), device=device)[None]
            st_cur = rec0.clone()
            losses = []
            with torch.no_grad():
                for pos in range(0, 256, 128):
                    lg, st_cur, _ = m.predict(ids[:, pos:pos+128], st_cur, lp[:, pos:pos+128])
                    losses.append(float(F.cross_entropy(lg[0], tgt[0, pos:pos+128])))
            chunk_nlls.append(float(np.mean(losses)))
            chunk_states.append((st_cur.detach().clone(), int(seg[-1]),
                                 bytes(seg[-48:]).decode('utf-8', errors='replace')))

        calm_idx, hard_idx = int(np.argmin(chunk_nlls)), int(np.argmax(chunk_nlls))
        T_traj = 48
        trajectories = {}
        for label_name, c_idx in (('calm', calm_idx), ('surprising', hard_idx)):
            st_init, last_b, snippet = chunk_states[c_idx]
            tan_rec = torch.randn_like(st_init)
            tan_rec = tan_rec / tan_rec.norm()
            eps_tr = 1e-3
            rec_pair = torch.cat([st_init.clone(), st_init.clone() + eps_tr * tan_rec], dim=1)
            ids_c = torch.full((2, 1), last_b, device=device, dtype=torch.long)
            lp_c = torch.as_tensor(np.log(np.clip(peek(e.table, last_b), 1e-9, 1.0)),
                                   device=device, dtype=torch.float32).reshape(1, 1, 256).expand(2, 1, 256)

            vel_list, wvel_list, ent_list, rel_list, lyap_list, h_list, bytes_out = [], [], [], [], [], [], []
            prev_h, prev_w = None, None
            for step_t in range(T_traj):
                lg, rec_pair, h_pair = batched_gdn_step(m, ids_c, rec_pair, lp_c)
                h0 = h_pair[0]
                w0_cur = rec_pair[:, 0, -w_size:]
                probs = lg[0, -1].softmax(-1)
                ent = float(-(probs * (probs + 1e-12).log()).sum())
                rel_id = int(relation_slot_ids(m, h0[None], rel_keys, rel_ids)[0])
                b_next = int(lg[0, -1].argmax(-1))
                bytes_out.append(b_next)
                vel = 0.0 if prev_h is None else float((h0 - prev_h).norm() / prev_h.norm().clamp_min(1e-6))
                wvel = 0.0 if prev_w is None else float((w0_cur - prev_w).norm() / prev_w.norm().clamp_min(1e-6))
                prev_h = h0.detach().clone()
                prev_w = w0_cur.detach().clone()
                delta_s = rec_pair[:, 1:2] - rec_pair[:, 0:1]
                d_norm = float(delta_s.norm())
                lyap = math.log(max(d_norm, 1e-12) / eps_tr)
                rec_pair[:, 1:2] = rec_pair[:, 0:1] + (eps_tr / max(d_norm, 1e-12)) * delta_s
                ids_c = torch.full((2, 1), b_next, device=device, dtype=torch.long)
                vel_list.append(vel); wvel_list.append(wvel); ent_list.append(ent)
                rel_list.append(rel_id); lyap_list.append(lyap); h_list.append(h0.cpu().numpy())

            # Quiet Hamiltonian from same context
            s_q = st_init[..., :mat_size].reshape(L, 1, 8, 32, 32)
            s_pair = torch.cat([s_q, s_q + eps_fli * tan_s], dim=1)
            p_pair = torch.cat([p0.clone(), p0.clone()], dim=1)
            ham_lyap, ham_rels, ham_vel = [], [], []
            prev_hq = None
            for step_t in range(T_traj):
                s_pair, p_pair = chunked_ham_quiet(
                    s_pair, p_pair, theta, ctx, dt_ham, freq_scale, impedance, layer_chunk=4
                )
                ds = s_pair[:, 1:2] - s_pair[:, 0:1]
                dp = p_pair[:, 1:2] - p_pair[:, 0:1]
                dn = float(torch.sqrt(ds.square().sum() + dp.square().sum()))
                ham_lyap.append(math.log(max(dn, 1e-12) / eps_fli))
                scale = eps_fli / max(dn, 1e-12)
                s_pair[:, 1:2] = s_pair[:, 0:1] + scale * ds
                p_pair[:, 1:2] = p_pair[:, 0:1] + scale * dp
                rec_q = st_init.clone()
                rec_q[..., :mat_size] = s_pair[:, 0:1].flatten(2)
                _, _, hq = batched_gdn_step(m, torch.full((1, 1), last_b, device=device, dtype=torch.long), rec_q, lp_c[:1])
                ham_rels.append(int(relation_slot_ids(m, hq, rel_keys, rel_ids)[0]))
                ham_vel.append(0.0 if prev_hq is None else float((hq[0] - prev_hq).norm() / prev_hq.norm().clamp_min(1e-6)))
                prev_hq = hq[0].detach().clone()

            trajectories[label_name] = dict(
                context_nll=chunk_nlls[c_idx], snippet=snippet,
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

        # =====================================================================
        # PROBE 4: Real Grand Closed-Loop Trajectory Anatomy (strict8 vs phase8)
        # =====================================================================
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
                    cp = r.get('causal_program', {})
                    su = r.get('shared_unknown', {})
                    w = r.get('world', {})
                    rows.append(dict(
                        cycle=r.get('cycle'),
                        choice=r.get('choice'),
                        response_nll=r.get('response_nll'),
                        metal_g=r.get('metal_g'),
                        workspace_rms=pp.get('workspace_rms'),
                        workset_retained=pp.get('retained'),
                        workset_slots=pp.get('slots'),
                        unknown_id=su.get('identity'),
                        relation_id=cp.get('roles', {}).get('identity'),
                        world_cursor=w.get('cursor'),
                        world_redirected=bool(w.get('matched_bytes', 0)),
                        regime=r.get('phase_coupled_rhythm', {}).get('last_event', {}).get('regime', 'strict'),
                    ))
            grand_loop[arm] = rows

        # =====================================================================
        # RENDER HIGH-RESOLUTION PUBLICATION FIGURES (3 RICH MULTI-PANEL PLOTS)
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

        # --- FIGURE 1: Local Quiet Hamiltonian (S, P) vs Closed-Loop GDN (S) ---
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.6), dpi=160)
        fig.suptitle(
            f"MengLi 2813 (32L, 30.5M) — arXiv:2609.04963 Fractal Basin Maps: Symplectic Quiet (S,P) vs Dissipative GDN+Water\n"
            f"Quiet Ham: Rel S_b={ham_metrics['relation_basin_entropy']['S_b']:.3f}, D_b={ham_metrics['relation_uncertainty']['fractal_dimension'] or 0:.3f}, "
            f"FLI_max={ham_metrics['fli_stats']['max']:.2f}  |  "
            f"GDN+Water: Rel S_b={gdn_on['metrics']['relation_basin_entropy']['S_b']:.3f}, D_b={gdn_on['metrics']['relation_uncertainty']['fractal_dimension'] or 0:.3f}, "
            f"Byte S_b={gdn_on['metrics']['byte_basin_entropy']['S_b']:.3f}, D_b={gdn_on['metrics']['byte_uncertainty']['fractal_dimension'] or 0:.3f}",
            fontsize=11, fontweight='bold', y=0.98
        )
        ext = [-radius_ham, radius_ham, -radius_ham, radius_ham]
        # Top row: Symplectic Quiet Hamiltonian
        r_idx, nr = remap_ids(ham_rel_2d)
        im = axes[0, 0].imshow(r_idx, origin='lower', extent=ext, cmap='turbo', interpolation='nearest')
        axes[0, 0].set_title(f"(1a) Quiet Hamiltonian: Relation Basins ({nr})")
        fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im = axes[0, 1].imshow(ham_fli_2d, origin='lower', extent=ext, cmap='inferno', interpolation='bilinear')
        axes[0, 1].set_title(f"(1b) Quiet Hamiltonian: FLI (med={ham_metrics['fli_stats']['median']:.2f})")
        fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im = axes[0, 2].imshow(ham_last_flip_2d, origin='lower', extent=ext, cmap='magma', interpolation='nearest')
        axes[0, 2].set_title(f"(1c) Quiet Hamiltonian: Last Flip Step (mean={ham_metrics['flips_mean']:.2f})")
        fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # Bottom row: Dissipative Closed-Loop 32L GDN + Water Co-evolution
        gr_idx, gnr = remap_ids(gdn_on['rel'])
        im = axes[1, 0].imshow(gr_idx, origin='lower', extent=ext, cmap='turbo', interpolation='nearest')
        axes[1, 0].set_title(f"(2a) GDN+Water Closed-Loop: Relation Basins ({gnr})")
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

        gb_idx, gnb = remap_ids(gdn_on['byte'])
        im = axes[1, 1].imshow(gb_idx, origin='lower', extent=ext, cmap='nipy_spectral', interpolation='nearest')
        axes[1, 1].set_title(f"(2b) GDN+Water Closed-Loop: Byte Attractors ({gnb})")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        im = axes[1, 2].imshow(gdn_on['sens'], origin='lower', extent=ext, cmap='inferno', interpolation='bilinear')
        axes[1, 2].set_title(f"(2c) GDN+Water: Sensitivity log(1+||dh/du||) [p95={gdn_on['metrics']['sensitivity_p95']:.2f}]")
        fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

        for ax in axes.ravel():
            ax.set_xlabel("u / ||S0||"); ax.set_ylabel("v / ||S0||")
            ax.plot(0, 0, '+', color='white', markersize=7, markeredgewidth=1.4)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        p_fig1 = fig_dir / 'cgb_fractal_basins_ham_vs_gdn_2609.png'
        fig.savefig(p_fig1, facecolor=fig.get_facecolor())
        fig.savefig(art_dir / p_fig1.name, facecolor=fig.get_facecolor())
        plt.close(fig)

        # --- FIGURE 2: Water Reservoir Co-evolution Dynamics (Ablation + Pure W_water Slice!) ---
        fig, axes = plt.subplots(2, 2, figsize=(12.2, 10.2), dpi=160)
        wm = water_only['metrics']
        fig.suptitle(
            f"MengLi 2813 Water Reservoir Co-evolution Dynamics (PastPresentField, 32L × 8 slots × 32 rank)\n"
            f"Water Ablation Flips {water_rel_flip_frac*100:.1f}% Rel / {water_byte_flip_frac*100:.1f}% Byte Basins  |  "
            f"Pure W_water Slice (S0 FIXED): {wm['byte_basin_entropy']['unique_basins']} Byte Basins, "
            f"S_b={wm['byte_basin_entropy']['S_b']:.3f}, D_b={wm['byte_uncertainty']['fractal_dimension'] or 0:.3f}",
            fontsize=11, fontweight='bold', y=0.98
        )
        # (a) Where turning Water Reservoir OFF changes final basin
        diff_map = (gdn_on['rel'] != gdn_off['rel']).astype(np.float32) + 2.0 * (gdn_on['byte'] != gdn_off['byte']).astype(np.float32)
        im = axes[0, 0].imshow(diff_map, origin='lower', extent=ext, cmap='viridis', interpolation='nearest', vmin=0, vmax=3)
        axes[0, 0].set_title(f"(a) Water ON vs OFF Basin Shift (0=same, 1=rel, 2=byte, 3=both)")
        axes[0, 0].set_xlabel("u / ||S0||"); axes[0, 0].set_ylabel("v / ||S0||")
        fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # (b) Settling / Escape Time under GDN + Water
        im = axes[0, 1].imshow(gdn_on['settle'], origin='lower', extent=ext, cmap='plasma', interpolation='nearest')
        axes[0, 1].set_title(f"(b) GDN+Water Settling/Escape Time (mean={gdn_on['metrics']['settle_mean']:.1f}, max={gdn_on['metrics']['settle_max']})")
        axes[0, 1].set_xlabel("u / ||S0||"); axes[0, 1].set_ylabel("v / ||S0||")
        fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # (c) Pure Water Reservoir W_water Slice -> Byte Attractors (S0 held 100% FIXED!)
        ext_w = [-2.5, 2.5, -2.5, 2.5]
        wb_idx, wnb = remap_ids(water_only['byte'])
        im = axes[1, 0].imshow(wb_idx, origin='lower', extent=ext_w, cmap='nipy_spectral', interpolation='nearest')
        axes[1, 0].set_title(f"(c) PURE Water Slice (S0 fixed): Byte Attractors ({wnb} symbols)")
        axes[1, 0].set_xlabel("u_W / ||W0||"); axes[1, 0].set_ylabel("v_W / ||W0||")
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # (d) Pure Water Reservoir W_water Slice -> Finite-Time Sensitivity
        im = axes[1, 1].imshow(water_only['sens'], origin='lower', extent=ext_w, cmap='inferno', interpolation='bilinear')
        axes[1, 1].set_title(f"(d) PURE Water Slice (S0 fixed): Sensitivity [p95={wm['sensitivity_p95']:.2f}]")
        axes[1, 1].set_xlabel("u_W / ||W0||"); axes[1, 1].set_ylabel("v_W / ||W0||")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        for ax in axes.ravel():
            ax.plot(0, 0, '+', color='white', markersize=7, markeredgewidth=1.4)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        p_fig2 = fig_dir / 'cgb_water_reservoir_basins_2609.png'
        fig.savefig(p_fig2, facecolor=fig.get_facecolor())
        fig.savefig(art_dir / p_fig2.name, facecolor=fig.get_facecolor())
        plt.close(fig)

        # --- FIGURE 3: Live Inference Trajectories + Grand Closed-Loop Anatomy ---
        fig, axes = plt.subplots(2, 2, figsize=(13, 9.6), dpi=160)
        c_tr, h_tr = trajectories['calm'], trajectories['surprising']
        fig.suptitle(
            f"MengLi 2813 Inference Structure (Calm NLL={c_tr['context_nll']:.2f} vs Surprising NLL={h_tr['context_nll']:.2f}) & Grand Closed-Loop\n"
            f"GDN Mean lambda: calm={c_tr['gdn_mean_lyapunov']:+.3f}, surprising={h_tr['gdn_mean_lyapunov']:+.3f}  |  "
            f"Ham Mean lambda: calm={c_tr['ham_mean_lyapunov']:+.4f}, surprising={h_tr['ham_mean_lyapunov']:+.4f}",
            fontsize=11, fontweight='bold', y=0.98
        )
        sx = np.arange(T_traj)
        axes[0, 0].plot(sx, c_tr['gdn_velocity'], color='#58a6ff', lw=2, label=f"Calm GDN h-vel (flips={c_tr['gdn_relation_flips']})")
        axes[0, 0].plot(sx, h_tr['gdn_velocity'], color='#ff7b72', lw=2, label=f"Surprising GDN h-vel (flips={h_tr['gdn_relation_flips']})")
        axes[0, 0].plot(sx, c_tr['water_velocity'], color='#3fb950', lw=1.4, ls='--', label='Calm Water W-vel')
        axes[0, 0].plot(sx, h_tr['water_velocity'], color='#d29922', lw=1.4, ls='--', label='Surprising Water W-vel')
        axes[0, 0].set_title("(a) Coupled Language h & Water Reservoir W Velocities")
        axes[0, 0].set_xlabel("Step t"); axes[0, 0].set_ylabel("Relative velocity")
        axes[0, 0].legend(facecolor='#0d1117', edgecolor='#30363d', fontsize=8)
        axes[0, 0].grid(alpha=0.15)

        axes[0, 1].axhline(0.0, color='#8b949e', ls=':', lw=1)
        axes[0, 1].plot(sx, c_tr['gdn_local_lyapunov'], color='#58a6ff', lw=2, label='Calm GDN+Water lambda(t)')
        axes[0, 1].plot(sx, h_tr['gdn_local_lyapunov'], color='#ff7b72', lw=2, label='Surprising GDN+Water lambda(t)')
        axes[0, 1].plot(sx, c_tr['ham_local_lyapunov'], color='#3fb950', lw=1.4, ls='--', label='Calm Quiet Ham lambda(t)')
        axes[0, 1].plot(sx, h_tr['ham_local_lyapunov'], color='#d29922', lw=1.4, ls='--', label='Surprising Quiet Ham lambda(t)')
        axes[0, 1].set_title("(b) Local Lyapunov Exponent lambda(t): Dissipative vs Symplectic")
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

        # (d) Real 8-cycle Grand Closed-Loop Bifurcation: strict8 vs phase8 (Water RMS + NLL + World Cursor)
        if 'strict8_baseline' in grand_loop and 'phase8_rhythm' in grand_loop:
            sr = grand_loop['strict8_baseline']
            pr = grand_loop['phase8_rhythm']
            cs = [r['cycle'] for r in sr]
            ax_d = axes[1, 1]
            ax_d.plot(cs, [r['response_nll'] for r in sr], color='#58a6ff', marker='o', lw=1.8, label='strict8 NLL')
            ax_d.plot(cs, [r['response_nll'] for r in pr], color='#ff7b72', marker='s', lw=1.8, label='phase8 NLL')
            ax_d2 = ax_d.twinx()
            ax_d2.plot(cs, [r['workspace_rms'] or 0 for r in sr], color='#79c0ff', ls='--', lw=1.4, label='strict8 Water RMS')
            ax_d2.plot(cs, [r['workspace_rms'] or 0 for r in pr], color='#ffa657', ls='--', lw=1.4, label='phase8 Water RMS')
            ax_d2.set_ylabel("Water Co-evolution RMS", color='#ffa657')
            ax_d2.tick_params(axis='y', colors='#ffa657')
            ax_d.set_title("(d) Real 8-Cycle Grand Closed-Loop: NLL & Water Reservoir Trajectory")
            ax_d.set_xlabel("Life cycle"); ax_d.set_ylabel("Response NLL (nats/byte)")
            ax_d.legend(loc='upper left', facecolor='#0d1117', edgecolor='#30363d', fontsize=8)
            ax_d2.legend(loc='upper right', facecolor='#0d1117', edgecolor='#30363d', fontsize=8)
            ax_d.grid(alpha=0.15)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        p_fig3 = fig_dir / 'cgb_inference_and_grandloop_2609.png'
        fig.savefig(p_fig3, facecolor=fig.get_facecolor())
        fig.savefig(art_dir / p_fig3.name, facecolor=fig.get_facecolor())
        plt.close(fig)

        traj_summary = {
            k: {tk: tv for tk, tv in v.items() if tk != 'gdn_h'}
            for k, v in trajectories.items()
        }
        report = dict(
            reference_paper='arXiv:2609.04963 (Fractal basins trap latent reasoning)',
            checkpoint=str(ckpt),
            cycle=loaded.cycle,
            read_only=True, updates=0, world_calls=0,
            total_seconds=round(time.perf_counter() - t_all, 2),
            probe1_hamiltonian_quiet=ham_metrics,
            probe2a_gdn_water_on=gdn_on['metrics'],
            probe2b_gdn_water_off=gdn_off['metrics'],
            water_ablation_flip_fraction=dict(relation=water_rel_flip_frac, byte=water_byte_flip_frac),
            probe2c_pure_water_slice=water_only['metrics'],
            probe3_inference_trajectories=traj_summary,
            probe4_grand_closed_loop=grand_loop,
            figures=[str(p_fig1), str(p_fig2), str(p_fig3)],
        )
        out_json = Path('results/published/cgb_fractal_basins_2609_04963_20261123.json')
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
        print('ALL DONE in', report['total_seconds'], 's ->', out_json)
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
