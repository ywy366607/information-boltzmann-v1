"""Lightweight Stage 1B objective reconciliation (read-only on cycle 2845 body).

Zero optimizer updates, zero world calls, zero checkpoint mutations.

1. Reconciles memory.log_temperature gradient across EVERY term of real
   production WoodObservationBody.objective:
       L = 10 * CE + 0.1 * (L_short + L_long) + 0.001 * L_sigreg
   and verifies the weighted sum matches autograd on model.objective() directly.

2. Reconciles log_relation_ridge between:
   - Production autograd path (where posterior() detaches residual_sum -> noise),
   - Full recomputation scoring curve (where changing ridge also shifts
     residual_sum -> noise),
   at 5 neighboring log-ridge points around the 2845 value (0.10008484).
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
TMP_WORK = Path('E:/mengli-batch2-regression-20261123/stage1b_reconcile_tmp').resolve()
OUT_PATH = PROJECT / 'results/published/cgb_stage1b_objective_reconciliation_20261123.json'
STARTS = [0, 95629, 191258, 286887]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.perf_counter()
    sha_before = sha256_file(CKPT)
    if TMP_WORK.exists():
        shutil.rmtree(TMP_WORK, ignore_errors=True)
    os.environ['CGB_CHECKPOINT_SPOOL'] = str(SPOOL)
    from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
    from scripts.ib_local.navigation_live_witness import frozen_table_logp
    from scripts.ib_local.causal_relation_program import mechanism, log_score
    from scripts.ib_local.language_navigation_kanli_v4l import live_sigreg

    opts = continuation_parser().parse_args([
        '--checkpoint', str(CKPT), '--output', str(TMP_WORK), '--cycles', '1',
        '--world-source', 'text', '--autonomous-events', '--optimizer-policy', 'budgeted',
        '--table-storage', str(TMP_WORK / 'table'), '--causal-storage', str(TMP_WORK / 'causal'),
    ])
    loaded = prepare_continuation(opts, resume_output=False)
    engine = loaded.engine
    model = engine.model
    device = engine.device
    versions_before = [p._version for p in model.parameters()]

    try:
        # =====================================================================
        # 1. Temperature gradient reconciliation on real WoodObservationBody.objective
        # =====================================================================
        raw_all = np.fromfile(VAL_BIN, dtype=np.uint8)
        # Use 4 real 128-byte chunks (same chunk length as production objective)
        chunks = [raw_all[s:s + 129] for s in STARTS]
        init_rec = engine.recurrent.detach().clone()
        torch.manual_seed(20261123)
        directions = torch.randn(model.encoder.out_features, 32, device=device)
        old_qctx = getattr(model, '_question_context', None)
        model._question_context = None

        term_grads = dict(ce=[], short_loss=[], long_loss=[], sigreg=[], total_objective=[])
        term_vals = dict(ce=[], short_loss=[], long_loss=[], sigreg=[], total_objective=[])
        lt_param = model.memory.log_temperature

        for raw in chunks:
            logp = torch.as_tensor(frozen_table_logp(engine.table, raw[:-1]), device=device)[None]
            ids = torch.as_tensor(raw[:-1].astype(np.int64), device=device)[None]
            targets = torch.as_tensor(raw[1:].astype(np.int64), device=device)[None]
            # Call actual production model.objective() first for exact ground truth
            loss_obj, _, h_obj, diag = model.objective(ids, targets, init_rec.clone(), logp, directions)
            # Note: clear read_credit receipts so no state leaks across probes
            model.memory._read_participation = []
            model.memory._read_credit_active = False
            model.memory._read_credit_entries = 0
            g_tot, = torch.autograd.grad(loss_obj, lt_param, allow_unused=True)
            term_grads['total_objective'].append(0.0 if g_tot is None else float(g_tot.detach()))
            term_vals['total_objective'].append(float(loss_obj.detach()))

            # Re-run exact same forward graph to get individual term gradients
            logits, _, _, h, _ = model.predict_observed(ids, init_rec.clone(), logp)
            ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            short = model.memory.short
            src = h[:-short]
            rec = model.memory.read(src, model.encoder)
            short_loss = F.mse_loss(model.predictor(torch.cat((model.encoder(src), rec.value), -1)),
                                    model.encoder(h[short:]))
            long_loss = model.long_prediction_loss(h)
            sigreg = live_sigreg(model.encoder(h), model.encoder(model.memory.history),
                                 model.memory.history_valid, directions)
            for name, tensor in (('ce', ce), ('short_loss', short_loss),
                                 ('long_loss', long_loss), ('sigreg', sigreg)):
                g, = torch.autograd.grad(tensor, lt_param, retain_graph=True, allow_unused=True)
                term_grads[name].append(0.0 if g is None else float(g.detach()))
                term_vals[name].append(float(tensor.detach()))
        model._question_context = old_qctx

        mean_g = {k: float(np.mean(v)) for k, v in term_grads.items()}
        mean_v = {k: float(np.mean(v)) for k, v in term_vals.items()}
        # WoodObservationBody.objective weights: 10*CE + 0.1*(short+long) + 0.001*sigreg
        weighted_contrib = dict(
            w10_ce=10.0 * mean_g['ce'],
            w01_short=0.1 * mean_g['short_loss'],
            w01_long=0.1 * mean_g['long_loss'],
            w0001_sigreg=0.001 * mean_g['sigreg'],
        )
        reconstructed_total = sum(weighted_contrib.values())

        # =====================================================================
        # 2. Ridge autograd path vs full-recomputation scoring curve
        # =====================================================================
        material = torch.load(MATERIAL, map_location=device, weights_only=False)
        orig_log_r = model.log_relation_ridge.detach().clone()
        saved_param = model._parameters.pop('log_relation_ridge')
        offsets = [-0.05, -0.01, 0.0, 0.01, 0.05]
        ridge_points = []
        try:
            def eval_ridge_autograd(log_r_tensor):
                r = log_r_tensor.exp().clamp(1e-3, 10.0)
                nlls = []
                for state, fact in material:
                    d = mechanism(model, state, fact.source, fact.action,
                                  exclude_fact=fact.fact_id, ridge=r)
                    nlls.append(-log_score(d, model.encoder(fact.outcome)).mean())
                return torch.stack(nlls).mean()

            for off in offsets:
                x = (orig_log_r + off).detach().clone().requires_grad_(True)
                val = eval_ridge_autograd(x)
                g1, = torch.autograd.grad(val, x, create_graph=True)
                g2, = torch.autograd.grad(g1, x)
                ridge_points.append(dict(
                    offset=off,
                    log_ridge=float(x.detach()),
                    ridge=float(x.detach().exp()),
                    neg_log_score=float(val.detach()),
                    autograd_grad=float(g1.detach()),
                    autograd_hess=float(g2.detach()),
                ))
        finally:
            model._parameters['log_relation_ridge'] = saved_param

        # Finite-difference full-curve 1st and 2nd derivatives at offset=0 (using +-0.01 and +-0.05)
        by_off = {p['offset']: p['neg_log_score'] for p in ridge_points}
        full_grad_001 = (by_off[0.01] - by_off[-0.01]) / 0.02
        full_hess_001 = (by_off[0.01] - 2.0 * by_off[0.0] + by_off[-0.01]) / (0.01 ** 2)
        full_grad_005 = (by_off[0.05] - by_off[-0.05]) / 0.10
        full_hess_005 = (by_off[0.05] - 2.0 * by_off[0.0] + by_off[-0.05]) / (0.05 ** 2)
        p0 = next(p for p in ridge_points if p['offset'] == 0.0)

        assert versions_before == [p._version for p in model.parameters()], 'read-only check failed!'
    finally:
        if hasattr(model, '_causal_archive') and hasattr(model._causal_archive, 'db'):
            model._causal_archive.db.close()
        if TMP_WORK.exists():
            shutil.rmtree(TMP_WORK, ignore_errors=True)
    assert sha256_file(CKPT) == sha_before

    report = dict(
        schema=1,
        source_checkpoint=str(CKPT),
        source_sha256=sha_before,
        source_cycle=2845,
        optimizer_updates=0,
        world_calls=0,
        elapsed_seconds=round(time.perf_counter() - t0, 2),
        temperature_objective_reconciliation=dict(
            formula='WoodObservationBody.objective = 10*CE + 0.1*(short_loss + long_loss) + 0.001*sigreg',
            current_temperature=float(model.memory.log_temperature.detach().exp()),
            term_values=mean_v,
            raw_term_grads_wrt_log_temp=mean_g,
            weighted_grad_contributions=weighted_contrib,
            reconstructed_total_grad=reconstructed_total,
            direct_model_objective_grad=mean_g['total_objective'],
            reconciliation_abs_error=abs(reconstructed_total - mean_g['total_objective']),
        ),
        ridge_autograd_vs_full_curve=dict(
            current_ridge=p0['ridge'],
            current_log_ridge=p0['log_ridge'],
            detached_path_note='posterior() detaches residual_sum into noise; autograd differentiates through mean and leverage only',
            autograd_grad_at_current=p0['autograd_grad'],
            autograd_hess_at_current=p0['autograd_hess'],
            full_curve_fd_grad_h001=full_grad_001,
            full_curve_fd_hess_h001=full_hess_001,
            full_curve_fd_grad_h005=full_grad_005,
            full_curve_fd_hess_h005=full_hess_005,
            same_sign=(p0['autograd_grad'] * full_grad_001 > 0),
            detached_noise_grad_contribution_h001=full_grad_001 - p0['autograd_grad'],
            neighbor_points=ridge_points,
        ),
    )
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
