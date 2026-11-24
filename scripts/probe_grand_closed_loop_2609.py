"""Grand Closed-Loop Perturbation Bifurcation & Multi-System Health Probe (MengLi 2813).

Tests how infinitesimal-to-small initial state perturbations eps in {0, 1e-4, 1e-3, 1e-2, 1e-1}
propagate across ALL FIVE coupled systems in the REAL Grand Closed Loop (4 live cycles each,
real natural-text world navigation, real BudgetedAdamW weight updates, real Water Reservoir
refresh_workset & eviction, real Wood question/unknown binding, real Fire action/speech):

  S/P -> Water (coevolution + memory + relation) -> Wood (question/plan) ->
  Fire (action/word/world cursor) -> Metal (text learning + BudgetedAdamW) -> S/P

Read-only with respect to original 2813 (all temporary copies isolated in E: spool and deleted).
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


def _param_vec(model):
    return torch.cat([p.detach().float().reshape(-1).cpu() for p in model.parameters()])


def run_branch(ckpt: Path, eps: float, direction_cpu: torch.Tensor, cycles: int = 4):
    """Run one completely isolated Grand Closed-Loop branch from original 2813."""
    import scripts.ib_local.run_mother_continuation as cont
    from scripts.ib_local.autonomous_cycle import dispatch
    from scripts.ib_local import phase_coupled_rhythm as pcr
    from scripts.ib_local.scoped_decision_graph import configure_fused_step
    from scripts.ib_local.native_replay_batch import configure as configure_replay
    from scripts.ib_local.epistemic_value_reference import configure as configure_epistemic_value
    from scripts.ib_local.native_amortized_decision import configure as configure_amortization
    from scripts.ib_local.persistent_rhythm import configure as configure_rhythm
    from scripts.ib_local.retention_scope_prior import configure as configure_scope_prior

    tmp = Path(tempfile.mkdtemp(prefix=f'grandloop_eps_{eps}_', dir='E:/mengli-runtime-staging-20261123'))
    loaded = None
    try:
        opts = cont.continuation_parser().parse_args([
            '--checkpoint', str(ckpt), '--output', str(tmp / 'body'), '--cycles', str(cycles),
            '--world-source', 'text', '--autonomous-events', '--optimizer-policy', 'budgeted',
            '--table-storage', str(tmp / 'table'), '--causal-storage', str(tmp / 'causal'),
        ])
        orig_save, orig_export = cont.save, cont.export_live
        cont.save = lambda *a, **kw: 'probe-ephemeral'
        cont.export_live = lambda *a, **kw: None
        try:
            loaded = cont.prepare_continuation(opts, resume_output=False)
        finally:
            cont.save, cont.export_live = orig_save, orig_export

        e = loaded.engine
        m = e.model
        # Match production runtime stack + phase_coupled_rhythm for fast 4-cycle rollout
        configure_epistemic_value(e, schema=5, word_execution='native_anchor')
        e.native_word_runtime = 'batch'
        m._preview_batch_enabled = True
        configure_amortization(e, dict(schema=1, seed=201120, planner='action_batch',
                                       correction='decision_budget', root_native_correction='observe'))
        configure_rhythm(e, interleave=True)
        configure_scope_prior(e, 'off')
        e.native_direct_question = True
        e.native_factorized_endpoints = True
        e.native_shared_utter_preview = True
        e.policy_proposal_batch_size = 4
        configure_replay(e, loaded.config, batch_size=8, reuse_actual=True)
        configure_fused_step(e, loaded.config)
        pcr.configure(e, loaded.config)

        # Apply exact initial state perturbation eps * ||rec0|| * unit_direction
        rec0_norm = float(e.recurrent.detach().float().norm())
        delta0 = 0.0
        if eps > 0.0:
            pert = (eps * rec0_norm) * direction_cpu.to(device=e.device, dtype=e.recurrent.dtype)
            e.recurrent = e.recurrent + pert
            delta0 = float(pert.float().norm())

        w_size = m.body.coevolution.state_size
        mat_size = m.body.base.layers[0].matrix_size

        t0 = time.perf_counter()
        cycle_records = []
        state_snapshots = []
        ext_done = 0
        polls = 0
        while ext_done < cycles and polls < cycles * 64:
            row = dispatch(e)
            polls += 1
            if row.get('autonomous_event', {}).get('kind') == 'external':
                ext_done += 1
                pp = row.get('past_present', {})
                w = row.get('world', {})
                su = row.get('shared_unknown', {})
                cp = row.get('causal_program', {})
                ad = row.get('adaptive_dynamics', {})
                bwf = ad.get('budget_write_fraction')
                pcr_ev = row.get('phase_coupled_rhythm', {}).get('last_event', {})
                cycle_records.append(dict(
                    cycle=int(row.get('cycle', loaded.cycle + ext_done)),
                    choice=row.get('choice'),
                    utterance_hex=row.get('utterance_hex', ''),
                    self_bytes=int(row.get('self_bytes', 0)),
                    response_nll=float(row.get('response_nll', 0.0)),
                    metal_g=float(row.get('metal_g', 0.0)),
                    world_cursor=int(w.get('cursor', -1)),
                    world_matched=int(w.get('matched_bytes', 0)),
                    unknown_id=su.get('identity'),
                    relation_id=cp.get('roles', {}).get('identity'),
                    workspace_rms=float(pp.get('workspace_rms', 0.0)),
                    workset_slots=list(pp.get('slots', [])),
                    workset_retained=int(pp.get('retained', 0)),
                    regime=pcr_ev.get('regime', 'unknown'),
                    ham_gate_mean=None if bwf is None else float(np.mean(bwf)),
                    ham_step_max=float(max(ad.get('per_layer_parameter_step', [0.0]))),
                ))
                state_snapshots.append(dict(
                    s=e.recurrent[..., :mat_size].detach().float().cpu().clone(),
                    w=e.recurrent[..., -w_size:].detach().float().cpu().clone(),
                    p=torch.stack([t.detach().float().cpu() for t in e.bfss_momenta]),
                    ret_pref=m.memory.retention_utility_first.detach().float().cpu().clone()
                    if hasattr(m.memory, 'retention_utility_first') else None,
                ))

        # Collect health metrics on final optimizer/body state (for eps==0 baseline)
        health = None
        if eps == 0.0:
            gate_stats = []
            for p_name, p in m.named_parameters():
                st = e.optimizer.state.get(p, {})
                bg = st.get('budget_gate')
                if bg is not None:
                    gate_stats.append((p_name, int(p.numel()), float(bg.detach().float().mean())))
            zero_gate = [x for x in gate_stats if x[2] == 0.0]
            low_gate = [x for x in gate_stats if 0.0 < x[2] < 0.05]
            active_gate = [x for x in gate_stats if x[2] >= 0.05]
            mem = m.memory
            occ = mem.occupied.detach().cpu()
            bank = m.causal_relations
            health = dict(
                total_budgeted_params=len(gate_stats),
                zero_gate_count=len(zero_gate),
                low_gate_count=len(low_gate),
                active_gate_count=len(active_gate),
                zero_gate_examples=zero_gate[:15],
                low_gate_examples=low_gate[:15],
                memory_occupied=int(occ.sum()),
                memory_capacity=int(occ.numel()),
                active_relations=int((bank.identity > 0).sum()),
                relation_capacity=int(bank.identity.numel()),
                coev_valid=int(m.body.coevolution.valid.sum()),
                coev_pref_min=float(m.body.coevolution.preference.min()),
                coev_pref_max=float(m.body.coevolution.preference.max()),
                read_gain_tanh=float(m.read_gain.tanh().detach()),
            )

        final_params = _param_vec(m)
        elapsed = round(time.perf_counter() - t0, 2)
        print(f'Branch eps={eps:g} DONE in {elapsed}s: choices={[r["choice"] for r in cycle_records]}, '
              f'cursors={[r["world_cursor"] for r in cycle_records]}, '
              f'unks={[r["unknown_id"] for r in cycle_records]}', flush=True)
        return dict(eps=eps, delta0=delta0, rec0_norm=rec0_norm, elapsed=elapsed,
                    records=cycle_records, snapshots=state_snapshots,
                    final_params=final_params, health=health)
    finally:
        if loaded is not None:
            arc = getattr(loaded.engine.model, '_causal_archive', None)
            if arc is not None and getattr(arc, 'db', None) is not None:
                arc.db.close()
            del loaded
            gc.collect()
            torch.cuda.empty_cache()
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    t_all = time.perf_counter()
    ckpt = Path('E:/mengli-native-supply-completed-20261122/original/last.pt.gz')
    fig_dir = Path('results/published/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    art_dir = Path('C:/Users/85750/.gemini/antigravity/brain/b52ce350-543f-4727-bcda-4a922c34066a')

    # Fixed deterministic unit perturbation vector in recurrent shape (32, 1, 10752)
    gen = torch.Generator().manual_seed(26092026)
    direction = torch.randn(32, 1, 10752, generator=gen)
    direction = direction / direction.norm()

    eps_list = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    branches = {}
    for eps in eps_list:
        branches[eps] = run_branch(ckpt, eps, direction, cycles=4)

    base = branches[0.0]
    comparisons = {}
    for eps in eps_list[1:]:
        br = branches[eps]
        per_cycle = []
        for t in range(4):
            rb, rp = base['records'][t], br['records'][t]
            sb, sp = base['snapshots'][t], br['snapshots'][t]
            ds = float((sp['s'] - sb['s']).norm())
            dw = float((sp['w'] - sb['w']).norm())
            dp = float((sp['p'] - sb['p']).norm())
            slots_b, slots_p = set(rb['workset_slots']), set(rp['workset_slots'])
            jacc = len(slots_b & slots_p) / max(1, len(slots_b | slots_p))
            per_cycle.append(dict(
                cycle=rb['cycle'],
                ds=ds,
                dw=dw,
                dp=dp,
                amplification=ds / max(br['delta0'], 1e-12),
                choice_same=(rb['choice'] == rp['choice']),
                utterance_same=(rb['utterance_hex'] == rp['utterance_hex']),
                cursor_same=(rb['world_cursor'] == rp['world_cursor']),
                cursor_diff=int(rp['world_cursor'] - rb['world_cursor']),
                unknown_same=(rb['unknown_id'] == rp['unknown_id']),
                relation_same=(rb['relation_id'] == rp['relation_id']),
                workset_jaccard=float(jacc),
                nll_diff=float(rp['response_nll'] - rb['response_nll']),
                metal_g_diff=float(rp['metal_g'] - rb['metal_g']),
            ))
        param_diff = float((br['final_params'] - base['final_params']).norm())
        param_rel = param_diff / float(base['final_params'].norm())
        comparisons[f'{eps:g}'] = dict(
            eps=eps,
            delta0=br['delta0'],
            final_param_l2_diff=param_diff,
            final_param_rel_diff=param_rel,
            per_cycle=per_cycle,
        )

    # Render Figure 4: Grand Closed-Loop Perturbation Bifurcation
    plt.rcParams.update({
        'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d', 'axes.labelcolor': '#e6edf3',
        'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
        'text.color': '#e6edf3', 'font.size': 9.5,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), dpi=160)
    fig.suptitle(
        "MengLi 2813 Grand Closed-Loop Perturbation Bifurcation (4 Live Cycles, Real World + Learning + Water)\n"
        "Tracing initial perturbation eps in {1e-4, 1e-3, 1e-2, 1e-1} through S/P, Water Reservoir, World Cursor & Weights",
        fontsize=11, fontweight='bold', y=0.98
    )
    cycles_x = [r['cycle'] for r in base['records']]
    colors = {'0.0001': '#58a6ff', '0.001': '#3fb950', '0.01': '#d29922', '0.1': '#ff7b72'}

    # (a) State divergence ||dS(t)|| (log scale)
    for k, comp in comparisons.items():
        ds_vals = [max(c['ds'], 1e-9) for c in comp['per_cycle']]
        axes[0, 0].semilogy(cycles_x, ds_vals, marker='o', lw=2, color=colors[k], label=f'eps={k} (d0={comp["delta0"]:.3g})')
    axes[0, 0].set_title("(a) GDN State Divergence ||dS(t)|| Across Grand Closed-Loop Cycles")
    axes[0, 0].set_xlabel("Life cycle"); axes[0, 0].set_ylabel("||S_eps - S_0||_2 (log)")
    axes[0, 0].legend(facecolor='#0d1117', edgecolor='#30363d', fontsize=8.5)
    axes[0, 0].grid(alpha=0.15)

    # (b) Water Reservoir Divergence ||dW(t)|| + Workset Jaccard
    for k, comp in comparisons.items():
        dw_vals = [max(c['dw'], 1e-9) for c in comp['per_cycle']]
        axes[0, 1].semilogy(cycles_x, dw_vals, marker='s', lw=2, color=colors[k], label=f'eps={k} ||dW||')
    axes[0, 1].set_title("(b) Water Reservoir Co-evolution Field Divergence ||dW_water(t)||")
    axes[0, 1].set_xlabel("Life cycle"); axes[0, 1].set_ylabel("||W_eps - W_0||_2 (log)")
    axes[0, 1].legend(facecolor='#0d1117', edgecolor='#30363d', fontsize=8.5)
    axes[0, 1].grid(alpha=0.15)

    # (c) Discrete Bifurcation Matrix (choice / utterance / world cursor / unknown / workset)
    labels_y = ['choice', 'utterance', 'world_cursor', 'unknown_id', 'workset_100%']
    mat = np.zeros((len(comparisons), 4))
    for i, (k, comp) in enumerate(comparisons.items()):
        for t, c in enumerate(comp['per_cycle']):
            # Count how many of the 5 discrete macro-ports diverged (0..5)
            div = (int(not c['choice_same']) + int(not c['utterance_same']) +
                   int(not c['cursor_same']) + int(not c['unknown_same']) +
                   int(c['workset_jaccard'] < 0.999))
            mat[i, t] = div
    im = axes[1, 0].imshow(mat, aspect='auto', cmap='magma', vmin=0, vmax=5)
    axes[1, 0].set_yticks(range(len(comparisons)))
    axes[1, 0].set_yticklabels([f'eps={k}' for k in comparisons.keys()])
    axes[1, 0].set_xticks(range(4)); axes[1, 0].set_xticklabels(cycles_x)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axes[1, 0].text(j, i, f'{int(mat[i, j])}/5', ha='center', va='center', color='white', fontweight='bold')
    axes[1, 0].set_title("(c) Macro-Port Bifurcation Count (choice/word/cursor/unknown/workset)")
    axes[1, 0].set_xlabel("Life cycle")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # (d) Permanent Weight Divergence ||dTheta(end)|| vs Initial Perturbation eps
    eps_x = [comp['eps'] for comp in comparisons.values()]
    p_diffs = [max(comp['final_param_l2_diff'], 1e-10) for comp in comparisons.values()]
    s_end = [max(comp['per_cycle'][-1]['ds'], 1e-10) for comp in comparisons.values()]
    axes[1, 1].loglog(eps_x, p_diffs, marker='D', lw=2.2, color='#ff7b72', label='Permanent weight ||dTheta_final||')
    axes[1, 1].loglog(eps_x, s_end, marker='o', lw=2.0, ls='--', color='#58a6ff', label='Final GDN state ||dS_final||')
    axes[1, 1].set_title("(d) Perturbation Sensitivity: State -> Permanent Synaptic Weights")
    axes[1, 1].set_xlabel("Initial perturbation eps"); axes[1, 1].set_ylabel("L2 divergence after 4 cycles (log)")
    axes[1, 1].legend(facecolor='#0d1117', edgecolor='#30363d', fontsize=8.5)
    axes[1, 1].grid(alpha=0.15)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p_fig4 = fig_dir / 'cgb_grand_closed_loop_bifurcation_2609.png'
    fig.savefig(p_fig4, facecolor=fig.get_facecolor())
    fig.savefig(art_dir / p_fig4.name, facecolor=fig.get_facecolor())
    plt.close(fig)

    report = dict(
        checkpoint=str(ckpt),
        cycles=4,
        total_seconds=round(time.perf_counter() - t_all, 2),
        baseline_records=base['records'],
        system_health=base['health'],
        comparisons=comparisons,
        figure=str(p_fig4),
    )
    out_json = Path('results/published/cgb_grand_closed_loop_bifurcation_20261123.json')
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    print('GRAND CLOSED-LOOP PROBE DONE in', report['total_seconds'], 's ->', out_json, flush=True)


if __name__ == '__main__':
    main()
