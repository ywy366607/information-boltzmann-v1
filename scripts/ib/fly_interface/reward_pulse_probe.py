"""Calibrate her answer's reward pulse (attach_reply_reward): does a pulse on the PAM cells make them fire, raise phasic
dopamine at the MBONs and write the compartment memory phi where Kenyon traces are standing?  Fixed weights except the
online compartment rule (it runs, as in life).  From the checkpoint's state, reading the lineage's own conversation
(mother.jsonl, 6 ticks per word); at 20 places, fork: control (no pulse) vs a pulse of `level` x rheobase for 4 ticks
at a word onset; 40 ticks per fork.  Per level: PAM extra spikes per cell, mean phasic dopamine (fast - slow) / slow
at the compartment edges, |delta phi| summed over the compartment edges and its share on edges whose Kenyon trace
was standing (> its median) at the pulse.  [min | median | mean | max] over places.
Usage: python reward_pulse_probe.py <checkpoint dir> <run dir> [out.json]"""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork, RUN = Path(sys.argv[1]), Path(sys.argv[2])
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else None
model, cfg, L = load(fork)
graph = ROOT / cfg['graph']
n_pam = fl.attach_reply_reward(model, graph, graph.parent / 'body-annotations.feather', level=1.0, ticks=4)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
s0 = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
if getattr(model, 'dopamine_sensitivity', None) is not None:
    opts['dopamine_s_class'] = fl.dopamine_class_sensitivity(model, s0.h)
import tiktoken
tok = tiktoken.get_encoding('gpt2')
lines = [json.loads(l) for l in (RUN / 'mother.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
stream = []
for r in lines[-120:]:
    if r['text'].strip() and r['text'].strip() != '...':
        stream += tok.encode(' ' + r['text'].strip())
stream = stream[:400]
pam = model.reward_cells
comp = getattr(model, 'comp_pre', None) is not None


def run(s, w, ticks, pulse):
    pam_spk = 0.0
    phasic = []
    for t in range(ticks):
        out = fl.advance_fly_token_stream(model, s, torch.tensor([int(w)], device='cuda'), [1], ticks=1,
                                          hold_input=bool(cfg.get('hold_input')), fresh=t == 0,
                                          reward_ticks=max(0, pulse - t), **opts)
        s = out[0]
        pam_spk += float((s.ring[0][0].index_select(0, pam) > 0).float().mean())
        if comp and s.dopamine.numel():
            d = s.dopamine.index_select(1, model.comp_post.long())
            phasic.append(float(((d[0] - d[1]) / d[1].clamp_min(1e-12)).mean()))
    return s, pam_spk, float(np.mean(phasic)) if phasic else None


rep = {'checkpoint': str(fork), 'pam_cells': n_pam, 'levels': {}}
places = list(range(20, 400, 19))[:20]
with torch.no_grad():
    s = s0
    states = []
    for i, w in enumerate(stream):
        if i in places:
            states.append((s, w))
        s = run(s, w, 6, 0)[0]
    for level in (0.5, 1.0, 2.0, 4.0):
        model.reward_level = level
        rows = []
        for s_p, w in states:
            trace = s_p.comp_trace.index_select(1, model.comp_edge_kc)[0] if comp and s_p.comp_trace.numel() else None
            s_c, pam_c, ph_c = run(s_p, w, 40, 0)
            s_r, pam_r, ph_r = run(s_p, w, 40, 4)
            row = {'pam_extra': (pam_r - pam_c) / 40, 'phasic_extra': (ph_r - ph_c) if ph_c is not None else None}
            if comp and s_r.comp_memory.numel():
                dphi = (s_r.comp_memory[0] - s_c.comp_memory[0]).abs()
                row['dphi_sum'] = float(dphi.sum())
                if trace is not None:
                    standing = trace > trace.median()
                    row['dphi_share_on_traces'] = float(dphi[standing].sum() / dphi.sum().clamp_min(1e-12))
            rows.append(row)
        rep['levels'][str(level)] = {k: [float(np.min(v)), float(np.median(v)), float(np.mean(v)), float(np.max(v))]
                                     for k in rows[0] if rows[0][k] is not None
                                     for v in [[r[k] for r in rows if r.get(k) is not None]]}
        print(level, json.dumps(rep['levels'][str(level)]), flush=True)
if OUT:
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
