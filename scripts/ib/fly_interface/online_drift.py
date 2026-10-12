"""Why does the release-gain memory drift?  Fixed weights of dasite_pre with the online rule attached at eta = 0
(no effect on the dynamics).  At the Kenyon terminals, per tick: phasic dopamine p, presynaptic trace e, spikes s.
Forward pairing (activity before dopamine) e*p, backward pairing (dopamine before activity) p_trace*s, their means and
the share of the mean that comes from correlation; the forward-minus-backward rule's drift."""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
fork = Path(sys.argv[1])
model, cfg, L = load(fork)
fl.attach_online_da_plasticity(model, ROOT / cfg['graph'], (ROOT / cfg['graph']).parent / 'body-annotations.feather', eta=0.0)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
kc = model.online_da_target[0] > 0
lam = float(torch.sigmoid(model.online_da_logit_trace))
rows = []
ptrace = None
original_step = fl.step_fly_physical_tick
def step(model_, current, *a, **k):
    global ptrace
    out = original_step(model_, current, *a, **k)
    if out.da_trace.numel():
        tonic = out.dopamine[1]
        level = tonic[tonic > 0].mean()
        p = ((out.dopamine[0] - tonic) / (tonic + level))[kc]
        s = (out.ring[0][0] > 0).float()[kc]
        e = out.da_trace[0][kc]
        ptrace = p.clone() if ptrace is None else lam * ptrace + (1 - lam) * p
        rows.append(torch.stack([p.mean(), p.abs().mean(), e.mean(), (e * p).mean(), (ptrace * s).mean(), s.mean(), p.std()]))
    return out
fl.step_fly_physical_tick = step
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
text = np.array(val[60_000:60_000 + 400], dtype=np.int64)
torch.manual_seed(0)
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
with torch.no_grad():
    for tok in text:
        s, *_ = fl.advance_fly_token_go(model, s, torch.tensor([int(tok)], device='cuda'), [], max_ticks=32,
                                        writer_baseline_clock='physical', **opts)
R = torch.stack(rows).cpu().numpy()[200:]
names = ['p mean', '|p| mean', 'e mean', 'e*p mean (forward)', 'p_trace*s mean (backward)', 's mean', 'p sd across KCs']
rep = {n: float(R[:, j].mean()) for j, n in enumerate(names)}
rep['e*p if uncorrelated'] = float((R[:, 2] * R[:, 0]).mean())
rep['forward - backward'] = rep['e*p mean (forward)'] - rep['p_trace*s mean (backward)']
rep['ticks'] = len(R)
print(json.dumps(rep, indent=1))
