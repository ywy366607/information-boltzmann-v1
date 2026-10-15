"""Why is the novelty circuit silent?  (2026-10-15: word_novelty_probe gave exactly 0 for MBON-alpha'3 and PPL104.)
Fixed weights except the online compartment memory.  Reading text (validation, 8 ticks per word, held input) from the
checkpoint's own state, 512 words.  For MBON16 / MBON17(-like) (alpha'3 MBONs), PPL104 (alpha'3 dopamine), all MBONs
and all PPL1 as references:
  firing rate per cell (spikes per tick), fraction of ticks with v / theta > 0.5, the median v / theta;
  input structure: excitatory and inhibitory edges in the static lists (and how many are pinned at 0), the compartment
  (Kenyon -> MBON) edges onto them with their efficacy comp_weight x (1 + phi), and the static MBON-alpha'3 -> PPL104
  edges;
  drive per tick split by presynaptic class (transmitted output x weight), mean over the stream.
Usage: python alpha3_probe.py <checkpoint dir> [out.json]"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else None
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
HOLD = bool(cfg.get('hold_input'))
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
N = model.n_neurons
quiet = torch.zeros(1, N, device='cuda')
g = np.load(ROOT / cfg['graph'], allow_pickle=True)
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
typ, cls, sup = (ann[c].astype(str).values for c in ('type', 'class', 'superclass'))
dan = np.zeros(N, bool); dan[g['dan_indices']] = True
GROUPS = {'MBON_a3': np.isin(typ, ['MBON16', 'MBON17', 'MBON17-like']), 'PPL104': typ == 'PPL104',
          'MBON_all': cls == 'MBON', 'PPL1_all': dan & np.char.startswith(typ.astype(str), 'PPL1')}
GT = {k: torch.as_tensor(v, device='cuda') for k, v in GROUPS.items()}
theta = opts['thresholds'].detach().float().expand(1, N)[0]
rep = {'checkpoint': str(fork), 'cells': {k: int(v.sum()) for k, v in GROUPS.items()}}

# input structure
struct = {}
for k, m in GROUPS.items():
    mt = GT[k]
    row = {}
    for sfx in ('e', 'i'):
        post = getattr(model, f'edge_post_{sfx}').long(); w = getattr(model, f'edge_weight_{sfx}').detach()
        sel = mt[post]
        row[f'{sfx}_edges'] = int(sel.sum()); row[f'{sfx}_pinned0'] = int((sel & (w == 0)).sum())
        row[f'{sfx}_weight_sum'] = float(w[sel].abs().sum())
    if getattr(model, 'comp_post', None) is not None:
        cs = mt[model.comp_post.long()]
        row['comp_edges'] = int(cs.sum()); row['comp_weight_sum'] = float(model.comp_weight[cs].abs().sum())
    struct[k] = row
a3 = torch.as_tensor(np.flatnonzero(GROUPS['MBON_a3']), device='cuda')
p104 = GT['PPL104']
pre_e, post_e = model.edge_pre_e.long(), model.edge_post_e.long()
pre_i, post_i = model.edge_pre_i.long(), model.edge_post_i.long()
isa3 = GT['MBON_a3']
rep['a3_to_PPL104_edges'] = {'e': int((isa3[pre_e] & p104[post_e]).sum()), 'i': int((isa3[pre_i] & p104[post_i]).sum())}
rep['structure'] = struct
# who inhibits them: the inhibitory weight onto each group by presynaptic type (top 6)
inh = {}
w_i = model.edge_weight_i.detach().abs()
for k in ('MBON_a3', 'PPL104', 'PPL1_all'):
    sel = GT[k][post_i]
    pre_types = pd.Series(w_i[sel].cpu().numpy(), index=typ[pre_i[sel].cpu().numpy()]).groupby(level=0).sum()
    inh[k] = {t: round(float(v), 3) for t, v in pre_types.sort_values(ascending=False).head(6).items()}
rep['inhibition_by_type'] = inh
rep['theta_median'] = {k: float(theta[GT[k]].median()) for k in GROUPS}

# activity while reading
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
words = np.array(val[60_000:60_512], dtype=np.int64)
rate = {k: 0.0 for k in GROUPS}; near = {k: 0.0 for k in GROUPS}; vt = {k: [] for k in GROUPS}
ticks = 0
with torch.no_grad():
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    import os
    if os.environ.get('ALPHA3_PHI0') and s.comp_memory.numel():
        # zero-shot: the compartment memory on the alpha'3 Kenyon edges reset to 0 (undepressed)
        mem = s.comp_memory.clone(); mem[0, GT['MBON_a3'][model.comp_post.long()]] = 0.0
        s = fl.FlyPhysicalState(**{**s.state_dict(), 'comp_memory': mem})
        rep['phi_a3_reset'] = True
    for w in words:
        token = torch.tensor([int(w)], device='cuda')
        for t in range(8):
            if t == 0 or HOLD:
                dr, baseline = fl.write_input(model, s, token)
                s = fl.step_fly_physical_tick(model, s, token, dr, baseline, opts, base_rates=opts['base_rates'])
            else:
                baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
                s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
            spk = (s.ring[0][0] > 0).float()
            ratio = s.h[0].float() / theta.clamp_min(1e-6)
            for k, m in GT.items():
                rate[k] += float(spk[m].mean()); near[k] += float((ratio[m] > 0.5).float().mean())
                if t == 7:
                    vt[k].append(float(ratio[m].median()))
            ticks += 1
        s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
rep['rate_per_tick'] = {k: rate[k] / ticks for k in GROUPS}
rep['frac_v_over_half_theta'] = {k: near[k] / ticks for k in GROUPS}
rep['median_v_over_theta'] = {k: [float(np.min(v)), float(np.median(v)), float(np.mean(v)), float(np.max(v))] for k, v in vt.items()}
rep['phi_a3'] = (None if getattr(model, 'comp_post', None) is None or not s.comp_memory.numel() else
                 [float(x) for x in s.comp_memory[0][GT['MBON_a3'][model.comp_post.long()]].float().quantile(
                     torch.tensor([0.0, 0.5, 1.0], device='cuda')).cpu()])
print(json.dumps(rep, indent=1), flush=True)
if OUT:
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
