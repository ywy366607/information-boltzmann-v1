"""What does the O(1) online rule itself learn beyond the BPTT window?  Fixed slow weights, self-paced reading, the
physical state carried through (never reset).

For several pairs of passages (A, B: 256 words each, from different parts of the validation text): warm up 64 words,
read A, B, then A again.  Revisit savings = NLL(A first) - NLL(A again).  Conditions:
  live     the release-gain memory keeps being written (the rule as trained)
  frozen   no writes during the test (eta = 0, no forgetting): the memory stays as it was at the start
The difference live - frozen is what the online rule learns in the stream, with every slow weight fixed.  Also the
memory's spread and the phasic dopamine at the Kenyon terminals.  For a fork without the rule only 'live' is run.
Usage: python online_revisit.py <fork dir> [pairs]
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1]); pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
max_ticks = int(L.get('go_max_ticks', 32))
online = getattr(model, 'online_da_eta', None) is not None
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
WARM, LEN = 64, 256


def nll_of(s, seq):
    out = []
    for i in range(len(seq) - 1):
        s, lat, _, _, _, _ = fl.advance_fly_token_go(model, s, torch.tensor([int(seq[i])], device='cuda'), [], max_ticks=max_ticks,
                                                     writer_baseline_clock='physical', **opts)
        out.append(float(F.cross_entropy(model.decoder(model.read_norm(lat[-1:])).float(), torch.tensor([int(seq[i + 1])], device='cuda'))))
    return s, float(np.mean(out))


def run(pair):
    a0 = 70_000 + pair * 4_000
    warm = np.array(val[a0 - WARM:a0], dtype=np.int64)
    A = np.array(val[a0:a0 + LEN + 1], dtype=np.int64)
    B = np.array(val[a0 + 2_000:a0 + 2_000 + LEN + 1], dtype=np.int64)
    torch.manual_seed(pair)
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    with torch.no_grad():
        s, _ = nll_of(s, warm)
        s, a1 = nll_of(s, A)
        s, b = nll_of(s, B)
        s, a2 = nll_of(s, A)
    mem = s.da_memory[0][model.online_da_target[0] > 0] if online and s.da_memory.numel() else None
    return {'A_first': a1, 'B': b, 'A_again': a2, 'savings': a1 - a2,
            'memory_sd': float(mem.std()) if mem is not None else None}


def summary(rows, key):
    v = np.array([r[key] for r in rows])
    return {'min': float(v.min()), 'median': float(np.median(v)), 'mean': float(v.mean()), 'max': float(v.max())}


report = {'fork': str(fork), 'online_rule': online, 'pairs': pairs}
conditions = ['live'] + (['frozen'] if online else [])
for cond in conditions:
    if cond == 'frozen':
        model.online_da_eta.data.zero_()
        model.online_da_logit_forget.data.fill_(-60.0)
    rows = [run(p) for p in range(pairs)]
    report[cond] = {'rows': rows, 'savings': summary(rows, 'savings'), 'A_first': summary(rows, 'A_first'),
                    'A_again': summary(rows, 'A_again')}
    print(cond, {k: {kk: round(vv, 4) for kk, vv in report[cond][k].items()} for k in ('savings', 'A_first', 'A_again')}, flush=True)
if online:
    d = np.array([l['savings'] - f['savings'] for l, f in zip(report['live']['rows'], report['frozen']['rows'])])
    report['savings_live_minus_frozen'] = {'min': float(d.min()), 'median': float(np.median(d)), 'mean': float(d.mean()), 'max': float(d.max())}
    print('savings live - frozen', {k: round(v, 4) for k, v in report['savings_live_minus_frozen'].items()}, flush=True)
json.dump(report, open(fork / 'online_revisit.json', 'w'), indent=1)
