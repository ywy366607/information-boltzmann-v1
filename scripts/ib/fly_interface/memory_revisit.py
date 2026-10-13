"""What does the O(1) online memory itself learn beyond the BPTT window?  (online_revisit.py for the current route: the
compartment memory phi on the Kenyon -> MBON synapses, and the older release-gain rule at Kenyon terminals if present;
world-paced reading, 8 ticks per word, pulse input, eyes open.)  Fixed slow weights, the physical state carried
through, never reset.  For several pairs of passages (A, B: 256 words each from different parts of the validation
text): warm up 64 words, read A, B, then A again; revisit savings = CE(A first) - CE(A again), CE of the next word read
at the last tick of each word.  Conditions:
  live     the online memories keep being written (as in life)
  frozen   no writes during the test (eta = 0, no forgetting): the memories stay as they were at the start
live - frozen is what the online rule learns in the stream with every slow weight fixed.  Also phi's spread and how
far reading A moved it.
Usage: python memory_revisit.py <checkpoint dir> [out.json] [pairs]
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else fork / 'memory_revisit.json'
pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
quiet = torch.zeros(1, model.n_neurons, device='cuda')
comp = getattr(model, 'comp_eta', None) is not None
terminal = getattr(model, 'online_da_eta', None) is not None
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
WARM, LEN, K = 64, 256, 8


def read(s, seq):
    ces = []
    for i in range(len(seq) - 1):
        token = torch.tensor([int(seq[i])], device='cuda')
        for t in range(K):
            if t == 0:
                drive, baseline = fl.write_input(model, s, token)
                s = fl.step_fly_physical_tick(model, s, token, drive, baseline, opts, base_rates=opts['base_rates'])
            else:
                baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
                s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
        lat = fl.read_fly_motor(model, s)
        ces.append(float(F.cross_entropy(model.decoder(model.read_norm(lat)).float(),
                                         torch.tensor([int(seq[i + 1])], device='cuda'))))
        s = fl.expect_next(model, s, lat)
    return s, float(np.mean(ces))


def phi(s):
    m = getattr(s, 'comp_memory', torch.empty(0))
    return m[0].clone() if m.numel() else None


def run(pair):
    a0 = 70_000 + pair * 4_000
    warm = np.array(val[a0 - WARM:a0], dtype=np.int64)
    A = np.array(val[a0:a0 + LEN + 1], dtype=np.int64)
    B = np.array(val[a0 + 2_000:a0 + 2_000 + LEN + 1], dtype=np.int64)
    torch.manual_seed(pair)
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    with torch.no_grad():
        s, _ = read(s, warm)
        p0 = phi(s)
        s, a1 = read(s, A)
        p1 = phi(s)
        s, b = read(s, B)
        s, a2 = read(s, A)
    row = {'A_first': a1, 'B': b, 'A_again': a2, 'savings': a1 - a2}
    if p0 is not None:
        row.update({'phi_sd': float(p1.std()), 'phi_absmax': float(p1.abs().max()),
                    'phi_moved_by_A_rms': float((p1 - p0).pow(2).mean().sqrt())})
    return row


def summary(rows, key):
    v = np.array([r[key] for r in rows])
    return {'min': float(v.min()), 'median': float(np.median(v)), 'mean': float(v.mean()), 'max': float(v.max())}


report = {'fork': str(fork), 'compartment_memory': comp, 'terminal_rule': terminal, 'pairs': pairs, 'ticks_per_word': K}
conditions = ['live'] + (['frozen'] if comp or terminal else [])
for cond in conditions:
    if cond == 'frozen':
        if comp:
            model.comp_eta.data.zero_(); model.comp_logit_forget.data.fill_(-60.0)
        if terminal:
            model.online_da_eta.data.zero_(); model.online_da_logit_forget.data.fill_(-60.0)
    rows = [run(p) for p in range(pairs)]
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    report[cond] = {'rows': rows, **{k: summary(rows, k) for k in keys}}
    print(cond, {k: {kk: round(vv, 4) for kk, vv in report[cond][k].items()} for k in ('savings', 'A_first', 'A_again')}, flush=True)
if len(conditions) == 2:
    d = np.array([l['savings'] - f['savings'] for l, f in zip(report['live']['rows'], report['frozen']['rows'])])
    report['savings_live_minus_frozen'] = {'min': float(d.min()), 'median': float(np.median(d)), 'mean': float(d.mean()), 'max': float(d.max())}
    print('savings live - frozen', {k: round(v, 4) for k, v in report['savings_live_minus_frozen'].items()}, flush=True)
json.dump(report, open(out, 'w'), indent=1)
