"""Does the central-brain core contribute its specific activity, or only a background drive?

Champion protocol (adaptive admission), 256 burn-in + 768 scored validation tokens, paired with the audit.  In the
replay arms every tick, before transmission, the core neurons' entries of the delayed pulse ring are replaced by core
pulses recorded elsewhere; the core's own membranes still evolve, but what it SENDS is replaced:
  self    the same run's own core pulses (plumbing check: must equal the full individual exactly)
  rest    pulses recorded while the brain idled with no input (same dynamics statistics, unrelated to the text)
  other   pulses recorded while it processed a different stretch of text (working statistics, wrong content)
"""
import sys, json, itertools, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
import information_boltzmann.core.fly_bptt_learning as fl
from scripts.ib.inspect_fly_bptt_credit import move_state

ROOT = Path('D:/information-boltzmann-v1')
BURN, SCORE = 256, 768
saved = torch.load('E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt', map_location='cpu', mmap=True, weights_only=False)
cfg, L = saved['config'], saved['learner']
model = FlyReservoirLM(ROOT / cfg['graph'], vocab_size=50257, d_model=cfg['d_model'], injection='topographic',
                       read_surface='output', synapse_model='coba', use_alif=True, use_stp=True,
                       decoder_bias=cfg['decoder_bias'], detach_reset=cfg.get('detach_reset', False),
                       surrogate_mode=cfg.get('surrogate_mode', 'absolute'), transmission_mode=cfg.get('transmission_mode', 'atomic'))
edges = ('edge_weight_e', 'edge_weight_i')
model.load_state_dict({k: v for k, v in saved['model'].items() if k not in edges}, strict=False)
model = model.cuda().eval().requires_grad_(False)
bufs = dict(model.named_buffers())
for k in edges:
    bufs[k].copy_(saved['model'][k])
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
kw = dict(min_settle_ticks=int(L.get('min_settle_ticks', 3)), max_settle_ticks=int(L.get('settle_ticks', 14)) or 14,
          flux_baseline=float(L.get('flux_baseline', 0.048)), writer_baseline_clock=L.get('writer_baseline_clock', 'physical'))
g = np.load(ROOT / cfg['graph'])
names = [str(x) for x in g['superclass_names']]
core = torch.as_tensor(np.flatnonzero(g['superclass_id'] == names.index('cb_intrinsic')), device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
cursor = int(saved['val_cursor'])
targets = np.array(val[cursor:cursor + BURN + SCORE], dtype=np.int64)
inputs = np.concatenate(([L['previous_token']], targets[:-1]))
start_state = lambda: move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')

original_step = fl.step_fly_physical_tick
record, replay = None, None         # list of recorded core pulse vectors / iterator over a recording


def step(model_, current, *a, **k):
    """One physical tick; optionally replace the core's outgoing pulses, optionally record what the core sent."""
    if replay is not None:
        ring = tuple(r.clone() for r in current.ring)
        for slot in range(4):
            ring[slot][0, core] = replay_ring[slot]
        current = fl.FlyPhysicalState(**{**current.state_dict(), 'ring': ring})
    out = original_step(model_, current, *a, **k)
    if record is not None:
        record.append(out.ring[0][0, core].cpu())
    if replay is not None:
        replay_ring.insert(0, next(replay).cuda().float())
        replay_ring.pop()
    return out


fl.step_fly_physical_tick = step


def run(tokens, nexts, state, score_from):
    scores, ticks = [], 0
    with torch.no_grad():
        for i, tok in enumerate(tokens):
            box = [None]
            state, z = fl.advance_fly_token_adaptive(model, state, torch.tensor([int(tok)], device='cuda'), box, **kw, **opts)
            ticks += 1 + box[0]
            if i >= score_from:
                logits = model.decoder(model.read_norm(z))
                scores.append(float(F.cross_entropy(logits.float(), torch.tensor([int(nexts[i])], device='cuda'))))
    return np.array(scores), ticks, state


# 1. full individual, recording what the core sends
record = []
full, full_ticks, end_state = run(inputs, targets, start_state(), BURN)
rec_self = record
# 2. the same brain idling with no input: record the core
record = []
with torch.no_grad():
    s = end_state
    for _ in range(full_ticks + 500):
        s = fl.step_fly_physical_tick(model, s, None, torch.zeros_like(s.h), model.topographic_writer.lambda_adapt * s.baseline, opts)
rec_rest = record
# 3. the same brain processing a different stretch of text: record the core
record = []
other_targets = np.array(val[cursor + 20000:cursor + 20000 + BURN + SCORE + 400], dtype=np.int64)
other_inputs = np.concatenate(([int(val[cursor + 19999])], other_targets[:-1]))
run(other_inputs, other_targets, start_state(), 10 ** 9)
rec_other = record
record = None
print(f'full: NLL {full.mean():.4f}, {full_ticks} ticks; recordings: self {len(rec_self)}, rest {len(rec_rest)}, other {len(rec_other)} ticks', flush=True)

def jaccard(a, b):
    a, b = a > 0, b > 0
    return float((a & b).sum()) / max(1, float((a | b).sum()))
start = 2000                                 # well past the shared starting state
same = [jaccard(rec_self[t], rec_other[t]) for t in range(start, min(len(rec_self), len(rec_other)), 50)]
lagged = [jaccard(rec_self[t], rec_self[t + 1]) for t in range(start, len(rec_self) - 1, 50)]
shift = [jaccard(rec_self[t], rec_other[t + 7]) for t in range(start, min(len(rec_self), len(rec_other)) - 7, 50)]
print(f'core spiking sets, same tick index, text A vs unrelated text B: Jaccard {np.mean(same):.3f}; '
      f'A vs B shifted by 7 ticks {np.mean(shift):.3f}; A tick vs A next tick {np.mean(lagged):.3f}', flush=True)
if '--only-compare' in sys.argv:
    sys.exit()
report = {'full_nll': float(full.mean()), 'arms': {}}
for label, rec in (('self', rec_self), ('rest', rec_rest), ('other', rec_other)):
    # the replay ring starts with what the core sent in the 4 ticks before the run (the full run's initial ring)
    init = start_state()
    replay_ring = [init.ring[slot][0, core].clone() for slot in range(4)]
    replay = itertools.cycle(rec)
    scores, ticks, _ = run(inputs, targets, start_state(), BURN)
    replay = None
    d = scores - full
    report['arms'][label] = {'nll': float(scores.mean()), 'delta': float(d.mean()), 'ticks': ticks,
                             'delta_blocks_128': [float(d[j:j + 128].mean()) for j in range(0, len(d), 128)],
                             'bitwise_equal_to_full': bool(np.array_equal(scores, full))}
    print(label, {k: v for k, v in report['arms'][label].items() if k != 'delta_blocks_128'}, flush=True)
json.dump(report, open(ROOT / 'scratch/fly_stream/core_replay.json', 'w'), indent=1)
