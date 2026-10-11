"""Inner life of a trained fork (fixed weights): silence content, core replay, released cells, ALIF.

Usage: python fork_analysis.py <fork dir with last.pt> [out.json]
The fork's own body is rebuilt from its config (go neuron, read bank, intrinsic drive) and its saved never-reset life
continues on fresh validation text with its own go-neuron rhythm (sampling seeded, identical across arms).
"""
import sys, json, math, itertools, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
import information_boltzmann.core.fly_bptt_learning as fl
from scripts.ib.inspect_fly_bptt_credit import move_state

ROOT = Path('D:/information-boltzmann-v1')
fork = Path(sys.argv[1])
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else fork / 'inner_life.json'
saved = torch.load(fork / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
cfg, L = saved['config'], saved['learner']
graph = ROOT / cfg['graph']
model = FlyReservoirLM(graph, vocab_size=50257, d_model=cfg['d_model'], injection='topographic', read_surface='output',
                       synapse_model='coba', use_alif=True, use_stp=True, decoder_bias=cfg['decoder_bias'],
                       detach_reset=cfg.get('detach_reset', False), surrogate_mode=cfg.get('surrogate_mode', 'absolute'),
                       transmission_mode=cfg.get('transmission_mode', 'atomic')).cuda()
if cfg.get('go_neuron') == 'virtual':
    fl.attach_go_neuron(model)
if cfg.get('read_bank'):
    fl.attach_read_bank(model)
if cfg.get('intrinsic'):
    fl.attach_intrinsic_drive(model, graph, graph.parent / 'body-annotations.feather')
edges = ('edge_weight_e', 'edge_weight_i')
weights = saved['model']
with torch.no_grad():
    for k in edges:
        getattr(model, k).copy_(weights[k])
    missing = model.load_state_dict({k: v for k, v in weights.items() if k not in edges}, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
model.eval().requires_grad_(False)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
g = np.load(graph)
names = [str(x) for x in g['superclass_names']]
core = torch.as_tensor(np.flatnonzero(g['superclass_id'] == names.index('cb_intrinsic')), device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
cursor = int(saved['val_cursor'])
BURN, SCORE = 256, 768
targets = np.array(val[cursor:cursor + BURN + SCORE + 16], dtype=np.int64)
inputs = np.concatenate(([L['previous_token']], targets[:-1]))
start_state = lambda: move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
max_ticks = int(L.get('go_max_ticks', 32))

original_step = fl.step_fly_physical_tick
record = replay = None
replay_ring = None


def step(model_, current, *a, **k):
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


def advance(state, token):
    if cfg.get('go_neuron'):
        state, lat, _, _, _ = fl.advance_fly_token_go(model, state, token, [], max_ticks=max_ticks,
                                                      writer_baseline_clock='physical', **opts)
        return state, lat[-1:], lat.shape[0]
    box = [None]
    state, z = fl.advance_fly_token_adaptive(model, state, token, box, writer_baseline_clock='physical', **opts)
    return state, z, 1 + box[0]


def run(tokens, nexts, state, score_from, silence_every=None, seed=0):
    torch.manual_seed(seed)
    scores, ticks, probes = [], 0, []
    with torch.no_grad():
        for i, tok in enumerate(tokens):
            state, z, used = advance(state, torch.tensor([int(tok)], device='cuda'))
            ticks += used
            if i >= score_from:
                logits = model.decoder(model.read_norm(z))
                scores.append(float(F.cross_entropy(logits.float(), torch.tensor([int(nexts[i])], device='cuda'))))
                if silence_every and (i - score_from) % silence_every == 0 and i + 1 < len(nexts):
                    probes.append(silence_probe(state, int(nexts[i])))
    return np.array(scores), ticks, state, probes


SILENT = (1, 2, 4, 8, 16, 32, 64)


def silence_probe(state, true_next):
    global record
    saved_record, record = record, None
    c = fl.FlyPhysicalState(**{k: tuple(t.clone() for t in v) if k == 'ring' else v.clone()
                               for k, v in state.detached().state_dict().items()})
    row = {}
    for t in range(1, max(SILENT) + 1):
        c = step(model, c, None, torch.zeros_like(c.h), writer.lambda_adapt * c.baseline, opts)
        if t in SILENT:
            logp = F.log_softmax(model.decoder(model.read_norm(fl.read_fly_motor(model, c))).float(), -1)[0]
            row[t] = (float(-logp[true_next]), float(-(logp.exp() * logp).sum()),
                      float((c.ring[0] > 0).float().mean()))
    record = saved_record
    return row


report = {'fork': str(fork), 'config': {k: cfg.get(k) for k in ('go_neuron', 'read_bank', 'intrinsic')}}
# 1. live reading, recording the core; silence probes every 48 tokens
record = []
full, full_ticks, end_state, probes = run(inputs, targets, start_state(), BURN, silence_every=48)
rec_self = record
report['nll'] = float(full.mean())
report['ticks_per_token'] = full_ticks / len(inputs)
unigram = np.bincount(np.array(val[:2_000_000], dtype=np.int64), minlength=50257) + 1.0
uni = -np.log(unigram / unigram.sum())
report['silence'] = {'unigram_true_next': float(np.mean([uni[int(targets[i])] for i in range(BURN, BURN + SCORE, 48)]))}
for t in SILENT:
    a = np.array([p[t] for p in probes])
    report['silence'][t] = {'true_next_nll': float(a[:, 0].mean()), 'entropy': float(a[:, 1].mean()),
                            'firing': float(a[:, 2].mean())}
print('reading NLL', round(report['nll'], 4), 'ticks/token', round(report['ticks_per_token'], 2), flush=True)
print('silence (true next NLL / entropy / firing):', {t: tuple(round(v, 3) for v in report['silence'][t].values())
                                                       for t in SILENT}, 'unigram', round(report['silence']['unigram_true_next'], 3))
if '--only-intrinsic' in sys.argv:
    record = None
    rec_self = []
# 2. the core while processing unrelated text, then replay arms
record = [] if '--only-intrinsic' not in sys.argv else None
other_t = np.array(val[cursor + 20000:cursor + 20000 + BURN + SCORE + 400], dtype=np.int64)
other_i = np.concatenate(([int(val[cursor + 19999])], other_t[:-1]))
if '--only-intrinsic' not in sys.argv:
    run(other_i, other_t, start_state(), 10 ** 9)
rec_other = record
record = None
report['replay'] = {}
for label, rec in ((('self', rec_self), ('other', rec_other)) if '--only-intrinsic' not in sys.argv else ()):
    init = start_state()
    replay_ring = [init.ring[slot][0, core].clone() for slot in range(4)]
    replay = itertools.cycle(rec)
    scores, _, _, _ = run(inputs, targets, start_state(), BURN)
    replay = None
    report['replay'][label] = float((scores - full).mean())
print('core replay delta (self should be 0):', {k: round(v, 4) for k, v in report['replay'].items()}, flush=True)
# 3. released cells
if getattr(model, 'intrinsic_kappa', None) is not None:
    kap = model.intrinsic_kappa.detach()
    report['intrinsic'] = {'n': int(kap.numel()), 'kappa_min_median_mean_max': [float(kap.min()), float(kap.median()),
                           float(kap.mean()), float(kap.max())], 'above_rheobase': float((kap > 1).float().mean())}
    print('released cells:', report['intrinsic'], flush=True)
# 3b. switch the released cells' intrinsic current off (kappa = 0), and replace their spikes' content
if getattr(model, 'intrinsic_kappa', None) is not None:
    kappa_saved = model.intrinsic_kappa.detach().clone()
    model.intrinsic_kappa.data.zero_()
    off, _, _, _ = run(inputs, targets, start_state(), BURN)
    model.intrinsic_kappa.data.copy_(kappa_saved)
    report['intrinsic_off_delta'] = float((off - full).mean())
    model.intrinsic_kappa.data.copy_(kappa_saved.mean().expand_as(kappa_saved))
    uniform, _, _, _ = run(inputs, targets, start_state(), BURN)
    model.intrinsic_kappa.data.copy_(kappa_saved)
    report['intrinsic_uniform_kappa_delta'] = float((uniform - full).mean())
    print('released cells off:', round(report['intrinsic_off_delta'], 4), ' learned kappa replaced by its mean:',
          round(report['intrinsic_uniform_kappa_delta'], 4), flush=True)
    if '--only-intrinsic' in sys.argv:
        json.dump(report, open(out_path, 'w'), indent=1)
        sys.exit()
# 4. ALIF causality on this individual
alif = model.get_alif_params()
opts['alif_params'] = (alif[0], torch.zeros_like(alif[1]))
no_alif, _, _, _ = run(inputs, targets, start_state(), BURN)
opts['alif_params'] = alif
report['no_alif_delta'] = float((no_alif - full).mean())
print('no ALIF delta', round(report['no_alif_delta'], 4), flush=True)
json.dump(report, open(out_path, 'w'), indent=1)
