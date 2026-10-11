"""Can the champion give a quick guess?  Inside its OWN rhythm (flux admission, unchanged), decode its own read at every
tick of every token, and compare the NLL of an early read with the read it actually made.  Then: does its silent
state carry content?  After reading, cut the input and keep reading the motor surface during silence; score those
reads against the real continuation of the text.
"""
import sys, json, math, numpy as np, torch, torch.nn.functional as F
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
min_t, max_t, flux0 = int(L.get('min_settle_ticks', 3)), int(L.get('settle_ticks', 14)) or 14, float(L.get('flux_baseline', 0.048))
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
cursor = int(saved['val_cursor'])
targets = np.array(val[cursor:cursor + BURN + SCORE + 64], dtype=np.int64)
inputs = np.concatenate(([L['previous_token']], targets[:-1]))
writer = model.topographic_writer


def ce_of(state, target):
    z = model.output_read(state.h[:, model.read_indices])
    return float(F.cross_entropy(model.decoder(model.read_norm(z)).float(), torch.tensor([int(target)], device='cuda')))


SILENT_TICKS = (1, 2, 4, 8, 16, 32, 64)
silence_rows = {t: [] for t in SILENT_TICKS}


def probe_silence(state, upcoming, after):
    """Copy the live state, cut the input, read the motor surface during silence; score each read against the token
    the text actually continues with (upcoming[0]) and the next 8, and against the token after it (after[0])."""
    c = state.detached()
    c = fl.FlyPhysicalState(**{k: tuple(t.clone() for t in v) if k == 'ring' else v.clone() for k, v in c.state_dict().items()})
    for t in range(1, max(SILENT_TICKS) + 1):
        c = fl.step_fly_physical_tick(model, c, None, torch.zeros_like(c.h), writer.lambda_adapt * c.baseline, opts)
        if t in SILENT_TICKS:
            logp = F.log_softmax(model.decoder(model.read_norm(model.output_read(c.h[:, model.read_indices]))).float(), -1)[0]
            silence_rows[t].append((float(-logp[int(upcoming[0])]), float(-logp[torch.as_tensor(upcoming, device='cuda')].mean()),
                                    float(-logp[int(after[0])]), float(-(logp.exp() * logp).sum())))


per_tick = {}          # tick k (1 = input tick) -> list of (ce at k, ce at the champion's own stop)
stops = []
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
sqrtN = math.sqrt(model.n_neurons)
with torch.no_grad():
    for i in range(BURN + SCORE):
        token = torch.tensor([int(inputs[i])], device='cuda')
        drive, base = writer.forward_with_state(model.embedding(token), s.h, s.baseline)
        s = fl.step_fly_physical_tick(model, s, token, drive, base, opts)
        ces = [ce_of(s, targets[i])] if i >= BURN else []
        prev_h, prev_flux = s.h.clone(), None
        for t in range(1, max_t + 1):
            s = fl.step_fly_physical_tick(model, s, None, torch.zeros_like(s.h), writer.lambda_adapt * s.baseline, opts)
            if i >= BURN:
                ces.append(ce_of(s, targets[i]))
            flux = float(torch.linalg.vector_norm(s.h - prev_h)) / sqrtN
            prev_h = s.h.clone()
            if t >= min_t and (flux <= flux0 or (prev_flux is not None and flux >= prev_flux)):
                break
            prev_flux = flux
        if i >= BURN and (i - BURN) % 48 == 0 and i + 9 < len(targets):
            probe_silence(s, targets[i:i + 8], targets[i + 1:i + 9])
        if i >= BURN:
            stops.append(len(ces))
            for k, c in enumerate(ces, start=1):
                per_tick.setdefault(k, []).append((c, ces[-1]))
    end_state = s

final = np.array([per_tick[k][j][1] for k in [1] for j in range(len(per_tick[1]))])
print(f'champion, its own rhythm: NLL at its own stop {final.mean():.4f}; ticks per token mean {np.mean(stops):.2f} '
      f'(min {min(stops)}, max {max(stops)})')
print('forced early read at tick k (1 = input tick), on the tokens that reached tick k:')
rows = {}
for k in sorted(per_tick):
    a = np.array(per_tick[k])
    rows[k] = {'n': len(a), 'nll_at_k': float(a[:, 0].mean()), 'nll_at_own_stop': float(a[:, 1].mean()),
               'gap': float((a[:, 0] - a[:, 1]).mean())}
    print(f'  tick {k:2d}  n={len(a):4d}  NLL {a[:, 0].mean():.3f}  (own stop {a[:, 1].mean():.3f}, gap {rows[k]["gap"]:+.3f})')

unigram = np.bincount(np.array(val[:2_000_000], dtype=np.int64), minlength=50257) + 1.0
uni = -np.log(unigram / unigram.sum())
print(f'silence probes at {len(silence_rows[1])} points (input cut right after a token was read); unigram NLL of the '
      f'true next token {np.mean([uni[int(targets[i])] for i in range(BURN, BURN + SCORE, 48)]):.3f}:')
silent = {}
for t in SILENT_TICKS:
    a = np.array(silence_rows[t])
    silent[t] = {'true_next_nll': float(a[:, 0].mean()), 'true_next8_nll': float(a[:, 1].mean()),
                 'token_after_nll': float(a[:, 2].mean()), 'entropy': float(a[:, 3].mean())}
    print(f'  silent tick {t:3d}: true next token {a[:, 0].mean():.3f}  next 8 {a[:, 1].mean():.3f}  '
          f'token after {a[:, 2].mean():.3f}  entropy {a[:, 3].mean():.3f}')
json.dump({'per_tick': rows, 'stops': stops, 'silent': silent},
          open(ROOT / 'scratch/fly_stream/early_read.json', 'w'), indent=1)
