"""Where is the information in the champion?  Fixed weights, linear (ridge) probes, held-out R^2.

The champion processes real validation text (1 input tick + 6 quiet ticks per token, never reset).  For every token we
keep the 7 ticks of its window from two populations of equal size:
  motor = the 2,333 read neurons (motor / descending / efferent: the action side of the blanket)
  core  = 2,333 random central-brain local interneurons (cb_intrinsic: ~70% of the brain's energy)
and probe these representations:
  V_k    membrane at tick k (k = 1..7; tick 1 = the input tick; tick 7 = the champion's read)
  S_k    spikes at tick k
  rate   spike count over the window (rate code)
  Vmean  window mean of the membrane (an integrator)
  Vtraj  the 7 membranes concatenated (population trajectory)
  Straj  the 7 spike vectors concatenated (spike timing pattern)
Targets: GPT-2 embedding of the CURRENT token (does the brain know what it sensed?) and of the NEXT token (does it
predict?).  Reported: held-out R^2 (fraction of embedding variance explained) and, for the current token,
nearest-embedding retrieval accuracy over the whole vocabulary.
References: next-token embedding predicted linearly from the current / previous token's own embedding.
"""
import sys, json, math, numpy as np, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
import information_boltzmann.core.fly_bptt_learning as fl
from scripts.ib.inspect_fly_bptt_credit import move_state

ROOT = Path('D:/information-boltzmann-v1')
BURN, N, TRAIN, VALID = 200, 4000, 3000, 500          # train = first 3000 (last 500 of them choose lambda), test = last 1000
TICKS = 7
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
g = np.load(ROOT / cfg['graph'])
names = [str(x) for x in g['superclass_names']]
rng = np.random.default_rng(0)
core_all = np.flatnonzero(g['superclass_id'] == names.index('cb_intrinsic'))
core = torch.as_tensor(np.sort(rng.choice(core_all, size=model.n_read, replace=False)), device='cuda')
motor = model.read_indices
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
cursor = int(saved['val_cursor'])
toks = np.array(val[cursor:cursor + BURN + N + 1], dtype=np.int64)

V = {r: torch.empty(N, TICKS, model.n_read, dtype=torch.float16) for r in ('motor', 'core')}
S = {r: torch.empty(N, TICKS, model.n_read, dtype=torch.uint8) for r in ('motor', 'core')}
nll = []
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
with torch.no_grad():
    for i in range(BURN + N):
        s, ticks = fl.advance_fly_input_event(model, s, torch.tensor([int(toks[i])], device='cuda'), settle_ticks=TICKS - 1,
                                              writer_baseline_clock='physical', return_ticks=True, **opts)
        if i >= BURN:
            j = i - BURN
            for k, st in enumerate(ticks):
                for r, idx in (('motor', motor), ('core', core)):
                    V[r][j, k] = st.h[0, idx].half().cpu()
                    S[r][j, k] = (st.ring[0][0, idx] > 0).to(torch.uint8).cpu()
            logits = model.decoder(model.read_norm(model.output_read(s.h[:, motor])))   # the champion's own read
            nll.append(float(torch.nn.functional.cross_entropy(logits.float(), torch.tensor([int(toks[i + 1])], device='cuda'))))
print(f'champion read on these tokens (fixed 6 quiet ticks): NLL {np.mean(nll):.4f}', flush=True)

emb = model.embedding.weight.detach().float()
cur_ids = torch.as_tensor(toks[BURN:BURN + N], device='cuda')
nxt_ids = torch.as_tensor(toks[BURN + 1:BURN + N + 1], device='cuda')
prv_ids = torch.as_tensor(toks[BURN - 1:BURN + N - 1], device='cuda')
targets = {'current': emb[cur_ids], 'next': emb[nxt_ids]}
emb_unit = torch.nn.functional.normalize(emb, dim=1)


def ridge(X, Y):
    """Dual ridge; lambda chosen on the last VALID training rows; returns test predictions and held-out R^2."""
    X = X.cuda().float()
    tr, te = slice(0, TRAIN), slice(TRAIN, N)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    keep = sd > 1e-6
    X = (X[:, keep] - mu[keep]) / sd[keep]
    ym = Y[tr].mean(0)
    Yc = Y - ym
    fit, vsl = slice(0, TRAIN - VALID), slice(TRAIN - VALID, TRAIN)
    best = None
    K_fit = X[fit] @ X[fit].T
    K_v = X[vsl] @ X[fit].T
    for lam in (1e-1, 1, 10, 100, 1e3, 1e4, 1e5):
        a = torch.linalg.solve(K_fit + lam * X.shape[1] / 1000 * torch.eye(K_fit.shape[0], device='cuda'), Yc[fit])
        err = ((K_v @ a - Yc[vsl]) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, lam)
    lam = best[1]
    K = X[tr] @ X[tr].T
    a = torch.linalg.solve(K + lam * X.shape[1] / 1000 * torch.eye(K.shape[0], device='cuda'), Yc[tr])
    pred = X[te] @ X[tr].T @ a + ym
    r2 = 1 - float(((pred - Y[te]) ** 2).sum() / ((Y[te] - Y[tr].mean(0)) ** 2).sum())
    return pred, r2


def retrieval(pred, ids):
    sims = torch.nn.functional.normalize(pred, dim=1) @ emb_unit.T
    return float((sims.argmax(1) == ids).float().mean())


results = {'champion_nll_fixed6': float(np.mean(nll)), 'probes': {}}
_, r_cur_next = ridge(targets['current'].cpu(), targets['next'])
_, r_prev_next = ridge(emb[prv_ids].cpu(), targets['next'])
results['reference'] = {'next_from_current_embedding': r_cur_next, 'next_from_previous_embedding': r_prev_next}
print(f'reference: next-token R2 from current embedding {r_cur_next:.3f}, from previous embedding {r_prev_next:.3f}', flush=True)

for region in ('motor', 'core'):
    v, sp = V[region].float(), S[region].float()
    feats = {f'V_{k + 1}': v[:, k] for k in range(TICKS)}
    feats.update({f'S_{k + 1}': sp[:, k] for k in range(TICKS)})
    feats.update(rate=sp.sum(1), Vmean=v.mean(1), Vtraj=v.reshape(N, -1), Straj=sp.reshape(N, -1))
    for name, X in feats.items():
        pc, rc = ridge(X, targets['current'])
        _, rn = ridge(X, targets['next'])
        acc = retrieval(pc, cur_ids[TRAIN:])
        results['probes'][f'{region}/{name}'] = {'current_r2': rc, 'next_r2': rn, 'current_retrieval': acc}
        print(f'{region:5s} {name:6s}  current R2 {rc:6.3f}  retrieval {100 * acc:5.1f}%   next R2 {rn:6.3f}', flush=True)
json.dump(results, open(ROOT / 'scratch/fly_stream/info_probe.json', 'w'), indent=1)
