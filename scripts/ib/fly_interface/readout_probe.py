"""Is a readout without a window worth it?  Frozen champion brain; only the read is trained, scored by held-out NLL.

The champion reads text (1 input tick + 6 quiet ticks per token, never reset).  Along the continuous tick stream (it
does not restart at token boundaries) every motor membrane is passed through exponential filters
f_k <- a_k f_k + (1 - a_k) h with time constants 1, 2, 4, 8, 16, 32 ticks, plus the instantaneous membrane (k = 0).
Both reads use the champion's own output_read, read_norm and decoder:
  single  z = output_read(h)                      (the champion's read; output_read fine-tuned on the same data)
  bank    z = output_read(sum_k g_k * f_k)        (per-neuron gains g_k, init g_0 = 1, others 0: identical to single)
Fit on 3,500 tokens, held-out NLL on the last 1,000.
"""
import sys, json, math, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
import information_boltzmann.core.fly_bptt_learning as fl
from scripts.ib.inspect_fly_bptt_credit import move_state

ROOT = Path('D:/information-boltzmann-v1')
BURN, N, TRAIN = 200, 4500, 3500
TAUS = (1, 2, 4, 8, 16, 32)
SETTLE = int(sys.argv[1]) if len(sys.argv) > 1 else 6      # quiet ticks per token: 6 = champion-like, 3 = streaming k=4
saved = torch.load('E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt', map_location='cpu', mmap=True, weights_only=False)
cfg, L = saved['config'], saved['learner']
model = FlyReservoirLM(ROOT / cfg['graph'], vocab_size=50257, d_model=cfg['d_model'], injection='topographic',
                       read_surface='output', synapse_model='coba', use_alif=True, use_stp=True,
                       decoder_bias=cfg['decoder_bias'], detach_reset=cfg.get('detach_reset', False),
                       surrogate_mode=cfg.get('surrogate_mode', 'absolute'), transmission_mode=cfg.get('transmission_mode', 'atomic'))
edges = ('edge_weight_e', 'edge_weight_i')
model.load_state_dict({k: v for k, v in saved['model'].items() if k not in edges}, strict=False)
model = model.cuda().eval().requires_grad_(False)
for k in edges:
    dict(model.named_buffers())[k].copy_(saved['model'][k])
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
cursor = int(saved['val_cursor'])
toks = np.array(val[cursor:cursor + BURN + N + 1], dtype=np.int64)
decay = torch.tensor([math.exp(-1 / t) for t in TAUS], device='cuda')[:, None]
K = 1 + len(TAUS)
feats = torch.empty(N, K, model.n_read, dtype=torch.float16)
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
filt = s.h[:, model.read_indices].repeat(len(TAUS), 1)
with torch.no_grad():
    for i in range(BURN + N):
        s, ticks = fl.advance_fly_input_event(model, s, torch.tensor([int(toks[i])], device='cuda'), settle_ticks=SETTLE,
                                              writer_baseline_clock='physical', return_ticks=True, **opts)
        for st in ticks:
            filt = decay * filt + (1 - decay) * st.h[:, model.read_indices]
        if i >= BURN:
            feats[i - BURN, 0] = s.h[0, model.read_indices].half().cpu()
            feats[i - BURN, 1:] = filt.half().cpu()
targets = torch.as_tensor(toks[BURN + 1:BURN + N + 1])


def fit(use_bank, epochs=6):
    W = model.output_read.weight.detach().clone().requires_grad_(True)
    gains = torch.zeros(K, model.n_read, device='cuda')
    gains[0] = 1.0
    gains.requires_grad_(use_bank)
    groups = [{'params': [W], 'lr': 3e-4}] + ([{'params': [gains], 'lr': 1e-2}] if use_bank else [])
    opt = torch.optim.Adam(groups)

    def nll(idx):
        x = feats[idx].cuda().float()
        h = (gains[None] * x).sum(1) if use_bank else x[:, 0]
        logits = model.decoder(model.read_norm(F.linear(h, W)))
        return F.cross_entropy(logits.float(), targets[idx].cuda())

    with torch.no_grad():
        start = float(np.mean([float(nll(torch.arange(j, min(j + 250, N)))) for j in range(TRAIN, N, 250)]))
    curve = []
    for ep in range(epochs):
        perm = torch.randperm(TRAIN)
        for j in range(0, TRAIN, 128):
            opt.zero_grad()
            nll(perm[j:j + 128]).backward()
            opt.step()
        with torch.no_grad():
            curve.append(float(np.mean([float(nll(torch.arange(j, min(j + 250, N)))) for j in range(TRAIN, N, 250)])))
    used = gains.detach().abs().mean(1).tolist() if use_bank else None
    return start, curve, used


report = {}
for name, bank in (('single', False), ('bank', True)):
    start, curve, used = fit(bank)
    report[name] = {'held_out_nll_before': start, 'held_out_nll_by_epoch': curve, 'mean_abs_gain_by_filter': used}
    print(name, 'held-out NLL before', round(start, 4), 'by epoch', [round(c, 4) for c in curve],
          '' if used is None else f'gains (instant, tau {TAUS}): {[round(u, 3) for u in used]}', flush=True)
json.dump(report, open(ROOT / f'scratch/fly_stream/readout_probe_settle{SETTLE}.json', 'w'), indent=1)
