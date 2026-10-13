"""Where does surprise stop on its way to dopamine?  (Diagnostic of the signal path, not a gate: fixed weights.)
The training world (paced intervals, input held), 4 places x 512 words after 256 words of burn-in from the inherited
state.  Per word: its surprisal (CE of the word under the read just before it arrived), and along the path
  writer innovation at its arrival (mean |packet(word) - reference|, reference = adapting baseline and expectation)
  -> olfactory receptors -> ALPN -> Kenyon cells -> MBONs -> dopamine cells (spikes over the last <= 4 ticks of the
  word, the spike ring) -> the dopamine cells' membrane relative to threshold -> phasic dopamine (fast - slow) / slow.
Spearman of each with the surprisal, [min | median | mean | max] over places.  The first link where it falls to ~0 is
where the path breaks.  Usage: python surprise_path_probe.py <checkpoint dir> [out.json]"""
import os, sys, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
HOLD = bool(cfg.get('hold_input'))
pmean, pmin = float(cfg.get('stream_pace_mean') or 0), float(cfg.get('stream_pace_min') or 0)
K = int(cfg.get('stream_ticks') or 8)
graph = np.load(ROOT / cfg['graph'], allow_pickle=True)
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(graph['neuron_body_ids'])
cls = ann['class'].astype(str).values
GROUPS = {'receptors': cls == 'olfactory', 'ALPN': cls == 'ALPN', 'Kenyon': cls == 'Kenyon_Cell', 'MBON': cls == 'MBON'}
G = {k: torch.as_tensor(np.flatnonzero(v), device='cuda') for k, v in GROUPS.items()}
G['DAN'] = torch.as_tensor(graph['dan_indices'].astype(np.int64), device='cuda')
theta = opts['thresholds'].expand(1, model.n_neurons)[0]
writer = model.topographic_writer
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
BURN, SCORE = 256, int(os.environ.get('PATH_SCORE', 512))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float('nan')


KEYS = ['innovation'] + [f'{k}_spikes' for k in ('receptors', 'ALPN', 'Kenyon', 'MBON', 'DAN')] + ['DAN_v_over_theta', 'phasic_DA']
per_place = []
with torch.no_grad():
    for place in range(int(os.environ.get('PATH_PLACES', 4))):
        rng = np.random.default_rng(3000 + place)
        start = 100_000 + place * 4_000
        words = np.array(val[start:start + BURN + SCORE + 1], dtype=np.int64)
        s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
        z = None
        rec = {k: [] for k in ['surprise'] + KEYS}
        pace = pmean
        for i in range(BURN + SCORE):
            if pmean > 0 and i % 32 == 0:
                pace = float(np.exp(rng.uniform(np.log(pmin), np.log(pmean)))) if 0 < pmin < pmean else pmean
            k = int(rng.geometric(1.0 / pace)) if pmean > 0 else K
            token = torch.tensor([int(words[i])], device='cuda')
            scored = i >= BURN and z is not None
            if scored:
                rec['surprise'].append(float(F.cross_entropy(model.decoder(model.read_norm(z)).float(), token)))
                packets = fl.word_packets(model, s, token)
                if packets is not None:
                    pred = packets[1]
                    beta = torch.sigmoid(model.writer_prediction_logit) if pred is not None else None
                    ref = s.baseline if pred is None else (1.0 - beta) * s.baseline + beta * pred
                    rec['innovation'].append(float((packets[0] - ref).abs().mean()))
                else:
                    rec['innovation'].append(float('nan'))
            s, z = fl.advance_fly_token_stream(model, s, token, ticks=k, hold_input=HOLD, writer_baseline_clock='physical', **opts)
            if scored:
                ring = torch.stack([r[0] for r in s.ring]).gt(0).float()          # [4, N]: the word's last ticks
                for g in ('receptors', 'ALPN', 'Kenyon', 'MBON', 'DAN'):
                    rec[f'{g}_spikes'].append(float(ring.index_select(1, G[g]).mean()))
                rec['DAN_v_over_theta'].append(float((s.h[0, G['DAN']] / theta[G['DAN']].clamp_min(1e-6)).mean()))
                if s.dopamine.numel():
                    fast, slow = s.dopamine[0], s.dopamine[1]
                    m = slow > 0
                    rec['phasic_DA'].append(float(((fast[m] - slow[m]) / slow[m]).mean()) if bool(m.any()) else 0.0)
                else:
                    rec['phasic_DA'].append(float('nan'))
        su = np.array(rec['surprise'])
        per_place.append({f'rho_surprise_{k}': spearman(su, rec[k]) for k in KEYS})
q = lambda v: [round(float(np.nanmin(v)), 4), round(float(np.nanmedian(v)), 4), round(float(np.nanmean(v)), 4), round(float(np.nanmax(v)), 4)]
rep = {'fork': str(fork), **{k: q([p[k] for p in per_place]) for k in per_place[0]}}
for k, v in rep.items():
    print(f'{k:>32}: {v}', flush=True)
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps(rep, indent=1), encoding='utf-8')
