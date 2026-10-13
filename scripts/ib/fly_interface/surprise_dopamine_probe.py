"""Does the brain's own surprise reach its dopamine?  (mainline link 2: prediction error -> phasic dopamine -> precision)
Fixed weights, the training world (paced intervals, input held), 4 places x 512 words after 256 words of burn-in from
the inherited state.  Per word: its surprisal (cross-entropy of the word under the expectation read just before it
arrived) and, at the end of the word's own ticks, (a) the dopamine cells' firing over the spike ring and (b) the
phasic dopamine at the targets, mean of (fast - slow) / slow concentration (fast tau ~50 ticks integrates the word).  Reports the Spearman correlation of surprisal with each (same word, and
the next word), the dopamine's word-to-word variability, and the [min | median | mean | max] over places.
Usage: python surprise_dopamine_probe.py <checkpoint dir> [out.json]
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
HOLD = bool(cfg.get('hold_input'))
K = int(cfg.get('stream_ticks') or 8)
pmean, pmin = float(cfg.get('stream_pace_mean') or 0), float(cfg.get('stream_pace_min') or 0)
graph = np.load(ROOT / cfg['graph'], allow_pickle=True)
dan = torch.as_tensor(graph['dan_indices'].astype(np.int64), device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
BURN, SCORE = 256, int(os.environ.get('DA_SCORE', 512))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else float('nan')


per_place = []
with torch.no_grad():
    for place in range(int(os.environ.get('DA_PLACES', 4))):
        rng = np.random.default_rng(2000 + place)
        start = 90_000 + place * 4_000
        words = np.array(val[start:start + BURN + SCORE + 1], dtype=np.int64)
        s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
        z = None
        surprise, dan_rate, phasic = [], [], []
        pace = pmean
        for i in range(BURN + SCORE):
            if pmean > 0 and i % 32 == 0:
                pace = float(np.exp(rng.uniform(np.log(pmin), np.log(pmean)))) if 0 < pmin < pmean else pmean
            k = int(rng.geometric(1.0 / pace)) if pmean > 0 else K
            token = torch.tensor([int(words[i])], device='cuda')
            if i >= BURN and z is not None:
                surprise.append(float(F.cross_entropy(model.decoder(model.read_norm(z)).float(), token)))
            # the word's ticks as the world gives them (one call: the efference copy is updated once per word, as in
            # training); the dopamine concentrations integrate over the ticks (fast tau ~50), so the state at the end of
            # the word carries its phasic response; the dopamine cells' firing from the spike ring (last ticks)
            s, z = fl.advance_fly_token_stream(model, s, token, ticks=k, hold_input=HOLD, writer_baseline_clock='physical', **opts)
            rates = [float(torch.stack([r[0] for r in s.ring]).index_select(1, dan).gt(0).float().mean())]
            ph = []
            if s.dopamine.numel():
                fast, slow = s.dopamine[0], s.dopamine[1]
                m = slow > 0
                ph.append(float(((fast[m] - slow[m]) / slow[m]).mean()) if bool(m.any()) else 0.0)
            if i >= BURN and len(surprise) == len(dan_rate) + 1:
                dan_rate.append(float(np.mean(rates)))
                phasic.append(float(np.mean(ph)) if ph else float('nan'))
        n = min(len(surprise), len(dan_rate))
        su, dr, phs = np.array(surprise[:n]), np.array(dan_rate[:n]), np.array(phasic[:n])
        per_place.append({'rho_surprise_dan_rate': spearman(su, dr), 'rho_surprise_phasic': spearman(su, phs),
                          'rho_surprise_next_phasic': spearman(su[:-1], phs[1:]),
                          'phasic_sd_words': float(np.nanstd(phs)), 'dan_rate_mean': float(dr.mean()),
                          'surprise_mean': float(su.mean())})
q = lambda v: [round(float(np.min(v)), 4), round(float(np.median(v)), 4), round(float(np.mean(v)), 4), round(float(np.max(v)), 4)]
rep = {'fork': str(fork)}
for key in per_place[0]:
    rep[key] = q([p[key] for p in per_place])
print(json.dumps(rep), flush=True)
if len(sys.argv) > 2:
    json.dump(rep, open(sys.argv[2], 'w'), indent=1)
