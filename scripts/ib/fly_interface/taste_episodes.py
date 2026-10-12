"""Taste world, arm A0 zero-shot: can the birth wiring plus the compartment memory learn a fresh cue -> valence mapping
within one 32-token window?  (Design: subagent colleague, FLY_CONSEQUENCE_WORLD_DESIGN_20261012.html.)  Fixed
weights, no gradient; a sweep of the memory's learning rate eta (forgetting kappa = |eta|).

Streaming clock, 8 ticks per token, one episode = 32 tokens = 256 ticks:
  tokens 0-3 cue A x4 | 4-7 distractors | 8-11 cue B x4 | 12-15 distractors (their last 32 ticks: baseline)
  16-19 test X1 x4 | 20-23 '?' x4 (read window ticks 128-192) | 24-27 test X2 x4 | 28-31 '?' x4 (read 192-256)
Each episode draws two cues from a pool of single-token nouns and makes one of them sweet (PAM cells driven) and the
other bitter (PPL1 cells driven): +DRIVE rheobase during ticks 16-40 of the cue's 32-tick block (nominal current,
after the input gain).  Test order random.  Between episodes the brain reads 32 tokens of text (same clock).
Act readout from the MBONs, fixed by the connectome: v_j = (Q_j - P_j) / (Q_j + P_j) from the PAM (P) and PPL1 (Q)
dopamine each MBON receives (PPL1 compartments approach +, PAM compartments avoid -); dn_j = spikes in the read window
- 2 x spikes in the baseline; A = sum v_j dn_j / sum |v_j|; go if A > 0.  Target: go for the sweet cue.
Every episode is also run from the same starting state with the memory frozen (eta = kappa = 0); the paired
difference isolates what the memory did.  Mechanism: in PAM-type MBONs, (response with memory - frozen) for the
sweet test cue vs for the bitter one.  Text head: logit(" yes") - logit(" no") at the read time, for reference.
Options: 'hold' holds each cue token on the sensory surface for all its ticks (a sustained odour; the writer's own
adaptation makes it a decaying innovation) instead of one pulse per token; 'etas=0.1,0.5' picks the sweep.
Mechanism measures: the Kenyon cells active while a cue is paired vs while the same cue (and the other cue) is tested
(Jaccard), and the input the plastic synapses give the MBONs during each test window (with memory - frozen).
Usage: python taste_episodes.py <fork dir> [episodes per eta] [hold] [etas=...]
"""
import sys, json, math, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
from information_boltzmann.core.triton_synapse import execute_delayed_synaptic_transmission

ETAS = (0.02, 0.1, 0.5)
TICKS, DRIVE, YES, NO, QUESTION = 8, 8.0, 3763, 645, 30
NOUNS = ('apple bread river stone cloud horse candle garden window mountain flower tree bird fish table chair door '
         'house water fire light paper music island forest ocean train ship coin glass shirt shoe hat road bridge '
         'moon star sun rain snow wind sand rock grass leaf milk cheese honey sugar salt egg wolf bear lion tiger '
         'mouse snake king queen ring sword wheel tower').split()
fork = Path(sys.argv[1])
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 40
HOLD = 'hold' in sys.argv[3:]
for a in sys.argv[3:]:
    if a.startswith('etas='):
        ETAS = tuple(float(x) for x in a[5:].split(','))
model, cfg, L = load(fork)
graph = ROOT / cfg['graph']; annotations = graph.parent / 'body-annotations.feather'
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
N = model.n_neurons
g = np.load(graph, allow_pickle=True)
ann = pd.read_feather(annotations).set_index('bodyId').reindex(g['neuron_body_ids'])
cls, typ = ann['class'].astype(str).values, ann['type'].astype(str).values
is_dan = np.zeros(N, bool); is_dan[g['dan_indices']] = True
pam, ppl1 = np.char.startswith(typ.astype(str), 'PAM') & is_dan, np.char.startswith(typ.astype(str), 'PPL1') & is_dan
mbon = np.flatnonzero(cls == 'MBON')
dpre, dpost, dw = g['dan_edge_pre'], g['dan_edge_post'], np.abs(g['dan_edge_weight']).astype(np.float64)
P = np.bincount(dpost[pam[dpre]], weights=dw[pam[dpre]], minlength=N)[mbon]
Q = np.bincount(dpost[ppl1[dpre]], weights=dw[ppl1[dpre]], minlength=N)[mbon]
v = np.where(P + Q > 0, (Q - P) / np.maximum(P + Q, 1e-12), 0.0)
v_t = torch.as_tensor(v, device='cuda', dtype=torch.float32)
pam_type = torch.as_tensor(v < -0.6, device='cuda')
import tiktoken
enc = tiktoken.get_encoding('gpt2')
pool = [enc.encode(' ' + w)[0] for w in NOUNS if len(enc.encode(' ' + w)) == 1]
print('cue pool', len(pool), 'MBONs', len(mbon), 'PAM-type', int(pam_type.sum()), 'PPL1-type', int((v > 0.6).sum()), flush=True)

s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
mb_idx = torch.as_tensor(mbon, device='cuda')
kc_idx = torch.as_tensor(np.flatnonzero(cls == 'Kenyon_Cell'), device='cuda')
K = float(s.dopamine[1, mb_idx].mean()) if s.dopamine.numel() == 2 * N else None     # receptors at the MBONs' background
n_edges = fl.attach_compartment_memory(model, graph, annotations, eta=-ETAS[0], forget=ETAS[0], affinity=K)
print('plastic Kenyon -> MBON synapses', n_edges, 'affinity K at MBONs', K, flush=True)
rates0 = model.get_decay_rates()[0].expand(1, N).detach()
rheobase = (model.get_thresholds().expand(1, N).detach() * -torch.log(rates0.clamp(1e-5, 1 - 1e-7)))
drives = {'sweet': DRIVE * rheobase * torch.as_tensor(pam, device='cuda').float().unsqueeze(0),
          'bitter': DRIVE * rheobase * torch.as_tensor(ppl1, device='cuda').float().unsqueeze(0)}
writer = model.topographic_writer
quiet = torch.zeros(1, N, device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
text = np.array(val[80_000:200_000], dtype=np.int64)


def plastic_input(s):
    mem = s.comp_memory[0] if s.comp_memory.numel() else torch.zeros_like(model.comp_weight)
    return execute_delayed_synaptic_transmission(s.ring, model.comp_pre, model.comp_post, model.comp_weight * (1.0 + mem),
                                                 model.comp_splits)[0, mb_idx]


def run_tokens(s, tokens, drive_at=None, record=False, held=()):
    """Stream tokens at TICKS ticks each; drive_at: {tick: drive tensor}; held: tokens held for all their ticks.
    Returns the state and, when recording, per-tick MBON spikes, Kenyon spikes and plastic input to the MBONs."""
    rec, kc_rec, pin_rec, t = [], [], [], 0
    for tok in tokens:
        for k in range(TICKS):
            model.external_drive = None if drive_at is None else drive_at.get(t)
            if k == 0 or (HOLD and tok in held):
                drive, baseline = fl.write_input(model, s, torch.tensor([int(tok)], device='cuda'))
                s = fl.step_fly_physical_tick(model, s, torch.tensor([int(tok)], device='cuda'), drive, baseline, opts,
                                              base_rates=opts['base_rates'])
            else:
                baseline = writer.lambda_adapt * s.baseline if writer is not None else s.baseline
                s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
            if record:
                rec.append((s.ring[0][0, mb_idx] > 0).float())
                kc_rec.append(s.ring[0][0, kc_idx] > 0)
                pin_rec.append(plastic_input(s))
            t += 1
    model.external_drive = None
    if not record:
        return s, None
    return s, (torch.stack(rec), torch.stack(kc_rec), torch.stack(pin_rec))


def yes_no(s):
    lg = model.decoder(model.read_norm(fl.read_fly_motor(model, s))).float().reshape(-1)
    return float(lg[YES] - lg[NO])


def episode(s, cues, sweet_first, order, rng_pos):
    a, b = cues
    dist1 = text[rng_pos:rng_pos + 4]; dist2 = text[rng_pos + 4:rng_pos + 8]
    x1, x2 = (a, b) if order == 0 else (b, a)
    toks = [a] * 4 + list(dist1) + [b] * 4 + list(dist2) + [x1] * 4 + [QUESTION] * 4 + [x2] * 4 + [QUESTION] * 4
    va, vb = ('sweet', 'bitter') if sweet_first else ('bitter', 'sweet')
    drive_at = {t: drives[va] for t in range(16, 41)}
    drive_at.update({64 + t: drives[vb] for t in range(16, 41)})
    s_mid, rec = None, []
    # run in two halves so the text head can be read at the end of each test's '?' block
    s1, r1 = run_tokens(s, toks[:24], drive_at, record=True, held=(a, b))
    y1 = yes_no(s1)
    s2, r2 = run_tokens(s1, toks[24:], {t - 192: d for t, d in drive_at.items() if t >= 192}, record=True, held=(a, b))
    y2 = yes_no(s2)
    R = torch.cat([r1[0], r2[0]])                         # 256 x n_MBON
    KC = torch.cat([r1[1], r2[1]])                        # 256 x n_KC
    PIN = torch.cat([r1[2], r2[2]])                       # 256 x n_MBON
    base = R[96:128].sum(0) * 2
    resp1, resp2 = R[128:192].sum(0) - base, R[192:256].sum(0) - base
    sweet_cue = a if sweet_first else b
    kc_set = lambda lo, hi: KC[lo:hi].any(0)
    jac = lambda x, y: float((x & y).sum()) / max(float((x | y).sum()), 1.0)
    paired = {a: kc_set(0, 32), b: kc_set(64, 96)}
    out = []
    for x, other, resp, y, lo in ((x1, x2, resp1, y1, 128), (x2, x1, resp2, y2, 192)):
        A = float((v_t * resp).sum() / v_t.abs().sum())
        test_set = kc_set(lo, lo + 32)
        out.append({'sweet': x == sweet_cue, 'A': A, 'pam_type_resp': float(resp[pam_type].mean()), 'yes_minus_no': y,
                    'kc_same': jac(test_set, paired[x]), 'kc_other': jac(test_set, paired[other]),
                    'kc_active': float(test_set.float().mean()),
                    'pin_pam_type': float(PIN[lo:lo + 64][:, pam_type].sum(0).mean()),
                    'pin_ppl1_type': float(PIN[lo:lo + 64][:, v_t > 0.6].sum(0).mean())})
    return s2, out


def set_memory(eta, frozen=False):
    with torch.no_grad():
        model.comp_eta.fill_(0.0 if frozen else -eta)
        model.comp_logit_forget.fill_(-30.0 if frozen else math.log(eta / (1 - eta)))


rng = np.random.default_rng(0)
report = {'fork': str(fork), 'hold': HOLD, 'affinity_K': K, 'plastic_edges': n_edges, 'drive_rheobase': DRIVE, 'episodes_per_eta': EPISODES}
torch.manual_seed(0)
with torch.no_grad():
    pos = 0
    s, _ = run_tokens(s, text[pos:pos + 128]); pos += 128
    for eta in ETAS:
        rows = []
        for ep in range(EPISODES):
            s, _ = run_tokens(s, text[pos:pos + 32]); pos += 32                 # live in text between episodes
            cues = tuple(int(c) for c in rng.choice(pool, 2, replace=False))
            sweet_first, order, rp = bool(rng.integers(2)), int(rng.integers(2)), int(rng.integers(0, len(text) - 16))
            set_memory(eta, frozen=True)
            _, frozen = episode(s, cues, sweet_first, order, rp)
            set_memory(eta)
            s, live = episode(s, cues, sweet_first, order, rp)
            for lv, fz in zip(live, frozen):
                rows.append({'kc_same': lv['kc_same'], 'kc_other': lv['kc_other'], 'kc_active': lv['kc_active'],
                             'dpin_pam_type': lv['pin_pam_type'] - fz['pin_pam_type'],
                             'dpin_ppl1_type': lv['pin_ppl1_type'] - fz['pin_ppl1_type'],
                             'sweet': lv['sweet'], 'A': lv['A'], 'A_frozen': fz['A'], 'dA': lv['A'] - fz['A'],
                             'pam_type_effect': lv['pam_type_resp'] - fz['pam_type_resp'],
                             'yes_minus_no': lv['yes_minus_no'], 'phi_sd': float(s.comp_memory.std()),
                             'phi_absmax': float(s.comp_memory.abs().max())})
        R = pd.DataFrame(rows)
        acc = float(((R.A > 0) == R.sweet).mean()); acc_fz = float(((R.A_frozen > 0) == R.sweet).mean())
        acc_d = float(((R.dA > 0) == R.sweet).mean())
        yn_med = float(R.yes_minus_no.median()); acc_text = float(((R.yes_minus_no > yn_med) == R.sweet).mean())
        sw, bt = R[R.sweet].pam_type_effect, R[~R.sweet].pam_type_effect
        res = {'tests': len(R), 'accuracy_sign_A': acc, 'accuracy_frozen': acc_fz, 'accuracy_sign_dA_paired': acc_d,
               'accuracy_text_head_median_split': acc_text,
               'pam_type_effect_sweet_min_med_mean_max': [float(sw.min()), float(sw.median()), float(sw.mean()), float(sw.max())],
               'pam_type_effect_bitter_min_med_mean_max': [float(bt.min()), float(bt.median()), float(bt.mean()), float(bt.max())],
               'dA_sweet_mean': float(R[R.sweet].dA.mean()), 'dA_bitter_mean': float(R[~R.sweet].dA.mean()),
               'phi_sd_end': float(R.phi_sd.iloc[-1]), 'phi_absmax_end': float(R.phi_absmax.iloc[-1]),
               'kc_jaccard_same_cue_paired_vs_test_median': float(R.kc_same.median()),
               'kc_jaccard_other_cue_median': float(R.kc_other.median()), 'kc_active_fraction_test': float(R.kc_active.median()),
               'plastic_input_change_pam_type_sweet_vs_bitter': [float(R[R.sweet].dpin_pam_type.mean()), float(R[~R.sweet].dpin_pam_type.mean())],
               'plastic_input_change_ppl1_type_sweet_vs_bitter': [float(R[R.sweet].dpin_ppl1_type.mean()), float(R[~R.sweet].dpin_ppl1_type.mean())]}
        report[f'eta={eta:g}'] = res
        print(f'eta={eta:g}', json.dumps(res), flush=True)
json.dump(report, open(fork / ('taste_episodes_hold.json' if HOLD else 'taste_episodes.json'), 'w'), indent=1)
print('done', flush=True)
