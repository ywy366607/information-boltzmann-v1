"""Taste world, zero-shot checks (design: docs/information_boltzmann/FLY_CONSEQUENCE_WORLD_DESIGN_20261012.html).
Fixed weights, no learning; the individual is the newborn control grown with homeostasis (b2_control).

e  dopamine sources in the model's own projection graph (dan_edge_*): PAM vs PPL1 share per Kenyon cell and per MBON
a  taste -> dopamine cells: stimulating the sugar or the bitter second-order taste neurons (Yao & Scott 2022 labels),
   or the gustatory sensory neurons that feed only one of them, for 30 ticks from the living state; spikes of PAM,
   PPL1 and other dopamine cells against the same 30 ticks without stimulation (paired, same starting state)
b  can the mushroom body reach the act?  logit(" yes") - logit(" no") at the read after 15 quiet ticks with: nothing,
   PAM-type MBONs driven, PPL1-type MBONs driven, all MBONs silenced, and an equal random set of core cells driven
   (does any stimulation move it?); against the natural spread of the difference across positions
c  cue words: Kenyon cells active while each of 8 candidate nouns is read after a silence, in 6 contexts; Jaccard
   overlap between words and between contexts of the same word; realized ticks per cue (d: delay)
clock  read latents under real text, one word repeated, and silence: how much of their variance the brain makes
   without changing input
Stimulation is a current of AMP times each cell's own rheobase (dimensionless, like the homeostatic drive), as a
dose-response over AMPS; per dose also the driven cells' own rate and how many neurons change their spike count.
Usage: python taste_phase0.py <fork dir>
"""
import sys, json, dataclasses, numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

AMPS, TASTE_TICKS, ACT_TICKS, TRIALS = (2.0, 5.0, 10.0, 20.0), 30, 15, 12
YES, NO = 3763, 645
CUES = {' apple': 17180, ' river': 7850, ' stone': 7815, ' cloud': 6279, ' bread': 8509, ' horse': 8223,
        ' candle': 26839, ' garden': 11376}

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
max_ticks = int(L.get('go_max_ticks', 32))
N = model.n_neurons
graph = ROOT / cfg['graph']
g = np.load(graph, allow_pickle=True)
ids = g['neuron_body_ids']
ann = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(ids)
typ, cls, sup = ann['type'].astype(str).values, ann['class'].astype(str).values, ann['superclass'].astype(str).values
syn = ann['synonyms'].astype(str).values
pam, ppl1 = np.char.startswith(typ.astype(str), 'PAM'), np.char.startswith(typ.astype(str), 'PPL1')
is_dan = np.zeros(N, bool); is_dan[g['dan_indices']] = True
kc, mbon = cls == 'Kenyon_Cell', cls == 'MBON'
sugar_sel = np.array(['Sugar SEL' in x for x in syn]); bitter_sel = np.array(['Bitter-SEL' in x for x in syn])
report = {'fork': str(fork), 'amps_rheobase': AMPS}

# ---- e: dopamine sources in the model's own projection graph ----
dpre, dpost, dw = g['dan_edge_pre'], g['dan_edge_post'], np.abs(g['dan_edge_weight']).astype(np.float64)
def da_in(src):
    e = src[dpre]
    return np.bincount(dpost[e], weights=dw[e], minlength=N)
from_pam, from_ppl1 = da_in(pam), da_in(ppl1)
pct = lambda x: [float(v) for v in np.percentile(x, [0, 10, 50, 90, 100])] + [float(np.mean(x))]
e_rep = {}
for name, m in (('Kenyon', kc), ('MBON', mbon)):
    tot = from_pam + from_ppl1
    ok = m & (tot > 0)
    share = from_pam[ok] / tot[ok]
    e_rep[name] = {'cells': int(m.sum()), 'receiving': int(ok.sum()), 'pam_share_min_p10_med_p90_max_mean': pct(share),
                   'frac_pam_dominated(>0.8)': float((share > 0.8).mean()), 'frac_ppl1_dominated(<0.2)': float((share < 0.2).mean())}
mb_tot = from_pam + from_ppl1
mbon_pam = mbon & (mb_tot > 0) & (from_pam / np.maximum(mb_tot, 1e-12) > 0.8)
mbon_ppl1 = mbon & (mb_tot > 0) & (from_pam / np.maximum(mb_tot, 1e-12) < 0.2)
report['e_dopamine_sources'] = e_rep
print('e', json.dumps(e_rep), flush=True)

# gustatory sensory neurons that feed only one taste line
pre, post, w = g['edge_pre'], g['edge_post'], np.abs(g['edge_weight']).astype(np.float64)
gus = cls == 'gustatory'
def to_set(dst):
    e = gus[pre] & dst[post]
    return np.bincount(pre[e], weights=w[e], minlength=N)
ws, wb = to_set(sugar_sel), to_set(bitter_sel)
sugar_grn = gus & (ws > 0) & (wb == 0); bitter_grn = gus & (wb > 0) & (ws == 0)
groups_np = {'sugar SEL': sugar_sel, 'bitter SEL': bitter_sel, 'sugar GRN': sugar_grn, 'bitter GRN': bitter_grn}
report['taste_cells'] = {k: int(v.sum()) for k, v in groups_np.items()}
print('taste cells', report['taste_cells'], flush=True)
core_np = (sup == 'cb_intrinsic') & ~np.isin(cls, ['CX', 'Kenyon_Cell', 'MBON', 'DAN']) & ~is_dan
rng = np.random.default_rng(0)
core_rand = np.zeros(N, bool); core_rand[rng.choice(np.flatnonzero(core_np), int(mbon.sum()), replace=False)] = True

T = lambda m: torch.as_tensor(m, device='cuda')
masks = {'PAM': T(pam & is_dan), 'PPL1': T(ppl1 & is_dan), 'other DA': T(is_dan & ~pam & ~ppl1), 'Kenyon': T(kc), 'MBON': T(mbon)}

# ---- stimulation and spike counting by wrapping the physical tick ----
mode = {'drive': None, 'silence': None, 'count': None, 'amp': 2.0}
original_step = fl.step_fly_physical_tick
rates0 = model.get_decay_rates()[0].expand(1, N).detach()
theta0 = model.get_thresholds().expand(1, N).detach()
rheobase = theta0 * -torch.log(rates0.clamp(1e-5, 1 - 1e-7))


def step(model_, current, token, source, *a, **k):
    if mode['drive'] is not None:
        extra = mode['amp'] * rheobase * mode['drive'].unsqueeze(0)
        source = extra if source is None else source + extra
    out = original_step(model_, current, token, source, *a, **k)
    s0 = out.ring[0]
    if mode['silence'] is not None:
        s0 = torch.where(mode['silence'].unsqueeze(0), torch.zeros_like(s0), s0)
        out = dataclasses.replace(out, ring=(s0, *out.ring[1:]))
    if mode['count'] is not None:
        mode['count'] += (s0[0] > 0).float()
    return out


fl.step_fly_physical_tick = step
writer = model.topographic_writer
quiet = torch.zeros(1, N, device='cuda')


def quiet_ticks(s, n, drive=None, silence=None, count=False, amp=2.0):
    mode.update(amp=amp, drive=None if drive is None else drive.float(), silence=silence,
                count=torch.zeros(N, device='cuda') if count else None)
    for _ in range(n):
        baseline = writer.lambda_adapt * s.baseline if writer is not None else s.baseline
        s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
    c = mode['count']
    mode.update(drive=None, silence=None, count=None)
    return s, c


def read_token(s, tok, count=False):
    mode.update(count=torch.zeros(N, device='cuda') if count else None)
    s, lat, _, _, _, _ = fl.advance_fly_token_go(model, s, torch.tensor([int(tok)], device='cuda'), [], max_ticks=max_ticks,
                                                 writer_baseline_clock='physical', **opts)
    c = mode['count']; mode['count'] = None
    return s, lat, c


def yes_no(s):
    logits = model.decoder(model.read_norm(fl.read_fly_motor(model, s))).float().reshape(-1)
    return float(logits[YES] - logits[NO])


val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
text = np.array(val[70_000:72_000], dtype=np.int64)
torch.manual_seed(0)
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
with torch.no_grad():
    for tok in text[:256]:
        s, _, _ = read_token(s, tok)
    # ---- a and b at 24 positions along the text ----
    a_rows, b_rows = [], []
    pos = 256
    taste = {'sugar (SEL + GRN)': T(groups_np['sugar SEL'] | groups_np['sugar GRN']), 'bitter (SEL)': T(groups_np['bitter SEL'])}
    arms_b = (('PAM-type MBONs driven', 'drive', T(mbon_pam)), ('PPL1-type MBONs driven', 'drive', T(mbon_ppl1)),
              ('random core cells driven', 'drive', T(core_rand)))
    for trial in range(TRIALS):
        for tok in text[pos:pos + 24]:
            s, _, _ = read_token(s, tok)
        pos += 24
        row = {}
        _, base = quiet_ticks(s, TASTE_TICKS, count=True)
        for gname, gm in taste.items():
            for amp in AMPS:
                _, c = quiet_ticks(s, TASTE_TICKS, drive=gm, count=True, amp=amp)
                row[f'{gname}|x{amp:g}|driven Hz'] = float(c[gm].mean()) / TASTE_TICKS * 1000.0
                row[f'{gname}|x{amp:g}|neurons changed'] = float(((c - base).abs() > 0).sum() - gm.sum())
                for k in ('PAM', 'PPL1', 'other DA'):
                    m = masks[k]
                    row[f'{gname}|x{amp:g}|{k} dHz'] = float((c[m] - base[m]).mean()) / TASTE_TICKS * 1000.0
        a_rows.append(row)
        brow = {}
        s0, _ = quiet_ticks(s, ACT_TICKS)
        brow['none'] = yes_no(s0)
        s1, _ = quiet_ticks(s, ACT_TICKS, silence=T(mbon))
        brow['all MBONs silenced'] = yes_no(s1)
        for name, _, gm in arms_b:
            for amp in AMPS:
                s1, _ = quiet_ticks(s, ACT_TICKS, drive=gm, amp=amp)
                brow[f'{name} x{amp:g}'] = yes_no(s1)
        b_rows.append(brow)
        print('trial', trial, {k: round(v, 2) for k, v in row.items() if 'x20' in k},
              {k: round(v - brow['none'], 4) for k, v in brow.items() if 'x20' in k or 'silenced' in k}, flush=True)
    A = pd.DataFrame(a_rows)
    report['a_taste_dose_response'] = {c: [float(A[c].min()), float(A[c].median()), float(A[c].mean()), float(A[c].max())] for c in A.columns}
    B = pd.DataFrame(b_rows)
    nat = float(B['none'].std())
    report['b_yes_minus_no'] = {'natural_sd_across_positions': nat, 'none_mean': float(B['none'].mean())}
    for c in B.columns[1:]:
        d = B[c] - B['none']
        report['b_yes_minus_no'][c] = {'delta_min_med_mean_max': [float(d.min()), float(d.median()), float(d.mean()), float(d.max())],
                                       'mean_abs_delta_over_natural_sd': float(d.abs().mean() / max(nat, 1e-9))}
    # ---- c: cue codes ----
    codes, ticks = {w: [] for w in CUES}, {w: [] for w in CUES}
    for ctx in range(6):
        for tok in text[pos:pos + 32]:
            s, _, _ = read_token(s, tok)
        pos += 32
        for word, tid in CUES.items():
            s1, _ = quiet_ticks(s, 40)
            _, lat, c = read_token(s1, tid, count=True)
            codes[word].append((c[masks['Kenyon']] > 0).cpu().numpy()); ticks[word].append(int(lat.shape[0]))
    jac = lambda x, y: float((x & y).sum() / max((x | y).sum(), 1))
    words = list(CUES)
    within = {w: float(np.mean([jac(codes[w][i], codes[w][j]) for i in range(6) for j in range(i + 1, 6)])) for w in words}
    between = {f'{u}|{v}': float(np.mean([jac(codes[u][i], codes[v][i]) for i in range(6)]))
               for k, u in enumerate(words) for v in words[k + 1:]}
    active = {w: float(np.mean([x.mean() for x in codes[w]])) for w in words}
    report['c_cue_codes'] = {'active_fraction': active, 'same_word_across_contexts': within, 'between_words': between,
                             'ticks_per_cue_mean': {w: float(np.mean(ticks[w])) for w in words}}
    print('c', json.dumps(report['c_cue_codes'], ensure_ascii=False), flush=True)
    # ---- clock: variance the brain makes without changing input ----
    def latents(kind, n=128):
        st, out = s, []
        for i in range(n):
            if kind == 'text':
                st, lat, _ = read_token(st, text[pos + i]); out.append(lat[-1])
            elif kind == 'repeat':
                st, lat, _ = read_token(st, CUES[' stone']); out.append(lat[-1])
            else:
                st, _ = quiet_ticks(st, 10); out.append(fl.read_fly_motor(model, st).reshape(-1))
        z = torch.stack([x.reshape(-1).float() for x in out]).cpu()
        z = z - z.mean(0)
        ev = torch.linalg.eigvalsh(z @ z.T).clamp_min(0); p = ev / ev.sum().clamp_min(1e-30)
        return float((z ** 2).sum(1).mean()), float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum()))
    clock = {k: dict(zip(('variance', 'effective_rank'), latents(k))) for k in ('text', 'repeat', 'silence')}
    for k in ('repeat', 'silence'):
        clock[k]['variance_over_text'] = clock[k]['variance'] / max(clock['text']['variance'], 1e-12)
    report['clock'] = clock
    print('clock', json.dumps(clock), flush=True)
json.dump(report, open(fork / 'taste_phase0.json', 'w'), indent=1, ensure_ascii=False)
print('done', flush=True)
