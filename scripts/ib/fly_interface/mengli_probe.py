"""Mengli's probe (user, 2026-10-15): does the mother's voice reach anything, does a signal still die after two hops,
is there a corollary discharge, is the repeated 'Mengli' caught by the familiarity memory, do the Kenyon and dopamine
cells matter?  A diagnostic on a copy (weights frozen; the online compartment rule runs as part of the dynamics), on
the lineage's own conversation (mother.jsonl, the last N tokens, GPT-2 BPE as the mother world feeds them), each word
held for TICKS ticks, from the checkpoint's own state (never reset).  Every effect is a paired branch from the same
state, so the difference is the cause alone (the main trajectory is the real one: voice on).

V  the voice: per mother word, k ticks with vs without the Johnston's organ current; firing rate difference per group
   (JO, hop 1..5 from JO, Kenyon, MBON, DAN, output cells), [min | median | mean | max] over words.
H  the text input: per word (every 4th), the word written vs a quiet interval; rate difference by hop from the cells
   the writer drives (does a sensory signal still die after two hops?).
C  corollary discharge: at sampled moments, a 'commit command' (current on the read cells in proportion to how much
   their firing raises the commit, ai_w[1] / ai_sigma, positive part) vs the same currents permuted over the read cells,
   vs none; the auditory first / second-order rates per tick after it, and the commit hazard (sanity: the command
   commits).
F  familiarity: the Kenyon -> MBON compartment memory phi on the edges of the Kenyon cells active for each word, by the
   word's count in the conversation (' Meng' 'li' vs the rest); the MBON rate with phi as it is vs phi = 0 per word.
K  the Kenyon and dopamine cells: per mother word, the next word's CE with the Kenyon cells (or the dopamine cells)
   held silent for the word's ticks (-3 x rheobase) vs intact (acute); and from the start for the whole stream
   (sustained, paired by position).
T  true silence: after sampled words, withdraw both text and voice for 64 ticks. Compare a word-written branch
   with a no-text branch from the same pre-word state; report content contrast and real next-word CE per delay.
Usage: python mengli_probe.py <checkpoint dir | .pt> <run dir> [tokens=600] [out.json]
"""
import sys, json, math, os, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

src = Path(sys.argv[1])


class _Fork:                     # load() reads fork / 'last.pt'; a .pt file is read as it is
    def __truediv__(self, name):
        return src if src.suffix == '.pt' else src / name


RUN = Path(sys.argv[2])
N_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 600
OUT = Path(sys.argv[4]) if len(sys.argv) > 4 else None
TICKS = int(os.environ.get('PROBE_TICKS', 6))
PARTS = os.environ.get('PROBE_PARTS', 'VHCFKTP')
model, cfg, L = load(_Fork())
graph_path = ROOT / cfg['graph']
ann_path = graph_path.parent / 'body-annotations.feather'
n_voice = fl.attach_mother_voice(model, graph_path, ann_path, level=float(cfg.get('mother_voice_level') or 1.5))
dev = 'cuda'
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
HOLD = bool(cfg.get('hold_input'))
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
N = model.n_neurons
quiet = torch.zeros(1, N, device=dev)
s0 = move_state(fl.FlyPhysicalState(**L['physical']), dev)
if getattr(model, 'dopamine_sensitivity', None) is not None:
    opts['dopamine_s_class'] = fl.dopamine_class_sensitivity(model, s0.h)
rates = opts['base_rates'][0].expand(1, N)
rheo = (opts['thresholds'].expand(1, N) * -torch.log(rates.clamp(1e-5, 1 - 1e-7))).detach()

# ---- cell groups ----
g = np.load(graph_path, allow_pickle=True)
ann = pd.read_feather(ann_path).set_index('bodyId').reindex(g['neuron_body_ids'])
typ, cls, sup = (ann[c].astype(str).values for c in ('type', 'class', 'superclass'))
pre = np.concatenate([g['edge_pre_e'], g['edge_pre_i']]).astype(np.int64)
post = np.concatenate([g['edge_post_e'], g['edge_post_i']]).astype(np.int64)
import scipy.sparse as sp
A = sp.csr_matrix((np.ones(pre.size, dtype=np.int32), (pre, post)), shape=(N, N))


def hops_from(seed, max_hop=5):
    d = np.full(N, 99, dtype=np.int64); d[seed] = 0
    front = np.zeros(N, dtype=bool); front[seed] = True
    for h in range(1, max_hop + 1):
        nxt = np.asarray(A.T @ front.astype(np.int32)).ravel() > 0
        new = nxt & (d == 99)
        d[new] = h; front = new
    return d


jo = model.voice_cells.cpu().numpy()
We = sp.csr_matrix((np.abs(g['edge_weight_e']).astype(np.float64), (g['edge_pre_e'], g['edge_post_e'])), shape=(N, N))
exc_in = np.asarray(We.sum(0)).ravel()
first = np.setdiff1d(np.flatnonzero(np.asarray(We[jo].sum(0)).ravel() >= 0.05 * np.maximum(exc_in, 1e-30)), jo)
second = np.setdiff1d(np.flatnonzero(np.asarray(We[first].sum(0)).ravel() >= 0.05 * np.maximum(exc_in, 1e-30)),
                      np.concatenate([jo, first]))
out_cells = np.flatnonzero(np.isin(sup, ('descending_neuron', 'vnc_motor', 'cb_motor', 'vnc_efferent')))
d_jo = hops_from(jo)
base = {'KC': np.flatnonzero(cls == 'Kenyon_Cell'), 'MBON': np.flatnonzero(cls == 'MBON'),
        'DAN': g['dan_indices'].astype(np.int64), 'output': out_cells,
        'aud1': first, 'aud2': second}
GROUPS_V = {'JO': jo, **{f'hop{h}': np.flatnonzero(d_jo == h) for h in range(1, 6)}, **base}


def masks(groups):
    return {k: torch.as_tensor(np.isin(np.arange(N), v), device=dev) for k, v in groups.items() if len(v)}


MV = masks(GROUPS_V)
from information_boltzmann.core.fly_activity_audit import CellActivityAudit
activity = CellActivityAudit(s0.h, getattr(model, 'graded_mask', None), model.refractory_ticks) if 'P' in PARTS else None

# ---- the conversation ----
import tiktoken
tok = tiktoken.get_encoding('gpt2')
lines = [json.loads(l) for l in (RUN / 'mother.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
max_trained = os.environ.get('PROBE_MAX_TRAINED')
if max_trained is not None:
    lines = [r for r in lines if int(r['trained']) < int(max_trained)]
stream, who = [], []
for r in reversed(lines):
    # An empty fly turn is logged as '...'; no such word reached the senses in the live conversation.
    if r['who'] == 'fly' and r['text'].strip() in ('', '...'):
        continue
    ids = tok.encode(' ' + r['text'].strip() + ('\n' if r['who'] == 'mother' else ''))
    stream = ids + stream; who = [r['who']] * len(ids) + who
    if len(stream) >= N_TOK + 1:
        break
stream, who = stream[-(N_TOK + 1):], who[-(N_TOK + 1):]
counts = pd.Series(stream).value_counts()
meng = set(tok.encode(' Mengli')) | set(tok.encode('Mengli'))


def tick(s, token, write, extra=None):
    """One tick: the word written (write) or a quiet tick; extra = external current [1, N] for this tick."""
    model.external_drive = extra
    if write:
        dr, baseline = fl.write_input(model, s, token)
        s = fl.step_fly_physical_tick(model, s, token, dr, baseline, opts, base_rates=opts['base_rates'])
    else:
        baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
        s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
    model.external_drive = None
    return s


def word(s, w, voice=False, silence=None, write=True, inject=None, groups=None, per_tick=False, muscle=None,
         track_activity=False):
    """k ticks of word w from s.  voice: the mother's voice current; silence: a bool mask held at -3 x rheobase;
    inject: an extra current on the first tick only.  Returns (state, rates {group: mean firing per cell per tick},
    per-tick rates if per_tick, hazard per tick)."""
    token = torch.tensor([int(w)], device=dev)
    vd = fl.mother_voice_drive(model, s.h) if voice else None
    sil = None if silence is None else torch.where(silence, -3.0 * rheo, torch.zeros_like(rheo))
    acc = {k: 0.0 for k in (groups or {})}
    rows, fired = [], torch.zeros(N, dtype=torch.bool, device=dev)
    counts, hazards = torch.zeros(N, device=dev), []
    for t in range(TICKS):
        extra = None
        for part in (vd, sil, inject if t == 0 else None):
            if part is not None:
                extra = part if extra is None else extra + part
        s = tick(s, token, write and (t == 0 or HOLD), extra)
        fired |= s.ring[0][0] > 0
        spk = (s.ring[0][0] > 0).float()
        counts += spk
        if muscle is not None:
            from information_boltzmann.core.fly_active_inference import ai_step
            hz, muscle, _ = ai_step(model, s, muscle, taped=False)
            hazards.append(float(hz))
        if per_tick:
            keys = list(groups or {})
            values = torch.stack([spk[groups[k]].mean() for k in keys]).cpu().tolist() if keys else []
            rows.append(dict(zip(keys, values)))
        if track_activity and activity is not None:
            activity.observe(s.ring[0][0])
    # One device-to-host transfer per word instead of one per group on every tick.
    keys = list(groups or {})
    values = torch.stack([counts[groups[k]].mean() / TICKS for k in keys]).cpu().tolist() if keys else []
    acc = dict(zip(keys, values))
    word.fired = fired
    word.counts = counts
    word.hazards = hazards
    word.muscle = muscle
    return s, acc, rows


def ce_next(s, target):
    lat = fl.read_fly_motor(model, s)
    logits = model.decoder(model.read_norm(lat)).float()
    return float(torch.nn.functional.cross_entropy(logits, torch.tensor([int(target)], device=dev))), lat


def dist(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=np.float64)
    if not x.size:
        return None
    return {'n': int(x.size), 'mean': float(x.mean()), 'se': float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else None,
            'min': float(x.min()), 'median': float(np.median(x)), 'max': float(x.max()), 'frac_pos': float((x > 0).mean())}


report = {'checkpoint': str(src), 'tokens': len(stream) - 1, 'ticks': TICKS, 'voice_cells': n_voice,
          'groups': {k: int(len(v)) for k, v in GROUPS_V.items()}, 'mengli_tokens': int(sum(t in meng for t in stream)),
          'scope': 'frozen-weight retrospective replay; online physical/compartment states evolve; no capability claim',
          'conversation_max_trained_exclusive': max_trained, 'parts': PARTS,
          'stream_sha256': __import__('hashlib').sha256(np.asarray(stream, dtype=np.int64).tobytes()).hexdigest()}
report['T_delay_unit'] = 'physical ticks after text and voice withdrawal (mother-world beat is 8 ticks)'
report['activity_unit'] = 'positive ring release; may include graded release as well as spikes'
print(json.dumps(report), flush=True)

# commit command (C): positive part of the commit readout per read cell, as a current on those cells
cmd = None
if 'C' in PARTS and getattr(model, 'ai_w', None) is not None:
    w1 = (model.ai_w[1] / model.ai_sigma[0]) if getattr(model, 'ai_sigma', None) is not None else model.ai_w[1]
    wpos = torch.relu(w1).float()
    wpos = wpos / wpos.max().clamp_min(1e-12)
    cmd = torch.zeros(1, N, device=dev)
    cmd[0, model.read_indices] = 1.5 * wpos * rheo[0, model.read_indices]
    perm = torch.randperm(model.read_indices.numel(), generator=torch.Generator().manual_seed(0)).to(dev)
    cmd_ctrl = torch.zeros(1, N, device=dev)
    cmd_ctrl[0, model.read_indices] = cmd[0, model.read_indices][perm]
    report['C_command_cells'] = int((wpos > 0.1).sum())

d_in = None
V, Hd, C, Fm, Kc = [], [], [], [], {'kc_acute': [], 'dan_acute': [], 'ce_intact': []}
suppression = {'kc_acute': [], 'dan_acute': [], 'kc_sustained': [], 'dan_sustained': []}
DELAYS = (0, 1, 2, 3, 4, 8, 16, 32, 64)
temporal = {t: [] for t in DELAYS}
sampling = np.random.default_rng(20261015)
def positions(every):
    size = len(stream) - 1
    return set(sampling.choice(size, size=max(1, size // every), replace=False).tolist())
sample_h, sample_c, sample_t = positions(4), positions(6), positions(24)
report['sample_positions'] = {'H': sorted(sample_h), 'C': sorted(sample_c), 'T': sorted(sample_t)}
KCm = torch.as_tensor(np.isin(np.arange(N), base['KC']), device=dev)
DANm = torch.as_tensor(np.isin(np.arange(N), base['DAN']), device=dev)
with torch.no_grad():
    if 'H' in PARTS:
        # the cells the writer drives: union over the stream's first 64 words
        drv = torch.zeros(N, dtype=torch.bool, device=dev)
        for w in stream[:64]:
            dr, _ = fl.write_input(model, s0, torch.tensor([int(w)], device=dev))
            drv |= dr[0].abs() > 0
        d_in = hops_from(np.flatnonzero(drv.cpu().numpy()))
        MH = masks({**{f'hop{h}': np.flatnonzero(d_in == h) for h in range(0, 6)}, **{k: base[k] for k in ('KC', 'MBON', 'DAN', 'output')}})
        report['H_groups'] = {k: int(m.sum()) for k, m in MH.items()}
    MC = masks({'aud1': first, 'aud2': second, 'output': out_cells})
    MM = masks({'MBON': base['MBON']})
    s = s0
    main_muscle = None
    if 'C' in PARTS and cmd is not None:
        from information_boltzmann.core.fly_active_inference import motor_rest
        muscle_saved = L.get('ai_g')
        main_muscle = muscle_saved.to(dev).clone() if muscle_saved is not None else motor_rest(model, s.h)
    for i in range(len(stream) - 1):
        w, nxt = stream[i], stream[i + 1]
        mom = who[i] == 'mother'
        if 'V' in PARTS and mom:
            _, on, _ = word(s, w, voice=True, groups=MV)
            counts_on = word.counts
            _, off, _ = word(s, w, voice=False, groups=MV)
            contrast = (counts_on - word.counts).abs() / TICKS
            V.append({k: on[k] - off[k] for k in on} | {f'{k}_off': off[k] for k in off}
                     | {f'{k}_abs': float(contrast[m].mean()) for k, m in MV.items()})
        if 'H' in PARTS and i in sample_h:
            _, wr, _ = word(s, w, voice=mom, groups=MH)
            counts_wr = word.counts
            _, qu, _ = word(s, w, voice=mom, write=False, groups=MH)
            contrast = (counts_wr - word.counts).abs() / TICKS
            Hd.append({k: wr[k] - qu[k] for k in wr} | {f'{k}_quiet': qu[k] for k in qu}
                      | {f'{k}_abs': float(contrast[m].mean()) for k, m in MH.items()})
        if 'C' in PARTS and cmd is not None and i in sample_c:
            groups_c = MC
            muscle = main_muscle
            _, _, r0 = word(s, w, voice=mom, groups=groups_c, per_tick=True, muscle=muscle)
            h0 = word.hazards
            _, _, rc = word(s, w, voice=mom, inject=cmd, groups=groups_c, per_tick=True, muscle=muscle)
            hc = word.hazards
            _, _, rp = word(s, w, voice=mom, inject=cmd_ctrl, groups=groups_c, per_tick=True, muscle=muscle)
            hp = word.hazards
            C.append([{k: (rc[t][k] - r0[t][k], rp[t][k] - r0[t][k]) for k in r0[t]}
                      | {'hazard': (hc[t] - h0[t], hp[t] - h0[t])} for t in range(TICKS)])
        if 'K' in PARTS and mom:
            sa, ra, _ = word(s, w, voice=True, silence=KCm, groups={'target': KCm})
            sb, rb, _ = word(s, w, voice=True, silence=DANm, groups={'target': DANm})
            suppression['kc_acute'].append(ra['target'])
            suppression['dan_acute'].append(rb['target'])
            Kc['kc_acute'].append(ce_next(sa, nxt)[0])
            Kc['dan_acute'].append(ce_next(sb, nxt)[0])
        if 'T' in PARTS and i in sample_t:
            # Both branches hear the same voice while this word is present. Then ALL sensory drive is withdrawn.
            present, _, _ = word(s, w, voice=mom)
            absent, _, _ = word(s, w, voice=mom, write=False)
            for delay in range(max(DELAYS) + 1):
                if delay:
                    present = tick(present, None, False, extra=None)
                    absent = tick(absent, None, False, extra=None)
                if delay in temporal:
                    cp, lp = ce_next(present, nxt)
                    ca, la = ce_next(absent, nxt)
                    delta = present.h - absent.h
                    spikes_p = present.ring[0] > 0
                    spikes_a = absent.ring[0] > 0
                    temporal[delay].append({
                        'name_word': w in meng,
                        'next_nll': cp, 'no_text_next_nll': ca, 'text_predictive_gain': ca - cp,
                        'membrane_contrast_rms': float(delta.square().mean().sqrt()),
                        'read_contrast_norm': float((lp - la).norm()),
                        'spike_difference_fraction': float((spikes_p != spikes_a).float().mean()),
                        'firing_rate': float(spikes_p.float().mean()),
                        'comp_memory_contrast_rms': float((present.comp_memory - absent.comp_memory).square().mean().sqrt())
                            if present.comp_memory.numel() else None,
                    })
        # the real trajectory: voice on for the mother's words
        s, r_main, _ = word(s, w, voice=mom, groups=MM if 'F' in PARTS else None, muscle=main_muscle,
                             track_activity=True)
        main_muscle = word.muscle
        fired_main = word.fired
        ce, lat = ce_next(s, nxt)
        s = fl.expect_next(model, s, lat)
        if 'K' in PARTS and mom:
            Kc['ce_intact'].append(ce)
        if 'F' in PARTS and getattr(model, 'comp_pre', None) is not None and s.comp_memory.numel():
            # phi on the compartment edges of the Kenyon cells that fired during this word, and the MBON rate without phi
            fired_edges = fired_main[model.comp_pre.long()]
            phi = s.comp_memory[0]
            phi_w = float(phi[fired_edges].mean()) if bool(fired_edges.any()) else None
            Fm.append({'i': i, 'w': int(w), 'count': int(counts.get(w, 0)), 'mengli': w in meng, 'mother': mom,
                       'phi_fired': phi_w, 'mbon': r_main.get('MBON')})
            if i % 3 == 0:
                s_nophi = fl.FlyPhysicalState(**{**s.state_dict(), 'comp_memory': torch.zeros_like(s.comp_memory)})
                _, r_np, _ = word(s_nophi, nxt, voice=who[i + 1] == 'mother', groups=MM)
                _, r_p, _ = word(s, nxt, voice=who[i + 1] == 'mother', groups=MM)
                Fm[-1].update(next_w=int(nxt), next_count=int(counts.get(nxt, 0)), next_mengli=nxt in meng,
                              mbon_phi_effect=r_p['MBON'] - r_np['MBON'])
        if i % 100 == 0:
            print(f'{i}/{len(stream) - 1}', flush=True)

    if 'K' in PARTS:
        # sustained: the Kenyon (or dopamine) cells silent for the whole stream, paired by position
        for name, mask in (('kc_sustained', KCm), ('dan_sustained', DANm)):
            s, ces = s0, []
            for i in range(len(stream) - 1):
                s, residual, _ = word(s, stream[i], voice=who[i] == 'mother', silence=mask, groups={'target': mask})
                ce, lat = ce_next(s, stream[i + 1])
                s = fl.expect_next(model, s, lat)
                if who[i] == 'mother':
                    ces.append(ce)
                    suppression[name].append(residual['target'])
            Kc[name] = ces

if V:
    report['V_voice_effect'] = {k: dist([r[k] for r in V]) for k in MV}
    report['V_voice_absolute_cell_effect'] = {k: dist([r[f'{k}_abs'] for r in V]) for k in MV}
    report['V_rate_without_voice'] = {k: float(np.mean([r[f'{k}_off'] for r in V])) for k in MV}
if Hd:
    report['H_input_effect'] = {k: dist([r[k] for r in Hd]) for k in MH}
    report['H_input_absolute_cell_effect'] = {k: dist([r[f'{k}_abs'] for r in Hd]) for k in MH}
    report['H_rate_quiet'] = {k: float(np.mean([r[f'{k}_quiet'] for r in Hd])) for k in MH}
if C:
    report['C_command'] = {k: [{'tick': t + 1, 'command': dist([c[t][k][0] for c in C]), 'permuted': dist([c[t][k][1] for c in C]),
                                'specific': dist([c[t][k][0] - c[t][k][1] for c in C])} for t in range(TICKS)]
                           for k in ('aud1', 'aud2', 'output', 'hazard')}
if Fm:
    df = pd.DataFrame(Fm)
    from scipy.stats import spearmanr
    rep = {'phi_fired_by_mengli': {'mengli': dist(df[df.mengli].phi_fired), 'other': dist(df[~df.mengli].phi_fired)},
           'phi_fired_vs_count_rho': float(spearmanr(df.phi_fired, df['count'], nan_policy='omit')[0])}
    if 'mbon_phi_effect' in df:
        e = df.dropna(subset=['mbon_phi_effect']).copy()
        e['next_mengli'] = e['next_mengli'].astype(bool)        # object column (NaN on rows without the branch)
        is_mengli = e['next_mengli'].fillna(False).astype(bool)
        rep['mbon_phi_effect'] = {'mengli': dist(e.loc[is_mengli, 'mbon_phi_effect']), 'other': dist(e.loc[~is_mengli, 'mbon_phi_effect']),
                                  'rho_with_count': float(spearmanr(e.mbon_phi_effect, e.next_count)[0])}
    rep['phi_all'] = dist(s0.comp_memory[0].float().cpu().numpy()) if s0.comp_memory.numel() else None
    report['F_familiarity'] = rep
if Kc['ce_intact']:
    base_ce = np.array(Kc['ce_intact'])
    report['K_ce_intact'] = dist(base_ce)
    for name in ('kc_acute', 'dan_acute', 'kc_sustained', 'dan_sustained'):
        if Kc.get(name):
            report[f'K_{name}_minus_intact'] = dist(np.array(Kc[name]) - base_ce)
    report['K_suppressed_population_residual_release'] = {k: dist(v) for k, v in suppression.items()}
if temporal[0]:
    report['T_true_silence'] = {str(t): {k: dist([r[k] for r in rows]) for k in rows[0] if k != 'name_word'}
                              for t, rows in temporal.items()}
    report['T_sample_name_fraction'] = float(np.mean([r['name_word'] for r in temporal[0]]))
if activity is not None:
    report['P_cell_activity'] = activity.report({'whole_brain': np.arange(N), **GROUPS_V})
    quiet_activity = CellActivityAudit(s.h, getattr(model, 'graded_mask', None), model.refractory_ticks)
    with torch.no_grad():
        for _ in range(64):
            s = tick(s, None, False, extra=None)
            quiet_activity.observe(s.ring[0][0])
    report['P_after_true_silence'] = quiet_activity.report({'whole_brain': np.arange(N), **GROUPS_V})
print(json.dumps(report, indent=1), flush=True)
if OUT:
    OUT.write_text(json.dumps(report, indent=1), encoding='utf-8')
