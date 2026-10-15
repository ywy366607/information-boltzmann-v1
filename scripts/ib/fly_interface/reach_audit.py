"""Temporal reachability audit: how many ticks does each signal need to reach each part of the brain, and how long does
it stay?  To be compared with the world's clock (ticks per word, decision windows, beats), the BPTT window in ticks,
and the trace / credit time constants (user, 2026-10-15: "tick reachability was never audited").

From the checkpoint's own state, after `rest` ticks of true silence (sensory_on=False), fixed weights except the online
rules, three stimuli, each averaged over several words of the lineage's conversation:
  text    a word held for `hold` ticks, then silence;
  voice   the same word with the mother's voice on Johnston's organ (attach_mother_voice);
  reward  silence with the reply-reward pulse on the PAM cells for 4 ticks (attach_reply_reward).
Per cell group, per tick: spike fraction.  Reported per group:
  onset    first tick whose rate exceeds the pre-stimulus rate by > 3 sd of the pre-stimulus ticks (and > 0.002)
  peak     tick of the maximum, t50 the first tick reaching half of (peak - pre)
  decay    ticks after the stimulus ends until the rate falls back below pre + half the evoked excess
Usage: python reach_audit.py <checkpoint dir> <run dir> [n_words=8] [hold=64] [rest=96] [out.json]"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork, RUN = Path(sys.argv[1]), Path(sys.argv[2])
N_W = int(sys.argv[3]) if len(sys.argv) > 3 else 8
HOLD = int(sys.argv[4]) if len(sys.argv) > 4 else 64
REST = int(sys.argv[5]) if len(sys.argv) > 5 else 96
OUT = Path(sys.argv[6]) if len(sys.argv) > 6 else None
AFTER = 96
model, cfg, L = load(fork)
graph = ROOT / cfg['graph']
ann_path = graph.parent / 'body-annotations.feather'
n_voice = fl.attach_mother_voice(model, graph, ann_path, level=float(cfg.get('mother_voice_level') or 1.5))
n_pam = fl.attach_reply_reward(model, graph, ann_path, level=1.0, ticks=4)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
s0 = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
if getattr(model, 'dopamine_sensitivity', None) is not None:
    opts['dopamine_s_class'] = fl.dopamine_class_sensitivity(model, s0.h)
g = np.load(graph, allow_pickle=True)
a = pd.read_feather(ann_path).set_index('bodyId').reindex(g['neuron_body_ids'])
cls, typ = a['class'].astype(str).values, a['type'].astype(str).values
sc = a['superclass'].astype(str).values if 'superclass' in a.columns else cls
n = len(cls)
read = np.zeros(n, bool); read[model.read_indices.cpu().numpy()] = True
voice = np.zeros(n, bool); voice[model.voice_cells.cpu().numpy()] = True
written = np.zeros(n, bool); written[model.topographic_writer.injection_index.cpu().numpy()] = True
dan = np.zeros(n, bool); dan[g['dan_indices'].astype(np.int64)] = True
kcsub = lambda k: (cls == 'Kenyon_Cell') & np.array([(k in t) for t in typ])
groups = {
    'written sensory': written, 'ALPN': cls == 'ALPN', 'Kenyon ab': kcsub('KCab'), "Kenyon a'b'": kcsub("KCa'b'"),
    'Kenyon g': kcsub('KCg'), 'APL': np.array(['APL' in t for t in typ]), 'MBON': cls == 'MBON',
    "MBON-a'3": np.isin(typ, ['MBON16', 'MBON17', 'MBON17-like']), 'PAM': np.char.startswith(typ.astype(str), 'PAM'),
    'PPL1': np.char.startswith(typ.astype(str), 'PPL1'), 'DAN (all)': dan, 'LHN': cls == 'LHLN',
    'central complex': cls == 'CX', 'descending': np.char.find(sc.astype(str), 'descending') >= 0,
    'output read (acts)': read, 'voice (JO-A/B)': voice, 'all': np.ones(n, bool)}
groups = {k: v for k, v in groups.items() if v.any()}
G = {k: torch.as_tensor(np.nonzero(v)[0], device='cuda') for k, v in groups.items()}
import tiktoken
tok = tiktoken.get_encoding('gpt2')
lines = [json.loads(l) for l in (RUN / 'mother.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
words = []
for r in lines[::-1]:
    t = r['text'].strip()
    if r['who'] == 'mother' and t and t != '...' and not t.startswith('Mengli'):
        words += tok.encode(' ' + t)[:1]
    if len(words) >= N_W:
        break


def run(w, kind):
    s = s0
    for t in range(REST):
        s = fl.advance_fly_token_stream(model, s, torch.tensor([w], device='cuda'), [1], ticks=1, hold_input=True,
                                        fresh=t == 0, sensory_on=False, **opts)[0]
    trace = {k: [] for k in G}
    pre = 16
    total = pre + HOLD + AFTER
    for t in range(total):
        on = pre <= t < pre + HOLD
        kw = dict(sensory_on=on and kind in ('text', 'voice'), voice=on and kind == 'voice',
                  reward_ticks=max(0, pre + 4 - t) if kind == 'reward' and t >= pre else 0)
        s = fl.advance_fly_token_stream(model, s, torch.tensor([w], device='cuda'), [1], ticks=1, hold_input=True,
                                        fresh=t in (0, pre), **kw, **opts)[0]
        spk = s.ring[0][0] > 0
        for k, idx in G.items():
            trace[k].append(float(spk.index_select(0, idx).float().mean()))
    return {k: np.array(v) for k, v in trace.items()}


rep = {'checkpoint': str(fork), 'hold': HOLD, 'rest': REST, 'words': len(words), 'voice_cells': n_voice, 'pam_cells': n_pam,
       'groups': {k: int(v.sum()) for k, v in groups.items()}, 'stimuli': {}}
with torch.no_grad():
    for kind in ('text', 'voice', 'reward'):
        runs = [run(w, kind) for w in (words if kind != 'reward' else words[:3])]
        res = {}
        for k in G:
            x = np.mean([r[k] for r in runs], 0)
            pre_x = x[:16]
            base, sd = float(pre_x.mean()), float(pre_x.std())
            stim = x[16:16 + HOLD] if kind != 'reward' else x[16:16 + 64]
            above = np.nonzero(stim > base + max(3 * sd, 0.002))[0]
            peak_i = int(np.argmax(stim))
            excess = float(stim[peak_i] - base)
            half = np.nonzero(stim >= base + 0.5 * excess)[0] if excess > 0.002 else []
            post = x[16 + HOLD:] if kind != 'reward' else x[16 + 4:]
            dec = np.nonzero(post < base + 0.5 * excess)[0] if excess > 0.002 else []
            res[k] = {'pre_rate': round(base, 5), 'peak_rate': round(float(stim[peak_i]), 5),
                      'onset_tick': int(above[0]) + 1 if len(above) else None,
                      't50_tick': int(half[0]) + 1 if len(half) else None,
                      'peak_tick': peak_i + 1 if excess > 0.002 else None,
                      'decay_ticks': (int(dec[0]) if len(dec) else f'>{len(post)}') if excess > 0.002 else None}
        rep['stimuli'][kind] = res
        print(kind, flush=True)
        for k, v in res.items():
            print(f'  {k:20s} pre {v["pre_rate"]:.4f} peak {v["peak_rate"]:.4f}  onset {v["onset_tick"]}  t50 {v["t50_tick"]}  '
                  f'peak@{v["peak_tick"]}  decay {v["decay_ticks"]}', flush=True)
cfg_t = {k: cfg.get(k) for k in ('window', 'stream_pace_mean', 'stream_pace_min', 'stream_ticks', 'mother_credit_tau',
                                 'mother_credit_ticks', 'act_tau', 'eg_tau')}
lam = float(torch.sigmoid(model.comp_logit_trace)) if getattr(model, 'comp_logit_trace', None) is not None else None
cfg_t['kenyon_trace_tau_ticks'] = (-1.0 / np.log(lam)) if lam else None
if getattr(model, 'dopamine_logit_fast', None) is not None:
    cfg_t['dopamine_tau_fast_ticks'] = float(-1.0 / torch.log(torch.sigmoid(model.dopamine_logit_fast)).mean())
    cfg_t['dopamine_tau_slow_ticks'] = float(-1.0 / torch.log(torch.sigmoid(model.dopamine_logit_slow)).mean())
rep['clocks'] = cfg_t
print(json.dumps(cfg_t))
if OUT:
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
