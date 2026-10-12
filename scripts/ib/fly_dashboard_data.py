"""Collect the fly runs into small JSON row files for the monitoring dashboard (CPU only, no torch).

runs.json      one row per logged window of each arm: loss, firing, dopamine, go ticks, online rule
evals.json     the trainer's active evaluations (fresh text, revisit)
status.json    one row per arm: tokens done / target, state
gestation.json group firing rates (Hz) over the gestation ticks; gest_summary.json its health numbers
brain.json     a sample of neurons with their 3D position (um), group, gestated rate, target rate and drive
rhythm.json    the adult lineage's population activity over 0.6 s of reading then 0.6 s of silence (1 tick = 1 ms)
Usage: python fly_dashboard_data.py <out dir>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / 'scratch' / 'fly_stream'
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
ARMS = {'对照': 'b2_control', '在线学习': 'b2_online'}


def jl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def num(x, nd=5):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


runs, evals, status = [], [], []
for arm, d in ARMS.items():
    m = jl(S / d / 'metrics.jsonl')
    for r in m:
        if 'bptt_train_tokens' not in r:
            continue
        runs.append({'kind': 'window', 'arm': arm, 'tokens': int(r['bptt_train_tokens']), 'loss': num(r.get('ema_stream_loss'), 4),
                     'firing_pct': num(100 * r.get('firing_rate', float('nan')), 4),
                     'dopamine_pct': num(100 * r['dopamine_cells_firing'], 4) if 'dopamine_cells_firing' in r else None,
                     'go_ticks': num(r.get('go_ticks_mean'), 3), 'grad_norm': num(r.get('grad_norm_before_clip'), 3),
                     'eta': num(r.get('online_da_eta'), 5), 'memory_sd': num(r.get('online_da_memory_sd'), 5),
                     'world': r.get('world')})
    for r in jl(S / d / 'lifelong_evaluation.jsonl'):
        t = r.get('bptt_train_tokens') or r.get('tokens')
        b = r.get('fresh_b_nll') or r.get('fresh_B_nll') or (r.get('fresh') or {}).get('nll') if isinstance(r.get('fresh'), dict) else r.get('fresh_b_nll')
        a = r.get('a_revisit_nll') or r.get('revisit_nll')
        if t is not None and (b is not None or a is not None):
            evals.append({'arm': arm, 'tokens': int(t), 'fresh_b': num(b, 4), 'a_revisit': num(a, 4)})
    last = m[-1] if m else {}
    done = int(last.get('bptt_train_tokens', 0)); target = int(last.get('target_bptt_tokens', 0) or 0)
    state = 'queued' if not m else ('done' if target and done >= target else 'running')
    status.append({'arm': arm, 'tokens_done': done, 'tokens_target': target, 'state': state,
                   'started_from': 'gestated newborn (200,000 ticks)', 'updated': last.get('updated_at') or ''})

# the trainer prints active evaluations to its log; parse them when the jsonl lacks them
if not evals:
    import re
    for arm, d in ARMS.items():
        log = S / (d + '.log')
        if log.exists():
            for mt in re.finditer(r'\[ACTIVE EVAL (\d+)\] fresh B NLL=([\d.]+); A revisit=([\d.]+)', log.read_text(encoding='utf-8', errors='ignore')):
                evals.append({'arm': arm, 'tokens': int(mt.group(1)), 'fresh_b': float(mt.group(2)), 'a_revisit': float(mt.group(3))})

gest, gsum = [], []
for r in jl(S / 'gestated' / 'gestation.jsonl'):
    for g, v in r['rate_per_group'].items():
        gest.append({'tick': r['tick'], 'group': g, 'hz': num(1000 * v, 3)})
    gsum.append({'tick': r['tick'], 'silent_pct': num(100 * r['fraction_silent_5k'], 2),
                 'within_2x_pct': num(100 * r['fraction_within_2x_of_target'], 2),
                 'median_abs_log_ratio': num(r['median_abs_log_rate_over_target'], 3),
                 'core_duty': num(r['core_duty_cycle_median'], 4), 'nll': num(r['nll'], 3)})

# brain sample
g = np.load(ROOT / 'data/malecns_v1/fly_reservoir_coba.npz', allow_pickle=True)
N = len(g['neuron_body_ids'])
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
cls, sup = ann['class'].astype(str).values, ann['superclass'].astype(str).values
dan = np.zeros(N, bool); dan[g['dan_indices']] = True
s_idx = np.load(ROOT / 'data/malecns_v1/sensory_partitions.npz')
sens = np.zeros(N, bool); sens[np.concatenate([s_idx[k] for k in s_idx.files])] = True
names = [str(x) for x in g['superclass_names']]
out_classes = ('vnc_motor', 'cb_motor', 'descending_neuron', 'cb_endocrine', 'vnc_endocrine', 'vnc_efferent', 'cb_efferent')
read = np.isin(g['superclass_id'], [names.index(n) for n in out_classes if n in names])
core = (sup == 'cb_intrinsic') & ~np.isin(cls, ['CX', 'Kenyon_Cell', 'MBON', 'DAN']) & ~dan
group = np.full(N, '其余', dtype=object)
group[sup == 'ol_intrinsic'] = '视叶'
group[core] = '核心'
group[cls == 'CX'] = '中央复合体'
group[cls == 'Kenyon_Cell'] = 'Kenyon 细胞'
group[cls == 'MBON'] = 'MBON'
group[sens] = '感觉'
group[read] = '运动读出面'
group[dan] = '多巴胺细胞'
rng = np.random.default_rng(0)
quota = {'多巴胺细胞': 400, 'MBON': 100, 'Kenyon 细胞': 2000, '中央复合体': 1200, '核心': 5000, '运动读出面': 1800,
         '感觉': 3000, '视叶': 6000, '其余': 4000}
rate_path = S / 'gestated' / 'gestation_rate_ema.npy'
rate = np.load(rate_path) * 1000 if rate_path.exists() else np.full(N, np.nan)
tau = np.asarray(g['tau_m'], dtype=np.float64)
z0 = np.random.default_rng(0)  # same draw as attach_homeostasis(seed=0) is torch-based; target recomputed below
try:
    import torch
    gen = torch.Generator().manual_seed(0)
    duty = torch.exp(math.log(0.05) + torch.randn(N, generator=gen)).numpy()
    target = np.minimum(duty / np.maximum(tau, 1.0), 0.2) * 1000
except Exception:
    target = np.full(N, np.nan)
drive = np.full(N, np.nan)
try:
    import torch
    P = torch.load(S / 'gestated' / 'last.pt', map_location='cpu', mmap=True, weights_only=False)['learner']['physical']
    drive = P['homeostasis'][0].numpy()
except Exception:
    pass
xyz = np.asarray(g['coords_um'], dtype=np.float64)
brain = []
for grp, k in quota.items():
    idx = np.flatnonzero(group == grp)
    pick = idx if len(idx) <= k else rng.choice(idx, k, replace=False)
    for i in pick:
        brain.append({'x': round(xyz[i, 0], 1), 'y': round(xyz[i, 1], 1), 'z': round(xyz[i, 2], 1), 'group': grp,
                      'rate_hz': num(rate[i], 3), 'target_hz': num(target[i], 3), 'drive': num(drive[i], 3)})

rhythm = []
labels = {'core (excitatory cells)': '核心兴奋', 'core (inhibitory cells)': '核心抑制', 'dopamine cells': '多巴胺细胞',
          'motor read surface': '运动读出面', 'sensory (injected)': '感觉'}
for who, f in (('成年谱系', 'heartbeat_dasite_pre.npz'), ('新生·对照', 'heartbeat_b2_control.npz'),
               ('新生·在线学习', 'heartbeat_b2_online.npz')):
    hb = S / f
    if not hb.exists():
        continue
    h = np.load(hb, allow_pickle=True)
    R, reading, gnames = h['rates'], int(h['reading']), list(h['names'])
    lo, hi = max(0, reading - 600), min(len(R), reading + 600)
    for t in range(lo, hi):
        for j, n in enumerate(gnames):
            if n in labels:
                rhythm.append({'individual': who, 'ms': t - reading, 'group': labels[n], 'hz': round(float(R[t, j] * 1000), 2),
                               'phase': 'reading' if t < reading else 'silence'})

for name, rows in (('runs', runs), ('evals', evals), ('status', status), ('gestation', gest), ('gest_summary', gsum),
                   ('brain', brain), ('rhythm', rhythm)):
    (out / f'{name}.json').write_text(json.dumps(rows, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(name, len(rows), 'rows', (out / f'{name}.json').stat().st_size // 1024, 'KB')
