"""Does a new WORD reach dopamine, and does the response habituate with repetition within life?  (Text version of
novelty_probe.py: the fly's alpha'3 compartment -- MBON-alpha'3 responds to novelty and habituates through its own
dopamine; a biological, within-life form of the epistemic value.)  Fixed weights except the online compartment memory
(it runs, as in life).  Pulse input, 8 ticks per word, running validation text.  For 6 rare words absent from the text
(a control: one frequent word), each presented 8 times with 12 text words between presentations: per presentation the
extra firing during the word's 8 ticks over the preceding text word's 8 ticks, for PPL104 (alpha'3 dopamine), all PAM,
all PPL1, MBON16/17 (alpha'3), all MBONs, Kenyon cells; the change of phi on the alpha'3 edges; and the read's CE of
the word when it arrives (does it become expected?).  Habituation = slope over presentations (negative = fades).
Usage: python word_novelty_probe.py <checkpoint dir> [out.json]
"""
import sys, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
graph = ROOT / cfg['graph']; ann_path = graph.parent / 'body-annotations.feather'
if getattr(model, 'comp_pre', None) is None:
    raise SystemExit('needs a checkpoint with compartment memory')
model.eval().requires_grad_(False)
g = np.load(graph, allow_pickle=True)
ann = pd.read_feather(ann_path).set_index('bodyId').reindex(g['neuron_body_ids'])
typ, cls = ann['type'].astype(str).values, ann['class'].astype(str).values
dan = np.zeros(len(typ), bool); dan[g['dan_indices']] = True
groups = {'PPL104': typ == 'PPL104', 'PAM': np.char.startswith(typ.astype(str), 'PAM') & dan,
          'PPL1': np.char.startswith(typ.astype(str), 'PPL1') & dan,
          'MBON_a3': np.isin(typ, ['MBON16', 'MBON17', 'MBON17-like']), 'MBON': cls == 'MBON', 'KC': cls == 'Kenyon_Cell'}
gm = {k: torch.as_tensor(v, device='cuda') for k, v in groups.items()}
a3_edges = torch.as_tensor(groups['MBON_a3'], device='cuda')[model.comp_post.long()]
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
N = model.n_neurons; writer = model.topographic_writer; quiet = torch.zeros(1, N, device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r'); text = np.array(val[50_000:70_000], dtype=np.int64)
train = np.load(ROOT / cfg['data'] / 'train.npy', mmap_mode='r')
counts = np.bincount(np.asarray(train[:20_000_000], dtype=np.int64), minlength=50257)
present = set(text.tolist())
rng = np.random.default_rng(5)
rare_pool = [w for w in np.where((counts >= 20) & (counts <= 200))[0] if w not in present]
novel = [int(w) for w in rng.choice(rare_pool, 6, replace=False)]
frequent = int(np.argmax(counts))
st = {'s': move_state(fl.FlyPhysicalState(**L['physical']), 'cuda'), 'i': 0}


def word(tok):
    """Present one word for 8 ticks; returns mean firing per group and the read's CE of this word before it came."""
    s = st['s']
    lat = fl.read_fly_motor(model, s)
    ce = float(F.cross_entropy(model.decoder(model.read_norm(lat)).float(), torch.tensor([tok], device='cuda')))
    acc = torch.zeros(N, device='cuda')
    t = torch.tensor([tok], device='cuda')
    for k in range(8):
        if k == 0:
            src, base = fl.write_input(model, s, t)
            s = fl.step_fly_physical_tick(model, s, t, src, base, opts, base_rates=opts['base_rates'])
        else:
            s = fl.step_fly_physical_tick(model, s, None, quiet, writer.lambda_adapt * s.baseline, opts, base_rates=opts['base_rates'])
        acc += (s.ring[0][0] > 0).float()
    st['s'] = fl.expect_next(model, s, fl.read_fly_motor(model, s))
    return {k: float(acc[m].mean() / 8) for k, m in gm.items()}, ce


def text_word():
    tok = int(text[st['i'] % len(text)]); st['i'] += 1
    return word(tok)


rep = {'checkpoint': str(fork), 'words': []}
with torch.no_grad():
    for _ in range(64):
        text_word()
    for w in novel + [frequent]:
        rows = []
        for k in range(8):
            for _ in range(11):
                text_word()
            before, _ = text_word()
            phi0 = st['s'].comp_memory.clone()
            during, ce = word(w)
            row = {k2: during[k2] - before[k2] for k2 in during}
            row['dphi_a3'] = float((st['s'].comp_memory - phi0)[0, a3_edges].mean())
            row['ce_on_arrival'] = ce
            rows.append(row)
        slope = {k2: float(np.polyfit(np.arange(8), [r[k2] for r in rows], 1)[0]) for k2 in rows[0]}
        kind = 'frequent' if w == frequent else 'novel'
        rep['words'].append({'token': w, 'kind': kind, 'train_count': int(counts[w]), 'per_presentation': rows, 'slope': slope})
        print(kind, w, 'first', {k2: round(v, 4) for k2, v in rows[0].items()},
              'slope', {k2: round(v, 5) for k2, v in slope.items()}, flush=True)
nov = [x for x in rep['words'] if x['kind'] == 'novel']
rep['novel_mean_first'] = {k: float(np.mean([x['per_presentation'][0][k] for x in nov])) for k in nov[0]['slope']}
rep['novel_mean_slope'] = {k: float(np.mean([x['slope'][k] for x in nov])) for k in nov[0]['slope']}
print('novel mean first', {k: round(v, 4) for k, v in rep['novel_mean_first'].items()})
print('novel mean slope', {k: round(v, 5) for k, v in rep['novel_mean_slope'].items()})
json.dump(rep, open(Path(sys.argv[2]) if len(sys.argv) > 2 else fork / 'word_novelty.json', 'w'), indent=1)
