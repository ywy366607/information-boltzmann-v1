"""Where does a word get lost on its way into the mushroom body?  Fixed weights.  From one living state (after a
silence of 40 ticks), each of 8 words is read for exactly 16 ticks (fixed, no go sampling) and every neuron's spikes
are counted; the same is done with no word (quiet).  Per group: how strongly the words drive it (mean |count(word) -
count(quiet)| per cell), and how word-specific the response is: 1 - Jaccard of the sets of cells whose count changed,
between two different words (0 = every word changes the same cells, 1 = disjoint).  Repeated in 6 contexts.
Groups along the olfactory path: written chemosensory cells (olfactory receptor neurons among them), olfactory
receptor neurons, antennal-lobe projection neurons (ALPN), Kenyon cells, MBONs, dopamine cells, motor read surface.
Usage: python word_path.py <fork dir> [<fork dir> ...]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

WORDS = [17180, 7850, 7815, 6279, 8509, 8223, 26839, 11376]
TICKS, CONTEXTS = 16, 6
original_step = fl.step_fly_physical_tick
box = {'count': None}


def step(model_, current, *a, **k):
    out = original_step(model_, current, *a, **k)
    if box['count'] is not None:
        box['count'] += (out.ring[0][0] > 0).float()
    return out


fl.step_fly_physical_tick = step
summary = {}
for fork in map(Path, sys.argv[1:]):
    model, cfg, L = load(fork)
    opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
                alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
    N = model.n_neurons
    graph = ROOT / cfg['graph']
    g = np.load(graph, allow_pickle=True)
    ann = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
    cls = ann['class'].astype(str).values
    written = np.zeros(N, bool); written[model.injection_index.cpu().numpy()] = True
    is_dan = np.zeros(N, bool); is_dan[g['dan_indices']] = True
    read = np.zeros(N, bool); read[model.read_indices.cpu().numpy()] = True
    groups = {'written sensory': written, 'olfactory receptor (written)': written & (cls == 'olfactory'),
              'olfactory receptor (all)': cls == 'olfactory', 'ALPN': cls == 'ALPN', 'Kenyon': cls == 'Kenyon_Cell',
              'MBON': cls == 'MBON', 'dopamine': is_dan, 'read surface': read}
    gm = {k: torch.as_tensor(v, device='cuda') for k, v in groups.items()}
    writer = model.topographic_writer
    quiet = torch.zeros(1, N, device='cuda')

    def run(s, word):
        box['count'] = torch.zeros(N, device='cuda')
        for t in range(TICKS):
            if t == 0 and word is not None:
                drive, baseline = fl.write_input(model, s, torch.tensor([word], device='cuda'))
                s = fl.step_fly_physical_tick(model, s, torch.tensor([word], device='cuda'), drive, baseline, opts, base_rates=opts['base_rates'])
            else:
                baseline = writer.lambda_adapt * s.baseline if writer is not None else s.baseline
                s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
        c = box['count']; box['count'] = None
        return c

    val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
    text = np.array(val[70_000:71_000], dtype=np.int64)
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    drive_rows, spec_rows = {k: [] for k in groups}, {k: [] for k in groups}
    torch.manual_seed(0)
    with torch.no_grad():
        pos = 0
        for ctx in range(CONTEXTS):
            for tok in text[pos:pos + 48]:
                s, _, _, _, _, _ = fl.advance_fly_token_go(model, s, torch.tensor([int(tok)], device='cuda'), [], max_ticks=32,
                                                           writer_baseline_clock='physical', **opts)
            pos += 48
            for _ in range(40):
                baseline = writer.lambda_adapt * s.baseline if writer is not None else s.baseline
                s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
            base = run(s, None)
            changed = {}
            for w in WORDS:
                c = run(s, w)
                d = (c - base).abs()
                for k, m in gm.items():
                    drive_rows[k].append(float(d[m].mean()))
                changed[w] = {k: (d[m] > 0).cpu().numpy() for k, m in gm.items()}
            for k in groups:
                for i in range(len(WORDS)):
                    for j in range(i + 1, len(WORDS)):
                        x, y = changed[WORDS[i]][k], changed[WORDS[j]][k]
                        union = (x | y).sum()
                        if union:
                            spec_rows[k].append(1.0 - float((x & y).sum() / union))
    rep = {}
    for k in groups:
        dr, sp = np.array(drive_rows[k]), np.array(spec_rows[k]) if spec_rows[k] else np.array([np.nan])
        rep[k] = {'cells': int(groups[k].sum()), 'mean_abs_count_change_per_cell': float(dr.mean()),
                  'word_specificity_min_med_mean_max': [float(np.nanmin(sp)), float(np.nanmedian(sp)), float(np.nanmean(sp)), float(np.nanmax(sp))]}
        print(fork.name, f'{k:30s}', rep[k], flush=True)
    summary[fork.name] = rep
    del model
    torch.cuda.empty_cache()
json.dump(summary, open('scratch/fly_stream/word_path.json', 'w'), indent=1)
print('done', flush=True)
