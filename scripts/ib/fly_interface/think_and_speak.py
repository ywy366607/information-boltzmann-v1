"""Two questions on the current best individual (virtual go neuron + no-window read), fixed weights.

A. Does it think longer when it is harder?  It reads 1,024 fresh validation tokens with its own go neuron; per token
   we record how many ticks it took and how hard the token was (its prediction error at the input tick, before the
   token could have been processed; its realized error; the target's corpus frequency).
B. Without any outside input, does it speak with content?  After reading a real passage, the world goes silent and the
   brain hears its own voice: every time its go neuron fires, the word it says (sampled from its prediction) is
   written back as its next input.  We decode what it says and test whether it relates to what it had read: mean
   embedding cosine of its words with ITS passage vs. with other passages, and word overlap.
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
import tiktoken
sys.path.insert(0, 'D:/information-boltzmann-v1')
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
import information_boltzmann.core.fly_bptt_learning as fl
from scripts.ib.inspect_fly_bptt_credit import move_state

ROOT = Path('D:/information-boltzmann-v1')
fork = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else ROOT / 'scratch/fly_stream/base_gb'
saved = torch.load(fork / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
cfg, L = saved['config'], saved['learner']
graph = ROOT / cfg['graph']
model = FlyReservoirLM(graph, vocab_size=50257, d_model=cfg['d_model'], injection='topographic', read_surface='output',
                       synapse_model='coba', use_alif=True, use_stp=True, decoder_bias=cfg['decoder_bias'],
                       detach_reset=cfg.get('detach_reset', False), surrogate_mode=cfg.get('surrogate_mode', 'absolute'),
                       transmission_mode=cfg.get('transmission_mode', 'atomic')).cuda()
if cfg.get('go_neuron') == 'virtual':
    fl.attach_go_neuron(model)
if cfg.get('read_bank'):
    fl.attach_read_bank(model)
if cfg.get('intrinsic'):
    fl.attach_intrinsic_drive(model, graph, graph.parent / 'body-annotations.feather')
edges = ('edge_weight_e', 'edge_weight_i')
with torch.no_grad():
    for k in edges:
        getattr(model, k).copy_(saved['model'][k])
    assert not model.load_state_dict({k: v for k, v in saved['model'].items() if k not in edges}, strict=False).unexpected_keys
model.eval().requires_grad_(False)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
max_ticks = int(L.get('go_max_ticks', 32))
enc = tiktoken.get_encoding('gpt2')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
cursor = int(saved['val_cursor'])
train = np.load(ROOT / cfg['data'] / 'train.npy', mmap_mode='r')
counts = np.bincount(np.array(train[:5_000_000], dtype=np.int64), minlength=50257) + 1.0
log_freq = np.log(counts / counts.sum())


def advance(state, token):
    state, lat, _, _, _ = fl.advance_fly_token_go(model, state, torch.tensor([int(token)], device='cuda'), [],
                                                  max_ticks=max_ticks, writer_baseline_clock='physical', **opts)
    return state, lat


report = {'fork': str(fork)}
# ---------------- A. thinking time vs difficulty ----------------
torch.manual_seed(0)
targets = np.array(val[cursor:cursor + 1280], dtype=np.int64)
inputs = np.concatenate(([L['previous_token']], targets[:-1]))
rows = []
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
with torch.no_grad():
    for i, tok in enumerate(inputs):
        s, lat = advance(s, tok)
        if i >= 256:
            ce = F.cross_entropy(model.decoder(model.read_norm(lat)).float(),
                                 torch.full((lat.shape[0],), int(targets[i]), device='cuda'), reduction='none')
            rows.append((lat.shape[0], float(ce[0]), float(ce[-1]), float(log_freq[targets[i]]), float(log_freq[tok])))
end_state = s
a = np.array(rows)
ticks = a[:, 0]


def corr(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


report['thinking'] = {
    'ticks_min_median_mean_max': [float(ticks.min()), float(np.median(ticks)), float(ticks.mean()), float(ticks.max())],
    'spearman_ticks_vs_error_at_input_tick': corr(ticks, a[:, 1]),
    'spearman_ticks_vs_realized_error': corr(ticks, a[:, 2]),
    'spearman_ticks_vs_target_log_frequency': corr(ticks, a[:, 3]),
    'spearman_ticks_vs_input_log_frequency': corr(ticks, a[:, 4]),
    'gain_by_thinking_vs_ticks': corr(ticks, a[:, 1] - a[:, 2])}
hard = a[:, 1] > np.median(a[:, 1])
report['thinking']['mean_ticks_hard_vs_easy'] = [float(ticks[hard].mean()), float(ticks[~hard].mean())]
print('A. thinking time:', json.dumps(report['thinking'], indent=1), flush=True)

if '--only-thinking' in sys.argv:
    json.dump(report, open(fork / 'think_and_speak.json', 'w'), indent=1)
    sys.exit()
# ---------------- B. silent self-talk ----------------
PROMPT, SPEAK = 160, 60
starts = [cursor + 3000 + 997 * j for j in range(6)]
passages, spoken = [], []
for j, st in enumerate(starts):
    torch.manual_seed(100 + j)
    passage = np.array(val[st:st + PROMPT], dtype=np.int64)
    s = end_state
    with torch.no_grad():
        for tok in passage[:-1]:
            s, lat = advance(s, tok)
        said, last = [], int(passage[-1])
        for _ in range(SPEAK):
            s, lat = advance(s, last)                       # the world is silent: it hears its own last word
            logits = model.decoder(model.read_norm(lat[-1:])).float()[0]
            top = torch.topk(logits, 40)
            last = int(top.indices[torch.multinomial(torch.softmax(top.values, -1), 1)])
            said.append(last)
    passages.append(passage)
    spoken.append(said)
    print(f'\n--- passage {j} (last 40 tokens read) ---\n{enc.decode(passage[-40:].tolist())!r}\n--- it then said ---\n{enc.decode(said)!r}', flush=True)

emb = torch.nn.functional.normalize(model.embedding.weight.detach().float(), dim=1).cpu().numpy()
def mean_vec(toks):
    v = emb[np.array(toks)].mean(0)
    return v / np.linalg.norm(v)
sim = np.array([[float(mean_vec(spoken[i]) @ mean_vec(passages[j])) for j in range(len(passages))] for i in range(len(spoken))])
own = np.diag(sim)
others = (sim.sum(1) - own) / (len(passages) - 1)
content_words = lambda toks: {t for t in toks if log_freq[t] < np.log(1e-5)}          # rarer words only
overlap_own = [len(content_words(spoken[i]) & content_words(passages[i].tolist())) for i in range(len(spoken))]
overlap_other = [np.mean([len(content_words(spoken[i]) & content_words(passages[j].tolist()))
                          for j in range(len(passages)) if j != i]) for i in range(len(spoken))]
distinct = [len(set(x)) / len(x) for x in spoken]
report['self_talk'] = {'embedding_cosine_own_passage': own.tolist(), 'embedding_cosine_other_passages': others.tolist(),
                       'rare_word_overlap_own': overlap_own, 'rare_word_overlap_other_mean': overlap_other,
                       'distinct_token_ratio': distinct,
                       'texts': [enc.decode(x) for x in spoken]}
print('\nB. self-talk content: cosine own vs other passages', np.round(own, 3).tolist(), np.round(others, 3).tolist(),
      '\n   rare-word overlap own vs other', overlap_own, np.round(overlap_other, 2).tolist(),
      '\n   distinct-token ratio', np.round(distinct, 2).tolist(), flush=True)
json.dump(report, open(fork / 'think_and_speak.json', 'w'), indent=1)
