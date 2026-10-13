"""Does a learned change carry the language?  Zero-shot swap on next-word NLL: load a checkpoint, put the birth values
back for parameter groups (env SWAP, groups as in swap_runaway_probe, plus 'model' = the whole birth model, 'noedges' = every synapse removed: the decoder alone), and read
1024 validation words in the training world (stream_ticks per word, input held if the world holds it), after 256 words
of burn-in from the inherited state.  Reports NLL [min | median | mean | max] over 4 places and the unigram NLL.
Usage: SWAP=edges_e BIRTH=<birth.pt> python swap_nll_probe.py <checkpoint dir>
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
GROUPS = {'edges_e': ['edge_weight_e'], 'edges_i': ['edge_weight_i'], 'threshold': ['log_threshold'],
          'alif': ['log_beta', 'log_tau_a'], 'membrane': ['log_tau_m'],
          'synapse': ['log_tau_s_e', 'log_tau_s_i', 'log_g_e', 'log_g_i'], 'intrinsic': ['intrinsic_kappa'],
          'dopamine': ['dopamine_sensitivity', 'dopamine_logit_fast', 'dopamine_logit_slow']}
swap = [g for g in os.environ.get('SWAP', '').split(',') if g]
fork = Path(sys.argv[1])
model, cfg, L = load(fork)
birth = torch.load(os.environ['BIRTH'], map_location='cpu', mmap=True, weights_only=False)['model']
params = dict(model.named_parameters()); params.update(dict(model.named_buffers()))
with torch.no_grad():
    names = list(birth) if 'model' in swap else [n for g in swap if g != 'noedges' for n in GROUPS[g]]
    if 'noedges' in swap:                       # the champion's control: no brain, the decoder alone (state kept)
        for n in ('edge_weight_e', 'edge_weight_i'):
            params[n].zero_()
    for n in names:
        if n in params and params[n].shape == birth[n].shape:
            params[n].copy_(birth[n].to(params[n].device, params[n].dtype))
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
K = int(cfg.get('stream_ticks') or 8); HOLD = bool(cfg.get('hold_input'))
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
train = np.load(ROOT / cfg['data'] / 'train.npy', mmap_mode='r')
counts = np.bincount(np.asarray(train[:5_000_000], dtype=np.int64), minlength=model.decoder.out_features) + 0.5
logp = np.log(counts / counts.sum())
BURN, SCORE = 256, int(os.environ.get('NLL_SCORE', 1024))
nll, uni = [], []
with torch.no_grad():
    for place in range(int(os.environ.get('NLL_PLACES', 4))):
        start = 80_000 + place * 4_000
        words = np.array(val[start:start + BURN + SCORE + 1], dtype=np.int64)
        s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
        out = []
        for i in range(BURN + SCORE):
            s, z = fl.advance_fly_token_stream(model, s, torch.tensor([int(words[i])], device='cuda'), ticks=K, hold_input=HOLD,
                                               writer_baseline_clock='physical', **opts)
            if i >= BURN:
                out.append(float(F.cross_entropy(model.decoder(model.read_norm(z)).float(), torch.tensor([int(words[i + 1])], device='cuda'))))
                uni.append(-logp[words[i + 1]])
        nll.append(float(np.mean(out)))
q = lambda v: [round(float(np.min(v)), 4), round(float(np.median(v)), 4), round(float(np.mean(v)), 4), round(float(np.max(v)), 4)]
rep = {'fork': str(fork), 'swap': swap or ['none'], 'ticks': K, 'hold': HOLD, 'nll_places_min_median_mean_max': q(nll), 'unigram': round(float(np.mean(uni)), 4)}
print(json.dumps(rep), flush=True)
