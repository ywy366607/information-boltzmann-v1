"""Which learned change makes the brain run away?  Zero-shot swap: load a checkpoint, put the birth values back for one
parameter group (env SWAP=group[,group]), and measure the brain rate and the cells latched above 200 Hz with the words
withheld (text_leak_probe's measure).  Groups: edges_e, edges_i, threshold, alif (log_beta, log_tau_a),
membrane (log_tau_m), synapse (log_tau_s_e, log_tau_s_i, log_g_e, log_g_i), intrinsic, dopamine (dopamine_*),
state (birth physical state), conserve_e / conserve_i (each cell's learned incoming synapses rescaled to its birth total).  Env BIRTH=<birth.pt>.
Usage: SWAP=threshold python swap_runaway_probe.py <checkpoint dir>
"""
import os, sys, torch
src = open('D:/information-boltzmann-v1/scripts/ib/fly_interface/text_leak_probe.py', encoding='utf-8').read()
GROUPS = {'edges_e': ['edge_weight_e'], 'edges_i': ['edge_weight_i'], 'threshold': ['log_threshold'],
          'alif': ['log_beta', 'log_tau_a'], 'membrane': ['log_tau_m'],
          'synapse': ['log_tau_s_e', 'log_tau_s_i', 'log_g_e', 'log_g_i'], 'intrinsic': ['intrinsic_kappa'],
          'dopamine': ['dopamine_sensitivity', 'dopamine_logit_fast', 'dopamine_logit_slow']}
swap = [g for g in os.environ.get('SWAP', '').split(',') if g]
birth = torch.load(os.environ['BIRTH'], map_location='cpu', mmap=True, weights_only=False)
hook = '''model, cfg, L = load(fork)
'''
inject = hook + '''with torch.no_grad():
    for g in SWAP_GROUPS:
        if g in ('conserve_e', 'conserve_i'):
            # heterosynaptic conservation: rescale every cell's incoming synapses so their total equals the birth total
            sfx = g[-1]
            w = getattr(model, f'edge_weight_{sfx}'); post = getattr(model, f'edge_post_{sfx}').long()
            w0 = BIRTH['model'][f'edge_weight_{sfx}'].to(w.device, w.dtype)
            s0 = torch.zeros(model.n_neurons, device=w.device).index_add_(0, post, w0)
            s1 = torch.zeros(model.n_neurons, device=w.device).index_add_(0, post, w.detach())
            w.mul_((s0 / s1.clamp_min(1e-12))[post])
            continue
        if g == 'state':
            L['physical'] = BIRTH['learner']['physical']
            continue
        for name in GROUPS[g]:
            p = dict(model.named_parameters()).get(name)
            if p is None:
                p = dict(model.named_buffers())[name]
            p.copy_(BIRTH['model'][name].to(p.device, p.dtype))
print('swapped to birth:', SWAP_GROUPS or 'none', flush=True)
'''
assert hook in src
src = src.replace(hook, inject, 1)
g = {'__name__': '__main__', 'SWAP_GROUPS': swap, 'BIRTH': birth, 'GROUPS': GROUPS}
exec(compile(src, 'text_leak_probe.py', 'exec'), g)
