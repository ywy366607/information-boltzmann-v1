"""Check the Lean equations' translation against production at a real anchor.

This numerical interface check is not training or a capability experiment.
NumPy follows reviews/NativeDynamics.lean; the other side calls the actual
PyTorch potential and quiet step. No random fixture or new state is generated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ib_local.adaptive_hamiltonian import potential, quiet


def scalar_potential(s, theta):
    total = 0.0
    heads = s.shape[1]
    for layer, x in enumerate(s):
        norm = (x*x).sum((-2, -1))
        correlation = np.einsum('hij,kij->hk', x, x) / np.sqrt(
            (1+norm[:, None])*(1+norm[None, :]))
        logits = (theta[layer, heads+2]*correlation + theta[layer, heads+3]*
                  (np.log1p(norm[:, None])+np.log1p(norm[None, :])))
        weight = 2*np.exp(-np.logaddexp(0.0, -logits))
        left = x[:, None] @ x[None].swapaxes(-1, -2)
        right = x[:, None].swapaxes(-1, -2) @ x[None]
        a = left-left.swapaxes(0, 1)
        b = right-right.swapaxes(0, 1)
        total += np.logaddexp(0.0, theta[layer, heads+1]) * (
            weight*((a*a).sum((-2, -1))+(b*b).sum((-2, -1)))).sum()/16
    return float(total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('anchor', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with np.load(args.anchor) as saved:
        a = {k: saved[k].astype(np.float64) for k in
             ('S', 'P', 'theta', 'context', 'frequency_scale', 'impedance')}
    s, p, theta = a['S'], a['P'], a['theta']
    if s.shape != (32, 8, 32, 32) or p.shape != s.shape:
        raise ValueError('anchor does not match the formalized native body')
    def tensor(x):
        return torch.tensor(x, dtype=torch.float64)
    th, context = tensor(theta), tensor(a['context'])
    scale, impedance = tensor(a['frequency_scale']), tensor(a['impedance'])
    def gradient(x):
        tx = tensor(x[:, None]).requires_grad_(True)
        value = potential(tx, th, 8).sum()
        return float(value.detach()), torch.autograd.grad(value, tx)[0].detach().numpy()[:, 0]
    production_value, grad = gradient(s)
    reference_value = scalar_potential(s, theta)
    probes = []
    for index in ((0, 0, 0, 0), (15, 3, 7, 11), (31, 7, 31, 31)):
        step = 1e-4*(1+abs(s[index]))
        plus, minus = s.copy(), s.copy()
        plus[index] += step
        minus[index] -= step
        finite = (scalar_potential(plus, theta)-scalar_potential(minus, theta))/(2*step)
        probes.append(dict(index=index, finite_difference=finite, production=float(grad[index]),
                           absolute_error=abs(finite-grad[index])))
    dt = .01
    w = np.logaddexp(0.0, theta[:, :8]+theta[:, 8:9]*a['context'].reshape(32, 8))
    w *= a['frequency_scale'].reshape(32, 8)
    z = a['impedance'].reshape(32, 8, 1, 1)
    angle = (w*dt)[..., None, None]
    half = p-dt/2*grad
    ns = np.cos(angle)*s+np.sin(angle)/z*half
    np_half = np.cos(angle)*half-z*np.sin(angle)*s
    _, next_grad = gradient(ns)
    np_final = np_half-dt/2*next_grad
    actual_s, actual_p = quiet(tensor(s[:, None]), tensor(p[:, None]), th, context, dt,
                               scale, impedance)
    errors = dict(potential=abs(reference_value-production_value),
                  S=float(abs(ns-actual_s.numpy()[:, 0]).max()),
                  P=float(abs(np_final-actual_p.numpy()[:, 0]).max()))
    passed = (errors['potential'] < 1e-10*(1+abs(production_value)) and
              max(errors['S'], errors['P']) < 1e-10 and
              all(v['absolute_error'] < 1e-6*(1+abs(v['production'])) for v in probes))
    root = Path(__file__).resolve().parents[1]
    report = dict(anchor=str(args.anchor), anchor_sha256=hashlib.sha256(args.anchor.read_bytes()).hexdigest(),
                  shape=list(s.shape), potential=production_value, errors=errors,
                  gradient_probes=probes, passed=passed,
                  source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
                      'lean/CGBAxioms/reviews/NativeDynamics.lean',
                      'scripts/ib_local/adaptive_hamiltonian.py',
                      'scripts/ib_local/bfss_gdn_capture.py')},
                  meaning='real-anchor equation translation check; no training or synchronization claim')
    args.output.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report))
    if not passed:
        raise SystemExit('equation bridge check failed')


if __name__ == '__main__':
    main()
