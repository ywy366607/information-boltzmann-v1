"""Same frozen 32-layer ports: warm proposal/commit and rejection costs."""
import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.ib_local.adaptive_hamiltonian import AdaptiveHamiltonian
from scripts.ib_local.contact_momentum import ContactMomentum
from scripts.quiet_force_jet import prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('scratch/cgb_math_contracts_20261122/source_ports_2537.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/published/cgb_prepared_quiet_20261122.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = torch.load(args.input, map_location='cpu', weights_only=False)
    state = {k.removeprefix('head_coupling.'): v for k, v in source['dynamics'].items()}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    module = AdaptiveHamiltonian(32, 8, 32, 32)
    module.contact = ContactMomentum(32, 8, 32, 32)
    module.load_state_dict(state); module.to(device)
    initial = copy.deepcopy(module.state_dict())
    base = SimpleNamespace(layers=[SimpleNamespace(heads=8, dk=32, dv=32, matrix_size=8192)]*32)
    s = source['engine']['recurrent'].to(device)
    p = [v.to(device) for v in source['engine']['bfss_momenta']]
    dt = source['engine']['bfss_dt']
    synchronize = torch.cuda.synchronize if device == 'cuda' else lambda: None
    results, reference = {}, {}
    for accepted in (False, True):
        for name in ('reference', 'prepared'):
            times, peaks = [], []
            for trial in range(5 if device == 'cuda' else 2):
                module.load_state_dict(initial)
                synchronize()
                if device == 'cuda': torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with torch.no_grad():
                    if name == 'reference':
                        ns, np = module.quiet(base, s, p, dt=dt, capture=False)
                        if accepted: ns, np = module.quiet(base, s, p, dt=dt, capture=True)
                    else:
                        proposal = prepare(module, base, s, p, dt=dt)
                        if accepted: ns, np = proposal.commit(s, p)
                        else:
                            ns, np = proposal.result, list(proposal.next_p.unbind())
                            proposal.close()
                synchronize()
                elapsed = time.perf_counter()-started
                outputs = [ns, torch.stack(np), module.sensitivity_s, module.sensitivity_p,
                           module.contact.sensitivity_s, module.contact.sensitivity_p]
                if name == 'reference': reference[accepted] = [v.detach().clone() for v in outputs]
                errors = [float((x-y).abs().max()) for x, y in zip(outputs, reference[accepted])]
                if trial >= (2 if device == 'cuda' else 1):
                    times.append(elapsed)
                    if device == 'cuda': peaks.append(torch.cuda.max_memory_allocated()/2**20)
            assert max(errors) < 2e-5
            results[f'{name}_{"accepted" if accepted else "rejected"}'] = dict(
                seconds=times, median_seconds=statistics.median(times),
                peak_mib=max(peaks) if peaks else None, max_errors=errors)
    r, q = results, results
    old_half = .5*(r['reference_accepted']['median_seconds']+r['reference_rejected']['median_seconds'])
    new_half = .5*(q['prepared_accepted']['median_seconds']+q['prepared_rejected']['median_seconds'])
    result = dict(device=device, cycle=source['cycle'], results=results,
                  equal_accept_reject_expected_speedup=old_half/new_half,
                  scope='frozen complete 32-layer internal proposal/commit port; actual selection frequency belongs to life trace',
                  source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  training_updates=0, world_calls=0)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__': main()
