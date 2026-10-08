"""Numerical corroboration of analytic response feasibility, without training."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from information_boltzmann.core.plastic_medium import PlasticMedium3D
from information_boltzmann.core.conductance_feasibility import conductance_energy_bound, local_ei_certificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(490)
    net = PlasticMedium3D((4,4,4), 8, bath_type='conductance', adaptive_conduction=True).double()
    initial = net.initial_state()
    old = replace(initial, field=torch.randn_like(initial.field),
                  flux=tuple(torch.randn_like(x) for x in initial.flux),
                  receptors=torch.rand_like(initial.receptors))
    _, info = net.advance(old, 0.1, substeps=8)
    bounds = conductance_energy_bound(net.prepare_evolution().response)
    proof = local_ei_certificate()
    eigen = lambda values: [[float(x.real), float(x.imag)] for x in values]
    report = {'purpose': 'analytic/numerical feasibility, no task capability or training',
              'fixed_point_residual': float(proof['fixed_point_residual'].abs().max()),
              'actual_ei_jacobian': proof['jacobian'].tolist(),
              'excitation_only_eigenvalues': eigen(proof['excitation_only_eigenvalues']),
              'closed_loop_eigenvalues': eigen(proof['closed_loop_eigenvalues']),
              'electrical_ledger_max_residual': float(info['response_energy_residual'].detach().abs().max()),
              'source_work': float(info['response_source_work'][0].detach()),
              'joule_heat': float(info['response_joule_heat'][0].detach()),
              'bound': {k: float(v.detach()) for k,v in bounds.items()},
              'scope': 'local inhibition-stabilized damped oscillation; no autonomous limit-cycle claim',
              'response_parameters_d128': sum(p.numel() for p in PlasticMedium3D(
                  channels=128,bath_type='conductance').conductance_response.parameters()),
              'receptor_state_bytes_d128_8x8x4_fp32': 256*2*128*4}
    if report['electrical_ledger_max_residual'] > 1e-11:
        raise RuntimeError('Electrical ledger failed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
