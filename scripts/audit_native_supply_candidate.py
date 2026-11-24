"""Bind SADL to raw/encoded ports of actual archived native experiences."""
import argparse
import json
from pathlib import Path

import torch

from scripts.ib_local.language_navigation_kanli_v4l import live_sigreg
from scripts.ib_local.native_supply_regularizer import sigreg_zero, supply


def examine(h, encoder, directions):
    h = h.float().clone().requires_grad_()
    z = h@encoder.T
    d = live_sigreg(z, z.new_empty(0, z.shape[1]), z.new_empty(0), directions)
    loss, values = supply(h, z, d, sigreg_zero(len(h), z))
    g, = torch.autograd.grad(loss, h)
    basis, _ = torch.linalg.qr(encoder.T.double(), mode='reduced')
    gd = g.double()
    unseen = gd-(gd@basis)@basis.T
    shared = h.detach().mean(0, keepdim=True)
    def variation(x):
        return float((x-x.mean(0)).square().sum()/x.square().sum().clamp_min(1e-12))
    return dict(**{key:float(value.detach()) if torch.is_tensor(value) else value
                    for key,value in values.items()}, loss=float(loss.detach()),
        gradient_norm=float(g.norm()),
        gradient_shared_sample_mode_fraction=float((len(g)*g.mean(0).square().sum())/
            g.square().sum().clamp_min(1e-30)),
        gradient_encoder_kernel_fraction=float(unseen.square().sum()/gd.square().sum().clamp_min(1e-30)),
        shared_mean_removal_directional_loss=float((-g*shared).sum()),
        full_feature_radial_derivative=float((g*h.detach()).sum()),
        raw_variation_energy_fraction=variation(h.detach()),
        encoded_variation_energy_fraction=variation(z.detach()),
        supply_branch='excitation' if values['k_exc'] <= values['k_tr'] else 'transmission')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ports', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    data = torch.load(a.ports, map_location='cpu', weights_only=False)
    e = data['model']['encoder.weight'].float()
    directions = data['directions']
    roles = {name:torch.stack([r[name] for r in data['episodes']])
             for name in ('source', 'action', 'mother_next')}
    roles['history'] = data['history'][data['history_valid'].bool()]
    results = {name:examine(h,e,directions) for name,h in roles.items()}
    chronological = [dict(first_cycle=data['episodes'][i]['receipt']['cycle'],
        last_cycle=data['episodes'][i+31]['receipt']['cycle'],
        **examine(roles['source'][i:i+32],e,directions)) for i in range(0,192,32)]
    # Euclidean projection records which raw variations the current encoder sees.
    q, _ = torch.linalg.qr(e.T.double(), mode='reduced')
    null_fractions = {}
    for name, raw in roles.items():
        x = raw.double()-raw.double().mean(0)
        residual = x-(x@q)@q.T
        null_fractions[name] = float(residual.square().sum()/x.square().sum().clamp_min(1e-12))
    report = dict(schema=1, cycle=data['cycle'],
        checkpoint=data['checkpoint'], checkpoint_sha256=data['checkpoint_sha256'],
        archive_sha256=data['archive_sha256'], events=data['events'],
        diagnostic_direction_seed=data['diagnostic_direction_seed'], roles=results,
        chronological_sources=chronological, encoder_kernel_variation_fraction=null_fractions,
        interface='actual pre-encoder raw body features -> shared live encoder',
        interpretation='supply deficit and gradient port; functions evaluated in paired complete life separately',
        world_calls_added=0, optimizer_updates_added=0)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(cycle=data['cycle'], roles=results,
        chronological_active=sum(r['loss']>0 for r in chronological),
        chronological_windows=len(chronological))))


if __name__ == '__main__':
    main()
