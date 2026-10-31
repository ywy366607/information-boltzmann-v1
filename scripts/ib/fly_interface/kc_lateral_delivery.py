"""One native receptor step from real archived packets, without whole-brain replay.

The thin module contains the unchanged graph endpoint arrays and full KC tree,
not an invented stimulus or a reduced geometry. This does not evolve the rest
of the brain, assess action/NLL utility or certify training admission.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_kc_terminals import CellwiseGeometry
from information_boltzmann.core.fly_kc_lateral import attach_kc_lateral, advance_kc_lateral, effector_gate
from scripts.ib.fly_interface.native_causal_audit import sha256


def audit(checkpoint, output):
    output.mkdir(parents=True, exist_ok=True)
    before = sha256(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    config, physical = saved['config'], saved['learner']['physical']
    model = nn.Module()
    with np.load(config['graph'], allow_pickle=False) as z:
        for name in ('edge_pre_e', 'edge_post_e'):
            model.register_buffer(name, torch.from_numpy(z[name]))
    with np.load(config['kc_terminal_geometry'], allow_pickle=False) as z:
        for attr, key in [('kct_index','kc_index'),('kct_owner','owner'),
                          ('kct_site_node','site_node'),('kct_site_owner','site_owner')]:
            model.register_buffer(attr, torch.from_numpy(z[key]).long())
        model.kct_geometries = (CellwiseGeometry(z['volume'], z['u'], z['v'], z['g'], z['owner']),)
    model.register_buffer('kc_terminal_digest', saved['model']['kc_terminal_digest'])
    model.kc_contact_memory_enabled = True
    attach_kc_lateral(model, config['kc_lateral_geometry'], saved=saved['model'])
    model.requires_grad_(False)
    current = SimpleNamespace(h=physical['h'], kc_lateral_receptor=physical['kc_lateral_receptor'],
                              kc_lateral_signal=physical['kc_lateral_signal'])
    packet = physical['kc_terminal_packets']
    sites = torch.arange(model.kct_site_owner.numel())
    contact_sites = saved['model']['kc_contact_ids'][:,0].long()
    baseline, results = None, {}
    with torch.no_grad():
        for arm in ('intact', 'sham', 'source_off', 'calcium_off', 'camp_off', 'restored'):
            model.kc_lateral_delivery_enabled = arm != 'source_off'
            model.kc_lateral_calcium_enabled = arm != 'calcium_off'
            model.kc_lateral_camp_enabled = arm != 'camp_off'
            receptor, signal = advance_kc_lateral(model, current, packet)
            calcium = effector_gate(model, signal, sites, 'calcium')
            camp = effector_gate(model, signal, contact_sites, 'camp')
            if baseline is None:
                baseline = receptor, signal, calcium, camp
            values = receptor, signal, calcium, camp
            results[arm] = {'receptor_max':float(receptor.max()), 'effector_max':float(signal.max()),
                'calcium_gate_min':float(calcium.min()), 'camp_gate_min':float(camp.min()),
                'calcium_gated_sites':int((calcium < 1-1e-6).sum()),
                'camp_gated_plastic_contacts':int((camp < 1-1e-6).sum()),
                'absmax_delta_from_intact':{key:float((a-b).abs().max())
                    for key,a,b in zip(('receptor','signal','calcium_gate','camp_gate'),values,baseline)}}
    controls = all(all(v == 0 for v in results[a]['absmax_delta_from_intact'].values())
                   for a in ('sham','restored'))
    delivered = results['intact']['calcium_gated_sites'] > 0 and results['intact']['camp_gated_plastic_contacts'] > 0
    report = {'checkpoint':str(checkpoint), 'sha256':before,
        'scope':'One receptor/effector tick on the full real KC morphology, actual saved physical-delay packet slots; no full-brain/action/NLL replay',
        'nodes':int(model.kct_owner.numel()),'source_packet_nonzero':int(torch.count_nonzero(packet)),
        'anatomy':model.kc_lateral_manifest,'arms':results,'controls_passed':controls,
        'dual_effector_delivery_observed':delivered,'source_unchanged':before==sha256(checkpoint),
        'full_trainer_restore':'unmeasured for this candidate', 'learning_or_behavior_benefit':'unmeasured',
        'status':'acute_dual_effector_delivery_passed' if controls and delivered else 'not_accepted'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)
    if report['status'] != 'acute_dual_effector_delivery_passed':
        raise RuntimeError('Actual-packet local delivery not accepted')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.checkpoint,args.output)
