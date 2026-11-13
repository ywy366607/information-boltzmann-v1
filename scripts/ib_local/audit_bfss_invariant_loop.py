"""Read-only S/P quiet-flow and existing mother-loop readout audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.bfss_gdn_capture import matrix_view, quiet_packed
from scripts.ib_local.bfss_gdn_reference import rotation_charge
from scripts.ib_local.language_action import peek
from scripts.ib_local.navigation_internal_rollout import next_boundary_prediction
from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
from scripts.ib_local.navigation_relation_wood import local_relation
from scripts.ib_local.run_causal_water_reply import build


@torch.no_grad()
def observe(engine, recurrent, momenta):
    model = engine.model
    engine.bfss_momenta = momenta
    model._question_context = None
    byte = torch.tensor([[engine.pending]], device=engine.device)
    probs = peek(engine.table, engine.pending)
    logp = torch.as_tensor(np.log(np.maximum(probs, 1e-6)), device=engine.device)[None, None]
    _, _, _, h, _ = model.predict_observed(byte, recurrent, logp)
    relation, _ = local_relation(model, h, h, ridge=engine.relation_ridge, scale=engine.relation_scale)
    p_read = decision_momentum_read(model, momenta, h)
    wood = next_boundary_prediction(model, h, h) + model.momentum_to_boundary(p_read) + relation
    q = engine._condition_mother_question(h[0], relation)
    model._question_context = q[0]
    logits, _, _, _, _ = model.predict_observed(byte, recurrent, logp)
    model._question_context = None
    return dict(hidden=h.clone(), wood=wood.clone(), p_read=p_read.clone(),
                q=q.clone(), prob=logits[0,0].softmax(-1).clone())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('output must be new')
    torch.set_num_threads(1)
    gdn_fused.enable()
    engine, _, _, provenance = build(args.checkpoint)
    if engine.__class__.__name__ != 'MotherBoundaryEngine':
        raise ValueError('requires the current mother-boundary individual')
    model = engine.model
    s0 = engine.recurrent.detach().clone()
    p0 = [x.detach().clone() for x in engine.bfss_momenta]
    def charges(s, ps):
        return [rotation_charge(matrix_view(layer,s[i]),ps[i])
                for i,layer in enumerate(model.body.base.layers)]
    q0 = charges(s0,p0)
    with torch.no_grad():
        baseline = observe(engine,s0,p0)
        rows = []
        for n in (0,1,8,64):
            s, ps = quiet_packed(model.body.base,s0.clone(),[x.clone() for x in p0],
                dt=engine.bfss_dt,frequency=engine.bfss_frequency,
                coupling=engine.bfss_coupling,steps=n)
            qs = charges(s,ps)
            obs = observe(engine,s,ps)
            rows.append(dict(steps=n, s_change=float((s-s0).norm()),
                p_change=float(torch.stack([(x-y).norm() for x,y in zip(ps,p0)]).norm()),
                q=[dict(initial_norms=[float(q0[i][j].norm()) for j in range(2)],
                        error_norms=[float((qs[i][j]-q0[i][j]).norm()) for j in range(2)])
                   for i in range(len(qs))],
                hidden_change=float((obs['hidden']-baseline['hidden']).norm()),
                wood_change=float((obs['wood']-baseline['wood']).norm()),
                p_read_change=float((obs['p_read']-baseline['p_read']).norm()),
                q_condition_change=float((obs['q']-baseline['q']).norm()),
                byte_probability_tv=float((obs['prob']-baseline['prob']).abs().sum()/2)))
    report = dict(source=provenance, config=dict(dt=engine.bfss_dt,frequency=engine.bfss_frequency,
        coupling=engine.bfss_coupling), rows=rows,
        interpretation='Q preservation and S/P readout changes are not a Q-specific causal intervention; no world step')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
