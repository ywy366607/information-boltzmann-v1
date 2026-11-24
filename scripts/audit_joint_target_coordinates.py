"""Read-only differential frame of encoded state AND actual momentum read.

Uses stored 2537 weights/S/P and actual native-bank source conditions. The
fixed-context frame is a sub-readout of the native packet. It determines the
local raw-source distinctions available jointly through these two ports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
from scripts.ib_local import native_amortized_decision as native


def build_model(parameters, momenta):
    w = parameters['encoder.weight'].double()
    model = nn.Module()
    model.encoder = nn.Linear(w.shape[1], w.shape[0], bias=False).double()
    model.encoder.weight.data.copy_(w)
    mp = parameters['momentum_to_boundary.weight'].double()
    model.momentum_to_boundary = nn.Linear(mp.shape[1], mp.shape[0], bias=False).double()
    model.momentum_to_boundary.weight.data.copy_(mp)
    layers = []
    for i, p in enumerate(momenta):
        heads, dk = p.shape[-3:-1]
        norm = nn.RMSNorm(w.shape[1], eps=torch.finfo(torch.float32).eps).double()
        norm.weight.data.copy_(parameters[f'body.base.layers.{i}.norm.weight'])
        q = nn.Linear(w.shape[1], heads*dk, bias=False).double()
        q.weight.data.copy_(parameters[f'body.base.layers.{i}.q_proj.weight'])
        norm.requires_grad_(False); q.requires_grad_(False)
        layers.append(SimpleNamespace(norm=norm, q_proj=q, heads=heads, dk=dk))
    model.body = SimpleNamespace(base=SimpleNamespace(layers=layers))
    model._packed_momentum_enabled = False
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ports', type=Path, default=Path('scratch/cgb_math_contracts_20261122/source_ports_2537.pt'))
    parser.add_argument('--queries', type=Path, default=Path('scratch/cgb_math_contracts_20261122/query_ports_2537.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/published/cgb_joint_target_coordinates_20261122.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    ports = torch.load(args.ports, weights_only=False, map_location='cpu')
    queries = torch.load(args.queries, weights_only=False, map_location='cpu')
    momenta = [p.double() for p in ports['engine']['bfss_momenta']]
    model = build_model(queries['parameters'], momenta)
    w = model.encoder.weight
    _, _, vh = torch.linalg.svd(w, full_matrices=True)
    null = vh[w.shape[0]:].T
    samples, seen = [], set()
    for record in ports['engine']['native_value_bank']['records']:
        raw = record['packet']['raw'][0].double()
        key = raw.numpy().tobytes()
        if key in seen:
            continue
        seen.add(key); samples.append((raw, record['packet']))
        if len(samples) == 8:
            break
    rows = []
    for index, (raw, packet) in enumerate(samples):
        def momentum(h):
            return model.momentum_to_boundary(decision_momentum_read(model, momenta, h[None]))[0]
        jac = torch.func.jacrev(momentum, chunk_size=16)(raw).detach()
        hidden = jac@null
        hidden_sv = torch.linalg.svdvals(hidden)
        joint = torch.cat((w, jac))
        joint_sv = torch.linalg.svdvals(joint)
        # The very same numerical source direction has zero encoded response
        # and a measured response through the actual momentum query/read.
        _, _, vhidden = torch.linalg.svd(hidden)
        direction = null@vhidden[0]
        step = raw.norm()*1e-5
        finite = (momentum(raw+step*direction)-momentum(raw-step*direction))/(2*step)
        derivative = jac@direction
        right = torch.linalg.solve(w@w.T, w).T
        # Bind the Lean reconstruction to actual matrices C=E^+ and
        # K=N(JN)^(-1), tested on all 256 coordinate directions at once.
        kernel_inverse = torch.linalg.solve(hidden.T, null.T).T
        reconstruction = right@w+kernel_inverse@(jac-jac@right@w)
        raw_blocks = packet['raw'].double().clone()
        raw_blocks[0] = raw
        raw_blocks[1] = right@momentum(raw)
        encoded = model.encoder(raw_blocks).detach()
        blocks, width = encoded.shape
        dz = encoded.new_zeros(blocks, width, len(raw))
        dz[0], dz[1] = w, jac
        norms = encoded.norm(dim=-1)
        dot = torch.einsum('bi,bij->bj', encoded, dz)
        safe = norms.clamp_min(1e-12)
        du = dz/safe[:, None, None]-encoded[:, :, None]*dot[:, None]/safe[:, None, None].pow(3)
        du = torch.where((norms > 1e-12)[:, None, None], du, dz/1e-12)
        magnitudes = torch.log1p(norms/math.sqrt(width))
        dm = dot/(math.sqrt(width)*safe[:, None]*(1+norms/math.sqrt(width))[:, None])
        scalars = encoded.new_tensor(packet['scalars'])
        augmented = torch.cat((magnitudes, scalars))
        da = torch.cat((dm, dm.new_zeros(len(scalars), len(raw))))
        den = augmented.norm()
        da = da/den-augmented[:, None]*(augmented@da)[None]/den.pow(3)
        feature_jac = torch.cat((du.flatten(0, 1)/math.sqrt(blocks), da))/math.sqrt(2)
        fsv = torch.linalg.svdvals(feature_jac)
        def actual_feature(h):
            updated = torch.cat((h[None], (right@momentum(h))[None], raw_blocks[2:]))
            return native.features(SimpleNamespace(model=model), [dict(packet, raw=updated)])[0]
        _, direct = torch.func.jvp(actual_feature, (raw,), (direction,))
        feature_finite = (actual_feature(raw+step*direction)-actual_feature(raw-step*direction))/(2*step)
        feature_jvp_error = float((feature_finite-feature_jac@direction).norm()/feature_finite.norm())
        q = actual_feature(raw).detach()
        # Differential RKHS distance for k(q,r)=(1+q.r)^2:
        # ||Dpsi(q)v||² = 2(1+||q||²)||v||²+2(q.v)².
        qtj = q@feature_jac
        kernel_metric = (2*(1+q.square().sum())*(feature_jac.T@feature_jac)
                         +2*torch.outer(qtj, qtj))
        kernel_sv = torch.linalg.eigvalsh(kernel_metric).clamp_min(0).sqrt()
        rows.append(dict(index=index, condition_sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
            encoder_rank=int(torch.linalg.matrix_rank(w)),
            momentum_rank_on_encoder_kernel=int(torch.linalg.matrix_rank(hidden)),
            joint_rank=int(torch.linalg.matrix_rank(joint)),
            hidden_singular_min=float(hidden_sv[-1]), hidden_singular_max=float(hidden_sv[0]),
            joint_singular_min=float(joint_sv[-1]), joint_singular_max=float(joint_sv[0]),
            joint_condition_number=float(joint_sv[0]/joint_sv[-1]),
            complementary_reconstruction_max_error=float((reconstruction-
                torch.eye(len(raw), dtype=raw.dtype)).abs().max()),
            packet_feature_rank=int(torch.linalg.matrix_rank(feature_jac)),
            packet_feature_singular_min=float(fsv[-1]), packet_feature_singular_max=float(fsv[0]),
            packet_feature_norm=float(q.norm()),
            feature_tangent_radial_component_max=float(qtj.abs().max()),
            kernel_metric_isotropy_relative_error=float((kernel_metric-
                2*(1+q.square().sum())*(feature_jac.T@feature_jac)).norm()/kernel_metric.norm()),
            packet_feature_condition_number=float(fsv[0]/fsv[-1]),
            polynomial_kernel_singular_min=float(kernel_sv[0]),
            polynomial_kernel_singular_max=float(kernel_sv[-1]),
            polynomial_kernel_condition_number=float(kernel_sv[-1]/kernel_sv[0]),
            packet_feature_jvp_relative_error=feature_jvp_error,
            packet_feature_check='central finite difference of production features',
            forward_ad_finite=bool(torch.isfinite(direct).all()),
            encoder_null_direction_response=float((w@direction).norm()),
            momentum_null_direction_response=float(derivative.norm()),
            finite_difference_relative_error=float((finite-derivative).norm()/derivative.norm())))
    result = dict(schema=1, cycle=ports['cycle'], samples=rows,
        interpretation='fixed current P, weights, and packet context; joint differential readout, not a task-performance score',
        zero_momentum_control_joint_rank=int(torch.linalg.matrix_rank(w)),
        ports_sha256=hashlib.sha256(args.ports.read_bytes()).hexdigest(),
        queries_sha256=hashlib.sha256(args.queries.read_bytes()).hexdigest(),
        world_calls=0, training_updates=0)
    assert rows and all(r['finite_difference_relative_error'] < 1e-4 for r in rows)
    assert all(r['packet_feature_jvp_relative_error'] < 1e-8 for r in rows)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
