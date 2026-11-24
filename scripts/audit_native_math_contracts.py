"""Read-only numerical bindings for the target/first-release math contracts.

Inputs are extracted, trusted snapshot ports. This checks feature inversion,
the production kernel's repeated-row Schur residual, and the actual 32-layer
quiet map's first-order transport. It performs no learning or world calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local import latent_action_planner as action
from scripts.ib_local import native_amortized_decision as native
from scripts.ib_local.adaptive_hamiltonian import AdaptiveHamiltonian, quiet
from scripts.ib_local.contact_momentum import ContactMomentum


def feature_binding(snapshot):
    weight = snapshot['encoder']['encoder.weight']
    encoder = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    encoder.weight.data.copy_(weight)
    engine = SimpleNamespace(model=SimpleNamespace(encoder=encoder))
    layers = snapshot['engine']['recurrent'].shape[0]
    reports = {}
    first_feature = None
    for name, module, key in [('native', native, 'native_value_bank'),
                              ('action', action, 'latent_action_bank')]:
        records = snapshot['engine'][key]['records']
        packets = [r['packet'] for r in records]
        if not packets:
            reports[name] = {'records': 0}
            continue
        with torch.no_grad():
            got = module.features(engine, packets)
            raw = torch.stack([p['raw'] for p in packets]).to(weight)
            z = encoder(raw).double()
        if first_feature is None:
            first_feature = got[:1].clone()
        batch, blocks, width = z.shape
        # q = [per-block direction/sqrt(blocks), normalized augmentation]/sqrt(2).
        split = blocks*width
        direction = got[:, :split].double().reshape(batch, blocks, width)*math.sqrt(2*blocks)
        augmented = got[:, split:].double()*math.sqrt(2)
        scalars = torch.tensor([p['scalars'] for p in packets], dtype=torch.float64)
        # recipe appends gamma, then 3 intero, layer RMS values, and 2 read statistics.
        index = scalars.shape[1]-(3+layers+2)-1 if name == 'native' else 6
        anchor = scalars[:, index]
        assert bool((anchor != 0).all())
        assert torch.allclose(anchor, anchor[:1].expand_as(anchor), atol=0, rtol=0)
        denominator = anchor/augmented[:, blocks+index]
        restored = augmented*denominator[:, None]
        magnitude = math.sqrt(width)*restored[:, :blocks].expm1()
        z_back = direction*magnitude.clamp_min(1e-12)[:, :, None]
        reports[name] = dict(records=len(records), role_blocks=blocks, width=width,
            feature_width=got.shape[1], anchor_scalar_index=index, anchor_value=float(anchor[0]),
            encoded_max_abs_error=float((z-z_back).abs().max()),
            encoded_relative_l2_error=float((z-z_back).norm()/z.norm().clamp_min(1e-12)),
            scalar_max_abs_error=float((scalars-restored[:, blocks:]).abs().max()),
            zero_encoded_blocks=int((z.norm(dim=-1)==0).sum()))
        assert reports[name]['encoded_relative_l2_error'] < 2e-5
    return reports, first_feature


def repeated_row_binding(feature, ridge):
    """An algebraic kernel witness using an actual snapshot feature.

    Alternating labels are declared numerical inputs, not observed conflicts.
    The test binds the closed formula to the actual Cholesky calculation.
    """
    rows = []
    feature = feature.double()
    k = float(native._kernel(feature, feature)[0, 0])
    for n in (2, 4, 8, 16):
        x = feature.expand(n, -1)
        gram = native._kernel(x, x)
        factor = torch.linalg.cholesky(gram+ridge*torch.eye(n, dtype=x.dtype))
        cross = native._kernel(feature, x)
        projected = torch.linalg.solve_triangular(factor, cross.T, upper=False)
        novelty = float((k-projected.square().sum())/k)
        expected = ridge/(ridge+n*k)
        labels = torch.tensor([1., -1.], dtype=x.dtype).repeat(n//2)
        prediction = float(cross@torch.cholesky_solve(labels[:, None], factor))
        mse = float((labels-prediction).square().mean())
        assert abs(novelty-expected) < 1e-10 and abs(mse-1) < 1e-10
        rows.append(dict(repetitions=n, actual_schur_novelty=novelty,
                         closed_form=expected, declared_label_mse=mse))
    return dict(kind='numerical identity binding; declared alternating labels',
                actual_feature_self_kernel=k, ridge=ridge, cases=rows)


def quiet_binding(snapshot):
    d = {k.removeprefix('head_coupling.'): v.clone()
         for k, v in snapshot['dynamics'].items() if k.startswith('head_coupling.')}
    recurrent = snapshot['engine']['recurrent'].clone()
    momenta = [v.clone() for v in snapshot['engine']['bfss_momenta']]
    layers = recurrent.shape[0]
    heads, dk, dv = momenta[0].shape[-3:]
    module = AdaptiveHamiltonian(layers, heads, dk, dv)
    module.contact = ContactMomentum(layers, heads, dk, dv)
    module.load_state_dict(d, strict=True)
    layout = SimpleNamespace(layers=[SimpleNamespace(heads=heads, dk=dk, dv=dv,
                                                    matrix_size=heads*dk*dv)]*layers)
    dt = snapshot['engine']['bfss_dt']
    t0 = time.perf_counter()
    with torch.no_grad():
        proposed_s, proposed_p = module.quiet(layout, recurrent, momenta, dt=dt, capture=False)
    proposal_seconds = time.perf_counter()-t0
    t0 = time.perf_counter()
    with torch.no_grad():
        committed_s, committed_p = module.quiet(layout, recurrent, momenta, dt=dt, capture=True)
    capture_seconds = time.perf_counter()-t0
    state_error = float((proposed_s-committed_s).abs().max())
    momentum_error = float((torch.stack(proposed_p)-torch.stack(committed_p)).abs().max())
    assert state_error < 2e-4 and momentum_error < 2e-4
    # Bind A E + B for every theta coordinate on the complete 32-layer state.
    # One substep is sufficient for this local jet; the production integrator
    # composes the same law across its declared subdivisions.
    # The stored theta eligibility was just consumed by learning. Use the
    # actual first quiet step's eligibility as the history of a second quiet
    # step, so both A E and B are nontrivial in the check below.
    s = committed_s[..., :heads*dk*dv].reshape(layers, 1, heads, dk, dv)
    p = torch.stack(committed_p)
    prior_sensitivity_s = module.sensitivity_s.clone()
    prior_sensitivity_p = module.sensitivity_p.clone()
    theta = d['theta']
    context = module.context.detach()
    sub_dt = dt/int(module.last_solver_substeps)
    fn = lambda xs, xp, xt: quiet(xs, xp, xt, context, sub_dt,
        module.frequency_scale, module.impedance if module.spectrum_enabled else None)
    zeros_s, zeros_p, zeros_theta = torch.zeros_like(s), torch.zeros_like(p), torch.zeros_like(theta)
    relative, maximum, transported_norms, full_norms = [], [], [], []
    for i in range(theta.shape[-1]):
        ds, dp = prior_sensitivity_s[i], prior_sensitivity_p[i]
        direction = torch.zeros_like(theta); direction[:, i] = 1
        with torch.no_grad():
            _, together = torch.func.jvp(fn, (s, p, theta), (ds, dp, direction))
            _, transported = torch.func.jvp(fn, (s, p, theta), (ds, dp, zeros_theta))
            _, inserted = torch.func.jvp(fn, (s, p, theta), (zeros_s, zeros_p, direction))
        error = torch.cat([(a-b-c).flatten() for a, b, c in zip(together, transported, inserted)])
        target = torch.cat([a.flatten() for a in together])
        maximum.append(float(error.abs().max()))
        relative.append(float(error.norm()/target.norm().clamp_min(1e-12)))
        transported_norms.append(float(torch.cat([a.flatten() for a in transported]).norm()))
        full_norms.append(float(target.norm()))
    assert max(relative) < 2e-5
    assert max(transported_norms) > 0
    return dict(layers=layers, heads=heads, theta_directions=len(relative),
        contact_directions=d['contact.sensitivity_s'].shape[0],
        context_was_valid=bool(d['context_valid']), subdivisions=int(module.last_solver_substeps),
        proposal_commit_state_max_abs_error=state_error,
        proposal_commit_momentum_max_abs_error=momentum_error,
        jet_transport_max_relative_error=max(relative), jet_transport_max_abs_error=max(maximum),
        jet_source='second quiet step after actual first-step eligibility capture',
        carried_theta_eligibility_norm=float(prior_sensitivity_s.norm()+prior_sensitivity_p.norm()),
        erase_history_gradient_difference_l2=float(torch.tensor(transported_norms).norm()),
        full_next_gradient_l2=float(torch.tensor(full_norms).norm()),
        captured_theta_sensitivity_norm=float(module.sensitivity_s.norm()+module.sensitivity_p.norm()),
        captured_contact_sensitivity_norm=float(module.contact.sensitivity_s.norm()+module.contact.sensitivity_p.norm()),
        cpu_proposal_seconds=proposal_seconds, cpu_capture_seconds=capture_seconds,
        timing_scope='one CPU contract check, not a throughput benchmark')


def stopping_binding(path):
    """First-release prefixes along already recorded, unchanged quiet paths."""
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    prefix, groups, skipped = [], [], 0
    for row in rows:
        if row['kind'] not in ('internal', 'release'):
            continue
        prefix.append(row)
        if row['kind'] != 'release':
            continue
        if row['choices'] != len(prefix):
            skipped += 1; prefix = []; continue
        stay = [r['admission']['probability'][1] /
                sum(r['admission']['probability']) for r in prefix]
        mass, tangent, release_mass, release_tangent = 1., 0., 0., 0.
        for c in stay:
            dc = c*(1-c)
            release_mass += mass*(1-c)
            release_tangent += tangent*(1-c)-mass*dc
            tangent = tangent*c+mass*dc
            mass *= c
        logp = sum(math.log(c) for c in stay[:-1])+math.log1p(-stay[-1])
        groups.append(dict(length=len(stay), tail=mass, tail_tangent=tangent,
            mass_error=abs(release_mass+mass-1),
            tangent_error=abs(release_tangent+tangent),
            path_logp_error=abs(logp-row['eligible_logp'])))
        prefix = []
    assert groups and max(r['mass_error'] for r in groups) < 1e-12
    assert max(r['tangent_error'] for r in groups) < 1e-12
    assert max(r['path_logp_error'] for r in groups) < 1e-5
    return dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        complete_quiet_prefixes=len(groups), skipped_incomplete_start=skipped,
        max_prefix_length=max(r['length'] for r in groups),
        max_mass_error=max(r['mass_error'] for r in groups),
        max_tangent_error=max(r['tangent_error'] for r in groups),
        max_actual_path_logp_error=max(r['path_logp_error'] for r in groups),
        mean_surviving_mass_at_observed_prefix=sum(r['tail'] for r in groups)/len(groups),
        derivative_direction='shared additive internal-vs-release logit field; fixed recorded quiet path',
        role='read-only binding of existing stopped-path receipt and finite-prefix tail')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--events', type=Path,
                   default=Path('scratch/native_signal_admission_2401_20261117/body/autonomous_events.jsonl'))
    args = p.parse_args()
    torch.set_num_threads(1)
    snapshot = torch.load(args.input, map_location='cpu', weights_only=False)
    features, first = feature_binding(snapshot)
    weight = snapshot['encoder']['encoder.weight'].double()
    singular = torch.linalg.svdvals(weight)
    rank = int(torch.linalg.matrix_rank(weight))
    result = dict(schema=1, cycle=snapshot['cycle'], source=snapshot['source'],
        extracted_input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        encoder_geometry=dict(shape=list(weight.shape), rank=rank,
            nullity=weight.shape[1]-rank, singular_min=float(singular[-1]),
            singular_max=float(singular[0])),
        feature_inversion=features,
        repeated_kernel_identity=repeated_row_binding(first, .1),
        quiet=quiet_binding(snapshot), stopping=stopping_binding(args.events),
        training_updates=0, world_calls=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
