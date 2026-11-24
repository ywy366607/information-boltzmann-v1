"""Read-only production-clock audit of the proposed Batch 3 diagnoses."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import torch

from scripts.ib_local.compressed_checkpoint import checkpoint_stream


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = [json.loads(x) for x in (args.run/'events.jsonl').read_text(encoding='utf8').splitlines()]
    external = [r for r in rows if r.get('autonomous_event', {}).get('kind') == 'external']
    result = dict(run=str(args.run), cycles=len(external), optimizer_updates=0, world_calls=0)
    groups = {}
    for shift in (0, 1):
        values = {}
        for i in range(1+shift, len(external)):
            j = i-shift
            advance = external[j]['world']['anchor']-external[j-1]['world']['anchor']
            key = 'sequential_960_bytes' if advance == 960 else 'other_address'
            values.setdefault(key, []).append(external[i]['past_present']['retained'])
        groups['current_world_grouping' if shift == 0 else 'prior_world_grouping'] = {
            k:dict(count=len(v), retained=v, zero_count=v.count(0), retained_fraction=sum(v)/(8*len(v)))
            for k, v in values.items()}
    result['workset'] = groups
    result['scales'] = {}
    for key in ('relation_ridge', 'memory_temperature'):
        values = [r['learnable_precision_scales'][key] for r in external]
        changes = [math.log(b/a) for a, b in zip(values, values[1:])]
        result['scales'][key] = dict(first=values[0], final=values[-1],
            positive=sum(x>0 for x in changes), negative=sum(x<0 for x in changes),
            zero=sum(x==0 for x in changes), max_log_change_per_external=max(map(abs, changes)),
            final_gate=external[-1]['learnable_precision_scales'][key+'_gate'])
    result['last_consumed_steps_histogram'] = dict(Counter(
        r['adaptive_dynamics']['last_consumed_steps'] for r in external))
    checkpoint = args.run/'body/last.pt.gz'
    before = sha(checkpoint)
    with checkpoint_stream(checkpoint) as stream:
        state = torch.load(stream, map_location='cpu', weights_only=False)
    sd = state['model']
    result['native_body'] = state['config']['native_body']
    prefix = next(k[:-len('theta_freq')] for k in sd if k.endswith('theta_freq'))
    theta = torch.cat([sd[prefix+k] for k in ('theta_freq','theta_context','theta_coupling','theta_pair')], -1)
    from scripts.ib_local.adaptive_hamiltonian import coefficients, force_analytic, quiet
    spec = state['config']['native_body']['backbone']
    heads, d = spec['heads'], spec['d']
    dk = dv = d//heads
    packed = state['engine']['recurrent']
    s = packed[..., :heads*dk*dv].reshape(len(packed), -1, heads, dk, dv)
    p = torch.stack(state['engine']['bfss_momenta']).reshape_as(s)
    context, scale, impedance = [sd[prefix+k] for k in ('context', 'frequency_scale', 'impedance')]
    omega, strength = coefficients(theta, context, heads, scale)
    force = force_analytic(s, theta, heads)
    linear = omega*impedance*s
    norm = lambda x: x.flatten(1).norm(dim=-1)
    ratio = norm(force)/norm(linear).clamp_min(1e-30)
    dt = .01
    actual = quiet(s, p, theta, context, dt, scale, impedance)
    isolated = theta.clone()
    isolated[:, heads+1] = -torch.inf
    removed = quiet(s, p, isolated, context, dt, scale, impedance)
    # A same-state, one-step direct numerical port effect, with no commit.
    result['coupling_at_saved_boundary'] = dict(
        coupling_min=float(strength.min()), coupling_max=float(strength.max()),
        state_rms=float(s.square().mean().sqrt()), momentum_rms=float(p.square().mean().sqrt()),
        force_to_oscillator_force_per_layer=ratio.tolist(),
        force_to_oscillator_force_median=float(ratio.median()),
        one_step_coupling_removed_state_relative_max=float((norm(actual[0]-removed[0])/norm(actual[0]).clamp_min(1e-30)).max()),
        one_step_coupling_removed_momentum_relative_max=float((norm(actual[1]-removed[1])/norm(actual[1]).clamp_min(1e-30)).max()),
        dt=dt, scope='fixed saved S/P and context; original quiet step, zero committed steps')
    result['optimizer_groups'] = [dict(
        index=i, parameter_ids=g['params'], lr=g['lr'], weight_decay=g['weight_decay'],
        betas=g['betas']) for i,g in enumerate(state['optimizer']['param_groups'])]
    # Verify scalar IDs by their explicit terminal two-parameter group and
    # exact log-scale initializer/trajectory, rather than len(opt.state)-N.
    ids = state['optimizer']['param_groups'][-1]['params']
    result['terminal_scale_group'] = []
    for index in ids:
        st = state['optimizer']['state'][index]
        result['terminal_scale_group'].append(dict(id=index, shape=list(st['exp_avg'].shape),
            step=int(st['step']), budget_count=st.get('budget_count'),
            gate=float(st['budget_gate']), m=float(st['exp_avg']), v=float(st['exp_avg_sq'])))
    # Exact Adam formula counterexample to the claimed universal |delta|<=lr.
    b1, b2, t = .9, .999, 10001
    direction = ((1-b1)/(1-b1**t))/math.sqrt((1-b2)/(1-b2**t))
    result['adam_step_bound_counterexample'] = dict(
        previous_gradients='all zero', next_gradient=1, step=t,
        bias_corrected_direction=direction, proposed_step_over_lr=direction)
    assert sha(checkpoint) == before
    result.update(source_sha256=before, source_unchanged=True)
    args.report.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('optimizer_groups','native_body')}, indent=2))


if __name__ == '__main__':
    main()
