"""Mass-preserving finite-population control variate for native previews.

The existing learned continuation is evaluated cheaply at every posterior
point. Original native decision programs evaluate only the sampled residuals.
Strata minimize the proxy's within-group dispersion at the inherited budget;
sampling within each stratum retains uniform finite-grid expectation for ANY
proxy. A private, fixed generator supplies common draws without advancing the
individual's CPU/CUDA/world RNG. Configured seed and mode survive restore.
"""
import torch


def partition(proxy, budget):
    """Greedy binary splits by largest exact reduction in proxy SSE.

Sorted sample indices are stable identities for ties. Flat groups split by
their original index, retaining coverage and the same finite cost budget.
"""
    if proxy.ndim != 1 or not len(proxy) or budget < 1 or not bool(torch.isfinite(proxy).all()):
        raise ValueError('finite one-dimensional proxy and positive budget required')
    order = torch.argsort(proxy, stable=True)
    groups = [order]
    for _ in range(min(int(budget), len(proxy))-1):
        proposals = []
        for i, indices in enumerate(groups):
            if len(indices) < 2:
                continue
            # Float64 cumulative sums keep constant proxies exactly flat.
            value = proxy[indices].double()
            centered = value-value.mean()
            prefix = centered.cumsum(0)[:-1]
            n = torch.arange(1, len(indices), device=proxy.device, dtype=torch.float64)
            gain = prefix.square()*(1/n+1/(len(indices)-n))
            maximum, index = gain.max(0)
            cut = int(index)+1 if float(maximum) > 0 else len(indices)//2
            proposals.append((float(maximum), len(indices), -i, i, cut))
        if not proposals:
            break
        _, _, _, index, cut = max(proposals)
        selected = groups.pop(index)
        groups[index:index] = [selected[:cut], selected[cut:]]
    return groups


def plan(proxy, budget, *, seed):
    groups = partition(proxy, budget)
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    # One independent uniform local index for each fixed stratum. The draw is
    # made AFTER its membership and proxy have been fixed.
    chosen = [int(group[int(torch.randint(len(group), (), generator=generator))])
              for group in groups]
    index = torch.tensor(chosen, device=proxy.device, dtype=torch.long)
    weights = proxy.new_tensor([len(group)/len(proxy) for group in groups])
    return index, weights, groups


def readout(proxy, index, weights, native_values):
    # Mean baseline plus unbiased finite-population residual correction.
    return proxy.mean() + (weights*(native_values-proxy[index])).sum(-1)


@torch.no_grad()
def cheap_proxy(engine, points):
    from scripts.ib_local.navigation_internal_rollout import continuation
    valid = engine.reward_label_valid(wood=True)
    return torch.cat([continuation(engine.model, chunk, reward_valid=valid)[0]
                      for chunk in points.split(128)])


@torch.no_grad()
def value(engine, state, source, action, mother, distribution, momentum):
    from scripts.ib_local.joint_successor import standardized
    from scripts.ib_local.prospective_decision import boundary, value as native_value
    from scripts.ib_local import successor_address_geometry as geometry
    if len(source) != 1:
        raise ValueError('one original typed successor branch required')
    config = engine.successor_quadrature
    factor = torch.linalg.cholesky(distribution['scale_matrix'][0])
    white = standardized(float(distribution['df']), distribution['mean'].shape[-1]).to(source)
    points = distribution['mean']+white@factor.T
    raw, _ = boundary(engine, state, source, action, mother, distribution['mean'])
    packet = geometry.snapshot(engine.model, raw, distribution['mean'])
    identities = geometry.addresses(packet, points)['identity']
    baseline = cheap_proxy(engine, points)
    index, weights, groups = plan(baseline, engine.joint_proposal_budget, seed=config['seed'])
    selected = points[index]
    def expand(tensor):
        return tensor.expand(len(selected), *tensor.shape[1:])
    values = native_value(engine, state, expand(source), expand(action), expand(mother),
                          selected, expand(momentum))
    expected = readout(baseline, index, weights, values)
    unique, counts = identities.unique(return_counts=True)
    strata = [dict(stratum=i, count=len(group), sample_index=int(index[i]),
                   representative_cell=int(identities[index[i]]),
                   proxy_mean=float(baseline[group].mean()),
                   proxy_std=float(baseline[group].std(unbiased=False)))
               for i, group in enumerate(groups)]
    engine.successor_distribution_last = dict(schema=2, budget=int(engine.joint_proposal_budget),
        branches=len(selected), weights=weights.tolist(), values=[values.tolist()],
        expected=[float(expected)], df=float(distribution['df']),
        scale_mean=float(distribution['scale2'].mean()),
        baseline_mean=float(baseline.mean()), sampled_baseline=baseline[index].tolist(),
        residuals=(values-baseline[index]).tolist(),
        joint=dict(samples=len(points), strata=strata,
            fresh_mass=float((identities==-1).float().mean()),
            address_mass={str(int(k)): int(n)/len(points) for k, n in zip(unique, counts)},
            seed=int(config['seed']),
            semantics='same posterior grid; learned terminal proxy; stratified native residuals'),
        semantics='all-point learned baseline plus mass-weighted original-native correction')
    return expected.reshape(1)


def configure(engine, config):
    config = dict(config)
    if config.get('schema') != 1 or config.get('mode') != 'residual':
        raise ValueError('explicit residual quadrature schema required')
    if getattr(engine, 'epistemic_value_schema', 0) != 5:
        raise ValueError('residual quadrature comparison retains schema5 successor law')
    if not isinstance(config.get('seed'), int):
        raise ValueError('persistent private integration seed required')
    engine.successor_quadrature = config
