"""Event-boundary adoption and bounded observations of the existing optimizer."""
import re

import torch

from information_boltzmann.budgeted_adamw import BudgetedAdamW, attach_budgeted_adamw


def configure(engine, config, policy, cycle):
    inherited = config.get('optimizer_plasticity', {}).get('kind', 'adamw')
    if policy is None:
        policy = inherited
    if policy not in ('adamw', 'budgeted'):
        raise ValueError('unknown optimizer policy')
    if policy == 'adamw' and isinstance(engine.optimizer, BudgetedAdamW):
        raise ValueError('do not silently discard adopted budget observer history')
    if policy == 'budgeted' and not isinstance(engine.optimizer, BudgetedAdamW):
        attach_budgeted_adamw(engine.optimizer, allocation=True, cooling=True, fused=True)
        config['optimizer_plasticity'] = dict(schema=2, kind='budgeted', adopted_at_cycle=cycle,
            cold_start_rule='recent_signal_upper_v2',
            clocks='per-parameter Adam plus local observer; no global age annealing',
            boundary='after real loss, delayed credit and existing clipping; before state continuation',
            retained='all parameters/groups/lrs/m/v/steps; complete S/P/M, archives, world, pending actor',
            layout=64, allocation=True, cooling=True, fused=True)
    if policy == 'budgeted' and config['optimizer_plasticity'].get('cold_start_rule') != 'recent_signal_upper_v2':
        prior = config['optimizer_plasticity']
        config['optimizer_plasticity'] = dict(prior,schema=2,cold_start_rule='recent_signal_upper_v2',
            rule_recorded_at_cycle=cycle,prior_rule=prior.get('cold_start_rule','unversioned_admission'))
    return policy


def configure_wiener_gate(engine, config, enabled: bool):
    opt = engine.optimizer
    if not isinstance(opt, BudgetedAdamW):
        raise ValueError('wiener_gate requires BudgetedAdamW')
    opt.defaults['wiener_gate'] = bool(enabled)
    for group in opt.param_groups:
        group['wiener_gate'] = bool(enabled)
    policy = dict(config.get('optimizer_plasticity', {}))
    policy['wiener_gate'] = bool(enabled)
    config['optimizer_plasticity'] = policy


@torch.no_grad()
def diagnostics(engine):
    """One compact device read for per-layer real write fractions, not per-byte."""
    opt = engine.optimizer
    if not isinstance(opt, BudgetedAdamW):
        return dict(kind='adamw')
    groups = {}
    roles = {}
    for name, p in engine.model.named_parameters():
        state = opt.state.get(p, {})
        if 'budget_gate' not in state: continue
        match = re.match(r'body\.base\.layers\.(\d+)\.', name)
        key = f'L{int(match[1])+1}' if match else 'auxiliary'
        groups.setdefault(key, []).append((state['budget_gate'].reshape(-1), p.numel()))
        if not match:
            role = ('.'.join(name.split('.')[:2]) if name.startswith(('body.','kanli.'))
                    else name.split('.')[0])
            roles.setdefault(role, []).append((state['budget_gate'].reshape(-1), p.numel()))
    labels, summaries = [], []
    for key, entries in [*groups.items(), *((f'role:{k}',v) for k,v in roles.items())]:
        total = sum(n for _,n in entries)
        # Param-weighted gate: a large projection cannot disappear behind many
        # tiny bias tensors. Exact zero is an observed hold, never a cutoff.
        mean = sum(g.mean()*(n/total) for g,n in entries)
        zero = sum((g == 0).float().mean()*(n/total) for g,n in entries)
        summaries.append(torch.stack((mean, zero)))
        labels.append(key)
    values = torch.stack(summaries).cpu().tolist() if summaries else []
    stats = {key:dict(mean_write_fraction=v[0], held_fraction=v[1]) for key,v in zip(labels, values)}
    return dict(kind='budgeted', layers={k:v for k,v in stats.items() if not k.startswith('role:')},
                mechanisms={k[5:]:v for k,v in stats.items() if k.startswith('role:')},
                extra_state_bytes=sum(v.numel()*v.element_size() for s in opt.state.values()
                    for k,v in s.items() if k.startswith('budget_') and torch.is_tensor(v)))
