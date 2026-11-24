"""Bind sparse native teachers to actual competing root action endpoints.

Only the current root preview sets this scope. Future labels remain pending
until real settlement. The current actor continues to use its frozen values.
"""
from contextlib import contextmanager
import hashlib

import torch


def enabled(engine):
    return getattr(engine, 'native_amortization', {}).get('teacher_selection') == 'competing_actions'


def action_key(source, action):
    return hashlib.sha256(source.detach().cpu().contiguous().numpy().tobytes()+
                          action.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@contextmanager
def decision_scope(engine, source, actions):
    from scripts.ib_local.root_native_correction import mode
    if not enabled(engine) and mode(engine) is None:
        yield
        return
    from scripts.ib_local.conditional_actor_reference import encoded_view
    view = encoded_view(engine.model, engine.model.encoder(source), engine.intero(),
                        reward_valid=engine.reward_label_valid())
    available = view.available[0].tolist()
    roots = {}
    for option, action in enumerate(actions):
        if available[option]:
            roots.setdefault(action_key(source, action[None]), []).append(option)
    old = getattr(engine, '_competing_action_roots', None)
    had_mass = hasattr(engine, '_root_native_mass')
    old_mass = getattr(engine, '_root_native_mass', None)
    engine._competing_action_roots = roots
    # _imagine averages conditional response calls for each physical root.
    # Accumulate each retained point's inner mass across those calls before
    # contracting the outer mean. Repeated/aliased roots get the same mean.
    engine._root_native_mass = {key: dict(calls=0, weights={}) for key in roots}
    # Earlier binding previews have a different body/question decision scope.
    engine._latent_teacher_queue = {}
    try:
        yield
    finally:
        engine._competing_action_roots = old
        if had_mass:
            engine._root_native_mass = old_mass
        else:
            del engine._root_native_mass


def root_node_mass(engine, root, key, local_mass):
    """Joint root coefficient: sum inner masses / original response calls."""
    if root is None:
        return local_mass, None
    row = engine._root_native_mass[root]
    if row['calls'] < 1 or key not in row['weights']:
        raise RuntimeError('root node lacks its actual response/successor mass')
    return row['weights'][key]/row['calls'], row['calls']


def retain(queue, item, capacity):
    """One best condition per root first; remaining slots retain novelty mass."""
    queue[item['key']] = item
    if len(queue) <= capacity:
        return
    ordered = sorted(queue.values(), key=lambda r: (-r['priority'], r['key']))
    by_root = {}
    for row in ordered:
        if row.get('root_key') is not None:
            by_root.setdefault(row['root_key'], row)
    retained = list(by_root.values())[:capacity]
    used = {r['key'] for r in retained}
    retained += [r for r in ordered if r['key'] not in used][:capacity-len(retained)]
    queue.clear()
    queue.update((r['key'], r) for r in retained)


def select(engine, menu, values):
    """Highest-valued distinct root endpoints each receive one native teacher.

    Within a root, mass*novelty chooses its most informative condition. Stable
    endpoint/content identities, rather than slot order, resolve equal scores.
    """
    roots = getattr(engine, '_competing_action_roots', None)
    if not roots or values is None:
        raise RuntimeError('competing teachers require the actual root action scope and values')
    by_root = {}
    for row in menu:  # original descending priority order
        if row.get('root_key') in roots:
            by_root.setdefault(row['root_key'], row)
    scores = values.detach().reshape(-1).cpu().tolist()
    ranked = sorted(by_root, key=lambda key: (-max(scores[i] for i in roots[key]), key))
    selected = [by_root[key] for key in ranked[:2]]
    if not selected:
        raise RuntimeError('no actual root endpoint survived the native teacher menu')
    return selected, dict(schema=1, available_roots=len(roots), covered_roots=len(by_root),
        selected_roots=ranked[:2], root_options={key: roots[key] for key in ranked},
        root_values={key: max(scores[i] for i in roots[key]) for key in ranked},
        top_margin=(max(scores[i] for i in roots[ranked[0]])-
                    max(scores[i] for i in roots[ranked[1]])) if len(ranked)>1 else None)
