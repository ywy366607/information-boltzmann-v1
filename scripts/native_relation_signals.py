"""Observe existing relation calculations without evaluating extra branches.

Records are local vector signals, not oscillator phases or filled cochains.
Missing relation/context evaluations remain missing. Every tensor is an input
or output of an invocation the body already performs.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
from pathlib import Path
import sys

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def state_payload(state):
    """Version alone does not identify functional imagined revisions."""
    return {name: getattr(state, name) for name in (
        'identity', 'generation', 'version', 'index', 'slots', 'stamps',
        'binding', 'variable_uses',
        'evidence_bound', 'constraints', 'extra', 'program_weights',
        'index', 'slots', 'stamps', 'valid', 'release', 'scope', 'revision_plan')}


def prediction_payload(distribution):
    return {name: distribution[name] for name in (
        'mean', 'scale2', 'df', 'base', 'query', 'x', 'y', 'coefficient',
        'ridge', 'noise', 'leverage', 'bound_evidence') if name in distribution}


def install_relation_signals(observer):
    """Reversible aliases retain original returned objects and autograd graph.

    Call IDs record nesting, not a new physical connection. No extra world,
    optimizer or prediction invocation is used to manufacture a signal.
    """
    from scripts.ib_local import causal_relation_program as cp
    from scripts.ib_local import generative_relations as gr
    from scripts.ib_local import joint_relation_update as jr
    from scripts.ib_local import shared_unknown_bindings as su

    originals = {
        'compose': cp.compose,
        'mechanism': cp.mechanism,
        'posterior': cp.posterior,
        'log_score': cp.log_score,
        'execute_programs': gr.execute_programs,
        'overlap_consensus': jr.overlap_consensus,
        '_bound_evidence_design': cp._bound_evidence_design,
        'relation_variables': jr.relation_variables,
        '_supports': su._supports,
        'revise': cp.revise,
    }
    serial, stack = 0, []

    def make(label, fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            nonlocal serial
            serial += 1
            call_id, parent_id = serial, stack[-1] if stack else None
            stack.append(call_id)
            try:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                a = bound.arguments
                result = fn(*args, **kwargs)
                data = dict(operation=label, call_id=call_id, parent_id=parent_id)
                caller = sys._getframe(1)
                data['caller'] = dict(module=caller.f_globals.get('__name__'),
                                      function=caller.f_code.co_name)
                del caller
                if label in ('compose', 'mechanism', 'execute_programs', 'relation_variables'):
                    data['state'] = state_payload(a['state'])
                if label == 'mechanism':
                    data.update(source=a['source'], action=a['action'],
                        kind=a['kind'], mother=a['mother'], momentum=a['momentum'],
                        ridge=a['ridge'], exclude=a['exclude'], exclude_fact=a['exclude_fact'],
                        prediction=prediction_payload(result))
                elif label == 'log_score':
                    data.update(prediction=prediction_payload(a['distribution']),
                                target=a['target'], score=result)
                elif label == 'posterior':
                    # This precedes mechanism's base addition. Subtracting
                    # two rounded final vectors need not recover this tensor.
                    data.update(design=a['x'], residual=a['y'], query=a['query'],
                        ridge=a['ridge'], noise_prior=a['noise_prior'],
                        prediction=prediction_payload(result))
                elif label == '_bound_evidence_design':
                    data.update(facts=a['facts'], kind=a['kind'], transport=a['transport'],
                        legacy=a['legacy'], design=result[0], residual=result[1],
                        evidence_row_counts=[len(e.source) for e in a['facts']])
                elif label == 'overlap_consensus':
                    data.update(predictions=[prediction_payload(d) for d in a['distributions']],
                        weights=a['weights'], result=result)
                elif label == 'execute_programs':
                    data.update(output_port='decision',
                        output_identity=f"{a['state'].identity}:decision",
                        predictions=[prediction_payload(d) for d in result['program_predictions']],
                        weights=result['program_weights'],
                        coordination=result.get('overlap_coordination'))
                elif label == 'relation_variables':
                    data['loaded_state'] = state_payload(result)
                elif label == '_supports':
                    data.update(unknown_identity=a['identity'], facts=result)
                elif label == 'revise':
                    data.update(state=state_payload(a['state']), evidence=a['evidence'],
                                revised_state=state_payload(result))
                observer.record('relation_signal', data)
                return result
            finally:
                stack.pop()
        return wrapped

    replacements = {name: make(name, fn) for name, fn in originals.items()}
    for module in list(sys.modules.values()):
        if module is None or not getattr(module, '__name__', '').startswith('scripts.ib_local'):
            continue
        for name, original in originals.items():
            if getattr(module, name, None) is original:
                observer._replace(module, name, replacements[name])
