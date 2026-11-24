"""A bounded, interruptible native action followed by contiguous reading.

One native actor owns the whole option. The first 960B retains the inherited
prediction targets; subsequent 960B blocks train the same byte objective and
retention machinery before the original action is settled once. No new actor
choice or imagined world observation is created for a continuation block.
"""
from __future__ import annotations

import math


def enabled(engine):
    return getattr(engine, 'bounded_reading_option_schema', None) == 1


def configure(engine, config=None, *, max_blocks=4, surprise_excess=0.5, hippocampal_anchor_scale=0.0):
    if max_blocks not in (1, 2, 4, 8) or not math.isfinite(surprise_excess) or surprise_excess < 0:
        raise ValueError('bounded reading needs K in 1/2/4/8 and finite nonnegative surprise excess')
    if not math.isfinite(hippocampal_anchor_scale) or hippocampal_anchor_scale < 0:
        raise ValueError('hippocampal_anchor_scale must be finite and nonnegative')
    if getattr(engine, 'phase_coupled_rhythm_schema', None) == 1:
        raise ValueError('select one scheduling policy per body')
    if type(engine.world).__name__ != 'NaturalTextWorld':
        raise TypeError('this option is bound to the actual natural-text world')
    if engine.world.navigation_policy.startswith('inquire_'):
        raise ValueError('header-bearing inquiry worlds require their own byte/port binding')
    if engine.response_bytes != 480 or engine.holdout_bytes != 480:
        raise ValueError('this implementation preserves the existing 480+480 micro-block')
    engine.bounded_reading_option_schema = 1
    engine.reading_option_spec = dict(schema=1, max_blocks=max_blocks,
                                      surprise_excess=float(surprise_excess))
    if hippocampal_anchor_scale > 0:
        engine.reading_option_spec['hippocampal_anchor_scale'] = float(hippocampal_anchor_scale)
    if not hasattr(engine, 'reading_option_state'):
        engine.reading_option_state = dict(options=0, blocks=0, surprise_ema=float(engine.intero_raw[0]),
                                          terminations={}, last=None)
    from scripts.ib_local import run_navigation_residual_wood as legacy
    legacy._DERIVED.add('_reading_option_session')
    if config is not None:
        config['bounded_reading_option'] = dict(engine.reading_option_spec)


def begin(engine):
    """Freeze the actual selected question's identity, after native planning."""
    if not enabled(engine):
        return None
    roles = getattr(engine, '_program_question_roles', {})
    qid = roles.get('unknown_identity')
    from scripts.ib_local import shared_unknown_bindings as shared
    archive = engine.model._causal_archive
    row = None if qid is None else shared._row(archive, qid)
    if row is None:
        raise RuntimeError('native shared question identity must resolve before extending an action')
    session = dict(blocks=0, block_nll=[], world_blocks=[],
                   unknown_id=qid, unknown_version=None if row is None else int(row['version']),
                   baseline=float(engine.reading_option_state['surprise_ema']), stop=None)
    engine._reading_option_session = session
    return session


def stop_reason(*, blocks, max_blocks, nll, baseline, excess, target_valid, source_remaining):
    if not math.isfinite(nll) or nll > baseline + excess:
        return 'surprise'
    if not target_valid:
        return 'target_changed'
    if source_remaining < 960:
        return 'source_boundary'
    if blocks >= max_blocks:
        return 'budget'
    return None


def completed_block(engine, session, nll, info):
    """Called AFTER both real _advance operations of the current block."""
    if session is None:
        return False
    session['blocks'] += 1
    session['block_nll'].append(float(nll))
    session['world_blocks'].append(dict(start=int(info['start']),
        source=info['source'], provider_calls=int(info.get('provider_calls', 0)),
        response_previously_seen_bytes=int(info.get('response_previously_seen_bytes', 0)),
        holdout_previously_seen_bytes=int(info.get('holdout_previously_seen_bytes', 0))))
    from scripts.ib_local import shared_unknown_bindings as shared
    qid = session['unknown_id']
    row = None if qid is None else shared._row(engine.model._causal_archive, qid)
    valid = qid is None or (row is not None and int(row['version']) == session['unknown_version'])
    base = engine.world.base
    remaining = len(base.data)-base.cursor
    # NaturalTextWorld wraps when a complete topic window no longer fits.
    remaining = remaining if remaining >= base.window else 0
    session['stop'] = stop_reason(blocks=session['blocks'], max_blocks=engine.reading_option_spec['max_blocks'],
        nll=float(nll), baseline=session['baseline'], excess=engine.reading_option_spec['surprise_excess'],
        target_valid=valid, source_remaining=remaining)
    return session['stop'] is None


def finish(engine, session):
    if session is None:
        return None
    if getattr(engine, '_reading_option_session', None) is not session:
        raise RuntimeError('reading option already settled or has a different owner')
    state = engine.reading_option_state
    state['options'] += 1
    state['blocks'] += session['blocks']
    for nll in session['block_nll']:
        state['surprise_ema'] = 0.9*state['surprise_ema'] + 0.1*nll
    reason = session['stop']
    state['terminations'][reason] = state['terminations'].get(reason, 0)+1
    receipt = dict(session, max_blocks=engine.reading_option_spec['max_blocks'],
        external_bytes=960*session['blocks'], language_updates=2*session['blocks'],
        native_plannings=1, action_settlements=1, proposals=engine.joint_proposal_budget,
        terminal_gain_discount=engine.gamma**(session['blocks']-1),
        bootstrap_discount=engine.gamma**session['blocks'])
    state['last'] = receipt
    engine._reading_option_session = None
    return receipt
