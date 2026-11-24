"""Actual internal-preview ports and learned dynamics feedback, private arms.

The live S/P evolution, world and full optimizer remain executing. The two
explicit switches act on future-state reads at the existing internal selector,
and on committing the existing 32x12 dynamics parameter proposal.
"""
import functools

import torch


def install(observer, *, hold_preview=False, hold_feedback=False):
    e, m = observer.engine, observer.module
    from scripts.ib_local import navigation_generative_cycle as module
    original_choice = e._sample_internal_choice
    original_source = e._read_source
    original_read = module.decision_momentum_read
    active = None
    receipts = []

    @functools.wraps(original_source)
    def source(*args, **kwargs):
        value = original_source(*args, **kwargs)
        if active is None:
            return value
        active['sources'].append(value.detach().clone())
        assert len(active['sources']) <= 2
        return active['sources'][0] if hold_preview and len(active['sources']) == 2 else value

    @functools.wraps(original_read)
    def read(*args, **kwargs):
        value = original_read(*args, **kwargs)
        if active is None:
            return value
        assert torch.is_tensor(value) and value.shape == (1, 8192)
        active['reads'].append(value.detach().clone())
        assert len(active['reads']) <= 2
        return active['reads'][0] if hold_preview and len(active['reads']) == 2 else value

    @functools.wraps(original_choice)
    def choice(*args, **kwargs):
        nonlocal active
        assert active is None
        active = dict(sources=[], reads=[])
        try:
            result = original_choice(*args, **kwargs)
            assert len(active['sources']) == len(active['reads']) == 2
            # These arrays are actual reads. The second source may change the
            # query used for the second P read in the hold-preview arm.
            observer.record('inner_loop_preview_ports',dict(
                hold_preview=hold_preview, source_now=active['sources'][0],
                source_after_quiet=active['sources'][1],
                momentum_now=active['reads'][0],momentum_after_quiet=active['reads'][1],
                selector_features=result[1]['features'],selector_prior=result[1]['prior'],
                admission=result[2]))
            receipts.append(dict(event=observer.clock,kind='preview',held=hold_preview,
                                 admission=result[2]))
            return result
        finally:
            active = None

    def commit(module, before):
        proposal = module.theta.detach().clone()
        if hold_feedback:
            with torch.no_grad():
                module.theta.copy_(before)
        delta = proposal-before
        observer.record('inner_loop_feedback_commit',dict(hold_feedback=hold_feedback,
            proposed_delta=delta,committed_delta=module.theta.detach()-before,
            semantics='same optimizer proposal and moments; select commit of 32x12 dynamics coordinates'))
        receipts.append(dict(event=observer.clock,kind='feedback',held=hold_feedback,
            proposed_norm=float(delta.norm()),committed_norm=float((module.theta.detach()-before).norm())))

    e._read_source = source
    e._sample_internal_choice = choice
    module.decision_momentum_read = read
    observer.after_optimizer_proposal = commit

    def undo():
        e._read_source = original_source
        e._sample_internal_choice = original_choice
        module.decision_momentum_read = original_read
        del observer.after_optimizer_proposal
    return receipts, undo
