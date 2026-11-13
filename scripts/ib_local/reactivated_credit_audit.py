"""Read-only attribution of current loss to actually reactivated local episodes.

This audits an existing forward graph; it does not manufacture a teacher or
change the learner. A receipt is valid only for the version actually replayed.
"""
import torch


def paired_reactivated_vjp(loss, wrapper, named_parameters, *, min_age=128):
    """Matched and cyclically mismatched teachers on the same old responses.

    The forward graph, response tensors, current loss and parameter values are
    identical. Null/recent paths are excluded; only complete strictly-old
    episodes with a current teacher are rotated within the same layer.
    """
    if min_age < 0:
        raise ValueError('min_age must be nonnegative')
    names, params = zip(*named_parameters) if named_parameters else ((), ())
    memory = wrapper.memory
    selected = []
    for slot, response in wrapper.replayed_by_slot.items():
        capsule = wrapper.capsules[slot]
        if not bool(memory.occupied[slot]) or capsule is None or capsule.version != int(memory.read_credit_version[slot]):
            raise RuntimeError('replayed episode identity changed')
        length = (capsule.complete_inputs.shape[1] if wrapper.reactivation_mode == 'complete_attention'
                  else capsule.inputs.shape[1])
        end = int(memory.birth[slot]) - (wrapper.replay_tokens - 1) + length - 1
        if int(memory.clock)-end > min_age and response.requires_grad:
            selected.append((int(slot), response))
    if not selected or not params:
        return dict(identities=[], matched={}, mismatched={}, reason='no eligible connected records or parameters')
    outputs = [x[1] for x in selected]
    teachers = torch.autograd.grad(loss, outputs, retain_graph=True, allow_unused=True)
    actual = [(slot, response, teacher.detach()) for (slot, response),teacher in zip(selected,teachers)
              if teacher is not None]
    if len(actual) < 2:
        return dict(identities=[x[0] for x in actual], matched={}, mismatched={}, reason='fewer than two actual teachers')
    responses = [x[1] for x in actual]
    factors = [x[2] for x in actual]
    groups = {}
    for index, response in enumerate(responses):
        groups.setdefault(tuple(response.shape), []).append(index)
    shuffled = factors.copy()
    rotated = 0
    for indices in groups.values():
        if len(indices) > 1:
            for dest, src in zip(indices, indices[-1:]+indices[:-1]):
                shuffled[dest] = factors[src]
            rotated += len(indices)
    if not rotated:
        return dict(identities=[x[0] for x in actual], matched={}, mismatched={},
                    reason='no two equal-shaped actual responses to mismatch')
    def contract(teachers):
        values = torch.autograd.grad(responses, params, grad_outputs=teachers,
                                     retain_graph=True, allow_unused=True)
        return {name: (torch.zeros_like(param) if value is None else value.detach())
                for name,param,value in zip(names,params,values)}
    return dict(identities=[x[0] for x in actual],
                teacher_norms={str(slot):float(teacher.float().norm()) for slot,_,teacher in actual},
                matched=contract(factors),
                mismatched=contract(shuffled), rotated=rotated, reason=None)


def audit_reactivated_credit(loss, wrapper, named_parameters, *, min_age=128):
    """Return local current-loss VJPs grouped by selected historical identity.

    Age is conservative: the *last* input in the reexecuted segment must be
    older than the requested boundary at the earliest query in this window.
    VJPs are local reexecution credit, not gradients through elapsed history.
    """
    if min_age < 0:
        raise ValueError('min_age must be nonnegative')
    names, params = zip(*named_parameters) if named_parameters else ((), ())
    memory = wrapper.memory
    records = []
    for slot, response in wrapper.replayed_by_slot.items():
        if not bool(memory.occupied[slot]):
            raise RuntimeError('replayed slot is no longer occupied')
        capsule = wrapper.capsules[slot]
        version = int(memory.read_credit_version[slot])
        if capsule is None or capsule.version != version:
            raise RuntimeError('replayed episode identity changed')
        length = (capsule.complete_inputs.shape[1] if wrapper.reactivation_mode == 'complete_attention'
                  else capsule.inputs.shape[1])
        end = int(memory.birth[slot]) - (wrapper.replay_tokens - 1) + length - 1
        age = int(memory.clock) - end
        if age <= min_age:
            continue
        item = dict(slot=int(slot), version=version, end_age=age, input_positions=int(length),
                    connected=bool(response.requires_grad), teaching_norm=None, gradients={})
        if response.requires_grad and loss.requires_grad:
            teaching, = torch.autograd.grad(loss, response, retain_graph=True, allow_unused=True)
            if teaching is not None:
                item['teaching_norm'] = float(teaching.detach().float().norm())
                gradients = torch.autograd.grad(response, params, grad_outputs=teaching.detach(),
                                                retain_graph=True, allow_unused=True) if params else ()
                item['gradients'] = {name: None if grad is None else float(grad.detach().float().norm())
                                     for name, grad in zip(names, gradients)}
        records.append(item)
    return records
