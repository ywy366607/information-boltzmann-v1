"""Present prediction errors assign credit to actually retrieved old records.

A bounded, ephemeral additive attention-logit probe records participation.
After backward its derivative is the signed first-order effect on the current
loss. The table retains read preference and bounded statistics, not a historical
graph, TD return, or rewritten observation. The mean rule is retained as a scale
diagnostic; the Adam rule learns preference using the body's optimizer settings.
This is read-time credit, not original-write credit.
"""
from dataclasses import dataclass

import torch


@dataclass
class ReadParticipation:
    bias: torch.Tensor
    slots: torch.Tensor
    versions: torch.Tensor
    valid: torch.Tensor


def configure_read_credit(memory, rule, optimizer_group=None):
    """Use the existing optimizer's settings; preserve stored evidence/contents."""
    if rule not in ('off', 'mean', 'adam'):
        raise ValueError(rule)
    memory.read_credit_enabled = rule != 'off'
    if rule != 'off':
        memory.read_credit_rule = rule
    if rule == 'adam':
        if optimizer_group is None:
            raise ValueError('read preference needs the actual optimizer settings')
        memory.read_credit_lr = float(optimizer_group['lr'])
        memory.read_credit_betas = tuple(optimizer_group['betas'])
        memory.read_credit_eps = float(optimizer_group['eps'])


def read_logits(memory, selected, slots, valid):
    logits = selected / memory.temperature
    if not memory.read_credit_enabled:
        return logits
    from scripts.ib_local.experience_consolidation import consolidates_reads
    if consolidates_reads(memory):
        from scripts.ib_local.experience_consolidation import bounded_preference
        bias = bounded_preference(memory)[slots].clone()
    else:
        bias = memory.read_credit_mean[slots].tanh().detach().clone()
    if torch.is_grad_enabled() and memory._read_credit_active:
        # Three bounded read batches cover prediction, successor learning and
        # causal choice replay. Misuse cannot accumulate an unbounded history.
        limit = getattr(memory, '_read_credit_limit',
                        3 * max(memory.capacity, memory.long) * memory.topk)
        if memory._read_credit_entries + slots.numel() > limit:
            raise RuntimeError('finish the bounded read-credit batch before more reads')
        bias.requires_grad_()
        memory._read_credit_entries += slots.numel()
        memory._read_participation.append(ReadParticipation(
            bias, slots.detach().clone(), memory.read_credit_version[slots].clone(),
            valid.detach().clone()))
    return logits + bias


@torch.no_grad()
def settle_read_credit(memory):
    """Settle current backward receipts once; reject replaced/merged identities."""
    receipts, memory._read_participation = memory._read_participation, []
    memory._read_credit_active = False
    memory._read_credit_entries = 0
    total = torch.zeros_like(memory.read_credit_mean)
    count = torch.zeros_like(memory.read_credit_count)
    stale = 0
    for receipt in receipts:
        if receipt.bias.grad is None:
            continue  # This read did not participate in this backward's loss.
        same = memory.read_credit_version[receipt.slots] == receipt.versions
        valid = receipt.valid & same & memory.occupied[receipt.slots]
        stale += int((receipt.valid & ~same).sum())
        credit = -receipt.bias.grad.detach()
        if not torch.isfinite(credit).all():
            raise FloatingPointError('nonfinite retrieved-record credit')
        index = receipt.slots[valid]
        if index.numel():
            # Repeated slot IDs need a fixed reduction order on CUDA as well
            # as CPU; floating-point atomic scatter would break exact resume.
            order = index.argsort(stable=True)
            unique, lengths = index[order].unique_consecutive(return_counts=True)
            reduced = torch.segment_reduce(credit[valid][order], 'sum', lengths=lengths)
            total[unique] += reduced
            count[unique] += lengths
    touched = count > 0
    new_count = memory.read_credit_count[touched] + count[touched]
    if bool((new_count <= memory.read_credit_count[touched]).any()):
        raise OverflowError('retrieved-record credit count exhausted')
    old_bias = memory.read_credit_mean[touched].tanh()
    old_applied_bias = old_bias
    from scripts.ib_local.experience_consolidation import consolidates_reads
    if consolidates_reads(memory):
        from scripts.ib_local.experience_consolidation import bounded_preference
        old_applied_bias = bounded_preference(memory)[touched]
    if memory.read_credit_rule == 'adam':
        # total is the negative derivative w.r.t. the additive logit probe.
        # Chain through tanh to train the underlying persistent preference.
        evidence_rule = (memory.consolidation_enabled and memory.consolidation_rule == 'evidence')
        # Evidence mode estimates the signed effect on the additive read
        # logit itself. It is not a derivative through optimizer moments.
        gradient = total[touched] if evidence_rule else total[touched] * (1-old_bias.square())
        b1, b2 = memory.read_credit_betas
        first = b1*memory.read_credit_first[touched] + (1-b1)*gradient
        second = b2*memory.read_credit_second[touched] + (1-b2)*gradient.square()
        steps = memory.read_credit_steps[touched] + 1
        corrected_first = first/(1-b1**steps)
        corrected_second = second/(1-b2**steps)
        mean = (memory.read_credit_mean[touched] + memory.read_credit_lr*corrected_first /
                (corrected_second.sqrt()+memory.read_credit_eps))
        if not (torch.isfinite(first).all() and torch.isfinite(second).all()
                and bool((steps > memory.read_credit_steps[touched]).all())):
            raise FloatingPointError('invalid retrieved-record Adam moments')
    else:
        mean = ((memory.read_credit_mean[touched] * memory.read_credit_count[touched]
                 + total[touched]) / new_count)
    if not torch.isfinite(mean).all():
        raise FloatingPointError('nonfinite retrieved-record credit mean')
    memory.read_credit_mean[touched] = mean
    memory.read_credit_count[touched] = new_count
    if memory.read_credit_rule == 'adam':
        memory.read_credit_first[touched] = first
        memory.read_credit_second[touched] = second
        memory.read_credit_steps[touched] = steps
    result = dict(records=int(touched.sum()), participations=int(count.sum()),
                  stale=stale, credit_sum=float(total.sum()))
    if getattr(memory, 'read_credit_diagnostics', False) and bool(touched.any()):
        age = memory.clock-memory.last_write[touched]
        applied = mean.tanh()
        if consolidates_reads(memory):
            applied = bounded_preference(memory)[touched]
        result.update(credit_abs_max=float(total[touched].abs().max()),
            per_read_credit_abs_max=float((total[touched]/count[touched]).abs().max()),
            preference_abs_max=float(mean.abs().max()),
            attention_logit_update_max=float((applied-old_applied_bias).abs().max()),
            records_over_128_bytes=int((age > 128).sum()),
            records_over_512_bytes=int((age > 512).sum()),
            max_age_bytes=int(age.max()))
    return result
