"""Reuse signed read preference for bounded retrieval and retention.

This is a candidate consolidation rule, not a proved attractor. The existing
bounded preference enters both retrieval logits and the lifetime multiplier
exp(b). The original rule uses optimizer displacement tanh(u); evidence mode
uses tanh(bias-corrected signed first moment / gradient RMS), reusing existing
statistics. Neither is a calibrated probability of absolute usefulness.
The retention rule uses the same moments only for lifetime competition; it
keeps the existing content selection and learned read-logit update unchanged.
The explicit utility candidate instead measures finite removal benefit. Its
three bounded per-slot buffers separate retention evidence from dose gradients;
no reward currency or historical graph is introduced.
Age is measured since writing, never refreshed merely by reading. Protection
is bounded in [exp(-1), exp(1)], so popularity cannot grant permanent tenure.
"""
from __future__ import annotations

import torch


def consolidates_reads(memory):
    """Legacy joint candidates alter retrieval; retention-only does not."""
    return memory.consolidation_enabled and memory.consolidation_rule not in ('retention', 'utility')


def bounded_preference(memory):
    if getattr(memory, 'consolidation_rule', '') == 'utility':
        return utility_preference(memory)
    if not memory.read_credit_enabled:
        return torch.zeros_like(memory.read_credit_mean)
    if (getattr(memory, 'consolidation_rule', 'preference') in ('evidence', 'retention')
            and memory.read_credit_rule == 'adam'):
        # Reuse the signed credit estimator, not optimizer displacement. A
        # record may get very few outcomes before replacement. Its utility
        # must not require ~1/lr outcomes merely to influence retention.
        steps = memory.read_credit_steps
        b1,b2 = memory.read_credit_betas
        active = steps > 0
        first = memory.read_credit_first / (1-b1**steps).clamp_min(memory.read_credit_eps)
        second = memory.read_credit_second / (1-b2**steps).clamp_min(memory.read_credit_eps)
        evidence = first / (second.sqrt()+memory.read_credit_eps)
        return torch.where(active,evidence.tanh(),torch.zeros_like(evidence)).detach()
    return memory.read_credit_mean.detach().tanh()


def utility_preference(memory):
    """Retention evidence is finite usefulness, distinct from read-dose credit."""
    steps = memory.retention_utility_steps
    b1, b2 = memory.read_credit_betas
    first = memory.retention_utility_first/(1-b1**steps).clamp_min(memory.read_credit_eps)
    second = memory.retention_utility_second/(1-b2**steps).clamp_min(memory.read_credit_eps)
    ratio = first/(second.sqrt()+memory.read_credit_eps)
    return torch.where(steps > 0, ratio.tanh(), torch.zeros_like(ratio)).detach()


@torch.no_grad()
def settle_retention_utility(memory, observations):
    """Settle observed L(without record)-L(with record) before overwriting.

Reuse actual read-optimizer EMA timescales. Three fixed-size buffers keep this
question separate from the existing gradient estimate controlling read dose.
"""
    b1,b2 = memory.read_credit_betas
    accepted, stale = 0, 0
    for item in observations:
        slot, version, value = item['slot'], item['version'], item['benefit']
        if not bool(memory.occupied[slot]) or int(memory.read_credit_version[slot]) != version:
            stale += 1
            continue
        value = torch.as_tensor(value, dtype=memory.retention_utility_first.dtype,
                                device=memory.retention_utility_first.device)
        if not bool(torch.isfinite(value)):
            raise FloatingPointError('nonfinite observed retention utility')
        if memory.retention_utility_steps[slot] == torch.iinfo(torch.long).max:
            raise OverflowError('retention utility count exhausted')
        first = memory.retention_utility_first[slot].mul(b1).add(value, alpha=1-b1)
        second = memory.retention_utility_second[slot].mul(b2).add(value.square(), alpha=1-b2)
        if not bool(torch.isfinite(first) & torch.isfinite(second)):
            raise FloatingPointError('nonfinite retention utility moments')
        memory.retention_utility_first[slot].copy_(first)
        memory.retention_utility_second[slot].copy_(second)
        memory.retention_utility_steps[slot].add_(1)
        accepted += 1
    return dict(records=accepted, stale=stale)


def select_read_slots(memory, scores):
    """Select using the same logits subsequently used by the live read.

Return original similarity scores: read_logits adds preference exactly once
and creates the current participation receipt. The top-k boundary is discrete.
"""
    preference = bounded_preference(memory)
    if not bool(preference.any()):
        return scores.topk(memory.topk, dim=-1)
    rank = (scores / memory.temperature + preference).masked_fill(
        ~memory.occupied, -torch.inf)
    slots = rank.topk(memory.topk, dim=-1).indices
    return scores.gather(-1, slots), slots


def replacement_slot(memory):
    """Evict greatest age/exp(b), preferring empty slots before any eviction.

All slots compete for the same fixed capacity. Opposing current evidence can
reduce b through the existing signed credit update. Elapsed events increase
age even when a record is repeatedly read; a saturated bias remains bounded.
"""
    empty = (~memory.occupied).nonzero(as_tuple=False).flatten()
    if empty.numel():
        return int(empty[0])
    age = (memory.clock - memory.last_write).to(torch.float64)
    if bool((age < 0).any()):
        raise ValueError('record write cannot lie after current event clock')
    effective_age = age * (-bounded_preference(memory).double()).exp()
    return int(effective_age.argmax())
