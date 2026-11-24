"""One gain function for C2 training, pre-outcome calibration and continuation.

Labels are absolute 10*finite-U gains. The action baseline is detached and
current; old absolute labels are rebased through this same forward function.
The first three outcome channels retain their original observation training.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def prediction(model, source_latent, action_delta, options):
    x = torch.cat((source_latent, action_delta,
                   F.one_hot(options, 3).to(source_latent)), -1)
    residual = model.kanli.outcome(x)[..., :4]
    evidence = model._causal_use_value_evidence
    base = evidence.bases()[options].detach()
    return base + residual


def gain_loss(engine, target):
    """Use the existing replay budget on eligible labels of this objective.

Ordinary outcome replay still learns the other three channels from its bank.
This delayed gain step reads only actual, compatible, individually bound
episodes; no new example or optimizer step is manufactured.
"""
    m = engine.model.memory
    slot = engine._causal_gain_current_slot
    valid = (m.occupied & (m.count == 1) & (m.wood_target_count == 1) &
             (m.wood_option >= 0) & engine.reward_label_valid(wood=True))
    valid[slot] = False  # current event is supplied exactly once below
    candidates = torch.nonzero(valid).flatten()
    count = min(len(candidates), engine.replay_batch)
    if count:
        ids = engine.replay_rng.choice(len(candidates), count, replace=False)
        picked = candidates[torch.as_tensor(ids, device=candidates.device)]
    else:
        picked = candidates
    source = torch.cat((engine._wood_source[None], m.source[picked].detach()))
    action = torch.cat((engine._wood_action[None], m.action[picked].detach()))
    options = torch.cat((m.wood_option.new_tensor([engine._pending_record['choice']]),
                         m.wood_option[picked]))
    labels = torch.cat((target[3:4], m.wood_target[picked, 3].detach()))
    z, a = engine.model.encoder(source), engine.model.encoder(action)
    raw = prediction(engine.model, z, a-z, options)[..., 3]
    loss = engine.outcome_weight * F.mse_loss(raw, labels)
    return loss, dict(schema=1, replay_samples=count, replay_slots=picked.tolist(),
        target_units='10*finite_U_nats_per_latent_coordinate',
        raw_mse=float((raw-labels).square().mean().detach()),
        current_raw=float(raw[0].detach()), current_target=float(labels[0]),
        base_role='current_compatible_gain_baseline', input_role='plain_shared_gain_head')
