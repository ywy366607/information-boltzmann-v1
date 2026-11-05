"""One-individual real-language connectivity inspection; no optimizer step.

The recorded local interventions are the user's specified pre-training gate,
not model comparisons or capability evaluations. Unconnected links remain null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.language_action import digest, peek
from scripts.ib_local.language_navigation_stream_v3 import StreamingIndividual
from scripts.ib_local.language_navigation_world_v3 import StreamingLanguageWorld
from scripts.ib_local.latent_navigation_v3 import LatentNavigationV3
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.run_ngram_fast_table import protocol_stream
from scripts.ib_local.train_gdn_fast_table import FLOOR, TableGDN


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bytes', type=int, default=600)
    p.add_argument('--seed', type=int, default=11)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve the existing check')
    if args.bytes <= 0:
        raise ValueError('positive inspection budget required')
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gdn_fused.enable()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    old = saved['config']
    if old['mode'] != 'off' or saved['step'] != old['steps']:
        raise ValueError('requires completed source individual')
    previous, _ = protocol_stream(old['steps'], old['tokens'])
    body = TableGDN(GatedDeltaNetLM(256,128,4,4,'decay_then_delta',256,4,32.,
                                    decay_init='log_timescale').to(device), True).to(device)
    body.load_state_dict(saved['model'])
    model = LatentNavigationV3(body).to(device).eval()
    table = RetrievedCreditTable.from_state_dict(saved['table'])
    root = Path('data/ib_owt_10000')
    offset = json.loads((root/'tokenizer.json').read_text())['offset']
    corpus = np.load(root/'train.npy', mmap_mode='r')
    raw = np.asarray(corpus[8_000_000:])
    data = (raw[raw >= offset] - offset).astype(np.uint8)
    world = StreamingLanguageWorld(data, seed=args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    individual = StreamingIndividual(model, table, world, saved['recurrent'].to(device), int(previous[-1]))
    say, silent, lengths = 0, 0, []
    for _ in range(args.bytes):
        utterance = individual.step()
        if utterance:
            say += 1
            lengths.append(len(utterance))
        else:
            silent += 1
    memory = model.memory
    # A read-only local intervention on the same state, not a training arm.
    ids = torch.tensor([[individual.pending]], device=device)
    logp = torch.as_tensor(np.log(np.maximum(peek(table, individual.pending), FLOOR)), device=device)[None, None]
    with torch.no_grad():
        _, state_on, _, h_on, _ = model.predict_observed(ids, individual.recurrent, logp)
        plan_on = model.plan_from_state(h_on)
        occupied = memory.occupied.clone()
        memory.occupied.zero_()
        _, state_off, _, h_off, _ = model.predict_observed(ids, individual.recurrent, logp)
        plan_off = model.plan_from_state(h_off)
        memory.occupied.copy_(occupied)
        q = torch.distributions.Bernoulli(logits=plan_on['decision_logits'][0,0])
        r = torch.distributions.Bernoulli(logits=plan_off['decision_logits'][0,0])
        kl = float(torch.distributions.kl_divergence(q, r))
        state_delta = float((state_on-state_off).norm())
        age = (memory.clock-memory.birth[plan_on['slots']]).clamp_min(0)
        old_weight = float((plan_on['weights'] * (age > memory.long)).sum())
    # Exactly one backward: inspect the current action logit through the same
    # fixed stored content. This is not the derivative of an old action.
    memory.action.requires_grad_(True)
    _, _, _, h, _ = model.predict_observed(ids, individual.recurrent, logp)
    plan = model.plan_from_state(h)
    local_policy_signal = torch.zeros((), device=device)
    policy_credit_grad_norm = 0.
    if individual.progress_receipts:
        receipt = individual.progress_receipts[-1]
        factor = float(receipt['progress'])
        eligibility = receipt['eligibility']
        bias_eligibility = float(receipt['bias_eligibility'])
        local_policy_signal = -factor * ((model.action_decisions.weight[0] * eligibility).sum() +
                                         model.action_decisions.bias[0] * bias_eligibility)
        policy_credit_grad_norm = abs(factor) * float(torch.sqrt(eligibility.square().sum() + bias_eligibility**2))
    (plan['decision_logits'][0,0] + local_policy_signal).backward()
    encoder_grad = float(model.encoder.weight.grad.norm()) if model.encoder.weight.grad is not None else 0.
    action_grad = memory.action.grad
    old = memory.occupied & ((memory.clock-memory.birth) > memory.long)
    old_grad = float(action_grad[old].norm()) if action_grad is not None and bool(old.any()) else 0.
    progress_count = int(memory.progress_valid.sum())
    actions = list(individual.world_actions)
    result = dict(scope='one real-OWT individual, forward interaction and one backward; no optimizer updates',
        checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
        source_raw_bounds=[8_000_000,len(corpus)], source_bytes=len(data), seed=args.seed,
        source_byte_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
        source_code_sha256={str(path):digest(path) for path in (
            Path(__file__), Path('scripts/ib_local/language_navigation_stream_v3.py'),
            Path('scripts/ib_local/language_navigation_world_v3.py'),
            Path('scripts/ib_local/latent_navigation_v3.py'),
            Path('scripts/ib_local/latent_navigation_v2.py'),
            Path('scripts/ib_local/gated_deltanet.py'))},
        external_bytes=individual.external_bytes, self_bytes=individual.self_bytes,
        decision_ticks=args.bytes, spoken=say, silent=silent,
        utterance_lengths=lengths, bounded_truncations=individual.truncated,
        dropped_unsettled=individual.dropped_experiences,
        overlapping_action_events=individual.overlapping_action_events,
        matched_actions=sum(e['source']=='matched' for e in actions),
        distinct_recent_response_starts=len({e['start'] for e in actions}),
        episodes=int(memory.episodes), occupied=int(memory.occupied.sum()),
        episodes_with_progress=progress_count,
        settled_progress_receipts=individual.progress_count,
        probe_diagnostics=dict(candidates=individual.probe_candidates,
            interrupted=individual.probe_interrupted, previously_seen_or_out_of_source=individual.probe_previously_seen,
            post_scored=individual.probe_post_scored, delivered=individual.progress_count),
        episodes_with_intervening_actions=int((memory.occupied & (memory.intervention_count>0)).sum()),
        oldest_content_age=int((memory.clock-memory.birth[memory.occupied]).max()) if bool(memory.occupied.any()) else None,
        pretraining_gate=dict(
            water_to_wood=dict(action_kl=kl,recurrent_state_l2=state_delta,status='measurements_only'),
            wood_to_fire=dict(current_encoder_gradient_l2=encoder_grad,status='partial_route_only'),
            fire_to_world=dict(matched_actions=sum(e['source']=='matched' for e in actions),status='repeat-query check pending'),
            fire_to_earth=dict(status='not checked: response-end intent probe not implemented'),
            earth_to_metal=dict(settled_progress_receipts=individual.progress_count,
                local_start_score_credit_gradient_l2=policy_credit_grad_norm,
                status='local frozen-feature start-score path only; full policy and useful credit unverified'),
            metal_to_water=dict(written_episodes=int(memory.episodes),with_progress=progress_count,status='fails: progress not connected'),
            water_to_wood_long=dict(old_read_weight=old_weight,old_value_gradient_l2=old_grad,status='partial local re-encoding only'),
            fire_rhythm=dict(spoken=say,silent=silent,truncated=individual.truncated,status='untrained sampling; not behavioral capability'),
            all_pass=False),
        warning='Stored features and local re-encoding are not exact long-range action credit; no life claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
