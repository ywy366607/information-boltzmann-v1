"""Locate schema5 content/order attenuation through the actual downstream body.

Frozen same-S/table interventions in layer2 only. Compare observed alternate
content, same-digit anagrams, fixed-weight value changes, fixed-value routing,
and researcher-selected first-position delivery. Oracle selection is a probe,
never a production address, loss or learning rule. No optimizer steps.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.ib_local.audit_small_native_content import collect_alternative
from scripts.ib_local.audit_small_native_failure import provenance
from scripts.ib_local.language_action import peek
from scripts.ib_local.pretrained_gdn_experience import LocalExperience
from scripts.ib_local.run_causal_water_reply import build
from scripts.ib_local.small_gdn_ruler import (
    consume_prompt, encode_case, probability_tensor, request_branch, sha256,
)
from scripts.ib_local.train_small_native_credit import continuation_digest


def rms(value):
    return float(value.double().square().mean().sqrt())


@contextmanager
def capture(bank, base, overrides=None):
    snapshots = {}
    originals = []
    boundary = {}
    decoder_calls = []

    def norm_hook(module, arguments, output):
        boundary['pre_norm'] = arguments[0][0, -1].detach().cpu().clone()
        boundary['normalized'] = output[0, -1].detach().cpu().clone()

    def decoder_hook(module, arguments, output):
        decoder_calls.append(1)
        boundary['neural_logits'] = output[0, -1].detach().cpu().clone()

    handles = [base.final_norm.register_forward_hook(norm_hook),
               base.decoder.register_forward_hook(decoder_hook)]
    for index, memory in enumerate(bank.memories):
        own = 'retrieve_context' in memory.__dict__
        previous = memory.retrieve_context
        originals.append((memory, own, previous))

        def read(query, block, index=index, memory=memory, previous=previous):
            result = previous(query, block)
            pr = memory.position_read
            weights = pr['weights'][0].detach().cpu()
            positions = pr['value_positions'].detach().cpu()
            owner = pr['record_ids'].detach().cpu()
            slots = pr['slots'].detach().cpu()
            responses = {slot: memory.replayed_by_slot[slot][0].detach().cpu().clone()
                         for slot in slots.tolist()}
            values = query.new_empty(len(weights), query.shape[-1], device='cpu')
            for record, slot in enumerate(slots.tolist()):
                selected = owner == record
                values[selected] = responses[slot][positions[selected]]
            recomputed = (weights.double() @ values.double())[None]
            error = float((recomputed-result.detach().cpu().double()).abs().max())
            if error > 2e-5:
                raise AssertionError(('value ordering mismatch', index+1, error))
            actual = result
            if overrides and index in overrides:
                actual = overrides[index](result).to(result)
            injection = memory.inlet.projection(actual.to(query.device))
            snapshots[index] = dict(query=query.detach().cpu().clone(),
                recall=actual.detach().cpu().clone(), injection=injection.detach().cpu().clone(),
                delivered=(query+injection).detach().cpu().clone(),
                weights=weights.clone(), values=values, value_positions=positions.clone(),
                owners=owner.clone(), slots=slots.clone(), responses=responses,
                value_reconstruction_max_error=error)
            return actual

        memory.retrieve_context = read
    try:
        yield snapshots
        assert len(decoder_calls) == 1
        snapshots['boundary'] = boundary
    finally:
        for handle in handles:
            handle.remove()
        for memory, own, previous in originals:
            if own:
                memory.retrieve_context = previous
            else:
                del memory.retrieve_context


def contrast(factual, changed, a, b):
    f, c = factual.double(), changed.double()
    return dict(stored_value_contrast=float((f[a]-f[b])-(c[a]-c[b])),
        correct_log_probability_benefit=float(f.log_softmax(-1)[a]-c.log_softmax(-1)[a]),
        maximum_logit_change=float((f-c).abs().max()))


@torch.no_grad()
def measure(engine, row, alternative, target=1):
    model, bank = engine.model, engine.model.local_replay
    number = row['outputs'][0]
    # Official answers have a nonzero first digit. Preserve that support when
    # changing order; a leading-zero contrast would probe an unseen format.
    rotated = next(number[k:]+number[:k] for k in range(1, len(number))
                   if number[k] != number[0] and number[k] != '0')
    assert sorted(rotated) == sorted(number) and rotated[0] != '0'
    alternatives = {name: collect_alternative(engine, row, value)[target]
                    for name, value in [('replacement', alternative), ('anagram', rotated)]}
    prefix, _ = encode_case(row)
    needle = prefix.index(number.encode())
    with request_branch(engine) as table, provenance(bank) as metadata:
        starts = [int(m.memory.clock) for m in bank.memories]
        _, state, pending = consume_prompt(model, table, prefix, 128)
        table.run_credit(np.array([pending, 32], dtype=np.uint8), max_window=1)
        logp = probability_tensor(model, peek(table, 32)[None])
        ids = torch.tensor([[32]], device=engine.device)
        memory = bank.memories[target]
        factual_capsules = {s: c for s, c in enumerate(memory.capsules) if c is not None}

        def restore():
            for slot, capsule in factual_capsules.items():
                memory.capsules[slot] = capsule

        def execute(overrides=None):
            with capture(bank, model.base, overrides) as snapshots:
                predicted = model.predict(ids, state, logp)
                logits = predicted[0][0, 0].detach().cpu()
                mixture_weight = float(predicted[2]['table_weight_mean'])
            snapshots['boundary']['table_weight'] = mixture_weight
            return logits, snapshots

        def install(name):
            restore()
            for slot, meta in metadata[target].items():
                bounds = meta['start']-starts[target], meta['end']-starts[target]
                if bounds in alternatives[name]:
                    c = alternatives[name][bounds]
                    memory.capsules[slot] = LocalExperience(c.inputs, c.boundary,
                        int(memory.memory.read_credit_version[slot]))

        def digit_vectors(snapshot):
            result = []
            for offset in range(len(number)):
                position = needle+offset
                matches = []
                for slot, meta in metadata[target].items():
                    lo, hi = meta['start']-starts[target], meta['end']-starts[target]
                    if lo <= position < hi:
                        matches.append(snapshot['responses'][slot][position-lo])
                if len(matches) != 1:
                    raise AssertionError(('missing or ambiguous observed digit', row['index'], offset))
                result.append(matches[0])
            return torch.stack(result)

        try:
            factual, baseline = execute()
            base = baseline[target]
            original_digits = digit_vectors(base)
            comparisons = []
            for name, replacement in [('replacement', alternative), ('anagram', rotated)]:
                install(name)
                changed, after = execute()
                altered = after[target]
                assert torch.equal(base['query'], altered['query'])
                assert torch.equal(base['slots'], altered['slots'])
                assert torch.equal(base['owners'], altered['owners'])
                assert torch.equal(base['value_positions'], altered['value_positions'])
                changed_digits = digit_vectors(altered)
                value_part = base['weights'].double() @ (base['values']-altered['values']).double()
                route_part = (base['weights']-altered['weights']).double() @ altered['values'].double()
                read_delta = (base['recall']-altered['recall']).double()[0]
                a, b = ord(number[0]), ord(replacement[0])
                per_digit_mass = []
                for offset in range(len(number)):
                    mask = torch.zeros(len(base['weights']), dtype=torch.bool)
                    for record, slot in enumerate(base['slots'].tolist()):
                        lo = int(memory.memory.birth[slot])-starts[target]
                        mask |= (base['owners'] == record) & (base['value_positions']+lo == needle+offset)
                    per_digit_mass.append(float(base['weights'][mask].sum()))
                restore()
                fixed_value_logits, _ = execute({target: lambda old: (
                    base['weights'] @ altered['values'])[None]})
                fixed_route_logits, _ = execute({target: lambda old: (
                    altered['weights'] @ base['values'])[None]})
                oracle_factual, _ = execute({target: lambda old: original_digits[0:1]})
                oracle_changed, _ = execute({target: lambda old: changed_digits[0:1]})
                flipped_factual, flipped_base = execute({target: lambda old: -old})
                install(name)
                flipped_changed, flipped_after = execute({target: lambda old: -old})
                clamped, _ = execute({i: (lambda old, i=i: baseline[i]['recall'])
                                      for i in range(target+1, len(bank.memories))})
                stage_deltas = []
                for index in range(target, len(bank.memories)):
                    stage_deltas.append(dict(layer=index+1, **{
                        key+'_delta_rms': rms(baseline[index][key]-after[index][key])
                        for key in ('query', 'recall', 'injection', 'delivered')}))
                fb, cb = baseline['boundary'], after['boundary']
                neural_delta = fb['neural_logits'].double()-cb['neural_logits'].double()
                lexical_delta = (fb['table_weight']-cb['table_weight'])*logp[0, 0].double().cpu()
                neural_contrast = float(neural_delta[a]-neural_delta[b])
                lexical_contrast = float(lexical_delta[a]-lexical_delta[b])
                actual_contrast = contrast(factual, changed, a, b)
                comparisons.append(dict(kind=name, alternative=replacement,
                    same_digit_multiset=sorted(number)==sorted(replacement),
                    actual=actual_contrast,
                    fixed_weights_changed_values=contrast(factual, fixed_value_logits, a, b),
                    fixed_values_changed_weights=contrast(factual, fixed_route_logits, a, b),
                    oracle_first_digit=contrast(oracle_factual, oracle_changed, a, b),
                    flipped_injection=contrast(flipped_factual, flipped_changed, a, b),
                    flipped_neural_contrast=contrast(flipped_base['boundary']['neural_logits'],
                        flipped_after['boundary']['neural_logits'], a, b)['stored_value_contrast'],
                    factual_first_digit_nll=float(-factual.double().log_softmax(-1)[a]),
                    flipped_factual_first_digit_nll=float(-flipped_factual.double().log_softmax(-1)[a]),
                    changed_content_with_downstream_recall_clamped=contrast(factual, clamped, a, b),
                    observed_digit_response_delta_rms=rms(original_digits-changed_digits),
                    recall_delta_rms=rms(read_delta), value_only_delta_rms=rms(value_part),
                    routing_only_delta_rms=rms(route_part),
                    read_decomposition_max_error=float((read_delta-value_part-route_part).abs().max()),
                    boundary=dict(pre_norm_delta_rms=rms(fb['pre_norm']-cb['pre_norm']),
                        normalized_delta_rms=rms(fb['normalized']-cb['normalized']),
                        factual_pre_norm_rms=rms(fb['pre_norm']),
                        factual_normalized_rms=rms(fb['normalized']),
                        neural_logits_delta_rms=rms(neural_delta),
                        neural_stored_value_contrast=neural_contrast,
                        lexical_stored_value_contrast=lexical_contrast,
                        contrast_decomposition_error=abs(actual_contrast['stored_value_contrast']-neural_contrast-lexical_contrast),
                        factual_table_weight=fb['table_weight'], changed_table_weight=cb['table_weight']),
                    per_digit_value_mass=per_digit_mass, stage_deltas=stage_deltas))
            return dict(case=row['index'], target_layer=target+1, number=number,
                answer_distance_bytes=len(prefix)-needle,
                inlet_gain=float(memory.inlet.projection.gain), comparisons=comparisons,
                output_weight_norm=(None if memory.inlet.projection.output_weight is None
                                    else float(memory.inlet.projection.output_weight.norm())),
                value_reconstruction_max_error=max(baseline[i]['value_reconstruction_max_error'] for i in range(len(bank.memories))))
        finally:
            restore()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'data', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--examples', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    if args.device == 'cuda':
        from scripts.ib_local import gdn_fused
        gdn_fused.enable()
    engine, _, _, _ = build(args.checkpoint, device=args.device)
    assert engine.model.local_replay.schema == 5
    before = continuation_digest(engine)
    source_hash, script_hash = sha256(args.checkpoint), sha256(__file__)
    rows = [json.loads(line) for line in args.data.read_text().splitlines()][:args.examples]
    report = dict(status='running', protocol=__doc__, retained_optimizer_updates=0,
        source_sha256=source_hash, script_sha256=script_hash, data_sha256=sha256(args.data),
        device=args.device, new_lean_coverage=0, records=[])
    def write():
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    start = time.time()
    try:
        for row in rows:
            alternate = next(r['outputs'][0] for r in rows if r['outputs'][0][0] != row['outputs'][0][0])
            report['records'].append(measure(engine, row, alternate))
            assert continuation_digest(engine) == before
            report['seconds'] = time.time()-start
            write()
            print('DELIVERY', row['index'], report['records'][-1]['comparisons'][1]['actual'], flush=True)
        report.update(status='complete', complete_state_unchanged=continuation_digest(engine)==before,
            source_unchanged=sha256(args.checkpoint)==source_hash,
            script_unchanged=sha256(__file__)==script_hash, seconds=time.time()-start)
    except Exception as exc:
        report.update(status='failed', error=repr(exc), seconds=time.time()-start)
        raise
    finally:
        write()


if __name__ == '__main__':
    main()
