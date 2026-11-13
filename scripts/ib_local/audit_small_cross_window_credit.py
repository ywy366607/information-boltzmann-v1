"""Isolate >128-byte native credit with fixed-forward, discarded Adam probes.

No retained training and no language-retention objective. Compare the exact
first-digit boundary teaching signal, strict old native VJPs, and content use
after shadow steps. A wrong-receiver control permutes equal-shape old traces
without changing any forward response. All parameters/Adam/memories restore.
"""
import argparse
import copy
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


def norm(values):
    return sum(float(v.double().square().sum()) for v in values)**.5


def cosine(left, right):
    denominator = norm(left)*norm(right)
    return None if denominator == 0 else sum(float((a.double()*b.double()).sum())
                                            for a, b in zip(left, right))/denominator


def gradients(loss, parameters, *, with_used=False, **kwargs):
    values = torch.autograd.grad(loss, parameters, allow_unused=True, **kwargs)
    result = [torch.zeros_like(p) if g is None else g.detach().clone()
              for p, g in zip(parameters, values)]
    return (result, [g is not None for g in values]) if with_used else result


@contextmanager
def capture_native(bank, strict_slots, *, cut=False):
    captures = []
    originals = []
    for index, memory in enumerate(bank.memories):
        own = 'retrieve_context' in memory.__dict__
        old = memory.retrieve_context
        originals.append((memory, own, old))

        def read(query, block, index=index, memory=memory, old=old):
            own_forward = 'forward' in block.__dict__
            original_forward = block.forward

            def forward(x, state):
                result = original_forward(x, state)
                members = [s for s in torch.nonzero(memory.memory.occupied).flatten().tolist()
                           if memory.capsules[s].inputs.shape[1] == x.shape[1]]
                if len(members) != x.shape[0]:
                    raise RuntimeError('historical replay batch identity mismatch')
                mask = torch.tensor([s in strict_slots[index] for s in members], device=x.device)
                hidden = torch.where(mask[:, None, None], result[0].detach(), result[0])
                return (hidden, *result[1:])

            if cut:
                block.forward = forward
            try:
                output = old(query, block)
            finally:
                if cut:
                    if own_forward:
                        block.forward = original_forward
                    else:
                        del block.forward
            for slot, response in memory.replayed_by_slot.items():
                captures.append(dict(layer=index, slot=slot, response=response,
                    strict=slot in strict_slots[index], position_read=memory.position_read))
            return output

        memory.retrieve_context = read
    try:
        yield captures
    finally:
        for memory, own, old in originals:
            if own:
                memory.retrieve_context = old
            else:
                del memory.retrieve_context


def native_vjp(receipts, teaching, native):
    selected = [(r['response'], g) for r, g in zip(receipts, teaching) if g is not None]
    if not selected:
        return [torch.zeros_like(p) for p in native]
    values = torch.autograd.grad([r for r, _ in selected], native,
        grad_outputs=[g.detach() for _, g in selected], allow_unused=True, retain_graph=True)
    return [torch.zeros_like(p) if g is None else g.detach().clone()
            for p, g in zip(native, values)]


def measure_case(engine, row, alternative, window):
    model = engine.model
    bank = model.local_replay
    parameters = list(engine.params)
    native = [p for layer in model.base.layers for p in layer.parameters()]
    native_indices = [next(i for i, p in enumerate(parameters) if p is q) for q in native]
    with torch.no_grad():
        substitutes = collect_alternative(engine, row, alternative)
    prefix, _ = encode_case(row)
    number = row['outputs'][0]
    needle = prefix.index(number.encode())
    with request_branch(engine) as table, provenance(bank) as metadata:
        bank.set_held(False)
        starts = [int(m.memory.clock) for m in bank.memories]
        with torch.no_grad():
            _, state, pending = consume_prompt(model, table, prefix, 128)
            table.run_credit(np.array([pending, 32], dtype=np.uint8), max_window=1)
            logp = probability_tensor(model, peek(table, 32)[None])
        ids = torch.tensor([[32]], device=engine.device)
        a, b = ord(number[0]), ord(alternative[0])
        strict = []
        details = []
        pairs = []
        for index, m in enumerate(bank.memories):
            old = set()
            slots = []
            for slot in torch.nonzero(m.memory.occupied).flatten().tolist():
                capsule = m.capsules[slot]
                lo = int(m.memory.birth[slot])-starts[index]
                hi = lo+capsule.inputs.shape[1]
                # The current observed space is at prefix position len(prefix).
                # Require even the newest replayed position to precede it by >W.
                minimum_age = len(prefix)-(hi-1)
                is_strict = minimum_age > window
                direct = lo < needle+len(number) and hi > needle
                if is_strict:
                    old.add(slot)
                slots.append(dict(slot=slot, minimum_age_bytes=minimum_age,
                    direct_number_slice=direct, strict_outside_window=is_strict))
                p = metadata[index].get(slot)
                if p is not None:
                    bounds = (p['start']-starts[index], p['end']-starts[index])
                    if bounds in substitutes[index]:
                        pairs.append((index, slot, substitutes[index][bounds]))
            strict.append(old)
            details.append(dict(layer=index+1, records=slots))
        assert len({i for i, s, c in pairs}) == len(bank.memories)
        with capture_native(bank, strict) as receipts:
            logits = model.predict(ids, state, logp)[0]
        # Observe actual addressing before the independent cut forward replaces
        # transient read receipts. Researcher position labels never enter read.
        routing = []
        for index, memory in enumerate(bank.memories):
            read = next(r['position_read'] for r in receipts if r['layer'] == index)
            weights = read['weights'][0].detach().double()
            positions = read['value_positions']
            mask = torch.zeros_like(weights, dtype=torch.bool)
            for record, slot in enumerate(read['slots'].tolist()):
                lo = int(memory.memory.birth[slot])-starts[index]
                mask |= ((read['record_ids'] == record)
                         & (positions+lo >= needle)
                         & (positions+lo < needle+len(number)))
            routing.append(dict(layer=index+1,
                inlet_gain=float(memory.inlet.projection.gain.detach()),
                direct_value_mass=float(weights[mask].sum()),
                direct_value_positions=int(mask.sum()),
                candidate_value_positions=len(weights),
                uniform_direct_mass=float(mask.sum())/(len(weights)+1),
                null_mass=float(read['null_weight'][0].detach()),
                maximum_position_weight=float(weights.max()),
                direct_teaching_squared=0., all_teaching_squared=0.))
        loss = -logits[0, 0].log_softmax(-1)[a]
        full, used = gradients(loss, parameters, with_used=True, retain_graph=True)
        signals = torch.autograd.grad(loss, [r['response'] for r in receipts],
                                      allow_unused=True, retain_graph=True)
        for receipt, signal in zip(receipts, signals):
            if signal is None:
                continue
            index, slot = receipt['layer'], receipt['slot']
            memory = bank.memories[index]
            lo = int(memory.memory.birth[slot])-starts[index]
            left, right = max(0, needle-lo), min(signal.shape[1], needle+len(number)-lo)
            routing[index]['all_teaching_squared'] += float(signal.double().square().sum())
            if right > left:
                routing[index]['direct_teaching_squared'] += float(signal[:, left:right].double().square().sum())
        for value in routing:
            value['direct_teaching_squared_fraction'] = value['direct_teaching_squared']/max(value['all_teaching_squared'], 1e-30)
        long_receipts = [r for r in receipts if r['strict']]
        long_signals = [g for r, g in zip(receipts, signals) if r['strict']]
        extra = native_vjp(long_receipts, long_signals, native)
        # Same old participation and identical forward, different arrived
        # result: check that the native long teaching actually reads the result.
        alternative_signals = torch.autograd.grad(-logits[0, 0].log_softmax(-1)[b],
            [r['response'] for r in long_receipts], allow_unused=True, retain_graph=True)
        alternative_extra = native_vjp(long_receipts, alternative_signals, native)
        # Reassign teaching only within the same layer and native span. Values
        # in the executed forward are never permuted or replaced.
        wrong_signals = list(long_signals)
        groups = {}
        for i, r in enumerate(long_receipts):
            groups.setdefault((r['layer'], tuple(r['response'].shape)), []).append(i)
        reassigned = 0
        for members in groups.values():
            if len(members) > 1:
                reassigned += len(members)
                for target, source in zip(members, members[1:]+members[:1]):
                    wrong_signals[target] = long_signals[source]
        wrong = native_vjp(long_receipts, wrong_signals, native)
        cut_expected = [g.clone() for g in full]
        for i, old in zip(native_indices, extra):
            cut_expected[i].sub_(old)
        # Independent graph-detachment reconstruction checks the VJP isolation.
        with capture_native(bank, strict, cut=True):
            cut_logits = model.predict(ids, state, logp)[0]
        forward_error = float((logits.detach()-cut_logits.detach()).abs().max())
        cut_actual = gradients(-cut_logits[0, 0].log_softmax(-1)[a], parameters)
        difference = [x-y for x, y in zip(cut_expected, cut_actual)]
        identity_error = norm(difference)/max(norm(full), 1e-30)
        assert forward_error == 0 and identity_error < 2e-5
        del logits, cut_logits, loss, receipts, signals, long_signals, wrong_signals
        original_params = [p.detach().clone() for p in parameters]
        original_adam = copy.deepcopy(engine.optimizer.state_dict())
        factual_capsules = {(i, s): bank.memories[i].capsules[s] for i, s, c in pairs}

        def restore_weights():
            with torch.no_grad():
                for p, value in zip(parameters, original_params):
                    p.copy_(value)
            engine.optimizer.load_state_dict(copy.deepcopy(original_adam))
            engine.optimizer.zero_grad(set_to_none=True)

        @torch.no_grad()
        def content(*, by_layer=False):
            factual = model.predict(ids, state, logp)[0][0, 0].double()
            layer_contrasts = []
            if by_layer:
                for layer in range(len(bank.memories)):
                    try:
                        for i, slot, c in pairs:
                            if i == layer:
                                bank.memories[i].capsules[slot] = LocalExperience(c.inputs, c.boundary,
                                    int(bank.memories[i].memory.read_credit_version[slot]))
                        altered = model.predict(ids, state, logp)[0][0, 0].double()
                        layer_contrasts.append(dict(layer=layer+1,
                            stored_value_contrast=float((factual[a]-factual[b])-(altered[a]-altered[b])),
                            maximum_logit_change=float((factual-altered).abs().max())))
                    finally:
                        for (i, slot), c in factual_capsules.items():
                            bank.memories[i].capsules[slot] = c
            try:
                for i, slot, c in pairs:
                    m = bank.memories[i]
                    m.capsules[slot] = LocalExperience(c.inputs, c.boundary,
                        int(m.memory.read_credit_version[slot]))
                substitute = model.predict(ids, state, logp)[0][0, 0].double()
            finally:
                for (i, slot), c in factual_capsules.items():
                    bank.memories[i].capsules[slot] = c
            return dict(first_digit_nll=float(-factual.log_softmax(-1)[a]),
                **({'per_layer_content': layer_contrasts} if by_layer else {}),
                content_contrast=float((factual[a]-factual[b])-(substitute[a]-substitute[b])))

        base = content(by_layer=True)
        shadow = {}
        update_vectors = {}
        wrong_full = [g.clone() for g in cut_expected]
        for i, old in zip(native_indices, wrong):
            wrong_full[i].add_(old)
        try:
            for name, values in (('matched', full), ('cut_long', cut_expected), ('wrong_long', wrong_full)):
                restore_weights()
                for p, g, active in zip(parameters, values, used):
                    p.grad = 10*g.clone() if active else None
                clips = [float(torch.nn.utils.clip_grad_norm_(group, 1., foreach=True)) for group in engine.groups]
                engine.optimizer.step()
                changes = [p.detach()-old for p, old in zip(parameters, original_params)]
                update_vectors[name] = [x.clone() for x in changes]
                value = content()
                shadow[name] = dict(**value, update_norm=norm(changes),
                    preclip_group_norms=clips, content_contrast_change=value['content_contrast']-base['content_contrast'])
        finally:
            restore_weights()
        layers = []
        offset = 0
        for index, layer in enumerate(model.base.layers):
            count = len(list(layer.parameters()))
            indices = native_indices[offset:offset+count]
            old = extra[offset:offset+count]
            total = [full[i] for i in indices]
            layers.append(dict(layer=index+1, long_native_gradient_norm=norm(old),
                full_native_gradient_norm=norm(total), long_to_full_norm_ratio=norm(old)/max(norm(total),1e-30),
                long_vs_remaining_cosine=cosine(old,[x-y for x,y in zip(total,old)])))
            offset += count
        return dict(case=row['index'], answer_distance_bytes=len(prefix)-needle,
            strict_window_bytes=window, strict_receipts=len(long_receipts), reassigned_receipts=reassigned,
            same_forward_max=forward_error, cut_vjp_identity_relative_error=identity_error,
            layers=layers, routing=routing, record_ages=details, matched_vs_wrong_native_cosine=cosine(extra, wrong),
            same_participation_alternative_result_long_cosine=cosine(extra, alternative_extra),
            alternative_result_long_gradient_difference=norm([a-b for a,b in zip(extra, alternative_extra)]),
            matched_vs_cut_parameter_update_difference=norm([a-b for a,b in zip(update_vectors['matched'],update_vectors['cut_long'])]),
            matched_vs_wrong_parameter_update_difference=norm([a-b for a,b in zip(update_vectors['matched'],update_vectors['wrong_long'])]),
            base=base, shadows=shadow,
            matched_minus_cut_content_contrast=shadow['matched']['content_contrast']-shadow['cut_long']['content_contrast'],
            matched_minus_wrong_content_contrast=shadow['matched']['content_contrast']-shadow['wrong_long']['content_contrast'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'data', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--examples', type=int, default=8)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    if args.device == 'cuda':
        from scripts.ib_local import gdn_fused
        gdn_fused.enable()
    engine, _, _, _ = build(args.checkpoint, device=args.device)
    assert engine.model.local_replay.schema == 5
    digest = continuation_digest(engine)
    source_hash = sha256(args.checkpoint)
    rows = [json.loads(line) for line in args.data.read_text().splitlines()][:args.examples]
    report = dict(status='running', source_sha256=source_hash, retained_optimizer_updates=0,
        executed_script_sha256=sha256(__file__),
        protocol=__doc__, device=args.device, new_lean_coverage=0,
        shadow_objective='10*CE on the current first answer digit, using source Adam and production group clipping; no auxiliary loss or memory preference commits. Not a full production next-update replay.',
        records=[])
    def write():
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    start = time.time()
    write()
    try:
        for row in rows:
            alternative = next(r['outputs'][0] for r in rows if r['outputs'][0][0] != row['outputs'][0][0])
            report['records'].append(measure_case(engine, row, alternative, 128))
            assert continuation_digest(engine) == digest
            report['seconds'] = time.time()-start
            write()
            last = report['records'][-1]
            print('CREDIT',row['index'],last['matched_vs_cut_parameter_update_difference'],
                  last['matched_minus_cut_content_contrast'],flush=True)
        report.update(status='complete', complete_state_unchanged=continuation_digest(engine)==digest,
            checkpoint_unchanged=sha256(args.checkpoint)==source_hash, seconds=time.time()-start,
            executed_script_unchanged=sha256(__file__)==report['executed_script_sha256'],
            matched_beats_cut_content_cases=sum(r['matched_minus_cut_content_contrast']>0 for r in report['records']),
            matched_beats_wrong_content_cases=sum(r['matched_minus_wrong_content_contrast']>0 for r in report['records']))
    except Exception as exc:
        report.update(status='failed', error=repr(exc), seconds=time.time()-start)
        raise
    finally:
        write()


if __name__ == '__main__':
    main()
