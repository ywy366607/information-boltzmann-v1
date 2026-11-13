"""Delayed labels settle actually emitted prediction participation.

A bounded pending answer stores no weights, optimizer or past graph. Each
decision pins its input, held native boundary and then-visible read view.
Feedback may teach those decisions later, without replaying real history or
introducing records written after the decision into its responsibility.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, fields, is_dataclass

import torch
from torch.nn import functional as F

from scripts.ib_local.pretrained_gdn_experience import copy_tree


@dataclass
class ReadView:
    memory: object
    capsules: list
    null_read_enabled: bool
    recall_placement: str
    read_interval: int | None


@dataclass
class PredictionParticipation:
    inputs: torch.Tensor
    boundary: dict | None
    views: dict
    clock_offset: int
    parameter_version: int
    prediction_seen: int
    logits: torch.Tensor | None = None
    emitted: int | None = None
    reads: dict | None = None


def capture_reads(learner):
    """The activities/values that actually participated, not a new retrieval."""
    result = {}
    for layer, wrapper in learner.memories.items():
        read = wrapper.last_read
        positions = wrapper.current['query_positions']
        result[layer] = dict(query=copy_tree(wrapper.current['source'][0, positions]),
            slots=copy_tree(read.slots), weights=copy_tree(read.weights),
            responses={slot:copy_tree(value) for slot,value in wrapper.replayed_by_slot.items()})
    return result


def anchor_participation(shadow, held):
    """Evaluate local derivatives at the actually participating activities.

    Checkpoint reexecution may round differently and switch a discrete top-k
    choice. The eligibility is the real query and response, not a newly chosen
    neighbour. Value substitution holds those observed forward activities;
    autograd still differentiates the shared maps and old native computation.
    It never supplies a reference answer or an unobserved response.
    """
    original = shadow.retrieve
    values = {id(shadow.capsules[slot]):value for slot,value in held['responses'].items()}

    def retrieve(query, replay):
        actual_query = held['query'].to(query)
        if query.shape != actual_query.shape:
            raise ValueError('one held activity per actual query required')
        query = actual_query + (query-query.detach())

        def local(capsule):
            current = replay(capsule)[:, -1:]
            actual = values[id(capsule)].to(current)[:, None]
            return actual + (current-current.detach())

        result = original(query, local)
        if not torch.equal(shadow.last_read.slots.cpu(), held['slots']):
            raise RuntimeError('reexecution changed the actually participating identities')
        torch.testing.assert_close(shadow.last_read.weights.detach().cpu(), held['weights'],
                                   atol=1e-6, rtol=1e-6)
        return result

    shadow.retrieve = retrieve


def release_view(shadow):
    """Break temporary bound-method/closure cycles after one local backward."""
    shadow.__dict__.pop('retrieve', None)
    shadow.current, shadow.last_read = None, None
    shadow.replayed, shadow.replayed_by_slot = [], {}
    shadow.memory._read_participation = []
    shadow.memory._read_credit_active = False


def capture_views(learner):
    result = {}
    for layer, wrapper in learner.memories.items():
        if wrapper.memory._read_credit_active:
            raise RuntimeError('cannot capture a view inside an unsettled backward')
        result[layer] = ReadView(copy.deepcopy(wrapper.memory), list(wrapper.capsules),
                                wrapper.null_read_enabled, wrapper.recall_placement,
                                wrapper.read_interval)
    return result


def fork_view(wrapper, view):
    """Functional replay view: share trainable maps, isolate historical buffers."""
    shadow = copy.copy(wrapper)
    shadow._modules = dict(wrapper._modules)
    shadow._parameters = dict(wrapper._parameters)
    shadow._buffers = dict(wrapper._buffers)
    # Never inherit the no-grad decode cache's bound retrieve method.
    shadow.__dict__.pop('retrieve', None)
    shadow.memory = copy.deepcopy(view.memory)
    shadow.capsules = list(view.capsules)
    shadow.null_read_enabled = view.null_read_enabled
    shadow.recall_placement = view.recall_placement
    shadow.read_interval = view.read_interval
    shadow.current, shadow.replayed, shadow.replayed_by_slot = None, [], {}
    shadow.last_read, shadow.query_clock_offsets = None, None
    shadow.suppress_older_than, shadow.suppressed_record = None, None
    shadow._replaying = False
    return shadow


def unique_tensor_bytes(value):
    seen = set()

    def count(item):
        if isinstance(item, torch.Tensor):
            key = (str(item.device), item.untyped_storage().data_ptr())
            if key in seen:
                return 0
            seen.add(key)
            return item.untyped_storage().nbytes()
        if isinstance(item, torch.nn.Module):
            return sum(count(t) for t in item.buffers())
        if is_dataclass(item):
            return sum(count(getattr(item, f.name)) for f in fields(item))
        if isinstance(item, dict):
            return sum(count(v) for v in item.values())
        if isinstance(item, (list, tuple)):
            return sum(count(v) for v in item)
        return 0

    return count(value)


class PendingAnswer:
    def __init__(self, *, capacity=16):
        if not 0 < capacity <= 32:
            raise ValueError('bounded answer participation required')
        self.capacity, self.records = capacity, []
        self.consumed = False
        self.answer_end_seen = None

    def add(self, record, emitted):
        if self.consumed or self.answer_end_seen is not None or len(self.records) >= self.capacity:
            raise RuntimeError('pending participation is closed or full')
        if record is None or record.logits is None:
            raise RuntimeError('actual prediction must exist before emission')
        if record.emitted is not None:
            raise RuntimeError('one prediction participation per actual emission')
        record.emitted = int(emitted)
        self.records.append(record)

    def finish_emission(self, seen):
        if not self.records or self.answer_end_seen is not None:
            raise RuntimeError('answer emission must finish exactly once')
        if seen <= max(r.prediction_seen for r in self.records):
            raise ValueError('all emitted tokens must already be heard')
        self.answer_end_seen = int(seen)

    @property
    def nbytes(self):
        return unique_tensor_bytes(self.records)

    def validate_feedback(self, learner, feedback_seen):
        if self.consumed or not self.records:
            raise RuntimeError('pending answer is absent or already settled')
        if any(r.parameter_version != learner.updates for r in self.records):
            raise RuntimeError('parameters changed before this held-participation settlement')
        if self.answer_end_seen is None or feedback_seen <= self.answer_end_seen:
            raise RuntimeError('feedback must arrive after actual answer emission')


@contextmanager
def differentiable_short_recurrence():
    """Use the existing FLA chunk derivative for <=64-token local reexecution.

    FLA's layer otherwise forces its inference-only fused recurrent operator
    for short sequences. Actual life decoding keeps that original operator.
    No new recurrent equation, manual backward or longer history is installed.
    """
    from importlib import import_module
    module = import_module('fla.layers.gated_deltanet')
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    original = module.fused_recurrent_gated_delta_rule

    def operation(*args, **kwargs):
        if torch.is_grad_enabled():
            return chunk_gated_delta_rule(*args, **kwargs)
        return original(*args, **kwargs)

    module.fused_recurrent_gated_delta_rule = operation
    try:
        yield
    finally:
        module.fused_recurrent_gated_delta_rule = original


def align_numeric_feedback(pending, tokenizer, reference):
    """World-side alignment to the scored numeric content, not punctuation.

    The current official task/tokenizer emits one token per digit. Reject
    unsupported formats rather than silently teaching a different sequence.
    No reference enters retrieval, action selection or the lived cache.
    """
    if not reference.isascii() or not reference.isdigit():
        raise ValueError('numeric answer contract requires ASCII digits')
    labels=[]
    for digit in reference:
        encoded=tokenizer.encode(' '+digit,add_special_tokens=False)
        candidates=[int(t) for t in encoded if tokenizer.decode([int(t)]).strip()==digit]
        if len(candidates)!=1:
            raise ValueError('this alignment requires one token per digit')
        labels.append(candidates[0])
    positions=[]
    started=False
    for index,record in enumerate(pending.records):
        token=tokenizer.decode([record.emitted]).strip()
        digit=len(token)==1 and token.isascii() and token.isdigit()
        if digit:
            started=True
            positions.append(index)
        elif started:
            break
    if not positions:
        raise ValueError('no actually emitted numeric participation to settle')
    count=min(len(positions),len(labels))
    return positions[:count],labels[:count]


def settle_answer(learner, pending, labels, *, feedback_seen, apply=True, positions=None,
                  return_native_gradients=False):
    """On-policy token-position correction, after the world's answer arrives.

    Labels align to actual emission positions, so replay always conditions on
    the actually emitted prefix, including mistakes. A reference prefix is
    never substituted into the lived state. EOS is an ordinary teaching label.
    """
    from scripts.ib_local.pretrained_gdn import new_recurrent_cache, clone_cache
    from scripts.ib_local.language_navigation_kanli_v4l import live_sigreg
    from scripts.ib_local.navigation_read_credit import configure_read_credit

    pending.validate_feedback(learner, feedback_seen)
    count = min(len(labels), len(pending.records))
    if not count:
        raise ValueError('external labels are required')
    labels=list(labels)
    if positions is None:
        labels=labels[:count]
        positions=list(range(count))
    else:
        positions=list(positions)
    if (len(positions)!=len(labels) or len(set(positions))!=len(positions)
            or positions!=sorted(positions) or not positions
            or positions[0]<0 or positions[-1]>=len(pending.records)):
        raise ValueError('one external label per distinct actually issued position required')
    count=len(positions)
    teaching=dict(zip(positions,labels))
    learner.optimizer.zero_grad(set_to_none=True)
    for _, parameter in learner.native:
        parameter.grad = None
    device = next(learner.model.parameters()).device
    live = learner.cache
    lived_seen, lived_tokens = live._seen_tokens, learner.tokens
    receipts = {i: [] for i in learner.memories}
    rows = []
    chain_cache = None
    with differentiable_short_recurrence():
        for emission_index,record in enumerate(pending.records[:positions[-1]+1]):
            shadows = {i: fork_view(w, record.views[i]) for i,w in learner.memories.items()}
            if record.reads is None:
                raise ValueError('actual read activities are required for delayed settlement')
            for i, shadow in shadows.items():
                anchor_participation(shadow, record.reads[i])
            if record.boundary is not None:
                cache = new_recurrent_cache(record.boundary['seen_tokens'])
                cache.states = copy_tree(record.boundary['states'], device)
            elif chain_cache is not None:
                cache = chain_cache
            else:
                raise ValueError('the first prediction needs an actual incoming boundary')
            # Reconstruct the next held slice with the original inference
            # operator and the actually emitted prefix. This bounds pending
            # storage to one native cache, rather than one cache per token.
            with torch.no_grad(), ExitStack() as stack:
                for i, shadow in shadows.items():
                    stack.enter_context(shadow.attach(learner.model, layer=i, capture=False,
                                                      clock_offset=record.clock_offset))
                original = learner.model(record.inputs.to(device),
                    past_key_values=clone_cache(cache), use_cache=True)
                chain_cache = clone_cache(original.past_key_values)
            if emission_index not in teaching:
                for shadow in shadows.values():
                    release_view(shadow)
                del original,shadows,cache
                continue
            label=teaching[emission_index]
            issued = record.logits.float()
            inference_error = float((original.logits[0, -1].float().cpu()-issued).abs().max())
            inference_ce = float(F.cross_entropy(original.logits[0, -1].float()[None],
                                                torch.tensor([label], device=device)))
            del original
            for shadow in shadows.values():
                shadow.current = None
                shadow.begin(learner.optimizer.param_groups[1], query_limit=record.inputs.shape[1])
            with ExitStack() as stack:
                for i, shadow in shadows.items():
                    stack.enter_context(shadow.attach(learner.model, layer=i, capture=False,
                                                      clock_offset=record.clock_offset))
                output = learner.model(record.inputs.to(device), past_key_values=clone_cache(cache), use_cache=True)
            logits = output.logits[0, -1].float()
            error = float((logits.detach().cpu()-record.logits.float()).abs().max())
            ce = F.cross_entropy(logits[None], torch.tensor([label], device=device))
            original_ce = float(F.cross_entropy(record.logits.float()[None], torch.tensor([label])))
            if error > .125 or abs(float(ce.detach())-original_ce) > .02:
                raise RuntimeError('held prediction must match actual issuance before applying feedback: '
                    f'seen={record.prediction_seen}, length={record.inputs.shape[1]}, '
                    f'inference_logit_error={inference_error}, inference_ce={inference_ce}, '
                    f'issued_ce={original_ce}, differentiable_logit_error={error}, '
                    f'differentiable_ce={float(ce.detach())}')
            regs, reads = [], {}
            for i, shadow in shadows.items():
                m, read = shadow.memory, shadow.last_read
                regs.append(live_sigreg(shadow.encoder(shadow.current['source'][0].float().cpu()),
                    shadow.encoder(m.history), m.history_valid, learner.directions))
                positions = torch.tensor(shadow.current['query_positions']) + record.clock_offset
                valid = m.occupied[read.slots]
                ages = (m.clock+positions[:, None]-m.birth[read.slots])[valid]
                reads[str(i+1)] = dict(queries=len(positions),
                    source_ages=ages.tolist(), old_participations=int((ages>128).sum()),
                    slots=read.slots.detach().tolist(),
                    versions=m.read_credit_version[read.slots].tolist())
            reg = torch.stack(regs).mean()
            (10*ce/count + .001*reg.to(device)/count).backward()
            for i, shadow in shadows.items():
                receipts[i].extend(shadow.memory._read_participation)
            rows.append(dict(emission_index=emission_index,emitted=record.emitted, teaching_target=int(label),
                ce=float(ce.detach()), lived_ce=original_ce,
                replay_logit_max_difference=error, sigreg=float(reg.detach()), reads=reads,
                inference_logit_max_difference=inference_error, inference_ce=inference_ce,
                decision_seen=record.prediction_seen,
                feedback_delay_tokens=feedback_seen-record.prediction_seen))
            for shadow in shadows.values():
                release_view(shadow)
            del output, logits, ce, reg, regs, shadows, cache
    norms = {n: None if p.grad is None else float(p.grad.float().norm()) for n,p in learner.native}
    gradients = ({n: (torch.zeros_like(p, dtype=torch.float32, device='cpu') if p.grad is None else
                      p.grad.detach().float().cpu().clone()) for n,p in learner.native}
                 if return_native_gradients else None)
    if learner.cache is not live or live._seen_tokens != lived_seen or learner.tokens != lived_tokens:
        raise AssertionError('delayed settlement changed actual history')
    credit = {}
    if apply:
        for i, wrapper in learner.memories.items():
            m = wrapper.memory
            if m._read_credit_active or m._read_participation:
                raise RuntimeError('cannot mix independent pending teaching batches')
            configure_read_credit(m, 'adam', learner.optimizer.param_groups[1])
            m.begin_read_credit()
            m._read_participation = receipts[i]
            credit[str(i+1)] = m.settle_read_credit()
        learner.apply_gradients()
        learner.updates += 1
        pending.consumed = True
    else:
        learner.optimizer.zero_grad(set_to_none=True)
        for _, parameter in learner.native:
            parameter.grad = None
    result = dict(supervised_positions=count, loss_ce=sum(r['ce'] for r in rows)/count,
        rows=rows, native_gradient_norms=norms, read_credit=credit,
        pending_cpu_bytes=pending.nbytes, updated=apply,
        live_seen=lived_seen, live_tokens=lived_tokens,
        replay_logit_max_difference=max(r['replay_logit_max_difference'] for r in rows))
    if gradients is not None:
        result['_native_gradients'] = gradients
    return result
