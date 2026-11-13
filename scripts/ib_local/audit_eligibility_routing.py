"""Diagnostic-only control of old participation/teaching correspondence.

Forward values, live encoders, current-state derivatives and retrieval receipts
stay unchanged. Only teaching vectors entering reexecuted native old responses
are permuted. No control is installed in a saved production body.
"""
from contextlib import contextmanager
import types

import torch


class RouteTeaching(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, permutation):
        ctx.save_for_backward(permutation.argsort())
        return values.clone()

    @staticmethod
    def backward(ctx, teaching):
        inverse, = ctx.saved_tensors
        return teaching.index_select(0, inverse), None


class RouteToOtherParticipation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, observed_values, other_participation):
        return observed_values.clone()

    @staticmethod
    def backward(ctx, teaching):
        return None, teaching


def old_permutation(ages, cutoff, *, exchange):
    permutation = torch.arange(len(ages), dtype=torch.long)
    old = [i for i, age in enumerate(ages) if age > cutoff]
    if exchange and len(old) > 1:
        for position, index in enumerate(old):
            permutation[index] = old[(position+1) % len(old)]
    return permutation


@contextmanager
def eligibility_routing(learner, *, exchange=False, cutoff=128, other_participation=False):
    """Temporarily audit existing old responses, without modifying the model."""
    receipts, originals = [], []
    for layer, wrapper in learner.memories.items():
        originals.append((wrapper, wrapper.retrieve))

        def retrieve(self, query, replay, *, _layer=layer):
            read = self.read(query)
            self.last_read = read
            slots = read.slots[0].tolist()
            if not bool(self.memory.occupied[slots].all()):
                raise ValueError('this paired audit requires occupied selected real histories')
            responses = []
            for slot in slots:
                capsule = self.capsules[slot]
                if capsule is None or capsule.version != int(self.memory.read_credit_version[slot]):
                    raise RuntimeError('qualification identity no longer matches its capsule')
                response = replay(capsule)[:, -1]
                responses.append(response.detach() if self.detach_replay else response)
            self.replayed.extend(responses)
            ages = (self.memory.clock-self.memory.birth[slots]).tolist()
            permutation = old_permutation(ages, cutoff, exchange=exchange)
            receivers = list(slots)
            if other_participation and not self.detach_replay:
                available = [i for i in range(self.memory.capacity)
                    if i not in slots and bool(self.memory.occupied[i])
                    and int(self.memory.clock-self.memory.birth[i]) > cutoff]
                for j,slot in enumerate(slots):
                    if ages[j] <= cutoff:
                        continue
                    if not available:
                        raise ValueError('no unused old participation available for the control')
                    preferred = (slot+self.memory.capacity//2) % self.memory.capacity
                    chosen = min(available,key=lambda s:(s-preferred) % self.memory.capacity)
                    receivers[j] = chosen
                    available.remove(chosen)
            receipt = dict(layer=_layer, slots=slots, ages_tokens=ages,
                versions=self.memory.read_credit_version[slots].tolist(),
                permutation=permutation.tolist(),
                exchanged_entries=int((permutation != torch.arange(len(slots))).sum()),
                receiver_slots=receivers,
                receiver_ages_tokens=(self.memory.clock-self.memory.birth[receivers]).tolist(),
                wrong_receiver_count=sum(a != b for a,b in zip(slots,receivers)))
            values = torch.stack(responses)
            if receivers != slots:
                other = []
                for original,receiver,response in zip(slots,receivers,responses):
                    capsule = self.capsules[receiver]
                    if capsule is None or capsule.version != int(self.memory.read_credit_version[receiver]):
                        raise RuntimeError('foreign qualification identity mismatch')
                    other.append(response if receiver == original else replay(capsule)[:, -1])
                credited_values = torch.stack(other)
                routed = RouteToOtherParticipation.apply(values, credited_values)
            else:
                credited_values = values
                routed = RouteTeaching.apply(values, permutation.to(values.device))
            if credited_values.requires_grad:
                def save_teaching(gradient, *, _receipt=receipt, _name='received'):
                    _receipt[_name] = gradient.detach().float().cpu().clone()
                credited_values.register_hook(save_teaching)
                routed.register_hook(lambda g, _r=receipt: save_teaching(g, _receipt=_r, _name='sent'))
            receipts.append(receipt)
            encoded = [self.encoder(response.float().cpu()) for response in routed.unbind(0)]
            weights = read.weights
            if self.suppress_older_than is not None:
                age = self.memory.clock-self.memory.birth[read.slots]
                weights = weights*(age <= self.suppress_older_than)
            return (torch.stack(encoded, dim=1)*weights[..., None]).sum(1)

        wrapper.retrieve = types.MethodType(retrieve, wrapper)
    try:
        yield receipts
    finally:
        for wrapper, original in originals:
            wrapper.retrieve = original


def receipt_summary(receipts):
    result = []
    for receipt in receipts:
        row = {key:value for key,value in receipt.items() if key not in ('sent','received')}
        sent, received = receipt.get('sent'), receipt.get('received')
        if sent is not None and received is not None:
            sent, received = sent.double().flatten(1), received.double().flatten(1)
            expected = sent[torch.tensor(receipt['permutation']).argsort()]
            row.update(sent_norm=float(sent.norm()), received_norm=float(received.norm()),
                correspondence_exact=torch.equal(expected, received),
                changed_teaching_norm=float((sent-received).norm()),
                per_record_sent_norm=sent.norm(dim=1).tolist(),
                per_record_received_norm=received.norm(dim=1).tolist())
        result.append(row)
    return result
