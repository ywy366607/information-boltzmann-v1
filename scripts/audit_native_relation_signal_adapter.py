"""Compare observer on/off on the actual checkpoint's relation operators.

Selective CPU extraction only. This is a numerical observer audit, not a
re-enactment of the full learning event or an assessment of memory usefulness.
No body, optimizer or world is constructed; all source files stay read-only.
"""
import argparse
import dataclasses
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ib_local import causal_relation_program as cp
from scripts.ib_local import joint_relation_update as jr
from scripts.ib_local.shared_relation_basis import SharedRelationBasis
from scripts.ib_local.audit_runtime_checkpoint_pair import digest
from scripts.native_relation_signals import install_relation_signals
from scripts.native_topology_observer import TensorStore, _json, load_tensor


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def selected_checkpoint(path):
    """Read each requested logical view, not its potentially huge storage."""
    with gzip.open(path, 'rb') as stream:
        with FakeTensorMode():
            saved = torch.load(stream, map_location='cpu', weights_only=False)
        engine = saved['engine']
        names = ('encoder.', 'long_predictor.', 'mother_from_boundary.',
                 'momentum_to_boundary.', 'relation_basis.')
        chosen = dict(weights={k: v for k, v in saved['model'].items() if k.startswith(names)},
            state=engine['_program_question_state'], source=engine['_internal_source'],
            actions=engine['_internal_action'], momentum=engine['_momentum_decision_read'],
            cycle=saved['cycle'], ridge=engine['relation_ridge'],
            revision=saved['config']['joint_relation_update']['implementation_revision'])
        tensors = {}

        def collect(value):
            if torch.is_tensor(value):
                if value.numel():
                    tensors[id(value)] = value
            elif dataclasses.is_dataclass(value):
                for field in dataclasses.fields(value):
                    collect(getattr(value, field.name))
            elif isinstance(value, dict):
                for v in value.values():
                    collect(v)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    collect(v)
        collect(chosen)
        requests = {}
        for tensor in tensors.values():
            if any(stride < 0 for stride in tensor.stride()):
                raise ValueError('negative view stride is unsupported')
            offset = tensor.untyped_storage()._checkpoint_offset
            start = offset+tensor.storage_offset()*tensor.element_size()
            span = (1+sum((n-1)*s for n, s in zip(tensor.shape, tensor.stride())))*tensor.element_size()
            requests[start, span] = None
        selected_bytes = sum(span for start, span in requests)
        if selected_bytes > 64*1024*1024:
            raise RuntimeError('selected view budget exceeds 64 MiB; refusing full storage load')
        for start, span in sorted(requests):
            stream.seek(start)
            raw = stream.read(span)
            if len(raw) != span:
                raise EOFError('incomplete selected tensor view')
            requests[start, span] = bytearray(raw)

        def concrete(value):
            if torch.is_tensor(value):
                if not value.numel():
                    return torch.empty(tuple(value.shape), dtype=value.dtype)
                offset = value.untyped_storage()._checkpoint_offset
                start = offset+value.storage_offset()*value.element_size()
                span = (1+sum((n-1)*s for n, s in zip(value.shape, value.stride())))*value.element_size()
                flat = torch.frombuffer(requests[start, span], dtype=value.dtype)
                return flat.as_strided(tuple(value.shape), tuple(value.stride())).clone()
            if dataclasses.is_dataclass(value):
                return type(value)(**{f.name: concrete(getattr(value, f.name))
                                     for f in dataclasses.fields(value)})
            if isinstance(value, dict):
                return {k: concrete(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(concrete(v) for v in value)
            return value
        print(json.dumps(dict(stage='selected', bytes=selected_bytes, views=len(requests))), flush=True)
        return concrete(chosen), selected_bytes


class Recorder:
    def __init__(self, path):
        self.records, self.undo = [], []
        self.tensors = TensorStore(path)

    def record(self, kind, data):
        self.records.append(_json(data, self.tensors))

    def _replace(self, owner, name, replacement):
        self.undo.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def close(self):
        for owner, name, fn in reversed(self.undo):
            setattr(owner, name, fn)
        self.tensors.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    before = sha(args.checkpoint)
    print(json.dumps(dict(stage='indexing', checkpoint_sha256=before)), flush=True)
    selected, selected_bytes = selected_checkpoint(args.checkpoint)
    sd, state = selected['weights'], selected['state']
    width, latent = sd['encoder.weight'].shape[1], sd['encoder.weight'].shape[0]
    with torch.random.fork_rng(devices=[]):
        encoder = nn.Linear(width, latent, bias=False)
        basis = SharedRelationBasis(latent)
    encoder.load_state_dict({'weight': sd['encoder.weight']})
    basis.load_state_dict({k.removeprefix('relation_basis.'): v for k, v in sd.items()
                           if k.startswith('relation_basis.')})

    def linear(name, x):
        return F.linear(x, sd[name+'.weight'], sd.get(name+'.bias'))

    model = SimpleNamespace(encoder=encoder, relation_basis=basis,
        long_predictor=lambda x: linear('long_predictor.2', F.silu(linear('long_predictor.0', x))),
        mother_from_boundary=lambda x: linear('mother_from_boundary', x),
        momentum_to_boundary=lambda x: linear('momentum_to_boundary', x),
        joint_relation_update_schema=1, joint_relation_update_revision=selected['revision'],
        causal_relation_ridge=selected['ridge'], memory=SimpleNamespace(
            source=torch.empty(0, width), action=torch.empty(0, width),
            mother_next=torch.empty(0, width), internal_next=torch.empty(0, width)))
    if not state.evidence_bound:
        raise ValueError('actual pending relation is not self-contained canonical evidence')
    evidence = next(e for e in state.constraints if e.kind == cp.Port.DECISION)
    if state.extra:
        raise ValueError('pending relation already has a temporary branch; preserve it rather than replace')
    actions = selected['actions']
    source = selected['source'].reshape(1, width).expand_as(actions)
    momentum = selected['momentum'].reshape(1, -1).expand(len(actions), -1)
    named_parameters = {'encoder.weight': encoder.weight,
                        **{'relation_basis.'+k: v for k, v in basis.named_parameters()}}
    for name, tensor in sd.items():
        if name.startswith(('long_predictor.', 'mother_from_boundary.', 'momentum_to_boundary.')):
            tensor.requires_grad_(True)
            named_parameters[name] = tensor
    parameters = list(named_parameters.values())
    input_state_hash = digest(dataclasses.asdict(state))
    parameter_hash = digest({k: p.detach() for k, p in named_parameters.items()})

    def run():
        mother = cp.mechanism(model, state, source, actions, ridge=selected['ridge'], momentum=momentum)
        direct = cp.mechanism(model, state, source, actions, kind=cp.Port.DECISION,
                              ridge=selected['ridge'], momentum=momentum)
        mediated = cp.mechanism(model, state, source, actions, kind=cp.Port.DECISION,
                                mother=mother['mean'], ridge=selected['ridge'])
        consensus = jr.overlap_consensus((mediated, direct), state.program_weights)
        # A previously observed real target is used only to compare the two
        # derivatives. This is not a new prequential score or a new exposure.
        target = encoder(evidence.outcome).detach().expand_as(consensus['mean'])
        score = cp.log_score(consensus, target)
        grads = torch.autograd.grad(score.sum(), parameters, allow_unused=True)
        revised = cp.revise(state, evidence, model=model)
        after = cp.mechanism(model, revised, source, actions, kind=cp.Port.DECISION,
                             ridge=selected['ridge'], momentum=momentum)
        return {k: v.detach().clone() for k, v in dict(mother=mother['mean'], direct=direct['mean'],
            mediated=mediated['mean'], consensus=consensus['mean'], score=score, revised=after['mean']).items()}, grads, revised

    rng = torch.random.get_rng_state().clone()
    expected, expected_grads, expected_state = run()
    if not torch.equal(rng, torch.random.get_rng_state()):
        raise AssertionError('reference relation call consumed RNG')
    with tempfile.TemporaryDirectory(prefix='cgb_relation_adapter_', dir=args.output.parent) as directory:
        tensor_path = Path(directory)/'tensors.bin'
        recorder = Recorder(tensor_path)
        try:
            install_relation_signals(recorder)
            actual, gradients, revised = run()
        finally:
            recorder.close()
        equality = {k: torch.equal(v, actual[k]) for k, v in expected.items()}
        grad_equal = {name: (a is b if a is None or b is None else torch.equal(a, b))
                      for name, a, b in zip(named_parameters, expected_grads, gradients)}
        message_errors, native_errors, base_subtraction_errors = [], [], []
        for record in recorder.records:
            if record['operation'] != 'posterior':
                continue
            p = record['prediction']
            refs = dict(x=record['design'], query=record['query'],
                        coefficient=p['coefficient'], mean=p['mean'])
            arrays32 = {k: torch.from_numpy(load_tensor(tensor_path, v).copy()) for k, v in refs.items()}
            arrays = {k: v.double() for k, v in arrays32.items()}
            correction = ((arrays['query']@arrays['x'].T)[:, :, None]
                          * arrays['coefficient'][None]).sum(1)
            message_errors.append(float((correction-arrays['mean']).abs().max()))
            native = (arrays32['query']@arrays32['x'].T)@arrays32['coefficient']
            native_errors.append(float((native-arrays32['mean']).abs().max()))
            parent = next(r for r in recorder.records if r['call_id'] == record['parent_id'])
            if parent['operation'] == 'mechanism':
                mean = torch.from_numpy(load_tensor(tensor_path, parent['prediction']['mean']).copy())
                base = torch.from_numpy(load_tensor(tensor_path, parent['prediction']['base']).copy())
                base_subtraction_errors.append(float((mean-base-arrays32['mean']).abs().max()))
        result = dict(cycle=selected['cycle'], selected_storage_bytes=selected_bytes,
            actual_relation=dict(identity=state.identity, generation=state.generation, version=state.version,
                                 binding=None if state.binding is None else dataclasses.asdict(state.binding)),
            observed_target_fact=evidence.fact_id, candidate_batch=len(actions),
            operation_counts={name: sum(r['operation'] == name for r in recorder.records)
                              for name in sorted({r['operation'] for r in recorder.records})},
            exact_output_equality=equality, exact_gradient_equality=grad_equal,
            rng_unchanged=torch.equal(rng, torch.random.get_rng_state()),
            functional_version_unchanged=revised.version == expected_state.version == state.version,
            temporary_evidence_equal=len(revised.extra) == len(expected_state.extra),
            complete_functional_state_equal=digest(dataclasses.asdict(revised)) == digest(dataclasses.asdict(expected_state)),
            original_relation_state_unchanged=digest(dataclasses.asdict(state)) == input_state_hash,
            parameters_unchanged=digest({k: p.detach() for k, p in named_parameters.items()}) == parameter_hash,
            evidence_message_reconstruction_max_error=max(message_errors),
            native_fp32_correction_reexecution_max_error=max(native_errors),
            final_mean_minus_base_vs_captured_correction_max_error=max(base_subtraction_errors),
            source_sha256=before, source_unchanged=sha(args.checkpoint) == before,
            all_weights_materialized=False, whole_event_equivalence=False,
            whole_event_note='This checks real current relation operators only, not an entire living/learning event.',
            role='observer numerical audit, not activity/synchronization/utility evidence',
            optimizer_updates=0, world_calls=0)
    if not (all(equality.values()) and all(grad_equal.values()) and result['rng_unchanged']
            and result['source_unchanged'] and result['complete_functional_state_equal']
            and result['original_relation_state_unchanged'] and result['parameters_unchanged']):
        raise AssertionError('observer changed an actual relation computation')
    if result['native_fp32_correction_reexecution_max_error'] != 0:
        raise AssertionError('captured actual posterior no longer reexecutes exactly')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: x.value), encoding='utf8')
    print(json.dumps(result, ensure_ascii=False, default=lambda x: x.value), flush=True)


if __name__ == '__main__':
    main()
