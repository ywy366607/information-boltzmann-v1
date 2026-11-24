"""Bind the real closed-loop learning archive to NativeLearning.lean.

No model construction, training step, world request, or checkpoint write.
Uses all completed observer segments. A metadata-only FakeTensor load identifies
optimizer settings and reads only the existing pending momentum-read tensors.
The adapter derivative probe is explicitly an interface test, not a new loss
applied to the individual and not a behavioral or credit-usefulness experiment.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ib_local.adaptive_hamiltonian import MomentumReadBridge


def read_array(handle, descriptor):
    ref = descriptor['tensor_ref']
    handle.seek(ref['offset'])
    raw = zlib.decompress(handle.read(ref['compressed_bytes']))
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('recorded tensor checksum mismatch')
    return np.broadcast_to(np.frombuffer(raw, dtype=ref['dtype']).reshape(
        ref['stored_shape']), ref['shape']).astype(np.float64)


def checkpoint_metadata_and_read(path):
    with gzip.open(path, 'rb') as stream:
        with FakeTensorMode():
            saved = torch.load(stream, map_location='cpu', weights_only=False)
        groups = saved['optimizer']['param_groups']
        # configure() appends precisely the dedicated theta group; check shape.
        group = groups[-1]
        if len(group['params']) != 1:
            raise ValueError('dedicated current dynamics optimizer group is ambiguous')
        parameter = group['params'][0]
        if tuple(saved['optimizer']['state'][parameter]['exp_avg'].shape) != (32, 12):
            raise ValueError('dedicated optimizer state does not match dynamics theta')
        settings = {k: v for k, v in group.items() if k != 'params'}
        pending = saved['engine']['internal_pending']
        fake_e = pending['momentum_read_eligibility']
        fake_read = pending['momentum_read']
        fake_live = saved['model']['head_coupling.sensitivity_p']
        def offset(x):
            return x.untyped_storage()._checkpoint_offset
        alias = offset(fake_e) == offset(fake_live)
        requests = sorted((offset(x), label, x) for label, x in
                          (('eligibility', fake_e), ('read', fake_read)))
        values, byte_count = {}, 0
        for start, label, fake in requests:
            size = fake.untyped_storage().nbytes()
            stream.seek(start)
            raw = bytearray(stream.read(size))
            if len(raw) != size:
                raise EOFError('incomplete selected checkpoint tensor')
            whole = torch.frombuffer(raw, dtype=fake.dtype)
            values[label] = whole.as_strided(tuple(fake.shape), tuple(fake.stride()),
                                            fake.storage_offset()).clone()
            byte_count += size
        return settings, values, dict(cycle=saved['cycle'], selected_storage_bytes=byte_count,
            pending_shape=list(fake_e.shape), pending_aliases_live_trace=alias,
            live_trace_shape=list(fake_live.shape), all_weights_materialized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task_root', type=Path)
    parser.add_argument('lean_output', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    settings, saved_read, selected = checkpoint_metadata_and_read(args.task_root/'body/last.pt.gz')
    beta1, beta2 = settings['betas']
    lr, decay, eps = settings['lr'], settings['weight_decay'], settings['eps']
    receipts, row_reports, sources = [], [], []
    stats = collections.Counter()
    maximum_error = maximum_ulp = 0.0
    gate_values = []
    for directory in sorted((args.task_root/'work').glob('collect_observations*')):
        archive = directory/'operations.jsonl.gz'
        with gzip.open(archive, 'rt', encoding='utf8') as stream:
            rows = [json.loads(line) for line in stream]
        sources.append(dict(path=str(archive), sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
        with (directory/'tensors.bin').open('rb') as handle:
            for row in rows:
                data = row['data']
                if row['kind'] == 'optimizer':
                    stats['optimizer_records'] += 1
                    before = read_array(handle, data['theta_before'])
                    after = read_array(handle, data['theta_after'])
                    if data['actual_gradient'] is None:
                        stats['theta_no_gradient'] += 1
                        if not np.array_equal(before, after):
                            raise AssertionError('parameter without gradient changed')
                        continue
                    stats['theta_gradient'] += 1
                    first = read_array(handle, data['exp_avg'])
                    second = read_array(handle, data['exp_avg_sq'])
                    count = int(read_array(handle, data['step']).item())
                    gate = read_array(handle, data['budget_gate'])
                    proposal = (first/(1-beta1**count))/(np.sqrt(second/(1-beta2**count))+eps)
                    expected = before-lr*gate*(proposal+decay*before)
                    error = abs(after-expected)
                    ulp = abs(np.spacing(before.astype(np.float32)).astype(np.float64))
                    maximum_error = max(maximum_error, float(error.max()))
                    maximum_ulp = max(maximum_ulp, float((error/ulp).max()))
                    gate_values.append(float(gate))
                    if float(gate) == 0:
                        stats['theta_zero_gate'] += 1
                        if not np.array_equal(before, after):
                            raise AssertionError('closed budget gate failed to hold the parameter')
                    if (error > 4*ulp).any():
                        raise AssertionError('observed write does not match real-arithmetic Adam/budget formula')
                elif row['kind'] == 'event' and data['autonomous_event']['kind'] == 'external':
                    stats['external_events'] += 1
                    cp, unknown = data['causal_program'], data['shared_unknown']
                    revision = unknown['revision']
                    committed, merged = cp['evidence_committed'], data['episode_merged']
                    stats['primary_commits' if committed else 'primary_rejections'] += 1
                    stats['merged_episodes'] += int(merged)
                    before_version = cp['roles']['version']
                    after_version = cp['stored_version']
                    unknown_before = revision.get('version_before', unknown['version'])
                    unknown_after = (f'some {revision["version_after"]}' if revision['committed'] else 'none')
                    current, delayed = data['momentum_current_credit_samples'], data['momentum_delayed_credit_samples']
                    stats['current_read_credit_samples'] += current
                    stats['delayed_read_credit_samples'] += delayed
                    event = data['autonomous_event']['event']
                    receipts.append(f'⟨{event}, {str(merged).lower()}, {before_version}, {after_version}, '
                        f'{str(committed).lower()}, {str(revision["committed"]).lower()}, '
                        f'{unknown_before}, {unknown_after}, {current}, {delayed}⟩')
                    row_reports.append(dict(event=event, cycle=row['external_cycle'],
                        primary_committed=committed, merged=merged, current=current, delayed=delayed))
    # A probe cotangent of the real saved read checks the adapter's exact
    # contraction. No claim that this is the individual's later prediction error.
    e, read = saved_read['eligibility'], saved_read['read']
    theta = torch.zeros((32, 12), requires_grad=True)
    class Receipt:
        last_momentum_gradient = torch.zeros_like(theta)
        momentum_credit_count = torch.zeros((), dtype=torch.long)
    adapter = Receipt()
    value = read.detach().clone().requires_grad_(True)
    bridged = MomentumReadBridge.apply(value, theta, e, adapter)
    (bridged*read).sum().backward()
    expected_credit = np.einsum('ld,pld->lp', read.double().numpy().reshape(32, 256),
                                e.double().numpy())
    credit_error = float(abs(theta.grad.numpy()-expected_credit).max())
    credit_scale = float(abs(expected_credit).max())
    if credit_error > 5e-6*(1+credit_scale) or selected['pending_aliases_live_trace']:
        raise AssertionError('saved delayed-read bridge does not match its abstraction')
    reference = ROOT/'lean/CGBAxioms/reviews/NativeLearning.lean'
    text = reference.read_text(encoding='utf8') + '''

namespace CGB.NativeLearning.Actual2401
set_option linter.style.longLine false
def receipts : List LearningReceipt := [
''' + ',\n'.join(receipts) + ''']
theorem actual_learning_receipts_valid :
    receipts.all learningReceiptValid = true := by native_decide
end CGB.NativeLearning.Actual2401
'''
    args.lean_output.write_text(text, encoding='utf8')
    result = dict(status='numeric_checks_passed_lean_not_yet_checked', sources=sources,
        stats=dict(stats), optimizer_settings=settings, selected_checkpoint=selected,
        adam_budget_formula_max_abs_error=maximum_error, adam_budget_formula_max_fp32_ulp=maximum_ulp,
        gate_min=min(gate_values), gate_max=max(gate_values),
        pending_read_bridge=dict(absolute_error=credit_error, gradient_scale=credit_scale,
            nonzero_eligibility_entries=int(torch.count_nonzero(e)),
            probe='interface cotangent = saved read; not a real outcome or usefulness score'),
        lean_output=str(args.lean_output),
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(), events=row_reports,
        model_updates=0, world_calls=0)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','events')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
