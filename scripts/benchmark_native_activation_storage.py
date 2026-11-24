"""Measure exact storage/recompute paths on a saved body's trained 32-layer backbone.

No optimizer update, world call, or checkpoint write. Full-loop equivalence is
covered separately by test_activation_runtime; this isolates GPU computation/I/O.
"""
import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.world_source import decoded_world_bytes
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.native_policy_activation_storage import activation_storage, MIB


def benchmark_table(data, directory):
    """Same real bytes, row updates and returned probabilities in RAM and mmap."""
    import tempfile
    import shutil
    ram = RetrievedCreditTable()
    work = Path(tempfile.mkdtemp(prefix='ngram_', dir=directory))
    disk = RetrievedCreditTable(storage=work)
    tables = (ram, disk)
    timings = []
    try:
        # Compile the CPU kernel before either measurement.
        for table in tables:
            table.run_credit(data[:17])
        for table in tables:
            started = time.perf_counter()
            digest = hashlib.sha256()
            for start in range(0, len(data)-1, 512):
                nll, probs, packet = table.run_credit(data[start:min(start+513, len(data))])
                digest.update(nll.tobytes())
                digest.update(probs.tobytes())
                del packet
            timings.append(dict(seconds=time.perf_counter()-started, output_sha256=digest.hexdigest()))
        assert timings[0]['output_sha256'] == timings[1]['output_sha256']
        for name in ram._array_specs(len(ram.count)):
            a, b = getattr(ram, name), getattr(disk, name)
            for start in range(0, len(a), 4096):
                assert np.array_equal(a[start:start+4096], b[start:start+4096])
        disk.flush()
        return dict(bytes=len(data)-1, table_mib=ram.nbytes/MIB,
                    ram=timings[0], mmap=timings[1], arrays_equal=True)
    finally:
        # Close this benchmark's private maps before removing only its workspace.
        for name in disk._array_specs(len(disk.count)):
            getattr(disk, name)._mmap.close()
        assert work.resolve().parent == directory.resolve() and work.name.startswith('ngram_')
        shutil.rmtree(work)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lengths', type=int, nargs='+', default=[128, 480])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--ngram', action='store_true')
    args = p.parse_args()
    if args.repeats < 1 or min(args.lengths) < 1:
        p.error('positive lengths and repeat count required')
    torch.set_num_threads(1)
    with gzip.open(args.checkpoint, 'rb') as f:
        saved = torch.load(f, map_location='cpu', weights_only=False)
    spec = saved['config']['native_body']['backbone']
    model = GatedDeltaNetLM(**spec).cuda()
    prefix = 'body.base.'
    weights = {k[len(prefix):]: v for k, v in saved['model'].items() if k.startswith(prefix)}
    model.load_state_dict(weights, strict=True)
    cycle = saved['cycle']
    del saved, weights
    gc.collect()
    gdn_fused.enable()
    root = Path(__file__).resolve().parents[1]
    offset = json.loads((root/'data/ib_owt_10000/tokenizer.json').read_text())['offset']
    raw = np.load(root/'data/ib_owt_10000/train.npy', mmap_mode='r')
    requested = max(max(args.lengths)+1, 8193 if args.ngram else 0)
    data = decoded_world_bytes(raw[8000000:8000000+4*requested], offset)[:requested]
    del raw
    result = dict(checkpoint=str(args.checkpoint.resolve()), cycle=cycle,
                  gpu=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.parameters()),
                  backbone=spec, optimizer_updates=0, world_calls=0, rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.ngram:
        result['ngram'] = benchmark_table(data, args.output.parent)
        print(json.dumps(dict(ngram=result['ngram'])), flush=True)
    for length in args.lengths:
        x = torch.tensor(data[:length].astype(np.int64), device='cuda')[None]
        y = torch.tensor(data[1:length+1].astype(np.int64), device='cuda')[None]
        initial = model.initial_state(1)
        reference = None
        for name, block, mode, gpu_mib in [('full_states', 1, 'device', 0),
                ('block16_device', 16, 'device', 0), ('block16_disk', 16, 'disk', 0),
                ('block16_tiered128', 16, 'tiered', 128), ('block16_tiered', 16, 'tiered', 512)]:
            for layer in model.layers:
                layer._gdn_state_block = block
            samples = []
            for repeat in range(args.repeats+1):
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with activation_storage(model.parameters(), args.output.parent/'spill', mode=mode,
                                        gpu_budget_bytes=gpu_mib*MIB) as (receipt, hooks):
                    with hooks():
                        loss, state, diagnostics = model(x, y, initial)
                    loss.backward()
                    torch.cuda.synchronize()
                    seconds = time.perf_counter()-started
                    peak = torch.cuda.max_memory_allocated()
                    value = float(loss.detach())
                # Compare every trained parameter gradient and final state,
                # outside timing and outside the measured GPU peak.
                gradients = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
                             if p.grad is not None}
                final = state.detach().cpu()
                if reference is None:
                    reference = (value, final.clone(), gradients)
                assert abs(reference[0]-value) < 1e-6
                torch.testing.assert_close(final, reference[1], atol=2e-5, rtol=2e-4)
                max_error = max(float((gradients[k]-reference[2][k]).abs().max()) for k in gradients)
                for k in gradients:
                    torch.testing.assert_close(gradients[k], reference[2][k], atol=5e-5, rtol=5e-4, msg=k)
                if repeat:
                    samples.append(dict(seconds=seconds, cuda_peak_bytes=peak, loss=value,
                                        gradient_max_abs_error=max_error, storage=dict(receipt)))
                del loss, state, diagnostics, gradients, final
            row = dict(length=length, name=name, block=block, mode=mode,
                       seconds_median=statistics.median(s['seconds'] for s in samples),
                       peak_mib=max(s['cuda_peak_bytes'] for s in samples)/MIB, samples=samples)
            result['rows'].append(row)
            args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
            print(json.dumps({k: v for k, v in row.items() if k != 'samples'}), flush=True)
        del reference, x, y, initial
    digest = hashlib.sha256()
    with args.checkpoint.open('rb') as f:
        for chunk in iter(lambda: f.read(MIB), b''):
            digest.update(chunk)
    result['checkpoint_sha256'] = digest.hexdigest()
    result['completed'] = True
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
