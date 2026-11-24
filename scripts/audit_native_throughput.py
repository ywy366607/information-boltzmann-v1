"""Compare complete continuations at their functional and carried-state ports."""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import io
import json
from pathlib import Path
import time
import types

import numpy as np
import torch


def load(path):
    # PyTorch's zip reader seeks backwards. Inflating the 0.87 GB stream once
    # avoids repeated gzip decompression during this serial CPU-only audit.
    buffer = io.BytesIO()
    with gzip.open(path, 'rb') as stream:
        while data := stream.read(8 * 1024**2):
            buffer.write(data)
    buffer.seek(0)
    return torch.load(buffer, map_location='cpu', weights_only=False)


def compare(a, b):
    result = dict(leaves=0, different=0, shape_differences=[], max_abs=0., examples=[])

    def walk(x, y, path):
        if isinstance(x, np.random.Generator) and isinstance(y, np.random.Generator):
            walk(x.bit_generator.state, y.bit_generator.state, path+'/bit_generator')
            return
        if isinstance(x, torch.Generator) and isinstance(y, torch.Generator):
            walk(x.get_state(), y.get_state(), path+'/generator_state')
            return
        if dataclasses.is_dataclass(x) and type(x) is type(y):
            for field in dataclasses.fields(x):
                walk(getattr(x, field.name), getattr(y, field.name), f'{path}/{field.name}')
            return
        if isinstance(x, (torch.Tensor, np.ndarray)) and isinstance(y, (torch.Tensor, np.ndarray)):
            x, y = torch.as_tensor(x), torch.as_tensor(y)
            if x.shape != y.shape or x.dtype != y.dtype:
                result['shape_differences'].append(path)
                return
            same = torch.equal(x, y)
            error = float((x.double()-y.double()).abs().max()) if x.numel() and not same else 0.
            result['max_abs'] = max(result['max_abs'], error)
        elif isinstance(x, dict) and isinstance(y, dict):
            for key in x.keys() | y.keys():
                walk(x.get(key), y.get(key), f'{path}/{key}')
            return
        elif isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)):
            if len(x) != len(y):
                result['shape_differences'].append(path)
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f'{path}/{i}')
            return
        elif (type(x) is type(y) and hasattr(x, '__dict__') and
              not isinstance(x, (type, types.FunctionType, types.MethodType))):
            # Serialized Running/ActionEvidence state is compared by contents,
            # rather than the identity of two freshly unpickled instances.
            walk(vars(x), vars(y), path+'/state')
            return
        else:
            same = x == y
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                result['max_abs'] = max(result['max_abs'], abs(x-y))
        result['leaves'] += 1
        if not same:
            result['different'] += 1
            if len(result['examples']) < 12:
                result['examples'].append(path)

    walk(a, b, '')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timing-reference', type=Path,
                        help='same-environment profile when an identical full state is retained elsewhere')
    args = parser.parse_args()
    started = time.perf_counter()
    result = dict(reference=str(args.reference), candidate=str(args.candidate))
    rows = []
    for work in (args.reference, args.candidate):
        rows.append([json.loads(line) for line in (work/'events.jsonl').read_text(encoding='utf8').splitlines()
                     if json.loads(line).get('autonomous_event', {}).get('kind') == 'external'])
    ports = ('cycle', 'choice', 'utterance_hex', 'external_bytes', 'self_bytes', 'updates',
             'metal_g', 'response_nll', 'continuation_nll', 'world', 'pre_outcome_gain',
             'joint_language_search', 'shared_unknown')
    result['actual_ports'] = {key: compare([r.get(key) for r in rows[0]],
                                         [r.get(key) for r in rows[1]]) for key in ports}
    a, b = (load(work/'body/last.pt.gz') for work in (args.reference, args.candidate))
    result['checkpoint'] = {key: compare(a[key], b[key]) for key in
        ('model', 'optimizer', 'table', 'world', 'torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng', 'cycle')}
    carried = ('recurrent', 'bfss_momenta', 'internal_pending', 'causal_use_pending',
               'native_value_bank', 'latent_action_bank', 'autonomous_eligibility')
    result['carried'] = {key: compare(a['engine'].get(key), b['engine'].get(key)) for key in carried}
    result['old_fact_scores'] = compare(
        json.loads((args.reference/'evaluations.json').read_text(encoding='utf8')),
        json.loads((args.candidate/'evaluations.json').read_text(encoding='utf8')))
    timing_reference = args.timing_reference or args.reference
    timings = [json.loads((work/'runtime_profile.json').read_text()) for work in
               (timing_reference, args.candidate)]
    result['timing'] = dict(reference_seconds=timings[0]['seconds'], candidate_seconds=timings[1]['seconds'],
        reference_profile=str(timing_reference/'runtime_profile.json'),
        speedup=timings[0]['seconds']/timings[1]['seconds'],
        candidate_peak_gpu_bytes=timings[1]['cuda_peak_allocated_bytes'])
    result['audit_seconds'] = time.perf_counter()-started
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
