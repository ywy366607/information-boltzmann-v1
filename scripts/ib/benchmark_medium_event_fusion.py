"""Execution equivalence and timing on a saved real-data medium individual."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.execution_cache import configure_execution_cache
from information_boltzmann.runtime.gpu_memory import total_gpu_memory
from information_boltzmann.runtime.medium_health import ChunkHealthCapture
from information_boltzmann.runtime.optimization import make_medium_optimizer
from information_boltzmann.runtime.training import belief_tensors, quiet_training_chunk
from scripts.ib.train_plastic_conductance import unpack_belief


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.75)
    configure_execution_cache('scratch/compiler_cache')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    model = PlasticMediumPorts3D(**config['constructor']).cuda().train()
    model.load_state_dict(saved['model'])
    model.source.embedding.weight.requires_grad_(False)
    belief = unpack_belief(saved['belief'], 'cuda')
    cursor = saved['cursor']
    del saved
    gc.collect()
    data = np.load('data/ib_owt_gpt2/train.npy', mmap_mode='r')
    ids = torch.from_numpy(np.array(data[cursor-1:cursor+31], dtype=np.int64)).cuda()[None]
    targets = torch.from_numpy(np.array(data[cursor:cursor+32], dtype=np.int64)).cuda()[None]
    health = ChunkHealthCapture(model, 32)
    parameters = [p for p in model.parameters() if p.requires_grad]

    def backward(compiled):
        model.zero_grad(set_to_none=True)
        output = quiet_training_chunk(model, ids, targets, belief,
            event_duration=config['event_duration'], health_capture=health,
            return_token_nll=True, activation_checkpointing=True, compile_event=compiled)
        output[0].backward()
        torch.cuda.synchronize()
        return output

    report = {'purpose': 'execution equivalence and speed; no capability training',
              'checkpoint': str(args.checkpoint), 'tokens_per_update': 32}
    print('Reference backward', flush=True)
    reference = backward(False)
    scores = reference[3].detach().cpu()
    states = [x.detach().cpu() for x in belief_tensors(reference[1])]
    gradients = [None if p.grad is None else p.grad.detach().cpu() for p in parameters]
    audit = [x.detach().cpu().clone() for x in
             (health.event_values, health.event_features, health.event_decode)]
    del reference
    print('Compiling full event and backward', flush=True)
    started = time.perf_counter()
    actual = backward(True)
    report['compile_and_first_backward_seconds'] = time.perf_counter() - started
    torch.testing.assert_close(actual[3].detach().cpu(), scores, atol=3e-5, rtol=3e-5)
    for x, y in zip(belief_tensors(actual[1]), states):
        torch.testing.assert_close(x.detach().cpu(), y, atol=3e-5, rtol=3e-4)
    for x, y in zip((health.event_values, health.event_features, health.event_decode), audit):
        torch.testing.assert_close(x.detach().cpu(), y, atol=3e-5, rtol=3e-4)
    squared_error = reference_energy = actual_energy = dot = 0.
    for p, expected in zip(parameters, gradients):
        assert (p.grad is None) == (expected is None)
        if expected is not None:
            actual_gradient = p.grad.detach().cpu()
            squared_error += float(torch.linalg.vector_norm(actual_gradient - expected)) ** 2
            reference_energy += float(torch.linalg.vector_norm(expected)) ** 2
            actual_energy += float(torch.linalg.vector_norm(actual_gradient)) ** 2
            dot += float((actual_gradient.double().flatten() @ expected.double().flatten()))
    report['gradient_relative_l2_error'] = (squared_error / reference_energy) ** .5
    report['gradient_cosine'] = dot / (reference_energy * actual_energy) ** .5
    assert report['gradient_relative_l2_error'] < 1e-3
    del actual, states, gradients, audit
    gc.collect()
    optimizer = make_medium_optimizer(model, lr=2e-4)
    for compiled in (False, True):
        samples = []
        for index in range(3):
            started = time.perf_counter()
            result = backward(compiled)
            optimizer.step()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - started)
            del result
        report['compiled' if compiled else 'eager'] = {
            'seconds_per_32_tokens': samples[1:],
            'tokens_per_second': 64 / sum(samples[1:])}
    report['memory'] = {'dedicated_mib': total_gpu_memory()['used_bytes'] / 2**20,
                        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
    assert report['memory']['dedicated_mib'] < 3900
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
