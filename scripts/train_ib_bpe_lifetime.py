"""Persistent GPT-2 BPE individual with CUDA Graph updates and held-out selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ib_bpe_window import BPEWindow, clock_inputs


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(.025)
    # Monitor readers/Windows scanners must not kill a training update.
    print(f'WARNING: deferred status replacement: {path}', file=sys.stderr, flush=True)


def atomic_save(path, data):
    tmp = path.with_suffix('.tmp')
    torch.save(data, tmp)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--particles', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--validate-every', type=int, default=250)
    parser.add_argument('--validation-tokens', type=int, default=4096)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=bool(args.resume))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['manifest_sha256'] = hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest()
    config['source_sha256'] = {p: hashlib.sha256((Path(__file__).parent / p).read_bytes()).hexdigest()
                               for p in ('train_ib_bpe_lifetime.py', 'ib_bpe_window.py', 'ib_fused_ou.py', 'ib_slice_collision.py')}
    config.update(seed=11, lr=3e-4, weight_decay=.01, validation_seed=99173,
                  burn_in_tokens=args.tokens, selection='minimum frozen held-out BPE NLL',
                  architecture='BPEWindow Slice Cayley, 4 Strang steps, local gamma, tied embedding')
    torch.set_num_threads(1)
    torch.manual_seed(11)
    torch.cuda.set_per_process_memory_fraction(.68)
    train = np.load(args.data / 'train.npy', mmap_mode='r')
    valid = np.load(args.data / 'validation.npy', mmap_mode='r')
    assert args.steps * args.tokens <= len(train), 'No implicit training stream wrap'
    assert args.validation_tokens % args.tokens == 0
    assert args.validation_tokens + args.tokens <= len(valid)
    model = BPEWindow(hidden=args.hidden, particles=args.particles).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=True, capturable=True)
    generator = torch.Generator(device='cuda').manual_seed(11)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    step, offset, best = 0, 0, float('inf')
    if args.resume:
        saved = torch.load(args.resume, map_location='cuda', weights_only=False)
        for key in ('manifest_sha256', 'tokens', 'particles', 'hidden', 'validation_tokens'):
            assert saved['config'][key] == config[key], key
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        x, v = saved['x'].detach().clone(), saved['v'].detach().clone()
        generator.set_state(saved['generator'].cpu())
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        step, offset, best = saved['step'], saved['events'], saved['best_validation_nll']
    else:
        with torch.cuda.stream(stream):
            state = model.core.initialize(torch.tensor([50256], device='cuda'), generator)
        torch.cuda.current_stream().wait_stream(stream)
        x, v = state.x, state.v
    assert offset == step * args.tokens
    ids = torch.empty(args.tokens, dtype=torch.long, device='cuda')
    targets = torch.empty_like(ids)
    clocks = clock_inputs(offset, args.tokens, model.steps, 'cuda')
    noise = torch.empty(args.tokens * model.steps * 4, args.particles, 4, device='cuda')

    def fill(data, cursor, rng):
        prefix = 50256 if cursor == 0 else int(data[cursor - 1])
        ids.copy_(torch.as_tensor(np.concatenate(([prefix], data[cursor:cursor + args.tokens - 1])), device='cuda', dtype=torch.long))
        targets.copy_(torch.as_tensor(np.array(data[cursor:cursor + args.tokens]), device='cuda', dtype=torch.long))
        clocks.copy_(clock_inputs(cursor, args.tokens, model.steps, 'cuda'))
        noise.normal_(generator=rng)

    @torch.no_grad()
    def validate():
        rng = torch.Generator(device='cuda').manual_seed(99173)
        state = model.core.initialize(torch.tensor([50256], device='cuda'), rng)
        vx, vv = state.x, state.v
        losses = []
        for cursor in range(0, args.validation_tokens + args.tokens, args.tokens):
            fill(valid, cursor, rng)
            item, vx, vv = model(vx, vv, ids, targets, clocks, noise)
            if cursor:
                losses.append(float(item))
        score = float(np.mean(losses))
        assert np.isfinite(score)
        return score

    def payload():
        return dict(model=model.state_dict(), optimizer=opt.state_dict(), x=x.detach(), v=v.detach(),
                    step=step, events=offset, generator=generator.get_state(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(), best_validation_nll=best, config=config)

    def update():
        opt.zero_grad(set_to_none=True)
        loss, nx, nv = model(x, v, ids, targets, clocks, noise)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
        opt.step()
        with torch.no_grad():
            x.copy_(nx)
            v.copy_(nv)
        return loss, norm

    def log(row):
        with (args.output / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row), flush=True)
        atomic_json(args.output / 'progress.json', dict(status='running', target_steps=args.steps, **row))

    def evaluate_save():
        nonlocal best
        atomic_json(args.output / 'progress.json', dict(status='validating', step=step, events=offset, target_steps=args.steps))
        started = time.perf_counter()
        score = validate()
        improved = score < best
        if improved:
            best = score
        data = payload()
        data['validation_nll'] = score
        if improved:
            atomic_save(args.output / 'BBest.pt', data)
        atomic_save(args.output / 'last.pt', data)
        atomic_save(args.output / f'age_{step:06d}.pt', data)
        log(dict(kind='validation', step=step, events=offset, validation_nll=score,
                 best_validation_nll=best, selected=improved, seconds=time.perf_counter() - started))

    atomic_json(args.output / 'config.json', config)
    try:
        if not args.resume:
            evaluate_save()
        graph = None
        while step < args.steps:
            started = time.perf_counter()
            fill(train, offset, generator)
            if graph is None:
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    loss, norm = update()
                torch.cuda.current_stream().wait_stream(stream)
            else:
                graph.replay()
            torch.cuda.synchronize()
            step += 1
            offset += args.tokens
            elapsed = time.perf_counter() - started
            with torch.no_grad():
                gamma = model.core.force.damping(x)
                phase = torch.cat((x, v), -1)
                centered = phase - phase.mean(0)
                covariance = centered.T @ centered / args.particles
                xvar, vvar = x.var(0, unbiased=False), v.var(0, unbiased=False)
                row = dict(kind='train', step=step, events=offset, seconds=elapsed,
                           nll=float(loss.detach()), grad_norm=float(norm.detach()),
                           energy=float(.5 * (x.square() + v.square()).sum(-1).mean()),
                           gamma_mean=float(gamma.mean()), gamma_std=float(gamma.std(unbiased=False)),
                           gamma_min=float(gamma.min()), gamma_max=float(gamma.max()),
                           kinetic_energy=float(.5 * v.square().sum(-1).mean()),
                           potential_energy=float(.5 * x.square().sum(-1).mean()),
                           x_variance=float(xvar.sum()), v_variance=float(vvar.sum()),
                           phase_covariance_trace=float(covariance.trace()),
                           mean_speed=float(v.norm(dim=-1).mean()),
                           peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
                for axis in range(4):
                    row[f'x_var_{axis}'] = float(xvar[axis])
                    row[f'v_var_{axis}'] = float(vvar[axis])
                if step == 1 or step % 10 == 0 or step == args.steps:
                    snapshot = dict(**row, wall_time=time.time(), x=x.detach().cpu().tolist(),
                                    v=v.detach().cpu().tolist(), gamma=gamma.flatten().cpu().tolist(),
                                    covariance=covariance.cpu().tolist(),
                                    covariance_eigenvalues=torch.linalg.eigvalsh(covariance).cpu().tolist())
                    atomic_json(args.output / 'live_state.json', snapshot)
                    with (args.output / 'phase_snapshots.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(snapshot, allow_nan=False) + '\n')
            row['seconds_with_monitor'] = time.perf_counter() - started
            assert all(np.isfinite(n) for n in row.values() if isinstance(n, float))
            log(row)
            x, v = x.detach(), v.detach()
            if step % args.validate_every == 0 or step == args.steps:
                evaluate_save()
            if graph is None and step < args.steps:
                # Capture does not consume data or advance the individual.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    loss, norm = update()
                torch.cuda.synchronize()
        atomic_json(args.output / 'progress.json', dict(status='complete', step=step, events=offset,
                    best_validation_nll=best, stopping_reason='5000-update budget, not proof of convergence'))
    except BaseException as exc:
        atomic_json(args.output / 'progress.json', dict(status='failed', step=step, events=offset, error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
