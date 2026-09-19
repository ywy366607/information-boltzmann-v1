"""Persistent local-collision BPE individual with held-out checkpoint selection."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ib_bpe_window import clock_inputs
from scripts.ib_local.window import LocalWindow
from scripts.ib_local.sampling import sample_window, padded_capacities


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
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(.025)
    raise PermissionError(f'Could not finalize checkpoint: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--particles', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--validate-every', type=int, default=250)
    parser.add_argument('--validation-tokens', type=int, default=4096)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--benchmark-updates', type=int, default=0)
    parser.add_argument('--compile-layer', action='store_true')
    parser.add_argument('--compile-evolve', action='store_true')
    parser.add_argument('--pad-candidates', action='store_true')
    parser.add_argument('--cuda-graph', action='store_true')
    parser.add_argument('--small-workspace', action='store_true')
    parser.add_argument('--no-recompute', action='store_true')
    parser.add_argument('--block-graph', action='store_true')
    parser.add_argument('--decoupled-clip', action='store_true')
    parser.add_argument('--write-operator', action='store_true')
    parser.add_argument('--coupling-mode', choices=['none', 'adaptive_force', 'message_coupling'], default='none')
    parser.add_argument('--score-scale', type=float, default=1.0)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    if args.small_workspace:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        os.environ['CUBLASLT_WORKSPACE_SIZE'] = '128'
        os.environ['TORCH_CUBLASLT_UNIFIED_WORKSPACE'] = '1'
    if args.cuda_graph and not (args.pad_candidates and args.compile_evolve):
        parser.error('--cuda-graph requires --pad-candidates and --compile-evolve')
    if args.block_graph and (args.cuda_graph or not args.pad_candidates):
        parser.error('--block-graph requires padding and replaces --cuda-graph')
    if args.wait_pid:
        import subprocess
        while True:
            result = subprocess.run(['tasklist', '/FI', f'PID eq {args.wait_pid}', '/FO', 'CSV'], capture_output=True, text=True, creationflags=0x08000000)
            if f'"{args.wait_pid}"' not in result.stdout:
                break
            time.sleep(5)
    args.output.mkdir(parents=True, exist_ok=bool(args.resume))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['manifest_sha256'] = hashlib.sha256((args.data / 'manifest.json').read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[2]
    sources = list((root / 'scripts/ib_local').glob('*.py'))
    sources += [root / 'fine_grain/information_boltzmann' / name for name in
                ('force.py', 'model.py', 'density.py', 'state.py', 'collision.py')]
    sources += [root / 'scripts' / name for name in ('ib_bpe_window.py', 'ib_fused_ou.py')]
    config['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    snapshot = 'source' if not args.resume else f'source_resume_{time.time_ns()}'
    config['source_snapshot'] = snapshot
    for source in sources:
        archived = args.output / snapshot / source.relative_to(root)
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(source.read_bytes())
    config.update(seed=11, lr=3e-4, weight_decay=.01, validation_seed=99173,
                  burn_in_tokens=args.tokens, selection='minimum frozen held-out BPE NLL',
                  architecture='birth attention flow; strict local Poisson elastic collision; Hc32; S4',
                  mask_mode='strict_local_v1', proposal='numpy PCG64 Poisson; all-candidate CPU DAG v1',
                  input_semantics='birth only; subsequent observed tokens enter force',
                  gradient='window-truncated pathwise + causal Bernoulli score prefix',
                  budget_label='3000-update short baseline, convergence not established')
    torch.set_num_threads(1)
    if sys.platform == 'win32':
        # PyTorch's static launcher narrows a capture-stream handle to C long
        # on this Windows runtime. Triton's standard launcher preserves it.
        torch._inductor.config.use_static_cuda_launcher = False
    torch.manual_seed(11)
    # Allocator budget; full process/driver memory is measured separately.
    torch.cuda.set_per_process_memory_fraction(.48)
    train = np.load(args.data / 'train.npy', mmap_mode='r')
    valid = np.load(args.data / 'validation.npy', mmap_mode='r')
    assert args.steps * args.tokens <= len(train), 'No implicit training stream wrap'
    assert args.validation_tokens % args.tokens == 0
    assert args.validation_tokens + args.tokens <= len(valid)
    score_scale = args.score_scale
    model = LocalWindow(hidden=args.hidden, particles=args.particles, use_write_operator=args.write_operator, coupling_mode=args.coupling_mode, score_scale=score_scale).cuda()
    model.recompute = not args.no_recompute
    if args.compile_layer:
        model.layer_fn = torch.compile(model.layer_fn, dynamic=True)
    if args.compile_evolve:
        model.evolve = torch.compile(model.evolve, dynamic=not args.pad_candidates)
    if args.block_graph:
        from scripts.ib_local.block_graph import BlockGraph
        model.block_engine = BlockGraph(model)
    config['parameters'] = sum(p.numel() for p in model.parameters())
    proposal_rng = np.random.default_rng(11)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=True, capturable=True)
    generator = torch.Generator(device='cuda').manual_seed(11)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    step, offset, best = 0, 0, float('inf')
    if args.resume:
        saved = torch.load(args.resume, map_location='cuda', weights_only=False)
        for key in ('manifest_sha256', 'tokens', 'particles', 'hidden', 'validation_tokens', 'architecture', 'mask_mode', 'proposal'):
            assert saved['config'][key] == config[key], key
        proposal_rng.bit_generator.state = saved['proposal_rng']
        if args.write_operator or args.coupling_mode in ('adaptive_force', 'message_coupling'):
            model.load_state_dict(saved['model'], strict=False)
            saved_opt = saved['optimizer']
            params = list(model.parameters())
            for i in range(len(saved_opt['param_groups'][0]['params'])):
                p = params[i]
                if i in saved_opt['state']:
                    opt.state[p] = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in saved_opt['state'][i].items()}
        else:
            model.load_state_dict(saved['model'])
            opt.load_state_dict(saved['optimizer'])
        x, v = saved['x'].detach().clone().requires_grad_(), saved['v'].detach().clone().requires_grad_()
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
        vrng = np.random.default_rng(99173)
        for cursor in range(0, args.validation_tokens + args.tokens, args.tokens):
            fill(valid, cursor, rng)
            validation_tables, _ = sample_window(vrng, args.tokens, model.steps, args.particles, 'cuda', padding=args.pad_candidates)
            _, item, vx, vv, _ = model(vx, vv, ids, targets, clocks, noise, validation_tables)
            if cursor:
                losses.append(float(item))
        score = float(np.mean(losses))
        assert np.isfinite(score)
        return score

    def payload():
        return dict(model=model.state_dict(), optimizer=opt.state_dict(), x=x.detach(), v=v.detach(),
                    step=step, events=offset, generator=generator.get_state(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(), proposal_rng=proposal_rng.bit_generator.state,
                    previous_token=50256 if offset == 0 else int(train[offset-1]),
                    physical_clock=float(offset), best_validation_nll=best, config=config)

    def update():
        opt.zero_grad(set_to_none=True)
        if x.is_leaf:
            x.grad = None
        if v.is_leaf:
            v.grad = None
        if not args.decoupled_clip:
            surrogate, loss, nx, nv, accepted = model(x, v, ids, targets, clocks, noise, tables)
            surrogate.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
            opt.step()
            with torch.no_grad():
                x.copy_(nx)
                v.copy_(nv)
            return loss, norm, accepted
        else:
            loss_p, nx, nv, ce_det = model.forward_path(x, v, ids, targets, clocks, noise, tables)
            loss_p.backward()
            grads_path = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
            model.zero_grad(set_to_none=True)

            score_val, _, _, accepted = model.forward_score(x, v, ids, targets, clocks, noise, tables, ce_det)
            score_val.backward()
            grads_score = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
            model.zero_grad(set_to_none=True)

            norm_p = torch.cat([g.flatten() for g in grads_path if g is not None]).norm()
            norm_s = torch.cat([g.flatten() for g in grads_score if g is not None]).norm()
            scale_p = min(1.0, float(1.0 / norm_p.item())) if norm_p > 0 else 1.0
            scale_s = min(1.0, float(1.0 / norm_s.item())) if norm_s > 0 else 1.0

            for p, gp, gs in zip(model.parameters(), grads_path, grads_score):
                if gp is not None and gs is not None:
                    p.grad = gp * scale_p + gs * scale_s * score_scale
                elif gp is not None:
                    p.grad = gp * scale_p
                elif gs is not None:
                    p.grad = gs * scale_s * score_scale
                else:
                    p.grad = None

            opt.step()
            with torch.no_grad():
                x.copy_(nx)
                v.copy_(nv)
            return loss_p, norm_p, accepted

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
        if not args.resume and not args.benchmark_updates:
            evaluate_save()
        graph = None
        graph_signature = None
        static_bases = None
        warm_updates = 0
        while step < args.steps:
            started = time.perf_counter()
            fill(train, offset, generator)
            tables, candidates = sample_window(proposal_rng, args.tokens, model.steps, args.particles, 'cuda', padding=args.pad_candidates)
            signature = tuple(tuple(len(layer[0]) for layer in table) for table in tables)
            replayed = graph is not None and signature == graph_signature
            if replayed:
                for destination, fresh in zip(static_bases, tables[0][0]):
                    destination.copy_(fresh._base)
                graph.replay()
                loss, norm, accepted = graph_outputs
            else:
                if graph is not None:
                    # A complete overflow table uses the ordinary path. Release
                    # the graph's pool before allocating its transient activations.
                    torch.cuda.synchronize()
                    graph.reset()
                    graph = graph_outputs = static_bases = graph_signature = None
                    loss = norm = accepted = None
                    opt.zero_grad(set_to_none=True)
                    x.grad = v.grad = None
                    gc.collect()
                    torch.cuda.empty_cache()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    loss, norm, accepted = update()
                torch.cuda.current_stream().wait_stream(stream)
                warm_updates += 1
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
                           candidates=candidates, accepted=int(accepted.detach()), graph_replayed=replayed,
                           block_graph=args.block_graph,
                           energy=float(.5 * (x.square() + v.square()).sum(-1).mean()),
                           gamma_mean=float(gamma.mean()), gamma_std=float(gamma.std(unbiased=False)),
                           gamma_min=float(gamma.min()), gamma_max=float(gamma.max()),
                           kinetic_energy=float(.5 * v.square().sum(-1).mean()),
                           potential_energy=float(.5 * x.square().sum(-1).mean()),
                           x_variance=float(xvar.sum()), v_variance=float(vvar.sum()),
                           phase_covariance_trace=float(covariance.trace()),
                           mean_speed=float(v.norm(dim=-1).mean()),
                           peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                           allocated_mib=torch.cuda.memory_allocated() / 2**20,
                           reserved_mib=torch.cuda.memory_reserved() / 2**20)
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
            # Keep the compiled input-gradient signature fixed across windows.
            # These are new leaves: no gradient crosses the update boundary.
            x, v = x.detach().requires_grad_(), v.detach().requires_grad_()
            if args.benchmark_updates and step >= args.benchmark_updates:
                atomic_json(args.output / 'benchmark.json', dict(updates=step, last=row))
                return
            if step % args.validate_every == 0 or step == args.steps:
                evaluate_save()
            elif step % 50 == 0 or (step < 50 and step % 10 == 0):
                atomic_save(args.output / 'last.pt', payload())
            regular_shape = all(shape == padded_capacities(args.particles) for shape in signature)
            if args.cuda_graph and graph is None and warm_updates >= 2 and regular_shape and step < args.steps:
                graph_signature = signature
                static_bases = tuple(t._base for t in tables[0][0])
                if any(t is None for t in static_bases):
                    raise RuntimeError('Graph proposal inputs must be views of packed buffers')
                torch.cuda.empty_cache()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    graph_outputs = update()
                torch.cuda.synchronize()
        atomic_json(args.output / 'progress.json', dict(status='complete', step=step, events=offset,
                    best_validation_nll=best, target_steps=args.steps, stopping_reason=f'{args.steps}-update budget, not proof of convergence'))
    except BaseException as exc:
        atomic_json(args.output / 'progress.json', dict(status='failed', step=step, events=offset, error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
