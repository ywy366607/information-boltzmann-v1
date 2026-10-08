"""Bounded CPU physical-response assay; no language-capability claim.

Use real induced MaleCNS edges and the production COBA/ALIF/STP step. Train
only a fresh response student against a fixed numerical oracle. Complete
stimulus trajectories are separated by batch lane before fitting. No GPU,
old checkpoint, vocabulary supervision, or validation-directed tuning.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState
from information_boltzmann.core.fly_rtc_student import TickResponseStudent
from information_boltzmann.core.fly_rtc_learning import (
    copy_physical, motor_features, physical_options, quiet_teacher_step,
    step_fly_physical_tick,
)


def edge_blocks(source: Path, block_size=250_000):
    """Read aligned NPZ edge arrays without allocating the 25M-edge graph."""
    with ZipFile(source) as archive, ExitStack() as stack:
        streams, dtypes, counts = [], [], []
        for name in ('edge_pre', 'edge_post', 'edge_weight', 'edge_delay'):
            stream = stack.enter_context(archive.open(name + '.npy'))
            version = np.lib.format.read_magic(stream)
            reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                      else np.lib.format.read_array_header_2_0)
            shape, fortran, dtype = reader(stream)
            if fortran or len(shape) != 1:
                raise ValueError('Flat aligned edge arrays required')
            streams.append(stream)
            dtypes.append(dtype)
            counts.append(shape[0])
        if len(set(counts)) != 1:
            raise ValueError('Edge arrays have different lengths')
        for start in range(0, counts[0], block_size):
            count = min(block_size, counts[0] - start)
            values = [np.frombuffer(stream.read(count * dtype.itemsize), dtype=dtype)
                      for stream, dtype in zip(streams, dtypes)]
            yield start, values


def extract_graph(source: Path, destination: Path, capacity: int) -> dict:
    """Selection uses anatomy/weights only, before observing any response."""
    with np.load(source, allow_pickle=False) as raw:
        classes, names = raw['superclass_id'], raw['superclass_names']
        signs = raw['nt_sign']
        sensory = np.isin(names[classes], FlyReservoirLM.SENSORY_CLASSES)
        motor = np.isin(names[classes], FlyReservoirLM.OUTPUT_CLASSES)
        direct_parts = []
        for _, (pre, post, weight, delay) in edge_blocks(source):
            mask = sensory[pre] & motor[post] & (signs[pre] > 0)
            direct_parts.append(np.stack((pre[mask], post[mask], weight[mask]), axis=1))
        direct = np.concatenate(direct_parts)
        if not len(direct):
            raise ValueError('Actual graph has no sensory-to-output seed edges')
        order = np.argsort(-np.abs(direct[:, 2]), kind='stable')
        chosen = set()
        for edge in order[:max(1, capacity // 8)]:
            chosen.update((int(direct[edge, 0]), int(direct[edge, 1])))
        seed = np.zeros(len(classes), dtype=bool)
        seed[list(chosen)] = True
        incident_parts = []
        for _, (pre, post, weight, delay) in edge_blocks(source):
            mask = seed[pre] | seed[post]
            incident_parts.append(np.stack((pre[mask], post[mask], weight[mask]), axis=1))
        incident = np.concatenate(incident_parts)
        for edge in np.argsort(-np.abs(incident[:, 2]), kind='stable'):
            for node in (int(incident[edge, 0]), int(incident[edge, 1])):
                if len(chosen) < capacity:
                    chosen.add(node)
            if len(chosen) == capacity:
                break
        nodes = np.array(sorted(chosen), dtype=np.int64)
        selected = np.zeros(len(classes), dtype=bool)
        selected[nodes] = True
        mapping = np.full(len(classes), -1, dtype=np.int64)
        mapping[nodes] = np.arange(len(nodes))
        parts = [[], [], [], []]
        boundary = 0
        for _, (pre, post, weight, delay) in edge_blocks(source):
            retained = selected[pre] & selected[post]
            boundary += int((selected[pre] ^ selected[post]).sum())
            for bucket, value in zip(parts, (mapping[pre[retained]], mapping[post[retained]],
                                            weight[retained], delay[retained])):
                bucket.append(value)
        kept_pre, kept_post, kept_weight, delays = [np.concatenate(p) for p in parts]
        saved = dict(neuron_body_ids=raw['neuron_body_ids'][nodes],
                     edge_pre=kept_pre, edge_post=kept_post,
                     edge_weight=kept_weight, edge_delay=delays,
                     superclass_id=classes[nodes], superclass_names=names,
                     nt_sign=raw['nt_sign'][nodes], tau_m=raw['tau_m'][nodes],
                     lambda_0=raw['lambda_0'][nodes])
        np.savez(destination, **saved)
        return dict(source=str(source), selection='strongest anatomical direct seeds plus incident edges',
                    neurons=len(nodes), edges=len(kept_pre),
                    body_ids=saved['neuron_body_ids'].tolist(),
                    original_node_indices=nodes.tolist(),
                    sensory=int(sensory[nodes].sum()), motor=int(motor[nodes].sum()),
                    delay_counts={str(d): int((delays == d).sum()) for d in range(1, 5)},
                    excitatory_edges=int((saved['nt_sign'][saved['edge_pre']] > 0).sum()),
                    inhibitory_edges=int((saved['nt_sign'][saved['edge_pre']] < 0).sum()),
                    dropped_boundary_edges=boundary,
                    graph_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                    scope='Induced boundary-value system, not full-brain response')


def initial_physical(model, batch):
    zero = torch.zeros(batch, model.n_neurons)
    return FlyPhysicalState(zero, tuple(zero.clone() for _ in range(4)),
                            zero.clone(), zero.clone(), zero.clone(), torch.ones_like(zero),
                            model.get_stp_params()[0].detach().expand_as(zero).clone(),
                            torch.zeros(batch, model.n_injection), zero.clone())


@torch.no_grad()
def responses(model, horizon, seed, *, return_surface=False, return_physical=False):
    """One continuing physical stream in each independent stimulus lane."""
    batch = 8
    student = model.rtc_student
    state = initial_physical(model, batch)
    history = student.initial_history(batch, 'cpu', torch.float32)
    history[0] = student.codec.encode(state)
    rng = np.random.default_rng(seed)
    masks = torch.tensor(rng.uniform(.5, 1.5, (batch, model.n_injection)), dtype=torch.float32)
    # These are experimental stimulus amplitudes, not imposed physical clocks.
    # The latter come unchanged from the numerical oracle's actual parameters.
    amplitudes = torch.tensor([1., 2., 3., 4., 1.5, 2.5, 3.5, 4.5])
    periods = torch.tensor([3, 5, 7, 9, 4, 6, 8, 10])
    thresholds = model.get_thresholds()[0, model.injection_index].detach()
    options = physical_options(model)
    anchors, future, features, roots = [], [], [], []
    surface_roots, surface_future, mean_roots = [], [], []
    physical_roots, physical_future = [], []
    counterfactual_motor = 0.
    for tick in range(1, 65):
        active = (tick % periods == 0).float()
        source = torch.zeros_like(state.h)
        source[:, model.injection_index] = (
            active[:, None] * amplitudes[:, None] * masks * thresholds[None])
        state = step_fly_physical_tick(model, state, None, source, state.baseline, options)
        observed = student.codec.encode(state)
        history = torch.cat((observed[None], history[:-1]))
        if tick % 8:
            continue
        anchors.append(history.clone())
        if return_physical:
            physical_roots.append(copy_physical(state))
        copied = copy_physical(state)
        z, y = [], []
        roots.append(motor_features(model, state))
        if return_surface:
            surface_roots.append(state.h[:, model.read_indices].clone())
            mean_roots.append(state.h_mean[:, model.read_indices].clone())
        future_membranes = []
        future_states = []
        for _ in range(horizon):
            copied = quiet_teacher_step(model, copied, options)
            if return_physical:
                future_states.append(copy_physical(copied))
            z.append(student.codec.encode(copied))
            y.append(motor_features(model, copied))
            if return_surface:
                future_membranes.append(copied.h[:, model.read_indices].clone())
        future.append(torch.stack(z))
        features.append(torch.stack(y))
        if return_surface:
            surface_future.append(torch.stack(future_membranes))
        if return_physical:
            physical_future.append(future_states)
        # Zero-drive, zero-state COBA oracle remains zero; actual motor membrane
        # energy is the counterfactual response, before any learned decoder.
        counterfactual_motor += float(state.h[:, model.read_indices].square().mean())
    # Preserve axes until lane-wise split. Each lane has its own full history.
    result = (torch.stack(anchors), torch.stack(future), torch.stack(features),
              torch.stack(roots), counterfactual_motor / len(anchors))
    if return_surface:
        return result + (torch.stack(surface_roots), torch.stack(surface_future),
                         torch.stack(mean_roots)) + ((physical_roots, physical_future) if return_physical else ())
    if return_physical:
        return result + (physical_roots, physical_future)
    return result


def subset(dataset, lanes):
    histories, future, features, roots, _ = dataset
    # [anchor, delay/horizon, batch, region, latent] -> [delay/horizon, cases,...]
    history = histories[:, :, lanes].transpose(0, 1).flatten(1, 2)
    target = future[:, :, lanes].transpose(0, 1).flatten(1, 2)
    feature = features[:, :, lanes].transpose(0, 1).flatten(1, 2)
    root = roots[:, lanes].flatten(0, 1)
    return history, target, feature, root


def prediction(student, data):
    history, target, features, root = data
    drafts, _ = student.rollout(history)
    return drafts[1:], student.draft_features(drafts, root)[1:]


def evaluate(student, data):
    with torch.no_grad():
        latent, motor = prediction(student, data)
        history, target, features, root = data
        hold_latent = (target - history[0][None]).square().mean(dim=tuple(range(1, target.ndim)))
        hold_motor = (features - root[None]).square().mean(dim=(1, 2))
        err_latent = (target - latent).square().mean(dim=tuple(range(1, target.ndim)))
        err_motor = (features - motor).square().mean(dim=(1, 2))
        # Give the read adapter exact future latent codes as a diagnostic oracle.
        # This isolates latent-rollout error from read-adapter fit; it is never
        # used by runtime, for training, or as a task-performance score.
        oracle_drafts = torch.cat((history[0][None], target))
        oracle_features = student.draft_features(oracle_drafts, root)[1:]
        oracle_error = (features - oracle_features).square().mean()
        return dict(latent_mse=float(err_latent.mean()), hold_latent_mse=float(hold_latent.mean()),
                    motor_delta_mse=float(err_motor.mean()), hold_motor_delta_mse=float(hold_motor.mean()),
                    latent_error_ratio=float(err_latent.mean() / hold_latent.mean().clamp_min(1e-20)),
                    motor_error_ratio=float(err_motor.mean() / hold_motor.mean().clamp_min(1e-20)),
                    oracle_latent_motor_error_ratio=float(oracle_error / hold_motor.mean().clamp_min(1e-20)),
                    per_tick=[dict(tick=i + 1, latent=float(err_latent[i]),
                                   hold_latent=float(hold_latent[i]), motor=float(err_motor[i]),
                                   hold_motor=float(hold_motor[i])) for i in range(len(err_latent))])


def oracle_linear_readout(student, train, validation):
    """No additional neural training: least-squares read of true future codes.

    Fit only training lanes. An oracle diagnostic, never a deployed forecast:
    failure separates insufficient linear observation from rollout error;
    success shows that a linear read is possible on this bounded dataset.
    """
    def pair(data):
        history, target, feature, root = data
        idx = student.codec.graph.motor_regions
        x = (target[:, :, idx] - history[0][None, :, idx]).flatten(2).flatten(0, 1)
        y = (feature - root[None]).flatten(0, 1)
        return x.double(), y.double()
    x, y = pair(train)
    xv, yv = pair(validation)
    solved = torch.linalg.lstsq(x, y, driver='gelsd')
    error = (xv @ solved.solution - yv).square().mean()
    return dict(heldout_error_ratio=float(error / yv.square().mean()),
                rank=int(solved.rank), dimensions=int(x.shape[-1]),
                training_cases=len(x), heldout_cases=len(xv),
                scope='Oracle true future latent, not autonomous draft')


def observable_rowspace_audit(model):
    """Can the fixed encoder exactly preserve the linear motor observable?

    Ambient linear-state test only. It does not assert that every null-space
    direction is visited by the actual nonlinear physical trajectory.
    """
    codec = model.rtc_student.codec
    weights = model.output_read.weight.detach().double()
    positions = {int(neuron): i for i, neuron in enumerate(model.read_indices)}
    sampled_energy, residual_energy = 0., 0.
    sample_count = codec.sample_indices.shape[1]
    for region in codec.graph.motor_regions:
        indices, mask = codec.sample_indices[region], codec.sample_mask[region]
        projection = codec.observation_projection[region].double()
        projection = projection * mask.repeat(10).double()[:, None]
        observable = torch.zeros(projection.shape[0], weights.shape[0], dtype=torch.float64)
        for slot in range(sample_count):
            if mask[slot] and int(indices[slot]) in positions:
                observable[slot] = weights[:, positions[int(indices[slot])]]
        reconstructed = projection @ (torch.linalg.pinv(projection) @ observable)
        sampled_energy += float(observable.square().sum())
        residual_energy += float((observable - reconstructed).square().sum())
    total = float(weights.square().sum())
    lost = (total - sampled_energy + residual_energy) / total
    return dict(motor_observable_outside_codec_rowspace_fraction=lost,
                scope='Ambient affine observable sufficiency; not reachable-manifold impossibility',
                interpretation='Positive residual means exact output reconstruction for arbitrary full physical states is not guaranteed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, default=ROOT / 'data/malecns_v1/fly_reservoir_coba.npz')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/published/fly_rtc_cpu_response_20261007.json')
    parser.add_argument('--neurons', type=int, default=256)
    parser.add_argument('--updates', type=int, default=120)
    parser.add_argument('--seed', type=int, default=11)
    args = parser.parse_args()
    preregistration = dict(seed=args.seed, optimizer_updates=args.updates,
        train_lanes=[0, 1, 2, 3], heldout_lanes=[4, 5, 6, 7],
        horizon=14, stopping='fixed bounded budget; no validation tuning',
        baseline='hold current true response across horizon',
        review='rtc_contract_review approved bounded mechanism test',
        forecast_prediction='If reduced dynamics are learnable at this budget, heldout response-delta error is below persistence.',
        failure_action='Retain timing implementation; reconsider codec/training sufficiency before expensive language run.',
        timestamp_prediction='Outputs available while teacher blocked; arrivals change only uncommitted suffix.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prereg_path = args.output.with_name(args.output.stem + '_preregistered.json')
    prereg_path.write_text(json.dumps(preregistration, indent=2), encoding='utf-8')
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    with tempfile.TemporaryDirectory(prefix='fly-rtc-cpu-') as scratch:
        graph = Path(scratch) / 'real_induced_graph.npz'
        metadata = extract_graph(args.graph, graph, args.neurons)
        model = FlyReservoirLM(graph, vocab_size=16, d_model=16, injection='sensory',
                               read_surface='output', synapse_model='coba',
                               use_alif=True, use_stp=True)
        model.rtc_student = TickResponseStudent.from_model(
            model, latent_dim=32, sample_per_region=32, horizon=14, seed=args.seed)
        model.eval()
        dataset = responses(model, 14, args.seed)
        train = subset(dataset, slice(0, 4))
        validation = subset(dataset, slice(4, 8))
        # Entire future trajectories/stimulus lanes are held out, not shuffled
        # adjacent windows from the same hidden state.
        before = evaluate(model.rtc_student, validation)
        optimizer = torch.optim.AdamW(model.rtc_student.parameters(), lr=2e-3, weight_decay=0)
        latent_scale = (train[1] - train[0][0][None]).square().mean().clamp_min(1e-12)
        motor_scale = (train[2] - train[3][None]).square().mean().clamp_min(1e-12)
        curve = []
        for update in range(args.updates):
            optimizer.zero_grad(set_to_none=True)
            latent, motor = prediction(model.rtc_student, train)
            loss = F.mse_loss(latent, train[1]) / latent_scale + F.mse_loss(motor, train[2]) / motor_scale
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite numerical student fit')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.rtc_student.parameters(), 1.)
            optimizer.step()
            if update == 0 or (update + 1) % 20 == 0:
                point = dict(update=update + 1, normalized_train_loss=float(loss.detach()))
                curve.append(point)
                print(json.dumps(point), flush=True)
        after = evaluate(model.rtc_student, validation)
        oracle = oracle_linear_readout(model.rtc_student, train, validation)
        by_lane = [evaluate(model.rtc_student, subset(dataset, slice(i, i + 1))) for i in range(4, 8)]
        response = dataset[-1]
        informative = response > 0 and before['hold_motor_delta_mse'] > 0
        improved = informative and after['motor_error_ratio'] < 1
        report = dict(scope='CPU numerical response learnability; not language capability or full-brain verdict',
                      preregistration=preregistration,
                      graph=metadata, device='cpu', checkpoints_written=0,
                      teacher='fresh fixed production COBA-ALIF-STP numerical oracle',
                      teacher_motor_counterfactual_energy=response,
                      input_currents='Only actual annotated sensory neurons; no motor injection',
                      updates=args.updates, curve=curve, before=before, after=after,
                      train_after=evaluate(model.rtc_student, train),
                      oracle_least_squares_readout=oracle,
                      observable_rowspace_audit=observable_rowspace_audit(model),
                      heldout_by_lane=by_lane,
                      verdict='heldout_motor_forecast_improved' if improved else 'inconclusive_or_no_gain',
                      elapsed_seconds=time.perf_counter() - start)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({k: report[k] for k in ('verdict', 'elapsed_seconds', 'teacher_motor_counterfactual_energy')}), flush=True)
        print(json.dumps(dict(before=before['motor_error_ratio'], after=after['motor_error_ratio'], report=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
