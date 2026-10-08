"""CPU numerical/interface audit on real OWT; no optimizer or capability claim."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.core.temporal_probes import CausalProbeFilterBank, sample_compact_probes
from information_boltzmann.core.gradient_norms import stable_grad_norm


def norm(parameters):
    return float(stable_grad_norm(parameters))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('results/medium_v8_joint_bptt32_96k_recovered/config.json'))
    parser.add_argument('--data', type=Path, default=Path('data/ib_owt_gpt2/train.npy'))
    parser.add_argument('--output', type=Path, default=Path('results/published/medium_temporal_probes_20261008.json'))
    parser.add_argument('--burn-in', type=int, default=64)
    parser.add_argument('--score', type=int, default=32)
    args = parser.parse_args()
    if args.burn_in < 1 or args.score < 1:
        raise ValueError('Positive real-text budgets required')
    torch.set_num_threads(4)
    torch.manual_seed(990)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    constructor = dict(config['constructor'])
    constructor.update(medium_execution='native', port_execution='native', read_mode='dynamic')
    model = PlasticMediumPorts3D(**constructor).cpu()
    duration = config['event_duration']
    dt = float(duration)
    n_probes, channels = model.readout.heads * model.readout.queries, model.medium.channels
    # Explicit coverage choices only: finite modes/half-lives, not an optimal clock.
    half_lives = torch.tensor([1.,4.,16.,64.]) * dt
    rates = math.log(2.) / half_lives
    omega = torch.tensor([0.,1/16,1/8,1/4]) * (math.pi / dt)
    bank = CausalProbeFilterBank(n_probes, 2*channels, rates, omega)
    # Modest shared temporal mixing; every local history remains individually stored.
    mode_mix = nn.Parameter(torch.randn(n_probes, bank.modes, 2) / math.sqrt(2*n_probes*bank.modes))
    projection = nn.Linear(2*channels, channels, bias=False)
    data = np.load(args.data, mmap_mode='r')
    offset = 96001  # published maturity cursor; this individual is explicitly NEW.
    values = np.array(data[offset:offset + args.burn_in + args.score + 1], dtype=np.int64)
    if len(values) != args.burn_in + args.score + 1:
        raise ValueError('Insufficient real data')
    belief, history = model.initial_belief(), bank.initial_state()
    model_before = {n: p.detach().clone() for n,p in model.named_parameters()}
    parameter_versions = {n: p._version for n,p in model.named_parameters()}

    def sample(belief, prepared):
        field = sample_compact_probes(model.readout, belief.medium.field)
        motion = model.read_time_reference * model.medium.field_rhs(belief.medium, prepared=prepared)
        return torch.cat((field, sample_compact_probes(model.readout, motion)), -1)

    def measure(belief, history, prepared):
        base, _ = model.read(belief, decode=False, prepared=prepared)
        temporal = (bank.features(history) * mode_mix[None,:,:,None,:]).sum((1,2,4))
        correction = projection(temporal)
        return model.decode(base + correction), model.decode(base.detach() + correction)

    started = time.perf_counter()
    with torch.no_grad():
        prepared = model.medium.prepare_evolution()
        token_features = F.normalize(model.source.embedding.weight, dim=-1)
        for token in values[:args.burn_in]:
            inp = torch.tensor([int(token)])
            belief, _ = model.assimilate(belief, inp, token_features=token_features, diagnostics=False)
            # Hold the endpoint observation over the just-executed interval.
            # Endpoint is already available at emission; this is event sampling,
            # not a claim of exact continuously observed physical integration.
            belief, _ = model.advance(belief, duration, prepared=prepared, diagnostics=False)
            history = bank(sample(belief, prepared), history, duration)
    burn_seconds = time.perf_counter()-started
    belief, history = belief.detach(), history.detach()
    prepared = model.medium.prepare_evolution()
    token_features = F.normalize(model.source.embedding.weight, dim=-1)
    features, snapshots, logits, baseline_logits, isolated_logits = [], [], [], [], []
    tick_seconds, read_seconds, filter_seconds = [], [], []
    for token in values[args.burn_in:-1]:
        begin = time.perf_counter()
        belief, _ = model.assimilate(belief, torch.tensor([int(token)]),
                                     token_features=token_features, diagnostics=False)
        belief, _ = model.advance(belief, duration, prepared=prepared, diagnostics=False)
        tick_seconds.append(time.perf_counter()-begin)
        begin = time.perf_counter()
        signal = sample(belief, prepared)
        read_seconds.append(time.perf_counter()-begin)
        begin = time.perf_counter()
        history = bank(signal, history, duration)
        filter_seconds.append(time.perf_counter()-begin)
        features.append(signal)
        snapshots.append(history.value.detach().clone())
        combined, isolated = measure(belief, history, prepared)
        logits.append(combined)
        isolated_logits.append(isolated)
        baseline_logits.append(model.read(belief, prepared=prepared)[0])
    targets = torch.from_numpy(values[args.burn_in+1:]).long()
    predictions = torch.cat(logits)
    loss = F.cross_entropy(predictions, targets)
    isolated_loss = F.cross_entropy(torch.cat(isolated_logits), targets)
    isolated_groups = {
        'filter_log_rates':[bank.log_rate], 'filter_frequencies':[bank.frequency],
        'temporal_mix':[mode_mix], 'temporal_projection':list(projection.parameters()),
        'probe_coordinates':[model.readout.probe_coords],
        'physical_medium':list(model.medium.parameters()),
        'write_agent':list(model.write_agent.parameters()),
        'source_writer':list(model.source.parameters()),
    }
    unique = {id(p):p for group in isolated_groups.values() for p in group}
    isolated_grad = torch.autograd.grad(isolated_loss, list(unique.values()),
                                       retain_graph=True, allow_unused=True)
    by_id = dict(zip(unique, isolated_grad))
    isolated_norms = {name:math.sqrt(sum(float(by_id[id(p)].detach().double().square().sum())
                             for p in params if by_id[id(p)] is not None))
                      for name,params in isolated_groups.items()}
    if any(not math.isfinite(n) or n <= 0 for n in isolated_norms.values()):
        raise AssertionError(f'Temporal-only credit failed: {isolated_norms}')
    loss.backward()
    groups = {
        'filter_log_rates': norm([bank.log_rate]),
        'filter_frequencies': norm([bank.frequency]),
        'temporal_mix': norm([mode_mix]),
        'temporal_projection': norm(projection.parameters()),
        'probe_coordinates': norm([model.readout.probe_coords]),
        'physical_medium': norm(model.medium.parameters()),
        'write_agent': norm(model.write_agent.parameters()),
        'source_writer': norm(model.source.parameters()),
        'readout': norm(model.readout.parameters()),
        'decoder': norm(model.decoder.parameters()),
    }
    if any(not math.isfinite(n) or n <= 0 for n in groups.values()):
        raise AssertionError(f'Nonfinite or disconnected required group: {groups}')
    changed = {n: float((p.detach()-model_before[n]).abs().max())
               for n,p in model.named_parameters() if not torch.equal(p.detach(),model_before[n])}
    if changed or parameter_versions != {n:p._version for n,p in model.named_parameters()}:
        raise AssertionError('Audit mutated model parameters')
    with torch.no_grad():
        # Continue on cached identical signals without restarting either physical run.
        # Separate replay states test value consistency only.
        initial = bank.initial_state()
        whole = initial
        for r in features:
            whole = bank(r.detach(), whole, duration)
        split = initial
        for r in features[:args.score//2]:
            split = bank(r.detach(), split, duration)
        split = type(split).from_state_dict(split.state_dict())
        for r in features[args.score//2:]:
            split = bank(r.detach(), split, duration)
        replay_diff = float((whole.value-split.value).abs().max())
        all_history = torch.stack(snapshots)
        real_power = all_history.real.double().square().mean((0,1,2,4))
        imag_power = all_history.imag.double().square().mean((0,1,2,4))
    report = {
        'scope':'fresh-weight real OWT numerical/interface acceptance; no predictive-quality claim',
        'seed':990, 'initialization':'new random model with saved architecture settings; not restored trained individual',
        'constructor':constructor, 'model_parameters':sum(p.numel() for p in model.parameters()),
        'added_parameters':sum(p.numel() for p in bank.parameters())+mode_mix.numel()+sum(p.numel() for p in projection.parameters()),
        'data':str(args.data), 'offset':offset, 'input_sha256':hashlib.sha256(values.tobytes()).hexdigest(),
        'burn_in_tokens':args.burn_in, 'score_tokens':args.score,
        'gradient_boundary':'no-grad burn-in; connected full score tape; physical and temporal state retained',
        'optimizer_updates':0, 'parameter_deltas':changed, 'physical_advances':args.burn_in+args.score,
        'duration':dt, 'elapsed_physical':float(belief.medium.elapsed[0]),
        'elapsed_filter':float(history.elapsed[0]), 'bank_state_bytes':history.value.numel()*history.value.element_size()+history.elapsed.numel()*history.elapsed.element_size(),
        'num_probes':n_probes, 'num_modes':bank.modes, 'num_input_channels':2*channels,
        'diagnostic_half_lives':half_lives.tolist(), 'diagnostic_angular_frequencies':omega.tolist(),
        'gradient_norms':groups, 'isolated_temporal_branch_gradient_norms':isolated_norms,
        'isolated_credit_contract':'base feature detached only for auxiliary audit; temporal signals remain attached to physical trajectory',
        'filter_state_finite':bool(history.value.isfinite().all()),
        'real_power_per_mode':real_power.tolist(), 'imaginary_power_per_mode':imag_power.tolist(),
        'same_signal_chunk_replay_maxdiff':replay_diff,
        'untrained_nll_for_gradient_only':float(loss.detach()),
        'baseline_untrained_nll_for_provenance_only':float(F.cross_entropy(torch.cat(baseline_logits).detach(),targets)),
        'timing_cpu_ms':{'burn_in_total':1000*burn_seconds,
                        'mean_write_and_advance':1000*np.mean(tick_seconds),
                        'mean_probe_field_rhs_and_pool':1000*np.mean(read_seconds),
                        'mean_bank_update':1000*np.mean(filter_seconds)},
        'limitations':['endpoint ZOH samples; not exact integration of varying physical waveform',
                       'fixed coordinates and weights during audit; no moving-probe history transport',
                       'new weights; NLL values used only to exercise actual CE derivative, not rank architectures',
                       'positive-rate contraction is per finite fixed parameter instance, not lifetime guarantee'],
        'elapsed_seconds':time.perf_counter()-started,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('scope','added_parameters','bank_state_bytes','isolated_temporal_branch_gradient_norms','timing_cpu_ms','same_signal_chunk_replay_maxdiff','elapsed_seconds')},indent=2),flush=True)


if __name__ == '__main__':
    main()
