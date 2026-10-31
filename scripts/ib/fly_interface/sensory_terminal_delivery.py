"""Real-clock check of the sensory terminal pathway on a full individual (no exposure, no learning).

1. Exact delivery: the first tick's site delivery from archived packets vs the
   point transmission of the same sensory pulses through the same edges.
2. Arms on the saved pending clock (same state/RNG/eyes): point (pathway off),
   sites with terminal input and gating off (must reproduce point up to
   summation order), active, and point again (source preserved).
Reports per-tick release-fraction distributions [min | median | mean | max]
and downstream divergence. Not a learning or behaviour benefit claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.fly_interface.native_causal_audit import restore_native, pending_queue_tape, clone_state, sha256
from scripts.ib.fly_interface.action_loop_v2_audit import state_digest
from scripts.ib.fly_interface.native_system_matrix import deterministic_replay
from information_boltzmann.core.fly_native_generative import _preserve_rollout
from information_boltzmann.core import fly_bptt_learning as fl
from information_boltzmann.core import fly_sensory_terminals as st


def dist(x):
    x = x.detach().double().reshape(-1)
    return {'min': float(x.min()), 'median': float(x.median()), 'mean': float(x.mean()), 'max': float(x.max())}


def run(checkpoint: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    source_sha = sha256(checkpoint)
    learner, saved, args, changed = restore_native(checkpoint, output, frozen=True, device='cpu')
    if changed:
        raise ValueError('Delivery check must not migrate the learning algorithm')
    model = learner.model
    if not getattr(model, 'sensory_terminal_enabled', False):
        raise ValueError('Archive does not carry the sensory terminal pathway')
    # 1. exact first-tick delivery
    with torch.no_grad():
        current = learner.state
        packets = st.terminal_values(model, current)[1]
        site_e, site_i = st.sensory_delivery(model, packets)
        mask = torch.zeros(1, model.n_neurons)
        mask[0, model.st_cell] = 1.
        only = tuple(p * mask for p in current.ring)
        point_e, point_i = model.transmit(only, 'e'), model.transmit(only, 'i')
        delivery = {}
        for label, a, b in (('excitatory', site_e, point_e), ('inhibitory', site_i, point_i)):
            diff = (a - b).abs()
            delivery[label] = {'max_abs_diff': float(diff.max()), 'point_total': float(b.sum()),
                               'site_total': float(a.sum()), 'max_point': float(b.abs().max()),
                               'targets_receiving': int((b != 0).sum())}
    print(json.dumps({'delivery': delivery}), flush=True)
    tape = pending_queue_tape(saved, 1, 1)[0]
    job = saved['learner'].get('native_context_state', {}).get('pending_teaching') or {}
    source = job.get('source', tape['forced']['sources'][0])
    duration = tape['forced']['intervals'][0]
    initial_digest = state_digest(learner.state.state_dict())
    weights_digest = state_digest(model.state_dict())
    initial_rng = torch.get_rng_state().clone()
    original = fl.advance_sensory_terminals
    stats = []

    def recording(model_, current_, raw, **kw):
        out = original(model_, current_, raw, **kw)
        if model_.sensory_terminal_gating_enabled:
            rates = kw.get('base_rates') or model_.get_decay_rates()
            gains = kw.get('conductance_gains') or model_.get_conductance_gains()
            frac = st.release_fraction(model_, out[0], rates, gains)
            stats.append({'fraction': dist(frac), 'sites_below_0_99': float((frac < .99).double().mean()),
                          'excitatory_signal_max': float(out[0][0].max()), 'inhibitory_signal_max': float(out[0][1].max())})
        return out

    fl.advance_sensory_terminals = recording
    eyes = None
    results = {}
    arms = (('point', False, True, True), ('sites_ungated', True, False, False),
            ('active', True, True, True), ('point_again', False, True, True))
    try:
        with deterministic_replay():
            for label, enabled, gating, inputs in arms:
                model.sensory_terminal_enabled = enabled
                model.sensory_terminal_gating_enabled = gating
                model.sensory_terminal_input_enabled = inputs
                stats.clear()
                started = time.perf_counter()
                spikes = []
                with _preserve_rollout(model), torch.no_grad():
                    current = clone_state(learner.state)
                    current = fl.prepare_sensory_expectation(model, current, learner.previous_token, source)
                    out = fl.advance_fly_token_stream(model, current,
                        torch.tensor([int(learner.previous_token)], device=current.h.device),
                        [duration], ticks=duration, hold_input=learner.hold_input,
                        writer_baseline_clock=learner.writer_baseline_clock,
                        fresh=True, sensory_on=True, voice=tape['last_voice'],
                        executed_action=source == 'self_hearing',
                        reward_ticks=fl.reward_ticks_left(model, tape['last_reward'], 0),
                        ai_g=learner.ai_g.detach().clone(), ai_tape=[False] * duration,
                        read_every=True, eye_forced=eyes, update_expectation=False)
                    if eyes is None:
                        eyes = out[-4].detach().cpu().reshape(-1).tolist()
                    final = fl.expect_next(model, out[0], out[1])
                    logits = model.decoder(model.read_norm(out[1]))
                    results[label] = {'h': final.h.clone(), 'ring0': final.ring[0].clone(),
                                      'logits': logits.clone(), 'muscles': out[-1].clone(),
                                      'physical': {k: state_digest(v) for k, v in final.state_dict().items()},
                                      'seconds': time.perf_counter() - started, 'ticks': [dict(s) for s in stats]}
                    del current, out, final
                print(json.dumps({'arm': label, 'seconds': results[label]['seconds']}), flush=True)
                gc.collect()
    finally:
        fl.advance_sensory_terminals = original
        model.sensory_terminal_enabled = model.sensory_terminal_gating_enabled = True
        model.sensory_terminal_input_enabled = True

    def compare(a, b):
        ra, rb = results[a], results[b]
        sens = model.st_cell
        others = torch.ones(model.n_neurons, dtype=torch.bool)
        others[sens] = False
        return {'max_abs_h': float((ra['h'] - rb['h']).abs().max()),
                'pulse_mismatch_cells': int((ra['ring0'] != rb['ring0']).sum()),
                'pulse_mismatch_non_sensory_cells': int((ra['ring0'] != rb['ring0'])[0, others].sum()),
                'max_abs_logit': float((ra['logits'] - rb['logits']).abs().max()),
                'max_abs_muscle': float((ra['muscles'] - rb['muscles']).abs().max()),
                'physical_fields_bitwise_equal': ra['physical'] == rb['physical']}

    source_unchanged = (initial_digest == state_digest(learner.state.state_dict())
                        and weights_digest == state_digest(model.state_dict())
                        and torch.equal(initial_rng, torch.get_rng_state())
                        and source_sha == sha256(checkpoint))
    report = {'at': datetime.now(timezone.utc).isoformat(), 'checkpoint': str(checkpoint), 'sha256': source_sha,
              'trained': int(saved['bptt_train_tokens']), 'new_training_inputs': 0, 'learning': 'frozen',
              'clock_source': tape['scope'], 'ticks': duration, 'source': source,
              'first_tick_delivery': delivery,
              'comparisons': {'sites_ungated_vs_point': compare('sites_ungated', 'point'),
                              'active_vs_point': compare('active', 'point'),
                              'point_again_vs_point': compare('point_again', 'point')},
              'active_per_tick': results['active']['ticks'],
              'arm_seconds': {k: v['seconds'] for k, v in results.items()},
              'source_state_parameters_rng_preserved': source_unchanged,
              'anatomy': {k: model.sensory_terminal_manifest.get(k) for k in
                          ('cells', 'nodes', 'release_sites', 'incoming_contacts_executed', 'excluded')},
              'scope': 'Execution and delivery check of the sensory terminal pathway; no learning or behaviour benefit claim'}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('first_tick_delivery', 'comparisons', 'source_state_parameters_rng_preserved')}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.checkpoint, a.output)
