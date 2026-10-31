"""One matched-clock, frozen-state audit for all enabled native system arms.

Uses pending real-world words and saved mother RNG; does not call the mother,
train, mutate source, force speech, or interpret acute delivery as learning.
"""
from __future__ import annotations

import argparse
import gc
from types import SimpleNamespace
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.fly_interface.native_causal_audit import (
    restore_native, pending_queue_tape, clone_state, state_difference, states_close, sha256, ensure_gpu_idle,
)
from scripts.ib.fly_interface.action_loop_v2_audit import state_digest
from information_boltzmann.core.fly_native_generative import _preserve_rollout
from information_boltzmann.core import fly_bptt_learning as fl
from information_boltzmann.core.fly_active_inference import motor_logits


SYSTEM_FLAGS = {'cx': 'cx_enabled', 'mb_regulators': 'mb_consolidation_enabled',
                'historical_body_proxy': 'body_proxy_compatibility', 'al_relay': 'olfactory_relay_enabled',
                'amines': 'amine_enabled', 'visual': 'visual_graded_enabled',
                'endocrine': 'endocrine_enabled', 'motor_terminal': 'native_motor_terminal_enabled',
                'mb_terminal': 'mb_terminal_enabled', 'mb_terminal_gaba': 'mb_terminal_gaba_enabled',
                'dpm_5ht': 'mb_dpm_5ht_enabled', 'apl_release': 'apl_output_enabled',
                'apl_axial': 'apl_axial_enabled',
                'apl_calcium_entry': 'apl_calcium_entry_enabled',
                'apl_release_resources': 'apl_release_resource_enabled'}
SYSTEM_FLAGS.update(kc_lateral_delivery='kc_lateral_delivery_enabled', kc_lateral_calcium='kc_lateral_calcium_enabled', kc_lateral_camp='kc_lateral_camp_enabled', kc_sites='kc_terminal_gating_enabled', kc_site_transport='kc_terminal_transport_enabled', kc_site_stp='kc_site_stp_enabled',
                    kc_contact_write='kc_contact_write_enabled', kc_contact_read='kc_contact_read_enabled',
                    kc_dopamine_delivery='kc_dopamine_delivery_enabled', kc_dopamine_teacher='kc_dopamine_teacher_enabled',
                    dan_local_feedback='dan_local_feedback_enabled',dan_local_source='dan_local_source_enabled',
                    dan_carried_credit='dan_carried_credit_enabled', dan_receiver_credit='dan_receiver_credit_enabled')


@contextmanager
def intervene(model, system):
    """Restore exact flags/buffers even if the paired replay fails."""
    if system in ('intact', 'sham', 'restored'):
        yield
        return
    flag = SYSTEM_FLAGS[system]
    original = getattr(model, flag)
    mask = None
    if system == 'visual':
        mask = model.graded_mask
        changed = mask.clone()
        changed[:, model.visual_graded_index] = 0.0
        model.graded_mask = changed
    else:
        setattr(model, flag, False)
    try:
        yield
    finally:
        setattr(model, flag, original)
        if mask is not None:
            model.graded_mask = mask


@contextmanager
def deterministic_replay():
    """Fix accumulation order, without relaxing the paired-state tolerance.

    Multi-target CUDA index_add sums otherwise use atomics. Their rounding can
    contaminate a long sham replay's carried sensitivities. The audit changes
    execution order only; it preserves values, physical time and state.
    """
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    transmission = fl.execute_delayed_synaptic_transmission
    from information_boltzmann.core.triton_synapse import PyTorchDelayedSynapticTransmission

    def ordered_delayed(packets, pre, post, weight, splits):
        # The small modulatory route's custom Triton atomic kernel does not
        # honor torch's deterministic flag. Use its existing same-equation
        # PyTorch implementation for frozen controls, with ordered index_add.
        return PyTorchDelayedSynapticTransmission.apply(*packets, pre, post, weight, splits)

    if previous is None:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    fl.execute_delayed_synaptic_transmission = ordered_delayed
    try:
        yield
    finally:
        fl.execute_delayed_synaptic_transmission = transmission
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        if previous is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)


def audit(checkpoint, output, targets=4, *, historical_body_proxy=False, initial_site_extension=False, systems=None, initial_contact_extension=False, initial_dan_extension=False, device='cuda'):
    with deterministic_replay():
        return _audit(checkpoint, output, targets, historical_body_proxy=historical_body_proxy,
                      initial_site_extension=initial_site_extension, systems=systems,
                      initial_contact_extension=initial_contact_extension,initial_dan_extension=initial_dan_extension,device=device)


def initial_kc_delivery(model, state):
    """Only for zero-exposure uniform extension, not future local trajectories."""
    from information_boltzmann.core.fly_kc_terminals import kc_arrival
    from information_boltzmann.core.fly_apl_cable import _segmented
    rows={}
    def compare(name,left,right):
        rows[name]={'absmax':float((left-right).abs().max()),
                    'close':bool(torch.allclose(left,right,rtol=1e-5,atol=1e-6))}
    mask=torch.zeros_like(state.h).index_fill(1,model.kct_index,1.)
    compare('all_fast_KC_output',kc_arrival(model,state,'fast'),model.transmit(tuple(p*mask for p in state.ring),'e'))
    weights=model.comp_weight*(1+state.comp_memory[0])
    local=kc_arrival(model,state,'comp',weights=weights)
    local=torch.zeros_like(state.h).index_add(1,model.comp_post.long(),local)
    expected=fl.execute_delayed_synaptic_transmission(state.ring,model.comp_pre,model.comp_post,weights,model.comp_splits)
    compare('plastic_KC_MBON',local,expected)
    compare('KC_DAN_feedback',kc_arrival(model,state,'kcd',instant=True),state.ring[0].index_select(1,model.kcd_pre))
    expected=state.h.new_zeros(1,model.aplc_owner.numel())
    for lag in range(1,5):
        pre=getattr(model,f'aplc_in_e_pre_{lag}');value=state.ring[lag-1].index_select(1,pre)[0]
        value=value*torch.isin(pre,model.kct_index).to(value.dtype)
        value=value*model.edge_weight_e.index_select(0,getattr(model,f'aplc_in_e_edge_{lag}'))*getattr(model,f'aplc_in_e_fraction_{lag}')
        expected=expected+_segmented(model,'aplc_in_e',lag,value,expected.shape[1]).unsqueeze(0)
    compare('local_KC_APL_feedback',kc_arrival(model,state,'apl'),expected)
    return {'routes':rows,'status':'passed' if all(v['close'] for v in rows.values()) else 'not_accepted',
        'scope':'initial uniform expansion only, preserves existing outgoing signals; no future equivalence claim'}


def _audit(checkpoint, output, targets=4, *, historical_body_proxy=False, initial_site_extension=False, systems=None, initial_contact_extension=False, initial_dan_extension=False, device='cuda'):
    admission = ensure_gpu_idle([])
    output.mkdir(parents=True, exist_ok=True)
    before_sha = sha256(checkpoint)
    overrides = {'body_proxy_compatibility': True} if historical_body_proxy else None
    learner, saved, cfg, changed = restore_native(checkpoint, output, frozen=True, overrides=overrides, device=device)
    if changed:
        raise RuntimeError('Frozen system audit must not migrate physical clock/algorithm')
    model = learner.model
    initial_delivery=None
    if initial_site_extension:
        with torch.no_grad(),_preserve_rollout(model):initial_delivery=initial_kc_delivery(model,learner.state)
        (output/'initial_delivery.json').write_text(json.dumps(initial_delivery,indent=2),encoding='utf-8')
        if initial_delivery['status']!='passed':raise RuntimeError('KC initial route totals differ; inspect initial_delivery.json')
    contact_delivery = None
    if initial_contact_extension:
        from information_boltzmann.core.fly_kc_contact_memory import contact_arrival
        from information_boltzmann.core.fly_kc_terminals import kc_arrival
        with torch.no_grad():
            local = contact_arrival(model, learner.state)
            reference = kc_arrival(model, learner.state, 'comp',
                weights=model.comp_weight * (1 + learner.state.comp_memory[0]))
            contact_delivery = {'absmax': float((local-reference).abs().max()),
                'close': bool(torch.allclose(local, reference, rtol=1e-5, atol=1e-6))}
        if not contact_delivery['close']:
            raise RuntimeError('Initial contact extension changed retained efficacy')
    tape = pending_queue_tape(saved, targets, targets)[0]
    intervals = tape['forced']['intervals']
    inputs = [int(learner.previous_token)] + tape['targets'][:-1]
    job = getattr(learner, 'native_context_state', {}).get('pending_teaching') or {}
    first_source = job.get('source', tape['forced']['sources'][0])
    sources = [first_source] + tape['forced']['sources'][:-1]
    group_index = {'visual': getattr(model, 'visual_graded_index', torch.empty(0, dtype=torch.long, device=learner.state.h.device)),
                   'CX': getattr(model, 'cx_index', torch.empty(0, dtype=torch.long, device=learner.state.h.device)),
                   'KC': getattr(model, 'comp_kc', torch.empty(0, dtype=torch.long, device=learner.state.h.device)),
                   'MBON': torch.unique(model.comp_post.long()), 'DAN': model.dan_indices.long()}
    if hasattr(model, 'amine_pre'):
        for channel, name in enumerate(('OA', '5HT')):
            group_index[name] = torch.unique(model.amine_pre[model.amine_channel == channel])
    trained = int(saved['bptt_train_tokens'])
    # The complete trainer restore was exercised above. Frozen physical replay
    # never consumes optimizers, accumulated parameter scores or mother queues.
    # Retain exactly the physical replay interface, not duplicate training state.
    learner = SimpleNamespace(model=model, state=learner.state, ai_g=learner.ai_g,
        previous_token=learner.previous_token, hold_input=learner.hold_input,
        writer_baseline_clock=learner.writer_baseline_clock)
    del saved, cfg, job
    gc.collect()
    torch.cuda.empty_cache()
    eye_tapes = []

    def replay(record_eyes=False):
        current, muscles = clone_state(learner.state), learner.ai_g.detach().clone()
        if hasattr(model, 'amine_pre') and not current.amine_concentration.numel():
            # First activation has no prior chemical history. Give BOTH arms
            # the same neutral coordinates before an on/off comparison.
            current.amine_concentration = current.h.new_zeros((2, model.n_neurons))
            current.amine_receptor = current.h.new_zeros((2, model.n_neurons))
        if hasattr(model, 'mb_terminal_kc_index') and not current.mb_terminal_gaba.numel():
            nk = model.mb_terminal_kc_index.numel()
            current.mb_terminal_gaba = current.h.new_zeros((model.mb_terminal_source_index.numel(), nk))
            current.mb_dpm_concentration = current.h.new_zeros((model.mb_terminal_n_dpm, nk))
            current.mb_dpm_receptor = current.h.new_zeros((1, nk))
        log_probs, action_probs, command_hazards, packet_ticks = [], [], [], 0
        for i, (word, duration, source) in enumerate(zip(inputs, intervals, sources)):
            current = fl.prepare_sensory_expectation(model, current, word, source)
            out = fl.advance_fly_token_stream(model, current,
                torch.tensor([word], device=current.h.device), [duration], ticks=duration,
                hold_input=learner.hold_input, writer_baseline_clock=learner.writer_baseline_clock,
                fresh=True, sensory_on=True, voice=tape['last_voice'] if i == 0 else True,
                executed_action=source == 'self_hearing',
                reward_ticks=fl.reward_ticks_left(model, tape['last_reward'] if i == 0 else False, 0),
                ai_g=muscles, ai_tape=[False] * duration, read_every=True,
                eye_forced=None if record_eyes else eye_tapes[i], update_expectation=False)
            if record_eyes:
                eye_tapes.append(out[-4].detach().cpu().reshape(-1).tolist())
            current, muscles, latent = out[0], out[-1], out[1]
            lp = model.decoder(model.read_norm(latent)).float().log_softmax(-1)[0]
            log_probs.append(lp.detach().cpu())
            if getattr(model, 'native_motor_port_enabled', False):
                from information_boltzmann.core.fly_native_motor_port import action_logits
                action_probs.append(action_logits(model, out[3][-1:]).float().log_softmax(-1)[0].cpu())
                packet_ticks += int(out[4].sum())
                command_hazards.append((torch.sigmoid(motor_logits(model, muscles)[0,1]) * out[4][-1]).cpu())
            current = fl.expect_next(model, current, latent)
        return (current, muscles.detach().cpu(), torch.stack(log_probs), motor_logits(model, muscles).detach().cpu(),
                torch.stack(action_probs) if action_probs else torch.empty(0),
                torch.stack(command_hazards) if command_hazards else torch.empty(0), packet_ticks)

    parameter_digest = state_digest(model.state_dict())
    physical_digest = state_digest(learner.state.state_dict())
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    with _preserve_rollout(model), torch.no_grad():
        replay(record_eyes=True)
    baseline = None
    report = {'at': datetime.now(timezone.utc).isoformat(), 'checkpoint': str(checkpoint),
              'sha256': before_sha, 'trained': trained,
              'scope': 'acute frozen delivery, not learning benefit, real action consequence, exploration or self',
              'execution_device': device, 'cuda_training_admission': False,
              'initial_site_delivery': initial_delivery, 'initial_contact_delivery': contact_delivery,
              'historical_body_proxy': bool(historical_body_proxy),
              'body_scope': ('synthetic historical replay only; not a native peripheral loop'
                             if historical_body_proxy else
                             'native CNS motor language peripheral; no simulated muscle/proprioception'),
              'clock_source': tape['scope'], 'intervals': intervals, 'ticks': sum(intervals),
              'reduction_order': 'torch deterministic algorithms and ordered native delayed modulatory sums; unsupported operators fail closed',
              'inputs': inputs, 'targets': tape['targets'], 'sources': sources,
              'motor_control': 'mother is speaking, no commits; identical recorded sensory gain per tick',
              'gpu_admission': admission, 'arms': {},
              'enabled': {key: bool(getattr(model, flag, False)) for key, flag in SYSTEM_FLAGS.items()}}
    if initial_dan_extension:
        if not getattr(model,'dan_local_enabled',False):raise ValueError('No local DAN source to verify')
        from information_boltzmann.core.fly_dan_release import dan_state,dan_arrival
        from information_boltzmann.core.fly_kc_dopamine import spatial_arrival
        with torch.no_grad():
            values=dan_state(model,learner.state)
            new_global=dan_arrival(model,values)
            old_global=fl.execute_delayed_synaptic_transmission(learner.state.ring,
                model.dan_edge_pre,model.dan_edge_post,model.dan_edge_weight,model.dan_delay_splits)/model.dan_scale
            new_kc=spatial_arrival(model,learner.state.ring,dan_values=values)
            old_kc=spatial_arrival(model,learner.state.ring)
            close_global=torch.allclose(new_global,old_global,rtol=3e-5,atol=2e-6)
            close_kc=torch.allclose(new_kc,old_kc,rtol=3e-5,atol=2e-6)
            report['initial_dan_site_delivery']={'global_absmax':float((new_global-old_global).abs().max()),
                'KC_spatial_absmax':float((new_kc-old_kc).abs().max()),'close':close_global and close_kc,
                'scope':'Initial copied somatic AP queue only, no pre-existing local chemistry/resource history or future equivalence claim'}
            if not close_global or not close_kc:raise ValueError('Initial DAN AP packet extension changed route totals')
    if getattr(model,'kc_dopamine_enabled',False):
        from information_boltzmann.core.fly_kc_dopamine import spatial_arrival
        with torch.no_grad():
            located = spatial_arrival(model,learner.state.ring)
            means = located.new_zeros(model.kct_index.numel()).index_add(0,
                model.kct_owner,located[0]/model.kct_density)
            expected = fl.execute_delayed_synaptic_transmission(learner.state.ring,
                model.dan_edge_pre,model.dan_edge_post,model.dan_edge_weight,model.dan_delay_splits)
            expected = (expected/model.dan_scale)[0].index_select(0,model.kct_index)
            close = torch.allclose(means,expected,rtol=3e-5,atol=2e-6)
            report['initial_spatial_dopamine_delivery'] = {
                'volume_weighted_KC_mean_vs_original_graph_absmax':float((means-expected).abs().max()),
                'close':close,'scope':'same current spike queue only; no added experience or terminal feedback intervention'}
            if not close:
                raise ValueError('Spatial dopamine input fails graph-total conservation')
    selected = [key for key, on in report['enabled'].items() if on] if systems is None else list(systems)
    if any(key not in report['enabled'] or not report['enabled'][key] for key in selected):
        raise ValueError('Requested system is absent or disabled in this checkpoint')
    report['requested_systems'] = selected
    report['frozen_archive_scope'] = 'Complete restore first; discard optimizer/parameter-score copies for frozen physical replay only, never saved back.'
    arms = ['intact', 'sham'] + selected + ['restored']
    for arm in arms:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        replay_start = time.perf_counter()
        with _preserve_rollout(model), intervene(model, arm), torch.no_grad():
            current, muscles, lp, logits, action_lp, command_hazard, packet_ticks = replay()
        torch.cuda.synchronize()
        replay_seconds = time.perf_counter() - replay_start
        replay_peak_mib = torch.cuda.max_memory_allocated() / 1024**2
        ce = -lp[torch.arange(len(tape['targets'])), torch.tensor(tape['targets'])].mean()
        if baseline is None:
            baseline = current, muscles, lp, logits, ce, action_lp, command_hazard
        physical, base_muscles, base_lp, base_logits, base_ce, base_action_lp, base_hazard = baseline
        differences = state_difference(current, physical)
        group_diff = {key: float((current.h[:, idx] - physical.h[:, idx]).abs().max())
                      for key, idx in group_index.items() if idx.numel()}
        report['arms'][arm] = {'replay_seconds': replay_seconds, 'peak_allocated_mib': replay_peak_mib, 'ce': float(ce), 'ce_delta': float(ce - base_ce),
                              'state_close_to_intact': states_close(current, physical),
                              'log_prob_absmax': float((lp-base_lp).abs().max()),
                              'muscle_absmax': float((muscles-base_muscles).abs().max()),
                              'motor_logit_absmax': float((logits-base_logits).abs().max()),
                              'action_log_prob_absmax': (float((action_lp-base_action_lp).abs().max()) if action_lp.numel() else None),
                              'command_hazard_absmax': (float((command_hazard-base_hazard).abs().max()) if command_hazard.numel() else None),
                              'native_motor_packet_ticks': packet_ticks,
                              'physical_absmax_by_field': differences, 'membrane_group_absmax': group_diff}
        if current.amine_concentration.numel():
            report['arms'][arm]['amine_terminal_snapshot'] = {
                suffix: {'concentration_max': float(current.amine_concentration[channel].max()),
                         'occupancy_max': float(current.amine_receptor[channel].max()),
                         'occupied_cells': int((current.amine_receptor[channel] > 0).sum()),
                         'response_absmax': float(getattr(model, f'amine_response_{suffix}').abs().max())}
                for channel, suffix in enumerate(('oa', '5ht'))}
        if current.mb_terminal_gaba.numel():
            report['arms'][arm]['mb_terminal_snapshot'] = {
                'source_body_ids': model.mb_terminal_source_body_ids.cpu().tolist(),
                'gaba_max_by_source': current.mb_terminal_gaba.max(dim=1).values.cpu().tolist(),
                '5ht_concentration_max_by_dpm': current.mb_dpm_concentration.max(dim=1).values.cpu().tolist(),
                '5ht_receptor_max': float(current.mb_dpm_receptor.max()),
                'spatial_scope': model.mb_terminal_manifest['scope']}
            if current.kc_terminal_gaba.numel():
                report['arms'][arm]['mb_terminal_snapshot']['spatial_scope']='historical pooled snapshot retained, inactive; current local signals are in kc_terminal_snapshot'
        if current.apl_voltage.numel():
            report['arms'][arm]['apl_cable_snapshot'] = {
                'nodes': current.apl_voltage.numel(),
                'local_voltage_min': float(current.apl_voltage.min()),
                'local_voltage_max': float(current.apl_voltage.max()),
                'local_voltage_sd': float(current.apl_voltage.std()),
                'packet_nonzero_nodes': int((current.apl_packets[0] != 0).sum()),
                'scope': 'all supplied native APL tree nodes; KC bouton cable not reconstructed'}
        if current.kc_terminal_packets.numel():
            packet=current.kc_terminal_packets[0].cpu().numpy()
            owner=model.kct_site_owner.cpu().numpy()
            starts=np.r_[0,np.flatnonzero(np.diff(owner))+1]
            spread=np.maximum.reduceat(packet,starts)-np.minimum.reduceat(packet,starts)
            report['arms'][arm]['kc_terminal_snapshot']={
                'nodes':current.kc_terminal_gaba.shape[1], 'release_sites':len(packet),
                'kc_with_different_site_release':int((spread>1e-6).sum()),
                'within_kc_release_range_max':float(spread.max(initial=0.)),
                'gaba_effector_max':float(current.kc_terminal_gaba.max()),
                '5ht_effector_max':float(current.kc_terminal_5ht.max()),
                'local_eligibility_max':float(current.kc_terminal_eligibility.max()),
                'resources_min':float(current.kc_site_resources.min()) if current.kc_site_resources.numel() else None,
                'utilization_max':float(current.kc_site_utilization.max()) if current.kc_site_utilization.numel() else None,
                'scope':'actual sites and spatial effective receptor signals; not axonal action-potential cable or identified molecular localization'}
        print(json.dumps({'arm': arm, 'seconds': replay_seconds, 'peak_allocated_mib': replay_peak_mib}), flush=True)
        if current.kc_contact_memory.numel():
            from information_boltzmann.core.fly_kc_contact_memory import contact_arrival
            from information_boltzmann.core.fly_kc_terminals import kc_arrival
            with torch.no_grad():
                expressed = contact_arrival(model, current)
                reference = kc_arrival(model, current, 'comp',
                    weights=model.comp_weight*(1+current.comp_memory[0]))
                read_delivery = float((expressed-reference).abs().max())
            edge = model.kcm_edge.cpu().numpy()
            order = np.argsort(edge, kind='stable')
            starts = np.r_[0, np.flatnonzero(np.diff(edge[order]))+1]
            phi = current.kc_contact_memory[0].cpu().numpy()[order]
            spread = np.maximum.reduceat(phi, starts) - np.minimum.reduceat(phi, starts)
            report['arms'][arm]['contact_memory_snapshot'] = {
                'actual_MBON_input_vs_retained_reference_absmax': read_delivery,
                'connections_with_local_phi_difference': int((spread > 1e-7).sum()),
                'within_connection_phi_range_max': float(spread.max(initial=0)),
                'contacts': current.kc_contact_memory.shape[1],
                'phi_min': float(current.kc_contact_memory.min()),
                'phi_max': float(current.kc_contact_memory.max()),
                'trace_max': float(current.kc_contact_trace.max()),
                'teacher': ('actual DAN-KC contact-local receptor signal' if getattr(model,'kc_dopamine_enabled',False) else 'MBON compartment dopamine')}
        if current.kc_dopamine_signal.numel():
            from information_boltzmann.core.fly_kc_dopamine import contact_teacher
            local = contact_teacher(model,(current.kc_dopamine_signal,current.kc_dopamine_credit))[0]
            reference = current.dopamine.index_select(1,model.kcm_post)
            report['arms'][arm]['kc_dopamine_snapshot'] = {
                'nodes': current.kc_dopamine_signal.shape[1],
                'signal_max': float(current.kc_dopamine_signal[0].max()),
                'signal_sd': float(current.kc_dopamine_signal[0].std()),
                'contact_teacher_vs_MBON_reference_absmax': float((local-reference).abs().max())}
        if current.dan_local_calcium.numel():
            owner=model.danl_site_owner.cpu().numpy();order=np.argsort(owner,kind='stable')
            starts=np.r_[0,np.flatnonzero(np.diff(owner[order]))+1]
            packet=current.dan_local_packets[0].cpu().numpy()[order]
            spread=np.maximum.reduceat(packet,starts)-np.minimum.reduceat(packet,starts)
            report['arms'][arm]['dan_local_snapshot']={
                'nodes':current.dan_local_calcium.shape[1], 'sites':current.dan_local_resources.shape[1],
                'calcium_max':float(current.dan_local_calcium.max()),
                'receptor_max':float(current.dan_local_receptor.max()),
                'resources_min':float(current.dan_local_resources.min()),
                'source_cells_with_different_site_release':int((spread>1e-7).sum()),
                'within_DAN_release_range_max':float(spread.max(initial=0)),
                'scope':'actual KC reciprocal input and full supplied DAN trees; local effective receptor/calcium release, somatic AP still existing cell spikes'}
        if current.dan_credit_receptor.numel():
            from information_boltzmann.core.fly_dan_credit import DAN_CREDIT_FIELDS
            report['arms'][arm]['DAN_carried_snapshot'] = {
                name: {'shape': list(getattr(current, name).shape),
                       'absmax': float(getattr(current, name).abs().max())}
                for name in DAN_CREDIT_FIELDS}
        if current.dan_credit_dopamine.numel():
            report['arms'][arm]['DAN_receiver_credit_snapshot'] = {
                'shape': list(current.dan_credit_dopamine.shape),
                'absmax': float(current.dan_credit_dopamine.abs().max()),
                'scope': 'Source-resolved global DA filters; separate KC spatial history remains unrepresented'}
    # Use the shared audit's scale-aware Float32 state contract. A single
    # absolute epsilon is invalid for large forward-credit sensitivities.
    report['control_tolerance'] = {'state_rtol': 1e-5, 'state_atol': 1e-6,
                                   'log_prob_atol': 2e-5,
                                   'source': 'native_causal_audit.states_close'}
    controls = all(report['arms'][name]['state_close_to_intact']
                   and report['arms'][name]['log_prob_absmax'] <= 2e-5
                   and (report['arms'][name]['action_log_prob_absmax'] or 0.) <= 2e-5
                   and (report['arms'][name]['command_hazard_absmax'] or 0.) <= 2e-5
                   for name in ('sham', 'restored'))
    unchanged = (parameter_digest == state_digest(model.state_dict())
                 and physical_digest == state_digest(learner.state.state_dict())
                 and torch.equal(cpu_rng, torch.get_rng_state())
                 and torch.equal(cuda_rng, torch.cuda.get_rng_state())
                 and sha256(checkpoint) == before_sha)
    report.update(controls_passed=controls, source_and_rng_preserved=unchanged,
                  status='controls_passed' if controls and unchanged else 'not_accepted')
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'arms': len(arms), 'ticks': report['ticks']}))
    if not controls or not unchanged:
        raise RuntimeError('Matched system audit controls failed')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--targets', type=int, default=4)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--initial-contact-extension', action='store_true')
    parser.add_argument('--initial-dan-extension', action='store_true')
    parser.add_argument('--systems', default=None, help='Comma-separated targeted interventions; omitted checks all enabled systems')
    parser.add_argument('--initial-site-extension',action='store_true',help='Verify initial uniform KC route totals before interventions')
    parser.add_argument('--historical-body-proxy', action='store_true',
                        help='Explicitly reproduce rejected historical body proxies; frozen audit only')
    args = parser.parse_args()
    if args.targets < 1 or args.output.resolve() == args.checkpoint.parent.resolve():
        parser.error('Use positive targets and a separate audit directory')
    audit(args.checkpoint, args.output, args.targets,
          historical_body_proxy=args.historical_body_proxy,initial_site_extension=args.initial_site_extension,
          systems=args.systems.split(',') if args.systems else None,
          initial_contact_extension=args.initial_contact_extension,initial_dan_extension=args.initial_dan_extension,device=args.device)
