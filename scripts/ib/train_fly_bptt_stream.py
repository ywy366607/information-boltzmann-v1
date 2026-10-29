"""Continuous real-OWT COBA learning with 32-event truncated BPTT.

Resume physical life from the archived online learner or this branch. Active
validation uses the same learner and scores each target before its update.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_odour_world import (OdourWorld, attach_act_head, ACT_PARAMETER_NAMES,
                                                        attach_sense_head, SENSE_PARAMETER_NAMES)
ACT_STATE_NAMES = ACT_PARAMETER_NAMES + ('act_mu', 'act_sigma', 'act_w_scale') + SENSE_PARAMETER_NAMES
from information_boltzmann.core.fly_active_inference import attach_ai_head, AI_PARAMETER_NAMES, AI_STATE_NAMES
from information_boltzmann.core.fly_bptt_learning import (WRITER_SENSE, attach_input_gain, init_input_gain_from_size,
                                                          attach_kenyon_terminal_release, attach_kenyon_habituation, 
    FlyPhysicalState, FlyBPTTLearner, FlyBPTTGraph, STP_PARAMETER_NAMES,
    GO_PARAMETER_NAMES, attach_go_neuron, attach_connectome_go, READ_BANK_PARAMETER_NAMES, attach_read_bank,
    INTRINSIC_PARAMETER_NAMES, attach_intrinsic_drive, PRECISION_PARAMETER_NAMES, attach_precision,
    PREDICTIVE_PARAMETER_NAMES, attach_predictive_writer, DOPAMINE_PARAMETER_NAMES, attach_dopamine_modulation,
    AMINE_PARAMETER_NAMES,
    CX_PARAMETER_NAMES,
    MB_CONSOLIDATION_PARAMETER_NAMES,
    BODY_LOOP_PARAMETER_NAMES,
    AL_PARAMETER_NAMES,
    ONLINE_DA_PARAMETER_NAMES, attach_online_da_plasticity, attach_kenyon_dendritic_only, attach_homeostasis,
    attach_regime_homeostasis, attach_tied_writer, REGIME_KENYON_Z,
    attach_compartment_memory, attach_taste_heads, COMPARTMENT_PARAMETER_NAMES, TASTE_PARAMETER_NAMES,
    attach_stream_jepa, STREAM_JEPA_PARAMETER_NAMES, attach_mother_voice, attach_pacemaker_homeostasis, attach_comp_tonic_forget, attach_comp_timing, attach_comp_unit_traces, attach_comp_relative_burst, attach_comp_camp, calibrate_comp_rate, set_time_scale, COMP_TONIC_PARAMETER_NAMES, attach_reply_reward, attach_apl_kenyon_dendritic_only, attach_compartment_dendritic,
    renormalize_kenyon_inputs, pin_kenyon_to_dopamine, pin_sensory_terminals, pin_modulatory_fast_edges,
    reset_inputs_to_birth, attach_graded_cells, GRADED_PARAMETER_NAMES, attach_receptor_writer,
)
from information_boltzmann.core.fly_amine_modulation import attach_amine_modulation, AMINE_KINETICS_VERSION
from information_boltzmann.core.fly_cx_state import attach_cx_state
from information_boltzmann.core.fly_apl_cable import (
    attach_apl_cable, APL_CABLE_PARAMETER_NAMES, APL_CABLE_ID_NAMES, APL_CABLE_VERSION,
)
from information_boltzmann.core.fly_apl_local_release import (
    attach_apl_local_release, APL_LOCAL_RELEASE_PARAMETER_NAMES,
    APL_LOCAL_RELEASE_VERSION,
)
from information_boltzmann.core.fly_mb_consolidation import attach_mb_consolidation
from information_boltzmann.core.fly_dan_release import (attach_dan_release, DAN_RELEASE_VERSION, DAN_RELEASE_PARAMETER_NAMES, DAN_RELEASE_ID_NAMES)
from information_boltzmann.core.fly_kc_dopamine import (attach_kc_dopamine, KC_DOPAMINE_VERSION, KC_DOPAMINE_PARAMETER_NAMES, KC_DOPAMINE_ID_NAMES)
from information_boltzmann.core.fly_kc_lateral import (attach_kc_lateral, KC_LATERAL_VERSION,
    KC_LATERAL_PARAMETER_NAMES, KC_LATERAL_ID_NAMES)
from information_boltzmann.core.fly_sensory_terminals import (SENSORY_TERMINAL_VERSION,
    SENSORY_TERMINAL_PARAMETER_NAMES, SENSORY_TERMINAL_ID_NAMES)
from information_boltzmann.core.fly_kc_contact_memory import (
    attach_kc_contact_memory, KC_CONTACT_VERSION, KC_CONTACT_ID_NAMES)
from information_boltzmann.core.fly_kc_site_stp import (
    attach_kc_site_stp, KC_SITE_STP_PARAMETER_NAMES, KC_SITE_STP_VERSION)
from information_boltzmann.core.fly_kc_terminals import (
    attach_kc_terminals, KC_TERMINAL_PARAMETER_NAMES, KC_TERMINAL_ID_NAMES, KC_TERMINAL_VERSION,
)
from information_boltzmann.core.fly_mb_terminal import (
    attach_mb_terminal, MB_TERMINAL_PARAMETER_NAMES, MB_TERMINAL_ID_NAMES, MB_TERMINAL_VERSION,
    quarantine_mb_pooled_credit,
)
from information_boltzmann.core.fly_body_loop import (attach_body_loop, attach_body_actuator,
    BODY_ACTUATOR_PARAMETER_NAMES, quarantine_body_proxy_credit)
from information_boltzmann.core.fly_endocrine_exchange import attach_endocrine_exchange, ENDOCRINE_PARAMETER_NAMES
from information_boltzmann.core.fly_native_motor_port import (
    attach_native_motor_port, MOTOR_PORT_PARAMETER_NAMES, checkpoint_model_weights,
)
from information_boltzmann.core.fly_olfactory_relay import attach_olfactory_relay
from information_boltzmann.core.fly_visual_physiology import attach_visual_physiology, VISUAL_PARAMETER_NAMES
from information_boltzmann.core.fly_native_runtime import attach_program_document, validate_program_bindings
from information_boltzmann.core.fly_native_assembly import assemble_spatial_components
from information_boltzmann.core.fly_activity_audit import instantaneous_release_metrics
from information_boltzmann.runtime.lifelong_evaluation import recovery_summary, savings_summary
from information_boltzmann.runtime.prequential_meter import FirstPassPredictionMeter


def atomic_json(path, payload):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(40):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(0.25 * min(attempt + 1, 8))
    raise PermissionError(f"Could not replace {path}")


def preserve_committed_resource_window(error, checkpoint):
    """A post-update limit must save the paired brain/world before stopping."""
    if error is not None:
        checkpoint()
        raise MemoryError(error + '; complete committed-window checkpoint saved')


_CHECKPOINT_DIR = None


def atomic_save(path, payload):
    if _CHECKPOINT_DIR is not None:
        path = Path(_CHECKPOINT_DIR) / path.name
        path.parent.mkdir(parents=True, exist_ok=True)
    # The 1.1 GB checkpoint write attracts transient Windows file locks
    # (indexer/antivirus corrupting the tmp mid-write AND locking the fresh
    # file); retry the WHOLE save with fresh names until one lands cleanly.
    last_error = None
    for attempt in range(5):
        # Release dead replay graphs and allocator reservations before making
        # CPU staging copies of a complete GPU individual. Live values remain
        # untouched. In particular, do not retain a failed torch.save traceback
        # (and its large temporary storages) across the next retry.
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        temporary = path.with_suffix(f'.pt.tmp{attempt}')
        try:
            torch.save(payload, temporary)
            for retry in range(40):
                try:
                    os.replace(temporary, path)
                    return
                except PermissionError as error:
                    last_error = error
                    time.sleep(0.5 * min(retry + 1, 8))
            break
        except (RuntimeError, OSError, MemoryError) as error:
            last_error = type(error)(str(error))
            try:
                temporary.unlink()
            except OSError:
                pass
            time.sleep(2.0 * (attempt + 1))
    raise type(last_error)(f"atomic_save failed after retries: {last_error}")


# Spike physics of an individual; omitted flags inherit the saved value, and individuals born before these options
# existed get the values they were born with (audit 2026-10-13: biophysical reversals, Shiu spike semantics, STP).
BIOPHYSICS_DEFAULTS = {'E_E': 1.0, 'E_I': -0.2, 'spike_consumes_drive': False, 'refractory_ticks': 0, 'use_stp': True,
                       'inhibitory_tau_ratio': 2.0}


def graph_is_uniform_synapse(graph_npz):
    """True for a graph converted with one efficacy per synapse (weights integer multiples of the smallest):
    it has no sqrt(K_in) birth normalization, so --kc-renorm / --dan-renorm (which undo that normalization) do not
    apply (audit 2026-10-13)."""
    w = np.abs(np.load(graph_npz, mmap_mode='r')['edge_weight_e'][:200_000]).astype(np.float64)
    w = w[w > 0]
    r = w / w.min()
    # most edges (a few synapse classes, e.g. the Kenyon claws, carry a measured factor on top)
    return bool((np.abs(r - np.round(r)) < 1e-3).mean() > 0.9)


def modulatory_pin_kwargs(args):
    if not getattr(args, 'pin_modulatory_types', False):
        return {}
    from information_boltzmann.core.fly_bptt_learning import MODULATORY_TYPE_PATTERN
    return dict(type_pattern=MODULATORY_TYPE_PATTERN, annotations=args.graph.parent / 'body-annotations.feather')


def stp_names(model):
    return STP_PARAMETER_NAMES if getattr(model, 'use_stp', False) else ()


def biophysics_kwargs(args):
    return dict(E_E=args.E_E, E_I=args.E_I, spike_consumes_drive=args.spike_consumes_drive,
                refractory_ticks=args.refractory_ticks, use_stp=args.use_stp, inhibitory_tau_ratio=args.inhibitory_tau_ratio)


def resolve_learning_modes(args, previous_config, previous_learner):
    """Omitted flags inherit the continuing individual's algorithm."""
    if getattr(args, 'detach_reset', None) is None:
        args.detach_reset = bool(previous_config.get('detach_reset', False))
    if getattr(args, 'surrogate_mode', None) is None:
        args.surrogate_mode = previous_config.get('surrogate_mode', 'absolute')
    if getattr(args, 'transmission_mode', None) is None:
        args.transmission_mode = previous_config.get('transmission_mode', 'atomic')
    for key, old in BIOPHYSICS_DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, previous_config.get(key, old))
    if args.read_centering is None:
        args.read_centering = bool(previous_learner.get('read_centering',
                                  previous_config.get('read_centering', False)))
    if args.dan_plastic_lr is None:
        args.dan_plastic_lr = float(previous_learner.get('dan_plastic_lr',
                                   previous_config.get('dan_plastic_lr', 0.0)))
    if args.dan_plastic_lr < 0:
        raise ValueError('DAN plasticity learning rate must be nonnegative')
    if getattr(args, 'use_read_gamma_trace', None) is None:
        args.use_read_gamma_trace = bool(previous_learner.get('use_read_gamma_trace',
                                         previous_config.get('use_read_gamma_trace', False)))
    if getattr(args, 'init_read_gamma', None) is None:
        saved_gamma = previous_learner.get('init_read_gamma', previous_config.get('init_read_gamma', 'anatomical'))
        try:
            args.init_read_gamma = float(saved_gamma)
        except (ValueError, TypeError):
            args.init_read_gamma = saved_gamma
    if getattr(args, 'use_latent_predictor', None) is None:
        args.use_latent_predictor = bool(previous_learner.get('use_latent_predictor',
                                         previous_config.get('use_latent_predictor', False)))
    if getattr(args, 'use_graph_observer', None) is None:
        args.use_graph_observer = bool(previous_learner.get('use_graph_observer',
                                       previous_config.get('use_graph_observer', False)))
    if getattr(args, 'obs_init_gamma', None) is None:
        args.obs_init_gamma = float(previous_learner.get('obs_init_gamma',
                                     previous_config.get('obs_init_gamma', 0.0)))
    if getattr(args, 'ctm_loss', None) is None:
        args.ctm_loss = bool(previous_learner.get('use_ctm_loss',
                             previous_config.get('ctm_loss', True)))
    if getattr(args, 'lambda_mcr2', None) is None:
        args.lambda_mcr2 = float(previous_learner.get('lambda_mcr2',
                                 previous_config.get('lambda_mcr2', 0.0)))
    if getattr(args, 'eps_mcr2', None) is None:
        args.eps_mcr2 = float(previous_learner.get('eps_mcr2',
                              previous_config.get('eps_mcr2', 0.5)))
    if getattr(args, 'adaptive_admission', None) is None:
        args.adaptive_admission = bool(previous_learner.get('adaptive_admission',
                                       previous_config.get('adaptive_admission', False)))
    if getattr(args, 'stream_ticks', None) is None:
        args.stream_ticks = int(previous_learner.get('stream_ticks', previous_config.get('stream_ticks', 0)))
    old_segment = int(previous_learner.get('stream_segment_ticks', previous_config.get('stream_segment_ticks', 16)))
    if getattr(args, 'stream_segment_ticks', None) is None:
        args.stream_segment_ticks = old_segment
    if args.stream_segment_ticks < 1:
        raise ValueError('Stream recomputation segment must be positive')
    if getattr(args, 'hold_input', None) is None:
        args.hold_input = bool(previous_learner.get('hold_input', previous_config.get('hold_input', False)))
    if getattr(args, 'read_lag', None) is None:
        args.read_lag = int(previous_learner.get('read_lag', previous_config.get('read_lag', 0)))
    if getattr(args, 'go_neuron', None) is None:
        previous_go = previous_config.get('go_neuron', 'none')
        args.go_neuron = previous_go if isinstance(previous_go, str) else ('virtual' if previous_go else 'none')
    if args.go_neuron == 'none':
        args.go_neuron = None
    if getattr(args, 'dopamine_modulation', None) is None:
        args.dopamine_modulation = bool(previous_config.get('dopamine_modulation', False))
    if getattr(args, 'amine_modulation', None) is None:
        args.amine_modulation = bool(previous_config.get('amine_modulation', False))
    if args.amine_modulation and previous_config.get('amine_modulation'):
        if previous_config.get('amine_kinetics_version', 1) != AMINE_KINETICS_VERSION:
            raise ValueError('Enabled legacy amine kinetics need an explicit state/optimizer repair; '
                             'refusing to silently reinterpret learned time constants')
    if getattr(args, 'cx_state', None) is None:
        args.cx_state = bool(previous_config.get('cx_state', False))
    if getattr(args, 'mb_consolidation', None) is None:
        args.mb_consolidation = bool(previous_config.get('mb_consolidation', False))
    if getattr(args, 'mb_terminal_projection', None) is None:
        args.mb_terminal_projection = bool(previous_config.get('mb_terminal_projection', False))
    if (args.mb_terminal_projection and previous_config.get('mb_terminal_projection')
            and previous_config.get('mb_terminal_version') != MB_TERMINAL_VERSION):
        raise ValueError('MB terminal kinetics/schema version requires explicit migration')
    if getattr(args, 'apl_cable', None) is None:
        args.apl_cable = bool(previous_config.get('apl_cable', False))
    if args.apl_cable and previous_config.get('apl_cable') and previous_config.get('apl_cable_version') != APL_CABLE_VERSION:
        raise ValueError('APL cable version requires explicit full-state migration')
    if getattr(args, 'apl_local_release', None) is None:
        args.apl_local_release = bool(previous_config.get('apl_local_release', False))
    if args.apl_local_release and not args.apl_cable:
        raise ValueError('Local APL calcium release requires the native cable')
    if previous_config.get('apl_local_release') and (not args.apl_local_release or
            previous_config.get('apl_local_release_version') != APL_LOCAL_RELEASE_VERSION):
        raise ValueError('Continuing APL calcium release requires matching enabled history')
    if args.apl_local_release and previous_config.get('apl_cable') and not previous_config.get('apl_local_release'):
        raise ValueError('Use explicit full-state APL calcium migration before continuing')
    if getattr(args,'dan_local_release',None) is None:
        args.dan_local_release = bool(previous_config.get('dan_local_release',False))
    from information_boltzmann.core.fly_dan_credit import DAN_CREDIT_VERSION
    if getattr(args, 'dan_carried_credit', None) is None:
        args.dan_carried_credit = bool(previous_config.get('dan_carried_credit', False))
    if previous_config.get('dan_carried_credit') and (not args.dan_carried_credit or
            previous_config.get('dan_credit_version') != DAN_CREDIT_VERSION):
        raise ValueError('Continuing DAN credit requires matching enabled history schema')
    if args.dan_carried_credit and not args.dan_local_release:
        raise ValueError('DAN carried credit requires the local source')
    if args.dan_carried_credit and previous_config.get('dan_local_release') and not previous_config.get('dan_carried_credit'):
        raise ValueError('Use explicit DAN credit migration to record the historical derivative boundary')
    from information_boltzmann.core.fly_dan_receiver_credit import DAN_RECEIVER_CREDIT_VERSION
    if getattr(args, 'dan_receiver_credit', None) is None:
        args.dan_receiver_credit = bool(previous_config.get('dan_receiver_credit', False))
    if previous_config.get('dan_receiver_credit') and (not args.dan_receiver_credit or
            previous_config.get('dan_receiver_credit_version') != DAN_RECEIVER_CREDIT_VERSION):
        raise ValueError('Continuing DAN receiver credit requires its enabled history schema')
    if args.dan_receiver_credit and not args.dan_carried_credit:
        raise ValueError('DAN receiver history requires source material history')
    if args.dan_receiver_credit and previous_config.get('dan_carried_credit') and not previous_config.get('dan_receiver_credit'):
        raise ValueError('Use explicit receiver credit migration to record its derivative boundary')
    if getattr(args,'dan_release_geometry',None) is None:
        args.dan_release_geometry = (Path(previous_config['dan_release_geometry'])
            if previous_config.get('dan_release_geometry') else None)
    if previous_config.get('dan_local_release') and (not args.dan_local_release or
            previous_config.get('dan_release_version') != DAN_RELEASE_VERSION):
        raise ValueError('Continuing local DAN source requires matching enabled schema')
    if getattr(args, 'kc_dopamine', None) is None:
        args.kc_dopamine = bool(previous_config.get('kc_dopamine', False))
    if getattr(args, 'kc_dopamine_geometry', None) is None:
        args.kc_dopamine_geometry = (Path(previous_config['kc_dopamine_geometry'])
            if previous_config.get('kc_dopamine_geometry') else None)
    if previous_config.get('kc_dopamine') and (not args.kc_dopamine or
            previous_config.get('kc_dopamine_version') != KC_DOPAMINE_VERSION):
        raise ValueError('Continuing DAN-KC state requires matching enabled schema')
    if getattr(args, 'kc_contact_memory', None) is None:
        args.kc_contact_memory = bool(previous_config.get('kc_contact_memory', False))
    if previous_config.get('kc_contact_memory') and not args.kc_contact_memory:
        raise ValueError('Cannot silently revert contact learning; use matched read/write interventions or an explicit migration')
    if previous_config.get('kc_contact_memory') and previous_config.get('kc_contact_version') != KC_CONTACT_VERSION:
        raise ValueError('KC contact memory version mismatch')
    if getattr(args, 'kc_lateral', None) is None:
        args.kc_lateral = bool(previous_config.get('kc_lateral', False))
    if getattr(args, 'kc_lateral_geometry', None) is None:
        args.kc_lateral_geometry = (Path(previous_config['kc_lateral_geometry'])
                                    if previous_config.get('kc_lateral_geometry') else None)
    if args.kc_lateral and not args.kc_contact_memory:
        raise ValueError('KC lateral coupling requires native plastic contacts')
    if previous_config.get('kc_lateral') and (not args.kc_lateral or
            previous_config.get('kc_lateral_version') != KC_LATERAL_VERSION):
        raise ValueError('Continuing KC lateral receptors requires matching enabled history')
    if getattr(args, 'sensory_terminal', None) is None:
        args.sensory_terminal = bool(previous_config.get('sensory_terminal', False))
    if getattr(args, 'sensory_terminal_geometry', None) is None:
        args.sensory_terminal_geometry = (Path(previous_config['sensory_terminal_geometry'])
                                          if previous_config.get('sensory_terminal_geometry') else None)
    if previous_config.get('sensory_terminal') and (not args.sensory_terminal or
            previous_config.get('sensory_terminal_version') != SENSORY_TERMINAL_VERSION):
        raise ValueError('Continuing sensory terminal state requires the matching enabled schema')
    if args.sensory_terminal and previous_config and not previous_config.get('sensory_terminal'):
        raise ValueError('Use the full-state sensory terminal migration before enabling the pathway')
    if args.kc_lateral and previous_config.get('kc_contact_memory') and not previous_config.get('kc_lateral'):
        raise ValueError('Use full-state KC lateral migration before enabling the new pathway')
    if getattr(args, 'kc_site_stp', None) is None:
        args.kc_site_stp = bool(previous_config.get('kc_site_stp', False))
    if args.kc_site_stp and previous_config.get('kc_site_stp') and previous_config.get('kc_site_stp_version') != KC_SITE_STP_VERSION:
        raise ValueError('KC site STP version mismatch')
    if getattr(args, 'kc_terminal', None) is None:
        args.kc_terminal = bool(previous_config.get('kc_terminal', False))
    if getattr(args, 'kc_terminal_geometry', None) is None:
        args.kc_terminal_geometry = Path(previous_config['kc_terminal_geometry']) if previous_config.get('kc_terminal_geometry') else None
    if args.kc_terminal and previous_config.get('kc_terminal') and previous_config.get('kc_terminal_version') != KC_TERMINAL_VERSION:
        raise ValueError('KC terminal version requires explicit full-state migration')
    if getattr(args, 'body_loop', None) is None:
        args.body_loop = bool(previous_config.get('body_loop', False))
    # Never inherit the historical proxy path as native physiology. Frozen
    # historical audits opt in explicitly; ordinary training cannot use it.
    args.body_proxy_compatibility = bool(getattr(args, 'body_proxy_compatibility', False))
    if args.body_proxy_compatibility and not getattr(args, 'native_audit_frozen', False):
        raise ValueError('Synthetic body feedback is quarantined; compatibility is for frozen historical audits only')
    if getattr(args, 'olfactory_relay', None) is None:
        args.olfactory_relay = bool(previous_config.get('olfactory_relay', False))
    if getattr(args, 'visual_graded', None) is None:
        args.visual_graded = bool(previous_config.get('visual_graded', False))
    if getattr(args, 'visual_graded_gain', None) is None:
        args.visual_graded_gain = float(previous_config.get('visual_graded_gain') or 0.025)
    if getattr(args, 'body_actuator', None) is None:
        args.body_actuator = bool(previous_config.get('body_actuator', False))
    if getattr(args, 'endocrine_exchange', None) is None:
        args.endocrine_exchange = bool(previous_config.get('endocrine_exchange', False))
    if getattr(args, 'native_motor_port', None) is None:
        args.native_motor_port = bool(previous_config.get('native_motor_port', False))
    if getattr(args, 'dopamine_site', None) is None:
        args.dopamine_site = previous_config.get('dopamine_site') or 'post'
    if getattr(args, 'dopamine_adapted', None) is None:
        args.dopamine_adapted = bool(previous_config.get('dopamine_adapted', False))
    if getattr(args, 'input_gain_size_exponent', None) is None:
        args.input_gain_size_exponent = previous_config.get('input_gain_size_exponent')
    if getattr(args, 'kc_dan_release', None) is None:
        args.kc_dan_release = bool(previous_config.get('kc_dan_release', False))
    if getattr(args, 'kc_habituation_tau', None) is None:
        args.kc_habituation_tau = previous_config.get('kc_habituation_tau')
    if getattr(args, 'kc_habituation_use', None) is None:
        args.kc_habituation_use = previous_config.get('kc_habituation_use') or 0.2
    if getattr(args, 'read_extra_classes', None) is None:
        args.read_extra_classes = previous_config.get('read_extra_classes') or ''
    if getattr(args, 'read_spikes', None) is None:
        args.read_spikes = bool(previous_config.get('read_spikes', False))
    if args.read_spikes:
        args.use_read_gamma_trace = True
    if getattr(args, 'evidence_plasticity', None) is None:
        args.evidence_plasticity = bool(previous_config.get('evidence_plasticity', False))
    if getattr(args, 'pull_tau', None) is None:
        args.pull_tau = float(previous_config.get('pull_tau') or 0.0)
    if getattr(args, 'eg_plasticity', None) is None:
        args.eg_plasticity = bool(previous_config.get('eg_plasticity', False))
    if getattr(args, 'eg_tau', None) is None:
        args.eg_tau = float(previous_config.get('eg_tau') or 4096.0)
    if getattr(args, 'online_da_plasticity', None) is None:
        args.online_da_plasticity = bool(previous_config.get('online_da_plasticity', False))
    if getattr(args, 'dopamine_affinity', None) is None:
        args.dopamine_affinity = previous_config.get('dopamine_affinity')
    if getattr(args, 'homeostasis', None) is None:
        args.homeostasis = bool(previous_config.get('homeostasis', False))
    if getattr(args, 'homeostasis_seed', None) is None:
        args.homeostasis_seed = int(previous_config.get('homeostasis_seed') or 0)
    if getattr(args, 'regime_homeostasis', None) is None:
        args.regime_homeostasis = bool(previous_config.get('regime_homeostasis', False))
    if getattr(args, 'regime_seed', None) is None:
        args.regime_seed = int(previous_config.get('regime_seed') or 0)
    if getattr(args, 'regime_kenyon_z', None) is None:
        args.regime_kenyon_z = previous_config.get('regime_kenyon_z')
    if getattr(args, 'tied_writer', None) is None:
        args.tied_writer = bool(previous_config.get('tied_writer', False))
    args.kc_renorm_newly = bool(getattr(args, 'kc_renorm', False)) and not bool(previous_config.get('kc_renorm', False))
    for name in ('modulatory_classes', 'dan_renorm'):
        if getattr(args, name, None) is None:
            setattr(args, name, bool(previous_config.get(name, False)))
    args.dan_renorm_newly = bool(args.dan_renorm) and not bool(previous_config.get('dan_renorm', False))
    if getattr(args, 'intrinsic_pattern', None) is None:
        args.intrinsic_pattern = previous_config.get('intrinsic_pattern')
    if getattr(args, 'intrinsic_kappa', None) is None:
        args.intrinsic_kappa = previous_config.get('intrinsic_kappa')
    if getattr(args, 'connectome_v2', None) is None:
        args.connectome_v2 = bool(previous_config.get('connectome_v2', False))
    # The resolver is also used with partial namespaces by restore interfaces.
    args.al_reset = bool(getattr(args, 'al_reset', False))
    args.al_reset_newly = args.al_reset and not bool(previous_config.get('al_reset', False))
    if previous_config.get('al_reset'):
        args.al_reset = True
    if getattr(args, 'kc_renorm', None) is None:
        args.kc_renorm = bool(previous_config.get('kc_renorm', False))
    if getattr(args, 'relative_edge_rate', None) is None:
        args.relative_edge_rate = float(previous_config.get('relative_edge_rate') or 0.0)
    if getattr(args, 'odour_world', None) is None:
        args.odour_world = bool(previous_config.get('odour_world', False))
    if getattr(args, 'ai', None) is None:
        args.ai = bool(previous_config.get('ai', False))
    if getattr(args, 'eye_binary', None) is None:
        args.eye_binary = bool(previous_config.get('eye_binary', False))
    if getattr(args, 'ai_depth', None) is None:
        args.ai_depth = int(previous_config.get('ai_depth') or 0)
    for name, default in (('motor_acts', False), ('act_brain_grad', False), ('miss_prior', False), ('kc_subtypes', False),
                          ('act_dopamine_precision', False), ('epistemic_belief', False),
                          ('carried_credit', False)):
        if getattr(args, name, None) is None:
            setattr(args, name, bool(previous_config.get(name, default)))
    for name, default in (('act_tau', 8.0), ('act_habit', 0.2), ('act_gamma_inv', 0.05)):
        if getattr(args, name, None) is None:
            setattr(args, name, float(previous_config.get(name) or default))
    if args.dan_carried_credit and not args.carried_credit:
        raise ValueError('DAN history requires --carried-credit')
    if getattr(args, 'sense_noise', None) is None:
        args.sense_noise = float(previous_config.get('sense_noise') or 0.0)
    if getattr(args, 'eye_init', None) is None:
        args.eye_init = float(previous_config.get('eye_init') or 0.95)
    if getattr(args, 'eye_gate', None) is None:
        args.eye_gate = str(previous_config.get('eye_gate') or 'atan')
    if getattr(args, 'noise_rate', None) is None:
        args.noise_rate = float(previous_config.get('noise_rate') or 0.0)
    for name, default in (('ai_epistemic', 0.0), ('ai_pragmatic', 0.0), ('ai_sigreg', 0.0)):
        if getattr(args, name, None) is None:
            setattr(args, name, float(previous_config.get(name) or default))
    if getattr(args, 'sense_prediction', None) is None:
        args.sense_prediction = bool(previous_config.get('sense_prediction', False))
    if getattr(args, 'compartment_memory', None) is None:
        args.compartment_memory = bool(previous_config.get('compartment_memory', False))
    if getattr(args, 'compartment_adapted', None) is None:
        args.compartment_adapted = bool(previous_config.get('compartment_adapted', False))
    if getattr(args, 'mother_reward', None) is None:
        args.mother_reward = bool(previous_config.get('mother_reward', False))
    if getattr(args, 'mother_reward_level', None) is None:
        args.mother_reward_level = float(previous_config.get('mother_reward_level') or 1.0)
    if getattr(args, 'mother_reward_ticks', None) is None:
        args.mother_reward_ticks = int(previous_config.get('mother_reward_ticks') or 4)
    if getattr(args, 'homeostat_skip_dans', None) is None:
        args.homeostat_skip_dans = bool(previous_config.get('homeostat_skip_dans', False))
    if getattr(args, 'pacemaker_homeostasis', None) is None:
        args.pacemaker_homeostasis = bool(previous_config.get('pacemaker_homeostasis', False))
    if getattr(args, 'dan_kappa', None) is None:
        args.dan_kappa = previous_config.get('dan_kappa', None)
    if getattr(args, 'comp_burst_k', None) is None:
        args.comp_burst_k = previous_config.get('comp_burst_k', None)
    if getattr(args, 'time_scale', None) is None:
        args.time_scale = float(previous_config.get('time_scale') or 1.0)
    if getattr(args, 'comp_camp_tau', None) is None:
        args.comp_camp_tau = previous_config.get('comp_camp_tau', None)
    if getattr(args, 'comp_burst_relative', None) is None:
        args.comp_burst_relative = bool(previous_config.get('comp_burst_relative', False))
    if getattr(args, 'comp_unit_traces', None) is None:
        args.comp_unit_traces = bool(previous_config.get('comp_unit_traces', False))
    if getattr(args, 'comp_timing', None) is None:
        args.comp_timing = bool(previous_config.get('comp_timing', False))
    if getattr(args, 'comp_tonic_forget', None) is None:
        args.comp_tonic_forget = float(previous_config.get('comp_tonic_forget') or 0.0)
    if getattr(args, 'intrinsic_mbon_kappa', None) is None:
        args.intrinsic_mbon_kappa = float(previous_config.get('intrinsic_mbon_kappa') or 0.0)
    if getattr(args, 'sense_gains_learnable', None) is None:
        args.sense_gains_learnable = bool(previous_config.get('sense_gains_learnable', False))
    if getattr(args, 'mother_credit_event_clock', None) is None:
        args.mother_credit_event_clock = bool(previous_config.get('mother_credit_event_clock', False))
    if getattr(args, 'mother_credit_brain_tau', None) is None:
        args.mother_credit_brain_tau = bool(previous_config.get('mother_credit_brain_tau', False))
    if getattr(args, 'mother_credit_differential', None) is None:
        args.mother_credit_differential = bool(previous_config.get('mother_credit_differential', False))
    if getattr(args, 'mother_credit_trace', None) is None:
        args.mother_credit_trace = bool(previous_config.get('mother_credit_trace', False))
    if getattr(args, 'mother_consequence_native', None) is None:
        args.mother_consequence_native = bool(previous_config.get('mother_consequence_native', False))
    if getattr(args, 'mother_native_credit', None) is None:
        args.mother_native_credit = bool(previous_config.get('mother_native_credit', False))
    if getattr(args, 'brain_learning', None) is None:
        args.brain_learning = previous_config.get('brain_learning') or 'bptt'
    if getattr(args, 'native_credit_cpu', None) is None:
        args.native_credit_cpu = bool(previous_learner.get('native_credit_cpu', previous_config.get('native_credit_cpu', False)))
    if getattr(args, 'mother_action_clock', None) is None:
        args.mother_action_clock = previous_config.get('mother_action_clock', 'arrival')
    if getattr(args, 'mother_generative_shadow', None) is None:
        args.mother_generative_shadow = bool(previous_config.get('mother_generative_shadow', False))
    if getattr(args, 'mother_native_generative', None) is None:
        args.mother_native_generative = bool(previous_config.get('mother_native_generative', False))
    if getattr(args, 'mother_native_context', None) is None:
        args.mother_native_context = bool(previous_config.get('mother_native_context', False))
    for key, default in (('mother_context_initial_variance', 1.), ('mother_context_process_variance', 1e-4)):
        if getattr(args, key, None) is None:
            setattr(args, key, float(previous_config.get(key, default)))
    if args.mother_native_context:
        if args.mother_action_clock != 'commit' or args.mother_credit != 'consequence':
            raise ValueError('Native context requires commit clock and consequence credit')
        if args.mother_native_generative or args.mother_native_credit or args.mother_consequence_native:
            raise ValueError('Native context is an explicit migration; disable other policy candidates')
    if args.stream_segment_ticks != old_segment and not args.mother_native_context:
        raise ValueError('Changing recomputation segments requires native-context endpoint expectation semantics')
    if args.mother_native_generative:
        if args.mother_action_clock != 'commit' or args.mother_credit != 'consequence':
            raise ValueError('Native generative control requires commit clock and consequence credit')
        if args.mother_native_credit or args.mother_consequence_native:
            raise ValueError('Disable legacy mother-native-credit and mother-consequence-native explicitly')
    if args.mother_generative_shadow and args.mother_action_clock != 'commit':
        raise ValueError('--mother-generative-shadow requires --mother-action-clock commit')
    if args.mother_native_credit and args.mother_consequence_native:
        raise ValueError('Native credit alone and the ordered consequence candidate are separate configurations')
    if args.mother_native_credit and args.mother_credit != 'consequence':
        raise ValueError('--mother-native-credit requires --mother-credit consequence')
    if getattr(args, 'mother_voice', None) is None:
        args.mother_voice = bool(previous_config.get('mother_voice', False))
    if getattr(args, 'mother_voice_level', None) is None:
        args.mother_voice_level = float(previous_config.get('mother_voice_level') or 1.5)
    if getattr(args, 'stream_jepa', None) is None:
        args.stream_jepa = bool(previous_config.get('stream_jepa', False))
    if getattr(args, 'stream_jepa_weight', None) is None:
        args.stream_jepa_weight = float(previous_config.get('stream_jepa_weight') if previous_config.get('stream_jepa_weight') is not None else 0.5)
    args.kc_dendritic_newly = bool(getattr(args, 'kc_dendritic_only', None)) and not bool(previous_config.get('kc_dendritic_only', False))
    args.apl_kc_dendritic_newly = bool(getattr(args, 'apl_kc_dendritic_only', None)) and not bool(previous_config.get('apl_kc_dendritic_only', False))
    args.compartment_dendritic_newly = bool(getattr(args, 'compartment_dendritic', None)) and not bool(previous_config.get('compartment_dendritic', False))
    if getattr(args, 'compartment_dendritic', None) is None:
        args.compartment_dendritic = bool(previous_config.get('compartment_dendritic', False))
    if getattr(args, 'apl_kc_dendritic_only', None) is None:
        args.apl_kc_dendritic_only = bool(previous_config.get('apl_kc_dendritic_only', False))
    if getattr(args, 'kc_dendritic_only', None) is None:
        args.kc_dendritic_only = bool(previous_config.get('kc_dendritic_only', False))
    if getattr(args, 'predictive_writer', None) is None:
        args.predictive_writer = bool(previous_config.get('predictive_writer', False))
    if getattr(args, 'precision', None) is None:
        args.precision = bool(previous_config.get('precision', False))
    if getattr(args, 'intrinsic', None) is None:
        args.intrinsic = bool(previous_config.get('intrinsic', False))
    if getattr(args, 'graded_types', None) is None:
        args.graded_types = previous_config.get('graded_types')
    if getattr(args, 'writer', None) is None:
        args.writer = previous_config.get('writer', 'topographic')
    if getattr(args, 'type_classes', None) is None:
        args.type_classes = bool(previous_config.get('type_classes', False))
    if getattr(args, 'pin_modulatory_types', None) is None:
        args.pin_modulatory_types = bool(previous_config.get('pin_modulatory_types', False))
    if getattr(args, 'intrinsic_jitter', None) is None:
        args.intrinsic_jitter = previous_config.get('intrinsic_jitter', 0.0)
    if getattr(args, 'read_bank', None) is None:
        args.read_bank = bool(previous_config.get('read_bank', False))
    if getattr(args, 'go_max_ticks', None) is None:
        args.go_max_ticks = int(previous_learner.get('go_max_ticks', previous_config.get('go_max_ticks', 32)))
    if getattr(args, 'go_tick_cost', None) is None:
        args.go_tick_cost = float(previous_learner.get('go_tick_cost', previous_config.get('go_tick_cost', 0.0)))
    if getattr(args, 'go_reference_energy', None) is None:
        args.go_reference_energy = float(previous_learner.get('go_reference_energy',
                                                              previous_config.get('go_reference_energy', 0.07)))
    if args.go_neuron and args.stream_ticks:
        raise ValueError('The go neuron (self-paced) and the streaming clock (externally paced) exclude each other')
    if args.stream_ticks or args.go_neuron:
        args.adaptive_admission = False          # the streaming clock never waits for settling
    if getattr(args, 'min_settle_ticks', None) is None:
        args.min_settle_ticks = int(previous_learner.get('min_settle_ticks',
                                    previous_config.get('min_settle_ticks', 3)))
    if getattr(args, 'flux_baseline', None) is None:
        args.flux_baseline = float(previous_learner.get('flux_baseline',
                                   previous_config.get('flux_baseline', 0.048)))
    if args.native_motor_port and not (getattr(args, 'mother_native_context', False)
            and args.mother_action_clock == 'commit' and args.mother_credit == 'consequence'):
        raise ValueError('Native motor port requires native-context consequence learning at the actual commit clock')


# The mainline individual (2026-10-13).  A chain names only what it changes; at birth every switch not named comes from
# here, on a resume from the checkpoint's own config (a switch the checkpoint never had takes the parser default, so an
# old individual is never silently given a new mechanism).  Switches went missing when each chain carried its own copy
# of a long flag string (the predictive writer, validated 10-11, was absent from R1-R3).
PROFILES = {
    'mainline': (
        # substrate: Shiu-scale connectome r2 (rescaled, sensory terminals out, antennal-lobe transmitters, DPM GABA)
        '--graph data/malecns_v1/fly_reservoir_coba_r2.npz --kc-dendritic-only --connectome-v2 --pin-modulatory-types '
        '--type-classes --intrinsic --intrinsic-jitter 0.1 --graded-types APL,CT1 --graded-gain 0.025 '
        # neuron engine (Shiu replication)
        '--E-E 0.55 --E-I -0.15 --spike-consumes-drive --refractory-ticks 1 --inhibitory-tau-ratio 4 --no-stp '
        '--alif-beta-init 0.2 --surrogate-mode threshold --detach-reset --transmission-mode incoming '
        # senses and world: text through every sensory cell, held for the word, prediction error written (efference)
        '--writer receptor_lang --hold-input --predictive-writer --writer-baseline-clock physical '
        '--stream-ticks 8 --stream-pace-mean 16 --stream-pace-min 2 --go-neuron none '
        # learning: synapses by normalized EG with the birth connectome as prior; class physiology frozen at birth;
        # dopamine phasic (adapted), presynaptic site; Kenyon -> MBON three-factor memory
        # class physiology (incl. the per-class input gain, started halfway to inverse-size) learns, evidence-gated
        '--eg-plasticity --eg-tau 4096 --relative-edge-rate 0.02 --lr-physiology 5e-3 --evidence-plasticity '
        '--input-gain-size-exponent 0.5 '
        '--dopamine-modulation --dopamine-site pre --dopamine-adapted --compartment-memory --comp-rate 2.0 '
        '--carried-credit --no-homeostasis --no-ctm-loss '
        # budgets and machine
        '--window 32 --no-graph --validate-every-tokens 4096 --eval-tokens 512 --replay-tokens 128 --vram-limit-mib 3800'),
    'none': '',
}


def _option_groups(tokens):
    groups, cur = [], None
    for t in tokens:
        if t.startswith('--'):
            cur = [t]
            groups.append(cur)
        elif cur is not None:
            cur.append(t)
    return groups


def explicit_dests(parser, argv):
    """The destinations the command line names (--x, --no-x, --x=v)."""
    table = parser._option_string_actions
    return {table[t.split('=')[0]].dest for t in argv if t.startswith('--') and t.split('=')[0] in table}


def apply_profile(parser, argv, name):
    """Parse argv on top of a profile: profile options whose destination the command line does not name come first."""
    table = parser._option_string_actions
    explicit = explicit_dests(parser, argv)
    tokens = []
    for group in _option_groups(PROFILES[name].split()):
        if table[group[0]].dest not in explicit:
            tokens += group
    args = parser.parse_args(tokens + argv)
    args.profile_dests = sorted({table[g[0]].dest for g in _option_groups(PROFILES[name].split())})
    args.explicit_dests = sorted(explicit)
    return args


def speak_credit_tau(model, args):
    """The speaking credit's decay: in ticks, or in mother words on the event clock (--mother-credit-event-clock).
    --mother-credit-brain-tau: the brain's own eligibility clock (the Kenyon trace time constant, learnable), converted
    to words by the world's mean word interval on the event clock."""
    brain = (float(-1.0 / torch.log(torch.sigmoid(model.comp_logit_trace.detach())))
             if args.mother_credit_brain_tau and getattr(model, 'comp_logit_trace', None) is not None else None)
    if args.mother_credit_event_clock:
        return brain / max(float(args.stream_pace_mean or 8.0), 1.0) if brain else float(args.mother_credit_tau_words)
    return brain if brain else args.mother_credit_tau


def make_sense_gains(model, args, saved_model=None):
    """--sense-gains-learnable: Johnston's organ's gain for the mother's voice and the PAM cells' sensitivity to her
    answer become learnable log gains (start at --mother-voice-level / --mother-reward-level), created before the
    optimizer so they train and resume.  Returns (names, names new to this individual)."""
    if not getattr(args, 'sense_gains_learnable', False) or not args.mother_world:
        return [], []
    made, new = [], []
    for name, level, on in (('voice_log_gain', args.mother_voice_level or 1.5, args.mother_voice),
                            ('reward_log_gain', args.mother_reward_level or 1.0, args.mother_reward)):
        if not on:
            continue
        known = saved_model is not None and name in saved_model
        value = saved_model[name].float() if known else torch.tensor(math.log(float(level)))
        setattr(model, name, torch.nn.Parameter(value.to(model.read_indices.device).clone()))
        made.append(name)
        if not known:
            new.append(name)
    return made, new


def attach_mother_senses(model, args):
    """Restore the world's physical sensory wiring without opening a network/world.

    The indices are nonpersistent buffers. Restoring weights alone does not
    restore Johnston input or PAM reply input; audits and training share this.
    Existing learned gain Parameters are preserved by the attach helpers.
    """
    routes = {}
    if not args.mother_world:
        return routes
    annotations = args.graph.parent / 'body-annotations.feather'
    if args.mother_reward:
        routes['reward_cells'] = attach_reply_reward(
            model, args.graph, annotations, learnable=bool(args.sense_gains_learnable),
            level=float(args.mother_reward_level or 1.0), ticks=int(args.mother_reward_ticks or 4))
        print(f'PAM reply sense restored: {routes["reward_cells"]} cells, {model.reward_ticks} ticks', flush=True)
    if args.mother_voice:
        routes['voice_cells'] = attach_mother_voice(
            model, args.graph, annotations, learnable=bool(args.sense_gains_learnable),
            level=float(args.mother_voice_level or 1.5))
        print(f'Mother voice sense restored: {routes["voice_cells"]} Johnston cells', flush=True)
    return routes


def inherit_profile_switches(parser, args, previous_config):
    """On a resume every profile switch the command line does not name is the checkpoint's own (or, if the
    checkpoint never had it, the parser default): a chain cannot drop or add a mechanism by omission."""
    actions = {a.dest: a for a in parser._actions}
    changed = []
    for dest in args.profile_dests:
        if dest in args.explicit_dests:
            continue
        value = previous_config.get(dest, actions[dest].default)
        if actions[dest].type is Path and isinstance(value, str):
            value = Path(value)
        if getattr(args, dest) != value:
            changed.append(dest)
        setattr(args, dest, value)
    return changed


def print_switches(parser, args, source):
    """One table of every profile switch: value, where it came from, and whether it differs from the mainline."""
    main = parser.parse_args(PROFILES['mainline'].split() + ['--from-scratch'])
    rows = []
    for dest in args.profile_dests:
        value = getattr(args, dest)
        how = 'command line' if dest in args.explicit_dests else source
        flag = '' if str(value) == str(getattr(main, dest)) else f'   <-- mainline: {getattr(main, dest)}'
        rows.append(f'  {dest:28s} {str(value):48s} [{how}]{flag}')
    print('Switches (' + source + '):' + chr(10) + chr(10).join(rows), flush=True)


def read_extra_index(graph_npz, classes):
    """Indices of the cells of the given annotation classes (e.g. 'Kenyon_Cell,MBON'), read besides the output cells."""
    if not classes:
        return None
    import pandas as pd
    ids = np.load(graph_npz, allow_pickle=True)['neuron_body_ids']
    cls = pd.read_feather(Path(graph_npz).parent / 'body-annotations.feather').set_index('bodyId').reindex(ids)['class']
    return np.flatnonzero(cls.astype(str).isin([c.strip() for c in classes.split(',')]).values)


def disable_alif(model):
    """beta = clamp(exp(log_beta), 1e-4, 2): log_beta = -20 sits below the clamp, so beta = 1e-4 and its gradient is 0."""
    if not getattr(model, 'use_alif', False):
        raise ValueError('This model has no ALIF to disable')
    with torch.no_grad():
        model.log_beta.fill_(-20.0)


def main():
    if os.environ.get('FLY_DETECT_ANOMALY'):
        torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, default=None,
                        help='Resume from a complete lifecycle checkpoint')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Initialize a newborn individual from scratch using pretrained embeddings and unigram prior')
    parser.add_argument('--pretrained-embedding', type=Path, default=Path('data/gpt2_model.safetensors'))
    parser.add_argument('--d-model', type=int, default=768)
    parser.add_argument('--decoder-bias', dest='decoder_bias', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--read-norm-init', type=float, default=0.1)
    parser.add_argument('--output', type=Path, default=Path('results/q8_fly_bptt32_continuous_100k'))
    parser.add_argument('--data', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--reference-train', type=Path, default=None,
                        help='Fixed train-only unigram reporting reference; retained across resumes')
    parser.add_argument('--graph', type=Path, default=Path('data/malecns_v1/fly_reservoir_coba.npz'))
    parser.add_argument('--additional-tokens', type=int, default=100000)
    parser.add_argument('--window', type=int, default=32)
    parser.add_argument('--settle-ticks', type=int, default=None,
                        help='Quiet physical ticks after each input pulse; default retains checkpoint timing')
    parser.add_argument('--adaptive-admission', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable event-driven adaptive admission settling based on causal relaxation')
    parser.add_argument('--circuit-observer', type=Path, default=None,
                        help='Read-only per-word circuit activity log of the live pass (fly_circuit_observer) into this '
                             'directory; diagnostic, never inherited')
    parser.add_argument('--brain-learning', choices=('bptt', 'local'), default=None,
                        help="'local': the brain runs forward only (its own local laws keep acting), only the readouts "
                             "take gradients; no autograd tape through any tick (inherit, else bptt)")
    parser.add_argument('--native-credit-cpu', action=argparse.BooleanOptionalAction, default=None,
                        help='Archive exact delayed native score tensors in CPU RAM; same policy/clock/values (inherit)')
    parser.add_argument('--stream-segment-ticks', type=int, default=None,
                        help='Native-context internal gradient recomputation segment; real input intervals unchanged (inherit, else16)')
    parser.add_argument('--stream-ticks', type=int, default=None,
                        help='Streaming clock: every token occupies exactly this many physical ticks and the next token '
                             'arrives regardless of settling (0 = serial pulse-settle-read; omitted = inherit)')
    parser.add_argument('--hold-input', action=argparse.BooleanOptionalAction, default=None,
                        help='Streaming: keep the token on the sensory surface for all its ticks (omitted = inherit)')
    parser.add_argument('--go-neuron', choices=('none', 'giant_fiber', 'virtual'), default=None,
                        help='Self-paced body: the spike of a go neuron is the act of speaking the prediction and taking '
                             'the next input. giant_fiber = the connectome DNp01 pair (nothing added); virtual = an added '
                             'leaky integrator of the motor surface (omitted = inherit)')
    parser.add_argument('--go-max-ticks', type=int, default=None, help='Safety ceiling for one token (default 32)')
    parser.add_argument('--intrinsic', action=argparse.BooleanOptionalAction, default=None,
                        help='Release the spontaneously active cells annotated in the connectome (clock, dopaminergic, '
                             'octopaminergic, serotonergic, DPM, APL) with a learnable intrinsic current (omitted = inherit)')
    parser.add_argument('--precision', action=argparse.BooleanOptionalAction, default=None,
                        help='Precision carried by the neuromodulatory cells: their concentration trace scales the '
                             'written innovation per sensory pathway (attention) and dopamine scales the synaptic '
                             'learning rate (omitted = inherit)')
    parser.add_argument('--predictive-writer', action=argparse.BooleanOptionalAction, default=None,
                        help='Predictive-coding writer: the innovation is taken against a mix of the adaptation '
                             'baseline and the brain own spoken prediction (learnable mix, start 1/2; omitted = inherit)')
    parser.add_argument('--dopamine-modulation', action=argparse.BooleanOptionalAction, default=None,
                        help='Dopamine as a neuromodulator along the connectome projections: its phasic part scales '
                             'target thresholds (per-superclass sensitivity, start 0) and its concentration scales the '
                             'learning of synapses onto the targets (omitted = inherit)')
    parser.add_argument('--amine-modulation', action=argparse.BooleanOptionalAction, default=None,
                        help='Native slow octopamine/serotonin delivery through immutable anatomical contacts, '
                             'mass-action receptor occupancy, and learnable existing-cell response (omitted = inherit)')
    parser.add_argument('--cx-state', action=argparse.BooleanOptionalAction, default=None,
                        help='Persistent central-complex activity trace with learnable decay and native CX threshold '
                             'feedback (omitted = inherit)')
    parser.add_argument('--mb-terminal-projection', action=argparse.BooleanOptionalAction, default=None,
                        help='Actual DPM/APL axonal contacts regulate individual KC release; persistent '
                             'DPM 5-HT receptors; pooled threshold/forgetting proxy archived')
    parser.add_argument('--mb-consolidation', action=argparse.BooleanOptionalAction, default=None,
                        help='Native DPM slow consolidation and APL slow state around existing mushroom-body '
                             'compartment memory (omitted = inherit)')
    parser.add_argument('--apl-cable', action=argparse.BooleanOptionalAction, default=None,
                        help='Full supplied APL forest with local inputs, passive cable and graded output; '
                             'requires MB axonal contacts and eager/checkpointed execution')
    parser.add_argument('--apl-local-release', action=argparse.BooleanOptionalAction, default=None,
                        help='Local calcium-dependent APL release and conserving vesicle resources; inherit on resume')
    parser.add_argument('--kc-terminal', action=argparse.BooleanOptionalAction, default=None,
                        help='Actual KC release sites, local receptor signals and compartment eligibility')
    parser.add_argument('--kc-terminal-geometry', type=Path, default=None)
    parser.add_argument('--kc-lateral', action=argparse.BooleanOptionalAction, default=None,
                        help='Native axo-axonal muscarinic inhibition of terminal release and cAMP')
    parser.add_argument('--kc-lateral-geometry', type=Path, default=None)
    parser.add_argument('--sensory-terminal', action=argparse.BooleanOptionalAction, default=None,
                        help='Sensory axo-axonal return contacts shunt release at actual terminal sites')
    parser.add_argument('--sensory-terminal-geometry', type=Path, default=None)
    parser.add_argument('--dan-local-release', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--dan-carried-credit', action=argparse.BooleanOptionalAction, default=None,
                        help='Carry conditional local DAN material history, including native action scores')
    parser.add_argument('--dan-receiver-credit', action=argparse.BooleanOptionalAction, default=None,
                        help='Carry source-resolved history through global dopamine filters')
    parser.add_argument('--dan-release-geometry', type=Path, default=None)
    parser.add_argument('--kc-dopamine', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--kc-dopamine-geometry', type=Path, default=None)
    parser.add_argument('--kc-contact-memory', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--kc-site-stp', action=argparse.BooleanOptionalAction, default=None,
                        help='Actual KC-site resources/facilitation; requires local terminal geometry')
    parser.add_argument('--body-loop', action=argparse.BooleanOptionalAction, default=None,
                        help='Restore historical body trace parameters/state (omitted = inherit); '
                             'synthetic feedback is quarantined and inactive in normal training')
    parser.add_argument('--body-proxy-compatibility', action='store_true',
                        help='Historical frozen diagnostic compatibility only; rejected by normal training')
    parser.add_argument('--olfactory-relay', action=argparse.BooleanOptionalAction, default=None,
                        help='Native antennal-lobe local inhibition: persistent AL local trace delivered through '
                             'fixed inhibitory AL-to-PN contacts with learnable decay and PN response (omitted = inherit)')
    parser.add_argument('--visual-graded', action=argparse.BooleanOptionalAction, default=None,
                        help='Use existing graded membrane release for evidence-scoped early visual cells; inherit on resume')
    parser.add_argument('--visual-graded-gain', type=float, default=None,
                        help='Initial positive visual release gain only for newly attached cells (default .025; learnable)')
    parser.add_argument('--body-actuator', action=argparse.BooleanOptionalAction, default=None,
                        help='Restore historical scalar actuator state; proxy feedback is inactive in normal training')
    parser.add_argument('--endocrine-exchange', action=argparse.BooleanOptionalAction, default=None,
                        help='LK secretion/clearance/receptor axis into annotated IPC/ITP cells (omitted = inherit)')
    parser.add_argument('--native-motor-port', action=argparse.BooleanOptionalAction, default=None,
                        help='Execute language only from annotated motor-cell release packets, with a separate action decoder')
    parser.add_argument('--dopamine-site', choices=('post', 'pre'), default=None,
                        help='Which end of a synapse the dopamine scales its learning by: post = the receiving neuron, '
                             'pre = the presynaptic terminal as at the fly mushroom body (omitted = inherit, else post)')
    parser.add_argument('--ablate', default=None,
                        help="Causal gate (never inherited; resume only): 'noedges' (every synapse cut: the read and decoder "
                             "learn alone), 'blind' (the words never reach the senses: what reading is worth) or a bool .npy "
                             "of cells to silence (outputs cut, reads zeroed); learning continues")
    parser.add_argument('--eye-floor', type=float, default=0.0,
                        help='Binary eyes: the sampled open probability is kept in [floor, 1 - floor] (never inherited)')
    parser.add_argument('--log-word-dopamine', action='store_true',
                        help='With --score-log: every word phasic dopamine and dopamine-cell firing (living gate 2)')
    parser.add_argument('--skip-eval', action='store_true',
                        help='No active evaluation (fresh B / A revisit) in this run: a gate arm that only scores the stream')
    parser.add_argument('--score-log', type=Path, default=None,
                        help='Append every window pre-update word scores (JSONL) for paired comparisons')
    parser.add_argument('--mother-world', choices=('local', 'api', 'gemini', 'deepseek', 'siliconflow', 'mailbox'), default=None,
                        help='Train in the mother world: live each window (the fly speaks in its turns, the mother answers), '
                             'then relive it with gradient (never inherited)')
    parser.add_argument('--mother-fallback', choices=('siliconflow',), default=None,
                        help='A second mouth that answers a turn the primary leaves silent after its retries (outage)')
    parser.add_argument('--mother-api-model', default=None, help='The API mother model id (the SDK reads its own env config)')
    parser.add_argument('--mother-temperature', type=float, default=1.0, help='Sampling temperature of the fly words')
    parser.add_argument('--mother-max-fly-words', type=int, default=16, help='A fly turn ends after this many words')
    parser.add_argument('--mother-credit', choices=('predict', 'epistemic', 'consequence'), default='predict',
                        help="Speaking act's credit: 'predict' (-CE of the following words) or 'epistemic' (what the words "
                             "arriving within --mother-credit-ticks taught: learning progress, applied one window late); "
                             "'consequence' (native KC action-consequence uncertainty; timing and word content credit)")
    parser.add_argument('--mother-consequence-native', action=argparse.BooleanOptionalAction, default=None,
                        help='Opt-in ordered word/action model, posterior-guided policy and native delayed credit')
    parser.add_argument('--mother-native-credit', action=argparse.BooleanOptionalAction, default=None,
                        help='Send existing consequence credit to native brain parameters without changing the posterior, policy or prediction targets')
    parser.add_argument('--mother-action-clock', choices=('arrival', 'commit'), default=None,
                        help='Explicit event-clock migration: commit sends the command on its actual motor tick; '
                             'uses actual external-arrival CE and excludes known self-command labels. Omitted inherits.')
    parser.add_argument('--mother-generative-shadow', action=argparse.BooleanOptionalAction, default=None,
                        help='Learn a durable marked-event likelihood from actual native context, commands and world '
                             'arrivals without changing actions/rewards. Requires the commit clock; experimental, '
                             'does not claim a subjective perception model or a complete active-inference policy.')
    parser.add_argument('--mother-native-generative', action=argparse.BooleanOptionalAction, default=None,
                        help='Actuate finite native categorical wait/token inference; actual external CE learns sensing, '
                             'response evidence trains native joint policy credit. Explicit experimental migration.')
    parser.add_argument('--mother-generative-reference', type=Path, default=None,
                        help='Retained complete checkpoint supplying the explicit ancestral native hypothesis')
    parser.add_argument('--mother-native-context', action=argparse.BooleanOptionalAction, default=None,
                        help='Experimental continuous context filtering and native compartment recognition learning')
    parser.add_argument('--mother-context-initial-variance', type=float, default=None)
    parser.add_argument('--mother-context-process-variance', type=float, default=None)
    parser.add_argument('--mother-generative-candidates', type=int, default=4,
                        help='Bounded evaluated word proposals; unassessed vocabulary retains its native probability')
    parser.add_argument('--mother-generative-ticks', type=int, default=0,
                        help='Native proposal lookahead in actual ticks; zero uses current world mean word interval')
    parser.add_argument('--mother-credit-ticks', type=int, default=64, help='Epistemic credit horizon in ticks')
    parser.add_argument('--intrinsic-mbon-kappa', type=float, default=None,
                        help='The MBONs join the spontaneously active cells with this start current (x rheobase, '
                             'learnable); 0 = off (omitted = inherit)')
    parser.add_argument('--sense-gains-learnable', action='store_true', default=None,
                        help="The mother's voice gain (Johnston's organ) and the PAM cells' sensitivity to her answer "
                             'are learnable, starting at their levels (omitted = inherit)')
    parser.add_argument('--mother-credit-event-clock', action='store_true', default=None,
                        help='The speaking credit decays per mother word since the decision, not per tick: her own word '
                             'does not discount what follows (review 2026-10-15; omitted = inherit)')
    parser.add_argument('--mother-credit-tau-words', type=float, default=3.0,
                        help='Event-clock decay in mother words (without --mother-credit-brain-tau)')
    parser.add_argument('--mother-credit-brain-tau', action='store_true', default=None,
                        help="The speaking credit decays on the brain's own eligibility clock: the Kenyon trace time "
                             "constant (comp_logit_trace, learnable) instead of --mother-credit-tau (omitted = inherit)")
    parser.add_argument('--mother-credit-differential', action='store_true', default=None,
                        help='With --mother-credit-trace: the average-reward return of an infinite stream (each tick '
                             'after a decision also costs her own long-run learning progress per tick; omitted = inherit)')
    parser.add_argument('--mother-credit-trace', action='store_true', default=None,
                        help='Speaking credit as a continuous O(1) eligibility trace on the act readout (decay '
                             '--mother-credit-tau, carried across windows, no horizon; --mother-credit-ticks unused)')
    parser.add_argument('--mother-credit-tau', type=float, default=0.0,
                        help='> 0: the epistemic return discounts each later word by exp(-delay/tau) (ticks) instead of the '
                             'flat horizon')
    parser.add_argument('--mother-interrupt', action='store_true',
                        help='With --mother-speak-act: a commit during the mother\'s turn cuts her off (no commit keeps '
                             'listening), both credited like speaking')
    parser.add_argument('--mother-speak-act', action='store_true',
                        help='v2: in its turn the motor acts decide at each arrival to say a word or yield the floor '
                             '(needs --ai --motor-acts; credit: fly_active_inference.speaking_value)')
    parser.add_argument('--kc-dan-release', action=argparse.BooleanOptionalAction, default=None,
                        help='Kenyon -> dopamine-terminal synapses raise local dopamine release (axo-axonal; learned gain '
                             'per type from 0.1); omitted = inherit, else off')
    parser.add_argument('--kc-habituation-tau', type=float, default=None,
                        help='Habituation of the Kenyon terminal release (alpha\'3 novelty): recovery time constant in '
                             'ticks; omitted = inherit, else off (needs --kc-dan-release)')
    parser.add_argument('--kc-habituation-use', type=float, default=None,
                        help='Fraction of a Kenyon cell\'s release availability one spike uses (default 0.2)')
    parser.add_argument('--input-gain-size-exponent', type=float, default=None,
                        help='Per-class learnable input gain (input conductance grows with the cell), started at '
                             '(median size / class size)^this (0 uniform, 1 inverse to size; omitted = inherit, else none)')
    parser.add_argument('--read-extra-classes', default=None,
                        help='Annotation classes whose cells the prediction is also read from (e.g. Kenyon_Cell,MBON; '
                             'the belief where it lives; acts stay on the output cells; omitted = inherit, else none)')
    parser.add_argument('--read-spikes', action=argparse.BooleanOptionalAction, default=None,
                        help='The read surface is its output: transmitted pulses through the Gamma (muscle) filter, not '
                             'the membranes (implies --use-read-gamma-trace; omitted = inherit, else off)')
    parser.add_argument('--evidence-plasticity', action=argparse.BooleanOptionalAction, default=None,
                        help='Synapses and class physiology step by gradient / one shared RMS per tensor (bounded), '
                             'not per-parameter Adam (omitted = inherit, else off)')
    parser.add_argument('--eg-plasticity', action=argparse.BooleanOptionalAction, default=None,
                        help='Synapses learn by normalized exponentiated gradient: log-space steps rate x momentum / its '
                             'running RMS, KL pull to the birth connectome (--eg-tau updates), every cell keeps its '
                             'birth E and I input totals (omitted = inherit, else off)')
    parser.add_argument('--eg-tau', type=float, default=None,
                        help='EG: time constant (updates) of the return to the birth connectome (omitted = inherit, else 4096)')
    parser.add_argument('--pull-tau', type=float, default=None,
                        help='With --evidence-plasticity: unconsolidated change decays to the birth value with this '
                             'time constant in updates (0 = off; omitted = inherit)')
    parser.add_argument('--dopamine-adapted', action=argparse.BooleanOptionalAction, default=None,
                        help='Receptors adapt to each target own slow dopamine level: threshold and plasticity see the '
                             'relative change of that target (omitted = inherit, else off)')
    parser.add_argument('--online-da-plasticity', action=argparse.BooleanOptionalAction, default=None,
                        help='O(1) online learning with dopamine as the third factor: every dopamine-receiving terminal '
                             'writes a release-gain memory from its activity trace and phasic dopamine (needs '
                             '--dopamine-modulation; omitted = inherit, else off)')
    parser.add_argument('--kc-dendritic-only', action=argparse.BooleanOptionalAction, default=None,
                        help='Kenyon -> Kenyon synapses drive the soma only for their calyx (dendritic) share (neuPrint '
                             'roiInfo); the 97%% that are axo-axonal are removed and kept at 0 (omitted = inherit, else off)')
    parser.add_argument('--dopamine-affinity', type=float, default=None,
                        help='Receptor half-saturation concentration K of the dopamine occupancy c / (c + K) '
                             '(omitted = inherit, else the lineage value 0.0108)')
    parser.add_argument('--homeostasis', action=argparse.BooleanOptionalAction, default=None,
                        help='Homeostatic intrinsic plasticity: every neuron drives itself toward its own target rate '
                             '(duty cycle / tau_m, log-normal across neurons; omitted = inherit, else off)')
    parser.add_argument('--homeostasis-seed', type=int, default=None,
                        help='The individual: seed of the neurons\' target duty cycles (omitted = inherit, else 0)')
    parser.add_argument('--regime-homeostasis', action=argparse.BooleanOptionalAction, default=None,
                        help='Regime homeostasis (attach_regime_homeostasis): each neuron keeps its threshold at a fixed '
                             'place in its own free-membrane distribution (omitted = inherit, else off)')
    parser.add_argument('--regime-seed', type=int, default=None, help='The individual of the regime homeostasis')
    parser.add_argument('--regime-kenyon-z', type=float, default=None,
                        help='Kenyon cells at a sparser quantile (omitted = inherit, else none)')
    parser.add_argument('--compartment-memory', action=argparse.BooleanOptionalAction, default=None,
                        help='Three-factor memory per Kenyon -> MBON synapse, dopamine at its MBON (omitted = inherit)')
    parser.add_argument('--compartment-dendritic', action=argparse.BooleanOptionalAction, default=None,
                        help='Every edge into Kenyon cells, MBONs, DANs, PNs and DNs scaled once by the share of its '
                             'synapses outside the axonal territory of the target (neuPrint ROI, per pair); edges with none '
                             'there pinned at 0.  Includes APL -> KC (omitted = inherit)')
    parser.add_argument('--apl-kc-dendritic-only', action=argparse.BooleanOptionalAction, default=None,
                        help='APL -> Kenyon inhibition on the soma only for its calyx synapses (84.5%% are axonal): '
                             'weights scaled once by the calyx share, axonal-only edges pinned at 0 (omitted = inherit)')
    parser.add_argument('--mother-reward', action=argparse.BooleanOptionalAction, default=None,
                        help='Her answer to the child speech as a consequence: a short pulse onto the PAM reward cells at '
                             'the first word of a normal reply (omitted = inherit, else off)')
    parser.add_argument('--mother-reward-level', type=float, default=None,
                        help='The reward pulse in rheobase units (omitted = inherit, else 1.0)')
    parser.add_argument('--mother-reward-ticks', type=int, default=None,
                        help='The reward pulse length in ticks (omitted = inherit, else 4)')
    parser.add_argument('--input-gain-kenyon-pooled', action='store_true', default=False,
                        help='Newborn: the Kenyon subtype classes start their input gain from the pooled Kenyon size '
                             '(the body volume is mostly axon; per-subtype size starved the alpha-prime beta-prime cells)')
    parser.add_argument('--comp-timing', action='store_true', default=None,
                        help="The fly's timing rule for the Kenyon -> MBON memory: dopamine before Kenyon potentiates, "
                             'Kenyon before dopamine depresses; use-dependent forgetting off (omitted = inherit)')
    parser.add_argument('--time-scale', type=float, default=None,
                        help='Newborn: dynamics k times faster per tick (time constants / k, conductances and sensory drive x k; '
                             'pair with --stream-pace-mean 32/k; omitted = inherit, 1 for a newborn)')
    parser.add_argument('--comp-unit-traces', action='store_true', default=None,
                        help='Newborn: the compartment eligibility is the terminal state (one spike lifts it to ~1, saturating), '
                             'rates start -1 / +1 (needs --comp-timing, --compartment-adapted; omitted = inherit)')
    parser.add_argument('--native-runtime-program', type=Path, default=None,
                        help='Data-only generated native mechanism document; resume inherits the embedded '
                             'program and complete physical history. New continuations require explicit migration.')
    parser.add_argument('--comp-rate-calibrate', action='store_true',
                        help='Newborn: start the compartment depression / potentiation rates so the median edge writes as fast '
                             'as it forgets (calibrate_comp_rate; needs --comp-tonic-forget)')
    parser.add_argument('--comp-camp-tau', type=float, default=None,
                        help='The burst judged on the Kenyon terminal cAMP: a leaky integral of the dopamine contrast, '
                             'time constant in ticks (start; learnable; needs --comp-burst-relative; omitted = inherit)')
    parser.add_argument('--comp-burst-relative', action='store_true', default=None,
                        help='The compartment burst threshold judged on the linear dopamine contrast (f - s) / (s + K), '
                             'occupancy amount ramped over one sd (needs --comp-timing; omitted = inherit)')
    parser.add_argument('--comp-burst-k', type=float, default=None,
                        help='Newborn: the burst threshold start in sd above edge mean (learnable; default 0.25; omitted = inherit)')
    parser.add_argument('--dan-kappa', type=float, default=None,
                        help='Newborn: intrinsic drive kappa for dopamine cells (learnable; start 0.5 = subthreshold / fluctuation-driven; '
                             'omitted = inherit or kappa)')
    parser.add_argument('--comp-trace-tau', type=float, default=None,
                        help="Newborn: the Kenyon eligibility trace's start time constant in ticks (learnable; default 50). "
                             'Set in word terms, about 2-3 word intervals of the world')
    parser.add_argument('--comp-tonic-forget', type=float, default=None,
                        help='Active forgetting of the Kenyon -> MBON memory: phi relaxes on its own clock (this many ticks '
                             'at tonic dopamine), whether or not the Kenyon cell fires; 0 = off (omitted = inherit)')
    parser.add_argument('--homeostat-skip-dans', action='store_true', default=None,
                        help='The pacemaker homeostat leaves the dopamine cells out (event-driven near threshold; their kappa '
                             'keeps its birth value; omitted = inherit)')
    parser.add_argument('--pacemaker-homeostasis', action=argparse.BooleanOptionalAction, default=None,
                        help='The spontaneously active modulatory cells keep their own firing: intrinsic_kappa leaves the '
                             'gradient and an intrinsic homeostat holds each cell near its own set point (omitted = inherit)')
    parser.add_argument('--mother-voice', action=argparse.BooleanOptionalAction, default=None,
                        help='The mother’s voice: while her word is on the senses, Johnston’s organ '
                             '(JO-A / JO-B) carries a steady current, her own words do not (omitted = inherit, else off)')
    parser.add_argument('--mother-voice-level', type=float, default=None,
                        help='The voice current in rheobase units (omitted = inherit, else 1.5)')
    parser.add_argument('--stream-jepa', action=argparse.BooleanOptionalAction, default=None,
                        help='JEPA term in the paced stream: the read after word t-1 predicts the Kenyon '
                             'representation (LeJEPA: no stop-gradient, SIGReg) (omitted = inherit, else off)')
    parser.add_argument('--stream-jepa-weight', type=float, default=None,
                        help='Weight of the stream JEPA term in the loss (omitted = inherit, else 0.5)')
    parser.add_argument('--compartment-adapted', action=argparse.BooleanOptionalAction, default=None,
                        help='The compartment memory\'s receptors adapt to their own slow dopamine, K_j = max(slow_j, K), '
                             'as every other target\'s do with --dopamine-adapted (omitted = inherit, else off)')
    parser.add_argument('--taste-every', type=int, default=0,
                        help='Taste world as a learning task: one episode after every N text windows (0 = off)')
    parser.add_argument('--ai', action=argparse.BooleanOptionalAction, default=None,
                        help='Unified perception and action: the eyes (text precision) and output (commit) act head '
                             '(omitted = inherit)')
    parser.add_argument('--ai-epistemic', type=float, default=None, help='Weight of the epistemic value of looking')
    parser.add_argument('--ai-pragmatic', type=float, default=None, help='Weight of the pragmatic value of output')
    parser.add_argument('--ai-sigreg', type=float, default=None, help='Weight of SIGReg on the read latents (VFE complexity)')
    parser.add_argument('--eye-forced-open', action='store_true', help='Control: eyes fixed open (gain 1, no gradient)')
    parser.add_argument('--eye-binary', action=argparse.BooleanOptionalAction, default=None,
                        help='M3: the eyes as a 0/1 decision (straight-through); omitted = inherit')
    parser.add_argument('--ai-depth', type=int, default=None,
                        help='M3: a question stays open over up to this many intervals while the eyes are closed')
    parser.add_argument('--noise-rate', type=float, default=None, help='Noisy TV: probability per window of a noise stretch')
    parser.add_argument('--noise-length', type=int, default=8, help='Noisy TV: words per stretch')
    parser.add_argument('--noise-alphabet', type=int, default=16, help='Noisy TV: alphabet size (random token ids)')
    parser.add_argument('--noise-seed', type=int, default=7, help='Noisy TV: the alphabet and the draws')
    parser.add_argument('--eye-init', type=float, default=None, help='M3: the eyes\' initial open probability')
    parser.add_argument('--eye-gate', choices=('atan', 'sigmoid'), default=None,
                        help='Sensory gating activation: normalized atan or sigmoid; omitted = inherit (default atan)')
    parser.add_argument('--kc-subtypes', action=argparse.BooleanOptionalAction, default=None,
                        help='Connectome audit: Kenyon subtypes (alpha/beta core, surface, posterior, alpha-prime/beta-prime, '
                             'gamma) get their own physiology instead of sharing that of cb_intrinsic (omitted = inherit)')
    parser.add_argument('--motor-acts', action=argparse.BooleanOptionalAction, default=None,
                        help='Acts read from the output cells\' spikes through a muscle filter, sampled (diversity first); '
                             'omitted = inherit')
    parser.add_argument('--act-tau', type=float, default=None, help='Muscle time constant in ticks (motor acts)')
    parser.add_argument('--act-habit', type=float, default=None, help='Habit (prior) commit hazard per tick')
    parser.add_argument('--act-gamma-inv', type=float, default=None, help='1/gamma: weight of KL(policy || habit)')
    parser.add_argument('--act-brain-grad', action=argparse.BooleanOptionalAction, default=None,
                        help='Let the policy gradient reach the brain through the motor spikes')
    parser.add_argument('--act-dopamine-precision', action=argparse.BooleanOptionalAction, default=None,
                        help='Policy precision gamma encoded by the dopamine cells of the brain: z = ln E + gamma (W m + b)')
    parser.add_argument('--epistemic-belief', action=argparse.BooleanOptionalAction, default=None,
                        help='Epistemic value of looking = Bayesian surprise of the brain: how far a word moves its own '
                             'prediction of what comes next (instead of an external word counter)')
    parser.add_argument('--lr-physiology', type=float, default=None,
                        help='Adam rate of the class physiology (log / logit space: a relative step); default = --lr')
    parser.add_argument('--carried-credit', action=argparse.BooleanOptionalAction, default=None,
                        help='Credit beyond the BPTT window at O(1): forward sensitivities of the slow states '
                             '(compartment memory, dopamine clocks, STP) carried across windows')
    parser.add_argument('--miss-prior', action=argparse.BooleanOptionalAction, default=None,
                        help='An unanswered question is worth minus the unigram (listener prior) CE of its answer')
    parser.add_argument('--sense-noise', type=float, default=None,
                        help='Sustained noisy input (with --hold-input): per-word noise sigma ~ U(0, this) relative to '
                             'the embedding norm per dimension; omitted = inherit')
    parser.add_argument('--eye-forced', type=float, default=None, help='Control: eyes fixed at this gain')
    parser.add_argument('--odour-world', action=argparse.BooleanOptionalAction, default=None,
                        help='Closed-loop odour world alongside the text stream; acts from the act head (omitted = inherit)')
    parser.add_argument('--odour-drive', type=float, default=32.0, help='Odour whiff current, rheobase units')
    parser.add_argument('--odour-receptors', type=int, default=8, help='Receptor types per odour')
    parser.add_argument('--outcome-dopamine', action=argparse.BooleanOptionalAction, default=True,
                        help='Contact also drives the dopamine cells directly (an electrode); --no-outcome-dopamine: '
                             'the outcome enters only through the taste receptors')
    parser.add_argument('--act-lambda', type=float, default=1.0, help='Weight of the act fitness in the BPTT loss')
    parser.add_argument('--sense-prediction', action=argparse.BooleanOptionalAction, default=None,
                        help='The brain also predicts its other senses (odour levels, taste) one tick ahead, and the world '
                             'sends gusts (world-caused odour rises) to test self vs world (omitted = inherit)')
    parser.add_argument('--sense-lambda', type=float, default=1.0, help='Weight of the sense prediction error')
    parser.add_argument('--levy-scale', type=float, default=0.0,
                        help='Exploration trial: heavy-tailed (Cauchy x scale, clipped at 1.5) spontaneous pushes into the act integrators (0.005: a jump past the bound ~1/300 ticks)')
    parser.add_argument('--info-value', action=argparse.BooleanOptionalAction, default=False,
                        help='Exploration trial: the fitness counts the lifetime value of tasting an odour of unknown valence')
    parser.add_argument('--act-init', type=Path, default=Path('scratch/fly_stream/act_init.pt'),
                        help='Initial act head (act_calibrate.py) for a lineage that gets the odour world now')
    parser.add_argument('--connectome-v2', action=argparse.BooleanOptionalAction, default=None,
                        help='Pin the point-model artifacts of the audit: Kenyon->dopamine terminal synapses, synapses onto '
                             'sensory terminals, fast edges of octopamine/serotonin neurons (omitted = inherit)')
    parser.add_argument('--al-reset', action='store_true',
                        help='Once: every input of the antennal-lobe cells back to its birth weight')
    parser.add_argument('--relative-edge-rate', type=float, default=None,
                        help='Relative plasticity: each synapse moves by this fraction of its birth weight per full Adam '
                             'step (omitted = inherit; 0 = absolute steps)')
    parser.add_argument('--modulatory-classes', action=argparse.BooleanOptionalAction, default=None,
                        help='Dopamine (PAM, PPL1, other), octopamine and serotonin cells get their own class physiology '
                             'instead of cb_intrinsic (after the Kenyon subtypes; omitted = inherit)')
    parser.add_argument('--dan-renorm', action=argparse.BooleanOptionalAction, default=None,
                        help='Redo the birth normalization of the inputs of the dopamine cells without their pinned '
                             'Kenyon partners, once (needs --connectome-v2; omitted = inherit)')
    parser.add_argument('--intrinsic-pattern', type=str, default=None,
                        help='Regex of the cell types released by --intrinsic (default: every annotated tonic type)')
    parser.add_argument('--intrinsic-kappa', type=float, default=None,
                        help='Initial intrinsic current of released cells, in units of their rheobase (default 1.2)')
    parser.add_argument('--writer', choices=('topographic', 'receptor', 'receptor_all', 'receptor_text', 'receptor_lang'), default=None,
                        help='receptor: text through the olfactory receptor neurons only, per receptor type, as a '
                             'bounded transduction conductance around spontaneous firing (ReceptorWriter); inherited')
    parser.add_argument('--stp-init', type=str, default=None, help="birth STP 'u0,tau_fac,tau_rec' (needs --stp)")
    parser.add_argument('--alif-beta-init', type=float, default=None, help='birth adaptation strength beta (default 0.05)')
    parser.add_argument('--type-classes', action=argparse.BooleanOptionalAction, default=None,
                        help='one physiology class per annotated cell type (birth only; inherited)')
    parser.add_argument('--inhibitory-tau-ratio', type=float, default=None,
                        help='tau_s_i = tau_m / ratio at birth (2: 10 ms, the old convention; 4: 5 ms as Shiu and fast GluCl / GABA-A)')
    parser.add_argument('--pin-modulatory-types', action=argparse.BooleanOptionalAction, default=None,
                        help='connectome v2 also pins the fast edges of OA-* / 5-HT* / PAM / PPL cells by type name')
    parser.add_argument('--intrinsic-jitter', type=float, default=None,
                        help='per-cell spread of the birth intrinsic current (kappa x (1 + jitter N(0,1))); desynchronises pacemakers')
    parser.add_argument('--graded-types', type=str, default=None,
                        help='Comma-separated cell types made non-spiking (graded release), e.g. APL,CT1; inherited on resume')
    parser.add_argument('--graded-gain', type=float, default=0.1,
                        help='Initial release gain of graded cells (learnable); 0.025 makes APL suppress Kenyon coding ~2.6x (blocking APL: 2-3x, Lin et al. 2014)')
    parser.add_argument('--kc-renorm', action=argparse.BooleanOptionalAction, default=None,
                        help='Redo the birth normalization of Kenyon inputs over dendritic partners, once (omitted = inherit)')
    parser.add_argument('--comp-rate', type=float, default=None,
                        help='Initial compartment-memory rate when it is attached now: eta = -rate (forgetting stays at its default 0.02)')
    parser.add_argument('--stream-pace-min', type=float, default=0.0,
                        help='With --stream-pace-mean M: each window\'s mean interval is log-uniform in [this, M] '
                             '(the world\'s tempo varies; 0 = fixed mean)')
    parser.add_argument('--stream-pace-mean', type=float, default=0.0,
                        help='With --stream-ticks: the world sets each word\'s interval, memoryless (geometric, >= 1 tick) '
                             'with this mean; the read is the expectation standing when the next word arrives (0 = fixed clock)')
    parser.add_argument('--tied-writer-scale', type=float, default=0.1,
                        help='Scale of the tied rows when the writer is tied now (1: each cell keeps its drive size)')
    parser.add_argument('--taste-drive', type=float, default=8.0,
                        help='Dopamine-cell drive during pairing, in rheobase units of each driven cell')
    parser.add_argument('--tied-writer', action=argparse.BooleanOptionalAction, default=None,
                        help='The writer drives chemosensory cells by receptor type (attach_tied_writer; omitted = inherit)')
    parser.add_argument('--bookshelf', choices=('progress', 'random', 'surprise', 'anti'), default=None,
                        help='Curious shelf v1: training text comes from a shelf of books (topic books, a repeated passage, '
                             'noise, a hidden pair) chosen after every 64-token segment by this selector (default: off)')
    parser.add_argument('--bookshelf-seed', type=int, default=0)
    parser.add_argument('--world-skip', type=int, default=0,
                        help='Mixed world: draw and discard this many windows of the world sequence first (resuming a '
                             'run that crashed, so it sees the same worlds as a run that did not)')
    parser.add_argument('--online-da-eta', type=float, default=-0.02,
                        help='Starting write rate eta of the online dopamine plasticity (used when it is first attached)')
    parser.add_argument('--dopamine-sensitivity-init', type=float, default=None,
                        help='Set every dopamine sensitivity to this value at start, also over a loaded checkpoint '
                             '(pass it only on the first segment of a run; omitted = keep)')
    parser.add_argument('--mixed-world', action='store_true',
                        help='Training windows alternate worlds: self-paced reading (go neuron), being spoken to '
                             '(external pace, k ticks per word, k random 3..8), and reading with silent gaps (each word '
                             'with probability 1/4 is followed by 8..32 silent ticks). Evaluation stays self-paced reading.')
    parser.add_argument('--world-mix', type=float, nargs=3, default=(0.5, 0.25, 0.25), metavar=('READ', 'LISTEN', 'GAP'),
                        help='Mixed world: fractions of windows that are self-paced reading, being spoken to, reading with gaps')
    parser.add_argument('--gap-prob', type=float, default=0.25, help='Mixed world, gap windows: chance a word is followed by silence')
    parser.add_argument('--gap-range', type=int, nargs=2, default=(8, 32), metavar=('MIN', 'MAX'),
                        help='Mixed world, gap windows: silence length in ticks (inclusive)')
    parser.add_argument('--read-bank', action=argparse.BooleanOptionalAction, default=None,
                        help='Read without a window: learnable exponential filters of the motor membranes along the '
                             'whole tick stream, starting as the plain read (omitted = inherit)')
    parser.add_argument('--go-tick-cost', type=float, default=None,
                        help='Optional energy prior: nats per tick of average activity (default 0 = no prior)')
    parser.add_argument('--go-reference-energy', type=float, default=None,
                        help='Average share of synapses carrying a spike per tick in the champion (0.07)')
    parser.add_argument('--read-lag', type=int, default=None,
                        help='Streaming: read the prediction of each token this many ticks after it began to be sensed '
                             '(output lags input; must stay below the sensory->read arrival, 1 for MaleCNS; omitted = inherit)')
    parser.add_argument('--min-settle-ticks', type=int, default=3,
                        help='Minimum physical quiet ticks per token ensuring causal signal arrival (default: 3)')
    parser.add_argument('--flux-baseline', type=float, default=0.048,
                        help='Membrane kinetic flux dissipation baseline for attractor admission (default: 0.048)')
    parser.add_argument('--ctm-loss', dest='ctm_loss', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable CTM dual-anchor (min loss + max certainty) training objective')
    parser.add_argument('--checkpointing', dest='checkpointing', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable gradient checkpointing per token to avoid VRAM blowup during deep settling')
    parser.add_argument('--detach-reset', action=argparse.BooleanOptionalAction, default=None,
                        help='Detach only hard-reset spike derivative; resumes retain saved convention')
    parser.add_argument('--surrogate-mode', choices=('absolute', 'threshold'), default=None,
                        help='Spike proxy width convention; resumes inherit the saved rule')
    parser.add_argument('--transmission-mode', choices=('atomic', 'incoming'), default=None,
                        help='Summation execution order; resumes inherit the saved convention')
    parser.add_argument('--E-E', dest='E_E', type=float, default=None,
                        help='Excitatory reversal (rest 0, threshold 0.1 = -55 / -45 mV); 0.55 = ACh near 0 mV')
    parser.add_argument('--E-I', dest='E_I', type=float, default=None, help='Inhibitory reversal; -0.15 = Cl- near -70 mV')
    parser.add_argument('--spike-consumes-drive', action=argparse.BooleanOptionalAction, default=None,
                        help='A spike zeroes the cell conductances (Shiu et al. 2024 g = 0 on reset)')
    parser.add_argument('--refractory-ticks', type=int, default=None, help='Ticks held at reset after a spike (Shiu: 2.2 ms -> 2)')
    parser.add_argument('--stp', dest='use_stp', action=argparse.BooleanOptionalAction, default=None,
                        help='Short-term plasticity (the cortical default kills transmission on the corrected birth)')
    parser.add_argument('--writer-baseline-clock', choices=('input', 'physical'), default=None,
                        help='Packet baseline clock; default reproduces the saved implementation')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr-decoder', type=float, default=None,
                        help='Decoder weight only; other read parameters retain --lr')
    parser.add_argument('--lr-synapse', type=float, default=None)
    parser.add_argument('--lr-sensory', type=float, default=None)
    parser.add_argument('--dan-plastic-lr', type=float, default=None,
                        help='Level-2: learning rate of the dopamine-gated three-factor '
                             'local rule on the DAN-innervated synapses (0 disables)')
    parser.add_argument('--read-centering', action=argparse.BooleanOptionalAction, default=None,
                        help='Center the read pathway on the running state baseline: '
                             'the readout sees only the conditional component')
    parser.add_argument('--use-read-gamma-trace', action=argparse.BooleanOptionalAction, default=None,
                        help='Continuous 2nd-order Gamma-trace temporal readout accumulator')
    parser.add_argument('--init-read-gamma', default='anatomical',
                        help='Initial decay rate for continuous Gamma-trace accumulator (default: anatomical)')
    parser.add_argument('--reinit-read-gamma', action='store_true',
                        help='Re-initialize logit_read_gamma with the new wide anatomical multi-scale timescale')
    parser.add_argument('--use-latent-predictor', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable LeJEPA latent predictive coding and SIGReg anti-collapse')
    parser.add_argument('--use-graph-observer', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable HX-1 Graph-Constrained Distributed State Observer with Predict-Arrive-Correct loop')
    parser.add_argument('--lambda-jepa', type=float, default=0.1,
                        help='Weight of LeJEPA latent dynamics loss in joint objective')
    parser.add_argument('--lambda-sigreg', type=float, default=0.2,
                        help='Weight of SIGReg anti-collapse regularizer in LeJEPA loss')
    parser.add_argument('--lambda-mcr2', type=float, default=None,
                        help='Weight of MCR^2 maximal coding rate reduction loss in joint objective (default: 0.0)')
    parser.add_argument('--eps-mcr2', type=float, default=None,
                        help='MCR^2 coding rate precision scale epsilon (default: 0.5)')
    parser.add_argument('--max-horizon', type=int, default=14,
                        help='Maximum forward conduction horizons for DAgger LeJEPA distillation (default: 14)')
    parser.add_argument('--dagger-beta', type=float, default=0.5,
                        help='DAgger interpolation parameter between teacher state and student prediction (default: 0.5)')
    parser.add_argument('--obs-init-gamma', type=float, default=0.0,
                        help='Initial value of coupling factor gamma for graph observer lookahead residual (default: 0.0)')
    parser.add_argument('--learn-stp', action='store_true', default=None,
                        help='Jointly learn existing STP release/recovery parameters; resumes inherit activation')
    parser.add_argument('--plasticity-optimizer', choices=('adamw', 'sgd'), default='adamw')
    parser.add_argument('--disable-alif', action='store_true',
                        help='Ablation: ALIF strength beta held at its lower clamp (1e-4) with zero gradient; '
                             'all other state, weights and the learner are unchanged')
    parser.add_argument('--validate-every-tokens', type=int, default=5000)
    parser.add_argument('--checkpoint-every-tokens', type=int, default=0,
                        help='Save continuous state at this training-token cadence without running ABA/AG (0 disables)')
    parser.add_argument('--log-every-tokens', type=int, default=128)
    parser.add_argument('--eval-tokens', type=int, default=256)
    parser.add_argument('--replay-tokens', type=int, default=128)
    parser.add_argument('--vram-limit-mib', type=float, default=3900)
    parser.add_argument('--no-graph', action='store_true')
    parser.add_argument('--calibrate', action='store_true', help='One real training window, save full continuation and exit')
    parser.add_argument('--checkpoint-dir', type=Path, default=None,
                        help='Write weight checkpoints here instead of the output dir '
                             '(for output dirs on drives where large writes are unstable)')
    parser.add_argument('--extend-budget', action='store_true',
                        help='Replace an unfinished target with current count + additional fresh tokens')
    parser.add_argument('--profile', choices=sorted(PROFILES), default='mainline',
                        help='Switch set every option not named on the command line comes from at birth '
                             '(a resume inherits the checkpoint config instead)')
    global _CHECKPOINT_DIR
    argv = sys.argv[1:]
    profile = parser.parse_known_args(argv)[0].profile
    args = apply_profile(parser, argv, profile)
    if args.mother_credit == 'consequence' and (not args.mother_world or not args.mother_speak_act):
        parser.error('consequence credit requires --mother-world and --mother-speak-act')
    if isinstance(args.init_read_gamma, str) and args.init_read_gamma != 'anatomical':
        args.init_read_gamma = float(args.init_read_gamma)      # a numeric decay, not the anatomical schedule
    _CHECKPOINT_DIR = str(args.checkpoint_dir) if args.checkpoint_dir else None
    if args.settle_ticks is not None and args.settle_ticks < 0:
        parser.error('settle-ticks must be nonnegative')
    args.lr_synapse = args.lr if args.lr_synapse is None else args.lr_synapse
    args.lr_sensory = args.lr if args.lr_sensory is None else args.lr_sensory
    if min(args.window, args.additional_tokens, args.eval_tokens, args.replay_tokens) < 1:
        parser.error('Budgets must be positive')
    if any(value % args.window for value in (args.additional_tokens, args.eval_tokens, args.replay_tokens)):
        parser.error('Training/evaluation budgets must be multiples of the captured window')
    if not torch.cuda.is_available():
        parser.error('CUDA required')
    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_storage = Path(args.checkpoint_dir) if args.checkpoint_dir else args.output
    if (ckpt_storage / 'last.pt').exists():
        if args.resume and args.resume.resolve() != (ckpt_storage / 'last.pt').resolve():
            parser.error('Output already contains a continuing individual')
        args.resume = ckpt_storage / 'last.pt'
        is_scratch_start = False
    elif (args.output / 'last.pt').exists():
        if args.resume and args.resume.resolve() != (args.output / 'last.pt').resolve():
            parser.error('Output already contains a continuing individual')
        args.resume = args.output / 'last.pt'
        is_scratch_start = False
    elif args.from_scratch:
        is_scratch_start = True
        print_switches(parser, args, f'profile {profile}')
    elif args.resume is not None:
        is_scratch_start = False
    else:
        parser.error('Either --resume or --from-scratch must be specified')
    torch.manual_seed(11)
    train = np.load(args.data / 'train.npy', mmap_mode='r')
    val = np.load(args.data / 'validation.npy', mmap_mode='r')

    if is_scratch_start:
        print('Initializing newborn individual from scratch with pretrained embeddings...', flush=True)
        if args.d_model is None:
            args.d_model = 768
        if args.decoder_bias is None:
            args.decoder_bias = True
        if args.detach_reset is None:
            args.detach_reset = False
        if args.surrogate_mode is None:
            args.surrogate_mode = 'absolute'
        for key, old in BIOPHYSICS_DEFAULTS.items():
            if getattr(args, key, None) is None:
                setattr(args, key, old)
        if args.transmission_mode is None:
            args.transmission_mode = 'atomic'
        if args.settle_ticks is None:
            args.settle_ticks = 0
        if args.writer_baseline_clock is None:
            args.writer_baseline_clock = 'physical'
        if args.read_centering is None:
            args.read_centering = False
        if args.use_read_gamma_trace is None:
            args.use_read_gamma_trace = False
        args.read_spikes = bool(args.read_spikes)
        args.read_extra_classes = args.read_extra_classes or ''
        if args.read_spikes:
            args.use_read_gamma_trace = True
        args.evidence_plasticity = bool(args.evidence_plasticity)
        args.pull_tau = float(args.pull_tau or 0.0)
        args.eg_plasticity = bool(args.eg_plasticity)
        args.eg_tau = float(args.eg_tau or 4096.0)
        if args.use_latent_predictor is None:
            args.use_latent_predictor = False
        if args.use_graph_observer is None:
            args.use_graph_observer = False
        if args.dan_plastic_lr is None:
            args.dan_plastic_lr = 0.0
        if args.reference_train is None:
            args.reference_train = args.data / 'train.npy'

        previous_config = {
            'd_model': args.d_model, 'decoder_bias': args.decoder_bias,
            'data': str(args.data), 'reference_train': str(args.reference_train),
        }
        fmt = 'fly-bptt-v1'

        model = FlyReservoirLM(args.graph, vocab_size=50257, d_model=args.d_model,
            injection='topographic', read_surface='output', synapse_model='coba',
            extra_read_index=read_extra_index(args.graph, args.read_extra_classes),
            use_alif=True, decoder_bias=args.decoder_bias, **biophysics_kwargs(args),
            read_centering=args.read_centering,
            use_read_gamma_trace=args.use_read_gamma_trace,
            init_read_gamma=args.init_read_gamma,
            use_latent_predictor=args.use_latent_predictor,
            use_graph_observer=args.use_graph_observer,
            lambda_obs=args.lambda_jepa,
            lambda_sigreg=args.lambda_sigreg,
            max_horizon=args.max_horizon,
            dagger_beta=args.dagger_beta,
            obs_init_gamma=args.obs_init_gamma,
            detach_reset=args.detach_reset, surrogate_mode=args.surrogate_mode,
            transmission_mode=args.transmission_mode).cuda()
        model.dan_plastic_lr = args.dan_plastic_lr

        from safetensors.torch import load_file
        pretrained = load_file(str(args.pretrained_embedding))
        with torch.no_grad():
            model.embedding.weight.copy_(pretrained['wte.weight'])
            model.decoder.weight.copy_(pretrained['wte.weight'])
            del pretrained
            if model.decoder.bias is not None:
                ref_train = np.load(args.reference_train, mmap_mode='r')
                ref_counts = np.ones(50257, dtype=np.float64)
                for left in range(0, len(ref_train), 1_000_000):
                    ref_counts += np.bincount(
                        ref_train[left:left+1_000_000].astype(np.int64), minlength=50257)
                prior = np.log(ref_counts / ref_counts.sum()).astype(np.float32)
                model.decoder.bias.copy_(torch.from_numpy(prior))
                del ref_train, ref_counts
            model.read_norm.weight.fill_(args.read_norm_init)
        if args.alif_beta_init is not None:
            # birth adaptation strength (threshold += beta x the cell's running rate per tick); 0.2 = +20% of
            # threshold at a sustained 100 Hz: ends most reverberating strongly coupled pairs (127 -> 9 cells above
            # 200 Hz without input) while sugar -> MN9 / bitter -> PPL1 keep +42 / +37 Hz (audit 2026-10-13); learnable
            with torch.no_grad():
                model.log_beta.fill_(math.log(args.alif_beta_init))
            print(f'Adaptation at birth: beta {args.alif_beta_init}', flush=True)
        if args.stp_init:
            # birth values of the short-term plasticity, 'u0,tau_fac,tau_rec' (ms); e.g. 0.25,5,50: depression only,
            # which ends the reverberation of strongly coupled pairs (audit 2026-10-13) and spares 100 Hz lines
            if not model.use_stp:
                raise ValueError('--stp-init needs --stp')
            u0, tf, tr = (float(x) for x in args.stp_init.split(','))
            with torch.no_grad():
                model.logit_u0.fill_(math.log(u0 / (1 - u0))); model.log_tau_fac.fill_(math.log(tf)); model.log_tau_rec.fill_(math.log(tr))
            print(f'STP at birth: u0 {u0}, tau_fac {tf} ms, tau_rec {tr} ms', flush=True)
        if args.writer is None:
            args.writer = 'topographic'
        if args.writer in WRITER_SENSE:
            if args.tied_writer:
                raise ValueError('--writer receptor already ties the rows per receptor type (no --tied-writer)')
            n_cells, n_types = attach_receptor_writer(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                                      sense_class=WRITER_SENSE[args.writer])
            print(f'Receptor writer (newborn): text through {n_cells} sensory cells, {n_types} rows', flush=True)

        # A newborn gets the same blanket and modulation as the resumed lineage (each only when asked for).
        if args.go_neuron in ('virtual', True):
            args.go_neuron = 'virtual'
            attach_go_neuron(model)
        elif args.go_neuron in ('none', None, False):
            args.go_neuron = None
        if args.read_bank:
            attach_read_bank(model)
        if args.predictive_writer:
            attach_predictive_writer(model)
        if args.dopamine_modulation:
            attach_dopamine_modulation(model, site=args.dopamine_site or 'post', adapted=bool(args.dopamine_adapted),
                                       sensitivity_init=float(args.dopamine_sensitivity_init or 0.0),
                                       affinity=args.dopamine_affinity)
            print(f'Dopamine scales learning at the {args.dopamine_site or "post"}synaptic end', flush=True)
            if args.online_da_plasticity:
                attach_online_da_plasticity(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                            eta=args.online_da_eta)
                print(f'Online dopamine plasticity at {int(model.online_da_target.sum())} terminals', flush=True)
        if args.kc_dendritic_only:
            n_kk, n_zero = attach_kenyon_dendritic_only(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                                        args.graph.parent / 'kc_kc_dendritic_synapses.feather', rescale=True)
            print(f'Kenyon -> Kenyon: {n_kk} edges, {n_zero} axo-axonal pinned at 0 (newborn)', flush=True)
            args.kc_dendritic_newly = False
        if args.compartment_dendritic:
            stats = attach_compartment_dendritic(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                                 args.graph.parent, rescale=True)
            print(f'Compartment rule (newborn): {stats} (edges, pinned, weighted dendritic share per list)', flush=True)
            args.compartment_dendritic_newly = False
        if args.apl_kc_dendritic_only and not args.compartment_dendritic:
            n_ak, n_az, mean_share = attach_apl_kenyon_dendritic_only(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                                                      args.graph.parent / 'apl_kc_dendritic_synapses.feather', rescale=True)
            print(f'APL -> Kenyon: {n_ak} edges, {n_az} axonal-only pinned at 0, mean calyx share {mean_share:.3f} (newborn)', flush=True)
            args.apl_kc_dendritic_newly = False
        # a newborn lives on the corrected connectome from its first tick (these were applied only on resume before)
        if args.connectome_v2:
            ann_path = args.graph.parent / 'body-annotations.feather'
            pins = {'kenyon_to_dopamine': pin_kenyon_to_dopamine(model, args.graph, ann_path),
                    'sensory_terminals': pin_sensory_terminals(model, args.graph),
                    'octopamine_serotonin_fast': pin_modulatory_fast_edges(model, args.graph,
                                                                           args.graph.parent / 'modulatory_neurons.feather',
                                                                           **modulatory_pin_kwargs(args))}
            print(f'Connectome v2 pins (newborn): {pins}', flush=True)
        if (args.kc_renorm or args.dan_renorm) and graph_is_uniform_synapse(args.graph):
            raise ValueError('--kc-renorm / --dan-renorm undo a sqrt(K_in) normalization this graph does not have')
        args.kc_renorm_newly = False
        args.dan_renorm_newly = False
        from information_boltzmann.core.fly_bptt_learning import split_kenyon_subtypes, split_modulatory_classes, split_by_type
        if args.input_gain_size_exponent is not None:
            attach_input_gain(model)                     # before the type split, so every type inherits it
        args.kc_dan_release = bool(args.kc_dan_release)
        if args.kc_dan_release:
            print(f'Kenyon -> dopamine-terminal release (newborn): {attach_kenyon_terminal_release(model, args.graph, args.graph.parent / "body-annotations.feather")} (edges, dopamine cells)', flush=True)
        if getattr(args, 'kc_habituation_tau', None):
            print(f'Kenyon release habituation (tau ticks, use): {attach_kenyon_habituation(model, args.kc_habituation_tau, args.kc_habituation_use or 0.2)}', flush=True)
        if args.type_classes:
            if args.kc_subtypes or args.modulatory_classes:
                raise ValueError('--type-classes already gives every cell type its own physiology (no --kc-subtypes / --modulatory-classes)')
            print(f'Type classes (newborn): {split_by_type(model, args.graph, args.graph.parent / "body-annotations.feather")} cell types with their own physiology', flush=True)
        if args.input_gain_size_exponent is not None:
            print(f'Input gain per class from size^{args.input_gain_size_exponent} (min, median, max): '
                  f'{init_input_gain_from_size(model, args.graph, args.input_gain_size_exponent, kenyon_pooled=bool(args.input_gain_kenyon_pooled))}', flush=True)
        if args.kc_subtypes:
            print(f'Kenyon subtypes (newborn): {split_kenyon_subtypes(model, args.graph, args.graph.parent / "body-annotations.feather")}', flush=True)
        if args.modulatory_classes:
            if not args.kc_subtypes:
                raise ValueError('--modulatory-classes comes after the Kenyon split (--kc-subtypes)')
            print(f'Modulatory classes (newborn): {split_modulatory_classes(model, args.graph, args.graph.parent / "body-annotations.feather")}', flush=True)
        if args.amine_modulation:
            n_contacts = attach_amine_modulation(model, args.graph,
                                                 args.graph.parent / 'modulatory_neurons.feather')
            print(f'Native OA/5-HT chemical delivery (newborn): {n_contacts} anatomical contacts', flush=True)
        if args.cx_state:
            n_cx = attach_cx_state(model, args.graph, args.graph.parent / 'body-annotations.feather')
            print(f'Native CX persistent state (newborn): {n_cx} CX neurons', flush=True)
        if args.mb_consolidation:
            n_dpm, n_apl = attach_mb_consolidation(model, args.graph, args.graph.parent / 'body-annotations.feather')
            print(f'Native MB consolidation (newborn): {n_dpm} DPM, {n_apl} APL neurons', flush=True)
        if args.mb_terminal_projection:
            print(f'Native MB axonal projection (newborn): {attach_mb_terminal(model, args.graph, args.graph.parent / "body-annotations.feather", args.graph.parent / "dendritic_share_Kenyon_Cell.feather")}', flush=True)
        if args.apl_cable:
            if not args.no_graph:
                raise ValueError('APL sparse solve requires --no-graph')
            attach_apl_cable(model, args.graph.parent / 'spatial/apl_cable_native_v1.npz')
        if args.apl_local_release:
            attach_apl_local_release(model)
        if args.body_loop:
            model.body_proxy_compatibility = args.body_proxy_compatibility
            n_dn, n_an = attach_body_loop(model, args.graph, args.graph.parent / 'body-annotations.feather')
            print(f'Historical body state (newborn): {n_dn} descending, {n_an} ascending; proxy feedback={model.body_proxy_compatibility}', flush=True)
        if args.body_actuator:
            if not args.mother_native_context:
                raise ValueError('Executed-action body interface requires native-context source tapes')
            print(f'Historical scalar actuator state (newborn): {attach_body_actuator(model, args.graph, args.graph.parent / "body-annotations.feather")} cells; proxy feedback={args.body_proxy_compatibility}', flush=True)
        if args.endocrine_exchange:
            print(f'LK receptor axis (newborn): {attach_endocrine_exchange(model, args.graph, args.graph.parent / "body-annotations.feather")}', flush=True)
        if args.olfactory_relay:
            n_al, n_pn, n_contacts = attach_olfactory_relay(model, args.graph, args.graph.parent / 'body-annotations.feather')
            print(f'Native antennal-lobe relay (newborn): {n_al} local, {n_pn} PN, {n_contacts} inhibitory contacts', flush=True)
        if args.intrinsic:
            if not args.intrinsic_pattern:
                from information_boltzmann.core.fly_bptt_learning import INTRINSIC_TYPE_PATTERN
                args.intrinsic_pattern = INTRINSIC_TYPE_PATTERN       # saved: a later default change cannot alter the lineage
            released = attach_intrinsic_drive(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                              **({'kappa': args.intrinsic_kappa} if args.intrinsic_kappa is not None else {}),
                                              **({'dan_kappa': args.dan_kappa} if getattr(args, 'dan_kappa', None) is not None else {}),
                                              pattern=args.intrinsic_pattern, jitter=float(args.intrinsic_jitter or 0.0),
                                              mbon_kappa=float(args.intrinsic_mbon_kappa or 0.0))
            print(f'Released {int(model.intrinsic_index.numel())} spontaneously active cells (newborn): {released}', flush=True)
        if args.graded_types:
            print(f'Graded (non-spiking) cells (newborn): {attach_graded_cells(model, args.graph, args.graph.parent / "body-annotations.feather", types=tuple(args.graded_types.split(",")), gain=args.graded_gain)}', flush=True)
        if args.visual_graded:
            counts = attach_visual_physiology(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                             gain=0.025 if args.visual_graded_gain is None else args.visual_graded_gain)
            print(f'Early visual graded physiology (newborn): {counts}', flush=True)
        if args.compartment_memory or args.taste_every:
            rate = {} if args.comp_rate is None else {'eta': -float(args.comp_rate)}
            if args.comp_trace_tau:
                rate['tau_trace'] = float(args.comp_trace_tau)    # newborn: the Kenyon trace's start (learnable)
            n_comp = attach_compartment_memory(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                               affinity=args.dopamine_affinity, **rate)
            attach_taste_heads(model, args.graph, args.graph.parent / 'body-annotations.feather')
            args.compartment_memory = True
            print(f'Compartment memory on {n_comp} Kenyon -> MBON synapses; taste heads on the MBONs (newborn)', flush=True)
            model.comp_adapted = bool(args.compartment_adapted)
        if getattr(args, 'ai', False):
            # acts from birth (2026-10-15): the same heads a resume attaches (look / speak muscles on the output cells)
            init = torch.load(args.act_init, map_location='cpu')
            if args.motor_acts:
                from information_boltzmann.core.fly_active_inference import attach_motor_acts
                attach_motor_acts(model, tau_muscle=float(args.act_tau or 8.0), eye_open=float(args.eye_init or 0.9),
                                  habit=float(args.act_habit if args.act_habit is not None else 0.2),
                                  brain_gradient=bool(args.act_brain_grad),
                                  dopamine_precision=bool(args.act_dopamine_precision),
                                  eye_gate=getattr(args, 'eye_gate', 'atan'))
            else:
                attach_ai_head(model, init, eye_open=float(getattr(args, 'eye_init', 0.95) or 0.95))
            print('Act head (eyes + output) attached at birth', flush=True)
        if args.native_motor_port:
            if getattr(model, 'ai_mode', None) != 'motor' or not args.mother_speak_act:
                raise ValueError('Native motor port needs motor speaking actions')
            print(f'Native motor language peripheral (newborn): {attach_native_motor_port(model, args.graph, args.graph.parent / "body-annotations.feather")} neurons', flush=True)
        if args.stream_jepa:
            print(f'Stream JEPA term on {attach_stream_jepa(model, args.graph, args.graph.parent / "body-annotations.feather")} Kenyon cells (newborn)', flush=True)
        if args.homeostasis:
            args.homeostasis_seed = int(args.homeostasis_seed or 0)
            attach_homeostasis(model, args.graph, seed=args.homeostasis_seed)
            print(f'Homeostatic intrinsic plasticity, individual {args.homeostasis_seed}', flush=True)

        n = model.n_neurons
        native_program = None
        if args.native_runtime_program is not None:
            if not args.no_graph:
                raise ValueError('Generated native execution currently requires --no-graph')
            document = json.loads(args.native_runtime_program.read_text(encoding='utf-8'))
            native_program = attach_program_document(model, document)
        native_assembly = assemble_spatial_components(model, args)
        validate_program_bindings(model)
        print(f'Native spatial assembly (newborn): {native_assembly}', flush=True)
        h = torch.zeros(1, n, device='cuda')
        ring = tuple(torch.zeros(1, n, device='cuda') for _ in range(4))
        ge = torch.zeros(1, n, device='cuda')
        gi = torch.zeros(1, n, device='cuda')
        b = torch.zeros(1, n, device='cuda')
        x = torch.ones(1, n, device='cuda')
        u = model.get_stp_params()[0].clone().expand_as(h).contiguous()
        baseline = torch.zeros(1, model.topographic_writer.n_total if model.topographic_writer else n, device='cuda')
        h_mean = torch.zeros(1, n, device='cuda')
        dan_gate = torch.zeros(1, n, device='cuda') if args.dan_plastic_lr > 0 else torch.empty(0, device='cuda')
        gamma_z1 = torch.zeros(1, model.n_read, device='cuda') if args.use_read_gamma_trace else torch.empty(0, device='cuda')
        gamma_z2 = torch.zeros(1, model.n_read, device='cuda') if args.use_read_gamma_trace else torch.empty(0, device='cuda')
        physical = FlyPhysicalState(h, ring, ge, gi, b, x, u, baseline, h_mean, dan_gate, gamma_z1, gamma_z2)
        if native_program is not None:
            native_device = 'cpu' if model.native_program_state_device == 'cpu' else h.device
            physical.native_state = native_program.layout.initialize(device=native_device, dtype=h.dtype)

        names = [
            'output_read.weight', 'read_norm.weight', 'decoder.weight', 'decoder.bias',
            'log_threshold', 'log_tau_m', 'log_beta', 'log_tau_a',
            'log_tau_s_e', 'log_tau_s_i', 'log_g_e', 'log_g_i',
            'topographic_writer.gate_linear.weight', 'topographic_writer.gate_linear.bias'
        ]
        if args.learn_stp:
            names.extend(STP_PARAMETER_NAMES)
        if args.intrinsic:
            names.extend(INTRINSIC_PARAMETER_NAMES)
        if args.graded_types:
            names.extend(GRADED_PARAMETER_NAMES)
        if args.input_gain_size_exponent is not None:
            names.append('log_input_gain')
        if args.kc_site_stp:
            names.extend(KC_SITE_STP_PARAMETER_NAMES)
        if args.kc_dan_release:
            names.append('log_kc_release')
        if args.compartment_memory:
            names.extend(n for n in COMPARTMENT_PARAMETER_NAMES + TASTE_PARAMETER_NAMES if n not in names)
            if args.comp_tonic_forget:
                print(f'Active forgetting of the compartment memory: tau {attach_comp_tonic_forget(model, args.comp_tonic_forget):.0f} '
                      f'ticks at tonic dopamine (newborn)', flush=True)
                names.extend(n for n in COMP_TONIC_PARAMETER_NAMES if n not in names)
            if args.comp_timing:
                print(f'Compartment timing rule (newborn): dopamine before Kenyon potentiates (start '
                      f'{attach_comp_timing(model, burst_k=args.comp_burst_k):.3f}), use-dependent forgetting off', flush=True)
                names.extend(n for n in ('comp_eta_pot', 'comp_burst_k') if n not in names)
                if args.comp_camp_tau and args.comp_burst_relative:
                    attach_comp_relative_burst(model)
                    print(f'cAMP integrator of the dopamine contrast (newborn): tau {attach_comp_camp(model, args.comp_camp_tau):g} ticks', flush=True)
                    names.append('comp_logit_camp')
                if args.comp_burst_relative:
                    print(f'Burst threshold on the dopamine contrast (newborn): {attach_comp_relative_burst(model)}', flush=True)
                if args.comp_unit_traces:
                    print(f'Unit eligibility traces (newborn), rates (dep, pot) start {attach_comp_unit_traces(model)}', flush=True)
        if args.stream_jepa:
            names.extend(n for n in STREAM_JEPA_PARAMETER_NAMES if n not in names)
        if args.use_read_gamma_trace:
            names.append('logit_read_gamma')
        if args.use_latent_predictor and model.latent_predictor is not None:
            for p_name, _ in model.latent_predictor.named_parameters():
                names.append(f'latent_predictor.{p_name}')
        if args.use_graph_observer and getattr(model, 'graph_observer', None) is not None:
            for p_name, _ in model.graph_observer.named_parameters():
                names.append(f'graph_observer.{p_name}')

        named_now = dict(model.named_parameters())
        for group in (GO_PARAMETER_NAMES, READ_BANK_PARAMETER_NAMES, PREDICTIVE_PARAMETER_NAMES, DOPAMINE_PARAMETER_NAMES,
                      AMINE_PARAMETER_NAMES,
                      CX_PARAMETER_NAMES,
                      MB_CONSOLIDATION_PARAMETER_NAMES,
                      BODY_LOOP_PARAMETER_NAMES,
                      AL_PARAMETER_NAMES,
                      VISUAL_PARAMETER_NAMES,
                      BODY_ACTUATOR_PARAMETER_NAMES, ENDOCRINE_PARAMETER_NAMES,
                      MOTOR_PORT_PARAMETER_NAMES,
                      ONLINE_DA_PARAMETER_NAMES):
            names.extend(name for name in group if name in named_now and name not in names)
        if getattr(args, 'pacemaker_homeostasis', False) and getattr(model, 'intrinsic_kappa', None) is not None:
            # the pacemakers' drive belongs to their homeostat, not to the prediction gradient
            print(f'Pacemaker homeostasis on {attach_pacemaker_homeostasis(model, exclude_cells=(np.load(args.graph, allow_pickle=True)["dan_indices"] if args.homeostat_skip_dans else None))} spontaneously active cells '
                  f'(kappa out of the gradient)', flush=True)
            names = [n for n in names if n not in INTRINSIC_PARAMETER_NAMES]
        names.extend(n for n in make_sense_gains(model, args)[0] if n not in names)
        args.time_scale = float(args.time_scale or 1.0)
        if args.time_scale != 1.0:
            print(f'Time scale x{set_time_scale(model, args.time_scale, birth=True):g} (newborn): time constants / k, '
                  f'conductances and sensory drive x k', flush=True)
        if getattr(args, 'ai', False):
            named_ai = dict(model.named_parameters())
            names.extend(n for n in AI_PARAMETER_NAMES if n in named_ai and n not in names)
        if getattr(model, 'mbon_kappa', None) is not None:
            names.extend(n for n in ('mbon_kappa',) if n not in names)
        if getattr(args, 'carried_credit', False):
            # credit beyond the BPTT window at O(1) memory for every slow state that outlives it and has trained
            # parameters (fly_carried_credit): compartment memory, dopamine clocks, STP (with --learn-stp)
            from information_boltzmann.core.fly_carried_credit import attach_carried_credit
            mech = [m for m, ok in (('comp', getattr(model, 'comp_eta', None) is not None),
                                    ('dopamine', getattr(model, 'dopamine_logit_slow', None) is not None),
                                    ('stp', bool(args.learn_stp))) if ok]
            print(f'Carried credit beyond the window: {attach_carried_credit(model, mech)}', flush=True)
        if args.disable_alif:
            disable_alif(model)
        model.read_spikes = bool(args.read_spikes)
        learner = FlyBPTTLearner(model, physical, adam_names=names,
            lr=args.lr, lr_synapse=args.lr_synapse, lr_sensory=args.lr_sensory,
            plasticity_optimizer=args.plasticity_optimizer, lr_decoder=args.lr_decoder,
            settle_ticks=args.settle_ticks, writer_baseline_clock=args.writer_baseline_clock,
            learn_stp=bool(args.learn_stp), lambda_jepa=args.lambda_jepa, lr_physiology=args.lr_physiology,
            use_ctm_loss=bool(args.ctm_loss),
            use_checkpointing=bool(args.checkpointing),
            lambda_mcr2=float(args.lambda_mcr2 or 0.0),
            eps_mcr2=float(args.eps_mcr2 or 0.5),
            adaptive_admission=bool(args.adaptive_admission),
            min_settle_ticks=args.min_settle_ticks,
            flux_baseline=args.flux_baseline,
            stream_ticks=int(args.stream_ticks or 0), hold_input=bool(args.hold_input),
            read_lag=int(args.read_lag or 0), go_neuron=bool(args.go_neuron),
            go_tick_cost=float(args.go_tick_cost or 0.0), go_reference_energy=float(args.go_reference_energy or 0.07),
            go_max_ticks=int(args.go_max_ticks or 32))
        learner.stream_pace_mean = float(args.stream_pace_mean or 0.0)
        learner.stream_segment_ticks = int(args.stream_segment_ticks)
        learner.native_credit_cpu = bool(args.native_credit_cpu)
        learner.stream_pace_min = float(getattr(args, 'stream_pace_min', 0.0) or 0.0)
        if getattr(args, 'comp_rate_calibrate', False) and getattr(model, 'comp_eta', None) is not None:
            # the compartment rates' start from the newborn's own dopamine: the median edge writes as fast as it forgets
            eta_c, pot_c, r_c = calibrate_comp_rate(model, learner.state, np.asarray(train[:128]),
                                                    ticks=int(round(args.stream_pace_mean or 8)), hold=bool(args.hold_input))
            print(f'Compartment rates calibrated on 128 words: depression {eta_c:.4f}, potentiation {pot_c:.4f} '
                  f'(active forgetting {r_c:.2e} per tick)', flush=True)
        if getattr(args, 'ai', False):
            # a newborn's omitted switches take the same defaults a resume would (acts from birth, 2026-10-15)
            learner.ai_epistemic, learner.ai_pragmatic = float(args.ai_epistemic or 0.0), float(args.ai_pragmatic or 0.0)
            learner.ai_sigreg = float(args.ai_sigreg or 0.0)
            learner.eye_forced = 1.0 if args.eye_forced_open else args.eye_forced
            learner.eye_binary, learner.ai_depth = bool(args.eye_binary), int(args.ai_depth or 0)
            learner.sense_noise_max = float(args.sense_noise or 0.0)
            learner.act_habit = float(args.act_habit if args.act_habit is not None else 0.2)
            learner.act_gamma_inv = float(args.act_gamma_inv if args.act_gamma_inv is not None else 0.05)
            learner.epistemic_source = 'belief' if args.epistemic_belief else 'count'
            if args.motor_acts and (getattr(model, 'ai_mu', None) is None
                                    or (getattr(model, 'ai_precision', False) and getattr(model, 'ai_da_ref', None) is None)):
                from information_boltzmann.core.fly_active_inference import calibrate_muscles
                spread = calibrate_muscles(model, learner.state, np.asarray(train[:128]),
                                           ticks=int(round(args.stream_pace_mean or 8)), hold=bool(args.hold_input))
                print(f'Muscles calibrated on 128 words (median spread {spread:.4f})'
                      + (f'; dopamine tone {float(model.ai_da_ref):.4f} (gamma = 1)' if getattr(model, 'ai_precision', False) else ''),
                      flush=True)
            if args.miss_prior:
                # the listener's prior: unigram CE of every word from the first 20M training tokens
                _c = np.bincount(np.asarray(train[:20_000_000], dtype=np.int64), minlength=50257).astype(np.float64) + 0.5
                learner.unigram_ce = (-np.log(_c / _c.sum())).astype(np.float32)
                print(f'Miss value: the listener prior (unigram CE, mean over tokens '
                      f'{float((learner.unigram_ce[np.asarray(train[:200_000])]).mean()):.3f})', flush=True)
            if args.eye_binary:
                # a new stretch of the world every chunk (pace and noise schedules seeded by the lifetime so far)
                learner.pace_seed = int(getattr(args, 'odour_seed_value', 0) or 0)
                learner.eye_init = float(args.eye_init)
                model.eye_floor = float(args.eye_floor or 0.0)
            if args.noise_rate > 0:
                import numpy as _np
                alphabet = _np.random.default_rng(args.noise_seed).choice(50257, args.noise_alphabet, replace=False)
                learner.noise_world = {'rate': args.noise_rate, 'length': args.noise_length,
                                       'alphabet': alphabet.tolist(), 'seed': args.noise_seed,
                                       'offset': int(getattr(args, 'odour_seed_value', 0) or 0)}
            print(f'M3: binary eyes {learner.eye_binary}, question depth {learner.ai_depth}, noise rate {args.noise_rate}',
                  flush=True)
            print(f'Active inference: epistemic {args.ai_epistemic}, pragmatic {args.ai_pragmatic}, SIGReg {args.ai_sigreg}'
                  f'{", eyes forced " + str(learner.eye_forced) if learner.eye_forced is not None else ""}', flush=True)
        if args.evidence_plasticity:
            learner.attach_evidence_plasticity(args.pull_tau, saved=None, edges=not args.eg_plasticity)
            print(f'Evidence plasticity: {"" if args.eg_plasticity else "synapses + "}{len(learner.physio_names)} physiology tensors, significance-gated (|mean| > z se), '
                  f'pull to birth tau {args.pull_tau:g} updates; read surface = '
                  f'{"spikes (Gamma)" if args.read_spikes else "membranes"}', flush=True)
        if args.eg_plasticity:
            learner.attach_eg_plasticity(args.relative_edge_rate or 0.02, args.eg_tau, saved=None)
            print(f'EG plasticity: log-space rate {args.relative_edge_rate or 0.02} per update at the running RMS, '
                  f'KL to birth tau {args.eg_tau:g} updates, per-cell E/I totals kept (update {learner.eg_count})', flush=True)
        if getattr(args, 'relative_edge_rate', 0.0) and not args.eg_plasticity:
            step = learner.set_relative_edge_steps(args.graph, args.relative_edge_rate)
            print(f'Relative plasticity: {args.relative_edge_rate} of the birth weight per step '
                  f'(median synapse step {step:.2e})', flush=True)
        if getattr(args, 'odour_world', False):
            learner.odour_world = OdourWorld(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                             seed=int(getattr(args, 'odour_seed_value', 0)),
                                             odour_drive=args.odour_drive, receptors=args.odour_receptors,
                                             outcome_dopamine=bool(args.outcome_dopamine),
                                             gusts=bool(getattr(args, 'sense_prediction', False)),
                                             levy_scale=float(args.levy_scale or 0.0), info_value=bool(args.info_value))
            learner.sense_lambda = float(getattr(args, 'sense_lambda', 1.0))
            learner.act_lambda = float(args.act_lambda)
            print(f'Odour world: {len(learner.odour_world.pool)} receptor types, {args.odour_receptors} per odour, '
                  f'outcome dopamine electrode {bool(args.outcome_dopamine)}, '
                  f'drive {args.odour_drive}, act lambda {args.act_lambda}', flush=True)
        learner.previous_token = int(train[0])

        initial_stp = {name: getattr(model, name).detach().clone() for name in stp_names(model)}
        origin = {
            'initialization': 'born from scratch with GPT-2 embedding/unigram prior and LeJEPA Latent Predictor Form B',
            'train_tokens': 0, 'events': 0, 'updates': 0,
        }
        trained = 0
        best = math.inf
        cursor = val_cursor = 0
        recent_a = []
        recent_scores = []
        target = args.additional_tokens
        budget_start = 0
    else:
        print('Loading archived weights and continuing physical state...', flush=True)
        saved = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True)
        fmt, previous_config = saved['format'], saved['config']
        changed = inherit_profile_switches(parser, args, previous_config)
        if changed:
            print(f'Inherited from the checkpoint (profile values not used): {changed}', flush=True)
        print_switches(parser, args, 'inherited')
        args.odour_seed_value = int(saved.get('bptt_train_tokens', 0) or 0)      # a new life sequence per chunk
        old = saved['learner']
        previous_centering = bool(old.get('read_centering', previous_config.get('read_centering', False)))
        previous_dan_lr = float(old.get('dan_plastic_lr', previous_config.get('dan_plastic_lr', 0.0)))
        resolve_learning_modes(args, previous_config, old)
        previous_gamma = bool(old.get('use_read_gamma_trace', previous_config.get('use_read_gamma_trace', False)))
        previous_predictor = bool(old.get('use_latent_predictor', previous_config.get('use_latent_predictor', False)))
        previous_observer = bool(old.get('use_graph_observer', previous_config.get('use_graph_observer', False)))
        mode_changed = (args.read_centering != previous_centering or
                        args.dan_plastic_lr != previous_dan_lr or
                        args.use_read_gamma_trace != previous_gamma or
                        args.use_latent_predictor != previous_predictor or
                        args.use_graph_observer != previous_observer or
                        args.native_motor_port != bool(previous_config.get('native_motor_port', False)) or
                        args.amine_modulation != bool(previous_config.get('amine_modulation', False)) or
                        args.mb_terminal_projection != bool(previous_config.get('mb_terminal_projection', False)) or
                        args.apl_cable != bool(previous_config.get('apl_cable', False)) or
                        args.apl_local_release != bool(previous_config.get('apl_local_release', False)) or
                        args.kc_terminal != bool(previous_config.get('kc_terminal', False)) or
                        args.kc_site_stp != bool(previous_config.get('kc_site_stp', False)) or
                        args.kc_contact_memory != bool(previous_config.get('kc_contact_memory', False)) or
                        args.kc_lateral != bool(previous_config.get('kc_lateral', False)) or
                        args.sensory_terminal != bool(previous_config.get('sensory_terminal', False)) or
                        args.kc_dopamine != bool(previous_config.get('kc_dopamine', False)) or
                        args.dan_local_release != bool(previous_config.get('dan_local_release',False)) or
                        args.body_proxy_compatibility != bool(previous_config.get(
                            'body_proxy_compatibility', previous_config.get('body_loop', False))) or
                        args.detach_reset != bool(previous_config.get('detach_reset', False)) or
                        args.surrogate_mode != previous_config.get('surrogate_mode', 'absolute') or
                        args.transmission_mode != previous_config.get('transmission_mode', 'atomic'))
        if args.reference_train is None:
            args.reference_train = Path(previous_config.get('reference_train', 'data/ib_owt_gpt2/train.npy'))
        if fmt not in ('fly-online-v2', 'fly-bptt-v1'):
            raise ValueError('Resume requires a complete lifecycle checkpoint')
        model = FlyReservoirLM(args.graph, vocab_size=50257, d_model=previous_config['d_model'],
            injection='topographic', read_surface='output', synapse_model='coba',
            extra_read_index=read_extra_index(args.graph, args.read_extra_classes),
            use_alif=True, decoder_bias=previous_config['decoder_bias'], **biophysics_kwargs(args),
            read_centering=args.read_centering,
            use_read_gamma_trace=args.use_read_gamma_trace,
            init_read_gamma=args.init_read_gamma,
            use_latent_predictor=args.use_latent_predictor,
            use_graph_observer=args.use_graph_observer,
            lambda_obs=args.lambda_jepa,
            lambda_sigreg=args.lambda_sigreg,
            max_horizon=args.max_horizon,
            dagger_beta=args.dagger_beta,
            obs_init_gamma=getattr(args, 'obs_init_gamma', 0.0),
            detach_reset=args.detach_reset, surrogate_mode=args.surrogate_mode,
            transmission_mode=args.transmission_mode).cuda()
        model.dan_plastic_lr = args.dan_plastic_lr
        if args.go_neuron == 'virtual':
            attach_go_neuron(model)
        elif args.go_neuron == 'giant_fiber':
            attach_connectome_go(model, args.graph)
        if args.read_bank:
            attach_read_bank(model)
        if args.intrinsic:
            released = attach_intrinsic_drive(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                              **({'kappa': args.intrinsic_kappa} if args.intrinsic_kappa is not None else {}),
                                              **({'dan_kappa': args.dan_kappa} if getattr(args, 'dan_kappa', None) is not None else {}),
                                              **({'pattern': args.intrinsic_pattern} if args.intrinsic_pattern else {}),
                                              mbon_kappa=float(args.intrinsic_mbon_kappa or 0.0))
            print(f'Released {int(model.intrinsic_index.numel())} spontaneously active cells: {released}', flush=True)
        if args.graded_types:
            print(f'Graded (non-spiking) cells: {attach_graded_cells(model, args.graph, args.graph.parent / "body-annotations.feather", types=tuple(args.graded_types.split(",")), gain=args.graded_gain)}', flush=True)
        if args.visual_graded:
            counts = attach_visual_physiology(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                             gain=args.visual_graded_gain)
            print(f'Early visual graded physiology (resume): {counts}', flush=True)
        if args.predictive_writer:
            attach_predictive_writer(model)
        if args.dopamine_modulation:
            attach_dopamine_modulation(model, site=args.dopamine_site or 'post', adapted=bool(args.dopamine_adapted),
                                       sensitivity_init=float(args.dopamine_sensitivity_init or 0.0),
                                       affinity=args.dopamine_affinity)
            print(f'Dopamine scales learning at the {args.dopamine_site or "post"}synaptic end', flush=True)
            if args.online_da_plasticity:
                attach_online_da_plasticity(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                            eta=args.online_da_eta)
                print(f'Online dopamine plasticity at {int(model.online_da_target.sum())} terminals', flush=True)
        if args.precision:
            n_mod, n_da = attach_precision(model, args.graph, args.graph.parent / 'body-annotations.feather')
            print(f'Precision carried by {n_mod} neuromodulatory cells ({n_da} dopaminergic)', flush=True)
        if args.writer in WRITER_SENSE:
            n_cells, n_types = attach_receptor_writer(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                                      sense_class=WRITER_SENSE[args.writer])
            print(f'Receptor writer: {n_cells} sensory cells, {n_types} rows', flush=True)
        tie_now = bool(args.tied_writer) and not bool(previous_config.get('tied_writer', False))
        if args.tied_writer and not tie_now:
            # a tied lineage: the writer's chemosensory rows are per receptor type (shapes before loading)
            attach_tied_writer(model, args.graph, args.graph.parent / 'body-annotations.feather', convert=False)
            print('Writer tied by chemosensory receptor type', flush=True)
        from information_boltzmann.core.fly_bptt_learning import split_kenyon_subtypes, split_modulatory_classes
        from information_boltzmann.core.fly_bptt_learning import split_by_type
        lineage_types = bool(previous_config.get('type_classes'))
        if previous_config.get('input_gain_size_exponent') is not None:
            attach_input_gain(model)                     # before the split replay (the saved gains load afterwards)
        if args.kc_dan_release:                          # before the split replay; a new switch starts every type at 0.1
            print(f'Kenyon -> dopamine-terminal release: {attach_kenyon_terminal_release(model, args.graph, args.graph.parent / "body-annotations.feather")} (edges, dopamine cells)', flush=True)
            if args.kc_habituation_tau:
                print(f'Kenyon release habituation (tau ticks, use): {attach_kenyon_habituation(model, args.kc_habituation_tau, args.kc_habituation_use)}', flush=True)
        if lineage_types:
            # a lineage with one physiology class per cell type: the same split before loading (supersedes the others)
            print(f'Type classes (carried over): {split_by_type(model, args.graph, args.graph.parent / "body-annotations.feather", record=False)}', flush=True)
        lineage_split = not lineage_types and (bool(previous_config.get('kc_subtypes')) or saved['model']['log_tau_m'].numel() > model.n_superclasses)
        lineage_mod = not lineage_types and bool(previous_config.get('modulatory_classes'))
        if lineage_mod and not lineage_split:
            raise ValueError('a modulatory-class lineage must also carry the Kenyon split (fixed class order)')
        if lineage_split:
            # a lineage whose Kenyon subtypes already have their own physiology: the same split before loading
            print(f'Kenyon subtypes (carried over): {split_kenyon_subtypes(model, args.graph, args.graph.parent / "body-annotations.feather", record=False)}', flush=True)
        if lineage_mod:
            print(f'Modulatory classes (carried over): {split_modulatory_classes(model, args.graph, args.graph.parent / "body-annotations.feather", record=False)}', flush=True)
        if args.amine_modulation:
            n_contacts = attach_amine_modulation(model, args.graph,
                                                 args.graph.parent / 'modulatory_neurons.feather')
            print(f'Native OA/5-HT chemical delivery (resume): {n_contacts} anatomical contacts', flush=True)
        if args.cx_state:
            n_cx = attach_cx_state(model, args.graph, args.graph.parent / 'body-annotations.feather')
            print(f'Native CX persistent state (resume): {n_cx} CX neurons', flush=True)
        if args.mb_consolidation:
            n_dpm, n_apl = attach_mb_consolidation(model, args.graph, args.graph.parent / 'body-annotations.feather')
            print(f'Native MB consolidation (resume): {n_dpm} DPM, {n_apl} APL neurons', flush=True)
        if args.body_loop:
            model.body_proxy_compatibility = args.body_proxy_compatibility
            n_dn, n_an = attach_body_loop(model, args.graph, args.graph.parent / 'body-annotations.feather')
            print(f'Historical body state (resume): {n_dn} descending, {n_an} ascending; proxy feedback={model.body_proxy_compatibility}', flush=True)
        if args.body_actuator:
            if not args.mother_native_context:
                raise ValueError('Executed-action body interface requires native-context source tapes')
            print(f'Historical scalar actuator state (resume): {attach_body_actuator(model, args.graph, args.graph.parent / "body-annotations.feather")} cells; proxy feedback={args.body_proxy_compatibility}', flush=True)
        if args.endocrine_exchange:
            print(f'LK receptor axis (resume): {attach_endocrine_exchange(model, args.graph, args.graph.parent / "body-annotations.feather")}', flush=True)
        if args.olfactory_relay:
            n_al, n_pn, n_contacts = attach_olfactory_relay(model, args.graph, args.graph.parent / 'body-annotations.feather')
            print(f'Native antennal-lobe relay (resume): {n_al} local, {n_pn} PN, {n_contacts} inhibitory contacts', flush=True)
        if saved['model']['log_tau_m'].numel() != model.n_superclasses:
            raise ValueError(f'class count mismatch: checkpoint {saved["model"]["log_tau_m"].numel()}, model {model.n_superclasses}')
        native_document = old.get('native_runtime_document')
        if args.native_runtime_program is not None:
            requested = json.loads(args.native_runtime_program.read_text(encoding='utf-8'))
            if native_document is None or json.dumps(requested, sort_keys=True) != json.dumps(native_document, sort_keys=True):
                raise ValueError('New native program requires explicit complete-state archive migration')
        if native_document is not None:
            if not args.no_graph:
                raise ValueError('Generated native execution currently requires --no-graph')
            attach_program_document(model, native_document, saved=saved['model'])
        weights = {key: value for key, value in saved['model'].items()
                   if key not in ('edge_weight_e', 'edge_weight_i')
                   and key not in MOTOR_PORT_PARAMETER_NAMES
                   and key != 'native_motor_body_ids'
                   and key not in MB_TERMINAL_PARAMETER_NAMES + MB_TERMINAL_ID_NAMES
                   and key not in APL_CABLE_PARAMETER_NAMES + APL_CABLE_ID_NAMES + APL_LOCAL_RELEASE_PARAMETER_NAMES
                   and key not in KC_TERMINAL_PARAMETER_NAMES + KC_TERMINAL_ID_NAMES + KC_SITE_STP_PARAMETER_NAMES + KC_CONTACT_ID_NAMES + KC_LATERAL_PARAMETER_NAMES + KC_LATERAL_ID_NAMES + SENSORY_TERMINAL_PARAMETER_NAMES + SENSORY_TERMINAL_ID_NAMES + KC_DOPAMINE_PARAMETER_NAMES + KC_DOPAMINE_ID_NAMES + DAN_RELEASE_PARAMETER_NAMES + DAN_RELEASE_ID_NAMES
                   and key not in COMPARTMENT_PARAMETER_NAMES + TASTE_PARAMETER_NAMES + ACT_STATE_NAMES + AI_STATE_NAMES
                   + STREAM_JEPA_PARAMETER_NAMES
                   # attached later in the resume (comp tonic / timing, make_sense_gains), each then loads its own
                   + COMP_TONIC_PARAMETER_NAMES + ('comp_eta_pot', 'comp_burst_k', 'comp_logit_camp', 'voice_log_gain', 'reward_log_gain')}
        if not args.go_neuron:
            dropped = [k for k in weights if k in GO_PARAMETER_NAMES]
            for k in dropped:
                weights.pop(k)
            if dropped:
                print(f'Go neuron off: its {len(dropped)} parameters are not carried over', flush=True)
        if args.reinit_read_gamma:
            weights.pop('logit_read_gamma', None)
            print('Re-initializing logit_read_gamma with full-causal anatomical distribution [1.0, 115.0] ticks.', flush=True)
        with torch.no_grad():
            for name in ('edge_weight_e', 'edge_weight_i'):
                getattr(model, name).copy_(saved['model'][name])
            if args.visual_graded and 'log_visual_graded_gain' not in saved['model']:
                # Only this newly attached parameter may be absent. Keep all
                # legacy gains, optimizer moments and physical states intact.
                weights['log_visual_graded_gain'] = model.log_visual_graded_gain.detach().clone()
            for name in BODY_ACTUATOR_PARAMETER_NAMES + ENDOCRINE_PARAMETER_NAMES:
                if hasattr(model, name) and name not in saved['model']:
                    weights[name] = getattr(model, name).detach().clone()
            loaded = model.load_state_dict(weights, strict=False)
            if previous_config.get('amine_modulation') and any(
                    name in loaded.missing_keys for name in AMINE_PARAMETER_NAMES):
                raise ValueError('Continuing amine checkpoint is missing learned parameters')
            missing = [k for k in loaded.missing_keys if k != 'logit_read_gamma' and not k.startswith('latent_predictor.') and not k.startswith('graph_observer.') and k not in GO_PARAMETER_NAMES and k not in READ_BANK_PARAMETER_NAMES and k not in INTRINSIC_PARAMETER_NAMES and k not in GRADED_PARAMETER_NAMES and k not in PRECISION_PARAMETER_NAMES and k not in PREDICTIVE_PARAMETER_NAMES and k not in DOPAMINE_PARAMETER_NAMES and k not in AMINE_PARAMETER_NAMES and k not in CX_PARAMETER_NAMES and k not in MB_CONSOLIDATION_PARAMETER_NAMES and k not in BODY_LOOP_PARAMETER_NAMES and k not in AL_PARAMETER_NAMES and k not in ONLINE_DA_PARAMETER_NAMES
                       and not (k == 'log_kc_release' and args.kc_dan_release)]     # a release switched on at this resume
            unexpected = [k for k in loaded.unexpected_keys if not k.startswith('latent_predictor.') and not k.startswith('graph_observer.')]
            if unexpected or missing:
                raise ValueError(f'Weight migration mismatch: unexpected={unexpected}, missing={missing}')
            if args.kc_subtypes and not lineage_split:
                print(f'Kenyon subtypes split from cb_intrinsic (own physiology, initialised from it): '
                      f'{split_kenyon_subtypes(model, args.graph, args.graph.parent / "body-annotations.feather")}', flush=True)
            if args.modulatory_classes and not lineage_mod:
                if not args.kc_subtypes:
                    raise ValueError('--modulatory-classes comes after the Kenyon split (--kc-subtypes)')
                print(f'Modulatory classes split (own physiology, initialised from their class): '
                      f'{split_modulatory_classes(model, args.graph, args.graph.parent / "body-annotations.feather")}', flush=True)
            if args.kc_dendritic_only:
                n_kk, n_zero = attach_kenyon_dendritic_only(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                                            args.graph.parent / 'kc_kc_dendritic_synapses.feather',
                                                            rescale=getattr(args, 'kc_dendritic_newly', True))
                print(f'Kenyon -> Kenyon: {n_kk} edges, {n_zero} axo-axonal pinned at 0'
                      f'{" (weights rescaled now)" if getattr(args, "kc_dendritic_newly", True) else ""}', flush=True)
            if args.compartment_dendritic:
                newly_c = bool(getattr(args, 'compartment_dendritic_newly', False))
                stats = attach_compartment_dendritic(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                                     args.graph.parent, rescale=newly_c)
                print(f'Compartment rule: {stats} (edges, pinned, weighted dendritic share per list)'
                      f'{" (weights rescaled now)" if newly_c else ""}', flush=True)
            if args.apl_kc_dendritic_only and not args.compartment_dendritic:
                newly = bool(getattr(args, 'apl_kc_dendritic_newly', False))
                n_ak, n_az, mean_share = attach_apl_kenyon_dendritic_only(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                                                          args.graph.parent / 'apl_kc_dendritic_synapses.feather', rescale=newly)
                print(f'APL -> Kenyon: {n_ak} edges, {n_az} axonal-only pinned at 0, mean calyx share {mean_share:.3f}'
                      f'{" (weights rescaled now)" if newly else ""}', flush=True)
            if args.connectome_v2:
                ann_path = args.graph.parent / 'body-annotations.feather'
                pins = {'kenyon_to_dopamine': pin_kenyon_to_dopamine(model, args.graph, ann_path),
                        'sensory_terminals': pin_sensory_terminals(model, args.graph),
                        'octopamine_serotonin_fast': pin_modulatory_fast_edges(model, args.graph,
                                                                               args.graph.parent / 'modulatory_neurons.feather',
                                                                               **modulatory_pin_kwargs(args))}
                print(f'Connectome v2 pins: {pins}', flush=True)
            if args.al_reset_newly:
                n_al = reset_inputs_to_birth(model, args.graph, args.graph.parent / 'body-annotations.feather')
                print(f'Antennal-lobe inputs reset to birth now: {n_al} edges', flush=True)
            if (args.kc_renorm and args.kc_renorm_newly or args.dan_renorm and args.dan_renorm_newly)                     and graph_is_uniform_synapse(args.graph):
                raise ValueError('--kc-renorm / --dan-renorm undo a sqrt(K_in) normalization this graph does not have')
            if args.kc_renorm and args.kc_renorm_newly:
                if not args.kc_dendritic_only:
                    raise ValueError('--kc-renorm needs --kc-dendritic-only (the axo-axonal inputs cut first)')
                factor = renormalize_kenyon_inputs(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                                   args.graph.parent / 'kc_kc_dendritic_synapses.feather')
                print(f'Kenyon inputs renormalized over dendritic partners (median factor {factor:.2f})', flush=True)
            if args.dan_renorm and args.dan_renorm_newly:
                if not args.connectome_v2:
                    raise ValueError('--dan-renorm needs --connectome-v2 (the Kenyon -> dopamine edges pinned first)')
                from information_boltzmann.core.fly_bptt_learning import renormalize_dopamine_inputs
                factor = renormalize_dopamine_inputs(model, args.graph, args.graph.parent / 'body-annotations.feather')
                print(f'Dopamine inputs renormalized without their Kenyon partners (median factor {factor:.2f})', flush=True)
            if args.homeostasis:
                attach_homeostasis(model, args.graph, seed=args.homeostasis_seed)
                print(f'Homeostatic intrinsic plasticity, individual {args.homeostasis_seed}', flush=True)
            if args.regime_homeostasis:
                attach_regime_homeostasis(model, args.graph, seed=args.regime_seed, kenyon_z=args.regime_kenyon_z,
                                          annotations=args.graph.parent / 'body-annotations.feather')
                print(f'Regime homeostasis, individual {args.regime_seed}'
                      f'{f", Kenyon z {args.regime_kenyon_z}" if args.regime_kenyon_z is not None else ""}', flush=True)
            if tie_now:
                # tie the writer now, from the per-cell rows just loaded
                attach_tied_writer(model, args.graph, args.graph.parent / 'body-annotations.feather', convert=True,
                                   scale=args.tied_writer_scale)
                print(f'Writer tied now by chemosensory receptor type (scale {args.tied_writer_scale})', flush=True)
            if args.compartment_memory or args.taste_every:
                comp_state = {k: v for k, v in saved['model'].items() if k in COMPARTMENT_PARAMETER_NAMES + TASTE_PARAMETER_NAMES}
                saved_tonic = {k: v for k, v in saved['model'].items() if k in COMP_TONIC_PARAMETER_NAMES}
                saved_tonic_all = {k: v for k, v in saved['model'].items() if k in ('comp_eta_pot', 'comp_burst_k', 'comp_logit_camp')}
                rate = {} if args.comp_rate is None else {'eta': -float(args.comp_rate)}      # forgetting stays slow (0.02)
                n_comp = attach_compartment_memory(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                                   affinity=args.dopamine_affinity, **rate)
                attach_taste_heads(model, args.graph, args.graph.parent / 'body-annotations.feather')
                if comp_state:
                    model.load_state_dict(comp_state, strict=False)
                args.compartment_memory = True
                print(f'Compartment memory on {n_comp} Kenyon -> MBON synapses; taste heads on the MBONs', flush=True)
                model.comp_adapted = bool(args.compartment_adapted)
                if model.comp_adapted:
                    print('Compartment receptors adapt to their own slow dopamine (K_j = max(slow_j, K))', flush=True)
            if args.stream_jepa:
                n_kc = attach_stream_jepa(model, args.graph, args.graph.parent / 'body-annotations.feather')
                jepa_state = {k: v for k, v in saved['model'].items() if k in STREAM_JEPA_PARAMETER_NAMES}
                if jepa_state:
                    model.load_state_dict(jepa_state, strict=False)
                print(f'Stream JEPA term on {n_kc} Kenyon cells ({"resumed" if jepa_state else "new"}), weight {args.stream_jepa_weight}', flush=True)
            if args.ai:
                ai_state = {k: v for k, v in saved['model'].items() if k in AI_STATE_NAMES}
                init = torch.load(previous_config.get('act_init') or args.act_init, map_location='cpu')
                # constants (mu, sigma, read scale) rebuilt from the same init; the learned parameters loaded after
                if args.motor_acts:
                    from information_boltzmann.core.fly_active_inference import attach_motor_acts
                    attach_motor_acts(model, tau_muscle=args.act_tau, eye_open=float(args.eye_init or 0.9),
                                      habit=args.act_habit, brain_gradient=args.act_brain_grad,
                                      dopamine_precision=args.act_dopamine_precision,
                                      eye_gate=getattr(args, 'eye_gate', 'atan'))
                    model.ai_eye_gate = getattr(args, 'eye_gate', 'atan')
                    from information_boltzmann.core.fly_active_inference import set_motor_constants, MOTOR_CONSTANTS
                    ai_state = {k: v for k, v in ai_state.items() if k in ('ai_w', 'ai_b') + MOTOR_CONSTANTS}
                    set_motor_constants(model, ai_state)
                else:
                    attach_ai_head(model, init, eye_open=float(getattr(args, "eye_init", 0.95) or 0.95))
                if ai_state:
                    model.load_state_dict(ai_state, strict=False)
                print(f'Act head (eyes + output) {"carried over" if ai_state else "initialized from " + str(args.act_init)}',
                      flush=True)
            if args.odour_world:
                act_state = {k: v for k, v in saved['model'].items() if k in ACT_STATE_NAMES}
                if act_state:
                    # the normalization (mu, sigma) and the read scale are constants of the initial head (buffers are
                    # not in the saved parameters): rebuild them from the same init, then load the learned parameters
                    init = torch.load(previous_config.get('act_init') or args.act_init, map_location='cpu')
                    attach_act_head(model, {**init, **{k: act_state[k] for k in ('act_mu', 'act_sigma') if k in act_state}})
                    model.load_state_dict(act_state, strict=False)
                    print('Act head carried over', flush=True)
                else:
                    attach_act_head(model, torch.load(args.act_init, map_location='cpu'))
                    print(f'Act head initialized from {args.act_init}', flush=True)
                if args.sense_prediction:
                    from information_boltzmann.core.fly_odour_world import INNATE
                    import pandas as _pd
                    _typ = _pd.read_feather(args.graph.parent / 'body-annotations.feather')['type'].astype(str)
                    size = len({t[4:] for t in _typ if t.startswith('ORN_')} - set(INNATE)) + 2
                    attach_sense_head(model, size)
                    sense_state = {k: v for k, v in saved['model'].items() if k in SENSE_PARAMETER_NAMES}
                    if sense_state:
                        model.load_state_dict(sense_state, strict=False)
                    print(f'Sense prediction: {size} channels{" (carried over)" if sense_state else ""}', flush=True)
            if args.dopamine_modulation and args.dopamine_sensitivity_init is not None:
                model.dopamine_sensitivity.fill_(float(args.dopamine_sensitivity_init))
                print(f'Dopamine sensitivities set to {args.dopamine_sensitivity_init}', flush=True)
        if args.native_motor_port or 'motor_port_projection.weight' in saved['model']:
            if getattr(model, 'ai_mode', None) != 'motor' or not args.mother_speak_act:
                raise ValueError('Native motor port needs motor speaking actions')
            if previous_config.get('native_motor_port') and 'motor_port_projection.weight' not in saved['model']:
                raise ValueError('Native motor checkpoint is missing its action parameters')
            count = attach_native_motor_port(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                             saved=saved['model'])
            model.native_motor_port_enabled = bool(args.native_motor_port)
            print(f'Native motor language peripheral (resume): {count} neurons, enabled={args.native_motor_port}', flush=True)
        native_assembly = assemble_spatial_components(model, args, saved=saved['model'],
                                                       previous_config=previous_config)
        validate_program_bindings(model)
        print(f'Native spatial assembly (resume): {native_assembly}', flush=True)
        previous_settle_ticks = old.get('settle_ticks', 0)
        previous_writer_clock = old.get('writer_baseline_clock', 'physical')
        if args.writer_baseline_clock is None:
            args.writer_baseline_clock = previous_writer_clock
        writer_clock_changed = args.writer_baseline_clock != previous_writer_clock
        if args.settle_ticks is None:
            args.settle_ticks = previous_settle_ticks
        if fmt == 'fly-online-v2':
            if old['counters']['pending'] or old['counters']['edge_pending']:
                raise ValueError('Migration requires the archived completed update boundary')
            physical = FlyPhysicalState.from_online(old, 'cuda')
            names = list(old['grads'])
        else:
            native_device = ('cpu' if getattr(model, 'native_program_state_device', 'brain') == 'cpu' else 'cuda')
            physical = FlyPhysicalState(**{key: tuple(t.to('cuda') for t in value) if key == 'ring'
                else value.to(native_device if key == 'native_state' else 'cuda')
                for key, value in old['physical'].items()})
            if physical.h_mean.numel() != physical.h.numel():
                physical.h_mean = torch.zeros_like(physical.h)
            if args.use_read_gamma_trace:
                num_read = model.n_read if model.read_surface != 'all' else model.n_neurons
                if physical.gamma_z1.numel() == 0 or args.reinit_read_gamma:
                    physical.gamma_z1 = torch.zeros(1, num_read, device='cuda')
                    physical.gamma_z2 = torch.zeros(1, num_read, device='cuda')
            if not hasattr(physical, 'observer_prior') or physical.observer_prior is None:
                physical.observer_prior = torch.empty(0, device='cuda')
            if not hasattr(physical, 'observer_history') or physical.observer_history is None:
                physical.observer_history = torch.empty(0, device='cuda')
            names = [n for n in old['adam_names']
                     if (args.use_graph_observer or not n.startswith('graph_observer.'))
                     and (args.use_latent_predictor or not n.startswith('latent_predictor.'))]
        previous_learn_stp = model.use_stp and all(name in names for name in STP_PARAMETER_NAMES)
        args.learn_stp = bool(args.learn_stp or previous_learn_stp)
        stp_newly_trainable = [name for name in stp_names(model)
                                if args.learn_stp and name not in names]
        stp_changed = bool(stp_newly_trainable)
        newly_trainable = list(stp_newly_trainable)
        if (args.reinit_read_gamma or (args.use_read_gamma_trace and 'logit_read_gamma' not in names)):
            newly_trainable.append('logit_read_gamma')
        if args.go_neuron == 'virtual':
            newly_trainable += [name for name in GO_PARAMETER_NAMES if name not in names]
        if args.read_bank:
            newly_trainable += [name for name in READ_BANK_PARAMETER_NAMES if name not in names]
        if args.intrinsic:
            newly_trainable += [name for name in INTRINSIC_PARAMETER_NAMES if name not in names]
        if args.graded_types:
            newly_trainable += [name for name in GRADED_PARAMETER_NAMES if name not in names]
        if args.precision:
            newly_trainable += [name for name in PRECISION_PARAMETER_NAMES if name not in names]
        if args.predictive_writer:
            newly_trainable += [name for name in PREDICTIVE_PARAMETER_NAMES if name not in names]
        if args.dopamine_modulation:
            newly_trainable += [name for name in DOPAMINE_PARAMETER_NAMES if name not in names]
        if args.amine_modulation:
            newly_trainable += [name for name in AMINE_PARAMETER_NAMES if name not in names]
        if args.cx_state:
            newly_trainable += [name for name in CX_PARAMETER_NAMES if name not in names]
        if args.mb_consolidation:
            newly_trainable += [name for name in MB_CONSOLIDATION_PARAMETER_NAMES if name not in names]
        if args.mb_terminal_projection:
            newly_trainable += [name for name in MB_TERMINAL_PARAMETER_NAMES if name not in names]
        if args.apl_cable:
            newly_trainable += [name for name in APL_CABLE_PARAMETER_NAMES if name not in names]
        if args.apl_local_release:
            newly_trainable += [name for name in APL_LOCAL_RELEASE_PARAMETER_NAMES if name not in names]
        if args.kc_terminal:
            newly_trainable += [name for name in KC_TERMINAL_PARAMETER_NAMES if name not in names]
        if args.kc_site_stp:
            newly_trainable += [name for name in KC_SITE_STP_PARAMETER_NAMES if name not in names]
        if args.kc_lateral:
            newly_trainable += [name for name in KC_LATERAL_PARAMETER_NAMES if name not in names]
        if args.sensory_terminal:
            newly_trainable += [name for name in SENSORY_TERMINAL_PARAMETER_NAMES if name not in names]
        if args.body_loop:
            extra = [name for name in BODY_LOOP_PARAMETER_NAMES if name not in names]
            names = list(names) + extra
            newly_trainable += extra
        if args.olfactory_relay:
            extra = [name for name in AL_PARAMETER_NAMES if name not in names]
            names = list(names) + extra
            newly_trainable += extra
        if args.visual_graded:
            extra = [name for name in VISUAL_PARAMETER_NAMES if name not in names]
            names = list(names) + extra
            newly_trainable += extra
        for enabled, parameters in ((args.body_actuator, BODY_ACTUATOR_PARAMETER_NAMES),
                                     (args.endocrine_exchange, ENDOCRINE_PARAMETER_NAMES),
                                     (args.native_motor_port, MOTOR_PORT_PARAMETER_NAMES)):
            if enabled:
                extra = [name for name in parameters if name not in names and name in dict(model.named_parameters())]
                names = list(names) + extra
                newly_trainable += extra
        if args.online_da_plasticity:
            newly_trainable += [name for name in ONLINE_DA_PARAMETER_NAMES if name not in names]
        if args.kc_dan_release and 'log_kc_release' not in names:
            names = list(names) + ['log_kc_release']
            newly_trainable.append('log_kc_release')
        if args.ai:
            extra = [name for name in AI_PARAMETER_NAMES if name not in names]
            names = list(names) + extra
            newly_trainable += extra
        if args.odour_world:
            extra = [name for name in ACT_PARAMETER_NAMES if name not in names]
            names = list(names) + extra
            newly_trainable += extra
        if args.compartment_memory:
            extra = [name for name in COMPARTMENT_PARAMETER_NAMES + TASTE_PARAMETER_NAMES if name not in names]
            names = list(names) + extra
            newly_trainable += extra
            if args.comp_tonic_forget:
                attach_comp_tonic_forget(model, args.comp_tonic_forget)
                tonic_state = {k: v for k, v in saved_tonic.items()}
                if tonic_state:
                    model.load_state_dict(tonic_state, strict=False)
                print(f'Active forgetting of the compartment memory: tau '
                      f'{float(1 / torch.sigmoid(model.comp_logit_tonic.detach())):.0f} ticks at tonic dopamine', flush=True)
                extra = [name for name in COMP_TONIC_PARAMETER_NAMES if name not in names]
                names = list(names) + extra
                newly_trainable += extra
            if args.comp_timing:
                attach_comp_timing(model, burst_k=args.comp_burst_k)
                if saved_tonic_all:
                    model.load_state_dict(saved_tonic_all, strict=False)
                print('Compartment timing rule: dopamine before Kenyon potentiates, use-dependent forgetting off', flush=True)
                if args.comp_unit_traces:
                    model.comp_unit_traces = True
                if args.comp_burst_relative:
                    attach_comp_relative_burst(model)
                    print('Burst threshold on the dopamine contrast', flush=True)
                    if args.comp_camp_tau:
                        attach_comp_camp(model, args.comp_camp_tau)
                        if 'comp_logit_camp' in saved['model']:
                            model.load_state_dict({'comp_logit_camp': saved['model']['comp_logit_camp']}, strict=False)
                        print(f'cAMP integrator: tau {float(1 / (1 - torch.sigmoid(model.comp_logit_camp.detach()))):.1f} ticks', flush=True)
                        if 'comp_logit_camp' not in names:
                            names = list(names) + ['comp_logit_camp']
                            if 'comp_logit_camp' not in saved['model']:
                                newly_trainable.append('comp_logit_camp')
                    print('Unit eligibility traces (rates from the checkpoint)', flush=True)
                for n_ in ('comp_eta_pot', 'comp_burst_k'):
                    if n_ not in names:
                        names = list(names) + [n_]
                        if n_ not in saved_tonic_all:
                            newly_trainable.append(n_)
        if args.stream_jepa:
            extra = [name for name in STREAM_JEPA_PARAMETER_NAMES if name not in names]
            names = list(names) + extra
            newly_trainable += extra
        if args.use_latent_predictor and model.latent_predictor is not None:
            for p_name, _ in model.latent_predictor.named_parameters():
                full_name = f'latent_predictor.{p_name}'
                if full_name not in names:
                    newly_trainable.append(full_name)
        if args.use_graph_observer and getattr(model, 'graph_observer', None) is not None:
            for p_name, _ in model.graph_observer.named_parameters():
                full_name = f'graph_observer.{p_name}'
                if full_name not in names:
                    newly_trainable.append(full_name)
        if getattr(args, 'pacemaker_homeostasis', False) and getattr(model, 'intrinsic_kappa', None) is not None:
            # the pacemakers' drive belongs to their homeostat, not to the prediction gradient
            print(f'Pacemaker homeostasis on {attach_pacemaker_homeostasis(model, exclude_cells=(np.load(args.graph, allow_pickle=True)["dan_indices"] if args.homeostat_skip_dans else None))} spontaneously active cells '
                  f'(kappa out of the gradient)', flush=True)
            names = [n for n in names if n not in INTRINSIC_PARAMETER_NAMES]
            newly_trainable = [n for n in newly_trainable if n not in INTRINSIC_PARAMETER_NAMES]
        if float(args.time_scale or 1.0) != 1.0:
            print(f'Time scale x{set_time_scale(model, args.time_scale, birth=False):g}: sensory drive x k (time constants '
                  f'carried in the checkpoint)', flush=True)
        sense, sense_new = make_sense_gains(model, args, saved['model'])
        names = list(names) + [n for n in sense if n not in names]
        newly_trainable += sense_new
        if getattr(model, 'mbon_kappa', None) is not None and 'mbon_kappa' not in names:
            names = list(names) + ['mbon_kappa']
            if 'mbon_kappa' not in saved['model']:
                newly_trainable.append('mbon_kappa')
        if getattr(args, 'carried_credit', False):
            # credit beyond the BPTT window at O(1) memory for every slow state that outlives it and has trained
            # parameters (fly_carried_credit): compartment memory, dopamine clocks, STP (with --learn-stp)
            from information_boltzmann.core.fly_carried_credit import attach_carried_credit
            mech = [m for m, ok in (('comp', getattr(model, 'comp_eta', None) is not None),
                                    ('dopamine', getattr(model, 'dopamine_logit_slow', None) is not None),
                                    ('stp', bool(args.learn_stp))) if ok]
            print(f'Carried credit beyond the window: {attach_carried_credit(model, mech)}', flush=True)
        if args.disable_alif:
            disable_alif(model)
        model.read_spikes = bool(args.read_spikes)
        learner = FlyBPTTLearner(model, physical, adam_names=names,
            lr=args.lr, lr_synapse=args.lr_synapse, lr_sensory=args.lr_sensory,
            plasticity_optimizer=args.plasticity_optimizer, lr_decoder=args.lr_decoder,
            settle_ticks=args.settle_ticks, writer_baseline_clock=args.writer_baseline_clock,
            learn_stp=args.learn_stp, lambda_jepa=args.lambda_jepa, lr_physiology=args.lr_physiology,
            use_ctm_loss=bool(args.ctm_loss),
            use_checkpointing=bool(args.checkpointing),
            lambda_mcr2=float(args.lambda_mcr2 or 0.0),
            eps_mcr2=float(args.eps_mcr2 or 0.5),
            adaptive_admission=bool(args.adaptive_admission),
            min_settle_ticks=args.min_settle_ticks,
            flux_baseline=args.flux_baseline,
            stream_ticks=int(args.stream_ticks or 0), hold_input=bool(args.hold_input),
            read_lag=int(args.read_lag or 0), go_neuron=bool(args.go_neuron),
            go_tick_cost=float(args.go_tick_cost or 0.0), go_reference_energy=float(args.go_reference_energy or 0.07),
            go_max_ticks=int(args.go_max_ticks or 32))
        learner.stream_pace_mean = float(args.stream_pace_mean or 0.0)
        learner.stream_segment_ticks = int(args.stream_segment_ticks)
        learner.native_credit_cpu = bool(args.native_credit_cpu)
        learner.stream_pace_min = float(getattr(args, 'stream_pace_min', 0.0) or 0.0)
        if getattr(args, 'ai', False):
            # a newborn's omitted switches take the same defaults a resume would (acts from birth, 2026-10-15)
            learner.ai_epistemic, learner.ai_pragmatic = float(args.ai_epistemic or 0.0), float(args.ai_pragmatic or 0.0)
            learner.ai_sigreg = float(args.ai_sigreg or 0.0)
            learner.eye_forced = 1.0 if args.eye_forced_open else args.eye_forced
            learner.eye_binary, learner.ai_depth = bool(args.eye_binary), int(args.ai_depth or 0)
            learner.sense_noise_max = float(args.sense_noise or 0.0)
            learner.act_habit = float(args.act_habit if args.act_habit is not None else 0.2)
            learner.act_gamma_inv = float(args.act_gamma_inv if args.act_gamma_inv is not None else 0.05)
            learner.epistemic_source = 'belief' if args.epistemic_belief else 'count'
            if args.motor_acts and (getattr(model, 'ai_mu', None) is None
                                    or (getattr(model, 'ai_precision', False) and getattr(model, 'ai_da_ref', None) is None)):
                from information_boltzmann.core.fly_active_inference import calibrate_muscles
                spread = calibrate_muscles(model, learner.state, np.asarray(train[:128]),
                                           ticks=int(round(args.stream_pace_mean or 8)), hold=bool(args.hold_input))
                print(f'Muscles calibrated on 128 words (median spread {spread:.4f})'
                      + (f'; dopamine tone {float(model.ai_da_ref):.4f} (gamma = 1)' if getattr(model, 'ai_precision', False) else ''),
                      flush=True)
            if args.miss_prior:
                # the listener's prior: unigram CE of every word from the first 20M training tokens
                _c = np.bincount(np.asarray(train[:20_000_000], dtype=np.int64), minlength=50257).astype(np.float64) + 0.5
                learner.unigram_ce = (-np.log(_c / _c.sum())).astype(np.float32)
                print(f'Miss value: the listener prior (unigram CE, mean over tokens '
                      f'{float((learner.unigram_ce[np.asarray(train[:200_000])]).mean()):.3f})', flush=True)
            if args.eye_binary:
                # a new stretch of the world every chunk (pace and noise schedules seeded by the lifetime so far)
                learner.pace_seed = int(getattr(args, 'odour_seed_value', 0) or 0)
                learner.eye_init = float(args.eye_init)
                model.eye_floor = float(args.eye_floor or 0.0)
            if args.noise_rate > 0:
                import numpy as _np
                alphabet = _np.random.default_rng(args.noise_seed).choice(50257, args.noise_alphabet, replace=False)
                learner.noise_world = {'rate': args.noise_rate, 'length': args.noise_length,
                                       'alphabet': alphabet.tolist(), 'seed': args.noise_seed,
                                       'offset': int(getattr(args, 'odour_seed_value', 0) or 0)}
            print(f'M3: binary eyes {learner.eye_binary}, question depth {learner.ai_depth}, noise rate {args.noise_rate}',
                  flush=True)
            print(f'Active inference: epistemic {args.ai_epistemic}, pragmatic {args.ai_pragmatic}, SIGReg {args.ai_sigreg}'
                  f'{", eyes forced " + str(learner.eye_forced) if learner.eye_forced is not None else ""}', flush=True)
        if args.evidence_plasticity:
            learner.attach_evidence_plasticity(args.pull_tau, saved=old, edges=not args.eg_plasticity)
            print(f'Evidence plasticity: {"" if args.eg_plasticity else "synapses + "}{len(learner.physio_names)} physiology tensors, significance-gated (|mean| > z se), '
                  f'pull to birth tau {args.pull_tau:g} updates (step {learner.evidence_count}); read surface = '
                  f'{"spikes (Gamma)" if args.read_spikes else "membranes"}', flush=True)
        if args.eg_plasticity:
            learner.attach_eg_plasticity(args.relative_edge_rate or 0.02, args.eg_tau, saved=old)
            print(f'EG plasticity: log-space rate {args.relative_edge_rate or 0.02} per update at the running RMS, '
                  f'KL to birth tau {args.eg_tau:g} updates, per-cell E/I totals kept (update {learner.eg_count})', flush=True)
        if getattr(args, 'relative_edge_rate', 0.0) and not args.eg_plasticity:
            step = learner.set_relative_edge_steps(args.graph, args.relative_edge_rate)
            print(f'Relative plasticity: {args.relative_edge_rate} of the birth weight per step '
                  f'(median synapse step {step:.2e})', flush=True)
        if args.ablate:
            # the causal gates: the same individual, still learning, with a part of the brain removed
            n = model.n_neurons
            if args.ablate == 'noedges':
                learner.attach_ablation(pre_cells=torch.ones(n, dtype=torch.bool, device='cuda'))
            elif args.ablate == 'blind':
                model.blind_input = True       # the words never reach the senses; every synapse and cell stays
            else:
                cells = torch.as_tensor(np.load(args.ablate), device='cuda')
                learner.attach_ablation(pre_cells=cells, read_cells=cells)
            print(f'Living ablation: {args.ablate}', flush=True)
        if getattr(args, 'odour_world', False):
            learner.odour_world = OdourWorld(model, args.graph, args.graph.parent / 'body-annotations.feather',
                                             seed=int(getattr(args, 'odour_seed_value', 0)),
                                             odour_drive=args.odour_drive, receptors=args.odour_receptors,
                                             outcome_dopamine=bool(args.outcome_dopamine),
                                             gusts=bool(getattr(args, 'sense_prediction', False)),
                                             levy_scale=float(args.levy_scale or 0.0), info_value=bool(args.info_value))
            learner.sense_lambda = float(getattr(args, 'sense_lambda', 1.0))
            learner.act_lambda = float(args.act_lambda)
            print(f'Odour world: {len(learner.odour_world.pool)} receptor types, {args.odour_receptors} per odour, '
                  f'outcome dopamine electrode {bool(args.outcome_dopamine)}, '
                  f'drive {args.odour_drive}, act lambda {args.act_lambda}', flush=True)
        learner.load_edge_signs(old)
        learner.load_adam_state(old['optimizer'], newly_trainable=newly_trainable)
        initial_stp = {name: getattr(model, name).detach().clone() for name in stp_names(model)}
        if fmt == 'fly-online-v2':
            learner.events = old['counters']['events']
            learner.previous_token = old['counters']['previous_token']
            learner.ema = old['counters']['ema']
            origin = {'checkpoint': str(args.resume), 'train_tokens': saved['train_cursor'],
                      'events': learner.events, 'old_optimizer_updates': old['counters']['updates'],
                      'migration': 'physical/weights/Adam moments retained; local eligibility and pending replay study archived; BPTT credit starts at this boundary'}
            trained = 0
            best = math.inf  # Different learner/budget: do not inherit a historical best headline.
        else:
            old_kind = old.get('plasticity_optimizer_kind', 'sgd')
            if old_kind == args.plasticity_optimizer:
                learner.sgd.load_state_dict(old['sgd'])
                for group in learner.sgd.param_groups:
                    for param in group['params']:
                        moments = learner.sgd.state.get(param)
                        if moments and any(isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape != param.shape
                                           for v in moments.values()):
                            learner.sgd.state.pop(param)      # a reshaped parameter (e.g. the writer just tied) starts afresh
                            print(f'Plasticity moments reset for a reshaped parameter {tuple(param.shape)}', flush=True)
            else:
                print(f'Plasticity optimizer migration: {old_kind} -> {args.plasticity_optimizer}; '
                      'new plasticity moments start at zero; existing Adam moments retained.', flush=True)
            for group, rate in zip(learner.sgd.param_groups, (args.lr_sensory, args.lr_synapse)):
                group['lr'] = rate
            for key in ('events', 'updates', 'previous_token', 'ema'):
                setattr(learner, key, old[key])
            origin = saved['origin']
            trained = saved['bptt_train_tokens']
            best = saved['best_live_nll']
            if old_kind != args.plasticity_optimizer:
                # A new optimizer arm has its own live-validation best; retain the
                # historical score as provenance rather than a new-arm headline.
                origin = {**origin, 'optimizer_migration_checkpoint': str(args.resume),
                          'optimizer_migration_train_cursor': saved['train_cursor'],
                          'prior_best_live_nll': best,
                          'plasticity_optimizer_migration': f'{old_kind}->{args.plasticity_optimizer}'}
                best = math.inf
        learner.physical_ticks = old.get('physical_ticks', learner.events)
        learner.load_interaction_state(old.get('interaction', {}))
        if quarantine_body_proxy_credit(learner):
            print('Synthetic-body score derivatives archived; real pending credit/world retained', flush=True)
        if quarantine_mb_pooled_credit(learner):
            print('Pooled MB score derivatives archived; real pending credit/world retained', flush=True)
        if args.ai and old.get('ai_g') is not None:
            from information_boltzmann.core.fly_active_inference import Familiarity
            learner.ai_g = old['ai_g'].to(learner.state.h.device)
            if old.get('familiarity_n') is not None:
                learner.familiarity = Familiarity(); learner.familiarity.n = old['familiarity_n'].copy()
        if args.dan_plastic_lr > 0 and old.get('dan_rule') != 'delayed-pulse-ema-terminal-coactivity-v1':
            mode_changed = True
            origin = {**origin, 'dan_rule_migration': {
                'from': old.get('dan_rule', 'legacy-terminal-membrane-or-disabled'),
                'to': 'delayed-pulse-ema-terminal-coactivity-v1',
                'new_gate_history': 'starts at zero; existing physical states and moments retained'}}
        if mode_changed:
            archive = args.output/'segments'/f'learning_mode_before_{trained}'
            if not archive.resolve().is_relative_to(args.output.resolve()):
                raise ValueError('Learning-mode archive must remain within the output directory')
            archive.mkdir(parents=True, exist_ok=True)
            atomic_json(archive/'config.json', previous_config)
            # A new output arm leaves its source assets untouched. Continuing in
            # place moves its best within the SAME checkpoint volume, without a
            # second GB-sized copy or an old-best/new-config ambiguity.
            if args.output.resolve() == Path(previous_config['output']).resolve():
                storage = Path(args.checkpoint_dir or previous_config.get('checkpoint_dir') or args.output)
                old_best = storage/'best.pt'
                if old_best.exists():
                    destination = storage/'segments'/f'learning_mode_before_{trained}'/'best.pt'
                    if not destination.resolve().is_relative_to(storage.resolve()):
                        raise ValueError('Best checkpoint archive must remain within its storage')
                    if destination.exists():
                        raise FileExistsError(f'Learning-mode archive already has {destination}')
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    old_best.replace(destination)
            origin = {**origin, 'learning_mode_migration': {
                'from_read_centering': previous_centering, 'to_read_centering': args.read_centering,
                'from_dan_plastic_lr': previous_dan_lr, 'to_dan_plastic_lr': args.dan_plastic_lr,
                'from_detach_reset': bool(previous_config.get('detach_reset', False)),
                'to_detach_reset': args.detach_reset,
                'from_surrogate_mode': previous_config.get('surrogate_mode', 'absolute'),
                'to_surrogate_mode': args.surrogate_mode,
                'from_transmission_mode': previous_config.get('transmission_mode', 'atomic'),
                'to_transmission_mode': args.transmission_mode,
                'train_cursor': saved['train_cursor'],
                'prior_best_live_nll': best if math.isfinite(best) else None}}
            best = math.inf
        if stp_changed:
            archive = args.output / 'segments' / f'stp_frozen_before_{trained}'
            archive.mkdir(parents=True, exist_ok=True)
            if not (archive/'config.json').exists():
                atomic_json(archive/'config.json', previous_config)
            old_best = args.output/'best.pt'
            if old_best.exists():
                destination = archive/'best.pt'
                if destination.exists():
                    raise FileExistsError(f'STP archive already has {destination}')
                old_best.replace(destination)
            origin = {**origin, 'stp_learning_migration': {
                'bptt_train_tokens': trained, 'train_cursor': saved['train_cursor'],
                'prior_best_live_nll': best if math.isfinite(best) else None,
                'newly_trainable': newly_trainable,
                'initial_parameters': {name: value.cpu().tolist() for name, value in initial_stp.items()},
                'state_and_optimizer': 'all physical values, weights, existing Adam moments and cursors retained; only new STP moments start at zero'}}
            best = math.inf
            print(f'STP joint learning activated: {newly_trainable}; existing Adam history retained.', flush=True)
        if args.settle_ticks != previous_settle_ticks:
            # Preserve the old phase's best asset without duplicating last.pt or
            # optimizer tensors. The next phase has a separate validation best.
            archive = args.output / 'segments' / f'timing_before_{trained}_settle{previous_settle_ticks}'
            root = args.output.resolve()
            if not archive.resolve().is_relative_to(root):
                raise ValueError('Timing archive must remain within the output directory')
            archive.mkdir(parents=True, exist_ok=True)
            if not (archive/'config.json').exists():
                atomic_json(archive/'config.json', previous_config)
            prior_progress = args.output/'progress.json'
            if prior_progress.exists() and not (archive/'progress.json').exists():
                atomic_json(archive/'progress.json', json.loads(prior_progress.read_text(encoding='utf-8')))
            old_best = args.output/'best.pt'
            if old_best.exists():
                destination = archive/'best.pt'
                if destination.exists():
                    raise FileExistsError(f'Timing archive already has {destination}')
                old_best.replace(destination)
            origin = {**origin, 'timing_migration': {
                'from_settle_ticks': previous_settle_ticks, 'to_settle_ticks': args.settle_ticks,
                'bptt_train_tokens': trained, 'train_cursor': saved['train_cursor'],
                'input_events': learner.events, 'physical_ticks': learner.physical_ticks,
                'prior_best_live_nll': best if math.isfinite(best) else None,
                'state_and_optimizer': 'all physical values, weights and Adam moments retained'}}
            best = math.inf
        if writer_clock_changed:
            archive = args.output / 'segments' / f'writer_clock_before_{trained}_{previous_writer_clock}'
            if not archive.resolve().is_relative_to(args.output.resolve()):
                raise ValueError('Writer clock archive must remain within the output directory')
            archive.mkdir(parents=True, exist_ok=True)
            for filename, payload in (('config.json', previous_config),
                    ('progress.json', json.loads((args.output/'progress.json').read_text(encoding='utf-8')))):
                if not (archive/filename).exists():
                    atomic_json(archive/filename, payload)
            old_best = args.output/'best.pt'
            if old_best.exists():
                destination = archive/'best.pt'
                if destination.exists():
                    raise FileExistsError(f'Writer clock archive already has {destination}')
                old_best.replace(destination)
            origin = {**origin, 'writer_clock_migration': {
                'from': previous_writer_clock, 'to': args.writer_baseline_clock,
                'bptt_train_tokens': trained, 'train_cursor': saved['train_cursor'],
                'input_events': learner.events, 'physical_ticks': learner.physical_ticks,
                'prior_best_live_nll': best if math.isfinite(best) else None,
                'state_and_optimizer': 'all physical values, weights, moments and cursors retained'}}
            best = math.inf
        learner.latent_window.copy_(old['latent_window'])
        cursor, val_cursor = saved['train_cursor'], saved['val_cursor']
        recent_a = saved.get('recent_a', [])
        recent_scores = saved.get('recent_scores', [])
    reference_train = np.load(args.reference_train, mmap_mode='r')
    reference_counts = np.ones(50257, dtype=np.float64)
    for left in range(0, len(reference_train), 1_000_000):
        reference_counts += np.bincount(
            reference_train[left:left+1_000_000].astype(np.int64), minlength=50257)
    reference_surprisal = -np.log(reference_counts/reference_counts.sum())
    reference_metadata = {
        'train': str(args.reference_train.resolve()), 'tokens': len(reference_train),
        'vocab_size': 50257, 'smoothing': 'add one', 'frozen': True,
        'validation_labels_used_for_prior': False,
        'counts_sha256': hashlib.sha256(reference_counts.tobytes()).hexdigest()}
    if is_scratch_start:
        first_pass_meter = FirstPassPredictionMeter(trained, cursor, reference_metadata)
        phase_meter = FirstPassPredictionMeter(trained, cursor, reference_metadata)
    else:
        if 'first_pass_meter' in saved:
            first_pass_meter = FirstPassPredictionMeter.from_state_dict(
                saved['first_pass_meter'], reference_metadata)
        else:
            # Historical sampled window logs cannot reconstruct complete totals.
            first_pass_meter = FirstPassPredictionMeter(trained, cursor, reference_metadata)
        first_pass_meter.verify_cursor(trained, cursor)
        if not writer_clock_changed and not stp_changed and not mode_changed and 'phase_first_pass_meter' in saved:
            phase_meter = FirstPassPredictionMeter.from_state_dict(
                saved['phase_first_pass_meter'], reference_metadata)
        else:
            phase_meter = FirstPassPredictionMeter(trained, cursor, reference_metadata)
        phase_meter.verify_cursor(trained, cursor)
        torch.set_rng_state(saved['rng_cpu'])
        torch.cuda.set_rng_state(saved['rng_cuda'])
        target = (trained + args.additional_tokens if fmt == 'fly-online-v2'
                  or args.extend_budget or trained >= saved['target_bptt_tokens'] else saved['target_bptt_tokens'])
        budget_start = trained if args.extend_budget else previous_config.get('budget_start_bptt_tokens', trained)
    del reference_train, reference_counts
    if args.data.resolve() != Path(previous_config['data']).resolve():
        old_train = np.load(Path(previous_config['data'])/'train.npy', mmap_mode='r')
        old_val = np.load(Path(previous_config['data'])/'validation.npy', mmap_mode='r')
        if len(train) < len(old_train) or len(val) < len(old_val):
            raise ValueError('Expanded stream is shorter than the continuing stream')
        for original, expanded in ((old_train, train), (old_val, val)):
            for left in range(0, len(original), 1048576):
                right = min(left+1048576, len(original))
                if not np.array_equal(original[left:right], expanded[left:right]):
                    raise ValueError('Expanded data changed the continuing stream prefix')
        del old_train, old_val
        print('Expanded train/validation prefixes verified; saved cursors retained.', flush=True)
    if cursor + (target - trained) + 1 > len(train):
        raise ValueError('Fresh OWT stream exhausted')
    evaluations_needed = target//args.validate_every_tokens - trained//args.validate_every_tokens
    if target % args.validate_every_tokens:
        evaluations_needed += 1
    if val_cursor + evaluations_needed*args.eval_tokens > len(val):
        raise ValueError('Fresh validation stream is insufficient for the full registered budget')
    # after every restore, migration and plasticity attachment: nothing later re-enables brain gradients
    learner.set_brain_learning(getattr(args, 'brain_learning', None) or 'bptt')
    if learner.brain_learning == 'local':
        from information_boltzmann.core.fly_native_runtime import compact_index_buffers
        compacted, freed = compact_index_buffers(model)
        torch.cuda.empty_cache()
        print(f'Local brain learning: readouts only; {compacted} index buffers stored as int32 '
              f'({freed / 2**20:.0f} MiB freed)', flush=True)
    if getattr(args, 'circuit_observer', None) is not None:
        from information_boltzmann.core.fly_circuit_observer import attach_circuit_observer
        attach_circuit_observer(learner, args.graph, Path('data/malecns_v1/body-annotations.feather'),
                                args.circuit_observer)
        print(f'Circuit observer: live-pass activity per word -> {args.circuit_observer}', flush=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(d_model=previous_config['d_model'], decoder_bias=previous_config['decoder_bias'],
        amine_kinetics_version=AMINE_KINETICS_VERSION if args.amine_modulation else None,
        mb_terminal_version=MB_TERMINAL_VERSION if args.mb_terminal_projection else None,
        apl_cable_version=APL_CABLE_VERSION if args.apl_cable else None,
        apl_local_release=bool(args.apl_local_release),
        apl_local_release_version=APL_LOCAL_RELEASE_VERSION if args.apl_local_release else None,
        kc_terminal_version=KC_TERMINAL_VERSION if args.kc_terminal else None,
        kc_site_stp_version=KC_SITE_STP_VERSION if args.kc_site_stp else None,
        kc_contact_version=KC_CONTACT_VERSION if args.kc_contact_memory else None,
        kc_lateral_version=KC_LATERAL_VERSION if args.kc_lateral else None,
        sensory_terminal_version=SENSORY_TERMINAL_VERSION if args.sensory_terminal else None,
        kc_dopamine_version=KC_DOPAMINE_VERSION if args.kc_dopamine else None,
        dan_release_version=DAN_RELEASE_VERSION if args.dan_local_release else None,
        dan_carried_credit=bool(args.dan_carried_credit),
        dan_credit_version=1 if args.dan_carried_credit else None,
        dan_receiver_credit=bool(args.dan_receiver_credit),
        dan_receiver_credit_version=1 if args.dan_receiver_credit else None,
        lr=args.lr, lr_synapse=args.lr_synapse, lr_sensory=args.lr_sensory,
        plasticity_optimizer=args.plasticity_optimizer,
        n_neurons=model.n_neurons, n_synapses=sum(p.numel() for p in learner.edges),
        trainable_parameters=sum(p.numel() for p in learner.trainable),
        credit=f'{args.window}-token whole-network ATan-surrogate BPTT; all within-event physical ticks included',
        physical_ticks_per_input=1+args.settle_ticks,
        event_order='one sensory input pulse, quiet propagation, output read, next target revelation',
        physical_tick_unit='unchanged graph-delay tick; no physiological milliseconds inferred from OWT',
        writer_baseline=('updates only at observed input events; held during quiet propagation'
                         if args.writer_baseline_clock == 'input'
                         else 'archived pulse update and zero-source decay at quiet ticks'),
        fixed_parameters=[name for name, parameter in model.named_parameters() if not parameter.requires_grad],
        stp_learning={'enabled': learner.learn_stp, 'parameter_names': list(STP_PARAMETER_NAMES),
                      'trainable_count': sum(getattr(model, name).numel() for name in STP_PARAMETER_NAMES)
                                         if learner.learn_stp else 0,
                      'weight_decay': 0.0, 'lr': args.lr},
        physical_state='h, delayed STP pulses, ge, gi, ALIF b, STP x/u, sensory adaptation; never reset',
        evaluation='same active learner; pre-update window scores; fresh B and A revisit with actual bridges',
        optimizer_cadence=f'one joint update per {args.window} observed input events, including active evaluation',
        origin=origin, pid=os.getpid(), budget_start_bptt_tokens=budget_start,
        fresh_training_budget=target-budget_start,
        budget_start_train_cursor=cursor if args.extend_budget else previous_config.get('budget_start_train_cursor', cursor),
        training_data_policy='monotone fresh train cursor, no wrap/shuffle/epoch replay; evaluation replay excluded from training budget')
    config['first_pass_measurement'] = {
        'reference': reference_metadata,
        'start_bptt_train_tokens': first_pass_meter.start_train_targets,
        'scope': 'every fresh training target scored before its update, including actual stream bridges; active B and A replay excluded',
        'recent_optimizer_windows': first_pass_meter.recent.maxlen,
        'ema_scope': 'historical EMA mixes training, fresh B and A replay traffic'}
    manifest = args.data / 'manifest.json'
    config['dataset_manifest'] = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else None
    config['train_shape'], config['validation_shape'] = list(train.shape), list(val.shape)
    config['tokenizer'] = 'GPT-2, 50257 vocabulary entries'
    # the conversation's own saved state outlives the big checkpoint dict (freed here)
    resume_mother = ({'mother_world': saved.get('mother_world'), 'mother_rng': saved.get('mother_rng')}
                     if not is_scratch_start else {})
    if not is_scratch_start:
        del weights, old, saved
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(args.output / 'config.json', config)
    if is_scratch_start:
        atomic_save(args.output / 'birth.pt', {
            'format': 'fly-bptt-v1', 'model': checkpoint_model_weights(model),
            'learner': learner.state_dict(), 'config': config, 'origin': origin,
            'train_cursor': 0, 'val_cursor': 0, 'bptt_train_tokens': 0,
            'target_bptt_tokens': target, 'best_live_nll': math.inf,
            'recent_a': [], 'recent_scores': [],
            'first_pass_meter': first_pass_meter.state_dict(),
            'phase_first_pass_meter': phase_meter.state_dict(),
            'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state()})
    (args.output / 'process.pid').write_text(str(os.getpid()))
    learner.record_word_da = bool(args.log_word_dopamine)       # birth and resume alike
    learner.stream_jepa_weight = float(0.5 if args.stream_jepa_weight is None else args.stream_jepa_weight)
    if not args.no_graph and not args.use_latent_predictor and not args.use_graph_observer:
        print('Capturing 32-event forward/backward; no physical reset...', flush=True)
        learner.runner = FlyBPTTGraph(learner, args.window)
    else:
        print('Running eager 32-event forward/backward; no physical reset...', flush=True)
    torch.cuda.synchronize()
    # Keep the initialization/warm-up peak in the reported budget as well.
    setup_peak_mib = torch.cuda.max_memory_allocated() / 2**20
    if max(setup_peak_mib, torch.cuda.memory_reserved()/2**20) > args.vram_limit_mib:
        raise MemoryError('BPTT setup exceeded the registered dedicated CUDA budget')
    start = time.perf_counter()
    start_trained = trained
    next_eval = ((trained // args.validate_every_tokens) + 1) * args.validate_every_tokens
    last_grad = {}

    def log(row, file='metrics.jsonl'):
        with (args.output / file).open('a', encoding='utf-8') as out:
            out.write(json.dumps(row, allow_nan=False) + '\n')

    def health():
        row = {'field_energy': float(learner.state.h.square().mean()),
            **instantaneous_release_metrics(learner.state.ring[0], getattr(model, 'graded_mask', None)),
            'read_norm_mean': float(model.read_norm.weight.detach().mean()),
            'vram_allocated_mib': torch.cuda.memory_allocated() / 2**20,
            'vram_reserved_mib': torch.cuda.memory_reserved() / 2**20,
            'vram_peak_mib': torch.cuda.max_memory_allocated() / 2**20,
            'setup_peak_mib': setup_peak_mib,
            'events': learner.events, 'physical_ticks': learner.physical_ticks,
            'physical_ticks_per_input': learner.stream_ticks or 1+learner.settle_ticks,
            'settle_ticks': learner.settle_ticks,
            'stream_ticks': learner.stream_ticks, 'hold_input': learner.hold_input, 'read_lag': learner.read_lag,
            'go_neuron': learner.go_neuron, 'world_windows': dict(world_counts),
            'writer_baseline_clock': learner.writer_baseline_clock,
            'bptt_optimizer_updates': learner.updates,
            'stp_learning_enabled': learner.learn_stp,
            'use_checkpointing': getattr(learner, 'use_checkpointing', False),
            'sensory_resource_mean': float(learner.state.x[:, model.topographic_writer.injection_index].mean())}
        if getattr(model, 'use_read_gamma_trace', False) and getattr(model, 'logit_read_gamma', None) is not None:
            decay = model.get_read_gamma_decay().detach()
            row['read_gamma_decay_mean'] = float(decay.mean())
            row['read_gamma_decay_min'], row['read_gamma_decay_max'] = float(decay.min()), float(decay.max())
        if getattr(model, 'latent_predictor', None) is not None:
            row['predictor_mix_alpha'] = float(model.latent_predictor.mix_alpha.item())
        if getattr(model, 'graph_observer', None) is not None:
            row['obs_gamma'] = float(model.graph_observer.gamma.item())
            row['obs_raw_gamma'] = float(model.graph_observer.raw_gamma.item())
            if learner.last_jepa_metrics:
                for k, v in learner.last_jepa_metrics.items():
                    row[f'obs_metric_{k}'] = float(v)
        # the brain's own signals (dopamine, intrinsic drive, graded cells...), logged whatever the go neuron
        if getattr(model, 'kcd_pre', None) is not None:
            active_types = torch.unique(model.superclass_id[model.kcd_post.long()])
            gain = model.log_kc_release.detach()[active_types].exp()
            row.update(kc_dan_release_enabled=True, kc_dan_release_active_types=int(active_types.numel()),
                       kc_dan_release_gain_mean=float(gain.mean()),
                       kc_dan_release_gain_min=float(gain.min()), kc_dan_release_gain_max=float(gain.max()))
        if getattr(model, 'dopamine_sensitivity', None) is not None and learner.state.dopamine.numel():
            dop = learner.state.dopamine.detach()
            sens = model.dopamine_sensitivity.detach()
            row.update(dopamine_sensitivity_absmax=float(sens.abs().max()), dopamine_sensitivity_mean=float(sens.mean()),
                       dopamine_concentration=float(dop[0].mean()),
                       dopamine_phasic_sd=float(((dop[0] - dop[1]) / dop[1][dop[1] > 0].mean().clamp_min(1e-12))[dop[1] > 0].std()),
                       dopamine_tau_fast=float(-1 / torch.log(torch.sigmoid(model.dopamine_logit_fast.detach()))),
                       dopamine_tau_slow=float(-1 / torch.log(torch.sigmoid(model.dopamine_logit_slow.detach()))),
                       dopamine_cells_firing=float((learner.state.ring[0][0, model.dan_indices] > 0).float().mean()))
        if getattr(model, 'online_da_eta', None) is not None and learner.state.da_memory.numel():
            phi = learner.state.da_memory[0][model.online_da_target[0] > 0].detach()
            row.update(online_da_eta_mean=float(model.online_da_eta.detach().mean()),
                       online_da_tau_trace=float(-1 / torch.log(torch.sigmoid(model.online_da_logit_trace.detach()))),
                       online_da_memory_sd=float(phi.std()), online_da_memory_absmax=float(phi.abs().max()))
        if getattr(model, 'amine_enabled', False) and learner.state.amine_concentration.numel():
            ac = learner.state.amine_concentration.detach()
            ar = learner.state.amine_receptor.detach()
            row.update(amine_concentration_oa=float(ac[0].mean()),
                       amine_concentration_5ht=float(ac[1].mean()),
                       amine_occupancy_oa=float(ar[0].mean()),
                       amine_occupancy_5ht=float(ar[1].mean()),
                       amine_response_oa_mean=float(model.amine_response_oa.detach().mean()),
                       amine_response_5ht_mean=float(model.amine_response_5ht.detach().mean()))
        if getattr(model, 'cx_enabled', False) and learner.state.cx_trace.numel():
            row.update(cx_trace_mean=float(learner.state.cx_trace.detach().mean()),
                       cx_trace_sd=float(learner.state.cx_trace.detach().std()),
                       cx_feedback_mean=float(model.cx_feedback.detach().mean()),
                       cx_tau_mean=float((-1.0 / torch.log(torch.sigmoid(model.cx_logit_decay.detach()))).mean()))
        row['mb_pooled_quarantined'] = bool(getattr(model, 'mb_pooled_quarantined', False))
        if (getattr(model, 'mb_consolidation_enabled', False) and learner.state.dpm_trace.numel()
                and not getattr(model, 'mb_pooled_quarantined', False)):
            row.update(dpm_trace=float(learner.state.dpm_trace.detach().mean()),
                       apl_trace=float(learner.state.apl_trace.detach().mean()),
                       dpm_consolidation=float(model.mb_dpm_consolidation.detach()),
                       apl_feedback=float(model.mb_apl_feedback.detach()),
                       dpm_tau=float(-1.0 / torch.log(torch.sigmoid(model.mb_dpm_logit_decay.detach()))),
                       apl_tau=float(-1.0 / torch.log(torch.sigmoid(model.mb_apl_logit_decay.detach()))))
        if getattr(model, 'mb_terminal_enabled', False) and learner.state.mb_terminal_gaba.numel():
            row.update(mb_terminal_gaba_by_source=learner.state.mb_terminal_gaba.detach().mean(1).cpu().tolist(),
                       mb_dpm_concentration_by_source=learner.state.mb_dpm_concentration.detach().mean(1).cpu().tolist(),
                       mb_dpm_receptor_max=float(learner.state.mb_dpm_receptor.detach().max()),
                       mb_terminal_source_body_ids=model.mb_terminal_source_body_ids.cpu().tolist())
        if learner.state.apl_voltage.numel():
            row.update(apl_cable_nodes=learner.state.apl_voltage.numel(),
                       apl_cable_voltage_min=float(learner.state.apl_voltage.min()),
                       apl_cable_voltage_max=float(learner.state.apl_voltage.max()),
                       apl_cable_voltage_sd=float(learner.state.apl_voltage.std()),
                       apl_log_axial=model.apl_log_axial.detach().cpu().tolist())
        if learner.state.apl_release_calcium.numel():
            row.update(apl_release_calcium_max=float(learner.state.apl_release_calcium.detach().max()),
                       apl_release_resources_min=float(learner.state.apl_release_resources.detach().min()),
                       apl_release_scope='local normalized calcium and conserving release resources; channel identity unmeasured')
        if learner.state.kc_lateral_receptor.numel():
            row.update(kc_lateral_receptor_max=float(learner.state.kc_lateral_receptor.detach().max()),
                       kc_lateral_signal_max=float(learner.state.kc_lateral_signal.detach().max()),
                       kc_lateral_scope='actual delayed KC axonal contacts, local inhibitory receptor/effector, release and cAMP coupling')
        if learner.state.kc_terminal_gaba.numel():
            row.update(kc_terminal_nodes=learner.state.kc_terminal_gaba.shape[1],
                       kc_terminal_release_sites=learner.state.kc_terminal_packets.shape[1],
                       kc_terminal_gaba_max_by_source=learner.state.kc_terminal_gaba.detach().amax(1).cpu().tolist(),
                       kc_terminal_5ht_max_by_source=learner.state.kc_terminal_5ht.detach().amax(1).cpu().tolist(),
                       kc_terminal_response_max=float(learner.state.kc_terminal_receptor.detach().max()),
                       kc_terminal_eligibility_max=float(learner.state.kc_terminal_eligibility.detach().max()),
                       kc_terminal_state_scope='local effective receptor signals at actual morphology volumes',
                       mb_terminal_state_scope='historical pooled snapshot retained; current local state uses kc_terminal_*')
        if learner.state.dan_local_calcium.numel():
            row.update(dan_local_calcium_max=float(learner.state.dan_local_calcium.detach().max()),
                dan_local_resources_min=float(learner.state.dan_local_resources.detach().min()),
                dan_local_source_scope='actual reciprocal contacts and DAN release sites; effective receptor/calcium kinetics, existing somatic AP drive')
        if learner.state.kc_dopamine_signal.numel():
            row.update(kc_dopamine_signal_max=float(learner.state.kc_dopamine_signal[0].detach().max()),
                kc_dopamine_scope='actual DAN-KC contacts, local effective receptor signals on all KC morphology nodes')
        if learner.state.kc_contact_memory.numel():
            row.update(kc_plastic_contacts=learner.state.kc_contact_memory.shape[1],
                       kc_contact_phi_min=float(learner.state.kc_contact_memory.detach().min()),
                       kc_contact_phi_max=float(learner.state.kc_contact_memory.detach().max()),
                       phi_scope='native release-position/MBON partner; legacy comp_memory historical',
                       dopamine_teacher_scope=('actual DAN-KC contact-local effective receptor signals' if args.kc_dopamine
                           else 'existing MBON compartment delivery, not spatial bouton dopamine'))
        if learner.state.kc_site_resources.numel():
            row.update(kc_site_resources_min=float(learner.state.kc_site_resources.detach().min()),
                       kc_site_resources_mean=float(learner.state.kc_site_resources.detach().mean()),
                       kc_site_utilization_max=float(learner.state.kc_site_utilization.detach().max()))
        if getattr(model, 'body_loop_enabled', False) and learner.state.body_action_trace.numel():
            row.update(body_action_trace=float(learner.state.body_action_trace.detach().mean()),
                       body_feedback_trace=float(learner.state.body_feedback_trace.detach().mean()),
                       body_feedback_mean=float(model.body_feedback.detach().mean()),
                       body_tau=float(-1.0 / torch.log(torch.sigmoid(model.body_logit_decay.detach()))),
                       body_descending_count=int(model.body_descending_index.numel()),
                       body_ascending_count=int(model.body_ascending_index.numel()))
        if getattr(model, 'body_actuator_enabled', False) and learner.state.body_muscle.numel():
            row.update(body_muscle=float(learner.state.body_muscle.detach().mean()),
                       body_reafference_absmax=float(learner.state.body_reafference.detach().abs().max()),
                       body_proprioceptive_count=int(model.body_proprioceptive_index.numel()),
                       body_proprioceptive_gain_norm=float(model.body_proprioceptive_gain.detach().norm()),
                       body_return_mode=('historical_scalar_proxy' if getattr(model, 'body_proxy_compatibility', False)
                                         else 'proxy_quarantined_native_periphery_unconnected'))
        if getattr(model, 'native_motor_port_enabled', False):
            row.update(native_motor_cells=int(model.native_motor_index.numel()),
                       native_motor_execution='actual_release_packet',
                       action_decoder='separate_from_prediction',
                       native_motor_terminal_enabled=bool(model.native_motor_terminal_enabled))
        if getattr(model, 'endocrine_enabled', False) and learner.state.endocrine_lk.numel():
            row.update(endocrine_lk=float(learner.state.endocrine_lk.detach().mean()),
                       endocrine_receptor=float(learner.state.endocrine_receptor.detach().mean()),
                       endocrine_calcium=float(learner.state.endocrine_calcium.detach().mean()),
                       endocrine_response_absmax=float(model.endocrine_response.detach().abs().max()),
                       endocrine_target_count=int(model.endocrine_target_index.numel()))
        if getattr(model, 'olfactory_relay_enabled', False) and learner.state.al_local_trace.numel():
            row.update(al_local_trace=float(learner.state.al_local_trace.detach().mean()),
                       al_pn_drive=float(learner.state.al_pn_drive.detach().mean()),
                       al_pn_feedback_mean=float(model.al_pn_feedback.detach().mean()),
                       al_local_tau=float(-1.0 / torch.log(torch.sigmoid(model.al_local_logit_decay.detach()))),
                       al_local_count=int(model.al_local_index.numel()),
                       al_pn_count=int(model.al_pn_index.numel()),
                       al_inhibitory_contacts=int(model.al_contact_weight.numel()))
        if getattr(model, 'writer_prediction_logit', None) is not None:
            row['writer_prediction_beta'] = float(torch.sigmoid(model.writer_prediction_logit.detach()))
        if getattr(model, 'precision_u', None) is not None and learner.state.modulator.numel():
            m = learner.state.modulator.detach()
            gain = 2 * torch.sigmoid(m @ model.precision_u.detach().T + model.precision_b.detach())[0]
            row.update(attention_gain_visual=float(gain[0]), attention_gain_chemo=float(gain[1]),
                       attention_gain_mechano=float(gain[2]), modulator_mean=float(m.mean()),
                       dopamine_level=float(m[0, model.dopamine_mask].mean()),
                       modulator_tau=float(-1 / torch.log(torch.sigmoid(model.modulator_logit_decay.detach()))))
        if getattr(model, 'intrinsic_kappa', None) is not None:
            kap = model.intrinsic_kappa.detach()
            spikes = (learner.state.ring[0][0, model.intrinsic_index] > 0).float()
            row.update(intrinsic_kappa_mean=float(kap.mean()), intrinsic_kappa_min=float(kap.min()),
                       intrinsic_kappa_max=float(kap.max()), intrinsic_firing=float(spikes.mean()))
        if getattr(model, 'read_bank_gain', None) is not None:
            gain = model.read_bank_gain.detach().abs().mean(1)
            row['read_bank_gain_now'] = float(gain[0])
            row['read_bank_gain_filters'] = float(gain[1:].mean())
            row['read_bank_tau_mean'] = float((-1 / torch.log(model.read_bank_decay().detach())).mean())
        if getattr(model, 'log_graded_gain', None) is not None:
            row['graded_gain'] = [round(float(x), 5) for x in model.log_graded_gain.detach().exp()]
        if getattr(model, 'log_visual_graded_gain', None) is not None:
            visual_gain = model.log_visual_graded_gain.detach().exp()
            row.update(visual_graded_count=int(model.visual_graded_index.numel()),
                       visual_graded_gain_mean=float(visual_gain.mean()),
                       visual_graded_gain_min=float(visual_gain.min()),
                       visual_graded_gain_max=float(visual_gain.max()),
                       visual_release_mean=float(learner.state.ring[0][:, model.visual_graded_index].detach().mean()))
        if getattr(learner, 'go_neuron', False):
            for k_go, v_go in getattr(learner, 'last_adaptive_metrics', {}).items():
                row[k_go] = float(v_go) if isinstance(v_go, (int, float)) else v_go
            if learner.go_neuron == 'virtual':
                row['go_leak'] = float(torch.sigmoid(model.go_logit_leak.detach()))
                row['go_drive'] = float(model.go_drive.detach())
                row['go_reset'] = float(model.go_reset.detach())
                row['go_read_norm'] = float(model.go_read.weight.detach().norm())
        if getattr(learner, 'adaptive_admission', False):
            row['adaptive_admission'] = True
            for k_ad, v_ad in getattr(learner, 'last_adaptive_metrics', {}).items():
                row[k_ad] = float(v_ad)
        if getattr(learner, 'use_ctm_loss', False):
            row['use_ctm_loss'] = True
            for k_ctm, v_ctm in getattr(learner, 'last_ctm_metrics', {}).items():
                row[k_ctm] = float(v_ctm)
        if getattr(learner, 'lambda_mcr2', 0.0) > 0.0:
            row['lambda_mcr2'] = learner.lambda_mcr2
            for k_mcr, v_mcr in getattr(learner, 'last_mcr2_metrics', {}).items():
                row[k_mcr] = float(v_mcr)
        for name in stp_names(model):
            parameter = getattr(model, name).detach()
            row[f'{name}_max_change_since_launch'] = float((parameter-initial_stp[name]).abs().max())
        if model.use_stp:
            row['stp_u0_mean'] = float(model.logit_u0.detach().sigmoid().clamp(.05, .95).mean())
            row['stp_tau_fac_mean'] = float(model.log_tau_fac.detach().exp().clamp(5, 1000).mean())
            row['stp_tau_rec_mean'] = float(model.log_tau_rec.detach().exp().clamp(5, 2000).mean())
        return row

    def prediction_accounting():
        # Keep complete historical totals and a distinct current implementation
        # phase, with no retrospective estimates and bounded memory.
        return {**first_pass_meter.summary(),
                **{f'phase_{key}': value for key, value in phase_meter.summary().items()}}

    def progress(status):
        row = {'status': status, 'bptt_train_tokens': trained, 'tokens_streamed': cursor,
            'target_bptt_tokens': target, 'ema_stream_loss': learner.ema,
            'fresh_budget_tokens_processed': trained-budget_start,
            'fresh_training_budget': target-budget_start,
            'lr': args.lr, 'lr_decoder': args.lr if args.lr_decoder is None else args.lr_decoder,
            'lr_synapse': args.lr_synapse, 'lr_sensory': args.lr_sensory,
            'best_live_nll': best if math.isfinite(best) else None,
            'speed_tokens_per_sec': (trained-start_trained) / max(time.perf_counter()-start, 1e-6),
            **health(), **last_grad, **prediction_accounting()}
        atomic_json(args.output / 'progress.json', row)
        return row

    def checkpoint(is_best=False):
        payload = {'format': 'fly-bptt-v1', 'model': checkpoint_model_weights(model),
            'learner': learner.state_dict(), 'config': config, 'origin': origin,
            'train_cursor': cursor, 'val_cursor': val_cursor, 'bptt_train_tokens': trained,
            'target_bptt_tokens': target, 'best_live_nll': best,
            'recent_a': recent_a, 'recent_scores': recent_scores,
            'first_pass_meter': first_pass_meter.state_dict(),
            'phase_first_pass_meter': phase_meter.state_dict(),
            'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state(),
            'mother_world': mother_world.state_dict() if mother_world is not None else None,
            'mother_rng': mother_rng.bit_generator.state if mother_rng is not None else None}
        atomic_save(args.output / 'last.pt', payload)
        if is_best:
            atomic_save(args.output / 'best.pt', {'format': 'fly-bptt-best-weights-v1',
                'model': checkpoint_model_weights(model), 'config': config,
                'bptt_train_tokens': trained, 'best_live_nll': best})

    world_rng = np.random.default_rng(1234)
    world_counts = {'read': 0, 'listen': 0, 'gap': 0}

    def sample_world(n):
        # The world, not the model: its pace and its silences.  Default fractions 1/2 read, 1/4 listen, 1/4 gaps.
        mix = np.asarray(args.world_mix, dtype=np.float64) / np.sum(args.world_mix)
        u = world_rng.random()
        if u < mix[0]:
            return {'mode': 'read'}
        if u < mix[0] + mix[1]:
            return {'mode': 'listen', 'k': int(world_rng.integers(3, 9))}
        lo, hi = args.gap_range
        gaps = np.where(world_rng.random(n) < args.gap_prob, world_rng.integers(lo, hi + 1, n), 0).tolist()
        return {'mode': 'gap', 'gaps': gaps}

    for _ in range(int(getattr(args, 'world_skip', 0) or 0)):
        sample_world(args.window)

    pending_resource_error = None

    def observe(values, training=False):
        nonlocal last_grad, pending_resource_error
        if training and learner.brain_learning == 'local' and getattr(learner, 'live_result', None) is not None:
            # the live pass was the life: commit it (no relived window, no tape)
            chunk, last_grad = learner.live_commit(values, learner.forced_world)
            learner.forced_world = None
            return chunk
        scores = []
        for left in range(0, len(values), args.window):
            world = sample_world(min(args.window, len(values) - left)) if (training and args.mixed_world) else None
            learner.world = world
            if world is not None:
                world_counts[world['mode']] += 1
            chunk, last_grad = learner.observe(np.array(values[left:left+args.window], dtype=np.int64))
            learner.world = None
            scores.extend(chunk)
            if torch.cuda.memory_reserved() / 2**20 > 0.9 * args.vram_limit_mib:
                # cached blocks from a long window (fragmentation), not memory in use: give them back between windows
                torch.cuda.empty_cache()
            if max(torch.cuda.memory_reserved(), torch.cuda.max_memory_allocated()) / 2**20 > args.vram_limit_mib:
                error = f'CUDA allocation/reservation exceeded the registered {args.vram_limit_mib:.0f} MiB budget'
                if training:
                    # observe() already updated weights and the world has moved.
                    # Complete counters/meters/logs, then save the paired state.
                    pending_resource_error = error
                else:
                    raise MemoryError(error)
        return scores

    taste = None
    if args.taste_every:
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
        nouns = ('apple bread river stone cloud horse candle garden window mountain flower tree bird fish table chair door '
                 'house water fire light paper music island forest ocean train ship coin glass shirt shoe hat road bridge '
                 'moon star sun rain snow wind sand rock grass leaf milk cheese honey sugar salt egg wolf bear lion tiger '
                 'mouse snake king queen ring sword wheel tower').split()
        taste = {'pool': [enc.encode(' ' + w)[0] for w in nouns if len(enc.encode(' ' + w)) == 1],
                 'rng': np.random.default_rng(int(args.seed) + 7919 if hasattr(args, 'seed') else 7919), 'windows': 0,
                 'log': open(args.output / 'taste.jsonl', 'a')}
        print(f'Taste world: one episode after every {args.taste_every} text windows, {len(taste["pool"])} cues', flush=True)

    def taste_episode():
        # 256 ticks: cue A 32 | distractors 32 | cue B 32 | distractors 32 (last 32: baseline) | test X1 32 | '?' 32 |
        # test X2 32 | '?' 32; one tick per token here (a held cue = the same word every tick).  Sweet cue: PAM cells
        # driven, bitter: PPL1, during ticks 16-40 of the cue's block; read windows 128-192 and 192-256.
        rng, tpt = taste['rng'], 1
        a, b = (int(c) for c in rng.choice(taste['pool'], 2, replace=False))
        sweet_first = bool(rng.integers(2)); order = int(rng.integers(2))
        at = int(rng.integers(0, len(train) - 128))
        d1, d2 = [int(x) for x in train[at:at + 32]], [int(x) for x in train[at + 32:at + 64]]
        x1, x2 = (a, b) if order == 0 else (b, a)
        tokens = [a] * 32 + d1 + [b] * 32 + d2 + [x1] * 32 + [30] * 32 + [x2] * 32 + [30] * 32
        with torch.no_grad():
            rates0 = model.get_decay_rates()[0].expand(1, model.n_neurons)
            rheobase = model.get_thresholds().expand(1, model.n_neurons) * -torch.log(rates0.clamp(1e-5, 1 - 1e-7))
            sweet = args.taste_drive * rheobase * model.taste_pam.float().unsqueeze(0)
            bitter = args.taste_drive * rheobase * model.taste_ppl1.float().unsqueeze(0)
        va, vb = (sweet, bitter) if sweet_first else (bitter, sweet)
        drives = {t: va for t in range(16, 41)}
        drives.update({64 + t: vb for t in range(16, 41)})
        sweet_cue = a if sweet_first else b
        targets = [int(x1 == sweet_cue), int(x2 == sweet_cue)]
        metrics = learner.observe_taste(tokens, drives, held=(a, b), targets=targets, read_windows=((128, 192), (192, 256)),
                                        baseline_window=(96, 128), token_ticks=tpt)
        row = {'bptt_train_tokens': trained, 'taste_loss': metrics['taste_loss'], 'mb_correct': metrics['taste_mb_correct'],
               'text_correct': metrics['taste_text_correct'], 'tests': metrics['taste_tests'],
               'comp_eta': float(model.comp_eta.detach()), 'taste_gain': float(model.taste_gain.detach()),
               'taste_bias': float(model.taste_bias.detach())}
        taste['log'].write(json.dumps(row) + '\n'); taste['log'].flush()
        if torch.cuda.memory_reserved() / 2**20 > 0.9 * args.vram_limit_mib:
            torch.cuda.empty_cache()
        return row

    mother_world = mother_rng = None
    if args.mother_world:
        # the user's mother world (docs/information_boltzmann/FLY_MOTHER_WORLD_DESIGN_20261014.md): the fly's speech goes to
        # a mother, her replies come back through the senses; a new conversation each run (each chunk: a new day)
        import tiktoken
        from information_boltzmann.core.fly_mother_world import (MotherWorld, LocalMother, ApiMother, GeminiApiMother,
                                                                 DeepSeekMother, SiliconFlowMother, FallbackMother, MailboxMother, live_window)
        # the raising plan (dashboard / the mother agent: data/harness/raising.json) sets the mother's side
        raising_path = Path('data/harness/raising.json')
        raising_mtime = 0.0
        raising_full = {}

        def refresh_raising_plan():
            nonlocal raising_mtime, raising_full
            if not raising_path.exists():
                return
            try:
                mt = raising_path.stat().st_mtime
                if mt != raising_mtime:
                    raising_full = json.loads(raising_path.read_text(encoding='utf-8'))
                    raising_mtime = mt
                    m_cfg = raising_full.get('mother', {})
                    args.mother_max_fly_words = int(m_cfg.get('max_fly_words', args.mother_max_fly_words))
                    args.mother_temperature = float(m_cfg.get('fly_temperature', args.mother_temperature))
                    if mother_world is not None:
                        mother_world.max_fly_words = args.mother_max_fly_words
                        mother_world.history_turns = int(m_cfg.get('history_turns', mother_world.history_turns))
                        mother_world.ticks_per_beat = int(m_cfg.get('ticks_per_beat', getattr(mother_world, 'ticks_per_beat', 8)))
                        mother_world.pause_beats_before_reply = int(m_cfg.get('pause_beats_before_reply', getattr(mother_world, 'pause_beats_before_reply', 1)))
                        mother_world.silent_beats_threshold = int(m_cfg.get('silent_beats_threshold', getattr(mother_world, 'silent_beats_threshold', 3)))
                        mother_world.call_prompt = m_cfg.get('call_prompt', getattr(mother_world, 'call_prompt', None))
                        mother_world.contingent = bool(m_cfg.get('contingent', getattr(mother_world, 'contingent', True)))
                    if hasattr(mother, 'update_plan'):
                        mother.update_plan(raising_full)
            except Exception as error:      # half written or invalid: keep the old plan, try again next window
                print(f'raising.json not applied ({type(error).__name__}): the previous plan stays', flush=True)

        if raising_path.exists():
            try:
                raising_full = json.loads(raising_path.read_text(encoding='utf-8'))
                raising_mtime = raising_path.stat().st_mtime
            except json.JSONDecodeError:
                print('raising.json is not valid JSON: ignored', flush=True)
        raising = raising_full.get('mother', {})
        args.mother_max_fly_words = int(raising.get('max_fly_words', args.mother_max_fly_words))
        args.mother_temperature = float(raising.get('fly_temperature', args.mother_temperature))
        ticks_per_beat = int(raising.get('ticks_per_beat', 8))
        pause_beats = int(raising.get('pause_beats_before_reply', 1))
        silent_thresh = int(raising.get('silent_beats_threshold', 3))
        call_prompt = raising.get('call_prompt')
        mother = (LocalMother() if args.mother_world == 'local' else
                  GeminiApiMother(args.mother_api_model or raising.get('model') or 'gemini-3.5-flash')
                  if args.mother_world == 'gemini' else
                  DeepSeekMother(args.mother_api_model or (raising.get('model') if 'deepseek' in str(raising.get('model'))
                                                           else None) or 'deepseek-flash')
                  if args.mother_world == 'deepseek' else
                  SiliconFlowMother(args.mother_api_model or (raising.get('model') if '/' in str(raising.get('model'))
                                                              else None) or 'zai-org/GLM-5.3-Flash')
                  if args.mother_world == 'siliconflow' else
                  MailboxMother(timeout=float(raising.get('mailbox_timeout_s', 900))) if args.mother_world == 'mailbox' else
                  ApiMother(args.mother_api_model))
        if args.mother_fallback == 'siliconflow' and args.mother_world in ('deepseek', 'gemini', 'api'):
            mother = FallbackMother(mother, SiliconFlowMother('zai-org/GLM-5.3-Flash'))
            print(f'Mouth: {mother.name}', flush=True)
        family = raising_full.get('family') or {}
        # The first contact establishes safety without teaching the mouth to prepend the child's name to every turn.
        # Naming remains available in the family memory and for meaningful attention calls, but is not a fixed greeting.
        opening = (f"Hello there. {family.get('mother', {}).get('name', 'Mama')} is here."
                   if family.get('child', {}).get('name') else 'Hello, little one.')
        state_path = args.output / 'mother_state.json'
        saved_talk = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
        saved_world = resume_mother.get('mother_world') if args.resume else None
        if saved_world is not None:
            opening = None
        elif saved_talk:
            # a continuing lineage: no new greeting -- the conversation goes on, the mother calls her by name
            opening = f"{family.get('child', {}).get('name', 'little one')}?"
        # A boundary token ends the turn; short motor pauses retain partial speech.
        # Sustained silence uses the world's existing patience, or the safety token cap.
        child_end = tuple(raising.get('child_end_of_turn') or ('\n', '.', '?', '!'))
        mother_world = MotherWorld(mother, tiktoken.get_encoding('gpt2'), max_fly_words=args.mother_max_fly_words,
                                   opening=opening, end_of_turn=child_end,
                                   history_turns=int(raising.get('history_turns', 20)),
                                   ticks_per_beat=ticks_per_beat,
                                   pause_beats_before_reply=pause_beats,
                                   silent_beats_threshold=silent_thresh,
                                   call_prompt=call_prompt,
                                   contingent=bool(raising.get('contingent', True)))
        if saved_world is not None:
            mother_world.load_state_dict(saved_world)
            print(f'Mother world: exact continuation, turn={mother_world.turn}, '
                  f'unread={len(mother_world.queue)}, partial speech={len(mother_world.fly_words)}', flush=True)
        elif saved_talk:
            # the restored history comes first, then this chunk's call; the mouth's cached prefix stays anchored
            mother_world.transcript = [tuple(x) for x in saved_talk['transcript']] + mother_world.transcript
            mother_world.history_anchor = int(saved_talk.get('history_anchor', 0))
            mother_world.silent_beats = int(saved_talk.get('silent_beats', 0))
            print(f'Mother world: the conversation continues ({len(saved_talk["transcript"])} lines restored, '
                  f'from {saved_talk.get("trained")} words)', flush=True)
        if saved_world is None:
            learner.last_voice = bool((saved_talk or {}).get('last_voice', False))
            if saved_talk:
                origin['mother_world_migration'] = 'legacy transcript only; unread queue and delayed credit unavailable'
        attach_mother_senses(model, args)
        print(f'Child end of turn: {child_end!r}, or {mother_world.speech_pause_limit()} silent ticks, '
              f'or {args.mother_max_fly_words} tokens; short pauses retain speech', flush=True)
        mother_rng = np.random.default_rng(int(trained) + 104729)
        if args.resume and resume_mother.get('mother_rng') is not None:
            mother_rng.bit_generator.state = resume_mother['mother_rng']
        print(f'Mother world ({args.mother_world}): turn-taking, self-hearing, beats ({ticks_per_beat} ticks/beat), '
              f'pause {pause_beats} beat(s), silent call threshold {silent_thresh}, fly T={args.mother_temperature}',
              flush=True)
    shelf = probe_fn = None
    if getattr(args, 'bookshelf', None):
        import information_boltzmann.core.fly_bptt_learning as fly_learning
        from scripts.ib.fly_bookshelf import Bookshelf, make_probe_fn
        shelf = Bookshelf(train, selector=args.bookshelf, seed=args.bookshelf_seed, log_path=args.output / 'bookshelf.jsonl')
        shelf_opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(),
                          conductance_gains=model.get_conductance_gains(), alif_params=model.get_alif_params(),
                          stp_params=model.get_stp_params())
        probe_fn = make_probe_fn(fly_learning, model, learner, int(args.go_max_ticks or 32), shelf_opts)
        print(f'Curious shelf: {len(shelf.books)} books, selector {args.bookshelf}', flush=True)
    try:
        while trained < target:
            speakers = None
            if mother_world is not None:
                refresh_raising_plan()
                # the mother world: live the window first (the fly's words sampled from its reads, her replies), then
                # train on exactly what happened (the same words and intervals, learner.forced_world)
                before = len(mother_world.transcript)
                learner.speak_credit = args.mother_credit
                learner.consequence_native_enabled = bool(args.mother_consequence_native)
                learner.mother_action_clock = args.mother_action_clock
                learner.generative_shadow_enabled = bool(args.mother_generative_shadow)
                learner.native_generative_enabled = bool(args.mother_native_generative)
                learner.native_context_enabled = bool(args.mother_native_context)
                learner.context_initial_variance = args.mother_context_initial_variance
                learner.context_process_variance = args.mother_context_process_variance
                learner.native_generative_reference = args.mother_generative_reference
                learner.native_generative_candidates = args.mother_generative_candidates
                learner.native_generative_ticks = args.mother_generative_ticks
                learner.native_credit_enabled = bool(args.mother_native_credit)
                if args.mother_credit == 'epistemic':
                    import information_boltzmann.core.fly_bptt_learning as _fl
                    window_state0 = _fl.FlyPhysicalState(**{k: (tuple(t.detach().clone() for t in v) if k == 'ring'
                                                                else v.detach().clone())
                                                            for k, v in learner.state.state_dict().items()})
                    window_token0 = int(learner.previous_token)
                targets_, forced_, speakers = live_window(learner, mother_world, args.window, mother_rng,
                                                          float(args.stream_pace_min or 0), float(args.stream_pace_mean or 0),
                                                          temperature=args.mother_temperature,
                                                          speak_act=args.mother_speak_act,
                                                          interrupt=args.mother_interrupt)
                values = np.array(targets_, dtype=np.int64)
                learner.forced_world = forced_
                with (args.output / 'mother.jsonl').open('a', encoding='utf-8') as out:
                    for index in range(before, len(mother_world.transcript)):
                        who, text = mother_world.transcript[index]
                        out.write(json.dumps({'trained': trained, 'who': who, 'text': text,
                                              **mother_world.turn_observation(index)}) + '\n')
                mouth = mother_world.mother
                request = getattr(mouth, 'last_prompt_messages', None)
                if request is not None and request is not getattr(mouth, '_logged_prompt_messages', None):
                    # Log the last mouth request in this window, without auth headers.
                    # Preserve rejected drafts separately from what the child actually heard.
                    with (args.output / 'mother_prompt_audit.jsonl').open('a', encoding='utf-8') as out:
                        out.write(json.dumps({'trained': trained, 'scope': 'last API reply in this window',
                            'model': mouth.model, 'messages': request,
                            'grounding_rejections': getattr(mouth, 'last_grounding_rejections', []),
                            'delivered_reply': getattr(mouth, 'last_delivered_reply', ''),
                            'raw_reply': getattr(mouth, 'last_raw_reply', '')}, ensure_ascii=False) + '\n')
                    mouth._logged_prompt_messages = request
                try:
                    mon_tmp = Path('data/harness/monitor.tmp')
                    mon_dst = Path('data/harness/monitor.json')
                    mon_data = {
                        'trained': int(trained),
                        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'monitor': mother_world.monitor(),
                        'recent_transcript': [{'who': w, 'text': t} for w, t in mother_world.transcript[-10:]],
                        'plan_topic': raising_full.get('parenting', {}).get('current_topic') or (raising_full.get('parenting', {}).get('topics') or [None])[0],
                        'plan_intent': raising_full.get('parenting', {}).get('intention'),
                        'plan_stage': raising_full.get('parenting', {}).get('stage'),
                        'plan_mtime': round(raising_mtime, 1),
                    }
                    mon_tmp.write_text(json.dumps(mon_data, ensure_ascii=False, indent=1), encoding='utf-8')
                    mon_tmp.replace(mon_dst)
                except Exception:
                    pass
            elif shelf is not None:
                values = shelf.next_window(args.window)
            else:
                values = np.array(train[cursor+1:cursor+args.window+1], dtype=np.int64)
            begin = time.perf_counter()
            if os.environ.get('FLY_TORCH_PROFILE') and trained == start_trained + 2 * args.window:
                # one window under the torch profiler (speed work): CPU and CUDA time per op, forward and backward
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    scores = observe(values, training=True)
                    torch.cuda.synchronize()
                Path(os.environ['FLY_TORCH_PROFILE']).write_text(
                    prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=45) + '\n\n' +
                    prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=30), encoding='utf-8')
            else:
                scores = observe(values, training=True)
            if mother_world is not None and args.mother_credit == 'epistemic':
                # what this window's words taught, now that it was learned: the speaking credit for the next window
                from information_boltzmann.core.fly_mother_world import epistemic_credit
                learner.last_epistemic = epistemic_credit(learner, window_state0, window_token0, targets_,
                                                          horizon_ticks=args.mother_credit_ticks,
                                                          tau_ticks=speak_credit_tau(model, args),
                                                          trace=args.mother_credit_trace,
                                                          differential=args.mother_credit_differential,
                                                          event_clock=args.mother_credit_event_clock)
            # the pacemakers' homeostat, after the update and the replay: next window's three passes share one kappa
            learner.pacemaker_homeostat_step()
            trace = getattr(learner, 'last_ai_trace', None)
            if trace is not None:
                with open(args.output / 'ai_trace.jsonl', 'a') as handle:
                    handle.write(json.dumps({'bptt_train_tokens': trained, **trace}) + chr(10))
                learner.last_ai_trace = None
            world = getattr(learner, 'odour_world', None)
            if world is not None and world.log:
                with open(args.output / 'odour.jsonl', 'a') as handle:
                    for row in world.log:
                        handle.write(json.dumps({'bptt_train_tokens': trained, **row}) + '\n')
                world.log.clear()
            if world is not None and world.acts:
                with open(args.output / 'odour_acts.jsonl', 'a') as handle:
                    for tick, act, phase in world.acts:
                        handle.write(json.dumps({'bptt_train_tokens': trained, 'tick': tick, 'act': act, 'phase': phase}) + chr(10))
                world.acts.clear()
            if taste is not None:
                taste['windows'] += 1
                if taste['windows'] % args.taste_every == 0:
                    taste_episode()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            cursor += args.window
            trained += args.window
            if mother_world is not None:
                # Sidecar is for inspection; the checkpoint holds the authoritative paired brain/world state.
                atomic_json(args.output / 'mother_state.json', {
                    'trained': int(trained), **mother_world.state_dict(),
                    'last_voice': bool(getattr(learner, 'last_voice', False))})
            if shelf is not None:
                with torch.no_grad():
                    shelf_opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(),
                                      conductance_gains=model.get_conductance_gains(), alif_params=model.get_alif_params(),
                                      stp_params=model.get_stp_params())
                probe_fn = make_probe_fn(fly_learning, model, learner, int(args.go_max_ticks or 32), shelf_opts)
                shelf.after_window(probe_fn)
            if args.score_log:
                plan_meta = {}
                if mother_world is not None and raising_full:
                    p_cfg = raising_full.get('parenting', {})
                    plan_meta = {
                        'plan_topic': p_cfg.get('current_topic') or (p_cfg.get('topics') or [None])[0],
                        'plan_intent': p_cfg.get('intention'),
                        'plan_stage': p_cfg.get('stage'),
                        'plan_mtime': round(raising_mtime, 1)
                    }
                with args.score_log.open('a', encoding='utf-8') as out:
                    out.write(json.dumps({'trained': trained, 't': round(time.perf_counter(), 3),
                                          'scores': [float(x) for x in scores],
                                          'unigram': [float(x) for x in reference_surprisal[values]],
                                          **({'word_da': learner.last_word_da}
                                             if getattr(learner, 'last_word_da', None) is not None else {}),
                                          **({'speakers': speakers, 'sources': forced_.get('sources'),
                                              'quiet_ticks': [sum(b - a for a, b in ranges)
                                                              for ranges in forced_.get('quiet_ranges', [])],
                                              'reply_pulses': sum(bool(x) for x in (forced_.get('reply_onsets') or [])),
                                              # per word, aligned with sources / word_da: the word that carried her
                                              # answer's dopamine pulse (B), so B-locked writes can be compared
                                              'reply_onsets': [bool(x) for x in (forced_.get('reply_onsets') or [])],
                                              'monitor': mother_world.monitor(), **plan_meta}
                                             if speakers is not None else {}),
                                          **({'epistemic': learner.last_epistemic}
                                             if getattr(learner, 'last_epistemic', None) else {}),
                                          **({'homeo': learner.last_homeo}
                                             if getattr(learner, 'last_homeo', None) else {})}) + '\n')
            first_pass_meter.record(scores, reference_surprisal[values])
            first_pass_meter.verify_cursor(trained, cursor)
            phase_meter.record(scores, reference_surprisal[values])
            phase_meter.verify_cursor(trained, cursor)
            recent_a = (recent_a + values.tolist())[-args.replay_tokens:]
            recent_scores = (recent_scores + scores)[-args.replay_tokens:]
            if trained == start_trained + args.window or trained % args.log_every_tokens == 0:
                row = progress('running')
                row.update(window_prequential_nll=float(np.mean(scores)),
                    seconds_per_window=seconds, milliseconds_per_token=1000*seconds/args.window)
                log(row)
                ctm_info = ""
                if 'ctm_k_cert' in last_grad:
                    ctm_info = f" (k_cert={last_grad['ctm_k_cert']:.1f}/{learner.settle_ticks}, t0={last_grad.get('ctm_loss_tick0',0):.2f}, tS={last_grad.get('ctm_loss_tickS',0):.2f})"
                adaptive_info = ""
                if getattr(learner, 'adaptive_admission', False) and 'adaptive_ticks_mean' in last_grad:
                    adaptive_info = f" (admit_ticks={last_grad['adaptive_ticks_mean']:.1f}/{learner.settle_ticks or 14}, min={last_grad.get('adaptive_ticks_min', 0)}, max={last_grad.get('adaptive_ticks_max', 0)})"
                mcr_info = ""
                if getattr(learner, 'lambda_mcr2', 0.0) > 0.0 and getattr(learner, 'last_mcr2_metrics', None):
                    mcr_info = f" [ΔR={learner.last_mcr2_metrics.get('mcr2_delta_R', 0):.2f}]"
                print(f'BPTT {trained}/{target} | NLL {np.mean(scores):.4f}{ctm_info}{adaptive_info}{mcr_info} | {seconds:.3f}s/{args.window} | peak {row["vram_peak_mib"]:.0f} MiB', flush=True)
            preserve_committed_resource_window(pending_resource_error, checkpoint)
            if trained == start_trained + args.window:
                checkpoint()
                if args.calibrate:
                    progress('calibrated')
                    return
            if (args.output / 'STOP').exists():
                checkpoint()
                progress('paused')
                return
            if (args.checkpoint_every_tokens > 0 and trained < target
                    and (trained - start_trained) % args.checkpoint_every_tokens == 0):
                checkpoint()
            if not args.skip_eval and (trained >= next_eval or trained == target):
                if val_cursor + args.eval_tokens > len(val):
                    raise ValueError('Fresh validation stream exhausted')
                event_start, updates_start = learner.events, learner.updates
                physical_start = learner.physical_ticks
                fresh = np.array(val[val_cursor:val_cursor+args.eval_tokens], dtype=np.int64)
                B = observe(fresh)
                val_cursor += args.eval_tokens
                A2 = observe(recent_a)
                live = float(np.mean(B))
                is_best = live < best
                best = min(best, live)
                z = learner.latent_window.detach().cpu()
                z = z-z.mean(0)
                eigenvalues = torch.linalg.eigvalsh(z@z.T).clamp_min(0)
                probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
                effective_rank = float((-(probabilities*probabilities.clamp_min(1e-30).log()).sum()).exp())
                report = {'bptt_train_tokens': trained, 'train_cursor': cursor,
                    'event_interval': [event_start, learner.events],
                    'physical_tick_interval': [physical_start, learner.physical_ticks],
                    'update_interval': [updates_start, learner.updates],
                    'fresh_validation_cursor': val_cursor, 'live_prequential_nll': live,
                    'B_curve': B, 'A1_curve': recent_scores, 'A2_replay_curve': A2,
                    'replay_tokens': recent_a, 'actual_intervening_events': args.eval_tokens,
                    'actual_intervening_physical_ticks': learner.physical_ticks - physical_start,
                    'A2_prequential_nll': float(np.mean(A2)),
                    'matched_revisit_nll_change': float(np.mean(recent_scores[1:])-np.mean(A2[1:])),
                    'bridge_scoring': 'all actual bridges scored; first A token excluded from matched gap',
                    'recovery': recovery_summary(B, block_tokens=16, hold_blocks=2,
                        reference_nll=reference_surprisal[fresh]),
                    'savings': savings_summary(recent_scores, A2, intervening_events=args.eval_tokens,
                        block_tokens=16, hold_blocks=2),
                    'centered_effective_rank': effective_rank,
                    'health_scope': 'field energy, activity and centered representation structure; FTLE not measured in this branch',
                    **health(), **prediction_accounting()}
                log(report, 'lifelong_evaluation.jsonl')
                ag_val = report['recovery'].get('ag')
                ag_str = f" AG={ag_val:.4f}" if ag_val is not None else ""
                print(f'[ACTIVE EVAL {trained}] fresh B NLL={live:.4f}{ag_str}; A revisit={np.mean(A2):.4f}', flush=True)
                checkpoint(is_best)
                progress('running')
                next_eval += args.validate_every_tokens
        if args.skip_eval:
            # Mother-world chunks must save their final state even while ABA/AG evaluation is deferred.
            checkpoint()
        progress('completed')
    except BaseException as exc:
        row = progress('failed')
        row['error'] = repr(exc)
        atomic_json(args.output / 'progress.json', row)
        raise


if __name__ == '__main__':
    main()
