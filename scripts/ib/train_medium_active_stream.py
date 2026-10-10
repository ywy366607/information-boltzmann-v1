"""Joint BPTT32 training of the latest 3D medium with live, never-reset evaluation.

This entry point joins the captured learner to genuine A->B->A traffic, with
optional local dynamic/temporal readout. Scores precede optimizer updates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from information_boltzmann.core.plastic_medium import COLLISION_PRESETS
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.evaluation import field_energy_statistics
from information_boltzmann.runtime.active_medium_training import ActiveMediumTrainer
from information_boltzmann.runtime.checkpoint_receipt import (
    CheckpointWriteError, append_attempt_journal, checkpoint_position,
    inspect_resume_gap, require_saved_position, save_complete_checkpoint,
)
from information_boltzmann.runtime.execution_cache import configure_execution_cache
from information_boltzmann.runtime.gpu_memory import total_gpu_memory
from information_boltzmann.runtime.lifelong_evaluation import recovery_summary, savings_summary
from information_boltzmann.runtime.medium_health import ChunkHealthCapture, MediumHealthAuditor, conditional_state_response, spatial_medium_snapshot
from information_boltzmann.runtime.optimization import make_medium_optimizer
from information_boltzmann.runtime.training import CapturedPlasticChunk
from scripts.ib.train_plastic_conductance import learning_rate, pack_belief, unpack_belief


@torch.no_grad()
def prenatal_development(model, windows, events, event_duration, rate, mu, device, solver_max_step=None):
    """Development before birth: spontaneous activity (uniformly random tokens) is written through the real ports,
    the medium evolves with its real physics, and the through-flow carves channels out of soil (Tero rule) once per
    window. No text, no loss, no optimizer: only the installed shares of the riverbed change."""
    medium = model.medium
    substeps = 1 if not solver_max_step else max(1, math.ceil(event_duration / solver_max_step - 1e-9))
    table = torch.nn.functional.normalize(model.source.embedding.weight, dim=-1)
    vocab = table.shape[0]
    generator = torch.Generator(device='cpu').manual_seed(20261010)
    belief = model.initial_belief()
    log = []
    for window in range(windows):
        prepared = medium.prepare_evolution()
        flow = None
        for _ in range(events):
            token = torch.randint(vocab, (1,), generator=generator).to(device)
            duration = model.event_time(belief, event_duration)
            belief, _ = model.assimilate(belief, token, token_features=table, diagnostics=False, training_terms=False)
            belief, _, _ = model.advance(belief, duration, substeps=substeps, prepared=prepared, diagnostics=False,
                                         return_motion=True)
            sample = medium.through_flow(belief.medium, prepared)
            flow = sample if flow is None else flow + sample
        info = medium.channel_growth_step(flow / events, rate, mu)
        info['window'] = window
        log.append(info)
        if window % 10 == 0 or window == windows - 1:
            print(f'PRENATAL window {window}: soil mean {info["soil_mean"]:.3f} cv {info["soil_cv"]:.3f} '
                  f'channel anisotropy {info["channel_anisotropy"]:.3f} flow cv {info["flow_cv"]:.3f}', flush=True)
    return log


def separate_ports(model, write_face=0.25, read_face=0.75):
    """Birth body plan: writes on the sensory face, reads on the motor face (half a period apart in x).

    Each face holds a balanced y-z grid; the half-cell offset of the default layout is kept so no footprint starts
    at a zero-derivative position. Radii, budgets and learnability are unchanged.
    """
    shape = model.medium.shape

    def face(count, x):
        ny = 1
        for d in range(1, int(count ** 0.5) + 1):
            if count % d == 0:
                ny = d
        nz = count // ny
        ny, nz = max(ny, nz), min(ny, nz)
        yz = torch.stack(torch.meshgrid((torch.arange(ny) + 0.5) / ny, (torch.arange(nz) + 0.5) / nz,
                                        indexing='ij'), -1).reshape(-1, 2)
        centers = torch.cat((torch.full((count, 1), float(x)), yz), -1)
        return (centers + 0.5 / torch.tensor(shape, dtype=centers.dtype)).remainder(1.0)

    ports = model.write_agent.local_ports
    probes = model.readout.probe_coords
    with torch.no_grad():
        ports.centers.copy_(face(ports.centers.shape[0], write_face).to(ports.centers))
        probes.copy_(face(probes[..., 0].numel(), read_face).reshape(probes.shape).to(probes))


def atomic_json(path, value):
    """Windows readers can briefly deny replacement; keep training through it."""
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for attempt in range(50):
        try:
            os.replace(temporary, path)
            return
        except (PermissionError, OSError):
            if attempt == 49:
                raise
            time.sleep(.02)


def content_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def train_only_reference(path, vocab_size=50257):
    values = np.load(path, mmap_mode='r')
    counts = np.ones(vocab_size, dtype=np.float64)
    for offset in range(0, len(values), 1_000_000):
        counts += np.bincount(np.array(values[offset:offset + 1_000_000], dtype=np.int64),
                              minlength=vocab_size)
    return -np.log(counts / counts.sum()), hashlib.sha256(counts.tobytes()).hexdigest()


def initialize_pretrained_vocabulary(model, path):
    """Best uncentered rank-d Frobenius projection of a pretrained word table.

    Retain its leading right-singular subspace, and match the decoder's existing
    initialization RMS with one global scale. Cosines in that subspace are intact.
    The source table is frozen, matching the fly baseline's embedding policy.
    """
    from safetensors import safe_open
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        table = handle.get_tensor('wte.weight').float()
    vocab, width = model.source.embedding.weight.shape
    if table.shape[0] != vocab or table.shape[1] < width:
        raise ValueError('Pretrained vocabulary or dimension mismatch')
    if table.shape[1] == width:
        projected = table
        retained_fraction = 1.0
    else:
        covariance = table.double().T @ table.double()
        eigenvalues, basis = torch.linalg.eigh(covariance)
        basis = basis[:, -width:].float()
        projected = table @ basis
        retained_fraction = float(eigenvalues[-width:].sum() / eigenvalues.sum())
    target_rms = float(model.decoder.weight.detach().square().mean().sqrt())
    scale = target_rms / float(projected.square().mean().sqrt())
    with torch.no_grad():
        model.source.embedding.weight.copy_(projected.to(model.source.embedding.weight))
        model.decoder.weight.copy_((projected * scale).to(model.decoder.weight))
    model.source.embedding.weight.requires_grad_(False)
    return {'path': str(path), 'sha256': content_hash(path),
            'method': 'direct word table' if table.shape[1] == width else 'uncentered leading right-singular subspace',
            'source_width': table.shape[1], 'medium_width': width,
            'retained_squared_norm_fraction': retained_fraction,
            'decoder_global_scale': scale, 'decoder_initialization_rms': target_rms,
            'embedding_frozen': True, 'decoder_trainable': True}


def prepare_hopf_continuation(expected, actual, *, source_checkpoint,
                              parameter_names, state_names, added_count):
    """Authorize only identity-initialized junctions and their audited code paths."""
    expected['constructor'] = dict(expected['constructor'])
    if expected['constructor'].get('hopf_recomposition', False):
        raise ValueError('Source already owns junctions; use exact continuation rather than adoption')
    expected['constructor']['hopf_recomposition'] = True
    for name in ('parameters', 'trainable_parameters', 'active_graph_parameters'):
        if actual[name] - expected[name] != added_count:
            raise ValueError(f'Junction adoption must add only its declared parameters: {name}')
        expected[name] = actual[name]
    expected['hopf_recomposition'] = actual['hopf_recomposition']
    if 'pretrained_vocabulary' in expected and 'pretrained_vocabulary' in actual:
        actual['pretrained_vocabulary'] = expected['pretrained_vocabulary']
    old_sources, new_sources = expected['source_hashes'], actual['source_hashes']
    allowed = {'scripts/ib/train_medium_active_stream.py',
               'information_boltzmann/core/hopf_recomposition.py',
               'information_boltzmann/core/medium_junction.py',
               'information_boltzmann/core/plastic_medium.py',
               'information_boltzmann/core/plastic_ports.py',
               'information_boltzmann/runtime/optimization.py',
               'information_boltzmann/runtime/active_medium_training.py',
               'information_boltzmann/runtime/medium_health.py',
               'information_boltzmann/runtime/training.py',
               'information_boltzmann/runtime/checkpoint_receipt.py'}
    changed = {key for key in old_sources.keys() | new_sources.keys()
               if old_sources.get(key) != new_sources.get(key)}
    if changed - allowed:
        raise ValueError(f'Unaudited unrelated junction migration sources: {changed - allowed}')
    compatibility_sources = {
        'information_boltzmann/runtime/active_medium_training.py',
        'information_boltzmann/runtime/medium_health.py',
        'information_boltzmann/runtime/training.py'}
    compatibility_evidence = []
    if changed & compatibility_sources:
        audit_path = (Path(__file__).resolve().parents[2] /
            'results/published/medium_gen3_source_continuation_audit_20261009.json')
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        records = {record['path']: record for record in audit['records']}
        for path in sorted(changed & compatibility_sources):
            evidence = records.get(path)
            if (evidence is None or not evidence['exact_reconstruction_verified'] or
                    evidence['before_sha256'] != old_sources.get(path) or
                    evidence['after_sha256'] != new_sources.get(path)):
                raise ValueError(f'Junction continuation compatibility source needs an exact reviewed diff: {path}')
            compatibility_evidence.append(evidence)
    expected['source_hashes'] = new_sources
    return {'source_checkpoint': str(source_checkpoint),
            'protocol': 'identity_initialized_persistent_flux_junction_adoption_v1',
            'before': old_sources, 'after': new_sources,
            'changed_sources': sorted(changed),
            'compatibility_source_evidence': compatibility_evidence,
            'solver_migration': ('declared local-junction angular-rate budget added to the existing numerical '
                                 'step ceiling; physical clock unchanged; deterministic integer schedule '
                                 'held fixed for conditional VJP replay, not a planner gradient'),
            'added_parameter_names': sorted(parameter_names),
            'added_parameter_count': added_count,
            'added_state_dict_names': sorted(state_names),
            'initialization': 'exact zero gate; no new physical state or changed existing parameter',
            'preserved': 'full physical belief and clocks, OU prior/sample, RNG, data cursors, evaluation/optimizer cadence and every inherited Adam moment'}


def validate_junction_pending_gradients(model, learner_state):
    """Only the new identity gate may lack gradient storage on adoption."""
    prefix = 'medium.hopf_pathway.'
    inherited = {name for name, _ in model.named_parameters() if not name.startswith(prefix)}
    saved_names = set(learner_state['pending_gradients'])
    if saved_names != inherited:
        raise ValueError('Junction adoption must preserve every inherited gradient entry; '
                         f'missing={sorted(inherited - saved_names)}, '
                         f'unexpected={sorted(saved_names - inherited)}')
    if learner_state['pending'] != 0:
        raise ValueError('Junction adoption requires a completed credit boundary')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pretrained-embedding', type=Path,
                        help='GPT2 safetensors word table; rank-d projection, frozen source')
    parser.add_argument('--data-dir', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--steps', type=int, default=3000, help='Fresh training groups, each tokens long')
    parser.add_argument('--compile-event', action='store_true')
    parser.add_argument('--medium-segment-graph', action='store_true',
                        help='Forward-only physical microstep CUDA graphs; full event VJP recomputation retained')
    parser.add_argument('--shared-credit-execution', action='store_true',
                        help='One primal/replay per event with separate full task and writer cotangents')
    parser.add_argument('--deferred-writer-credit', action='store_true',
                        help='Exact window GEMMs for single-row writer/readout weight gradients; requires shared credit')
    parser.add_argument('--credit-history-offload', action='store_true',
                        help='Exact host storage of inactive recurrent field history; requires shared credit')
    parser.add_argument('--fused-optimizer', action='store_true',
                        help='Fused CUDA AdamW arithmetic with unchanged saved hyperparameters and moments')
    parser.add_argument('--read-key-execution', choices=('dense', 'support'), default='dense',
                        help='Exact finite-support key projection; no field or parameter compression')
    parser.add_argument('--allow-execution-change', action='store_true',
                        help='Audited execution-only continuation; all learning and physical settings fixed')
    parser.add_argument('--activation-checkpointing', action='store_true')
    parser.add_argument('--checkpoint-granularity', choices=('nested', 'event'), default='nested',
                        help='Event-only replay for short physical intervals; nested replay for long ones')
    parser.add_argument('--optimizer-state-offload', action='store_true',
                        help='Eager only: stage idle Adam moments on CPU during full-window physics/backward')
    parser.add_argument('--channels', type=int, default=128)
    parser.add_argument('--shape', type=int, nargs=3, default=[8, 8, 4])
    parser.add_argument('--material-initialization', choices=('uniform', 'spectral-xavier'), default='uniform')
    parser.add_argument('--material-field-std', type=float, default=1.0,
                        help='Reference-grid material standard deviation; unit variance by default')
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--chunk-tokens', type=int, default=32)
    parser.add_argument('--event-duration', type=float, required=True)
    parser.add_argument('--intrinsic-time-reference', type=float,
                        help='Explicit initial physical interval for a pre-input history clock')
    parser.add_argument('--intrinsic-max-duration', type=float,
                        help='Optional execution ceiling; exceeding it stops rather than clips physics')
    parser.add_argument('--solver-max-step', type=float,
                        help='Numerical physical integration resolution, independent of event time')
    parser.add_argument('--observer-max-step', type=float,
                        help='Temporal probe sampling resolution, independent of solver substeps')
    parser.add_argument('--max-evolution-steps', type=int, default=4096,
                        help='Explicit per-event execution budget, not a biological constant')
    parser.add_argument('--material-bandwidth', choices=('legacy', 'runtime'), default='legacy',
                        help='Runtime gives the slow material all modes supported by the run grid')
    parser.add_argument('--temporal-time-reference', type=float,
                        help='Physical unit for dimensionless learned omega*time_reference')
    parser.add_argument('--temporal-physical-reference', type=float,
                        help='Freeze physical filter initialization independently of new input cadence')
    parser.add_argument('--write-aperture-budget', type=float,
                        help='Sum of physical support-box volumes over all input ports')
    parser.add_argument('--read-aperture-budget', type=float,
                        help='Sum of physical support-box volumes over all read ports')
    parser.add_argument('--structure-config', type=Path,
                        help='Explicit Gaussian/resource/OU/maintenance scales for the structural candidate')
    parser.add_argument('--structure-dual-lr', type=float,
                        help='Optional nonnegative maintenance-supply dual update; otherwise fixed declared price')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--decay', type=int, default=300)
    parser.add_argument('--validate-every-tokens', type=int, default=5000)
    parser.add_argument('--eval-tokens', type=int, default=256)
    parser.add_argument('--eval-block-tokens', type=int, default=16)
    parser.add_argument('--replay-tokens', type=int, default=128,
                        help='Total return budget: one bridge plus 127 matched A targets by default')
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--seed', type=int, default=449)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--execution', choices=('eager', 'graph'), default='graph')
    parser.add_argument('--vram-limit-mib', type=int, default=3900)
    parser.add_argument('--calibrate-only', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--adopt-capacity-growth', action='store_true',
                        help='Explicit completed-window rule migration to a distinct output; preserve all other state')
    parser.add_argument('--no-stp', action='store_true', help='Birth without short-term plasticity (measured idle)')
    parser.add_argument('--no-conduction', action='store_true', help='Birth without conduction plasticity (measured idle)')
    parser.add_argument('--no-health', action='store_true', help='Skip per-event energy ledgers and fourth-pillar capture in the physics step')
    parser.add_argument('--structural-gravity', type=float, default=0.0,
                        help='Installed-capacity self-gravity: linear e-folding rate per window of every density mode (0 = off)')
    parser.add_argument('--structural-gravity-screening', type=float, default=0.0,
                        help='kappa^2 of the finite-range gravity force (lattice Laplacian units); needs --structural-gravity')
    parser.add_argument('--structural-gravity-diffusion', type=float, default=0.0,
                        help='D of the weak diffusion that, with the screening, selects the structure wavelength')
    parser.add_argument('--internal-time-factor', type=int, default=1,
                        help='K: event duration = K x --event-duration at the SAME solver step (K substeps, one observer interval)')
    parser.add_argument('--structure-noise', type=float, default=1.0,
                        help='Scale of the riverbed sampling noise per structural window; 0 takes credit at the posterior mean')
    parser.add_argument('--collision-preset', choices=sorted(COLLISION_PRESETS), default=None,
                        help='Named collision operator (boltzmann: bilinear pair collisions; boltzmann-mlp: learned MLP operator)')
    parser.add_argument('--soil-absorption', type=float, default=0.0,
                        help='Unified riverbed: absorption rate (1/time) of a pure-soil (idle) site; 0 = idle is inert')
    parser.add_argument('--channel-growth', type=float, default=0.0,
                        help='Unified riverbed: share relaxation rate per window toward the through-flow target (Tero rule); 0 = off')
    parser.add_argument('--channel-mu', type=float, default=2.0, help='Tero feedback exponent of the channel target')
    parser.add_argument('--prenatal-windows', type=int, default=0,
                        help='Token-free development before birth: windows of spontaneous random writes with channel growth')
    parser.add_argument('--freeze', choices=('none', 'interface', 'interface+medium'), default='none',
                        help='Block loss-reducing paths outside the medium: interface freezes the write agent, the readout '
                             '(probes, attention, correction MLP) and the temporal read, leaving the medium and the linear '
                             'decoder trainable; interface+medium also freezes every medium parameter (reservoir control; '
                             'structure still develops before birth)')
    parser.add_argument('--brain-interface', action='store_true',
                        help='Only the medium computes and remembers: no readout correction MLP, no write-side token '
                             'prediction (the observed feature is written), and no temporal read (incompatible with --temporal-read)')
    parser.add_argument('--read-heads', type=int, default=4, help='Read heads (default 4)')
    parser.add_argument('--read-queries', type=int, default=4,
                        help='Read probes per head (default 4); heads*queries probes in total. For an output bottleneck use fewer '
                             'probes and scale --read-aperture-budget by the same factor so each probe keeps its size.')
    parser.add_argument('--write-split', action='store_true',
                        help='Each write port carries only its own block of channels (the token is divided over space)')
    parser.add_argument('--port-layout', choices=('grid', 'separated'), default='grid',
                        help='Birth port positions: grid (interleaved, reads next to writes) or separated '
                             '(writes on the sensory face x=1/4, reads on the motor face x=3/4); positions stay learnable')
    parser.add_argument('--plasticity-options', type=json.loads, default=None,
                        help='JSON: spectrum [tau_min, tau_max], depression, compensation for the conduction/STP fast layers')
    parser.add_argument('--collision-options', type=json.loads, default=None,
                        help='JSON of opt-in collision activity: center_receptors, position_rates, bed_rates (default: legacy)')
    parser.add_argument('--hopf-recomposition', action='store_true',
                        help='Enable capacity-paid physical flux junctions with CK edit metadata')
    parser.add_argument('--adopt-hopf-branch', action='store_true',
                        help='Identity-initialized capacity-paid flux junction adoption; preserve the complete continuous individual')
    parser.add_argument('--check-resume-only', action='store_true',
                        help='Validate complete continuation and exit without writing or consuming any target')
    parser.add_argument('--anisotropic-transport', action='store_true',
                        help='Learn a positive-definite local propagation tensor')
    parser.add_argument('--transport-capacity-budget',
                        help='Opt-in slow-material mean trace(A) ceiling: positive number or birth')
    parser.add_argument('--allow-default-preserving-extension', action='store_true',
                        help='Audited continuation with disabled optional capacity and read-only energy-current telemetry')
    parser.add_argument('--dynamic-read', action='store_true',
                        help='Explicit finite-aperture f and analytic physical-motion read branch')
    parser.add_argument('--temporal-read', action='store_true',
                        help='Dynamic local read plus persistent signed complex probe histories')
    parser.add_argument('--temporal-half-life-events', type=float, nargs='+',
                        default=[1., 4., 16., 64.],
                        help='Initial filter half-lives in intrinsic-reference units (fixed event duration for legacy); learned thereafter')
    parser.add_argument('--temporal-frequency-fractions', type=float, nargs='+',
                        default=[0., 1/16, 1/8, 1/4],
                        help='Initial angular frequencies as fractions of pi/initial physical interval; learned thereafter')
    parser.add_argument('--allow-entrypoint-change', action='store_true',
                        help='Allow audited operational trainer changes; model/learner hashes still must match')
    args = parser.parse_args()
    if args.internal_time_factor < 1:
        raise SystemExit('--internal-time-factor must be a positive integer')
    if args.internal_time_factor > 1:
        if args.solver_max_step is None:
            raise SystemExit('--internal-time-factor needs --solver-max-step (the substep size that stays fixed)')
        args.event_duration *= args.internal_time_factor      # K substeps per token at the unchanged solver step
        args.observer_max_step = args.event_duration          # ONE observer interval: more would repeat the whole event step
    spectrum = (args.plasticity_options or {}).get('spectrum')
    if spectrum is not None and spectrum[1] == 'horizon':
        # no human-set upper limit: the slowest identifiable time constant is the total observed time of this run
        args.plasticity_options = {**args.plasticity_options,
                                   'spectrum': [float(spectrum[0]), float(args.steps * args.tokens * args.event_duration)]}
    if args.deferred_writer_credit and (not args.shared_credit_execution or args.compile_event):
        parser.error('Deferred writer credit requires shared credit and uncompiled event orchestration')
    if args.credit_history_offload and not args.shared_credit_execution:
        parser.error('Credit history offload requires shared credit')
    if (args.deferred_writer_credit or args.credit_history_offload) and (
            not args.activation_checkpointing or args.checkpoint_granularity != 'event'
            or args.structure_config is None):
        parser.error('Deferred/history execution requires structural event checkpoints')
    if args.check_resume_only and args.resume is None:
        parser.error('Continuation checking requires --resume')
    if args.adopt_capacity_growth:
        if args.resume is None or args.structure_config is None:
            parser.error('Capacity rule migration requires a source --resume and --structure-config')
        if args.output.resolve() == args.resume.resolve().parent:
            parser.error('Retain the source baseline; capacity migration requires a distinct output')
        if args.allow_execution_change or args.allow_default_preserving_extension or args.allow_entrypoint_change:
            parser.error('Capacity rule migration cannot be combined with broad continuation exceptions')
    if args.adopt_hopf_branch:
        if args.resume is None or not args.hopf_recomposition:
            parser.error('Hopf branch adoption requires a source --resume and --hopf-recomposition')
        if args.output.resolve() == args.resume.resolve().parent:
            parser.error('Retain the source baseline; Hopf branch adoption requires a distinct output')
        if args.allow_execution_change or args.allow_default_preserving_extension or args.allow_entrypoint_change or args.adopt_capacity_growth:
            parser.error('Hopf branch adoption cannot be combined with broad continuation exceptions')
    if sum((args.allow_execution_change, args.allow_default_preserving_extension,
            args.allow_entrypoint_change)) > 1:
        parser.error('Choose at most one continuation authorization mode')
    if args.medium_segment_graph and args.execution == 'graph':
        parser.error('The whole-window CUDA graph subsumes --medium-segment-graph; do not nest them')
    if args.optimizer_state_offload and args.execution != 'eager':
        parser.error('Optimizer state staging requires --execution eager')
    if args.checkpoint_granularity != 'nested' and not args.activation_checkpointing:
        parser.error('Event-only checkpoints require activation checkpointing')
    if args.temporal_physical_reference is not None and (not args.temporal_read
            or not math.isfinite(args.temporal_physical_reference) or args.temporal_physical_reference <= 0):
        parser.error('A positive physical filter reference requires temporal read')
    if (args.solver_max_step is None) != (args.observer_max_step is None):
        parser.error('Declare solver and observer resolution together')
    if args.intrinsic_time_reference is not None and args.solver_max_step is None:
        parser.error('Intrinsic time requires explicit solver and observer resolution')
    if args.solver_max_step is not None and (args.compile_event or (
            args.execution == 'graph' and args.intrinsic_time_reference is not None)):
        parser.error('An adaptive clock uses --execution eager with fused medium kernels (a fixed clock may use graph); omit --compile-event')
    if args.structure_config is not None and (not args.anisotropic_transport
            or args.transport_capacity_budget is not None or args.solver_max_step is None):
        parser.error('Structural posterior requires resolved tensor evolution and replaces the old capacity projection')
    if args.structure_config is not None and args.compile_event:
        parser.error('Structural evidence windows cannot use the compiled whole-event path')
    if args.structure_dual_lr is not None and (args.structure_config is None
            or not math.isfinite(args.structure_dual_lr) or args.structure_dual_lr < 0):
        parser.error('A finite nonnegative dual learning rate requires structural configuration')
    if args.transport_capacity_budget is not None:
        if not args.anisotropic_transport:
            parser.error('Capacity constraint requires tensor transport')
        if args.transport_capacity_budget != 'birth':
            try:
                args.transport_capacity_budget = float(args.transport_capacity_budget)
            except ValueError:
                parser.error('Capacity must be a positive number or birth')
            if not math.isfinite(args.transport_capacity_budget) or args.transport_capacity_budget <= 0:
                parser.error('Capacity must be finite and positive')
    if min(args.steps, args.tokens, args.chunk_tokens, args.eval_tokens,
           args.validate_every_tokens, args.replay_tokens, args.save_every, args.eval_block_tokens) < 1:
        parser.error('Positive budgets required')
    if (args.tokens % args.chunk_tokens or not 2 <= args.replay_tokens <= 128 or
            args.eval_tokens % args.tokens or args.replay_tokens % args.tokens):
        parser.error('Use fixed BPTT chunks and replay <= the 128-target live ledger')
    if not math.isfinite(args.event_duration) or args.event_duration <= 0:
        parser.error('Finite positive physical cadence required')
    if args.temporal_read:
        if args.dynamic_read:
            parser.error('Temporal read already includes dynamic read; select one flag')
        if (len(args.temporal_half_life_events) != len(args.temporal_frequency_fractions)
                or any(not math.isfinite(x) or x <= 0 for x in args.temporal_half_life_events)
                or any(not math.isfinite(x) for x in args.temporal_frequency_fractions)):
            parser.error('One finite frequency per positive filter half-life required')
    if args.device == 'cpu' and args.execution == 'graph':
        parser.error('CPU execution is eager')
    if not 0 < args.min_lr <= args.lr or not 0 < args.vram_limit_mib <= 4096:
        parser.error('Invalid learning rate or GPU budget')
    if not args.calibrate_only and (min(args.warmup, args.decay) < 0 or args.warmup + args.decay > args.steps):
        parser.error('WSD phases exceed training budget')
    if (args.output / 'config.json').exists() and args.resume is None:
        parser.error('Existing individual requires --resume')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    configure_execution_cache(Path('scratch/compiler_cache'))
    if args.device == 'cuda':
        memory = total_gpu_memory()
        if memory['used_bytes'] / 2**20 > 2800:
            raise RuntimeError('Another GPU job leaves insufficient capture headroom; release it before launch')
        total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
        # Leave driver/context headroom inside the requested dedicated budget.
        torch.cuda.set_per_process_memory_fraction(min(1., (args.vram_limit_mib - 192) / total_mib))
    train = np.load(args.data_dir / 'train.npy', mmap_mode='r')
    validation = np.load(args.data_dir / 'validation.npy', mmap_mode='r')
    if args.steps * args.tokens + 1 > len(train):
        parser.error('No corpus wrapping allowed')
    prior, prior_hash = train_only_reference(args.data_dir / 'train.npy')
    constructor = dict(vocab_size=50257, shape=tuple(args.shape), channels=args.channels,
                       bath_type='conductance', write_exchange='contact_mode', port_scope='compact',
                       activity_adaptation=True, short_term_plasticity=not args.no_stp, pre_decoder_norm=True,
                       medium_execution='fused' if args.device == 'cuda' else 'native',
                       port_execution='fused' if args.device == 'cuda' else 'native')
    if args.read_key_execution != 'dense':
        constructor['read_key_execution'] = args.read_key_execution
    if args.medium_segment_graph:
        constructor['medium_segment_graph'] = True
    if args.shared_credit_execution:
        constructor['shared_credit_execution'] = True
    if args.deferred_writer_credit:
        constructor['deferred_writer_credit'] = True
    if args.credit_history_offload:
        constructor['credit_history_offload'] = True
    if args.hopf_recomposition:
        constructor['hopf_recomposition'] = True
    constructor['anisotropic_transport'] = args.anisotropic_transport
    if args.no_conduction:
        constructor['adaptive_conduction'] = False
    if args.plasticity_options:
        constructor['plasticity_options'] = dict(args.plasticity_options)
    if args.write_split:
        constructor['write_split'] = True
    if args.brain_interface:
        if args.temporal_read:
            raise ValueError('--brain-interface keeps memory inside the medium; drop --temporal-read')
        constructor['read_correction'] = False
        constructor['write_prediction'] = False
    if (args.read_heads, args.read_queries) != (4, 4):
        constructor['heads'], constructor['queries'] = args.read_heads, args.read_queries
    if args.collision_preset or args.collision_options:
        constructor['collision_options'] = {**COLLISION_PRESETS.get(args.collision_preset, {}), **(args.collision_options or {})}
    if args.structure_config is not None:
        constructor['structure_options'] = json.loads(args.structure_config.read_text(encoding='utf-8'))
    if args.material_bandwidth == 'runtime':
        constructor['material_reference_shape'] = None
    for name in ('intrinsic_time_reference', 'intrinsic_max_duration', 'solver_max_step',
                 'observer_max_step', 'write_aperture_budget', 'read_aperture_budget'):
        if getattr(args, name) is not None:
            constructor[name] = getattr(args, name)
    if args.solver_max_step is not None:
        constructor['max_evolution_steps'] = args.max_evolution_steps
    if isinstance(args.transport_capacity_budget, float):
        constructor['transport_capacity_budget'] = args.transport_capacity_budget
    if tuple(args.shape) != (8, 8, 4):
        # Preserve physical communication apertures when refining the grid.
        # Derived from CompactTorusPorts on the original 8x8x4 reference grid.
        constructor.update(write_port_radius=(0.1875, 0.1875, 0.25),
                           read_port_radius=(0.125, 0.1875, 0.25))
    if args.dynamic_read:
        constructor['read_mode'] = 'dynamic'
    if args.temporal_read:
        initial_interval = args.temporal_physical_reference or args.intrinsic_time_reference or args.event_duration
        constructor.update(read_mode='temporal',
            temporal_time_reference=(args.temporal_time_reference if args.temporal_time_reference is not None
                                     else args.intrinsic_time_reference or 1.),
            temporal_rates=[math.log(2.) / (x * initial_interval)
                            for x in args.temporal_half_life_events],
            temporal_frequencies=[x * math.pi / initial_interval
                                  for x in args.temporal_frequency_fractions])
    saved = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=False) if args.resume else None
    source_receipt = None if saved is None else saved.get('checkpoint_receipt')
    source_position = None if saved is None else checkpoint_position(saved)
    source_gap = None if saved is None else inspect_resume_gap(args.resume.parent, source_position)
    if args.transport_capacity_budget == 'birth' and saved:
        if saved['config']['constructor'].get('transport_capacity_budget') is None:
            raise ValueError('Cannot introduce birth budget into an unconstrained continuous individual')
        constructor['transport_capacity_budget'] = saved['config']['constructor']['transport_capacity_budget']
    model = PlasticMediumPorts3D(**constructor).train()
    model.structure_dual_learning_rate = args.structure_dual_lr
    if not saved and args.material_initialization == 'spectral-xavier':
        # A declared birth is reproducible on CPU and CUDA. This also lets the
        # resource/time calibration reconstruct the actual initial material.
        model.medium.material.initialize_spectral_xavier(field_std=args.material_field_std)
    if not saved and args.plasticity_options:
        model.medium.initialize_plasticity_spectrum()
    if not saved and args.port_layout == 'separated':
        separate_ports(model)
    model = model.to(args.device)
    if args.transport_capacity_budget == 'birth' and not saved:
        from information_boltzmann.core.structural_resource import TransportCapacityBudget, transport_capacity
        with torch.no_grad():
            material = model.medium.material_field()
            speed = (model.medium.speed_reference * model.medium.log_speed(material).exp())[None]
            birth_capacity = transport_capacity(model.medium.raw_transport_factor(material, speed)).item()
        model.medium.transport_capacity_limit = birth_capacity
        model.medium.transport_budget = TransportCapacityBudget(birth_capacity).to(args.device)
        constructor['transport_capacity_budget'] = birth_capacity
    vocabulary_initialization = None
    if args.pretrained_embedding:
        vocabulary_initialization = initialize_pretrained_vocabulary(model, args.pretrained_embedding)
    if saved:
        if args.adopt_capacity_growth:
            from information_boltzmann.runtime.optimization import initialize_capacity_growth_branch
            saved_optimizer = initialize_capacity_growth_branch(
                model, saved['model'], saved['optimizer'], saved['learner'])
        elif args.adopt_hopf_branch:
            from information_boltzmann.runtime.optimization import initialize_hopf_branch
            validate_junction_pending_gradients(model, saved['learner'])
            saved_optimizer = initialize_hopf_branch(
                model, saved['model'], saved['optimizer'], saved['learner'])
        else:
            model.load_state_dict(saved['model'])
            saved_optimizer = saved['optimizer']
        belief = unpack_belief(saved['belief'], args.device)
    else:
        with torch.no_grad():
            model.decoder.bias.copy_(torch.from_numpy(-prior).to(model.decoder.bias))
        belief = model.initial_belief()
    model.medium.soil_absorption = args.soil_absorption
    if args.freeze != 'none':
        frozen = ['write_agent.', 'readout.', 'temporal_readout.']
        if args.freeze == 'interface+medium':
            frozen.append('medium.')
        for name, parameter in model.named_parameters():
            if name.startswith(tuple(frozen)) and not name.startswith('medium.structural_posterior.'):
                parameter.requires_grad_(False)
    prenatal = None
    if not saved and args.prenatal_windows:
        prenatal = prenatal_development(model, args.prenatal_windows, args.tokens, args.event_duration,
                                        args.channel_growth or 0.1, args.channel_mu, args.device,
                                        solver_max_step=args.solver_max_step)
        belief = model.initial_belief()                         # birth: development activity is not carried over
    optimizer = make_medium_optimizer(model, lr=args.lr,
                                      saved_state=saved_optimizer if saved else None,
                                      fused=args.fused_optimizer)
    if saved:
        del saved_optimizer
    health = None if args.no_health else MediumHealthAuditor(model, window_tokens=256, block_tokens=32)
    capture = None
    if args.execution == 'graph':
        sample = torch.from_numpy(np.array(train[:args.chunk_tokens + 1], dtype=np.int64)).to(args.device)[None]
        capture = CapturedPlasticChunk(model, sample[:, :-1], sample[:, 1:], belief,
                                       event_duration=args.event_duration,
                                       loss_scale=args.chunk_tokens / args.tokens,
                                       health_capture=None if args.no_health else ChunkHealthCapture(model, args.chunk_tokens),
                                       activation_checkpointing=args.activation_checkpointing, compile_event=args.compile_event,
                                       checkpoint_granularity=args.checkpoint_granularity)
    learner = ActiveMediumTrainer(model, optimizer, belief, carry_token=int(train[0]),
                                  event_duration=args.event_duration, chunk_tokens=args.chunk_tokens,
                                  tokens_per_update=args.tokens, captured=capture, health=health,
                                  activation_checkpointing=args.activation_checkpointing, compile_event=args.compile_event,
                                  optimizer_state_offload=args.optimizer_state_offload,
                                  checkpoint_granularity=args.checkpoint_granularity)
    learner.structure_noise = args.structure_noise
    learner.structural_gravity = args.structural_gravity
    learner.structural_gravity_screening = args.structural_gravity_screening
    learner.structural_gravity_diffusion = args.structural_gravity_diffusion
    learner.channel_growth = args.channel_growth
    learner.freeze_structure = args.freeze == 'interface+medium'
    learner.skip_unstable_structure = args.freeze != 'none'
    learner.channel_mu = args.channel_mu
    step, cursor, val_cursor, next_eval, best_gain = 0, 1, 0, args.validate_every_tokens, -float('inf')
    config = {'constructor': constructor, 'parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'active_graph_parameters': sum(p.numel() for _, p in model.learning_named_parameters()),
              'pretrained_vocabulary': vocabulary_initialization,
              'prenatal_development': None if prenatal is None else {'windows': len(prenatal), 'first': prenatal[0],
                                                                    'last': prenatal[-1]},
              'unified_riverbed': {'soil_absorption': args.soil_absorption, 'channel_growth': args.channel_growth,
                                   'channel_mu': args.channel_mu, 'prenatal_windows': args.prenatal_windows,
                                   'port_layout': args.port_layout, 'write_split': args.write_split,
                                   'freeze': args.freeze},
              'fresh_training_tokens': args.steps * args.tokens, 'training_groups': args.steps,
              'compile_event': args.compile_event,
              'activation_checkpointing': args.activation_checkpointing,
              'checkpoint_granularity': args.checkpoint_granularity,
              'optimizer_state_offload': args.optimizer_state_offload,
              'material_initialization': args.material_initialization,
              'material_field_std': args.material_field_std,
              'bptt_chunk_tokens': args.chunk_tokens, 'tokens_per_update': args.tokens,
              'event_duration': args.event_duration, 'lr': args.lr, 'min_lr': args.min_lr,
              'warmup': args.warmup, 'decay': args.decay, 'seed': args.seed,
              'execution': args.execution, 'calibrate_only': args.calibrate_only,
              'initialization': ('projected pretrained word table; train-only unigram decoder bias'
                                 if vocabulary_initialization else
                                 'random learned embedding/decoder weights; train-only unigram decoder bias'),
              'prior_counts_sha256': prior_hash,
              'data_sha256': {name: content_hash(args.data_dir / name)
                              for name in ('train.npy', 'validation.npy')},
              'manifest_sha256': hashlib.sha256((args.data_dir / 'manifest.json').read_bytes()).hexdigest(),
              'fourth_pillar': {'protocol': 'persistent_medium_health_v1', 'window_tokens': 256,
                                'block_tokens': 32, 'conditional_response_directions': 2},
              'evaluation': {'protocol': 'live prequential A->B->A', 'reset': False, 'learning_active': True,
                             'b_targets': args.eval_tokens, 'block_tokens': args.eval_block_tokens, 'a_replay_targets': args.replay_tokens - 1,
                             'scored_return_bridge_targets': 1,
                             'validate_every_tokens': args.validate_every_tokens},
              'stopping': 'fresh budget, STOP file, nonfinite state/gradient/loss, dedicated GPU cap',
              'source_hashes': {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (
                  'scripts/ib/train_medium_active_stream.py',
                  'information_boltzmann/runtime/active_medium_training.py',
                  'information_boltzmann/runtime/medium_health.py',
                  'information_boltzmann/runtime/lifelong_evaluation.py',
                  'information_boltzmann/runtime/training.py',
                  'information_boltzmann/runtime/optimization.py',
                  'information_boltzmann/runtime/checkpoint_receipt.py',
                  'information_boltzmann/runtime/optimizer_storage.py',
                  'information_boltzmann/core/state_checkpoint.py',
                  'information_boltzmann/core/plastic_medium.py',
                  'information_boltzmann/core/segment_graph.py',
                  'information_boltzmann/core/shared_credit.py',
                  'information_boltzmann/core/deferred_linear_credit.py',
                  'information_boltzmann/core/gradient_norms.py',
                  'information_boltzmann/core/plastic_ports.py',
                  'information_boltzmann/core/conductance_response.py',
                  'information_boltzmann/core/conduction_plasticity.py',
                  'information_boltzmann/core/short_term_plasticity.py',
                  'information_boltzmann/core/local_ports.py',
                  'information_boltzmann/core/mode_port.py',
                  'information_boltzmann/core/torus3d.py',
                  'information_boltzmann/core/readout_probes.py')}}
    if args.structure_config is not None:
        config['structural_learning'] = {
            'configuration': constructor['structure_options'],
            'dual_learning_rate': args.structure_dual_lr,
            'likelihood': 'one next-token task likelihood; writer auxiliary has local parameter credit only',
            'window': 'one fixed noise sample and prior per optimizer update',
            'maintenance': 'installed volume-weighted active capacity; idle uncharged',
            'fast_utilization': 'sigmoid(conduction) * STP_resource * STP_utilization <= 1',
            'continuation': 'posterior/prior/sample/ledger in model; pending gradients and physical start in learner'}
        path = 'information_boltzmann/core/structural_posterior.py'
        config['source_hashes'][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if model.medium.structural_posterior.capacity_growth is not None:
            config['structural_learning']['capacity_update'] = {
                'rule': 'volume-weighted simplex pullback natural step in continuous Fourier coefficients',
                'credit': 'unclipped accumulated task plus existing structural objective gradient',
                'prior_curvature': '1/(window_targets * latched prior variance); no additional loss',
                'variance': 'frozen posterior log_std; original fixed-noise and OU continuation',
                'ownership': 'mean excluded from Adam, independent post-window update',
                'configuration': constructor['structure_options']['capacity_growth']}
            path = 'information_boltzmann/core/capacity_growth.py'
            config['source_hashes'][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if args.hopf_recomposition:
        config['hopf_recomposition'] = model.hopf_pathway.descriptions()
        config['hopf_recomposition']['parameters'] = sum(p.numel() for p in model.hopf_pathway.parameters())
        for path in ('information_boltzmann/core/hopf_recomposition.py',
                     'information_boltzmann/core/medium_junction.py'):
            config['source_hashes'][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if args.fused_optimizer:
        config['fused_optimizer'] = True
        path = 'information_boltzmann/runtime/optimization.py'
        config['source_hashes'][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if args.anisotropic_transport:
        config['transport_law'] = {
            'factor': ('v_ref * diag(simplex_active_capacity) @ unit_direction_rows; bounded fast utilization'
                       if args.structure_config is not None else
                       'diag(positive_edge_speed) @ unit_lower_triangular(material)'),
            'tensor': 'A = B @ B.T',
            'energy': 'unchanged Euclidean field-plus-flux energy',
            'solver': 'exact skew edge rotations; first-order splitting',
            'initialization': 'zero shear exactly recovers axis transport',
            'extra_parameters': sum(p.numel() for p in model.medium.transport_shear.parameters())}
    if args.transport_capacity_budget is not None:
        path = 'information_boltzmann/core/structural_resource.py'
        config['source_hashes'][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if args.solver_max_step is not None:
        from information_boltzmann.core.intrinsic_time import factor_characteristic_time
        with torch.no_grad():
            factor = model.medium.current_transport_factor(belief.medium)
            scales = factor_characteristic_time(factor, cell_spacing=[1 / n for n in args.shape])
        config['physical_interval'] = {
            'clock': 'pre-observation compact-read history' if model.intrinsic_time is not None else 'fixed duration',
            'initial_duration': args.intrinsic_time_reference or args.event_duration,
            'reference_policy': 'explicit calibrated constant; never recomputed as inverse learned speed',
            'cell_rotation_time_estimate': float(scales.cell_crossing_time),
            'domain_traversal_time_estimate': float(scales.torus_crossing_time),
            'solver_max_step': args.solver_max_step, 'observer_max_step': args.observer_max_step,
            'max_evolution_steps': args.max_evolution_steps,
            'credit': 'all physical microsteps; cross-event truncation still explicit',
            'activation_storage': (
                ('structural event first-order VJP; local event tape; full credit'
                 if args.structure_config is not None and args.checkpoint_granularity == 'event' else
                 'structural event first-order VJP; observer and physical-step recomputation; full credit'
                 if args.structure_config is not None else
                 'event and observer-interval recomputation; no detach within event')
                if args.activation_checkpointing else 'full graph storage'),
            'decode': 'once after the chosen physical interval',
            'execution': 'eager scheduling with fused medium; no fixed graph substitution'}
        if saved:
            # These are birth measurements, not mutable policy inputs. The
            # restored material has learned since birth; recomputing its scales
            # must not make an otherwise exact continuation look incompatible.
            birth_interval = saved['config'].get('physical_interval', {})
            for name in ('cell_rotation_time_estimate', 'domain_traversal_time_estimate'):
                if name in birth_interval:
                    config['physical_interval'][name] = birth_interval[name]
        path = 'information_boltzmann/core/intrinsic_time.py'
        config['source_hashes'][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if args.temporal_read:
        config['temporal_read'] = {
            'sampling': ('one held endpoint per observer interval'
                         if args.solver_max_step is not None else
                         'one held endpoint per physical advance call'),
            'history_frame': 'persistent sensor identity along learned probe trajectory',
            'initial_half_life_events': args.temporal_half_life_events,
            'initial_frequency_fractions': args.temporal_frequency_fractions,
            'initialization_interval': args.temporal_physical_reference or args.intrinsic_time_reference or args.event_duration,
            'frequency_time_reference': constructor['temporal_time_reference'],
            'coverage_status': 'diagnostic initialization, not an optimal physical timescale',
            'rates_and_frequencies_learned': True,
            'port_arithmetic': ('compiled CUDA ports; eager physical-interval scheduling'
                                if args.device == 'cuda' and args.solver_max_step is not None else
                                'compiled CUDA ports' if args.device == 'cuda' else 'native CPU ports')}
        path = 'information_boltzmann/core/temporal_probes.py'
        config['source_hashes'][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if saved:
        expected = dict(saved['config'])
        # Historical execution always used the nested policy.
        expected.setdefault('checkpoint_granularity', 'nested')
        actual = dict(config)
        expected.pop('entrypoint_continuation', None)
        expected.pop('execution_continuation', None)
        expected.pop('default_preserving_continuation', None)
        expected.pop('capacity_growth_continuation', None)
        expected.pop('hopf_branch_continuation', None)
        if args.adopt_hopf_branch:
            prefix = 'medium.hopf_pathway.'
            config['hopf_branch_continuation'] = prepare_hopf_continuation(
                expected, actual, source_checkpoint=args.resume,
                parameter_names=[prefix + name for name, _ in model.hopf_pathway.named_parameters()],
                state_names=[prefix + name for name in model.hopf_pathway.state_dict()],
                added_count=sum(p.numel() for p in model.hopf_pathway.parameters()))
        if args.adopt_capacity_growth:
            # Narrow, recorded policy change. All unrelated settings must still
            # compare exactly: data, evaluation, clock, budget and other credit.
            expected['constructor'] = dict(expected['constructor'])
            old_structure = dict(expected['constructor']['structure_options'])
            new_structure = dict(actual['constructor']['structure_options'])
            growth_options = new_structure.pop('capacity_growth', None)
            if growth_options is None or old_structure != new_structure:
                raise ValueError('Capacity migration must preserve all inherited structure scales')
            expected['constructor']['structure_options'] = constructor['structure_options']
            frozen_count = model.medium.structural_posterior.log_std.numel()
            if expected['parameters'] != actual['parameters']:
                raise ValueError('Capacity migration must preserve total model parameter count')
            for name in ('trainable_parameters', 'active_graph_parameters'):
                if expected[name] - actual[name] != frozen_count:
                    raise ValueError(f'Capacity migration must freeze only log_std: {name}')
                expected[name] = actual[name]
            old_learning = dict(expected['structural_learning'])
            new_learning = dict(actual['structural_learning'])
            new_learning.pop('capacity_update', None)
            new_learning['configuration'] = old_structure
            if old_learning != new_learning:
                raise ValueError('Capacity migration changed another structural objective/ledger')
            expected['structural_learning'] = actual['structural_learning']
            old_sources, new_sources = expected['source_hashes'], actual['source_hashes']
            allowed = {'scripts/ib/train_medium_active_stream.py',
                       'information_boltzmann/runtime/checkpoint_receipt.py',
                       'information_boltzmann/core/structural_posterior.py',
                       'information_boltzmann/core/capacity_growth.py',
                       'information_boltzmann/runtime/active_medium_training.py',
                       'information_boltzmann/runtime/optimization.py'}
            changed = {k for k in old_sources.keys() | new_sources.keys()
                       if old_sources.get(k) != new_sources.get(k)}
            if changed - allowed:
                raise ValueError(f'Unaudited unrelated capacity migration sources: {changed - allowed}')
            expected['source_hashes'] = new_sources
            config['capacity_growth_continuation'] = {
                'source_checkpoint': str(args.resume),
                'before': old_sources, 'after': new_sources,
                'discarded_optimizer_entries': ['medium.structural_posterior.mean',
                                                 'medium.structural_posterior.log_std'],
                'preserved': 'all physical belief, OU prior, RNG, data cursors, cadence and other Adam moments'}
        if args.allow_default_preserving_extension:
            old_sources = expected.pop('source_hashes')
            new_sources = actual.pop('source_hashes')
            allowed = {'scripts/ib/train_medium_active_stream.py',
                       'information_boltzmann/runtime/checkpoint_receipt.py',
                       'information_boltzmann/core/plastic_medium.py',
                       'information_boltzmann/core/plastic_ports.py',
                       'information_boltzmann/runtime/active_medium_training.py',
                       'information_boltzmann/runtime/medium_health.py'}
            if saved['config']['constructor'].get('transport_capacity_budget') is not None:
                # Enabled-capacity individuals admit only telemetry/entrypoint
                # extensions; the constitutive law and budget must stay fixed.
                allowed = {'scripts/ib/train_medium_active_stream.py',
                           'information_boltzmann/runtime/checkpoint_receipt.py',
                           'information_boltzmann/runtime/medium_health.py',
                        'information_boltzmann/runtime/lifelong_evaluation.py'}
            changed = {key for key in old_sources.keys() | new_sources.keys()
                       if old_sources.get(key) != new_sources.get(key)}
            if changed - allowed:
                raise ValueError('Unaudited source change during telemetry continuation')
            config['default_preserving_continuation'] = {
                'reason': 'Audited telemetry extension; unchanged physical/learning config and capacity law',
                'before': old_sources, 'after': new_sources}
        if args.allow_execution_change:
            old_sources = expected.pop('source_hashes')
            new_sources = actual.pop('source_hashes')
            allowed = {'scripts/ib/train_medium_active_stream.py',
                       'information_boltzmann/runtime/checkpoint_receipt.py',
                       'information_boltzmann/runtime/training.py',
                       'information_boltzmann/runtime/active_medium_training.py',
                       'information_boltzmann/runtime/optimizer_storage.py',
                       'information_boltzmann/runtime/medium_health.py',
                       'information_boltzmann/core/readout_probes.py',
                       'information_boltzmann/core/plastic_ports.py'}
            allowed.update({'information_boltzmann/core/plastic_medium.py',
                            'information_boltzmann/core/segment_graph.py',
                            'information_boltzmann/core/shared_credit.py',
                            'information_boltzmann/core/deferred_linear_credit.py',
                            'information_boltzmann/runtime/optimization.py',
                            'information_boltzmann/runtime/medium_health.py'})
            changed = {key for key in old_sources.keys() | new_sources.keys()
                       if old_sources.get(key) != new_sources.get(key)}
            if changed - allowed:
                raise ValueError('Execution continuation changes unaudited model/evaluation sources')
            expected.pop('compile_event', None)
            actual.pop('compile_event', None)
            old_fused_optimizer = expected.pop('fused_optimizer', False)
            new_fused_optimizer = actual.pop('fused_optimizer', False)
            expected['constructor'] = dict(expected['constructor'])
            actual['constructor'] = dict(actual['constructor'])
            old_read_keys = expected['constructor'].pop('read_key_execution', 'dense')
            new_read_keys = actual['constructor'].pop('read_key_execution', 'dense')
            old_segments = expected['constructor'].pop('medium_segment_graph', False)
            new_segments = actual['constructor'].pop('medium_segment_graph', False)
            old_shared_credit = expected['constructor'].pop('shared_credit_execution', False)
            new_shared_credit = actual['constructor'].pop('shared_credit_execution', False)
            old_deferred_credit = expected['constructor'].pop('deferred_writer_credit', False)
            new_deferred_credit = actual['constructor'].pop('deferred_writer_credit', False)
            old_history_offload = expected['constructor'].pop('credit_history_offload', False)
            new_history_offload = actual['constructor'].pop('credit_history_offload', False)
            old_port_execution = expected['constructor'].pop('port_execution', 'native')
            new_port_execution = actual['constructor'].pop('port_execution', 'native')
            old_port_metadata = new_port_metadata = None
            if 'temporal_read' in expected and 'temporal_read' in actual:
                expected['temporal_read'] = dict(expected['temporal_read'])
                actual['temporal_read'] = dict(actual['temporal_read'])
                old_port_metadata = expected['temporal_read'].pop('port_arithmetic', None)
                new_port_metadata = actual['temporal_read'].pop('port_arithmetic', None)
            config['execution_continuation'] = {'reason': 'audited execution-only workspace and bounded forward CUDA segments with fused RHS; physical/learning policy unchanged',
                'before': old_sources, 'after': new_sources,
                'read_key_execution_before': old_read_keys,
                'read_key_execution_after': new_read_keys,
                'medium_segment_graph_before': old_segments,
                'medium_segment_graph_after': new_segments,
                'shared_credit_before': old_shared_credit,
                'shared_credit_after': new_shared_credit,
                'deferred_writer_credit_before': old_deferred_credit,
                'deferred_writer_credit_after': new_deferred_credit,
                'credit_history_offload_before': old_history_offload,
                'credit_history_offload_after': new_history_offload,
                'fused_optimizer_before': old_fused_optimizer,
                'fused_optimizer_after': new_fused_optimizer,
                'port_execution_before': old_port_execution,
                'port_execution_after': new_port_execution,
                'port_arithmetic_metadata_before': old_port_metadata,
                'port_arithmetic_metadata_after': new_port_metadata,
                'compile_event_before': saved['config'].get('compile_event', False),
                'compile_event_after': args.compile_event}

        if args.allow_entrypoint_change:
            entry = 'scripts/ib/train_medium_active_stream.py'
            receipt_source = 'information_boltzmann/runtime/checkpoint_receipt.py'
            expected['source_hashes'] = dict(expected['source_hashes'])
            actual['source_hashes'] = dict(actual['source_hashes'])
            old_hash = expected['source_hashes'].pop(entry)
            new_hash = actual['source_hashes'].pop(entry)
            old_receipt_hash = expected['source_hashes'].pop(receipt_source, None)
            new_receipt_hash = actual['source_hashes'].pop(receipt_source)
            config['entrypoint_continuation'] = {'before': old_hash, 'after': new_hash,
                'checkpoint_helper_before': old_receipt_hash,
                'checkpoint_helper_after': new_receipt_hash,
                'reason': 'Operational telemetry, actual complete-save receipts and restart provenance; physical/learning settings unchanged'}
        if actual != expected:
            def different_paths(left, right, prefix=''):
                if isinstance(left, dict) and isinstance(right, dict):
                    result = []
                    for name in sorted(left.keys() | right.keys()):
                        key = f'{prefix}.{name}' if prefix else name
                        if name not in left or name not in right:
                            result.append(key)
                        else:
                            result.extend(different_paths(left[name], right[name], key))
                    return result
                return [] if left == right else [prefix]
            raise ValueError('Resume must preserve source, budget, cadence and evaluation policy; '
                             f'differing fields: {different_paths(actual, expected)}')
        learner.load_state_dict(saved['learner'])
        if args.adopt_capacity_growth:
            # Completed old gradients carry no credit under the new owner.
            model.medium.structural_posterior.mean.grad = None
            model.medium.structural_posterior.log_std.grad = None
        step, cursor, val_cursor, next_eval, best_gain = (
            saved[key] for key in ('step', 'cursor', 'val_cursor', 'next_eval', 'best_gain'))
        torch.set_rng_state(saved['cpu_rng'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'])
        # GPU model, optimizer and learner now own their restored state. Keeping
        # the full CPU payload during Inductor compilation can exhaust Windows'
        # commit limit even when physical RAM appears available.
        del saved
        import gc
        gc.collect()
    if args.check_resume_only:
        print(json.dumps({'status': 'continuation_validated', 'step': step,
                          'cursor': cursor, 'val_cursor': val_cursor,
                          'next_eval': next_eval, 'optimizer_updates': learner.optimizer_updates,
                          'pending_events': learner.pending, 'targets_consumed': 0,
                          'source_checkpoint_position': source_position,
                          'retained_source_log_gap': source_gap,
                          'production_files_written': False}), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    attempt_id = uuid.uuid4().hex
    def live_position():
        return {'step': step, 'cursor': cursor, 'val_cursor': val_cursor,
                'next_eval': next_eval, 'fresh_training_tokens': cursor - 1,
                'events': learner.events, 'optimizer_updates': learner.optimizer_updates,
                'pending_gradient_events': learner.pending}
    restart = append_attempt_journal(args.output, attempt_id=attempt_id,
        position=live_position(), source_checkpoint=args.resume,
        source_receipt=source_receipt, source_gap=source_gap)
    atomic_json(args.output / 'config.json', config)
    stop = False
    def request_stop(*unused):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    progress = {'status': 'initializing', 'pid': os.getpid(), 'attempt_id': attempt_id,
                'restart': restart}
    last_saved_receipt = (source_receipt if args.resume is not None
                          and args.resume.resolve() == (args.output / 'last.pt').resolve()
                          else None)
    spatial_published = False
    spatial_published_at = 0.0
    def publish(**values):
        nonlocal spatial_published, spatial_published_at
        if values.get('status') in ('completed', 'stopped'):
            require_saved_position(last_saved_receipt, live_position())
        if args.device == 'cuda' and values.get('status') == 'training':
            values['memory'] = memory_guard()
        if getattr(model, 'hopf_recomposition', False) and getattr(model, 'hopf_pathway', None) is not None:
            junction = model.hopf_pathway
            values['hopf_junction'] = {**junction.descriptions(),
                'gate_weight_norm': float(junction.gate.weight.detach().norm()),
                'gate_bias_norm': float(junction.gate.bias.detach().norm()),
                'compiled_route_mask': junction.route_mask.detach().cpu().tolist()}
        progress.update(values, step=step, fresh_training_tokens=cursor - 1,
                        train_cursor=cursor, validation_cursor=val_cursor,
                        next_evaluation_fresh_tokens=next_eval, **learner.summary())
        progress['last_saved_checkpoint'] = last_saved_receipt
        saved_position = {} if last_saved_receipt is None else last_saved_receipt['position']
        progress.update(last_saved_step=saved_position.get('step'),
                        last_saved_fresh_training_tokens=saved_position.get('fresh_training_tokens'),
                        last_saved_train_cursor=saved_position.get('cursor'),
                        last_saved_validation_cursor=saved_position.get('val_cursor'),
                        last_saved_optimizer_updates=saved_position.get('optimizer_updates'),
                        checkpoint_matches_live_state=bool(last_saved_receipt) and all(
                            saved_position.get(key) == value for key, value in live_position().items()))
        if args.medium_segment_graph:
            progress['execution_detail'] = {
                'mode': 'bounded_forward_segments_and_fused_rhs',
                'backward': 'complete event VJP recomputation',
                'credit_execution': ('shared_primal_two_full_cotangent_lanes'
                                     if model.shared_credit_execution else 'separate_auxiliary_replay'),
                'linear_weight_credit': ('exact_window_factor_gemm'
                                         if model.deferred_writer_credit else 'per_event_outer_product'),
                'credit_history_storage': ('host_exact' if model.credit_history_offload else 'gpu'),
                'vram_limit_mib': args.vram_limit_mib,
                'segments': {name: {'graphs': len(segment.records), 'replays': segment.replays}
                             for name in ('advance', 'rhs')
                             if (segment := getattr(model.medium, f'_segment_{name}', None)) is not None}}
        atomic_json(args.output / 'progress.json', progress)
        if values.get('status') == 'training' and (step % 10 == 0 or not spatial_published
                or time.monotonic() - spatial_published_at >= 30):
            spatial = spatial_medium_snapshot(model, learner.belief)
            spatial.update(step=step, fresh_training_tokens=cursor - 1, wall_time=time.time())
            atomic_json(args.output / 'spatial.json', spatial)
            spatial_published = True
            spatial_published_at = time.monotonic()
    def save(name):
        nonlocal last_saved_receipt
        if args.calibrate_only:
            return
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'belief': pack_belief(learner.belief), 'learner': learner.state_dict(),
                   'step': step, 'cursor': cursor, 'val_cursor': val_cursor,
                   'next_eval': next_eval, 'best_gain': best_gain, 'config': config,
                   'cpu_rng': torch.get_rng_state(),
                   'cuda_rng': torch.cuda.get_rng_state() if args.device == 'cuda' else None}
        receipt = save_complete_checkpoint(args.output / name, payload, attempt_id=attempt_id)
        if name == 'last.pt':
            last_saved_receipt = receipt
        return receipt
    def consume(values, phase):
        targets = torch.from_numpy(np.array(values, dtype=np.int64)).to(args.device)
        if phase == 'train_first_pass':
            return learner.consume(targets, phase=phase, prior_nll=prior)
        # The actual learner already uses these same 32-event windows. Publish
        # between them so a live context-change assessment is visibly advancing.
        rows = []
        phase_name = phase if isinstance(phase, str) else 'revisit_A'
        for left in range(0, len(targets), args.chunk_tokens):
            right = min(left + args.chunk_tokens, len(targets))
            labels = phase if isinstance(phase, str) else phase[left:right]
            rows.extend(learner.consume(targets[left:right], phase=labels, prior_nll=prior))
            publish(status='training', evaluation_phase=phase_name,
                    evaluation_completed_events=right, evaluation_total_events=len(targets))
        return rows
    def memory_guard():
        if args.device == 'cuda':
            torch.cuda.synchronize()
            memory = total_gpu_memory()
            if memory['used_bytes'] / 2**20 > args.vram_limit_mib:
                raise MemoryError('Dedicated VRAM budget exceeded')
            return {'total_dedicated_mib': memory['used_bytes'] / 2**20,
                    'allocated_mib': torch.cuda.memory_allocated() / 2**20,
                    'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                    'reserved_mib': torch.cuda.memory_reserved() / 2**20,
                    'inactive_split_mib': torch.cuda.memory_stats().get(
                        'inactive_split_bytes.all.current', 0) / 2**20}
        return None
    try:
        # Persist the complete birth before first-event compilation can be
        # interrupted. This also makes a reboot at zero updates resumable.
        if not args.calibrate_only and (not args.resume or args.adopt_capacity_growth or args.adopt_hopf_branch):
            save('last.pt')
        publish(status='calibrating' if args.calibrate_only else 'training', memory=memory_guard())
        while step < args.steps and not stop and not (args.output / 'STOP').exists():
            rate = args.lr if args.calibrate_only else learning_rate(
                step + 1, steps=args.steps, warmup=args.warmup, decay=args.decay,
                peak=args.lr, floor=args.min_lr)
            for group in optimizer.param_groups:
                group['lr'] = rate
            if args.device == 'cuda':
                torch.cuda.synchronize()
            started = time.perf_counter()
            source_start = cursor
            scored = consume(train[cursor:cursor + args.tokens], 'train_first_pass')
            cursor += args.tokens
            step += 1
            if args.device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            row = {'step': step, 'fresh_training_tokens': cursor - 1,
                   'attempt_id': attempt_id, 'record_id': uuid.uuid4().hex,
                   'source_split': 'train', 'source_sha256': config['data_sha256']['train.npy'],
                   'source_start': source_start, 'source_end': cursor,
                   'events': learner.events, 'optimizer_updates': learner.optimizer_updates,
                   'train_prequential_nll': sum(x[2] for x in scored) / len(scored),
                   'fixed_unigram_nll': sum(x[3] for x in scored) / len(scored),
                   'seconds_per_group': elapsed, 'tokens_per_second': args.tokens / elapsed,
                   'gradient_norm_before_clip': learner.last_gradient_norm,
                   'lr': rate, 'physical_time': float(learner.belief.medium.elapsed[0]),
                   **field_energy_statistics(learner.belief.medium.field)}
            if model.temporal_readout is not None:
                temporal = learner.belief.temporal
                bank = model.temporal_readout.bank
                row['temporal_read'] = {
                    'physical_time': float(temporal.elapsed[0]),
                    'rates': bank.log_rate.detach().exp().cpu().tolist(),
                    'frequencies': bank.physical_frequency.detach().cpu().tolist(),
                    'dimensionless_frequencies': bank.frequency.detach().cpu().tolist(),
                    'real_power_per_mode': temporal.value.real.square().mean((0, 1, 3)).cpu().tolist(),
                    'imag_power_per_mode': temporal.value.imag.square().mean((0, 1, 3)).cpu().tolist()}
            if (cursor - 1 >= next_eval and
                    len(learner.recent) >= args.replay_tokens - 1):
                a = list(learner.recent)[-(args.replay_tokens - 1):]
                a_tokens = [a[0][0]] + [x[1] for x in a]
                if any(x[4] != 'train_first_pass' for x in a):
                    raise ValueError('Revisit must pair actual first-pass A scores')
                if val_cursor + args.eval_tokens > len(validation):
                    raise ValueError('Fresh B corpus exhausted')
                # Preserve the exact pre-evaluation individual if an optional
                # diagnostic or evaluation operation raises before completion.
                save('last.pt')
                publish(status='training', **row, evaluation_phase='fresh_B',
                        evaluation_completed_events=0, evaluation_total_events=args.eval_tokens)
                response = conditional_state_response(model, learner.belief,
                    torch.tensor([learner.carry_token], device=args.device),
                    event_duration=args.event_duration, directions=2)
                b = consume(validation[val_cursor:val_cursor + args.eval_tokens], 'fresh_B')
                val_cursor += args.eval_tokens
                b_health = None if health is None else health.summary()
                # One real bridge +127 real replay targets =128 events. Execute
                # across phase labels in the SAME 32-event tapes; no padding,
                # discarded gradients, state reset or persistent 31+1 fallback.
                returned = consume(a_tokens, ['revisit_bridge'] + ['revisit_A'] * len(a))
                bridge, revisit = returned[:1], returned[1:]
                evaluation = {'fresh_training_tokens': cursor - 1, 'B_curve': [x[2] for x in b],
                              'attempt_id': attempt_id, 'record_id': uuid.uuid4().hex,
                              'source_split': 'validation',
                              'source_sha256': config['data_sha256']['validation.npy'],
                              'source_start': val_cursor - args.eval_tokens, 'source_end': val_cursor,
                              'fresh_validation_cursor': val_cursor,
                              'B_source_start': val_cursor - args.eval_tokens,
                              'B_prior_curve': [x[3] for x in b], 'B_nll': sum(x[2] for x in b) / len(b),
                              'B_prior_nll': sum(x[3] for x in b) / len(b),
                              'conditional_state_response': response, 'B_health': b_health,
                              'bridge_nll': bridge[0][2], 'learning_active': True, 'reset': False,
                              'shock': recovery_summary([x[2] for x in b], block_tokens=args.eval_block_tokens, hold_blocks=2,
                                  reference_nll=[x[3] for x in b]),
                              'savings': savings_summary([x[2] for x in a], [x[2] for x in revisit],
                                  intervening_events=len(b) + 1, block_tokens=args.eval_block_tokens, hold_blocks=2),
                              **learner.summary()}
                with (args.output / 'lifelong_evaluation.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(evaluation, allow_nan=False) + '\n')
                next_eval += args.validate_every_tokens
                pooled_gain = learner.summary()['phases']['fresh_B']['gain']
                if pooled_gain > best_gain:
                    best_gain = pooled_gain
                    save('best.pt')
                # An appended active evaluation must have a resumable complete
                # individual after it, even before the periodic fresh save.
                save('last.pt')
                print(f"ACTIVE {cursor-1}: B={evaluation['B_nll']:.4f} paired prior={evaluation['B_prior_nll']:.4f} pooled gain={pooled_gain:+.4f}", flush=True)
            with (args.output / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
            publish(status='calibrating' if args.calibrate_only else 'training', **row,
                    evaluation_phase=None, evaluation_completed_events=0, evaluation_total_events=0,
                    memory=memory_guard() if args.calibrate_only or step <= 5 or step % 25 == 0 else progress.get('memory'))
            if step == 1 or step % args.save_every == 0:
                save('last.pt')
            if step == 1 or step % 50 == 0:
                print(f"MEDIUM {cursor-1} tokens, NLL={row['train_prequential_nll']:.4f} {row['tokens_per_second']:.1f} token/s", flush=True)
        save('last.pt')
        publish(status=('calibration_completed' if args.calibrate_only else
                        'completed' if step == args.steps else 'stopped'))
    except Exception as error:
        failed_save = None
        if isinstance(error, (PermissionError, OSError)) and not isinstance(error, CheckpointWriteError):
            # A monitoring error occurs after a complete valid group. Preserve
            # it rather than forcing recovery from an older periodic save.
            try:
                save('last.pt')
            except Exception as save_error:
                failed_save = f'{type(save_error).__name__}: {save_error}'
        try:
            publish(status='failed', error=f'{type(error).__name__}: {error}',
                    emergency_checkpoint_error=failed_save)
        except OSError as progress_error:
            print(f'Unable to publish failed status: {progress_error}; original error: {error}', flush=True)
        raise


if __name__ == '__main__':
    main()
