"""Native birth of the complete navigation organism, without a weight ancestor.

Birth starts empty physical states and evidence once. Afterwards the existing
complete-state continuation protocol retains S/P, Adam, history and world state.
The byte lookup port and the deep experience port are deliberately independent.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]


def validate_spec(spec):
    if spec.get('schema') != 1:
        raise ValueError('unknown native body schema')
    b = spec['backbone']
    if b['vocab_size'] != 256 or b['d'] % b['heads']:
        raise ValueError('native byte body needs divisible head dimensions')
    if not 0 <= spec['byte_table_layer'] < spec['experience_layer'] < b['layers']:
        raise ValueError('experience must receive deeper content than byte lookup')
    if spec['latent_dim'] < 8 or spec['memory_topk'] < 8:
        raise ValueError('complete typed relation graph needs eight evidence slots')
    if spec['memory_capacity'] < spec['memory_topk']:
        raise ValueError('evidence budget exceeds memory')
    if spec['response_bytes'] + 32 > spec['long_horizon'] or spec['holdout_bytes'] > spec['long_horizon']:
        raise ValueError('initial observation must fit the declared long horizon')
    if spec['expression_cap'] != 128 or not 0 < spec['weight_std'] < 1:
        raise ValueError('undeclared expression or initialization contract')


def component_config(spec, manifest):
    """Algorithm configuration only: no old individual, moments or evidence."""
    validate_spec(spec)
    one = lambda **kw: dict(schema=1, **kw)
    return dict(
        stage='native normally initialized complete continuous language body',
        native_body=copy.deepcopy(spec), seed=spec['seed'], mode='real',
        source_raw_bounds=[8000000, 56000000], topic_window=4096,
        natural_text_corpus=copy.deepcopy(manifest), world_source='text',
        text_navigation_policy='stable_context_v1',
        wood_memory_features=14, wood_evidence_schema=2,
        wood_gain_mode='direct_occurrence', observer_mode='joint_only',
        common_settlement='world_stream',
        internal_rollout=one(mode='real'),
        bfss_dynamics=one(enabled=True, dt=.01, frequency=.8, coupling=.005, steps=1),
        momentum_wood=one(), relation_wood=one(ridge=.1, scale=.1),
        question_context=one(), shadow_preview=one(samples=2, mode='real', candidate_kind='centered'),
        mother_boundary=one(), persistent_question=one(), bound_question_shadow=one(),
        joint_question_update=one(), align_question_boundary=one(), mother_shadow=one(),
        question_parent_carry=one(), causal_program=one(),
        joint_world_cycle=one(proposal_budget=spec['proposal_budget']),
        generative_relations=one(), shared_relation_basis=one(features=[4, 5]),
        joint_relation_update=one(implementation_revision=4), content_question=one(),
        variable_expression=one(cap=128), shared_unknowns=one(),
        gain_units=one(critic_scale=100., wood_scale=10., raw_scale=1.),
        checkpoint_compression_level=0,
        birth=dict(kind='new individual', parent_checkpoint=None, inherited_weights=False,
                   inherited_optimizer=False, inherited_evidence=False,
                   continuous_after_birth=True,
                   algorithm_reference='CGB_MASTER_PLAN_20261114.md; complete current graph',
                   boundary_names='mother=responsive world prefix; ambient=following real text; no API'))


def construct_engine(config, world, *, device='cpu', restoring=False,
                     response_bytes=None, holdout_bytes=None):
    """Construct the same shapes/group order at birth and restoration."""
    from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
    from scripts.ib_local.train_gdn_fast_table import TableGDN
    from scripts.ib_local.language_navigation_kanli_v4m import WoodObservationBody
    from scripts.ib_local.language_navigation_kanli_v4b import attach_imagination
    from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
    from scripts.ib_local.run_causal_water_reply import WaterReplyEngine
    spec = config['native_body']
    validate_spec(spec)
    base = TableGDN(GatedDeltaNetLM(**spec['backbone']), True,
                    inject_layer=spec['byte_table_layer']).to(device)
    optimizer = torch.optim.AdamW(base.parameters(), lr=spec['learning_rate'], foreach=True)
    model = attach_imagination(WoodObservationBody(base,
        latent_dim=spec['latent_dim'], capacity=spec['memory_capacity'],
        short=spec['short_horizon'], long=spec['long_horizon'],
        topk=spec['memory_topk'], seed=spec['seed'], max_chunk=spec['long_horizon']))
    model.experience_layer = spec['experience_layer']
    optimizer.add_param_group(dict(params=[p for n,p in model.named_parameters()
        if not n.startswith('body.')], lr=spec['learning_rate']))
    # During restore the loaded table replaces this tiny allocation immediately.
    table = RetrievedCreditTable(orders=(0,), heads=1, slots_log2=1)
    engine = WaterReplyEngine(model, optimizer, model.initial_state(1), table, world,
        int(world.data[0]), np.random.default_rng(spec['seed']), 'active', 32,
        response_bytes or spec['response_bytes'], holdout_bytes or spec['holdout_bytes'], capture=False)
    # The growth constructor already uses the guarded causal chunk replay.
    # Native birth AND restoration need the same runtime; otherwise a long
    # detached audit silently executes the entire 32-layer graph byte by byte.
    engine.policy_replay_backend = 'batched_guarded'
    return engine


@torch.no_grad()
def initialize_native(engine, spec):
    """Nonzero neural matrices; norm/bias/timescale priors retain their meaning.

    A_log and dt_bias parameterize physical time constants, not inherited
    knowledge. Empty S/P, evidence and optimizer are not neural residual gates.
    """
    residual_std = spec['weight_std'] / math.sqrt(2*spec['backbone']['layers'])
    matrices = []
    for name, module in engine.model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = residual_std if name.endswith(('.o_proj', '.mlp.down')) else spec['weight_std']
            nn.init.normal_(module.weight, std=std)
            # Keep meaningful handover/decay/mixing birth biases from their
            # constructors; other biases start at the ordinary zero prior.
            if getattr(module, 'bias', None) is not None and not name.endswith(
                    ('.alpha_proj', '.beta_proj', '.mix', 'handover')):
                nn.init.zeros_(module.bias)
            matrices.append(dict(name=name, shape=list(module.weight.shape), std=std,
                                 rms=float(module.weight.square().mean().sqrt())))
    # Birth revision 1 visits both aliases in module order. Their one shared
    # weight ends with the decoder's recorded draw; never split the parameter.
    assert engine.model.base.decoder.weight is engine.model.base.source.embedding.weight
    assert not engine.optimizer.state
    ids = [id(p) for g in engine.optimizer.param_groups for p in g['params']]
    parameters = list(engine.model.parameters())
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in parameters}:
        raise RuntimeError('every native parameter must appear exactly once in AdamW')
    if not all(p.requires_grad for p in parameters):
        raise RuntimeError('native birth must remain fully trainable')
    return dict(schema=1, residual_std=residual_std, matrices=matrices,
                parameters=sum(p.numel() for p in parameters),
                zero_residual_layers=[], optimizer_states=len(engine.optimizer.state),
                feature_width=engine.model.encoder.in_features,
                latent_width=engine.model.encoder.out_features,
                byte_table_layer=spec['byte_table_layer']+1,
                experience_input_layer=spec['experience_layer']+1,
                reset_policy='empty only at birth; complete continuous state thereafter')


def create(config, world, *, device='cpu', table_storage=None, causal_storage=None, table=None):
    """Assemble, initialize and bind the complete native graph once."""
    from scripts.ib_local.run_causal_water_reply import configure_components
    from scripts.ib_local.navigation_settlement import configure_settlement, configure_reward_memory
    from scripts.ib_local.navigation_read_credit import configure_read_credit
    from scripts.ib_local.navigation_gain_units import configure_gain_units
    spec = config['native_body']
    random.seed(spec['seed']); np.random.seed(spec['seed']); torch.manual_seed(spec['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec['seed'])
    engine = construct_engine(config, world, device=device)
    configure_components(engine, config)
    engine.observer_mode = 'joint_only'
    configure_settlement(engine, 'world_stream')
    configure_reward_memory(engine, 'observed')
    configure_gain_units(engine)
    configure_read_credit(engine.model.memory, 'adam', engine.optimizer.param_groups[0])
    engine.model.memory.consolidation_enabled = True
    engine.model.memory.consolidation_rule = 'utility'
    receipt = initialize_native(engine, spec)
    from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
    if table is None:
        # New bodies allocate the large row array directly in its final tier.
        engine.table = RetrievedCreditTable(storage=table_storage)
        table_storage = None
    else:
        engine.table = table
    if table_storage is not None:
        table_storage = Path(table_storage)
        table_storage.mkdir(parents=True, exist_ok=False)
        for name in ('rows','count','tag','err','epoch','last_write'):
            original = getattr(engine.table, name)
            mapped = np.lib.format.open_memmap(table_storage/(name+'.npy'), mode='w+',
                shape=original.shape, dtype=original.dtype)
            mapped[:] = original; mapped.flush()
            setattr(engine.table, name, mapped)
        engine.table.storage = table_storage
    if causal_storage is not None:
        from scripts.ib_local.causal_archive import attach_archive
        from scripts.ib_local.shared_unknown_bindings import configure_shared_unknowns
        from scripts.ib_local.joint_relation_update import configure_joint_relation_update
        budget = spec['archive']
        attach_archive(engine, causal_storage, episode_capacity=budget['episodes'],
            relation_capacity=budget['relations'], max_bytes=budget['max_bytes'])
        configure_shared_unknowns(engine)
        configure_joint_relation_update(engine, revision=4)
    # No episodic evidence exists yet, and no world event occurred at birth.
    assert int(engine.model.memory.occupied.sum()) == 0
    assert int(engine.model.memory.episodes) == 0
    assert engine.prev is None
    config['neural_parameters'] = receipt['parameters']
    return engine, receipt


def restore_birth_counters(engine, cycle, journal):
    """Keep birth exposure/update accounting through complete-state resumes."""
    names = ('native_optimizer_updates', 'native_external_bytes', 'native_self_bytes')
    present = [hasattr(engine, name) for name in names]
    if all(present):
        return
    if any(present):
        raise ValueError('incomplete native lifetime counters')
    rows = []
    if cycle:
        rows = [json.loads(line) for line in Path(journal).read_text(encoding='utf-8').splitlines() if line]
        rows = [r for r in rows if r.get('cycle', cycle+1) <= cycle]
        if [r['cycle'] for r in rows] != list(range(1, cycle+1)):
            raise ValueError('cannot reconstruct native lifetime accounting from an incomplete journal')
    for name, key in zip(names, ('updates', 'external_bytes', 'self_bytes')):
        setattr(engine, name, sum(int(r[key]) for r in rows))


def record_birth_event(engine, row):
    for name, key in (('native_optimizer_updates','updates'),
                      ('native_external_bytes','external_bytes'), ('native_self_bytes','self_bytes')):
        setattr(engine, name, getattr(engine, name) + int(row[key]))
    row.update(since_birth_updates=engine.native_optimizer_updates,
        since_birth_external_bytes=engine.native_external_bytes,
        since_birth_self_bytes=engine.native_self_bytes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('birth requires a new directory; resume existing bodies through the continuation runner')
    torch.set_num_threads(1)
    if args.device == 'cuda':
        from scripts.ib_local import gdn_fused
        gdn_fused.enable()
    spec = json.loads(args.spec.read_text(encoding='utf-8'))
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    config = component_config(spec, manifest)
    from scripts.ib_local.world_source import decoded_world_bytes
    from scripts.ib_local.language_navigation_kanli_v4h import ReplyTopicWorld
    from scripts.ib_local.natural_text_world import NaturalTextWorld, extended_source
    data_root = ROOT/'data/ib_owt_10000'
    offset = json.loads((data_root/'tokenizer.json').read_text())['offset']
    train = np.load(data_root/'train.npy', mmap_mode='r')
    lo, hi = config['source_raw_bounds']
    prefix = decoded_world_bytes(np.asarray(train[lo:hi]), offset)
    manifest['prefix_bytes'] = len(prefix)
    config['natural_text_corpus'] = copy.deepcopy(manifest)
    source = extended_source(prefix, manifest)
    del train
    world = NaturalTextWorld(ReplyTopicWorld(source, window=4096, seed=spec['seed']),
        {}, source_manifest=manifest, navigation_policy='stable_context_v1')
    args.output.mkdir(parents=True)
    engine, receipt = create(config, world, device=args.device,
        table_storage=args.output/'fast_table_work', causal_storage=args.output/'causal_work')
    identity = hashlib.sha256((json.dumps(spec,sort_keys=True)+str(args.output.resolve())).encode()).hexdigest()[:16]
    config['life_identity'] = dict(name='梦漓', individual_id='mengli-native-'+identity)
    from scripts.ib_local.life_card import branch_id_for, card_config_for, export_live
    branch = branch_id_for(config, args.output/'last.pt.gz')
    config['life_card'] = card_config_for(config, branch)
    config['birth']['source_sha256'] = hashlib.sha256(source).hexdigest()
    config['birth']['source_bytes'] = len(source)
    config['birth']['initialization_receipt'] = 'birth.json'
    from scripts.ib_local.mother_checkpoint import save
    save(engine, args.output, 0, config)
    export_live(engine, checkpoint=args.output/'last.pt.gz', config=config, cycle=0,
        output=config['life_card']['path'], branch_id=branch)
    (args.output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (args.output/'birth.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='native_complete_birth_saved', cycle=0,
        identity=config['life_identity'], parameters=receipt['parameters'],
        checkpoint=str(args.output/'last.pt.gz'),
        cuda_peak_mib=torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else None)))


if __name__ == '__main__':
    main()
