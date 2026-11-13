"""Stream, score, actual-proposal credit and migration contracts, no capability task."""
import copy

import numpy as np
import pytest
import torch

from scripts.ib_local.world_learning_cycle import observed_pieces, settle_sources
from scripts.ib_local.navigation_joint_cycle import migrate_joint_cycle, proposal_log_probability
from test_causal_relation_program import configured


@pytest.mark.parametrize('mother', [0, 1, 2, 97, 480])
def test_public_stream_partition_learns_each_byte_once_and_probes_only_future(mother):
    response = np.arange(480, dtype=np.int64).astype(np.uint8)
    tail = np.arange(480, 960, dtype=np.int64).astype(np.uint8)
    data, pieces = observed_pieces(response, tail, dict(source='mother', mother_bytes=mother))
    assert np.array_equal(np.concatenate([data[p.start:p.end] for p in pieces]),
                          np.concatenate((response, tail)))
    coverage = np.zeros(len(data), dtype=int)
    for p in pieces:
        coverage[p.start:p.end] += 1
        if p.probe_start is not None:
            assert p.end == p.probe_start < p.probe_end
            if p.source == 'mother':
                assert p.probe_end == mother
            else:
                assert p.start >= mother
    assert (coverage == 1).all()


def test_unavailable_source_has_no_fabricated_gain_and_background_cannot_outvote_by_length():
    readings = [dict(source='mother',bytes=7,pre=2.,post=1.),
                dict(source='ambient',bytes=470,pre=1.,post=1.5)]
    assert settle_sources(readings) == .25
    readings[1]['bytes'] = 470000
    assert settle_sources(readings) == .25
    assert settle_sources([dict(bytes=1,pre=1.,post=2.)]) == -1.
    with pytest.raises(ValueError):
        settle_sources([])
    with pytest.raises(ValueError):
        settle_sources([dict(bytes=1,pre=float('nan'),post=0.)])


def test_joint_sampling_credits_unchosen_proposals_once_and_choice_once():
    choice = torch.tensor(-.7, requires_grad=True)
    a = torch.tensor([-.1,-.2], requires_grad=True)
    b = torch.tensor([-.3,-.4], requires_grad=True)
    loss = proposal_log_probability(choice, [a,b], torch.tensor(-.5))
    loss.backward()
    assert choice.grad == 1
    assert torch.equal(a.grad, torch.ones_like(a))
    assert torch.equal(b.grad, torch.ones_like(b))
    assert float(loss.detach()) == pytest.approx(-2.2)


def test_joint_migration_preserves_real_facts_parameters_pending_credit_and_rng():
    e = configured()
    e.settlement_kind, e.reward_memory_kind = 'continuation_joint', 'observed'
    e.read_gain_state = copy.deepcopy(e.internal_gain_state)
    e.prev = dict(g=.3,actor=(torch.tensor(2.),),settlement_kind='continuation_joint')
    pending = copy.deepcopy(e.prev)
    before = copy.deepcopy(e.model.state_dict())
    params = [id(p) for p in e.model.parameters()]
    rng = torch.get_rng_state().clone()
    migrate_joint_cycle(e, proposal_budget=2)
    assert e.settlement_kind == 'world_stream' and e.joint_proposal_budget == 2
    assert params == [id(p) for p in e.model.parameters()]
    assert torch.equal(rng, torch.get_rng_state())
    assert e.prev['g'] == pending['g'] and torch.equal(e.prev['actor'][0],pending['actor'][0])
    assert e.prev['settlement_kind'] == 'continuation_joint'
    for k,v in before.items():
        if torch.is_tensor(v):
            assert torch.equal(v,e.model.state_dict()[k])
    assert not e.shadow_gain_state['count'].any()
    assert e.joint_cycle_archive['settlement'] == 'continuation_joint'


def test_invalid_public_boundaries_and_unbounded_budgets_fail_before_observation():
    with pytest.raises(ValueError):
        observed_pieces(np.zeros(4,dtype=np.uint8),np.zeros(4,dtype=np.uint8),
                        dict(source='mother',mother_bytes=5))
    e = configured()
    e.settlement_kind,e.reward_memory_kind='continuation_joint','observed'
    e.read_gain_state=copy.deepcopy(e.internal_gain_state)
    with pytest.raises(ValueError):
        migrate_joint_cycle(e,proposal_budget=e.model.memory.topk+1)


def complete_interface_engine():
    """Small tensors exercising the production graph; no capability score."""
    from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
    from scripts.ib_local.train_gdn_fast_table import TableGDN
    from scripts.ib_local.language_navigation_kanli_v4b import attach_imagination
    from scripts.ib_local.language_navigation_kanli_v4m import WoodObservationBody
    from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
    from scripts.ib_local.run_causal_water_reply import WaterReplyEngine
    from scripts.ib_local.navigation_direct_read import configure_direct_read
    from scripts.ib_local.navigation_internal_rollout import configure_internal_rollout
    from scripts.ib_local.navigation_bfss_dynamics import configure_momentum
    from scripts.ib_local.navigation_momentum_wood import configure_momentum_wood
    from scripts.ib_local.navigation_relation_wood import configure_relation_wood
    from scripts.ib_local.navigation_question_context import configure_question_context
    from scripts.ib_local.navigation_shadow_preview import configure_shadow_preview
    from scripts.ib_local.navigation_mother_boundary import configure_mother_boundary
    from scripts.ib_local.navigation_persistent_question import configure_persistent_question
    from scripts.ib_local.navigation_causal_program import migrate_causal_program
    from scripts.ib_local.navigation_settlement import configure_settlement, configure_reward_memory

    class PublicWorld:
        anchor = offset = moves = 0
        def state_dict(self):
            return dict(anchor=self.anchor, offset=self.offset, moves=self.moves)

        def load_state_dict(self, state):
            self.anchor, self.offset, self.moves = (state[k] for k in ('anchor', 'offset', 'moves'))

        def take(self, words, n, h):
            raw = np.frombuffer((b'The reply follows the words. The background continues. '*4),np.uint8)
            self.offset += n+h
            return raw[:n].copy(),raw[n:n+h].copy(),dict(source='mother',mother_bytes=32,
                match=0,ambient_match=0,holdout_position_fresh=True,anchor=0)

    torch.manual_seed(89)
    base = TableGDN(GatedDeltaNetLM(256,16,2,2,'decay_then_delta',32,4,32.),True)
    model = attach_imagination(WoodObservationBody(base,latent_dim=8,capacity=128,
        short=4,long=128,topk=4,history_size=8,max_chunk=128))
    optimizer = torch.optim.AdamW(base.parameters(),lr=.0003)
    optimizer.add_param_group(dict(params=[p for n,p in model.named_parameters()
                                          if not n.startswith('body.')],lr=.0003))
    e = WaterReplyEngine(model,optimizer,model.initial_state(1),
        RetrievedCreditTable(orders=(0,),heads=1,slots_log2=1),PublicWorld(),32,
        np.random.default_rng(89),'active',32,64,64,capture=False)
    configure_direct_read(e,'real','joint_only')
    configure_settlement(e,'continuation_joint');configure_reward_memory(e,'observed')
    configure_internal_rollout(e,'real');configure_momentum(e);configure_momentum_wood(e)
    configure_relation_wood(e);configure_question_context(e);configure_shadow_preview(e)
    configure_mother_boundary(e);configure_persistent_question(e)
    e.bound_question_shadow=e.joint_question_update=e.align_question_boundary=e.mother_shadow_enabled=True
    migrate_causal_program(e);migrate_joint_cycle(e,2)
    return e


def test_entire_production_hook_chain_new_world_score_replay_and_input_free_continuation(monkeypatch):
    torch.set_num_threads(1)
    from scripts.ib_local import world_learning_cycle as cycle
    read_prefixes = []
    original_read = cycle.frozen_table_logp
    def capture_prefix(table, ids):
        read_prefixes.append(ids.copy())
        return original_read(table,ids)
    monkeypatch.setattr(cycle,'frozen_table_logp',capture_prefix)
    e = complete_interface_engine()
    actual_updates = []
    optimizer_step = e.optimizer.step
    def counted_step(*a,**kw):
        actual_updates.append(1)
        return optimizer_step(*a,**kw)
    e.optimizer.step = counted_step
    before = int(e.model.memory.clock)
    row = e.step(torch.randn(8,4))
    assert row['updates'] == len(actual_updates)
    assert row['world_learning']['source_roles'] == ['mother','mother','ambient','ambient']
    assert row['world_learning']['external_bytes_once'] == 128
    assert len(row['joint_language_search']['candidates']) == 2
    assert len(e._draft_actions) in (2,3) and len(e._causal_words) == len(e._draft_actions)
    assert row['policy_replay_max_error'] < 1e-3
    assert row['joint_language_search']['replay_max_error'] < 1e-3
    assert int(e.model.memory.clock)-before == 128+row['self_bytes']
    assert row['causal_program']['evidence_committed']
    assert e.prev['g'] == row['world_learning']['settlement']
    assert len(read_prefixes) == 4
    assert np.array_equal(read_prefixes[0],read_prefixes[1])
    assert np.array_equal(read_prefixes[2],read_prefixes[3])
    old_slot,old_gain = row['episode_slot'],row['metal_g']
    clock,episodes,world = int(e.model.memory.clock),int(e.model.memory.episodes),e.world.offset
    weights = [p.detach().clone() for p in e.model.parameters()]
    fact_versions = e.model.memory.last_write.clone()
    inner = e.continue_internal(1)
    assert inner['new_world_evidence'] is False and inner['ticks'] == 1
    assert int(e.model.memory.clock) == clock and int(e.model.memory.episodes) == episodes
    assert e.world.offset == world and torch.equal(fact_versions,e.model.memory.last_write)
    assert all(torch.equal(p,v) for p,v in zip(e.model.parameters(),weights))
    actual_updates.clear()
    second = e.step(torch.randn(8,4))
    assert second['updates'] == len(actual_updates)
    assert second['td_previous']['memory_label_kind'] == 'observed_gain'
    assert second['internal_rollout']['settled_two_turn'] is not None
    if second['episode_slot'] != old_slot:
        assert float(e.model.memory.progress[old_slot]) == pytest.approx(old_gain)


def test_v3_constructor_does_not_capture_the_immediately_discarded_v2_graph(monkeypatch):
    from scripts.ib_local import language_navigation_v2 as v2
    captures=[]
    original=v2.DecisionGraph
    def observed(*a,**kw):
        captures.append(kw.get('capture',True))
        return original(*a,**kw)
    monkeypatch.setattr(v2,'DecisionGraph',observed)
    complete_interface_engine()
    assert captures == [False]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_sequential_proposal_credit_matches_original_joint_autograd(device):
    """All four proposal graphs may be freed separately without changing credit."""
    torch.set_num_threads(1)
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    e = complete_interface_engine()
    if device == 'cuda':
        from scripts.ib_local.language_navigation_turns_v3 import DecisionGraphV3
        from scripts.ib_local.run_causal_water_reply import to_device
        e.model.to(device)
        for name, value in list(vars(e).items()):
            if name not in {'model', 'params', 'optimizer', 'groups', 'decision', 'world', 'table'}:
                setattr(e, name, to_device(value, device))
        e.device = torch.device(device)
        e.decision = DecisionGraphV3(e.model, capture=False)
    e.joint_proposal_budget = 4
    words, record = e._utter()
    targets = torch.tensor(list(words), dtype=torch.long, device=e.device)[None]
    scalar, expected_replay = e._policy_logp(None, None, targets, len(words), record)
    expected = torch.autograd.grad(scalar, e.params, allow_unused=True)
    expected = tuple(None if g is None else g.detach().clone() for g in expected)
    actual, replay = e._policy_grad(targets, len(words), record)
    assert replay['choice'] == expected_replay['choice']
    assert replay['stops'] == expected_replay['stops']
    if 'byte_logp' in replay:
        torch.testing.assert_close(replay['byte_logp'], expected_replay['byte_logp'])
    assert len(e._joint_proposals) == 4
    for before, after in zip(expected, actual):
        assert (before is None) == (after is None)
        if before is not None:
            torch.testing.assert_close(after, before, rtol=2e-5, atol=2e-7)


def assert_same_saved_values(left, right):
    """Compare nested optimizer, receipt and recurrent values exactly."""
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_saved_values(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same_saved_values(a, b)
    else:
        assert left == right


def test_restore_audit_preserves_values_while_canonicalizing_receipt_storage():
    from scripts.ib_local.audit_joint_restore import canonical
    from scripts.ib_local.checkpoint_digest import state_digest
    view = torch.arange(24).reshape(3, 8)[:, :2]
    restored = view.contiguous()
    assert view.stride() != restored.stride()
    assert state_digest(canonical(view))['sha256'] == state_digest(canonical(restored))['sha256']
    restored[0, 0] += 1
    assert state_digest(canonical(view))['sha256'] != state_digest(canonical(restored))['sha256']


def test_saved_joint_cycle_replays_next_real_transition_and_delayed_credit_exactly(tmp_path):
    """Production serialization, same continuation: no reset or capability task."""
    import gzip
    import json
    from scripts.ib_local.navigation_checkpoint import restore, save

    torch.set_num_threads(1)
    e = complete_interface_engine()
    first = e.step(torch.randn(8, 4))
    e.continue_internal(1)
    saved_actor = tuple(None if g is None else g.clone() for g in e.prev['actor'])
    save(e, tmp_path, cycle=1,
         config=dict(joint_world_cycle=dict(schema=1, proposal_budget=2)))
    with gzip.open(tmp_path/'last.pt.gz', 'rb') as stream:
        saved = torch.load(stream, map_location='cpu', weights_only=False)
    assert saved['cycle'] == 1 and saved['engine']['joint_internal_ticks'] == 1
    assert saved['engine']['settlement_kind'] == 'world_stream'
    assert_same_saved_values(saved['engine']['prev']['actor'], saved_actor)

    expected_row = e.step(torch.randn(8, 4))
    # Small fixture tensors only; retain evidence for exact next-update equality.
    expected_model = copy.deepcopy(e.model.state_dict())
    expected_optimizer = copy.deepcopy(e.optimizer.state_dict())
    expected_recurrent = e.recurrent.clone()
    expected_momenta = [p.clone() for p in e.bfss_momenta]
    expected_actor = tuple(None if g is None else g.clone() for g in e.prev['actor'])
    expected_rng = torch.get_rng_state().clone()
    expected_table = copy.deepcopy(e.table.state_dict())
    expected_world = e.world.state_dict()

    # Rebuild the fixture graph and opt-in class, then use the production
    # restore. Destroy the old instance so transient Python fields cannot
    # accidentally supply missing carried state after loading.
    del e
    e = complete_interface_engine()
    restore(e, saved)
    assert e.joint_internal_ticks == 1
    assert_same_saved_values(e.prev['actor'], saved_actor)
    assert e.prev['g'] == first['world_learning']['settlement']
    replay_row = e.step(torch.randn(8, 4))
    assert json.dumps(replay_row, sort_keys=True) == json.dumps(expected_row, sort_keys=True)
    assert_same_saved_values(e.model.state_dict(), expected_model)
    assert_same_saved_values(e.optimizer.state_dict(), expected_optimizer)
    assert_same_saved_values(e.prev['actor'], expected_actor)
    assert torch.equal(e.recurrent, expected_recurrent)
    assert_same_saved_values(e.bfss_momenta, expected_momenta)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert_same_saved_values(e.table.state_dict(), expected_table)
    assert e.world.state_dict() == expected_world
