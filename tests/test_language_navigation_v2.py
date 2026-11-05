"""Actual body/world/action causal wiring, not capability tests."""
import numpy as np
import pytest
import torch

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.language_navigation_v2 import DecisionGraph, DialogueEngine
from scripts.ib_local.language_navigation_world import LanguageNavigationWorld
from scripts.ib_local.latent_navigation_v2 import LatentNavigationV2
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.train_gdn_fast_table import TableGDN


@pytest.mark.parametrize('mode', ['active','random','silence'])
def test_same_body_self_hears_and_learns_each_observed_segment(mode):
    torch.manual_seed(34); torch.set_num_threads(1)
    base = GatedDeltaNetLM(256, 16, 2, 2, 'decay_then_delta', 32, 4, 32., decay_init='log_timescale')
    model = LatentNavigationV2(TableGDN(base, True), 8, 16, 2, 8, 2,
                               max_chunk=8, history_size=4)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
    table = RetrievedCreditTable(orders=(0,1,2), heads=1, slots_log2=5)
    data = np.frombuffer(b'ordinary language changes and returns repeatedly. '*80, np.uint8)
    world = LanguageNavigationWorld(data, buckets=512)
    engine = DialogueEngine(model, optim, model.initial_state(1), table, world, ord('a'),
                            np.random.default_rng(4), mode, 4, 4, 4)
    initial = model.body.base.decoder.weight.detach().clone()
    row = engine.step(torch.randn(8, 6), age_ablate=True)
    assert row['external_bytes'] == 8 and row['updates'] == 2
    assert row['self_bytes'] == (0 if mode == 'silence' else 4)
    assert table.events == int(model.memory.clock) == row['external_bytes'] + row['self_bytes']
    assert not torch.equal(initial, model.body.base.decoder.weight)
    assert np.isfinite(row['progress']) and np.isfinite(row['response_nll'])
    assert set(row['age_component_ablation']) == {'0_4k','4k_64k','64k_256k','256k_plus'}
    if mode == 'silence':
        assert row['policy_replay_max_error'] is None
    else:
        assert row['policy_replay_max_error'] < 1e-4
        assert len(row['utterance_hex']) == 8


def test_active_policy_with_frozen_latent_gain_excludes_frozen_parameter_from_credit():
    torch.manual_seed(35); torch.set_num_threads(1)
    base = GatedDeltaNetLM(256,16,2,2,'decay_then_delta',32,4,32.,decay_init='log_timescale')
    model = LatentNavigationV2(TableGDN(base,True),8,16,2,8,2,max_chunk=8,history_size=4)
    model.read_gain.requires_grad_(False)
    optim = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0)
    table = RetrievedCreditTable(orders=(0,1),heads=1,slots_log2=4)
    world = LanguageNavigationWorld(np.frombuffer(b'actual language with a changing context. '*80,np.uint8),buckets=512)
    engine = DialogueEngine(model,optim,model.initial_state(1),table,world,ord('a'),
                            np.random.default_rng(4),'active',4,4,4)
    engine.step(torch.randn(8,6))
    assert model.read_gain.item() == 0


def test_seen_probe_supplies_no_novelty_progress_but_baseline_is_unconditionally_applied():
    torch.manual_seed(36); torch.set_num_threads(1)
    base = GatedDeltaNetLM(256,16,2,2,'decay_then_delta',32,4,32.,decay_init='log_timescale')
    model = LatentNavigationV2(TableGDN(base,True),8,16,2,8,2,max_chunk=8,history_size=4)
    optim = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0)
    table = RetrievedCreditTable(orders=(0,1),heads=1,slots_log2=4)
    data = np.frombuffer(b'continuous language and repeated familiar passage. '*80,np.uint8)
    world = LanguageNavigationWorld(data,buckets=512)
    world.mark_seen(0,len(data))
    engine = DialogueEngine(model,optim,model.initial_state(1),table,world,ord('a'),
                            np.random.default_rng(4),'active',4,4,4,baseline=.25)
    row = engine.step(torch.randn(8,6))
    assert not row['world']['holdout_position_fresh']
    assert row['position_fresh_progress'] is None and row['actor_update_used']
    assert row['novelty_conditioned_signal'] == 0 and row['advantage'] == -.25
    assert engine.baseline == pytest.approx(.25*.95)
    assert table.events == 12 and row['updates'] == 2


def test_cuda_decision_graph_matches_eager_across_continuous_bytes():
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from scripts.ib_local import gdn_fused
    gdn_fused.enable(); torch.manual_seed(4); torch.set_num_threads(1)
    base = GatedDeltaNetLM(256,16,2,2,'decay_then_delta',32,4,32.,decay_init='log_timescale').cuda()
    model = LatentNavigationV2(TableGDN(base,True).cuda(),8,16,2,8,2,max_chunk=8,history_size=4).cuda()
    graph, eager = DecisionGraph(model,True), DecisionGraph(model,False)
    start = model.initial_state(1,'cuda'); graph.start(start); eager.start(start)
    for byte in (1,3,7,9):
        prob = np.random.default_rng(byte).dirichlet(np.ones(256)).astype(np.float32)
        a, b = graph.step(byte,prob), eager.step(byte,prob)
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(graph.state,eager.state,atol=2e-6,rtol=2e-5)
