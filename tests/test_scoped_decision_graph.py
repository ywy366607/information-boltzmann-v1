import pytest
import torch
import numpy as np

from scripts.ib_local import gdn_fused
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.train_gdn_fast_table import TableGDN
from scripts.ib_local.latent_navigation import LatentNavigationBody
from scripts.ib_local.scoped_decision_graph import ScopedDecisionGraph


def test_runtime_configuration_keeps_carried_state_and_cpu_restore_path():
    from types import SimpleNamespace
    from scripts.ib_local.scoped_decision_graph import configure_fused_step
    model = LatentNavigationBody(TableGDN(GatedDeltaNetLM(256,32,2,1,'decay_then_delta',64,4),True,0),
                                 latent_dim=8,capacity=8,horizon=2,max_chunk=8)
    state = model.initial_state(1).normal_(0., .01)
    engine = SimpleNamespace(model=model, recurrent=state, decision=ScopedDecisionGraph(model, False))
    saved = state.clone()
    rng = torch.get_rng_state()
    config = {}
    configure_fused_step(engine, config)
    assert config['decode_runtime']['fused_step'] and not engine.decision.capture
    assert torch.equal(saved, engine.recurrent) and torch.equal(rng, torch.get_rng_state())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph input check')
@pytest.mark.parametrize('fused_step', [False, True])
def test_graph_uses_current_question_and_budget_without_changing_output_rank(fused_step):
    gdn_fused.enable()
    model=LatentNavigationBody(TableGDN(GatedDeltaNetLM(256,32,2,1,'decay_then_delta',64,4),True,0),
                              latent_dim=8,capacity=8,horizon=2,max_chunk=8).cuda()
    model.question_to_hidden=torch.nn.Linear(8,64,bias=False,device='cuda')
    model.question_to_logits=torch.nn.Linear(8,256,bias=False,device='cuda')
    model.expression_budget_to_hidden=torch.nn.Linear(1,32,bias=False,device='cuda')
    model.handover=torch.nn.Linear(33,1,device='cuda')
    graph=ScopedDecisionGraph(model,True,capture_handover=True,fused_step=fused_step)
    state=model.initial_state(1,device='cuda')
    probability=np.full(256,1/256,dtype=np.float32)
    for budget in (1.,.25):
        with torch.no_grad():
            model.base.layers[0].q_proj.weight.add_(.001)
        model._question_context=torch.randn(8,device='cuda')
        model._expression_remaining=budget
        # Rare replay cleanup may release only unused allocator blocks. The
        # existing captured graph and its live tensors must remain replayable.
        torch.cuda.empty_cache()
        graph.start(state)
        logits,h=graph.step(97,probability)
        with torch.no_grad():
            direct=model.predict_observed(torch.tensor([[97]],device='cuda'),state,
                torch.full((1,1,256),-np.log(256),device='cuda'))
        assert logits.shape==(256,) and h.shape==(32,)
        torch.testing.assert_close(logits,direct[0][0,0],atol=2e-5,rtol=2e-5)
        from scripts.ib_local.variable_expression import handover_logits
        with torch.no_grad():
            stop = handover_logits(model, h, budget).reshape(())
        torch.testing.assert_close(graph.handover_logit, stop, atol=2e-5, rtol=2e-5)
        assert all(not hasattr(layer, '_decode_step_bundle') for layer in model.base.layers)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_draft_cache_refresh_live_masks_and_training_are_equivalent(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA graph cache contract')
    from test_past_present_coevolution import engine, populated, SPEC
    from scripts.ib_local.past_present_coevolution import configure_coevolution
    torch.manual_seed(76)
    _, e = engine()
    configure_coevolution(e, SPEC)
    populated(e)
    m = e.model.to(device)
    state = e.recurrent.to(device)
    if device == 'cuda':
        gdn_fused.enable()
    graph = ScopedDecisionGraph(m, capture=device == 'cuda')
    probability = np.full(256, 1/256, dtype=np.float32)
    logp = torch.full((1, 1, 256), -np.log(256), device=device)
    ids = torch.tensor([[97]], device=device)
    f = m.body.coevolution
    for turn in range(3):
        with torch.no_grad():
            # Both a learned encoder and actual facts can change between drafts.
            m.encoder.weight.add_(.001)
            m.memory.action[:8].add_(.1)
            m.memory.long_successor[:8].sub_(.05)
            f.anchors.add_(.02)
            f.bind.weight.add_(.001)
            graph.start(state)
            # These remain live graph inputs, not cached masks/credit.
            f.enabled.fill_(turn != 1)
            f.preference.add_(.1)
            logits, _ = graph.step(97, probability)
            direct = m.predict_observed(ids, state, logp)
        torch.testing.assert_close(logits, direct[0][0, 0], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(graph.state, direct[1], atol=3e-5, rtol=3e-5)
        assert not hasattr(m.memory, '_decode_projection_cache')
        assert not hasattr(f, '_decode_projection_cache')
        assert not hasattr(m.body, '_decode_no_diagnostics')
        assert 'alpha_mean' in direct[2]
    # Cached inference did not sever ordinary learning or add checkpoint state.
    f.enabled.fill_(1)
    result = m.predict_observed(ids, state, logp)
    result[0].square().mean().backward()
    assert m.encoder.weight.grad.norm() > 0
    assert f.bind.weight.grad.norm() > 0
    assert not any('cache' in name for name in m.state_dict())
