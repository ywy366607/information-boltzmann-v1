"""Causal wiring and derivative tests, not capability studies."""
import numpy as np
import pytest
import torch

from scripts.ib_local.language_action import (
    ACTION_BYTES, ActionTrainer, TextChoiceWorld, joint_loss, peek,
)
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.train_gdn_fast_table import TableGDN


def small_table():
    table = RetrievedCreditTable(orders=(0,1,2),heads=1,slots_log2=4)
    table.run_credit(np.frombuffer(b'the world reads words and the world learns ',np.uint8))
    return table


def test_peek_is_read_only_and_matches_actual_first_prediction():
    table = small_table()
    before = table.state_dict()
    p = peek(table,ord('t'))
    assert table.history == before['history'] and table.events == before['events']
    for name,value in before['arrays'].items():
        np.testing.assert_array_equal(getattr(table,name),value)
    _,probs,_ = table.run_credit(np.frombuffer(b't0new language',np.uint8))
    np.testing.assert_array_equal(p,probs[0])


def test_world_intervention_and_cursor_restore_are_real_and_bounded():
    raw = np.array([0,101,102,103,0,104,105,106,0,201,202,203,204,205,206],np.int64)
    world = TextChoiceWorld(raw,4,((0,8),(8,15)))
    a = world.take(0,3)
    assert world.cursors == [4,8]
    saved = list(world.cursors)
    b = world.take(1,3)
    assert not np.array_equal(a,b)
    clone = TextChoiceWorld(raw,4,world.bounds,saved)
    np.testing.assert_array_equal(b,clone.take(1,3))
    assert clone.cursors == world.cursors
    with pytest.raises(EOFError):
        clone.take(1,10)


def test_joint_estimator_matches_exact_expected_shared_parameter_derivative():
    torch.manual_seed(9)
    theta = torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    direction = torch.randn(1,4,256,dtype=torch.float64)
    fixed = torch.randn(2,1,4,256,dtype=torch.float64)
    fixed[1,:,0] = fixed[0,:,0]  # both actions have the same causal prefix
    targets = torch.tensor([[48,65,66,67]])
    losses, policies, surrogates = [], [], []
    for action in range(2):
        logits = fixed[action]+theta*direction
        surrogate, loss, logp = joint_loss(logits,targets,torch.tensor(action),torch.tensor(.7))
        losses.append(loss)
        policies.append(logp.exp()[0,action])
        surrogates.append(surrogate)
    exact = sum(p*l for p,l in zip(policies,losses))
    expected_estimator = sum(p.detach()*l for p,l in zip(policies,surrogates))
    a = torch.autograd.grad(exact,theta,retain_graph=True)[0]
    b = torch.autograd.grad(expected_estimator,theta)[0]
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)


def test_self_command_is_not_scored_as_external_evidence():
    torch.manual_seed(1)
    logits = torch.randn(1,4,256,requires_grad=True)
    targets = torch.tensor([[48,65,66,67]])
    _,loss,_ = joint_loss(logits,targets,torch.tensor(0),torch.tensor(0.))
    changed = targets.clone()
    changed[0,0] = 200
    _,other,_ = joint_loss(logits,changed,torch.tensor(0),torch.tensor(0.))
    torch.testing.assert_close(loss,other)
    grad = torch.autograd.grad(loss,logits)[0]
    assert torch.count_nonzero(grad[:,0]) == 0
    assert torch.count_nonzero(grad[:,1:]) > 0


def test_replay_control_has_only_direct_external_language_gradients():
    from scripts.ib_local.replay_language_action import ReplayTrainer
    torch.set_num_threads(2)
    torch.manual_seed(19)
    def make():
        return TableGDN(GatedDeltaNetLM(256,16,2,2,'decay_then_delta',32,4,32.,
                                        decay_init='log_timescale'),True)
    model,reference=make(),make()
    reference.load_state_dict(model.state_dict())
    runner=ReplayTrainer(model,8,capture=False)
    runner.ids.copy_(torch.randint(0,256,(1,8)))
    runner.targets.copy_(torch.randint(0,256,(1,8)))
    runner.action.fill_(1)
    runner.baseline.fill_(-9)  # Would create a large policy term if accidentally retained.
    loss,_,_=runner._backward()
    logits,_,_=reference.predict(runner.ids,runner.state,runner.table)
    direct=torch.nn.functional.cross_entropy(logits[:,1:].reshape(-1,256),
                                            runner.targets[:,1:].reshape(-1))
    direct.backward()
    torch.testing.assert_close(loss,direct)
    for p,q in zip(model.parameters(),reference.parameters()):
        if p.grad is not None or q.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_graph_matches_eager_and_full_continuation():
    from scripts.ib_local import gdn_fused
    gdn_fused.enable()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    def make():
        base = GatedDeltaNetLM(256,16,2,2,'decay_then_delta',32,4,32.,decay_init='log_timescale').cuda()
        return TableGDN(base,True).cuda()
    first,second = make(),make()
    second.load_state_dict(first.state_dict())
    graph = ActionTrainer(first,8,True)
    eager = ActionTrainer(second,8,False)
    table = small_table()
    p = peek(table,ord('t'))
    proposed = graph.decide(ord('t'),p)
    np.testing.assert_allclose(proposed,eager.decide(ord('t'),p),rtol=2e-5,atol=2e-6)
    raw = np.frombuffer(b't0abcdefg',np.uint8)
    _,probs,_ = table.run_credit(raw)
    a = graph.step(raw,probs,0,.7)
    b = eager.step(raw,probs,0,.7)
    assert a[0] == pytest.approx(b[0],rel=1e-6)
    np.testing.assert_allclose(a[1],proposed,atol=2e-6)
    torch.testing.assert_close(graph.state,eager.state,rtol=1e-5,atol=2e-6)
    for x,y in zip(first.parameters(),second.parameters()):
        torch.testing.assert_close(x,y,rtol=1e-5,atol=2e-6)
    # Restore the complete neural/optimizer/table transition in a fresh runner.
    import copy
    snapshot = copy.deepcopy(dict(model=first.state_dict(),optimizer=graph.optimizer.state_dict(),
                                  recurrent=graph.state,table=table.state_dict()))
    third = make()
    third.load_state_dict(snapshot['model'])
    restored = ActionTrainer(third,8,True)
    restored.optimizer.load_state_dict(snapshot['optimizer'])
    with torch.no_grad():
        restored.state.copy_(snapshot['recurrent'])
    clone = RetrievedCreditTable.from_state_dict(snapshot['table'])
    raw2 = np.frombuffer(b'g1hijklmn',np.uint8)
    _,p1,_ = table.run_credit(raw2)
    _,p2,_ = clone.run_credit(raw2)
    np.testing.assert_array_equal(p1,p2)
    one = graph.step(raw2,p1,1,1.2)
    two = restored.step(raw2,p2,1,1.2)
    assert one[0] == two[0]
    for x,y in zip(first.parameters(),third.parameters()):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
