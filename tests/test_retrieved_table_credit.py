"""Numerical/interface checks, not synthetic capability experiments."""
import numpy as np
import pytest

from scripts.ib_local.ngram_fast_table import ByteNgramFastTable
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable


def seeded_table():
    table = RetrievedCreditTable(orders=(0,1,2), heads=1, slots_log2=4)
    table.run_credit(np.frombuffer(b'one two one three one two one ',dtype=np.uint8))
    return table


def test_disabled_credit_matches_original_prequential_table():
    old = ByteNgramFastTable(orders=(0,1,2), heads=1, slots_log2=4)
    new = RetrievedCreditTable(orders=(0,1,2), heads=1, slots_log2=4)
    raw = np.random.default_rng(4).integers(0,256,100,dtype=np.uint8)
    for a,b in [(0,31),(31,65),(65,99)]:
        n,p = old.run(raw[a:b+1],True)
        m,q,receipt = new.run_credit(raw[a:b+1])
        new.settle(receipt,np.zeros_like(q),mode='off')
        np.testing.assert_allclose(m,n,atol=1e-12)
        np.testing.assert_array_equal(p,q)
    for name in ('rows','count','tag','err'):
        np.testing.assert_array_equal(getattr(old,name),getattr(new,name))


def test_actual_count_read_derivative_and_causal_receipt():
    table = seeded_table()
    before = table.state_dict()
    raw=np.array([ord('o'),ord('n')],np.uint8)
    _,p,receipt=table.run_credit(raw)
    g=np.random.default_rng(1).normal(size=256)
    for h,slot in enumerate(receipt.slots[0]):
        if slot<0:continue
        # Float32 mixture output needs a finite displacement for this audit.
        eps=1e-2
        losses=[]
        for sign in (-1,1):
            other=RetrievedCreditTable.from_state_dict(before)
            other.count[slot]+=sign*eps
            _,prob,_=other.run_credit(raw)
            losses.append(np.dot(np.log(prob[0]),g))
        analytic=np.dot(receipt.eligibility[0,h],g)
        assert (losses[1]-losses[0])/(2*eps)==pytest.approx(analytic,abs=3e-3,rel=2e-2)
    changed=RetrievedCreditTable.from_state_dict(before)
    _,q,r=changed.run_credit(np.array([ord('o'),ord('z')],np.uint8))
    np.testing.assert_array_equal(q,p)
    np.testing.assert_array_equal(r.eligibility,receipt.eligibility)


def test_settlement_conserves_local_counts_and_changes_future_assimilation():
    table=seeded_table()
    _,p,packet=table.run_credit(np.frombuffer(b'one two one three ',dtype=np.uint8))
    before=table.count.copy()
    row_before=table.rows.copy()
    unmodified=RetrievedCreditTable.from_state_dict(table.state_dict())
    report=table.settle(packet,np.random.default_rng(3).normal(size=p.shape),rate=.2)
    assert report['gross_count_exchange']>0
    np.testing.assert_allclose(table.count.reshape(3,-1).sum(1),before.reshape(3,-1).sum(1),atol=1e-12)
    np.testing.assert_array_equal(table.rows,row_before)
    assert np.all(table.count>=0)
    assert np.any(1/(table.count+2)!=1/(before+2))
    raw=np.frombuffer(b'one two one ',dtype=np.uint8)
    _,p,_=table.run_credit(raw)
    _,q,_=unmodified.run_credit(raw)
    assert np.max(np.abs(p-q))>1e-8
    assert np.max(np.abs(table.rows-unmodified.rows))>1e-8


def test_numerical_ceiling_shortens_whole_exchange_without_losing_mass():
    from scripts.ib_local.retrieved_table_credit import ReadCredit, MAX_OCCUPANCY
    table=RetrievedCreditTable(orders=(0,),heads=1,slots_log2=1)
    table.count[:]=[MAX_OCCUPANCY-1,MAX_OCCUPANCY/2]
    e=np.zeros((1,2,256),np.float32)
    e[0,:,0]=[-20,20]
    receipt=ReadCredit(np.array([[0,1]]),np.zeros((1,2),np.uint64),
                       np.full((1,2),4096,np.uint64),e)
    total=table.count.sum()
    result=table.settle(receipt,np.ones((1,256)),rate=.5)
    assert result['boundary_limited_pools']==1
    assert result['reads_older_than_window']==2
    assert result['gross_count_exchange']>0
    assert np.all((table.count>0)&(table.count<=MAX_OCCUPANCY))
    assert table.count.sum()==pytest.approx(total,abs=1e-8)


def test_replaced_owner_cannot_receive_old_credit():
    table=seeded_table()
    _,p,packet=table.run_credit(np.frombuffer(b'one two ',dtype=np.uint8))
    ids=np.unique(packet.slots[packet.slots>=0])
    table.epoch[ids]+=1
    before=table.count.copy()
    report=table.settle(packet,np.ones_like(p))
    assert report['eligible_reads']==0
    np.testing.assert_array_equal(before,table.count)


def test_complete_table_restore_continues_identically_and_has_bounded_packet():
    a=seeded_table(); b=RetrievedCreditTable.from_state_dict(a.state_dict())
    raw=np.frombuffer(b'one two one three ',dtype=np.uint8)
    for table in (a,b):
        _,p,packet=table.run_credit(raw)
        table.settle(packet,np.random.default_rng(5).normal(size=p.shape))
    for k in a.state_dict()['arrays']:
        np.testing.assert_array_equal(getattr(a,k),getattr(b,k))
    assert a.events==b.events and a.history==b.history
    assert packet.nbytes==(len(raw)-1)*3*(24+256*4)
    with pytest.raises(ValueError):a.run_credit(np.zeros(514,np.uint8))


def test_zero_factor_no_exchange_and_shuffle_is_deterministic():
    a=seeded_table()
    _,p,packet=a.run_credit(np.frombuffer(b'one two one three ',dtype=np.uint8))
    b=RetrievedCreditTable.from_state_dict(a.state_dict())
    c=RetrievedCreditTable.from_state_dict(a.state_dict())
    before=a.count.copy()
    a.settle(packet,np.zeros_like(p))
    np.testing.assert_allclose(a.count,before,atol=1e-12)
    g=np.random.default_rng(12).normal(size=p.shape)
    for x in (b,c):x.settle(packet,g,mode='shuffled',rng=np.random.default_rng(7))
    np.testing.assert_array_equal(b.count,c.count)


def test_cuda_table_factor_matches_eager_and_full_restore(tmp_path):
    import copy
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from scripts.ib_local import gdn_fused
    from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
    from scripts.ib_local.train_gdn_fast_table import TableGDN
    from scripts.ib_local.train_retrieved_credit import CreditGraphTrainer, save_snapshot, restore_snapshot
    gdn_fused.enable()
    torch.set_num_threads(1);torch.manual_seed(7)
    model=TableGDN(GatedDeltaNetLM(256,32,2,2,'decay_then_delta',64,4,32.,
                     decay_init='log_timescale').cuda(),True).cuda()
    eager=copy.deepcopy(model)
    trainer=CreditGraphTrainer(model,16,8)
    ids=torch.randint(256,(1,16),device='cuda'); targets=torch.randint(256,(1,16),device='cuda')
    p=torch.randn(1,16,256,device='cuda').log_softmax(-1).detach().requires_grad_(True)
    state=eager.initial_state(1,device='cuda')
    ls=[]
    for i in (0,8):
        loss,state,_=eager(ids[:,i:i+8],targets[:,i:i+8],state,p[:,i:i+8])
        (loss/2).backward();ls.append(loss.detach());state=state.detach()
    loss,g=trainer.step_credit(ids,targets,p.detach())
    torch.testing.assert_close(loss,torch.stack(ls).mean(),atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(g,p.grad*16,atol=2e-6,rtol=2e-4)
    torch.testing.assert_close(trainer.state,state,atol=2e-6,rtol=2e-5)
    assert g.abs().sum()>0
    table=seeded_table();rng=np.random.default_rng(3)
    save_snapshot(tmp_path/'full.pt',trainer,table,rng,1,{'unit_test':True})
    snapshot=torch.load(tmp_path/'full.pt',weights_only=False,map_location='cpu')
    loss1,g1=trainer.step_credit(ids,targets,p.detach())
    parameters={k:v.clone() for k,v in model.state_dict().items()}
    restored=restore_snapshot(snapshot,trainer,rng)
    loss2,g2=trainer.step_credit(ids,targets,p.detach())
    torch.testing.assert_close(loss1,loss2,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(g1,g2,atol=1e-6,rtol=1e-6)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,parameters[k],atol=1e-6,rtol=1e-6)
    np.testing.assert_array_equal(restored.rows,table.rows)
