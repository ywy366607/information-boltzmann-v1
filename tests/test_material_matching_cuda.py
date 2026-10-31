"""GPU arithmetic, stock ordering and complete continuation, not capability."""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction as F
from itertools import product
import random

import pytest
import torch

from information_boltzmann.material_matching_cuda import CudaMatchingMaterial, CudaMaterialOptimizer, COIN_LEVELS
from information_boltzmann.conjugate_gradient_port import prepare_readout_fiber
from information_boltzmann.core.cgb_local_medium import LocalLaw, LocalMedium, OpenMedium, ReservoirLaw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make(states, rows, cols, stock=0, seed=19):
    packed = sum(((p*3+q)*3+w) << (5*e) for e,(p,q,w) in enumerate(states))
    law = LocalLaw(rows,cols,2,2,tuple(n*n for n in range(2*max(rows,cols)+1)),
                   epsilon=0,coupling=12,pool_reference_size=2)
    medium = OpenMedium(LocalMedium(law,packed.to_bytes(law.state_bytes,'little')),
                        ReservoirLaw(stock,content_cost=16,chemical_base=2,stimulus=0))
    codes = torch.tensor([(p*3+q)*3+w for p,q,w in states],device='cuda',dtype=torch.uint8).reshape(rows,cols)
    parameter = torch.nn.Parameter(torch.empty(rows,cols,device='cuda'))
    owner = CudaMatchingMaterial(parameter,codes,stock=stock,quantum=F(1,8),
                                 offset=F(-3,8),energy_unit=F(1,128),seed=seed)
    return owner,medium


def cutoff(power):
    with localcontext() as ctx:
        ctx.prec=70
        p = Decimal(2)**(-Decimal(power.numerator)/Decimal(power.denominator))
        return int(p*COIN_LEVELS)


def serial_color(owner, medium, gradient, tickets, color):
    values=gradient.detach().cpu().view(-1).tolist()
    words=tickets.cpu().tolist()
    powers={}
    for lane, (direction,choice,coin) in enumerate(words):
        row,col = ((lane,(lane+color)%owner.cols) if owner.rows<=owner.cols
                   else ((lane+color)%owner.rows,lane))
        edge=row*owner.cols+col
        direction=1 if direction else -1
        block=prepare_readout_fiber(medium,edge,direction,lambda e: F.from_float(values[e])*16)
        if not block:
            continue
        state,power,sign=block[choice%len(block)]
        powers[lane]=float(power)
        if coin < cutoff(power):
            medium._commit(((edge,state),),sign)
    return powers


def assert_body(owner, medium):
    codes=owner.codes.cpu().tolist()
    assert codes == [(p*3+q)*3+w for p,q,w in (medium.body.get(e) for e in range(owner.rows*owner.cols))]
    assert owner.stock.item()==medium.stock
    assert owner.row_p.cpu().tolist()==[medium.body.total(0,r) for r in range(owner.rows)]
    assert owner.col_q.cpu().tolist()==[medium.body.total(1,c) for c in range(owner.cols)]
    expected=torch.tensor([float(F(-3,8)+F(1,8)*sum(medium.body.get(e)))
                           for e in range(owner.rows*owner.cols)]).view(owner.rows,owner.cols)
    assert torch.equal(owner.parameter.detach().cpu(),expected)
    assert sum(sum(medium.body.get(e)) for e in range(owner.rows*owner.cols))+owner.stock.item()==owner.supply


@pytest.mark.parametrize('shape',[(1,1),(1,3),(3,1),(2,3),(3,2),(4,4),(3,7)])
def test_cuda_matching_exact_serial_stock_and_candidate_reference(shape):
    rows,cols=shape;rng=random.Random(573)
    cells=list(product(range(3),repeat=3))
    for stock in (0,1,5):
        states=[rng.choice(cells) for _ in range(rows*cols)]
        owner,medium=make(states,rows,cols,stock=stock)
        for t in range(2*max(shape)):
            gradient=torch.tensor([rng.choice([-8.,-2.,-.125,0.,.125,2.,8.])
                                   for _ in states],device='cuda').reshape(shape)
            tickets=torch.tensor([[rng.randrange(2),rng.randrange(468),
                                   rng.choice([0,COIN_LEVELS-1,rng.randrange(COIN_LEVELS)])]
                                  for _ in range(owner.lanes)],device='cuda',dtype=torch.int32)
            color=owner.color
            audit=owner.color_step(gradient,tickets=tickets,audit=True)
            powers=serial_color(owner,medium,gradient,tickets,color)
            actual=audit['powers'].cpu().tolist()
            cutoffs=audit['cutoffs'].cpu().tolist()
            for lane,value in powers.items():
                assert actual[lane]==pytest.approx(value,abs=2e-10)
                # One-count tolerance records numerical exponential error;
                # it is not equality with the unlimited-bit CPU sampler.
                with localcontext() as ctx:
                    ctx.prec=70
                    exact=int((Decimal(2)**(-Decimal.from_float(value)))*COIN_LEVELS)
                assert abs(cutoffs[lane]-exact)<=1
            assert_body(owner,medium)


def test_cuda_matching_zero_stock_blocks_internal_upward_edit():
    owner,medium=make([(1,1,1),(2,2,2)],1,2,stock=0)
    # Lane has an upward block whose chosen candidate has the same mass.
    tickets=torch.tensor([[1,0,0]],device='cuda',dtype=torch.int32)
    owner.color_step(torch.zeros_like(owner.parameter),tickets=tickets)
    assert_body(owner,medium)


def test_cuda_matching_checkpoint_rng_continuation_and_rejected_mutation():
    states=[(2,1,2),(0,0,0),(1,1,1),(1,2,2),(2,2,2),(0,1,0)]
    a,_=make(states,2,3,stock=1)
    gradient=torch.tensor([1.,-2.,3.,-4.,5.,-6.],device='cuda').reshape(2,3)
    a.sweep(gradient)
    saved=deepcopy(a.state_dict())
    b,_=make(states,2,3,stock=1,seed=908)
    b.load_state_dict(saved)
    for _ in range(3):
        a.sweep(gradient);b.sweep(gradient)
        sa,sb=a.state_dict(),b.state_dict()
        for key in sa:
            if isinstance(sa[key],torch.Tensor):assert torch.equal(sa[key],sb[key])
            else:assert sa[key]==sb[key]
        assert torch.equal(a.parameter,b.parameter)
    for mutation in ('supply','fee_units','codes','rng'):
        bad=deepcopy(b.state_dict());before=deepcopy(b.state_dict())
        if mutation=='codes':bad['codes'][0]=27
        elif mutation=='rng':bad['rng']=torch.zeros(3,dtype=torch.uint8)
        else:bad[mutation]+=1
        with pytest.raises((ValueError,RuntimeError)):
            b.load_state_dict(bad)
        after=b.state_dict()
        assert all(torch.equal(before[k],after[k]) if isinstance(before[k],torch.Tensor)
                   else before[k]==after[k] for k in before)


def test_cuda_matching_nondefault_stream_and_no_moments():
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        owner,medium=make([(1,1,1)]*6,2,3,stock=1)
        gradient=torch.full_like(owner.parameter,-8.)
        tickets=torch.tensor([[1,467,0],[0,467,0]],device='cuda',dtype=torch.int32)
        color=owner.color
        owner.color_step(gradient,tickets=tickets)
    stream.synchronize()
    serial_color(owner,medium,gradient,tickets,color)
    assert_body(owner,medium)
    assert owner.codes.numel()*owner.codes.element_size()==6
    assert owner.stock.numel()==1 and owner.row_p.numel()+owner.col_q.numel()==5
    assert not any(k in vars(owner) for k in ('moments','age','original_weight','history'))


def test_cuda_matching_invalid_input_precedes_rng_and_mutation():
    owner,_=make([(1,1,1)]*6,2,3,stock=1)
    for gradient in (owner.parameter.detach(),torch.full_like(owner.parameter,float('nan'))):
        before=deepcopy(owner.state_dict())
        with pytest.raises(ValueError):owner.color_step(gradient)
        after=owner.state_dict()
        assert all(torch.equal(before[k],after[k]) if isinstance(before[k],torch.Tensor)
                   else before[k]==after[k] for k in before)
    gradient=torch.zeros_like(owner.parameter)
    with pytest.raises(ValueError):
        owner.color_step(gradient,tickets=torch.full((2,3),-1,device='cuda',dtype=torch.int32))


def test_cuda_matching_prefix_reject_release_then_fund_later_lane():
    states=[(0,0,0)]*9
    states[4]=(2,2,2)
    owner,medium=make(states,3,3,stock=0)
    gradient=torch.zeros_like(owner.parameter)
    gradient[0,0]=-8.;gradient[1,1]=8.;gradient[2,2]=-8.
    tickets=torch.tensor([[1,1,0],[0,0,0],[1,1,0]],device='cuda',dtype=torch.int32)
    owner.color_step(gradient,tickets=tickets)
    serial_color(owner,medium,gradient,tickets,0)
    assert_body(owner,medium)
    assert owner.codes[0].item()==0  # first intake was unfunded
    assert sum(medium.body.get(4))==5  # actual middle release
    assert sum(medium.body.get(8))==1 and owner.stock.item()==0


def test_cuda_matching_missing_gradient_still_runs_physics():
    states=[(1,1,1)]*4
    a,_=make(states,2,2,stock=2)
    b,_=make(states,2,2,stock=2)
    a.sweep(None)
    b.sweep(torch.zeros_like(b.parameter))
    assert torch.equal(a.codes,b.codes) and torch.equal(a.parameter,b.parameter)
    assert torch.equal(a.generator.get_state(),b.generator.get_state())


def test_cuda_matching_graph_preserves_changed_gradient_rng_and_resume():
    states=[(1,1,1),(2,2,2),(0,0,0),(1,2,2),(2,1,2),(0,1,0)]
    a,_=make(states,2,3,stock=1)
    b,_=make(states,2,3,stock=1)
    gradient=torch.ones_like(a.parameter)
    before=deepcopy(a.state_dict())
    a.capture_sweep(gradient)
    after=a.state_dict()
    assert all(torch.equal(before[k],after[k]) if isinstance(before[k],torch.Tensor)
               else before[k]==after[k] for k in before)
    for t in range(4):
        gradient.fill_((-1 if t%2 else 1)*(t+1))
        a.graph_sweep(gradient)
        b.sweep(gradient)
        sa,sb=a.state_dict(),b.state_dict()
        assert all(torch.equal(sa[k],sb[k]) if isinstance(sa[k],torch.Tensor)
                   else sa[k]==sb[k] for k in sa)
        assert torch.equal(a.parameter,b.parameter)
    a.load_state_dict(deepcopy(b.state_dict()))
    with pytest.raises(ValueError):a.graph_sweep(gradient)
    a.capture_sweep(gradient)
    a.graph_sweep(gradient);b.sweep(gradient)
    assert torch.equal(a.parameter,b.parameter)
    assert torch.equal(a.generator.get_state(),b.generator.get_state())


def test_cuda_matching_raw_writes_update_autograd_version_and_guard_graph_storage():
    owner,_=make([(1,1,1)]*4,2,2,stock=1)
    gradient=torch.ones_like(owner.parameter)
    owner.capture_sweep(gradient)
    version=owner.parameter._version
    old_loss=owner.parameter.square().sum()
    owner.graph_sweep(gradient)
    assert owner.parameter._version==version+1
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):
        old_loss.backward()
    gradient.set_(gradient.clone())
    before=deepcopy(owner.state_dict())
    with pytest.raises(ValueError):owner.graph_sweep(gradient)
    after=owner.state_dict()
    assert all(torch.equal(before[k],after[k]) if isinstance(before[k],torch.Tensor)
               else before[k]==after[k] for k in before)


def equal_nested(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal_nested(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_nested(x,y) for x,y in zip(a,b))
    return a==b


def parameter_list():
    return [torch.nn.Parameter(torch.linspace(-.4,.4,n,device='cuda').reshape(shape))
            for shape,n in [((2,3),6),((12,),12),((2,2,2),8)]]


def test_cuda_optimizer_nearest_initialization_and_independent_compartments():
    params=parameter_list();before=[p.detach().clone() for p in params]
    opt=CudaMaterialOptimizer(params,quantum=F(1,8),energy_unit=F(1,128),seed=18)
    assert isinstance(opt,torch.optim.Optimizer) and opt.state=={}
    for p,old,entry in zip(params,before,opt._entries):
        expected=((old+.375)/.125).round().clamp(0,6)*.125-.375
        assert torch.equal(p,expected)
        owner=entry['owner'];assert owner.codes.numel()==p.numel()
        assert torch.equal(owner.parameter.view_as(p),p)
        assert owner.stock.item()==p.numel()
    assert [(e['owner'].rows,e['owner'].cols) for e in opt._entries]==[(2,3),(3,4),(2,4)]


@pytest.mark.parametrize('use_graph',[False,True])
def test_cuda_optimizer_actual_backward_zero_grad_and_full_restore(use_graph):
    params=parameter_list()
    opt=CudaMaterialOptimizer(params,quantum=F(1,8),energy_unit=F(1,128),seed=7,use_graph=use_graph)
    def update(parameters,optimizer):
        optimizer.zero_grad()
        # Analytic autograd/interface check, not a convergence experiment.
        loss=sum((p*p).sum() for p in parameters)
        loss.backward();optimizer.step()
        return [p.detach().clone() for p in parameters]
    update(params,opt)
    snapshot=deepcopy(opt.state_dict());saved_parameters=[p.detach().clone() for p in params]
    other=[torch.nn.Parameter(p.clone()) for p in saved_parameters]
    restored=CudaMaterialOptimizer(other,quantum=F(1,8),energy_unit=F(1,128),seed=951,use_graph=not use_graph)
    restored.load_state_dict(snapshot)
    assert equal_nested(opt.state_dict(),restored.state_dict())
    for _ in range(3):
        actual=update(params,opt);expected=update(other,restored)
        assert all(torch.equal(a,b) for a,b in zip(actual,expected))
        assert equal_nested(opt.state_dict(),restored.state_dict())
    assert opt.state=={} and opt.state_dict()['state']=={}


def test_cuda_optimizer_last_bad_gradient_or_checkpoint_is_transactional():
    params=parameter_list();opt=CudaMaterialOptimizer(params,quantum=F(1,8),seed=1)
    for p in params:p.grad=torch.ones_like(p)
    before=deepcopy(opt.state_dict());cache=[p.detach().clone() for p in params]
    params[-1].grad.fill_(float('nan'))
    with pytest.raises(ValueError):opt.step()
    assert equal_nested(before,opt.state_dict())
    assert all(torch.equal(p,q) for p,q in zip(params,cache))
    bad=deepcopy(before);bad['cuda_material']['compartments'][-1]['codes'][0]=27
    with pytest.raises(ValueError):opt.load_state_dict(bad)
    assert equal_nested(before,opt.state_dict())
    assert all(torch.equal(p,q) for p,q in zip(params,cache))


def test_cuda_optimizer_lr_change_and_missing_gradients_keep_full_continuation():
    a=parameter_list();b=parameter_list()
    opt=CudaMaterialOptimizer(a,quantum=F(1,8),seed=911,use_graph=True)
    reference=CudaMaterialOptimizer(b,quantum=F(1,8),seed=911,use_graph=False)
    for lr in (1.,.5,0.,2.):
        opt.param_groups[0]['lr']=reference.param_groups[0]['lr']=lr
        for pa,pb in zip(a,b):pa.grad=torch.ones_like(pa);pb.grad=torch.ones_like(pb)
        a[-1].grad=b[-1].grad=None
        opt.step();reference.step()
        assert equal_nested(opt.state_dict(),reference.state_dict())
        assert all(torch.equal(pa,pb) for pa,pb in zip(a,b))
    reference.load_state_dict(deepcopy(opt.state_dict()))
    assert equal_nested(opt.state_dict(),reference.state_dict())


def test_cuda_optimizer_group_units_and_alias_rejected_before_initial_mutation():
    p=torch.nn.Parameter(torch.ones(2,3,device='cuda'))
    with pytest.raises(ValueError):
        CudaMaterialOptimizer([p,p],quantum=F(1,8))
    assert torch.equal(p,torch.ones_like(p))
    with pytest.raises(ValueError):
        CudaMaterialOptimizer([p],quantum=F(1,8),stock_per_cell=1<<62)
    assert torch.equal(p,torch.ones_like(p))
    params=parameter_list();opt=CudaMaterialOptimizer(params,quantum=F(1,8),seed=1)
    params[0].grad=params[0].detach()
    before=deepcopy(opt.state_dict())
    with pytest.raises(ValueError):opt.step()
    assert equal_nested(before,opt.state_dict())
