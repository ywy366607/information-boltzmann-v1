import torch

from scripts.clock_credit_jet import compose, first_arrival_prefix, mark


def test_ordered_matrix_jet_matches_clock_credit_finite_differences():
    torch.manual_seed(2211)
    values = [(torch.rand(3,3,dtype=torch.double),torch.rand(3,3,dtype=torch.double),
               torch.randn(3,3,dtype=torch.double),torch.randn(3,3,dtype=torch.double)) for _ in range(3)]
    jets = [mark(*row) for row in values]
    left, right = compose(compose(*jets[:2]),jets[2]), compose(jets[0],compose(*jets[1:]))
    for key in vars(left):
        torch.testing.assert_close(getattr(left,key),getattr(right,key))
    def evaluate(eta,delta):
        product = torch.eye(3,dtype=torch.double)
        for p,t,dp,dt in values:
            product = product@((p+delta*dp)*torch.exp(-eta*(t+delta*dt)))
        return product
    h = 1e-4
    mixed = -(evaluate(h,h)-evaluate(h,-h)-evaluate(-h,h)+evaluate(-h,-h))/(4*h*h)
    torch.testing.assert_close(left.mixed,mixed,atol=2e-6,rtol=2e-6)
    clock = -(evaluate(h,0)-evaluate(-h,0))/(2*h)
    torch.testing.assert_close(left.clock,clock,atol=1e-6,rtol=1e-6)


def test_first_arrival_keeps_time_dependent_operators_and_tail():
    probabilities = [.7,.4,.9]
    q = [mark(torch.tensor([[p]],dtype=torch.double),torch.tensor([[2.+i]],dtype=torch.double))
         for i,p in enumerate(probabilities)]
    r = [mark(torch.tensor([[1-p]],dtype=torch.double),torch.tensor([[.5]],dtype=torch.double))
         for p in probabilities]
    hit, tail = first_arrival_prefix(q,r)
    torch.testing.assert_close(hit.mass+tail.mass,torch.ones(1,1,dtype=torch.double))
    expected_tail = .7*.4*.9*(2+3+4)
    assert abs(float(tail.clock)-expected_tail) < 1e-12
    expected_hit = .3*.5+.7*.6*(2+.5)+.7*.4*.1*(2+3+.5)
    assert abs(float(hit.clock)-expected_hit) < 1e-12
