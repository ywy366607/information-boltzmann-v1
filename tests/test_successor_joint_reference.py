"""Bind matrix residual scatter to the actual conjugate solver."""
import torch

from scripts.ib_local.causal_relation_program import posterior
from scripts.ib_local.successor_joint_reference import residual_scatter,scale_matrix


def test_matrix_diagonal_is_original_solver_and_rotates_with_outputs():
    rng=torch.Generator().manual_seed(91)
    x=torch.randn(7,5,generator=rng,dtype=torch.float64)
    y=torch.randn(7,12,generator=rng,dtype=torch.float64)
    query=torch.randn(1,5,generator=rng,dtype=torch.float64)
    d=posterior(x,y,query,.1,noise_prior=torch.full((12,),.2,dtype=x.dtype))
    full=scale_matrix(d,.1)
    torch.testing.assert_close(full.diag(),d['scale2'][0])
    assert torch.linalg.eigvalsh(full).min()>0
    rotation=torch.linalg.qr(torch.randn(12,12,generator=rng,dtype=x.dtype)).Q
    torch.testing.assert_close(residual_scatter(x,y@rotation,.1),
        rotation.T@residual_scatter(x,y,.1)@rotation)
