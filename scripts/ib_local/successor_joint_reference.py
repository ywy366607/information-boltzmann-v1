"""Read-only joint-output candidates for the existing predictive marginals.

The residual evidence-space solve yields a PSD scatter Y' (I-XA^-1X') Y.
Its diagonal is the residual_sum already used by posterior(). This reference
retains its off-diagonal entries, then coordinates program disagreement with
outer products. No observed successor is used to construct these matrices.
"""
import torch


def residual_scatter(x,y,ridge):
    x,y=x.double(),y.double()
    kernel=x@x.T+ridge*torch.eye(len(x),device=x.device,dtype=x.dtype)
    return ridge*y.T@torch.linalg.solve(kernel,y)


def scale_matrix(program,ridge,*,prior_correlation=None):
    scatter=residual_scatter(program['x'],program['y'],ridge)
    df=float(program['df'])
    noise=program['noise'].double()
    if prior_correlation is None:
        off=scatter-torch.diag(scatter.diag())
        full_noise=torch.diag(noise)+off/df
    else:
        prior=(noise*df-scatter.diag()).clamp_min(torch.finfo(noise.dtype).eps)
        prior_matrix=prior.sqrt()[:,None]*prior_correlation*prior.sqrt()[None,:]
        full_noise=(prior_matrix+scatter)/df
    scale=full_noise*(1+program['leverage'].double().reshape(-1)[0])
    return (scale+scale.T)/2


def coordinated_scale(case,ridge,*,prior_correlation=None):
    mean=case['distribution']['mean'].double()[0]
    weights=case['extra']['weights'].double()
    weights=weights/weights.sum()
    scale=torch.zeros(len(mean),len(mean),dtype=mean.dtype)
    for weight,program in zip(weights,case['extra']['programs']):
        delta=program['mean'].double()[0]-mean
        scale+=weight*(scale_matrix(program,ridge,prior_correlation=prior_correlation)+delta[:,None]*delta[None,:])
    return (scale+scale.T)/2


def precision_consensus(case,ridge):
    """Joint quadratic reconciliation, reusing the two actual program weights."""
    programs=case['extra']['programs'];weights=case['extra']['weights'].double()
    weights=weights/weights.sum()
    scales=[scale_matrix(p,ridge) for p in programs]
    precision=[torch.linalg.inv(s) for s in scales]
    total=sum(w*p for w,p in zip(weights,precision))
    rhs=sum(w*(p@d['mean'].double()[0]) for w,p,d in zip(weights,precision,programs))
    mean=torch.linalg.solve(total,rhs)
    scale=sum(w*(s+(d['mean'].double()[0]-mean)[:,None]*(d['mean'].double()[0]-mean)[None,:])
              for w,s,d in zip(weights,scales,programs))
    return mean[None],(scale+scale.T)/2


def multivariate_score(mean,scale,df,target):
    delta=target.double()-mean.double()
    factor=torch.linalg.cholesky(scale)
    solved=torch.linalg.solve_triangular(factor,delta.T,upper=False).T
    d=delta.shape[-1];df=torch.as_tensor(df,device=delta.device,dtype=delta.dtype)
    return (torch.lgamma((df+d)/2)-torch.lgamma(df/2)-d/2*torch.log(df*torch.pi)-
        factor.diag().log().sum()-(df+d)/2*torch.log1p(solved.square().sum(-1)/df))/d
