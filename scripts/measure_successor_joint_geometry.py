"""Matched-marginal joint laws evaluated on native cells of frozen real bodies."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtri,gammaincinv
from scipy.stats import qmc

from scripts.ib_local.successor_address_geometry import addresses
from scripts.ib_local.successor_joint_reference import coordinated_scale,multivariate_score,precision_consensus


def measure(capsule,ridge,power):
    case=next(c for c in capsule['cases'] if c['name']=='actual_response')
    d=case['distribution'];mean=d['mean'].double();df=float(d['df']);width=mean.shape[-1]
    scale=coordinated_scale(case,ridge)
    diag_error=float((scale.diag()-d['scale2'][0]).abs().max())
    assert diag_error<3e-5,diag_error
    uniform=qmc.Sobol(d=width+1,scramble=True,seed=191119).random_base2(power)
    normal=torch.from_numpy(ndtri(uniform[:,:width]))
    radial=torch.from_numpy(np.sqrt(2*gammaincinv(df/2,uniform[:,-1])/df))[:,None]
    output=[];joint_mean,joint_scale=precision_consensus(case,ridge)
    for name,location,matrix in [('diagonal_joint',mean,torch.diag(scale.diag())),
                                 ('coherent_joint',mean,scale),
                                 ('joint_precision',joint_mean,joint_scale)]:
        points=(location+normal@torch.linalg.cholesky(matrix).T/radial).float()
        assigned=addresses(case['geometry'],points)['identity']
        unique,count=assigned.unique(return_counts=True)
        mass={str(int(i)):int(n)/len(points) for i,n in zip(unique,count)}
        output.append(dict(name=name,mass=mass,actual_relation_mass=mass.get(str(capsule['actual_relation']),0.),
            fresh_mass=mass.get('-1',0.),
            actual_joint_log_score=float(multivariate_score(location,matrix,df,case['actual_latent'])[0]),
            mean_shift=float((location-mean).norm()),
            median_norm=float(points.norm(dim=-1).median())))
    eig=torch.linalg.eigvalsh(scale)
    return dict(cycle=capsule['cycle'],actual_relation=capsule['actual_relation'],cases=output,
        diagonal_error=diag_error,eigen_min=float(eig[0]),eigen_max=float(eig[-1]),
        effective_rank=float(eig.sum().square()/eig.square().sum()),samples=2**power)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ridge',type=float,required=True);p.add_argument('--power',type=int,default=14)
    a=p.parse_args();torch.set_num_threads(1)
    rows=[measure(torch.load(path,map_location='cpu',weights_only=False),a.ridge,a.power)
          for path in sorted(a.work.glob('geometry_*.pt'))]
    result=dict(observations=rows,training_updates=0,world_calls=0,
        purpose='same marginals and mean, measured coherent residual covariance versus diagonal control')
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
