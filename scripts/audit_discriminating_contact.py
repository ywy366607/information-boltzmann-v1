"""2594: value of hearing an answer on the fixed, pre-world root menu.

Columns are the body's two shared operator draws (including their paired
quadrature noise in the continuation payoff), not two actual future lives.
The observable channel uses the same conditional Gaussian as hypothesis_evidence.
"""
import gzip
import io
import json
import math
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.special import ndtr
import torch

from scripts.audit_revisit_exchange import alternate, natural, row, fact
from scripts.ib_local.navigation_causal_program import score_frozen
from scripts.ib_local.causal_relation_program import hypothesis_evidence

NAMES = ('silence', 'generate', 'reuse')


def hearing(payoff, distance):
    """Exact E[max posterior payoff] for two equiprobable equal-noise normals."""
    v = np.asarray(payoff, dtype=float)
    blind = float(v.mean(1).max())
    if distance == 0:
        return dict(blind=blind, adaptive=blind, gain=0., regions=[])
    cuts = {0., 1.}
    for a, b in combinations(v, 2):
        d0, d1 = a-b
        if d0*d1 < 0:
            cuts.add(float(d0/(d0-d1)))
    # P(H=1|answer) is logistic(L). Under the two hypotheses the log
    # likelihood ratio L is N(-D²/2,D²) or N(D²/2,D²).
    def cdfs(p):
        if p in (0., 1.): return np.full(2, p)
        return ndtr(math.log(p/(1-p))/distance + np.array([.5, -.5])*distance)
    adaptive, regions = 0., []
    cuts = sorted(cuts)
    for lo, hi in zip(cuts, cuts[1:]):
        p = (lo+hi)/2
        a = int((v @ [1-p, p]).argmax())
        mass = .5*(cdfs(hi)-cdfs(lo))
        adaptive += float(mass @ v[a]); regions.append(dict(p1=[lo,hi], action=a, mass=float(mass.sum())))
    return dict(blind=blind, adaptive=adaptive, gain=adaptive-blind, regions=regions)


def archive_forecast():
    # Extract only tensors frozen by the original _relation_value before the
    # actor/world call. No post-update model is used; no temporary disk image.
    with gzip.open(alternate/'body/last.pt.gz', 'rb') as src, io.BytesIO() as buf:
        shutil.copyfileobj(src, buf, 2**20); buf.seek(0)
        saved = torch.load(buf, weights_only=False, map_location='cpu')
    assert saved['cycle']==2594 and saved['config']['epistemic_value']['schema']==5
    eng = saved['engine']; plan = eng['_program_forecasts']
    return eng['_program_mother_forecast'], plan['hypotheses'], plan['path']['mother']['noise']


def forecast(root, frozen, centers, noise):
    e = root['causal_use']['epistemic_value']; routes=e['successor_routes']
    v = np.array([[r['expected'][0] for r in action] for action in routes])
    assert v.shape==(3,2) and centers.shape[:2]==(3,2) and min(root['choice_probability'])>0
    np.testing.assert_allclose(v.mean(1), e['contextual_future'], atol=2e-7)
    torch.testing.assert_close(centers.mean(1),frozen['mean'],atol=2e-6,rtol=1e-5)
    result={}
    for i,k in enumerate(NAMES):
        d=float(((centers[i,1]-centers[i,0]).double().square()/noise.double()).sum().sqrt())
        result[k]=dict(distance=d, **hearing(v,d))
    return dict(menu=NAMES, continuation_by_shared_draw=v.tolist(), full_information=float(v.max(0).mean()),
                hearing=result, contextual_total=e['contextual_total'], actor_probability=root['choice_probability'])


def main():
    root,alt=row(natural,2594),row(alternate,2594)
    e=root['causal_use']['epistemic_value']; assert e['contextual_future']==alt['causal_use']['epistemic_value']['contextual_future']
    assert root['joint_language_search']['candidates']==alt['joint_language_search']['candidates']
    frozen,centers,noise=archive_forecast()
    prediction=forecast(root,frozen,centers,noise)
    reveal={}
    for actual, folder, r in [('generate',alternate,alt),('reuse',natural,root)]:
        assert r['choice']==actual
        y=fact(folder,r['causal_program']['committed_episode_identity']).outcome
        scores={}
        for i,k in enumerate(NAMES):
            f=dict(frozen,mean=frozen['mean'][i:i+1],scale2=frozen['scale2'][i:i+1]);scores[k]=score_frozen(f,y)['log_score']
        assert abs(scores[actual]-r['causal_program']['mother_prequential']['log_score'])<3e-6
        z=torch.nn.functional.linear(y,frozen['encoder_weight']);i=NAMES.index(actual)
        posterior=hypothesis_evidence(centers[i:i+1],noise,z).double().exp().flatten().tolist()
        values=np.asarray(prediction['continuation_by_shared_draw'])@posterior
        reveal[actual]=dict(response_scores=scores,posterior=posterior,fixed_menu_best=NAMES[values.argmax()])
    later=json.loads(Path('results/published/cgb_active_evidence_quality_2594_20261120.json').read_text())
    assert later['signature']==root['shared_unknown']['signature']==alt['shared_unknown']['signature']
    report=dict(cycle=2594,revisit=2605,question=later['question'],prediction=prediction,observations=reveal,
        later_benefits={k:v['benefit'] for k,v in later['actions'].items()},actual_choice=root['choice'],
        scope='fixed root menu, two common quadrature draws, conditional continuation only; not the next live actor',
        world_calls=0,updates=0)
    out=Path('results/published/cgb_discriminating_contact_2594_20261121.json');out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report))


if __name__=='__main__':
    torch.set_num_threads(1); main()
