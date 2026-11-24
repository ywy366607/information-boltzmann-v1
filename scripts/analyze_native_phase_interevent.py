"""Track qualified real S/P phase during quiet steps and between them."""
import argparse,collections,gzip,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    samples=[];kindmap={}
    for seg,d in enumerate(sorted(a.work.glob('collect_observations*'))):
        with gzip.open(d/'operations.jsonl.gz','rt',encoding='utf8') as f:
            for line in f:
                r=json.loads(line)
                if r['kind']=='event':kindmap[(seg,r['event'])]=r['data'].get('autonomous_event',{}).get('kind')
        for f in sorted(d.glob('physical_*.npz')):
            with np.load(f,allow_pickle=False) as z:
                for i,k in enumerate(z['kind']):
                    if k in ('quiet_before','quiet_after'):
                        samples.append((seg,int(z['event'][i]),str(k),z['complex_overlap_real'][i]+1j*z['complex_overlap_imag'][i],z['omega'][i],float(z['dt'][i])))
    ii,jj=np.triu_indices(8,1);cycles=np.zeros((32,8));qualified=np.zeros((32,28),int)
    for seg,event,k,c,w,dt in samples:
        if k=='quiet_after':cycles+=w*dt/(2*np.pi);qualified+=(abs(c[:,ii,jj])>=.1)
    eligible=(qualified>=10)&(cycles[:,ii]>=1)&(cycles[:,jj]>=1);result=[]
    for layer,edge in np.argwhere(eligible):
        prev=None;groups=collections.defaultdict(list); angles=[]
        for seg,event,k,c,w,dt in samples:
            value=c[layer,ii[edge],jj[edge]]
            if k=='quiet_after' and abs(value)>=.1:angles.append(float(np.angle(value)))
            if prev and prev[0]==seg and min(abs(prev[3]),abs(value))>=.1:
                change=float(abs(np.angle(value*np.conj(prev[3]))))
                if prev[2]=='quiet_before' and k=='quiet_after':group='within_quiet'
                elif prev[2]=='quiet_after' and k=='quiet_before':group='between_'+str(kindmap.get((seg,prev[1])))+'_to_'+str(kindmap.get((seg,event)))
                else:group=None
                if group:groups[group].append(change)
            prev=(seg,event,k,value)
        result.append(dict(layer=int(layer),heads=[int(ii[edge]),int(jj[edge])],
            cycles=[float(cycles[layer,ii[edge]]),float(cycles[layer,jj[edge]])],
            qualified_samples=len(angles),phase_concentration=float(abs(np.exp(1j*np.array(angles)).mean())),
            changes={name:dict(count=len(v),median_abs_radians=float(np.median(v))) for name,v in groups.items()}))
    out=dict(pairs=result,note='Cycle counts cover the full integrated quiet time, not necessarily the qualified subset; no continuity across recording segments. Interevent gaps can include multiple actual operations and are not isolated interventions.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out))


if __name__=='__main__':main()
