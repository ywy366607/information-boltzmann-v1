"""Observed runs and native conditional projection, without simulated data."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.ib.generative.r7_lifetime_constraints import observed_intervals


def main(base):
    folder=base/'cgb_r7_lifetime_constraints_20261022'
    report=json.loads((folder/'report.json').read_text())
    native=json.loads((base/'cgb_r7_generator_bridge_20261022/report.json').read_text())
    obs=json.loads((base/'cgb_r7_observation_audit_20261022/observations.json').read_text())
    control=sorted([w for w in obs['workbooks'] if w['kind']=='filopodium' and w['genotype']=='Control'],key=lambda w:w['file'])
    duration,ended=observed_intervals(control[2:])
    terminal=duration+ended
    empirical=[1.];value=1.
    for time in range(1,60):
        risk=np.sum(terminal>=time)
        exits=np.sum((terminal==time)&(ended>0))
        if risk:value*=1-exits/risk
        empirical.append(value)
    time=np.arange(60)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.2))
    ax=axes[0]
    ax.step(time,empirical,where='post',color='#222222',label='Observed CG3 (72 incident tracks)')
    for family,color in [('single','#cc7953'),('mixture','#298d8d')]:
        m=report['models'][family]['parameters']
        survival=np.sum(np.asarray(m['weights_at_first_observation'])[:,None]
                        *np.exp(-np.asarray(m['rates_per_minute'])[:,None]*time),axis=0)
        ax.plot(time,survival,color=color,label='Single timescale' if family=='single' else 'Two timescales')
    row=next(r for r in native['rows'] if r['total_support']==native['selected_support'])
    matrix=np.asarray(row['transient_generator'])*row['clock_reference_ms_per_minute']
    initial=np.array([row['initially_unbound_probability'],1-row['initially_unbound_probability']])
    survival=[float(initial@expm(matrix*t)@np.ones(2)) for t in time]
    ax.plot(time,survival,color='#7666a5',linestyle='--',label='Native support sector (exploratory)')
    ax.set(xlabel='Minutes after first observation',ylabel='Remaining-run survival',ylim=(-.02,1.03),xlim=(0,59))
    ax.legend(frameon=False,fontsize=8)
    ax.set_title('A   Same grouped observations, explicit censoring',loc='left',fontsize=10)
    ax=axes[1]
    for i,w in enumerate(control):
        ax.step(time,w['counts'],where='post',label=f"CG{i+1} ({'fit' if i<2 else 'held out'})",linewidth=1.3)
    ax.set(xlabel='Actual observation minute',ylabel='Visible filopodia count',xlim=(0,59))
    ax.legend(frameon=False,fontsize=8)
    ax.set_title('B   Actual counts: no independent-frame error bars',loc='left',fontsize=10)
    fig.suptitle('Real R7 development constrains a candidate generator',fontsize=13)
    fig.text(.5,.015,'Filo2 public observations, fixed commit 872d059; three control growth cones. '
             'Conditional projection does not establish the full biological law.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94))
    fig.savefig(folder/'r7_constraints.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--base',type=Path,default=Path('F:/fly_stream_overflow'))
    main(p.parse_args().base)
