"""Compile every original pair to explicit mechanism requirements, without cuts.

This is the anatomical input to a mechanism linker, not a claim that an
unresolved receptor or local electrical endpoint has been implemented.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROUTES = {
    'fast_cell_baseline':1,
    'dopamine_delivery':2,
    'octopamine_delivery':4,
    'serotonin_delivery':8,
    'KC_terminal_release':16,
    'KC_MBON_contact_learning':32,
    'KC_DAN_local_feedback':64,
    'KC_KC_axonal_receptor':128,
    'APL_local_release':256,
    'DPM_local_cotransmission':512,
    'sensory_local_return_required':1024,
}


def checksum(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def compile_routes(ids,pre,post,annotations,transmitters,dan_ids):
    """Each original pair keeps one row; multiple chemical roles are bit flags."""
    annotation=annotations.set_index('bodyId').reindex(ids)
    nt=transmitters.set_index('body')['consensus_nt'].reindex(ids).fillna('unclear').to_numpy()
    cls=annotation['class'].fillna('').astype(str).to_numpy()
    typ=annotation['type'].fillna('').astype(str).to_numpy()
    sc=annotation['superclass'].fillna('').to_numpy(dtype=str)
    sensory=np.char.find(sc,'sensory')>=0
    kc=cls=='Kenyon_Cell';mbon=cls=='MBON'
    dan=np.isin(ids,dan_ids)
    apl=typ=='APL';dpm=typ=='DPM'
    # DPM's existing reviewed cotransmission must not be overwritten by the
    # uncorrected transmitter prediction in the source table.
    oa=nt=='octopamine';serotonin=(nt=='serotonin') | dpm
    rows=np.zeros(pre.size,dtype=np.uint16)
    counts={}
    selectors={
        'fast_cell_baseline':lambda a,b:~sensory[b] & ~dan[a] & ~oa[a] & ~serotonin[a] & ~kc[a] & ~apl[a] & ~dpm[a],
        'dopamine_delivery':lambda a,b:dan[a],
        'octopamine_delivery':lambda a,b:oa[a],
        'serotonin_delivery':lambda a,b:serotonin[a],
        'KC_terminal_release':lambda a,b:kc[a],
        'KC_MBON_contact_learning':lambda a,b:kc[a] & mbon[b],
        'KC_DAN_local_feedback':lambda a,b:kc[a] & dan[b],
        'KC_KC_axonal_receptor':lambda a,b:kc[a] & kc[b],
        'APL_local_release':lambda a,b:apl[a],
        'DPM_local_cotransmission':lambda a,b:dpm[a],
        'sensory_local_return_required':lambda a,b:sensory[b],
    }
    # Bounded boolean/gather workspace; no edge-by-mechanism dense matrix.
    for start in range(0,len(pre),500_000):
        end=min(start+500_000,len(pre));a,b=pre[start:end],post[start:end]
        value=rows[start:end]
        for name,selector in selectors.items():
            selected=selector(a,b)
            value[selected] |= ROUTES[name]
            counts[name]=counts.get(name,0)+int(selected.sum())
    if np.any(rows==0):
        raise ValueError('Unassigned anatomical pairs; no implicit fallback allowed')
    if np.any((rows & ROUTES['fast_cell_baseline'] != 0) & sensory[post]):
        raise ValueError('Sensory return incorrectly assigned to somatic baseline')
    return rows,counts,annotation,nt


def main(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new immutable compilation directory')
    args.output.mkdir(parents=True,exist_ok=True)
    with np.load(args.master,allow_pickle=False) as master,np.load(args.current,allow_pickle=False) as current:
        ids=master['neuron_body_ids']
        if not np.array_equal(ids,current['neuron_body_ids']):
            raise ValueError('Cell order changed; continuing states cannot be silently remapped')
        pre,post=master['edge_pre'],master['edge_post']
        ann=pd.read_feather(args.annotations,columns=['bodyId','status','superclass','class','type',
            'receptorType','birthtime','trumanHl','entryNerve','exitNerve'])
        nt=pd.read_feather(args.transmitters,columns=['body','consensus_nt'])
        bits,counts,annot,ligands=compile_routes(ids,pre,post,ann,nt,ids[current['dan_indices']])
        np.savez_compressed(args.output/'program.npz',neuron_body_ids=ids,route_bits=bits)
        neurons=annot.reset_index(names='bodyId')
        neurons['source_consensus_nt']=ligands
        neurons.to_feather(args.output/'cell_program_inputs.feather')
        master_edges=len(pre)
    report={'created_at':datetime.now(timezone.utc).isoformat(),'version':1,
        'master_graph':str(args.master.resolve()),'master_sha256':checksum(args.master),
        'current_graph':str(args.current.resolve()),'current_sha256':checksum(args.current),
        'neuron_count':len(ids),'master_pair_rows':master_edges,'compiled_pair_rows':len(bits),
        'unassigned_pair_rows':int((bits==0).sum()),'all_original_pairs_retained':len(bits)==master_edges,
        'route_opcodes':ROUTES,'route_requirement_counts':counts,
        'program_sha256':checksum(args.output/'program.npz'),
        'new_route_state_bytes':0,'static_route_program_uncompressed_bytes':bits.nbytes,
        'state_identity_rule':'Original neuron order and master pair row identity; no morphology or live states pooled',
        'sensory_return_status':'Explicit requirements, awaiting contact-local endpoint/receptor binding; NOT somatic currents',
        'KC_pair_status':'Requirement includes both calyx and axonal contacts; execution must split using actual contact locations, not apply muscarinic inhibition to every pair',
        'fast_baseline_status':'Existing general cell dynamics, not a declaration of fully known specialized physiology',
        'runtime_binding_complete':False,'runtime_changed':False,'checkpoint_changed':False,
        'scope':'Complete anatomical mechanism requirement program. This is a compiler input; unresolved execution binding is not activation.'}
    (args.output/'program.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--master',type=Path,default=Path('data/malecns_v1/fly_reservoir_delayed.npz'))
    p.add_argument('--current',type=Path,default=Path('data/malecns_v1/fly_reservoir_coba_r2.npz'))
    p.add_argument('--annotations',type=Path,default=Path('data/malecns_v1/body-annotations.feather'))
    p.add_argument('--transmitters',type=Path,default=Path('data/malecns_v1/body-neurotransmitters.feather'))
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
