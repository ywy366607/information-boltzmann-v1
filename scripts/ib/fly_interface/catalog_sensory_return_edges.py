"""Catalog original sensory-return contacts without restoring somatic currents."""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd


def main(args):
    if args.output.exists():
        raise ValueError('Preserve earlier catalogs; use a fresh path')
    args.output.mkdir(parents=True)
    chunks=[]
    with np.load(args.original,allow_pickle=False) as original, np.load(args.current,allow_pickle=False) as current:
        ids=original['neuron_body_ids']
        assert np.array_equal(ids,current['neuron_body_ids'])
        sc=original['superclass_names'].astype(str)[original['superclass_id']]
        sensory=np.char.find(sc,'sensory')>=0
        for channel in ('e','i'):
            post=original['edge_post_'+channel]
            slots=np.flatnonzero(sensory[post])
            pre=original['edge_pre_'+channel][slots]
            target=post[slots]
            splits=original['delay_splits_'+channel]
            delay=np.searchsorted(splits[1:],slots,side='right')+1
            weight=original['edge_weight_'+channel][slots]
            chunks.append(pd.DataFrame({'body_pre':ids[pre],'body_post':ids[target],
                'source_graph_edge':slots,'original_channel':channel,'delay_tier':delay,
                'reference_original_weight':weight,'target_superclass':sc[target]}))
            # The current point-cell graph actually excludes all such fast
            # edges; this catalog does not claim that other runtime pathways
            # have or have not restored their anatomical local effects.
            assert not sensory[current['edge_post_'+channel]].any()
        total_current=sum(current[name].size for name in ('edge_pre_e','edge_pre_i','dan_edge_pre'))
        total_original=sum(original[name].size for name in ('edge_pre_e','edge_pre_i','dan_edge_pre'))
    table=pd.concat(chunks,ignore_index=True)
    table.to_feather(args.output/'sensory_return_edges.feather')
    report={'original_graph':str(args.original),'current_graph':str(args.current),
        'original_all_channels_edges':int(total_original),'current_all_channels_edges':int(total_current),
        'sensory_return_pairs':len(table),
        'source_channel_counts':table.original_channel.value_counts().to_dict(),
        'target_superclass_counts':table.target_superclass.value_counts().to_dict(),
        'total_count_gap_matches_sensory_catalog':total_original-total_current==len(table),
        'scope':'Anatomical routing candidates only. Counts are not full edge-set equality; current specialized-module coverage still needs endpoint audit.',
        'runtime_changed':False,'checkpoint_changed':False,
        'requirements':['Obtain actual postsynaptic contact locations and justified receptor/effectors',
            'Route local axonal/dendritic effects without turning all feedback into somatic excitation',
            'Reference old weights are provenance, not current learned parameters or an approved reinitialization']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--original',type=Path,default=Path('data/malecns_v1/fly_reservoir_coba.npz'))
    p.add_argument('--current',type=Path,default=Path('data/malecns_v1/fly_reservoir_coba_r2.npz'))
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
