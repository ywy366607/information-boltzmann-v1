"""Analysis of measured native movement; no surrogate five-node complex.

Reports graph-Hodge amplitudes, complex matrix coherence and actual phase/time
coverage. Higher faces, native edge rotors and mode-specific usefulness remain
unresolved unless their own evidence has been obtained, not assigned zeros.
"""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import numpy as np


def analyze(directory):
    directory=Path(directory)
    keys=('quiet_time','dt','omega','node_energy','edge_weight','edge_flux','edge_graph_harmonic',
          'complex_overlap_real','complex_overlap_imag','temporal_overlap','phase_delta','theta')
    chunks={key:[] for key in keys}
    samples=0
    for path in sorted(directory.glob('physical_*.npz')):
        with np.load(path,allow_pickle=False) as data:
            selected=data['kind']=='quiet_after'
            for key in keys:chunks[key].append(data[key][selected])
            samples+=int(selected.sum())
    if not samples:raise ValueError('no actual quiet movement recorded')
    a={key:np.concatenate(value) for key,value in chunks.items()}
    ii,jj=np.triu_indices(a['node_energy'].shape[-1],1)
    overlap=a['complex_overlap_real'][:,:,ii,jj]+1j*a['complex_overlap_imag'][:,:,ii,jj]
    amplitude=abs(overlap)
    # Phase concentration is reported WITH overlap, never as the phases of
    # arbitrary scalar errors. This is per-layer physical head-pair coherence.
    qualified=amplitude>=.5
    numerator=np.where(qualified,overlap/np.maximum(amplitude,1e-30),0).sum(0)
    count=qualified.sum(0)
    plv=np.divide(abs(numerator),count,out=np.full_like(abs(numerator),np.nan),where=count>0)
    # Missing qualified observations are undefined, not zero synchrony.
    per_layer_plv=[float(np.nanmax(row)) if np.isfinite(row).any() else None for row in plv]
    # Occupation of graph cycle directions: NOT a high-dimensional Betti count
    # and NOT evidence that the actual flow protects those directions.
    flux_energy=(a['edge_flux'].astype(float)**2).sum(-1)
    harmonic_energy=(a['edge_graph_harmonic'].astype(float)**2).sum(-1)
    fraction=np.divide(harmonic_energy,flux_energy,out=np.zeros_like(harmonic_energy),where=flux_energy>1e-30)
    phase_path=(a['omega']*a['dt'][:,None,None]).sum(0)/(2*np.pi)
    kinds={};external=0;updates=0;operation_cells={};actual_version_uses=0
    with gzip.open(directory/'operations.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);kind=row['kind'];kinds[kind]=kinds.get(kind,0)+1
            if kind=='event':
                d=row['data'];external+=d.get('autonomous_event',{}).get('kind')=='external'
                updates+=d.get('updates',0)
                actual_version_uses+=len(d.get('joint_relation_update',{}).get('variable_uses',[]))
            if kind=='typed_execution':
                for call in row['data']['calls']:
                    name=call['operation'];operation_cells[name]=operation_cells.get(name,0)+1
    return dict(schema=1,scope='actual all-layer movement during complete learning; preliminary acquisition analysis',
        quiet_calls=samples,quiet_time=float(a['dt'].sum()),external_events=external,
        optimizer_updates=updates,operation_counts=kinds,actual_typed_operations=operation_cells,
        actual_version_uses=actual_version_uses,
        phase_coverage=dict(definition='integral of actual intrinsic omega over measured quiet time; not observed phase locking',
            periods_per_layer_head=phase_path.tolist(),minimum=float(phase_path.min()),
            maximum=float(phase_path.max()),eight_periods_covered=bool(phase_path.min()>=8),
            max_intrinsic_phase_per_recorded_call=float(abs(a['omega']*a['dt'][:,None,None]).max()),
            qualification='intrinsic sampling check only; actual matrix dephasing and coupling also require inspection'),
        node_edge_movement=dict(per_layer_graph_harmonic_fraction=fraction.mean(0).tolist(),
            per_layer_mean_complex_overlap=amplitude.mean((0,2)).tolist(),
            per_layer_max_qualified_pair_phase_concentration=per_layer_plv,
            qualified_pair_sample_counts=count.tolist(),
            per_layer_mean_temporal_matrix_overlap=a['temporal_overlap'].mean(0).tolist(),
            qualified_pair_sample_fraction=float(qualified.mean()),
            actual_head_weight_range=[float(a['edge_weight'].min()),float(a['edge_weight'].max())]),
        completion=dict(real_structure_and_life='recorded at actual operations; all 32 layers retained',
            structural_modes='graph-mode occupation only; preservation and collective mode identification still pending',
            native_sync='matrix overlaps measured; slow-time sufficiency checked separately',
            dirac='not evaluated: no identified native edge phase and validated native Dirac law',
            activity_regulates_structure='real weights/revisions/applied updates recorded; regulatory dynamics analysis pending',
            higher_faces='not fabricated; no validated native face chain supplied',
            functional_intervention='pending identified mode and information identity'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=analyze(args.observations)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('quiet_calls','quiet_time','external_events','optimizer_updates','completion')},ensure_ascii=False))


if __name__=='__main__':main()
