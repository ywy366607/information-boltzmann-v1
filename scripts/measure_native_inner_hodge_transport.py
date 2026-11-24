"""Transport the first actual internal motion into the inherited Hodge/Dirac.

Reuse certified edge supports by stable fact identity. The actual new posterior
rows determine the signal; source/preview clocks and the fixed scope are checked.
"""
import argparse
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.linalg import eigh

from scripts.measure_native_inner_loop import ROOT, load, sha
from scripts.measure_native_paper_phase_dirac import decode_cells
from scripts.audit_native_dowker_measurement import make_boundary, to_sparse


def first_calls(path):
    result={}
    for line in gzip.open(path,'rt',encoding='utf8'):
        r=json.loads(line)
        if r['event']>1:break
        if r['kind']=='relation_signal':result[r['data']['call_id']]=r
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args()
    source_path=ROOT/'results/published/cgb_native_shared_response_scoped_2401_20261117.json'
    source=json.loads(source_path.read_text(encoding='utf8'))
    old=first_calls(Path(source['source']['operations']))
    refs={r['call_id']:r for r in source['calls']}
    proof=Path(source['scope_lean']['path']); assert sha(proof)==source['scope_lean']['sha256']
    text=proof.read_text(encoding='utf8')
    cells=[decode_cells(text,f'expanded{n}') for n in range(3)]
    b1=to_sparse(make_boundary(cells[0],cells[1]),len(cells[0])).astype(float)
    b2=to_sparse(make_boundary(cells[1],cells[2]),len(cells[1])).astype(float)
    ids=json.loads((ROOT/'results/published/cgb_native_harmonic_observability_2401_20261117.json').read_text(encoding='utf8'))['identities']['relations']
    nodes=sorted([tuple(x) for x in ids]+[tuple(x) for x in source['retained_saved_scope_vertices']])
    eigen,vectors=eigh((b1@b1.T).toarray());active=eigen>1e-10
    inverse=(vectors[:,active]/eigen[active])@vectors[:,active].T
    with np.load(source['arrays']['path']) as z:h=z['harmonic_basis']
    arms={};saved={};max_error=0.;physical_reference=None;certificates=[]
    for arm in ('baseline','bridge_cut','matched_cut','order_only'):
        obs=args.work/arm/'observations';calls=first_calls(obs/'operations.jsonl.gz')
        with np.load(obs/'physical_00000.npz') as z:
            indices=[i for i,k in enumerate(z['kind']) if str(k) in ('preview_quiet_before','preview_quiet_after')]
            assert len(indices)==2
            physical={k:z[k][indices].copy() for k in ('complex_overlap_real','complex_overlap_imag','complex_norm','theta')}
        if physical_reference is None:physical_reference=physical
        else:assert all(np.array_equal(v,physical_reference[k]) for k,v in physical.items())
        records={};signals={}
        with (obs/'tensors.bin').open('rb') as stream:
            for cid in (11,14,17,23,26,29):
                row=calls[cid];d=row['data'];ref=refs[cid];prior=old[cid]['data']
                assert row['event']==1 and row['update_clock']==0
                assert d['state']['variable_uses']==prior['state']['variable_uses']
                children=[r['data'] for r in calls.values() if r['data']['parent_id']==cid and r['data']['operation']=='_bound_evidence_design']
                assert len(children)==1 and all(n==1 for n in children[0]['evidence_row_counts'])
                keys=[(f['fact_id'],f['kind']) for f in children[0]['facts']]
                old_keys=[tuple(k) for k in ref['fact_keys']]
                assert set(keys)==set(old_keys) and len(keys)==len(set(keys))==8
                permutation=[old_keys.index(k) for k in keys]
                masks=np.array(ref['mask'],dtype=float)[:,permutation]
                assert masks.shape == (4,8)
                certificates.append((permutation,ref['mask'],masks.astype(int).tolist()))
                pred=d['prediction'];x,y,q=[load(stream,pred[k]) for k in ('x','y','query')]
                ridge=load(stream,pred['ridge']) if isinstance(pred['ridge'],dict) else pred['ridge']
                alpha=np.linalg.solve(x@x.T+float(np.asarray(ridge))*np.eye(len(x)),(q@x.T).T).T
                response=np.einsum('bf,ef,fd->ebd',alpha,masks,y)
                rid=tuple(ref['relation'])
                signs=np.array([1 if nodes[cells[1][e][1]]==rid else -1 for e in ref['edge_ids']])
                phi=np.zeros((len(cells[1]),128));phi[ref['edge_ids']]=(response*signs[:,None,None]).reshape(-1,128)
                grad=b1.T@(inverse@(b1@phi));harm=h@(h.T@phi);curl=phi-grad-harm
                err=max(float(abs(phi-grad-harm-curl).max()),float(abs(b1@curl).max()),float(abs(b2.T@grad).max()))
                max_error=max(max_error,err)
                records[cid]=dict(call_id=cid,event=1,update_clock=0,permutation=permutation,
                    inherited_edge_ids=ref['edge_ids'],fact_keys=keys,
                    total_energy=float(np.square(phi).sum()),
                    hodge_energies={k:float(np.square(v).sum()) for k,v in [('gradient',grad),('curl',curl),('harmonic',harm)]},
                    dirac_node_energy=float(np.square(b1@phi).sum()),dirac_face_energy=float(np.square(b2.T@phi).sum()))
                signals[cid]=(phi,grad,curl,harm)
                saved[f'{arm}_phi_{cid}']=phi[ref['edge_ids']]
        changes=[]
        for a,b in ((11,23),(14,26),(17,29)):
            delta=[v-u for u,v in zip(signals[a],signals[b])]
            changes.append(dict(before_call=a,after_call=b,
                delta_hodge_energy={k:float(np.square(v).sum()) for k,v in zip(('total','gradient','curl','harmonic'),delta)},
                delta_D_node_energy=float(np.square(b1@delta[0]).sum()),
                delta_D_face_energy=float(np.square(b2.T@delta[0]).sum())))
        arms[arm]=dict(calls=records,motion_changes=changes)
    np.savez_compressed(args.output.with_suffix('.npz'),**saved)
    result=dict(status='measured',inherited_measurement=str(source_path),inherited_sha256=sha(source_path),
        support_proof=str(proof),support_proof_sha256=sha(proof),
        mapping='same actual 213/4 scope and dependency versions; stable fact permutation of certified masks',
        first_preview_physical_state_readouts_equal_across_four_arms=True,
        maximum_reconstruction_boundary_error=max_error,arms=arms)
    proof_path=ROOT/'lean/CGBAxioms/reviews/NativeInnerHodgeTransportActual2401.lean'
    proof_text='''import Mathlib.Tactic
namespace CGB.NativeInnerHodgeTransportActual2401
open Matrix
/-- The certified support mask follows the same eight actual fact identities
    as their 128-coordinate response rows. -/
theorem transport_mask (M : Matrix (Fin 8) (Fin 4) ℝ)
    (R : Matrix (Fin 8) (Fin 128) ℝ) (P : Matrix (Fin 8) (Fin 8) ℝ)
    (orthogonal : P.transpose * P = 1) :
    (P * M).transpose * (P * R) = M.transpose * R := by
  simp only [Matrix.transpose_mul, Matrix.mul_assoc]
  rw [← Matrix.mul_assoc P.transpose P R, orthogonal, Matrix.one_mul]
'''
    for i,(perm,old_masks,new_masks) in enumerate(certificates):
        proof_text+=f'example : ({perm} : List Nat).Perm (List.range 8) := by decide\n'
        for label,rows_ in [('old',old_masks),('new',new_masks)]:
            proof_text+=f'def {label}{i} : Array (Array Nat) := #['+', '.join('#'+str(row) for row in rows_)+']\n'
        proof_text+=f'def perm{i} : Array Nat := #'+str(perm)+'\n'
        proof_text+=f'example : (List.range 4).all (fun e => (List.range 8).all (fun j => (new{i}[e]!)[j]! == (old{i}[e]!)[perm{i}[j]!]!)) = true := by native_decide\n'
    proof_text+='end CGB.NativeInnerHodgeTransportActual2401\n'
    proof_path.write_text(proof_text,encoding='utf8')
    checked=subprocess.run(['lake','env','lean',str(proof_path.relative_to(ROOT/'lean/CGBAxioms'))],
        cwd=ROOT/'lean/CGBAxioms',capture_output=True,text=True,encoding='utf8')
    args.output.with_suffix('.lean.log').write_text(checked.stdout+checked.stderr,encoding='utf8')
    if checked.returncode:raise RuntimeError(checked.stdout+checked.stderr)
    result['lean']=dict(path=str(proof_path),sha256=sha(proof_path),exit_code=0,actual_mask_instances=len(certificates))
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v['motion_changes'] for k,v in arms.items()}))


if __name__=='__main__':main()
