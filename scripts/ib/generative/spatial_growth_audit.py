"""Finite-volume deposition from local conserved exchange; no target tree."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange,save_exchange
from information_boltzmann.cgb.spatial_exchange import SpatialExchange,SpatialExchangeParams,cubic_mesh
from scripts.ib.generative.compare_support_transport import digest,same
from scripts.ib.generative.driven_ports_audit import input_voltage


def construct(params):
    xyz,pre,post = cubic_mesh(9)
    n,e = len(xyz),len(pre)
    center = int(np.flatnonzero(np.all(xyz==[4.,4.,4.],axis=1))[0])
    seed = (pre==center)|(post==center)
    installed = np.zeros(e)
    installed[seed] = .8
    neighbors = np.r_[post[pre==center],pre[post==center]]
    diagonal = int(np.flatnonzero(np.all(xyz==[4.,5.,5.],axis=1))[0])
    ports = np.r_[center,neighbors,diagonal]
    assert len(ports)==8 and seed.sum()==6
    state = SpatialExchange(pre,post,installed,np.full(n,.1),np.zeros(e),np.zeros(n),
        np.full(n,.8),np.full(n,.2),np.ones(e),np.full(n,.8),np.zeros(n),
        params=params,coordinates=xyz,trace=0.,port_nodes=ports,
        world_capacitance=2.,port_conductance=.05,source_conductance=.1)
    return state,seed


def geometry(state,seed):
    n = state.capacity
    mid = .5*(state.coordinates[state.pre]+state.coordinates[state.post])
    centroid = np.average(mid,axis=0,weights=n)
    radius = float(np.sqrt(np.average(np.sum((mid-centroid)**2,axis=1),weights=n)))
    levels = {}
    for cut in (.1,.3,.5):
        keep = n>=cut
        pre,post = state.pre[keep],state.post[keep]
        nodes = np.unique(np.r_[pre,post])
        degree = np.bincount(np.r_[pre,post],minlength=state.n)
        graph = coo_matrix((np.ones(2*len(pre)),(np.r_[pre,post],np.r_[post,pre])),shape=(state.n,state.n)).tocsr()
        components = connected_components(graph[nodes][:,nodes],directed=False,return_labels=False) if len(nodes) else 0
        levels[str(cut)] = dict(edges=int(keep.sum()),new_edges=int(np.sum(keep&~seed)),
            nodes=len(nodes),components=components,junctions=int(np.sum(degree>=3)),tips=int(np.sum(degree==1)),
            cycle_rank=int(keep.sum()-len(nodes)+components))
    return dict(installed=float(n.sum()),radius_of_gyration=radius,centroid=centroid.tolist(),levels=levels)


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    shelf_path = ROOT/'data/harness/reading/shelf.json'
    story = json.loads(shelf_path.read_text(encoding='utf-8'))['entries'][0]
    data = story['text'].encode()[:64]
    (args.out/'reading_shelf_snapshot.json').write_bytes(shelf_path.read_bytes())
    paths = [Path(__file__)] + [ROOT/'information_boltzmann/cgb'/name for name in (
        'spatial_exchange.py','trace_exchange.py','driven_exchange.py','flow_exchange.py',
        'structural_exchange.py','finite_exchange.py','material_reactions.py','exchange_checkpoint.py')]
    paths += [ROOT/'docs/information_boltzmann/CGB_TRACE_GROWTH_PREREG_20261022.md']
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in paths}
    snapshot = args.out/'source_snapshot'
    snapshot.mkdir()
    for p in paths:
        (snapshot/p.name).write_bytes(p.read_bytes())
    base = SpatialExchangeParams(assembly_rate=.05,structure_energy=2.,
        conversion_rate=0.,active_rate=0.,write_rate=.0005)
    arms,scenes = {},[]
    for j,d in ((0.,.005),(.6,.005),(0.,.1),(.6,.1)):
        label = f'J{j:g}_D{d:g}'
        p = replace(base,cohesion_energy=j,material_diffusion=d,support_diffusion=d/2,drive_diffusion=d/2)
        m,seed = construct(p)
        initial = geometry(m,seed)
        errors = {name:0. for name in m.balances()}
        frames = []
        def record():
            frames.append(dict(ms=m.t,installed=round(float(m.capacity.sum()),8),
                material=round(float(m.material.sum()),8),source_work=m.source_work,
                density=np.round(m.capacity,7).tolist(),trace=np.round(m.trace,7).tolist()))
        record()
        least,peak_cg = 0.,0
        fork = None
        started = time.perf_counter()
        for step in range(128):
            v = input_voltage(data,step,2.)
            update = m.step(2.,world_voltage=v)
            least = min(least,update['free_energy_drop'])
            peak_cg = max(peak_cg,m.last_cg_iterations)
            if fork is not None:
                fork.step(2.,world_voltage=v)
            for name,value in m.balances().items():
                errors[name] = max(errors[name],float(np.max(np.abs(value),initial=0.)))
            if (step+1)%8==0:
                record()
            if step+1==64:
                save_exchange(args.out/(label+'_half.npz'),m,dict(next_step=64,dt_ms=2.,
                    byte_interval_ms=4.,byte_stream_hex=data.hex(),params=asdict(p)))
                if j==.6 and d==.005:
                    fork,_ = load_exchange(args.out/(label+'_half.npz'))
        if fork is not None:
            same(m.state_dict(),fork.state_dict())
        assert max(errors.values())<1e-8 and least>=-1e-9
        archive = args.out/(label+'_complete.npz')
        save_exchange(archive,m,dict(next_step=128,dt_ms=2.,byte_interval_ms=4.,
            byte_stream_hex=data.hex(),params=asdict(p),kind='spatial_material_not_neuron_tree'))
        restored,_ = load_exchange(archive)
        same(m.state_dict(),restored.state_dict())
        report = dict(params=asdict(p),initial=initial,final=geometry(m,seed),
            elapsed_ms=m.t,seconds=time.perf_counter()-started,source_work=m.source_work,
            final_material=float(m.material.sum()),final_charged=float(m.charged.sum()),
            trace_written=float(m.trace_written.sum()),maximum_balance_errors=errors,
            minimum_irreversible_g_drop=least,peak_cg_iterations=peak_cg,
            archive=str(archive),sha256=digest(archive),full_restore_loaded=True,
            future_bitwise_tested=fork is not None,
            physical_state_bytes=sum(getattr(m,k).nbytes for k in m._arrays)+sum(v.nbytes for v in (m.pre,m.post,m.capacity,m.port_nodes)))
        arms[label] = report
        scenes.append(dict(id=label,label=f'结合强度 {j:g} · 材料扩散 {d:g}',frames=frames))
        print(json.dumps(dict(arm=label,**report),ensure_ascii=False),flush=True)
    scene = dict(scope='spatial physical material, not neuron arbors',coordinates=m.coordinates.tolist(),
        pre=m.pre.tolist(),post=m.post.tolist(),arms=scenes,default='J0.6_D0.005',
        display='All density retained; cutoff only affects viewing, never dynamics.')
    (args.out/'material_scene.json').write_text(json.dumps(scene,separators=(',',':'),ensure_ascii=False),encoding='utf-8')
    report = dict(date='2026-10-22',scope='spatial finite material deposition and retraction; not complete neuronal morphogenesis',
        baseline_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        preregistered='docs/information_boltzmann/CGB_TRACE_GROWTH_PREREG_20261022.md',
        nodes=m.n,spatial_positions=len(m.pre),initial_seed_bonds=6,
        story=story['title'],story_source=story['source'],reading_shelf_sha256=digest(shelf_path),
        bytes_presented=len(data),byte_stream_hex=data.hex(),byte_ms=4.,dt_ms=2.,arms=arms,
        code_sha256=hashes,no_target_coordinates_or_branch_count=True,
        constitutive_scope='equal-volume lattice material with short-range cohesion and surface catalysis',
        neuronal_lineages_generated=False,cell_contact_identity_matching_invoked=False,
        continuum_mesh_convergence='unmeasured; independent diffusion time-refinement test is separate',
        language_or_predictive_learning_benefit='unmeasured',passed=True)
    assert hashes=={str(p.relative_to(ROOT)):digest(p) for p in paths}
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
