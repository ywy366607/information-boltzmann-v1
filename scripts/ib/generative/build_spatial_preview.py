"""Build actual candidate development and local-medium state for 3D replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.birth import BirthParams, seed_from_contacts
from information_boltzmann.cgb.checkpoint import save_medium
from information_boltzmann.cgb.development import SpatialDevelopment, encounter, geometry_features
from information_boltzmann.cgb.medium import CellClass, SynapseParams, TrophicParams
from information_boltzmann.cgb.structural_budget import MaterialParams, StructuralBudget
from information_boltzmann.cgb.type_generator import TypeStrengthGenerator
from information_boltzmann.cgb.morphology import SpatialIdentityProgram
from information_boltzmann.cgb.occurrence import OccurrenceProgram
from information_boltzmann.cgb.tree_development import ConditionalTreeDevelopment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
        default=lambda a: a.tolist() if isinstance(a,np.ndarray) else a.item()), encoding='utf-8')


def floors_from_training(work):
    with np.load(work/'e1_types3000.npz', allow_pickle=True) as z:
        sizes = z['sizes']
    floors = []
    for scale in (4, 2):
        matrix = np.load(work/f'e1c_overlap_trainlabels_{scale}um.npy', mmap_mode='r')
        floor = np.inf
        for start in range(0, len(sizes), 128):
            v = matrix[start:start+128]/(sizes[start:start+128,None]*sizes[None,:])
            if np.any(v>0):
                floor = min(floor, float(v[v>0].min())*.5)
        floors.append(floor)
    return floors


def main(args):
    started = time.perf_counter()
    if (args.out/'scene.json').exists():
        raise FileExistsError('Keep the previous candidate; choose a new output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    generator = TypeStrengthGenerator.load(args.generator)
    rng = np.random.default_rng(21)
    spatial = SpatialIdentityProgram.load(args.spatial) if args.spatial else None
    occurrence = OccurrenceProgram.load(args.occurrence) if args.occurrence else None
    for program in (spatial, occurrence):
        if program is not None and not np.array_equal(program.values['types'], generator.types):
            raise ValueError('Birth programs must share an exact identity ordering')
    eligible = np.flatnonzero(spatial.values['known']) if spatial else np.arange(len(generator.types))
    types = np.sort(rng.choice(eligible,48,replace=False))
    cell_type = np.repeat(types,4)
    plan = spatial.describe(cell_type, np.tile(['L','L','R','R'],len(types))) if spatial else None
    if args.tree:
        if plan is None: raise ValueError('Tree birth requires a spatial program')
        with np.load(args.tree,allow_pickle=False) as q: tree={k:q[k] for k in q.files}
        growth=ConditionalTreeDevelopment(generator.values['code'],cell_type,plan,tree)
    else:
        growth = SpatialDevelopment(generator.values['code'],cell_type,spatial_plan=plan)
    frames = []

    def frame(phase, **extra):
        b = growth.budget if 'medium' not in extra else extra.pop('medium').material
        row = dict(phase=phase, generation=growth.generation, segments=len(growth.start),
                   free=float(b.free.sum()), reserved=float(b.reserved.sum()),
                   structure=0., activity=0, balance=0., word='')
        row.update(extra)
        frames.append(row)
    frame('seed')
    for step in range(args.growth_steps):
        if args.tree and not np.any(growth.cone_active): break
        growth.step()
        frame('growth', balance=float(np.max(np.abs(growth.budget.balance(np.zeros(growth.n))))))
    # Complete growth and branch RNG/pools are retained separately from display rounding.
    write_json(args.out/'development_state.json',growth.state_dict())
    seg = growth.segments()
    print(json.dumps({'stage':'arbors_generated','segments':len(seg['cell']),
                      'free_material':float(growth.budget.free.sum())}),flush=True)
    overlaps = [encounter(seg,growth.n,scale) for scale in (4,2)]
    all_overlaps = [encounter(seg,growth.n,scale,polarized=False) for scale in (4,2)] if occurrence else overlaps
    opportunities = all_overlaps[0].tocoo()
    valid = opportunities.row != opportunities.col
    pre, post, e = opportunities.row[valid],opportunities.col[valid],opportunities.data[valid]
    raw = geometry_features(overlaps,cell_type,pre,post,floors_from_training(args.work))
    probability = (occurrence.probability(cell_type[pre],cell_type[post],e,
        np.asarray(all_overlaps[1][pre,post]).ravel()) if occurrence else -np.expm1(-args.binding*e))
    medium,birth = seed_from_contacts(generator,cell_type,[CellClass()],np.zeros(growth.n,int),
        pre,post,raw,probability,params=BirthParams(1,seed=21),
        syn=SynapseParams(gmax=.04,eta=4e-5,m0=.02,slide_A0=5,
                          aK=.2,thP=.7,nP=4,tau_bap=10,bap_amp=2))
    desired = medium.g.copy()
    medium.g[:] = 0
    # Carry the original donor account into the new edge layout, without a new endowment.
    budget_state = growth.budget.state_dict()
    budget_state['post'] = medium.post.copy()
    medium.material = StructuralBudget.from_state_dict(budget_state)
    medium.material.params = MaterialParams(supply=2e-5,clearance=.0001,maintenance=1e-5,recycle=.5)
    medium._set_structure(desired)
    birth.update(material_enabled=True, funded_structure=float(medium.g.sum()),
                 arbor_material_carried=float(medium.material.reserved.sum()))
    if occurrence:
        birth['unverified'].remove('occurrence calibration')
        birth['occurrence_calibration_scope']='Observed real cell pairs; generated geometry transfer not validated'
    np.savez(args.out/'birth_program_features.npz',pre=pre,post=post,raw_strength_features=raw,
             total_overlap4=e,total_overlap2=np.asarray(all_overlaps[1][pre,post]).ravel(),probability=probability)
    medium.trophic = TrophicParams(rule='constitutive',capture='tag',q0=.001,
                                   T_star=1,lambda0=1/20000)
    medium.trophic_edges[:] = True
    for phase in ('contact','wiring'):
        frame(phase, medium=medium, structure=float(medium.g.sum()),
              balance=float(np.max(np.abs(medium.material.balance(medium.g)))))
    shelf=json.loads((ROOT/'data/harness/reading/shelf.json').read_text(encoding='utf-8'))
    story=shelf['entries'][0]
    words=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:]",story['text'])[:48]
    half_path=args.out/'medium_before_reading.npz'
    provenance={'scope':'Candidate spatial development and prescribed real-text sensory drive; no capability claim',
                'generator_sha256':digest(args.generator),'story':story['title'],
                'story_source':story['source'],'binding_rate_unfitted':None if occurrence else args.binding,
                'geometry_law':'Known-type spatial descriptors with candidate local Gaussian growth' if spatial else 'Identity projection to gradient receptors; unfitted',
                'spatial_program_sha256':digest(args.spatial) if spatial else None,
                'occurrence_program_sha256':digest(args.occurrence) if occurrence else None,
                'code_path':str(ROOT/'information_boltzmann/cgb/development.py')}
    save_medium(half_path,medium,provenance)
    initial_g=medium.g.copy()
    print(json.dumps({'stage':'connections_funded','opportunities':len(pre),'edges':medium.E}),flush=True)
    drives={}
    for word in set(words):
        h=int.from_bytes(hashlib.blake2b(word.lower().encode(),digest_size=8).digest(),'little')
        drive=np.zeros(medium.N)
        drive[np.random.default_rng(h).choice(medium.N,6,replace=False)]=.18
        drives[word]=drive
    spikes=np.zeros(medium.N,bool)
    max_error=0.
    duration_steps=int(len(words)*200/medium.dt)
    for k in range(duration_steps):
        word=words[min(len(words)-1,int(medium.t//200))]
        # Fixed transduction, no learned embedding and no externally chosen spike outcome.
        medium.step(I_ext=drives[word])
        spikes |= medium.spike
        if (k+1)%80==0:
            error=float(np.max(np.abs(medium.material.balance(medium.g))))
            max_error=max(max_error,error)
            frame('reading',medium=medium,structure=float(medium.g.sum()),activity=int(spikes.sum()),
                  balance=error,word=word,t_ms=medium.t,u=np.round(medium.u,4),
                  spikes=np.flatnonzero(spikes),g=np.round(medium.g,7),
                  support=np.round(np.bincount(medium.post,weights=medium.T,minlength=medium.N),4),
                  free_by_cell=np.round(medium.material.free,6))
            spikes[:]=False
    save_medium(args.out/'medium_complete.npz',medium,provenance)
    normalized=(raw-generator.values['feature_mean'])/generator.values['feature_scale']
    report={'scope':provenance['scope'],'cells':growth.n,'identity_types':len(types),
            'segments':len(seg['cell']),'opportunities':len(pre),'birth':birth,
            'binding_rate_unfitted':None if occurrence else args.binding,'generated_feature_abs_z_max':float(np.abs(normalized).max()),
            'generated_feature_fraction_abs_z_over_6':float((np.abs(normalized)>6).mean()),
            'generated_log_feature_fraction_abs_z_over_6':float((np.abs(normalized[:,[0,2,4,6]])>6).mean()),
            'generated_log_feature_abs_z_max':float(np.abs(normalized[:,[0,2,4,6]]).max()),
            'max_local_material_error':max_error,'simulated_reading_ms':medium.t,
            'story_title':story['title'],'words_presented':len(words),
            'changed_structural_edges':int(np.count_nonzero(medium.g!=initial_g)),
            'readout_training':False,'world_consequence_learning':False,
            'growth_steps':growth.generation,'initial_material':float(growth.budget.initial.sum()),
            'spatial_descriptors_calibrated':bool(spatial),'occurrence_calibrated':bool(occurrence),
            'growth_dynamics_calibrated':False,'polarity_calibrated':False,
            'mean_occurrence_probability':float(probability.mean()),
            'sampled_edges':int(medium.E),
            'conditional_tree_program':bool(args.tree),
            'tree_growth_completed':bool(not np.any(growth.cone_active)) if args.tree else None,
            'seconds':time.perf_counter()-started,
            'source_sha256':{str(p):digest(p) for p in [Path(__file__),
                ROOT/'information_boltzmann/cgb/development.py',ROOT/'information_boltzmann/cgb/structural_budget.py',
                ROOT/'docs/information_boltzmann/CGB_SPATIAL_PREVIEW_20261021.md']}}
    if spatial:
        length=np.linalg.norm(seg['end']-seg['start'],axis=1)
        actual=np.bincount(seg['cell'],weights=length,minlength=growth.n)
        center=np.column_stack([np.bincount(seg['cell'],weights=length*(seg['start'][:,j]+seg['end'][:,j])/2,
            minlength=growth.n)/np.maximum(actual,1e-30) for j in range(3)])
        target=plan['soma_mean']+plan['centroid_offset']
        report['spatial_generation']={'length_fraction_quantiles':np.quantile(actual/plan['length'],[0,.5,1]).tolist(),
            'cable_centroid_error_um_median':float(np.median(np.linalg.norm(center-target,axis=1))),
            'soma_noise_sampled':False,'trees_topology_validated':False,
            'cone_workspace_per_cell':growth.params.cone_slots_per_cell,
            'cumulative_cones_max':int(np.bincount(growth.cone_cell,minlength=growth.n).max()),
            'endpoint_field_correlation_steps_unfitted':8}
    if max_error>1e-10:
        raise RuntimeError('Material balance failed')
    scene={'version':1,'title':'生成脑 · 出生观察室','scope':'候选空间发育 · 真实状态回放',
           'report':report,'soma':np.round(growth.soma,4),'cell_type':cell_type,
           'type_names':[str(generator.types[t]) for t in cell_type],
           'segments':{k:np.round(v,4) if v.dtype.kind=='f' else v for k,v in seg.items()},
           'edges':{'pre':medium.pre,'post':medium.post,'initial_g':initial_g},
           'opportunities':{'pre':pre,'post':post},'frames':frames,'story':story}
    write_json(args.out/'scene.json',scene)
    write_json(args.out/'report.json',report)
    source=args.out/'source_snapshot'
    source.mkdir(exist_ok=True)
    import shutil
    copied={}
    for name in ('development','tree_development','tree_geometry','morphology','occurrence','structural_budget','medium','birth','type_generator','checkpoint'):
        p=ROOT/f'information_boltzmann/cgb/{name}.py'
        shutil.copy2(p,source/p.name);copied[str(p.relative_to(ROOT))]=digest(p)
    shutil.copy2(__file__,source/Path(__file__).name)
    copied[str(Path(__file__).relative_to(ROOT))]=digest(__file__)
    for p in (args.generator,args.spatial,args.occurrence,args.tree):
        if p:
            shutil.copy2(p,source/p.name);copied[str(p)]=digest(p)
    write_json(source/'manifest.json',copied)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--generator',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--binding',type=float,default=.15)
    parser.add_argument('--spatial',type=Path)
    parser.add_argument('--occurrence',type=Path)
    parser.add_argument('--growth-steps',type=int,default=64)
    parser.add_argument('--tree',type=Path)
    main(parser.parse_args())
