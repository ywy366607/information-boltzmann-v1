"""Deploy inherited identity components on actual moving-material contacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import subprocess

import numpy as np
import scipy.sparse as sp
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.identity_material import IdentityMaterialProgram, IdentityMobileExchange
from information_boltzmann.cgb.mobile_exchange import MobileExchangeParams
from information_boltzmann.cgb.occurrence import OccurrenceProgram
from information_boltzmann.cgb.type_generator import TypeStrengthGenerator
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from scripts.ib.generative.mobile_material_audit import preparation, morphology
from scripts.ib.generative.compare_support_transport import digest, same
from scripts.ib.generative.driven_ports_audit import input_voltage


def verify_sources(program, strength_path, occurrence_path, panel_path):
    strength, occurrence = TypeStrengthGenerator.load(strength_path), OccurrenceProgram.load(occurrence_path)
    with np.load(panel_path/'occurrence_panel.npz', allow_pickle=False) as z:
        body, types, weights = z['body'], z['cell_type'], z['weights']
    pre, post = np.nonzero(weights > 0)
    a, b = types[pre], types[post]
    e4 = np.asarray(sp.load_npz(panel_path/'cell_overlap_4um.npz').tocsr()[pre, post]).ravel()
    e2 = np.asarray(sp.load_npz(panel_path/'cell_overlap_2um.npz').tocsr()[pre, post]).ravel()
    raw = occurrence.raw_features(e4, e2)
    reconstructed = expit(occurrence.values['bias']
        +((raw-occurrence.values['feature_mean'])/occurrence.values['feature_scale'])@occurrence.values['geometry']
        +program.recognition_field(a, b)+program.occurrence_center)
    original = occurrence.probability(a, b, e4, e2)
    error_o = float(np.max(np.abs(reconstructed-original)))
    # Subtracting the reference geometric/intercept part is a numerical
    # decomposition, not an invented geometry measurement for new particles.
    reference = np.broadcast_to(strength.values['feature_mean'], (len(a), len(strength.values['feature_mean'])))
    original_s = strength.log_strength(a, b, reference)
    reconstructed_s = strength.values['bias']+program.transport_log_scale(a, b)+program.strength_center
    error_s = float(np.max(np.abs(original_s-reconstructed_s)))
    assert error_o < 1e-12 and error_s < 1e-5
    chosen = np.argsort(body)[np.linspace(0, len(body)-1, 24).astype(int)]
    cell_type, selected_body = types[chosen], body[chosen]
    aa, bb = np.meshgrid(cell_type, cell_type, indexing='ij')
    logs = program.transport_log_scale(aa.ravel(), bb.ravel())
    field = program.recognition_field(aa.ravel(), bb.ravel())
    audit = dict(positive_real_cell_pairs=len(a), panel_cells=len(body), panel_types=int(len(np.unique(types))),
        all_gene_types=len(program.types), shared_type_order_and_code_exact=True,
        maximum_occurrence_probability_reconstruction_error=error_o,
        maximum_reference_log_strength_reconstruction_error=error_s,
        reference_geometry='fixed algebraic feature mean only for decomposition; not used on moving contacts',
        selected_transport_log_range=[float(logs.min()), float(logs.max())],
        selected_recognition_field_range=[float(field.min()), float(field.max())],
        selected_recognition_direction_difference=float(np.max(np.abs(field.reshape(24,24)-field.reshape(24,24).T))),
        selected_body_ids=selected_body.tolist(), selected_type_indices=cell_type.tolist(),
        selected_type_names=program.types[cell_type].tolist(), passed=True)
    return audit, cell_type


def construct(params, program, cell_type, label):
    base = preparation(params)
    if label == 'shuffled_identity':
        cell_type = cell_type[np.random.default_rng(812).permutation(len(cell_type))]
    return IdentityMobileExchange(base.coordinates, base.material, base.free, base.charged, base.depleted,
        end_capacity=base.end_capacity, carried=base.carried, charge=base.charge,
        capacitance=base.capacitance, pre=base.pre, post=base.post, bound=base.bound,
        trace=base.trace, unit_conductance=base.unit_conductance, params=params,
        port_nodes=base.port_nodes, world_charge=base.world_charge, world_capacitance=base.world_capacitance,
        port_conductance=base.port_conductance, source_conductance=base.source_conductance,
        program=program, cell_type=cell_type, recognition_energy_fraction=1. if label == 'energy_storage' else 0.,
        recognition_enabled=label != 'no_recognition', transport_enabled=label != 'no_transport_identity', seed=7521)


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    program = IdentityMaterialProgram.from_sources(args.strength, args.occurrence)
    source_audit, cell_type = verify_sources(program, args.strength, args.occurrence, args.panel)
    (args.out/'source_program_audit.json').write_text(json.dumps(source_audit, indent=2, ensure_ascii=False), encoding='utf-8')
    prior = json.loads((args.baseline/'report.json').read_text(encoding='utf-8'))
    params = MobileExchangeParams(**prior['arms']['coupled']['params'])
    sources = [Path(__file__), ROOT/'scripts/ib/generative/mobile_material_audit.py',
        ROOT/'scripts/ib/generative/driven_ports_audit.py',
        ROOT/'docs/information_boltzmann/CGB_IDENTITY_MATERIAL_PREREG_20261022.md']
    sources += [ROOT/'information_boltzmann/cgb'/name for name in ('mobile_exchange.py', 'identity_material.py',
        'occurrence.py', 'type_generator.py', 'exchange_checkpoint.py', 'spatial_exchange.py')]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    snapshot = args.out/'source_snapshot'; snapshot.mkdir()
    for p in sources: (snapshot/p.name).write_bytes(p.read_bytes())
    generator_sources = args.out/'generator_sources'; generator_sources.mkdir()
    for p in (args.strength, args.occurrence): (generator_sources/p.name).write_bytes(p.read_bytes())
    np.savez_compressed(args.out/'identity_program.npz', **program.arrays, types=program.types,
        metadata=np.array(json.dumps(dict(provenance=program.provenance, sha256=program.fingerprint()))))
    shelf_path = ROOT/'data/harness/reading/shelf.json'
    story = json.loads(shelf_path.read_text(encoding='utf-8'))['entries'][0]
    data = story['text'].encode('utf-8')[:64]
    (args.out/'reading_shelf_snapshot.json').write_bytes(shelf_path.read_bytes())
    reports, scenes, finals = {}, [], {}
    for label in ('coupled', 'no_recognition', 'no_transport_identity', 'shuffled_identity', 'energy_storage'):
        m = construct(params, program, cell_type, label)
        before, initial_u, initial_g = morphology(m), m.energy(), m.free_energy()
        initial_recognition = m._recognition_total()
        initial_pairs = {tuple(pair) for pair in m.encounters().tolist()}
        counts = dict(formations=0, new_spatial_pair_formations=0, removals=0,
                      summed_formation_recognition_free_energy=0.)
        log = (args.out/(label+'_contacts.jsonl')).open('w', encoding='utf-8')
        def observer(event):
            if event['kind'] != 'assembly': return
            if event['sign'] > 0:
                counts['formations'] += 1
                if tuple(sorted((event['i'], event['j']))) not in initial_pairs:
                    counts['new_spatial_pair_formations'] += 1
                f = float(m.recognition_free_energy(event['i'], event['j']))
                counts['summed_formation_recognition_free_energy'] += f
                event['recognition_free_energy'] = f
                event['transport_multiplier'] = float(m._transport_factor(event['i'], event['j']))
                event['types'] = m.cell_type[[event['i'], event['j']]].tolist()
            else: counts['removals'] += 1
            log.write(json.dumps(event, separators=(',', ':'))+'\n')
        m.event_observer = observer
        frames, fork = [], None
        errors = {k: 0. for k in m.balances()}
        def record():
            frames.append(dict(ms=m.t, coordinates=np.round(m.coordinates, 7).tolist(),
                pre=m.pre.tolist(), post=m.post.tolist(), trace=m.trace.tolist(), bound=m.bound.tolist(),
                material=int(m.material.sum()), source_work=m.work_in, heat_out=m.heat_out,
                morphology=morphology(m)))
        record()
        started = time.perf_counter()
        for step in range(128):
            voltage = input_voltage(data, step, 2.)
            m.step(2., voltage)
            if fork is not None: fork.step(2., voltage)
            for k, v in m.balances().items(): errors[k] = max(errors[k], abs(float(v)))
            if (step+1)%8 == 0: record()
            if step == 63:
                save_exchange(args.out/(label+'_half.npz'), m,
                    dict(next_step=64, dt_ms=2., byte_ms=4., byte_stream_hex=data.hex(), genome=program.fingerprint()))
                if label == 'coupled': fork, _ = load_exchange(args.out/(label+'_half.npz'))
        log.close()
        if fork is not None: same(m.state_dict(), fork.state_dict())
        path = args.out/(label+'_complete.npz')
        save_exchange(path, m, dict(next_step=128, dt_ms=2., byte_ms=4., byte_stream_hex=data.hex(),
            genome=program.fingerprint(), source_programs_embedded=True))
        restored, _ = load_exchange(path); same(m.state_dict(), restored.state_dict())
        assert max(errors.values()) < 1e-8
        paired = np.stack((m.pre, m.post), axis=1)
        reciprocal = sum((int(b), int(a)) in set(zip(m.pre, m.post)) for a,b in paired)//2
        reports[label] = dict(initial=before, final=morphology(m), initial_energy=initial_u, initial_free_energy=initial_g,
            initial_recognition_free_energy=initial_recognition, final_recognition_free_energy=m._recognition_total(),
            reciprocal_bridge_pairs=int(reciprocal), contact_events=counts, elapsed_ms=m.t,
            seconds=time.perf_counter()-started, recognition_energy_fraction=m.recognition_energy_fraction,
            source_work=m.work_in, heat_out=m.heat_out, reaction_heat=m.reaction_heat,
            bound_support=int(m.bound.sum()), occupied_trace=int(m.trace.sum()), reaction_counts=m.reaction_counts,
            maximum_balance_errors=errors, archive=str(path), sha256=digest(path),
            future_bitwise_tested=fork is not None, full_restore_loaded=True,
            state_bytes=sum(getattr(m,k).nbytes for k in m._arrays)+m.pre.nbytes+m.post.nbytes+m.capacity.nbytes+m.port_nodes.nbytes)
        scenes.append(dict(id=label, frames=frames))
        finals[label] = m
        print(json.dumps(dict(arm=label, final=reports[label]['final'], contacts=counts,
                             error=max(errors.values())), ensure_ascii=False), flush=True)
    a, b = finals['coupled'], finals['energy_storage']
    for name in a._arrays+('pre', 'post', 'capacity'):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for rng in ('rng_motion', 'rng_reaction', 'rng_field'):
        assert getattr(a,rng).bit_generator.state == getattr(b,rng).bit_generator.state
    df = a._recognition_total()-reports['coupled']['initial_recognition_free_energy']
    heat_check = b.heat_out-a.heat_out+df
    assert abs(heat_check) < 1e-8
    result = dict(date='2026-10-22', baseline_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        prior_material_report=str(args.baseline/'report.json'), prior_material_report_sha256=digest(args.baseline/'report.json'),
        params=prior['arms']['coupled']['params'], nodes=24, elapsed_ms=256., dt_ms=2., byte_ms=4.,
        byte_stream_hex=data.hex(), source_program_audit=source_audit, code_sha256=hashes,
        source_programs=program.provenance, identity_program_fingerprint=program.fingerprint(),
        genome_array_bytes=sum(v.nbytes for v in program.arrays.values()), arms=reports,
        energy_entropy_identifiability=dict(physical_arrays_and_rng_bitwise=True,
            maximum_free_energy_difference=abs(a.free_energy()-b.free_energy()),
            heat_difference=b.heat_out-a.heat_out, recognition_change=df, heat_relation_error=heat_check),
        scope='relative identity calibration and transport material interface on actual new encounters',
        biological_binding_energy_identified=False, absolute_conductance_calibrated=False,
        predictive_learning_benefit='not measured by this interface audit', passed=True)
    assert hashes == {str(p.relative_to(ROOT)): digest(p) for p in sources}
    (args.out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (args.out/'mobile_scene.json').write_text(json.dumps(dict(box_size=params.box_size,arms=scenes,
        default='coupled',scope=result['scope']),ensure_ascii=False,separators=(',',':')),encoding='utf-8')


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=Path('F:/fly_stream_overflow/cgb_mobile_bounded_20261022'))
    p.add_argument('--strength',type=Path,default=Path('F:/fly_stream_overflow/cgb_v3d_20261021/directed64_deploy.npz'))
    p.add_argument('--occurrence',type=Path,default=Path('F:/fly_stream_overflow/cgb_birth_calibration_20261021/occurrence_program.npz'))
    p.add_argument('--panel',type=Path,default=Path('F:/fly_stream_overflow/cgb_birth_calibration_20261021'))
    main(p.parse_args())
