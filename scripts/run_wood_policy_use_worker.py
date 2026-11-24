"""One unmodified life line, or one conditional two-transition continuation."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('natural','branch'),required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--temporary',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--option',type=int,choices=(0,1,2))
    p.add_argument('--option-basis-calibration',action='store_true')
    p.add_argument('--root-calibration-shadow',action='store_true')
    p.add_argument('--persistent-residual-refinement',action='store_true')
    p.add_argument('--selector-sign', type=int, choices=(-1, 1))
    p.add_argument('--quiet-depth', type=int, choices=range(8))
    a=p.parse_args()
    assert (a.mode=='branch')==(a.option is not None)
    assert not (a.option_basis_calibration and a.root_calibration_shadow)
    assert not a.persistent_residual_refinement or not (a.option_basis_calibration or a.root_calibration_shadow)
    assert not a.root_calibration_shadow or a.mode=='branch'
    a.work.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(a.protocol.read_text(encoding='utf8'))
    natural_rounds=int(protocol.get('natural_rounds',16))
    assert natural_rounds>0
    import torch
    from scripts.ib_local import run_mother_continuation as continuation,autonomous_cycle
    from scripts.ib_local.action_value_pair_audit import first_action
    from scripts.ib_local.finite_choice_contract import trace_choices
    from scripts.ib_local.wood_policy_measurement import action_receipt,score,segments,serializable_engine
    from scripts.ib_local.mother_checkpoint import save
    from scripts.run_root_native_correction_comparison import digest
    from scripts import run_native_preview_restore_comparison as runner
    from scripts.option_basis_calibration import calibrated_predictions
    torch.set_num_threads(1)
    prepare,dispatch=continuation.prepare_continuation,autonomous_cycle.dispatch
    loaded_holder=[]; opened=[]; count=0
    calibration_stats=None
    refinement_stats=None
    material=segments(protocol['validation'])
    observations={}; readings={}
    bounds=protocol['observation_rounds']
    score_rounds=set(protocol.get('score_rounds',[]))
    assert all(0<r<=natural_rounds for r in score_rounds)
    prefix_saved=set()
    remaining_quiet = a.quiet_depth
    if remaining_quiet is not None:
        assert a.mode == 'natural' and natural_rounds == 1 and a.selector_sign is None
    displacement = None
    if a.selector_sign is not None:
        assert a.mode == 'natural' and natural_rounds == 2
        displacement = json.loads(Path(protocol['selector_direction']).read_text(encoding='utf8'))
        assert displacement['paired_admission'] and digest(a.checkpoint) == displacement['source_sha256']

    def read(e,key):
        readings[str(key)]=score(e,material)
        write(a.work/'independent_scores.json',dict(material=protocol['validation'],readings=readings))
        return readings[str(key)]

    def prepared(*args,**kwargs):
        nonlocal calibration_stats,refinement_stats
        loaded=prepare(*args,**kwargs);loaded_holder.append(loaded)
        e=loaded.engine
        assert loaded.config['retention_scope_prior']['mode']=='off'
        assert e.persistent_rhythm_interleave and e.native_amortization['root_native_correction']=='observe'
        if protocol.get('record_selector_age'):
            from scripts.ib_local.native_delayed_credit import attach
            attach(e)
        if protocol.get('old_fact_material'):
            from shutil import copyfile
            copyfile(protocol['old_fact_material'], a.work/'run'/'material.pt')
        if a.option_basis_calibration:
            calibration=calibrated_predictions(e)
            calibration_stats=calibration.__enter__();opened.append(calibration)
        if a.persistent_residual_refinement:
            from scripts.native_value_residual_filter import refined_root_values
            refinement=refined_root_values(e)
            refinement_stats=refinement.__enter__();opened.append(refinement)
            loaded.config['persistent_value_refinement']=dict(schema=1,mode='root_two_fit',steps=2)
        if a.quiet_depth is not None:
            import hashlib
            from unittest.mock import patch
            take = e.world.take

            def checked_take(words, *args, **kwargs):
                assert not words, 'same-evidence pair requires the audited silent action'
                reply, holdout, info = take(words, *args, **kwargs)
                received = hashlib.sha256(reply.tobytes()+holdout.tobytes()).hexdigest()
                assert received == protocol['required_observation_sha256']
                return reply, holdout, info | dict(observation_sha256=received)

            context = patch.object(e.world, 'take', checked_take)
            context.__enter__(); opened.append(context)
        read(e,0)
        original=e._utter
        if a.mode=='natural':
            def utter():
                boundary=count+1
                if boundary not in bounds and not protocol.get('record_actions'):return original()
                assert str(boundary) not in observations
                with trace_choices() as trace:
                    words,record=original()
                receipt=action_receipt(e,words,record,trace)
                observation=dict(action=receipt)
                if boundary in bounds:
                    observation.update(source=prefix_saved_path(boundary),
                        source_sha256=digest(Path(prefix_saved_path(boundary))))
                observations[str(boundary)]=observation
                write(a.work/'observations.json',observations)
                return words,record
            e._utter=utter
        else:
            if a.root_calibration_shadow:
                from scripts.root_option_calibration_shadow import observe,choice_receipt
                observer=observe(e); shadows=observer.__enter__();opened.append(observer)
            def publish(receipt):
                write(a.work/'pre_outcome_action.json',receipt)
                if a.root_calibration_shadow:
                    assert len(shadows)==1
                    write(a.work/'root_calibration_shadow.json',choice_receipt(e,receipt,shadows[0]))
            context=first_action(e,a.option,publish,keep_actor_learning=True)
            context.__enter__();opened.append(context)
        return loaded

    def prefix_saved_path(boundary):
        return str(Path(protocol['prefix_paths'][str(boundary)])/'last.pt.gz')

    def observed_dispatch(e):
        nonlocal count, displacement, remaining_quiet
        if displacement is not None:
            assert not e.autonomous_ready and e.autonomous_event_clock+1 == displacement['selector_event']
            weight = e.model.binding_choice.weight
            with torch.no_grad():
                weight.add_(weight.new_tensor(displacement['delta']), alpha=a.selector_sign)
            displacement = None
        boundary=count+1
        if a.mode=='natural' and boundary in bounds and boundary not in prefix_saved:
            assert getattr(e,'_rhythm_runtime',None) is None
            loaded=loaded_holder[0]
            destination=Path(protocol['prefix_paths'][str(boundary)])
            destination.mkdir(parents=True,exist_ok=False)
            with serializable_engine(e):
                save(e,destination,loaded.cycle+count,loaded.config)
            write(destination/'config.json',loaded.config)
            read(e,f'before_{boundary}')
            prefix_saved.add(boundary)
        started=time.perf_counter()
        if remaining_quiet is None:
            row = dispatch(e)
        else:
            from unittest.mock import patch
            draw, sample = torch.multinomial, e._sample_internal_choice
            internal = remaining_quiet > 0

            def conditional_sample():
                def conditional_draw(probs, n, *args, **kwargs):
                    assert probs.shape == (2,) and n == 1
                    return torch.full_like(draw(probs, n, *args, **kwargs), int(internal))

                with patch.object(torch, 'multinomial', conditional_draw):
                    return sample()

            with patch.object(e, '_sample_internal_choice', conditional_sample):
                row = dispatch(e)
            assert row['autonomous_event']['kind'] == ('internal' if internal else 'release')
            remaining_quiet = remaining_quiet-1 if internal else None
        elapsed=time.perf_counter()-started
        if protocol.get('record_dispatch_timing'):
            with (a.work/'dispatch_timing.jsonl').open('a',encoding='utf8') as timing:
                timing.write(json.dumps(dict(dispatch_seconds=elapsed,
                    event=row['autonomous_event'],before_external=count,
                    internal_ticks=e.joint_internal_ticks),ensure_ascii=False)+'\n')
        if row['autonomous_event']['kind']!='external':return row
        count+=1
        if a.mode=='branch':
            read(e,count)
            if count==2:
                while opened:opened.pop().__exit__(None,None,None)
        else:
            if count in [b+1 for b in bounds] or count in score_rounds or count==natural_rounds:
                read(e,count)
            if count==natural_rounds:
                # Runner subsequently saves a complete body without this observer.
                del e._utter
        return row

    continuation.prepare_continuation=prepared
    autonomous_cycle.dispatch=observed_dispatch
    argv=sys.argv
    sys.argv=[argv[0],'--checkpoint',str(a.checkpoint),'--work',str(a.work/'run'),
        '--cycles',str(natural_rounds if a.mode=='natural' else 2),
        '--native-decision','--native-word-execution','--successor-distribution',
        '--joint-successor','--batch-native-words','--batch-preview-scores',
        '--amortized-decision','--latent-action-planner','--persistent-rhythm','coupled',
        '--root-native-correction','observe','--retention-scope-prior','off',
        '--table-storage',str(a.temporary/'table'),
        '--world-activations',str(a.temporary/'world'),
        '--policy-activations',str(a.temporary/'policy')]
    if protocol.get('old_fact_material'):
        sys.argv.extend(('--evaluate-every', '1'))
    try:
        runner.main()
        assert count==(natural_rounds if a.mode=='natural' else 2)
        if a.mode=='natural':assert prefix_saved==set(bounds)
    finally:
        while opened:opened.pop().__exit__(None,None,None)
        if calibration_stats is not None:
            write(a.work/'option_basis_runtime.json',dict(fits=calibration_stats['fits'],
                queries=calibration_stats['queries'],channel='local finite-use q',
                mode='private experiment'))
        if refinement_stats is not None:
            write(a.work/'persistent_residual_runtime.json',dict(**refinement_stats,
                channel='actual final-root persistent scalar value',mode='private experiment'))
        continuation.prepare_continuation=prepare
        autonomous_cycle.dispatch=dispatch
        sys.argv=argv


if __name__=='__main__':main()
