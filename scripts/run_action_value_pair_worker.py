"""One private two-transition continuation, intervening only on its first actor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')


def write(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--temporary', type=Path, required=True)
    p.add_argument('--option', type=int, choices=(0,1,2), required=True)
    p.add_argument('--capture-cross-score', action='store_true')
    a = p.parse_args()
    a.work.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from torch.nn import functional as F
    from scripts.ib_local import run_mother_continuation as continuation
    from scripts.ib_local import autonomous_cycle
    from scripts.ib_local.action_value_pair_audit import first_action, tensor_digest
    from scripts.ib_local.navigation_live_witness import frozen_table_logp
    from scripts import run_native_preview_restore_comparison as runner
    torch.set_num_threads(1)
    prepare, dispatch = continuation.prepare_continuation, autonomous_cycle.dispatch
    scores, captured = [], []
    opened = []
    validation_identity = {}

    @torch.no_grad()
    def score(e, segments, event):
        params = tuple(p._version for p in e.model.parameters())
        state_hash = tensor_digest(e.recurrent)
        before = (int(e.model.memory.clock),int(e.table.history),e.model._causal_archive.db.total_changes)
        rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        q = getattr(e.model,'_question_context',None)
        e.model._question_context = None
        losses = []
        try:
            for raw in segments:
                logp = torch.as_tensor(frozen_table_logp(e.table,raw[:-1]),device=e.device)[None]
                ids = torch.as_tensor(raw[:-1].astype(np.int64),device=e.device)[None]
                target = torch.as_tensor(raw[1:].astype(np.int64),device=e.device)[None]
                state = e.recurrent.detach().clone()
                part = []
                for start in range(0,len(raw)-1,min(128,e.model.max_chunk)):
                    end = min(start+min(128,e.model.max_chunk),len(raw)-1)
                    logits,state,_ = e.model.predict(ids[:,start:end],state,logp[:,start:end])
                    part.extend(F.cross_entropy(logits[0],target[0,start:end],reduction='none').cpu().tolist())
                losses.append(part)
        finally:
            e.model._question_context = q
        assert params == tuple(p._version for p in e.model.parameters())
        assert tensor_digest(e.recurrent) == state_hash
        assert before == (int(e.model.memory.clock),int(e.table.history),e.model._causal_archive.db.total_changes)
        assert torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(x,y) for x,y in zip(cuda_rng,torch.cuda.get_rng_state_all()))
        result = dict(event=event,mean_nll=float(np.mean(losses)),
                      segment_nll=[float(np.mean(v)) for v in losses],per_byte_loss=losses,
                      evaluation='read-only current-body clone; shared external validation bytes; no table/memory assimilation')
        scores.append(result)
        write(a.work/'independent_scores.json',dict(material=validation_identity,scores=scores))

    def prepared(*args,**kwargs):
        loaded = prepare(*args,**kwargs)
        e = loaded.engine
        assert loaded.config['retention_scope_prior']['mode'] == 'off'
        assert e.persistent_rhythm_interleave and e.native_amortization['root_native_correction'] == 'observe'
        manifest = loaded.config['natural_text_corpus']
        path = Path(manifest['validation_path'])
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == manifest['validation_sha256']
        starts = np.linspace(0,len(raw)-2049,4,dtype=np.int64).tolist()
        segments = [np.frombuffer(raw[start:start+2049],np.uint8).copy() for start in starts]
        validation_identity.update(path=str(path),sha256=manifest['validation_sha256'],
                                   starts=starts,bytes_per_segment=2048,total_scored_bytes=8192,
                                   training_path=manifest['train_path'],used_in_training=False)
        if a.capture_cross_score:
            from scripts.ib_local.action_value_cross_score import capture_second_settlement
            snapshot = loaded.config['causal_archive']['snapshot']
            observer = capture_second_settlement(a.work/'second_settlement.pt', snapshot)
            observer.__enter__()
            opened.append(observer)
        score(e,segments,0)
        def publish(receipt):
            captured.append(receipt)
            write(a.work/'pre_outcome_action.json',receipt)
        context = first_action(e,a.option,publish)
        context.__enter__()
        opened.append(context)
        e._action_pair_segments = segments  # removed before standard checkpoint save
        return loaded

    count = 0
    def observed_dispatch(e):
        nonlocal count
        row = dispatch(e)
        if row['autonomous_event']['kind'] == 'external':
            count += 1
            score(e,e._action_pair_segments,count)
            if count == 1:
                assert row['choice'] == ('silence','generate','reuse')[a.option]
                row['action_pair_intervention'] = captured[0]['behavior']
            if count == 2:
                del e._action_pair_segments
                while opened:
                    opened.pop().__exit__(None,None,None)
        return row

    continuation.prepare_continuation = prepared
    autonomous_cycle.dispatch = observed_dispatch
    old_argv = sys.argv
    # Explicitly retain the runner's saved execution modes; it rejects omitted
    # amortization/rhythm flags and otherwise defaults native words to serial.
    sys.argv = [old_argv[0],'--checkpoint',str(a.checkpoint),'--work',str(a.work/'run'),
                '--cycles','2','--native-decision','--native-word-execution',
                '--successor-distribution','--joint-successor','--batch-native-words',
                '--batch-preview-scores','--amortized-decision','--latent-action-planner',
                '--persistent-rhythm','coupled','--root-native-correction','observe',
                '--retention-scope-prior','off','--table-storage',str(a.temporary/'table'),
                '--world-activations',str(a.temporary/'world'),
                '--policy-activations',str(a.temporary/'policy')]
    try:
        runner.main()
        assert count == 2 and len(captured) == 1 and not opened
    finally:
        while opened:
            opened.pop().__exit__(None,None,None)
        continuation.prepare_continuation = prepare
        autonomous_cycle.dispatch = dispatch
        sys.argv = old_argv


if __name__ == '__main__':
    main()
