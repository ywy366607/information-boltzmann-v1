"""Compare original and private-batched replays of a complete saved body."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
from scripts.ib_local.native_retention_feedback import without_identity
from scripts.ib_local.native_replay_batch import logits as replay_logits
from scripts.ib_local.native_word_batch import table_payload
from scripts.ib_local.question_language_port import observation_scope


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--storage', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--length', type=int, default=480)
    args = p.parse_args()
    opts = continuation_parser().parse_args([
        '--checkpoint', str(args.checkpoint), '--output', str(args.checkpoint.parent),
        '--cycles', '1', '--resume-output', '--world-source', 'text',
        '--life-card', 'data/harness/life/replay-batch-readonly.json',
        '--table-storage', str(args.storage/'table'), '--causal-storage', str(args.storage/'causal')])
    loaded = prepare_continuation(opts, resume_output=True)
    e, m = loaded.engine, loaded.engine.model
    memory, field = m.memory, m.body.coevolution
    source = e.recurrent.clone()
    from scripts.ib_local.adaptive_hamiltonian import bridge_state
    start = bridge_state(m, source)
    candidate = [(-1,-1)] + [(int(s),int(v)) for s,v,ok in zip(field.slots_id,field.versions,field.valid) if bool(ok)]
    cursor = int(e.world.cursor)
    words = bytes(np.asarray(e.world.data[cursor:cursor+args.length-1],dtype=np.uint8))
    ids, lp = table_payload(e, words)
    ids, lp = torch.as_tensor(ids, device=e.device)[None], torch.as_tensor(lp, device=e.device)[None]
    expected = []
    timings = {width:[] for width in (1,2,4,8)}
    errors = {}
    peak = {}
    with torch.no_grad(), observation_scope(m, 0, len(ids[0])):
        e.decision._refresh_projections()
        def run(width):
            if width == 1:
                result=[]
                for s,v in candidate:
                    with without_identity(m,s,v):
                        result.append(m.predict(ids,start,lp)[0])
                return result
            return list(replay_logits(e,ids,start,lp,candidate,batch_size=width))
        with e.decision._decode_context():
            for width in timings:
                actual = run(width)
                if width == 1:
                    expected = [x.cpu() for x in actual]
                else:
                    errors[width] = max(float((x.cpu()-y).abs().max()) for x,y in zip(actual,expected))
                    for x,y in zip(actual,expected):
                        torch.testing.assert_close(x.cpu(),y,atol=3e-5,rtol=2e-4)
                del actual
            for width in (1,4,2,8,8,2,4,1):
                gc.collect(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                began = time.perf_counter()
                actual = run(width)
                torch.cuda.synchronize()
                timings[width].append(time.perf_counter()-began)
                peak[width] = torch.cuda.max_memory_allocated()
                del actual
    result = dict(source=str(args.checkpoint), source_sha256=loaded.source_sha,
        bytes=len(ids[0]), branches=len(candidate), seconds=timings,
        median_seconds={k:float(np.median(v)) for k,v in timings.items()},
        peak_gpu_bytes=peak, logit_max_errors=errors,
        carried_state_unchanged=torch.equal(source,e.recurrent), optimizer_updates=0, world_calls=0)
    assert result['carried_state_unchanged']
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))
    m._causal_archive.db.close()


if __name__ == '__main__':
    main()
