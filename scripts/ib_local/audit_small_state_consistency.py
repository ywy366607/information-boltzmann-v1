"""Real-text zero-update state consistency audit, not another digit study."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.run_causal_water_reply import build
from scripts.ib_local.small_gdn_ruler import fingerprint, request_branch, sha256
from scripts.ib_local.small_gdn_stream_eval import CarriedStream
from scripts.ib_local.train_small_native_credit import continuation_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bytes', type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    gdn_fused.enable()
    report = dict(status='running', protocol=__doc__, persistent_optimizer_updates=0,
        real_data='existing OWT validation, first512 bytes after tokenizer offset filtering',
        source_script_sha256=sha256(__file__), observed_bytes=args.bytes, window=128, records=[])
    started = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def write():
        report['seconds'] = time.time()-started
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temp.replace(args.output)
    write()
    try:
        for checkpoint in args.checkpoints:
            engine, raw, offset, _ = build(checkpoint)
            selected = np.asarray(raw[:2*(args.bytes+1)])
            observed = (selected[selected >= offset]-offset)[:args.bytes].astype(np.uint8)
            assert len(observed) == args.bytes
            del raw, selected
            before = continuation_digest(engine)
            outputs, summary = {}, {}
            for mode in ('training_forward', 'carried', 'reset_native', 'reset_history', 'reset_both'):
                reset_native = mode in ('reset_native', 'reset_both')
                reset_history = mode in ('reset_history', 'reset_both')
                with request_branch(engine, reset_history=reset_history) as table:
                    reader = CarriedStream(engine, table, reset_native=reset_native)
                    result = reader.observe(observed, training_forward=mode == 'training_forward')
                    result['observed_memory_table_sha256'] = fingerprint(engine.model, table)
                assert continuation_digest(engine) == before
                outputs[mode] = result
                summary[mode] = {k:v for k,v in result.items() if k not in ('logits','targets','final_state')}
                print('STATE', checkpoint.parent.name, mode, result['nll'], flush=True)
            reference, carried = outputs['training_forward'], outputs['carried']
            max_error = float((reference['logits']-carried['logits']).abs().max())
            exact = dict(logits_max_error=max_error,
                recurrent_exact=torch.equal(reference['final_state'], carried['final_state']),
                targets_exact=torch.equal(reference['targets'], carried['targets']),
                pending_exact=reference['final_pending']==carried['final_pending'],
                observed_memory_table_exact=reference['observed_memory_table_sha256']==carried['observed_memory_table_sha256'])
            assert max_error == 0 and all(v for k,v in exact.items() if k != 'logits_max_error')
            record = dict(checkpoint=str(checkpoint), source_sha256=sha256(checkpoint), modes=summary,
                frozen_training_inference=exact, source_state_preserved=True)
            for mode in ('reset_native', 'reset_history', 'reset_both'):
                record['modes'][mode]['delta_nll_vs_carried'] = outputs[mode]['nll']-carried['nll']
                record['modes'][mode]['max_logit_change_vs_carried'] = float(
                    (outputs[mode]['logits']-carried['logits']).abs().max())
            report['records'].append(record)
            write()
            del engine, outputs
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status='complete', new_lean_coverage=0,
            script_unchanged=sha256(__file__)==report['source_script_sha256'])
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        write()


if __name__ == '__main__':
    main()
