"""Inspect the actual native retention consumer and its persisted evidence.

Complete private restore, no optimizer update or world call. An enabled rule
is reported separately from evidence accumulated by its production writer.
"""
import argparse
import inspect
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--measure-cost', action='store_true')
    args = p.parse_args()
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
    from scripts.ib_local.run_causal_water_reply import sha256_file
    from scripts.ib_local.experience_consolidation import bounded_preference, replacement_slot
    torch.set_num_threads(1)
    before = sha256_file(args.checkpoint)
    opts = continuation_parser().parse_args(['--checkpoint', str(args.checkpoint),
        '--output', str(args.work/'body'), '--cycles', '1', '--world-source', 'text',
        '--autonomous-events', '--optimizer-policy', 'budgeted',
        '--table-storage', str(args.work/'table'), '--causal-storage', str(args.work/'causal')])
    loaded = prepare_continuation(opts)
    e, model = loaded.engine, loaded.engine.model
    try:
        m = model.memory
        stats = {}
        for name in ('read_credit_count', 'read_credit_steps', 'read_credit_mean',
                     'retention_utility_steps', 'retention_utility_first', 'retention_utility_second'):
            x = getattr(m, name).detach().double()
            stats[name] = dict(nonzero=int(torch.count_nonzero(x)), total=float(x.sum()),
                minimum=float(x.min()), median=float(x.median()), mean=float(x.mean()), maximum=float(x.max()))
        preference = bounded_preference(m)
        out = dict(cycle=loaded.cycle, source=str(args.checkpoint), source_sha256=before,
            source_unchanged=before == sha256_file(args.checkpoint), optimizer_updates=0, world_calls=0,
            engine=type(e).__name__, model=type(model).__name__,
            native_retention_feedback_mode=getattr(e, 'native_retention_feedback_mode', 'off'),
            configured_feedback=loaded.config.get('native_retention_feedback'),
            runtime_methods={name:dict(file=inspect.getsourcefile(method),
                line=inspect.getsourcelines(method)[1]) for name,method in
                [('objective',model.objective),('advance',e._advance),('step',e.step)]},
            retention=dict(enabled=m.consolidation_enabled, rule=m.consolidation_rule,
                read_credit_enabled=m.read_credit_enabled, read_credit_rule=m.read_credit_rule,
                capacity=m.capacity, occupied=int(m.occupied.sum()),
                preference_nonzero=int(torch.count_nonzero(preference)),
                replacement_slot=replacement_slot(m), statistics=stats),
            scope='Persisted evidence and actual runtime entry points, not a future-use experiment.')
        if args.measure_cost:
            import numpy as np
            from scripts.ib_local.language_action import peek
            from scripts.ib_local.native_retention_feedback import configure, measure
            from scripts.ib_local.past_present_coevolution import refresh_workset
            configure(e, 'observe')
            refresh_workset(e)
            # Fixed real text is numerical admission material only. It is not
            # a later-use observation or training exposure and is never paid.
            raw = np.asarray(e.world.data[:481], dtype=np.uint8)
            ids = torch.tensor(raw[:-1].astype(np.int64), device=e.device)[None]
            targets = torch.tensor(raw[1:].astype(np.int64), device=e.device)[None]
            q = np.asarray(peek(e.table, e.pending))
            logp = torch.tensor(np.log(np.maximum(q, 1e-6)), device=e.device)[None,None]
            logp = logp.expand(1, ids.shape[1], 256).contiguous()
            start = e.recurrent.detach().clone()
            with torch.no_grad():
                logits = model.predict(ids, start, logp)[0]
                diag = dict(logits=logits)
                measure(e, ids, targets, start, logp, diag)
                out['cost_admission'] = diag['retention_feedback']
                out['cost_admission']['material'] = 'fixed corpus prefix, no settlement, not a future-use result'
                assert torch.equal(start, e.recurrent)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        print(json.dumps(out, ensure_ascii=False))
    finally:
        model._causal_archive.close()


if __name__ == '__main__':
    main()
