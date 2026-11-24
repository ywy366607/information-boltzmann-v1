"""ABBA known-payload execution on the complete real body, without learning."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
from scripts.ib_local.variable_expression import film
from scripts.ib_local.table_branch import table_branch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--body', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    args = continuation_parser().parse_args(['--checkpoint', str(a.body/'last.pt.gz'),
        '--output', str(a.body), '--cycles', '1', '--resume-output', '--world-source', 'text',
        '--table-storage', str(a.output/'table'), '--causal-storage', str(a.output/'causal')])
    loaded = prepare_continuation(args, resume_output=True)
    e = loaded.engine
    report = dict(cycle=loaded.cycle, source_sha=loaded.source_sha, lengths={})
    for size in (0, 25, 128):
        words = bytes(e.world.data[int(e.world.cursor):int(e.world.cursor)+size])
        def call(batch):
            if batch: e._utterance_payload_cache = {}
            with table_branch(e.table) as branch:
                result = film(e, words, branch)
            state = e.decision.state.clone()
            endpoint = e._draft_actions.pop()
            if batch: del e._utterance_payload_cache
            return result, state, endpoint
        expected, state, feature = call(False)
        actual, following, endpoint = call(True)
        times = {False:[], True:[]}
        for batch in (False, True, True, False, False, True, True, False):
            torch.cuda.synchronize(); start = time.perf_counter()
            call(batch)
            torch.cuda.synchronize(); times[batch].append(time.perf_counter()-start)
        report['lengths'][str(size)] = dict(before_seconds=statistics.median(times[False]),
            after_seconds=statistics.median(times[True]), samples=times,
            state_max_error=float((state-following).abs().max()),
            endpoint_max_error=float((feature-endpoint).abs().max()))
    (a.output/'known_payload.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report), flush=True)
    e.model._causal_archive.db.close()


if __name__ == '__main__': main()
