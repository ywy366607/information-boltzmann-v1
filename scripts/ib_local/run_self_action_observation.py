"""Existing mother continuation with passive action-forecast instrumentation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--expected-cycle', required=True, type=int)
    extra.add_argument('--observer-binding', type=Path)
    args, continuation_args = extra.parse_known_args()
    from scripts.ib_local import run_mother_continuation as runner
    from scripts.ib_local.self_action_observer import attach_self_action_observer
    original_prepare = runner.prepare_continuation

    def prepare(options, **kwargs):
        loaded = original_prepare(options, **kwargs)
        if loaded.cycle != args.expected_cycle:
            raise ValueError('continuation is not the explicitly approved completed boundary')
        attach_self_action_observer(loaded.engine)
        loaded.config['self_action_observation'] = dict(
            schema=1, start_cycle=loaded.cycle, planned_cycles=options.cycles,
            change='passive pre-response forecast copies only; original learning retained',
            controls=['same source across options', 'four generated drafts within one action type'],
            stopping='complete this observation budget, then assess the existing whole loop')
        if args.observer_binding is not None:
            path = args.observer_binding
            binding = json.loads(path.read_text(encoding='utf-8'))
            expected_body = options.checkpoint.resolve().parent
            if (Path(binding['body']).resolve() != expected_body or
                    binding['individual_id'] != loaded.config['life_identity']['individual_id'] or
                    binding['branch_id'] != loaded.life_branch):
                raise ValueError('refuse to redirect an unrelated workbench individual')
            (options.output.parent/'previous-observer-binding.json').write_text(
                json.dumps(binding, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            body = str(options.output.resolve())
            binding.update(run_id='self-action-observation-from-'+str(loaded.cycle),
                           body=body, life_card=loaded.life_card['path'],
                           guard_status=str((options.output.parent/'status.json').resolve()),
                           target_cycle=loaded.cycle+options.cycles)
            if body not in binding['segments']:
                binding['segments'].append(body)
            if len(binding['segments']) > 32:
                raise ValueError('observation exceeds the declared workbench segment budget')
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(binding, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            temporary.replace(path)
        return loaded

    runner.prepare_continuation = prepare
    sys.argv = [sys.argv[0], *continuation_args]
    runner.main()


if __name__ == '__main__':
    main()
