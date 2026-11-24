"""Single continuous body owner. Launch this file directly with Python.

Do not use -m: the legacy ib_local package initializer eagerly imports torch.

stdin/stdout are JSONL: {id,command:load|step|close,...} ->
{id,ok,result} or {id,ok:false,error:{type,message}}. No model imports or API
requests occur at startup. Node serializes requests and chooses control mode;
step always performs exactly one original complete body cycle and save.
"""
from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path
import sys
import time


class ContinuationBody:
    @classmethod
    def load(cls, request):
        # Imports and restore diagnostics belong to stderr, not the wire.
        root = str(Path(__file__).resolve().parents[3])
        if root not in sys.path:
            sys.path.insert(0, root)
        from scripts.ib_local import run_mother_continuation as runner
        from scripts.ib_local.mengli_harness.family_provider import FamilyCaregiver

        checkpoint = Path(request['checkpoint']).resolve()
        output = Path(request['output']).resolve()
        if not checkpoint.is_file():
            raise ValueError('checkpoint must name an existing complete snapshot')
        argv = ['--checkpoint', str(checkpoint), '--output', str(output),
                '--cycles', '1', '--mother-context', 'dsh']
        for key in ('mother_harness_url', 'mother_session', 'life_card'):
            if request.get(key) is not None:
                argv += ['--' + key.replace('_', '-'), str(request[key])]
        args = runner.continuation_parser().parse_args(argv)
        loaded = runner.prepare_continuation(args, provider_factory=FamilyCaregiver,
                                              resume_output=True)
        body = cls()
        body.loaded, body.runner = loaded, runner
        body.started = time.perf_counter()
        body.poisoned = False
        if loaded.life_card_data is None and loaded.life_card is not None:
            card_path = Path(loaded.life_card['path'])
            if card_path.is_file():
                card = json.loads(card_path.read_text(encoding='utf-8'))
                if (card.get('branch_id') == loaded.life_branch
                        and card.get('source', {}).get('cycle') == loaded.cycle):
                    loaded.life_card_data = card
        return body

    def status(self):
        loaded = self.loaded
        provider = loaded.engine.world.mother
        return dict(cycle=loaded.cycle,
                    checkpoint=str((loaded.output/'last.pt.gz').resolve()),
                    branch_id=provider.branch_id,
                    mother_checkpoint_ref=copy.deepcopy(provider.checkpoint_ref),
                    response_bytes=loaded.engine.response_bytes,
                    input_limit=dict(unit='utf8_bytes', capacity=loaded.engine.response_bytes,
                        includes_appended_newline=True, truncation='raw byte prefix; may split UTF-8'),
                    life_card=None if loaded.life_card is None else loaded.life_card['path'],
                    life_card_data=copy.deepcopy(loaded.life_card_data),
                    plan_version=None if provider.plan is None else provider.plan['version'])

    def step(self, request):
        if self.poisoned:
            raise RuntimeError('previous cycle failed; restart from the latest committed checkpoint')
        loaded, runner = self.loaded, self.runner
        engine = loaded.engine
        provider = engine.world.mother
        cycle = loaded.cycle + 1
        provider.begin_turn(caregiver=request.get('caregiver', 'mother'),
            text=request.get('text'), plan=request.get('plan'), cycle=cycle,
            response_bytes=engine.response_bytes)
        updates = 0
        original_step = engine.optimizer.step

        def counted_step(*args, **kwargs):
            nonlocal updates
            result = original_step(*args, **kwargs)
            updates += 1
            return result

        engine.optimizer.step = counted_step
        try:
            row = engine.step(runner.torch.randn(64, 32, device=engine.device))
            row['updates'] = updates
            row.update(cycle=cycle, elapsed_seconds=time.perf_counter() - self.started)
            # No acknowledgement until both complete body save and live export.
            runner.save(engine, loaded.output, cycle, loaded.config)
            loaded.cycle = cycle
            if loaded.life_card is not None:
                loaded.life_card_data = runner.export_live(engine,
                    checkpoint=loaded.output/'last.pt.gz', config=loaded.config,
                    cycle=cycle, row=row, output=loaded.life_card['path'],
                    branch_id=loaded.life_branch)
            with (loaded.output/'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            event = provider.last_event
            utterance = bytes.fromhex(row['utterance_hex'])
            return dict(self.status(), caregiver=event['caregiver'],
                utterance=dict(text=utterance.decode('utf-8', 'replace'),
                    hex=utterance.hex(), spoke=bool(utterance),
                    timing='before_caregiver_input'),
                caregiver_text=event['caregiver_text'],
                delivered=copy.deepcopy(event['delivered']),
                event_id=event['event_id'], choice=row['choice'], updates=updates,
                row=row)
        except Exception:
            # A partial real cycle must never be silently repeated or rolled back
            # in this process. Recovery is from the last complete on-disk commit.
            self.poisoned = True
            raise
        finally:
            engine.optimizer.step = original_step


class BodyOwner:
    def __init__(self, session_factory=None):
        self.session_factory = session_factory or ContinuationBody.load
        self.session = None
        self.closed = False
        self.load_attempted = False

    def dispatch(self, request):
        if self.closed:
            raise RuntimeError('body owner is closed')
        command = request.get('command')
        if command == 'close':
            self.closed = True
            return dict(closed=True, cycle=None if self.session is None else self.session.status()['cycle'])
        if command == 'load':
            if self.load_attempted:
                raise RuntimeError('body owner accepts one load; restart to change the individual')
            if not request.get('checkpoint') or not request.get('output'):
                raise ValueError('load requires explicit checkpoint and output')
            self.load_attempted = True
            self.session = self.session_factory(request)
            return self.session.status()
        if command == 'step':
            if self.session is None:
                raise RuntimeError('load an explicit checkpoint before stepping')
            return self.session.step(request)
        raise ValueError('command must be load, step, or close')


def serve(input_stream=None, output_stream=None, *, owner=None):
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    owner = BodyOwner() if owner is None else owner
    for line in input_stream:
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError('request must be an object')
            request_id = request.get('id')
            with contextlib.redirect_stdout(sys.stderr):
                result = owner.dispatch(request)
            response = dict(id=request_id, ok=True, result=result)
        except Exception as error:
            response = dict(id=request_id, ok=False,
                            error=dict(type=type(error).__name__, message=str(error)))
        output_stream.write(json.dumps(response, ensure_ascii=False) + '\n')
        output_stream.flush()
        if owner.closed:
            break


def main():
    # Node uses UTF-8 regardless of the Windows console codepage.
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    serve()


if __name__ == '__main__':
    main()
