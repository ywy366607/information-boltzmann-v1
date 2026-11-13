"""Bounded real-API wiring audit using a checkpoint's actual caregiver history.

This is a researcher-side mother conversation, not the child's life or training.
It requests two normal replies, one official compaction, and a cached fork replay.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

import torch

from scripts.ib_local.mengli_harness.mengli_world_client import DshMother, branch_for_output


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/cgb_causal_program_joint_20261111/joint_program_resume_02/last.pt.gz'


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(source, output, endpoint):
    source = source.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', source=str(source), source_sha256=file_hash(source),
                  scope='researcher-side caregiver wiring; zero child optimizer updates',
                  normal_api_budget=2, summary_api_budget=1,
                  diagnostic_override=dict(retain_tokens=64), calls=[])
    try:
        with gzip.open(source, 'rb') as stream:
            saved = torch.load(stream, map_location='cpu', weights_only=False)
        world, cycle, config = saved['world'], int(saved['cycle']), saved['config']
        history = copy.deepcopy(world['mother_transcript'])
        questions = [text for who, text in history if who == 'child']
        if len(questions) < 2:
            raise ValueError('audit needs two existing real child utterances')
        mother = DshMother(endpoint_url=endpoint, branch_id=branch_for_output(output),
            policy=config['mother_policy'], settings=dict(retain_tokens=64),
            provenance=dict(source_checkpoint=str(source), cycle=cycle,
                            source_sha256=report['source_sha256'],
                            scope=report['scope']))
        mother.load_state_dict(world['mother_provider_state'])
        mother.inherit_world_history(history)
        del saved, world

        def turn(provider, words):
            request = history + [('child', words)]
            reply = provider.reply(request)
            provider.commit_turn()
            report['calls'].append(dict(child_text=words, mother_reply=reply,
                telemetry=copy.deepcopy(provider.last_telemetry),
                checkpoint_ref=copy.deepcopy(provider.checkpoint_ref)))
            return reply

        reply = turn(mother, questions[0])
        history += [('child', questions[0]), ('mother', reply)]
        before_version = mother.checkpoint_ref.get('summary_version', 0)
        compacted = mother._call('compact', dict(branch_id=mother.branch_id,
                                                checkpoint_ref=mother.checkpoint_ref))
        mother.checkpoint_ref = copy.deepcopy(compacted['checkpoint_ref'])
        mother.usage_totals = copy.deepcopy(compacted['usage_totals'])
        report['compaction'] = compacted
        if mother.checkpoint_ref.get('summary_version', 0) <= before_version:
            raise RuntimeError('official compaction did not commit a new summary')
        boundary = mother.state_dict()
        (output/'before_replay.json').write_text(json.dumps(boundary, ensure_ascii=False,
                                                           indent=2)+'\n', encoding='utf-8')
        reply = turn(mother, questions[1])
        first_key = mother.last_telemetry.get('request_key')

        restored = DshMother(endpoint_url=endpoint,
            branch_id=branch_for_output(output/'restored'), policy=mother.policy,
            settings=dict(retain_tokens=64))
        restored.load_state_dict(boundary)
        replay = turn(restored, questions[1])
        second_key = restored.last_telemetry.get('request_key')
        if not first_key or first_key != second_key or reply != replay:
            raise RuntimeError('restored caregiver prefix changed the next actual request')
        if restored.last_telemetry.get('source') not in ('local', 'shared'):
            raise RuntimeError('fork replay was not served by the exact real-reply cache')
        report.update(status='passed', same_next_request=True, same_real_reply=True,
                      replay_source=restored.last_telemetry['source'])
        (output/'restored_provider.json').write_text(json.dumps(restored.state_dict(),
            ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['source_unchanged'] = file_hash(source) == report['source_sha256']
        (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n',
                                         encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8900')
    args = parser.parse_args()
    audit(args.source, args.output, args.endpoint)


if __name__ == '__main__':
    main()
