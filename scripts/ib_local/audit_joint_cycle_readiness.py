"""Read the trained individual and exercise joint planning without API or updates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.ib_local.run_causal_water_reply import build, sha256_file
from scripts.ib_local.navigation_joint_cycle import migrate_joint_cycle


def parameter_digest(model):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='loading',source=str(args.checkpoint),source_sha256=sha256_file(args.checkpoint),
                optimizer_updates=0,api_calls=0,training_ready=False,
                scope='existing trained body; internal continuation and language planning only')
    def write():args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    write()
    e,_,_,source=build(args.checkpoint,device='cpu')
    migrate_joint_cycle(e,4)
    report.update(status='loaded',cycle=source['cycle'],parameters=sum(p.numel() for p in e.model.parameters()),
                  parameter_sha256_before=parameter_digest(e.model))
    write()
    m=e.model.memory
    facts=m.last_write.clone();clock=int(m.clock);episodes=int(m.episodes)
    inner=e.continue_internal(1)
    report.update(status='internal_complete',internal=inner,
        internal_did_not_write_observations=(torch.equal(facts,m.last_write) and int(m.clock)==clock
                                            and int(m.episodes)==episodes))
    write()
    with torch.no_grad():words,record=e._utter()
    report.update(status='planning_complete',choice=record['choice'],utterance_hex=words.hex(),
        selection=e._joint_selection,proposal_count=len(e._joint_proposals),
        outer_action_count=len(e._draft_actions),outer_word_count=len(e._causal_words),
        parameter_sha256_after=parameter_digest(e.model),
        source_unchanged=sha256_file(args.checkpoint)==report['source_sha256'],
        caveat='planning/restore verified; full real event training admission remains separate')
    if report['parameter_sha256_before']!=report['parameter_sha256_after']:
        raise RuntimeError('readiness changed weights')
    write()
    print(json.dumps(report,ensure_ascii=True))


if __name__=='__main__':main()
