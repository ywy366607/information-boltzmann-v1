"""Continuous five-room mechanism test; no organism update changes."""
import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local import gdn_fused, language_navigation_kanli_v4h
from scripts.ib_local import language_navigation_kanli_v4i
from scripts.ib_local import run_language_navigation_kanli_v4 as runner
from scripts.ib_local.language_navigation_kanli_v4m import WoodObservationBody
from scripts.ib_local.language_navigation_kanli_v4t import TypedWoodEngine
from scripts.ib_local.learning_rooms_world import LearningRoomsWorld, ROOMS, validation_banks
from scripts.ib_local.navigation_checkpoint import restore, save, snapshot
from scripts.ib_local.navigation_live_witness import frozen_table_logp
from scripts.ib_local.navigation_settlement import configure_settlement, configure_reward_memory
from scripts.ib_local.navigation_value_scale import configure_value_scale
from scripts.ib_local.symbol_navigation_world import SymbolNavigationWorld


class RoomAuditEngine(TypedWoodEngine):
    def _utter(self):
        self._candidate_words=[]
        return super()._utter()

    def _film(self,*args,**kwargs):
        result=super()._film(*args,**kwargs)
        self._candidate_words.append(result[0])
        return result

    def step(self,*args,**kwargs):
        old_room=self.world.anchor
        result=super().step(*args,**kwargs)
        # Pure diagnostic after the action: never changes its distribution.
        rooms=[LearningRoomsWorld.destination(w,old_room,self.world.routing) for w in self._candidate_words]
        choice=self._pending_record['choice']
        assert rooms[choice]==result['world']['anchor']
        result.update(candidate_rooms=rooms,candidate_words_hex=[w.hex() for w in self._candidate_words],
            candidate_g_predictions=(self._forecast['used'][:len(rooms),3]/10).tolist())
        return result


@torch.no_grad()
def score_bank(engine,bank):
    versions=tuple(p._version for p in engine.model.parameters())
    clock,events=int(engine.model.memory.clock),engine.table.events
    recurrent=engine.recurrent.clone()
    rng=torch.cuda.get_rng_state_all()
    result=[]
    for data in bank:
        raw=data[:-1]
        logp=torch.tensor(frozen_table_logp(engine.table,raw),device=engine.device)[None]
        state=engine.model.initial_state(1,device=engine.device)
        total=0.
        for i in range(0,len(raw),512):
            ids=torch.tensor(raw[i:i+512].astype(np.int64),device=engine.device)[None]
            targets=torch.tensor(data[i+1:i+1+len(ids[0])].astype(np.int64),device=engine.device)
            logits,state,_=engine.model.predict(ids,state,logp[:,i:i+512])
            total+=float(F.cross_entropy(logits[0],targets,reduction='sum'))
        result.append(total/len(raw))
    assert versions==tuple(p._version for p in engine.model.parameters())
    assert clock==int(engine.model.memory.clock) and events==engine.table.events
    assert torch.equal(recurrent,engine.recurrent)
    assert all(torch.equal(a,b) for a,b in zip(rng,torch.cuda.get_rng_state_all()))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cycles',type=int,default=512)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--settlement', choices=('witness','continuation','continuation_joint'), default='witness')
    parser.add_argument('--value-normalization', choices=('none','popart'), default='none')
    parser.add_argument('--reward-memory', choices=('td','observed'), default='td')
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);gdn_fused.enable()
    runner.LatentNavigationV3=WoodObservationBody
    language_navigation_kanli_v4i.FilmEngine=RoomAuditEngine
    language_navigation_kanli_v4h.ReplyTopicWorld=lambda *a,**kw:SymbolNavigationWorld(kw['seed'])
    engine,unused_table,unused_valid,cfg=runner.build(equal=True,reply=True,film=True,
        capture=False,validation_bytes=4096,seed=11)
    del unused_table,unused_valid
    source=Path('results/cgb_symbol_navigation_20261105/last.pt.gz')
    with gzip.open(source,'rb') as stream:
        state=torch.load(stream,weights_only=False)
    restore(engine,state);del state
    configure_settlement(engine,args.settlement)
    configure_value_scale(engine,args.value_normalization)
    configure_reward_memory(engine,args.reward_memory)
    engine.world=LearningRoomsWorld(11,switch_turn=256,routing='direct_bits')
    engine.response_bytes=engine.holdout_bytes=257
    banks=validation_banks()
    cfg.update(stage='five learning-progress rooms, full continuing symbol individual',
        source_snapshot=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        retained='all learned and carried organism state, optimizer, fast table, episode/witness memories, pending credit; only world replaced',
        world_source_bytes=None,response_bytes=257,holdout_bytes=257,cycles=args.cycles,
        switch_turn=256,seed=11,parameter_count=sum(p.numel() for p in engine.model.parameters()),
        routing='direct_bits',routing_note='last three byte parities address all rooms; periodic continuation leads directly to a learnable room',
        settlement=args.settlement,settlement_migration=getattr(engine,'settlement_migration',None),
        value_normalization=args.value_normalization,
        reward_memory=args.reward_memory,reward_memory_migration=getattr(engine,'reward_memory_migration',None),
        validation='no-write frozen-current-table/no-learning score, five independently generated 1024-byte banks, pre/post same bank',
        validation_sha256={str(k):hashlib.sha256(b''.join(x.tobytes() for x in v)).hexdigest() for k,v in banks.items()},
        sources_new={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__),Path('scripts/ib_local/learning_rooms_world.py'),
            Path('scripts/ib_local/navigation_live_witness.py'),
            Path('scripts/ib_local/language_navigation_kanli_v4t.py'),
            Path('scripts/ib_local/navigation_settlement.py'),
            Path('scripts/ib_local/navigation_value_scale.py'),
            Path('scripts/ib_local/navigation_checkpoint.py'),
            Path('scripts/ib_local/language_navigation_kanli_v4b.py'),
            Path('scripts/ib_local/language_navigation_kanli_v4c.py'),
            Path('scripts/ib_local/language_navigation_kanli_v4p.py'),
            Path('scripts/ib_local/language_navigation_kanli_v4q.py'))})
    (root/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    initial=score_bank(engine,banks[0])
    (root/'initial.json').write_text(json.dumps(dict(zip(ROOMS,initial)),indent=2),encoding='utf8')
    counted=[0];original_step=engine.optimizer.step
    def counted_step(*a,**kw):
        result=original_step(*a,**kw);counted[0]+=1;return result
    engine.optimizer.step=counted_step
    rows=[];start=time.perf_counter()
    with (root/'metrics.jsonl').open('w',encoding='utf8') as stream:
        for i in range(args.cycles):
            audit=i%16==0 or (i+1)%64==0 or i+1==args.cycles
            bank=banks[int(engine.world.turns>=engine.world.switch_turn)]
            pre=score_bank(engine,bank) if audit else None
            before=counted[0]
            row=engine.step(torch.randn(64,32,device=engine.device))
            row.update(cycle=i+1,optimizer_steps=counted[0]-before)
            assert np.isfinite(row['response_nll']) and 0<=row['match_probability']<=1
            assert row['metal_g']==row['settlement']==row['usable_signal']
            reward_field={'witness':'witness_g','continuation':'progress','continuation_joint':'joint_progress'}[args.settlement]
            assert row['settlement']==row[reward_field]
            assert engine.prev['g']==row['settlement']
            assert abs(row['wood_target'][3]-10*row['settlement'])<1e-4
            assert engine.intero_raw[1]==100*row['settlement']
            if args.reward_memory == 'observed':
                td = row['td_previous']
                assert td is None or td['memory_label_kind'] == 'observed_gain'
                memory = engine.model.memory
                slot = row['episode_slot']
                label_valid = bool(engine.reward_label_valid(torch.tensor([slot], device=engine.device))[0])
                complete = (int(memory.wood_target_count[slot]) == int(memory.count[slot])
                            and int(memory.wood_option[slot]) == int(memory.option[slot])
                            and bool(engine.reward_label_valid(torch.tensor([slot], device=engine.device),wood=True)[0]))
                difference = None
                if label_valid and complete:
                    difference = abs(float(memory.progress[slot])-float(memory.wood_target[slot,3])/10.)
                    assert difference < 1e-4
                row['observed_memory_audit'] = dict(valid=label_valid,
                    measured_gain=float(memory.progress[slot]),count=int(memory.count[slot]),
                    complete_wood_mean=complete,mean_difference=difference)
            if audit:
                post=score_bank(engine,bank)
                row.update(audit_room_pre=pre,audit_room_post=post,
                    audit_balanced_g=float(np.mean(pre)-np.mean(post)),
                    audit_destination_g=pre[row['world']['anchor']]-post[row['world']['anchor']])
                # Diagnose which observed alphabets the existing gold ruler
                # remembers, without changing its admission or weighting.
                windows=torch.cat(engine.w_ids,0).cpu().numpy()
                counts={name:0 for name in ROOMS};counts['mixed']=0
                for words in windows:
                    group=(words.astype(np.int64)-ord('a'))//2
                    key=ROOMS[int(group[0])] if 0<=group[0]<5 and np.all(group==group[0]) else 'mixed'
                    counts[key]+=1
                row['witness_alphabets']=counts
            if (i+1)%64==0 or i+1==args.cycles:
                save(engine,root,i+1,cfg)
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            if (i+1)%16==0 or i+1==args.cycles:
                print(json.dumps(dict(cycle=i+1,elapsed=time.perf_counter()-start,
                    choice=row['choice'],room=row['world']['room'],g=row['metal_g'],
                    moves=engine.world.moves,rooms_seen=sorted(set(r['world']['anchor'] for r in rows)),
                    future_nll=row.get('audit_room_post'))),flush=True)
    end=snapshot(engine)
    first=engine.step(torch.randn(64,32,device=engine.device))
    weights={k:v.detach().clone() for k,v in engine.model.state_dict().items()
             if torch.is_tensor(v) and v.is_floating_point() and v.numel()}
    restore(engine,end)
    second=engine.step(torch.randn(64,32,device=engine.device))
    error=max(float((engine.model.state_dict()[k]-v).abs().max()) for k,v in weights.items())
    assert error==0 and first['world']==second['world'] and first['metal_g']==second['metal_g']
    assert first['utterance_hex']==second['utterance_hex']
    restore(engine,end)
    summary=dict(cycles=len(rows),elapsed_seconds=time.perf_counter()-start,
        settlement=args.settlement,
        value_normalization=args.value_normalization,
        reward_memory=args.reward_memory,
        external_bytes=sum(r['external_bytes'] for r in rows),self_bytes=sum(r['self_bytes'] for r in rows),
        optimizer_steps=sum(r['optimizer_steps'] for r in rows),
        replay_samples=sum(r['wood_replay_samples'] for r in rows),
        parameter_count=cfg['parameter_count'],initial_nll=dict(zip(ROOMS,initial)),
        final_nll=dict(zip(ROOMS,rows[-1]['audit_room_post'])),
        visits={name:sum(r['world']['room']==name for r in rows) for name in ROOMS},
        moves=engine.world.moves,
        choices={a:sum(r['choice']==a for r in rows) for a in ('silence','generate','reuse')},
        max_policy_replay_error=max(r['policy_replay_max_error'] for r in rows),
        exact_next_step_parameter_error=error,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        scope='semantic-free mechanism diagnostic, one continuing individual; no real-language capability claim')
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
