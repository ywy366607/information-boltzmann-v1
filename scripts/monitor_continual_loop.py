"""Follow existing complete-body learning without loading Torch or the body."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import time
from collections import deque

from scripts.analyze_complete_optimizer_loop import summary
from scripts.audit_logged_choice_effect import effects


def read_rows(path):
    lines=path.read_text(encoding='utf-8').splitlines()
    rows=[]
    for index,line in enumerate(lines):
        try: rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1: raise
            break  # The live writer may be finishing the final record.
    return rows


class EventTail:
    """Incremental UTF-8 JSONL reader; memory stays bounded during long runs."""
    def __init__(self,path,limit=512):
        self.stream=path.open(encoding='utf-8')
        self.rows=deque(maxlen=limit)
        self.events=self.external_bytes=self.updates=0
        self.last_validation_cycle=None

    def poll(self):
        while True:
            offset=self.stream.tell()
            line=self.stream.readline()
            if not line:break
            if not line.endswith('\n'):
                self.stream.seek(offset);break
            row=json.loads(line)
            self.rows.append(row)
            self.events+=1;self.external_bytes+=row['external_bytes'];self.updates+=row['updates']
            if row.get('fixed_validation'):self.last_validation_cycle=row['cycle']
        return list(self.rows)

    def close(self):
        self.stream.close()


def report(rows, initial, target_cycle):
    if not rows: return dict(events=0,main_written=False)
    failures={key:[] for key in ('cycle_gap','source_bytes','external_accounting',
                                 'paid_world_calls','policy_replay','workspace_layers')}
    for index,r in enumerate(rows):
        cycle=r['cycle']; world=r['world']
        if index and cycle != rows[index-1]['cycle']+1: failures['cycle_gap'].append(cycle)
        if sum(world['text_source_bytes'].values()) != r['external_bytes']:
            failures['source_bytes'].append(cycle)
        if r.get('world_learning',{}).get('external_bytes_once') != r['external_bytes']:
            failures['external_accounting'].append(cycle)
        if world.get('provider_calls',0) or world.get('mother_bytes',0):failures['paid_world_calls'].append(cycle)
        if r.get('policy_replay_max_error',0)>1e-3:failures['policy_replay'].append(cycle)
        if r['past_present']['layers']!=32:failures['workspace_layers'].append(cycle)
    return dict(schema=1,scope='execution integrity and causal usefulness kept separate',
        main_written=False,optimizer_updates_by_monitor=0,world_calls_by_monitor=0,
        first_cycle=rows[0]['cycle'],last_cycle=rows[-1]['cycle'],events=len(rows),
        external_bytes=sum(r['external_bytes'] for r in rows),
        actual_optimizer_updates=sum(r['updates'] for r in rows),
        loop=summary(rows),integrity=failures,initial_validation=initial,
        independent_validation=[dict(cycle=r['cycle'],**r['fixed_validation'])
                                for r in rows if r.get('fixed_validation')],
        local_plasticity=[dict(cycle=r['cycle'],**r['optimizer_plasticity'])
                          for r in rows if r.get('optimizer_plasticity')],
        distinct_navigation_anchors=len({r['world']['anchor'] for r in rows}),
        matched_navigation_events=sum(r['world'].get('navigation_source')=='matched' for r in rows),
        last_runtime=rows[-1].get('runtime'),
        selection_effects=dict(one_loop=effects(rows,horizon=1),two_loops=effects(rows,horizon=2)),
        remaining_to_target=max(0,target_cycle-rows[-1]['cycle']),
        interpretation='read/write activity is execution evidence; usefulness needs stable causal contrasts; no dialogue capability verdict')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--body',type=Path,required=True)
    p.add_argument('--guard',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--target-cycle',type=int,required=True)
    a=p.parse_args()
    if os.name!='nt':raise ValueError('Windows process-handle watcher required')
    guard=json.loads(a.guard.read_text(encoding='utf-8'))
    initial=json.loads((a.body/'initial_validation.json').read_text(encoding='utf-8'))
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    # Keep a handle to this exact supervisor, not a possibly recycled PID.
    handle=kernel.OpenProcess(0x100000,False,int(guard['pid']))
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    bucket=None;published_validation=None
    tail=EventTail(a.body/'metrics.jsonl')
    try:
        while True:
            rows=tail.poll()
            wait=kernel.WaitForSingleObject(handle,0)
            if wait not in (0,258):raise ctypes.WinError(ctypes.get_last_error())
            alive=wait==258
            if (bucket is None or tail.events//128!=bucket or not alive or
                    tail.last_validation_cycle != published_validation):
                result=report(rows,initial,a.target_cycle)
                result['history_scope']='rolling most recent 512 events; independent lifetime totals'
                result['lifetime_totals']=dict(events=tail.events,external_bytes=tail.external_bytes,updates=tail.updates)
                current=json.loads(a.guard.read_text(encoding='utf-8'))
                result.update(metrics_source=str(a.body/'metrics.jsonl'),supervisor_live=alive,
                    supervisor={k:current.get(k) for k in ('status','pid','child_pid','resources','child_memory','guard_stop_reason')})
                a.report.parent.mkdir(parents=True,exist_ok=True)
                tmp=a.report.with_suffix(a.report.suffix+'.tmp')
                tmp.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');tmp.replace(a.report)
                print(json.dumps(dict(events=len(rows),cycle=result.get('last_cycle'),supervisor_live=alive)),flush=True)
                bucket=tail.events//128
                published_validation=tail.last_validation_cycle
            if not alive:break
            time.sleep(5)
    finally:
        tail.close()
        kernel.CloseHandle(handle)


if __name__=='__main__':main()
