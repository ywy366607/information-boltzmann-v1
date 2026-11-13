"""Stage memory audit for the carried body; no world call or optimizer step."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path

import psutil
import torch


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                ('total_phys', ctypes.c_ulonglong), ('avail_phys', ctypes.c_ulonglong),
                ('total_page', ctypes.c_ulonglong), ('avail_page', ctypes.c_ulonglong),
                ('total_virtual', ctypes.c_ulonglong), ('avail_virtual', ctypes.c_ulonglong),
                ('avail_extended_virtual', ctypes.c_ulonglong)]


def memory_snapshot(stage):
    process = psutil.Process(os.getpid())
    info = process.memory_info()
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('GlobalMemoryStatusEx failed')
    data = dict(stage=stage, process_rss=info.rss,
                process_private=getattr(info, 'pagefile', None),
                available_physical=status.avail_phys,
                available_commit=status.avail_page)
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        data.update(cuda_allocated=torch.cuda.memory_allocated(),
                    cuda_reserved=torch.cuda.memory_reserved(),
                    cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved=torch.cuda.max_memory_reserved())
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--table-storage', type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    events = []

    def mark(stage):
        events.append(memory_snapshot(stage))
        args.output.write_text(json.dumps(dict(events=events, api_calls=0,
            optimizer_steps=0, completed=False), indent=2), encoding='utf-8')
        print(json.dumps(events[-1]), flush=True)

    torch.set_num_threads(1)
    mark('start')
    from scripts.ib_local import gdn_fused
    from scripts.ib_local.run_causal_water_reply import build
    gdn_fused.enable()
    mark('imports')
    engine, _, _, source = build(args.checkpoint, device='cuda', table_storage=args.table_storage)
    if getattr(engine, 'joint_world_cycle_schema', None) != 1:
        raise ValueError('expected the already migrated joint individual')
    mark('loaded')
    with torch.no_grad():
        words, record = engine._utter()
    mark('planned')
    torch.cuda.reset_peak_memory_stats()
    targets = torch.tensor(list(words), dtype=torch.long, device=engine.device)[None]
    actor, _ = engine._policy_grad(targets, len(words), record)
    mark('credit_complete')
    if not any(item is not None for item in actor):
        raise RuntimeError('missing proposal/action credit')
    result = dict(events=events, api_calls=0, optimizer_steps=0, completed=True,
                  cycle=source['cycle'], proposals=len(engine._joint_proposals),
                  choice=record['choice'], replay_error=engine._joint_proposal_replay_error,
                  scope='actual carried body, GPU planning and action-credit calculation')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
