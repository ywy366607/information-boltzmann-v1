"""Exact streaming of receipts for the two read-only maximum-clock scans.

This changes storage traversal only. Original journals and their replay/checksum
validation continue using production complete_jsonl unchanged.
"""
import json
import os
from pathlib import Path


def iter_complete_jsonl(path):
    path=Path(path)
    if not path.exists():return
    valid=0;torn=False
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                torn=True;break
            valid+=len(line)
            yield json.loads(line)
    if torn:
        with path.open('r+b') as stream:
            stream.truncate(valid);stream.flush();os.fsync(stream.fileno())


def install_runtime_scan(runtime):
    original=runtime.complete_jsonl
    def scan(path):
        if Path(path).name in ('metrics.jsonl','autonomous_events.jsonl'):
            return iter_complete_jsonl(path)
        return original(path)
    runtime.complete_jsonl=scan
    return original
