"""Persistent Session Store for MengLi Harness.

Follows dsh-session-persistence-jsonl design:
- Append-only event stream recording every interaction between MengLi and Mother.
- Zero loss of raw historical events, even after compaction summaries are committed.
- Replayable and audit-ready across sessions and crashes.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from scripts.ib_local.mengli_harness.telemetry import MengLiVitalSigns


class PersistentSession:
    def __init__(self, session_id: str, storage_dir: Path):
        self.session_id = session_id
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.storage_dir / f"{session_id}.jsonl"
        self.metadata_file = self.storage_dir / f"{session_id}.meta.json"
        self._init_metadata()

    def _init_metadata(self):
        if not self.metadata_file.exists():
            meta = {
                "session_id": self.session_id,
                "created_at": time.time(),
                "last_updated": time.time(),
                "total_turns": 0,
                "total_compaction_events": 0,
            }
            self.metadata_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def append_event(self, event_type: str, data: Dict[str, Any]):
        record = {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "event_type": event_type,  # 'turn' | 'compaction' | 'checkpoint'
            "data": data,
        }
        with open(self.session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Update metadata
        try:
            meta = json.loads(self.metadata_file.read_text(encoding="utf-8"))
            meta["last_updated"] = time.time()
            if event_type == "turn":
                meta["total_turns"] = meta.get("total_turns", 0) + 1
            elif event_type == "compaction":
                meta["total_compaction_events"] = meta.get("total_compaction_events", 0) + 1
            self.metadata_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass

    def record_turn(
        self,
        cycle: int,
        mengli_text: str,
        mother_reply: str,
        vitals: Optional[MengLiVitalSigns] = None,
        telemetry: Optional[Dict[str, Any]] = None,
    ):
        data = {
            "cycle": cycle,
            "child_text": mengli_text,
            "mother_reply": mother_reply,
            "vitals": vitals.to_dict() if vitals else {},
            "telemetry": telemetry or {},
        }
        self.append_event("turn", data)

    def record_compaction(self, old_summary: str, new_summary: str, version: int):
        data = {
            "version": version,
            "old_summary": old_summary,
            "new_summary": new_summary,
        }
        self.append_event("compaction", data)

    def load_recent_turns(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.session_file.exists():
            return []
        turns = []
        with open(self.session_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("event_type") == "turn":
                    turns.append(rec)
        return turns[-limit:]
