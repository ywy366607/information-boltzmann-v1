"""A bounded SSD causal archive behind the existing episode/relation worksets.

Episodes include their optional utterance; relations reference the same episode
IDs. Only selected facts move to CUDA, where the unchanged model re-encodes them.
SQLite connections are runtime resources, never tensors or checkpoint payloads.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def columns(engine):
    """One record, including existing attachment and settlement provenance."""
    model, memory = engine.model, engine.model.memory
    result = {k: v for k, v in memory.named_buffers(recurse=False)
              if v.ndim and v.shape[0] == memory.capacity and k not in
              {'pending', 'pending_view', 'pending_valid', 'history', 'history_clock', 'history_valid'}}
    result.update({'@'+k: v for k, v in model.retrieval.named_buffers(recurse=False)})
    for name in ('settlement_versions', 'wood_settlement_versions'):
        if hasattr(engine, name):
            result['$'+name] = getattr(engine, name)
    return result


class CausalArchive:
    def __init__(self, engine, directory, *, manifest=None, episode_capacity=65536,
                 relation_capacity=65536, max_bytes=512*1024*1024):
        if min(episode_capacity, relation_capacity) < 1 or max_bytes < 4096:
            raise ValueError('positive archive budgets required')
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / ('working_'+uuid.uuid4().hex+'.sqlite')
        self.engine = engine
        self.db = sqlite3.connect(self.path)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA cache_size=-2048')  # 2 MiB host page-cache target
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA journal_mode=DELETE')
        if manifest is not None:
            source = Path(manifest['snapshot'])
            if manifest.get('schema') != 1 or sha256(source) != manifest['sha256']:
                self.close()
                raise ValueError('causal archive snapshot identity mismatch')
            source_db = sqlite3.connect(source.as_uri()+'?mode=ro', uri=True)
            try:
                source_db.backup(self.db)
            finally:
                source_db.close()
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT, origin_slot INTEGER NOT NULL,
                stamp INTEGER NOT NULL, payload BLOB NOT NULL, touched INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS episode_origin ON episodes(origin_slot,stamp);
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY, generation INTEGER NOT NULL, version INTEGER NOT NULL,
                condition BLOB NOT NULL, touched INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS evidence (
                relation_id INTEGER NOT NULL REFERENCES relations(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL, episode_id INTEGER NOT NULL REFERENCES episodes(id),
                PRIMARY KEY (relation_id,ordinal));
            CREATE INDEX IF NOT EXISTS evidence_episode ON evidence(episode_id);
            CREATE TABLE IF NOT EXISTS relation_program (
                relation_id INTEGER PRIMARY KEY REFERENCES relations(id) ON DELETE CASCADE,
                payload TEXT NOT NULL);
        ''')
        self.fields = columns(engine)
        schema = {k: dict(shape=list(v.shape[1:]), dtype=v.detach().cpu().numpy().dtype.str)
                  for k, v in self.fields.items()}
        old = self._meta('schema')
        if old is not None and old != schema:
            self.close()
            raise ValueError('archive/body episode schema differs')
        self.schema = schema
        self.layouts = []
        offset = 0
        for k in sorted(schema):
            spec = schema[k]
            count = int(np.prod(spec['shape'])) if spec['shape'] else 1
            size = count*np.dtype(spec['dtype']).itemsize
            self.layouts.append((k, spec, offset, size))
            offset += size
        self.record_bytes = offset
        saved_budget = self._meta('budgets')
        budgets = dict(episodes=episode_capacity, relations=relation_capacity, max_bytes=max_bytes)
        if saved_budget is not None and saved_budget != budgets:
            self.close()
            raise ValueError('archive budget change requires an explicit migration')
        self.budgets = budgets
        self.hot = self._meta('hot', {})
        self.counters = dict(warmed_relations=0, loaded_episodes=0, scans=0)
        self.db.execute('PRAGMA max_page_count='+str(max_bytes//4096))
        with self.db:
            self._set_meta('schema', schema)
            self._set_meta('budgets', budgets)
        # Restored snapshots must refer to exactly the supplied hot state.
        if manifest is not None:
            for slot, mapping in self.hot.items():
                if self._signature(int(slot)) != mapping['signature']:
                    self.close()
                    raise ValueError('archive snapshot does not match body workset')
        try:
            self.sync()
        except BaseException:
            self.close()
            raise

    def _meta(self, name, default=None):
        row = self.db.execute('SELECT value FROM meta WHERE key=?', (name,)).fetchone()
        return default if row is None else json.loads(row[0])

    def _set_meta(self, name, value):
        self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (name, json.dumps(value)))

    def close(self):
        self.db.close()

    def _signature(self, slot):
        m = self.engine.model.memory
        return [int(m.last_write[slot]), int(m.read_credit_version[slot]),
                int(m.birth[slot]), int(m.count[slot])]

    def _pack(self, slot, arrays):
        return b''.join(arrays[k][slot].tobytes() for k, _, _, _ in self.layouts)

    def _unpack(self, payload):
        if len(payload) != self.record_bytes:
            raise ValueError('truncated episode payload')
        return {k: np.frombuffer(payload, dtype=spec['dtype'], count=size//np.dtype(spec['dtype']).itemsize,
                                 offset=offset).reshape(spec['shape']).copy()
                for k, spec, offset, size in self.layouts}

    def _episode(self, slot, arrays):
        m = self.engine.model.memory
        signature = self._signature(slot)
        old = self.hot.get(str(slot))
        payload = self._pack(slot, arrays)
        if old is not None and old['signature'] == signature:
            eid = old['id']
            self.db.execute('UPDATE episodes SET payload=?,touched=? WHERE id=?',
                            (payload, int(m.last_touch[slot]), eid))
        else:
            self._room('episodes')
            cursor = self.db.execute('INSERT INTO episodes(origin_slot,stamp,payload,touched) VALUES (?,?,?,?)',
                (slot, int(m.last_write[slot]), payload, int(m.last_touch[slot])))
            eid = cursor.lastrowid
            self.hot[str(slot)] = dict(id=eid, signature=signature)
        return eid

    def _room(self, table):
        if self.db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] < self.budgets[table]:
            return
        if table == 'episodes':
            protected = [v['id'] for v in self.hot.values()]
            self.db.execute('CREATE TEMP TABLE IF NOT EXISTS pinned(id INTEGER PRIMARY KEY)')
            self.db.execute('DELETE FROM pinned')
            self.db.executemany('INSERT OR IGNORE INTO pinned VALUES (?)', ((i,) for i in protected))
            row = self.db.execute('''SELECT id FROM episodes WHERE id NOT IN (SELECT id FROM pinned)
                AND id NOT IN (SELECT episode_id FROM evidence) ORDER BY touched,id LIMIT 1''').fetchone()
        else:
            live = self.engine.model.causal_relations.identity.detach().cpu().tolist()
            placeholders = ','.join('?' for _ in live)
            row = self.db.execute('SELECT id FROM relations WHERE id NOT IN ('+placeholders+
                ') ORDER BY touched,id LIMIT 1', live).fetchone()
        if row is None:
            raise RuntimeError('archive budget full: all release candidates are referenced or hot')
        self.db.execute('DELETE FROM '+table+' WHERE id=?', row)

    @torch.no_grad()
    def sync(self):
        """Persist one committed boundary, with shared payloads and FK evidence."""
        m, bank = self.engine.model.memory, self.engine.model.causal_relations
        arrays = {k: v.detach().cpu().contiguous().numpy() for k, v in self.fields.items()}
        old_hot = json.loads(json.dumps(self.hot))
        try:
            with self.db:
                for slot in torch.nonzero(m.occupied).flatten().tolist():
                    self._episode(slot, arrays)
                for i in torch.nonzero(bank.identity > 0).flatten().tolist():
                    rid, version = int(bank.identity[i]), int(bank.version[i])
                    old = self.db.execute('SELECT version FROM relations WHERE id=?', (rid,)).fetchone()
                    if old is None:
                        self._room('relations')
                        self.db.execute('INSERT INTO relations VALUES (?,?,?,?,?)',
                            (rid, int(bank.generation[i]), version,
                             bank.condition[i].detach().cpu().numpy().tobytes(), int(bank.last_use[i])))
                    else:
                        self.db.execute('UPDATE relations SET version=?,condition=?,touched=? WHERE id=?',
                            (version, bank.condition[i].detach().cpu().numpy().tobytes(), int(bank.last_use[i]), rid))
                    if hasattr(bank, 'program_owner'):
                        from scripts.ib_local.generative_relations import EXTRA_FIELDS
                        payload = {name: getattr(bank, name)[i].detach().cpu().tolist()
                                   for name in EXTRA_FIELDS}
                        self.db.execute('INSERT OR REPLACE INTO relation_program VALUES (?,?)',
                                        (rid, json.dumps(payload)))
                    # If a physical slot disappeared without a logical revision,
                    # keep the disk edge. Physical eviction is not a refutation.
                    if old is None or old[0] != version:
                        self.db.execute('DELETE FROM evidence WHERE relation_id=?', (rid,))
                    for j in torch.nonzero(bank.valid[i]).flatten().tolist():
                        slot, stamp = int(bank.slots[i, j]), int(bank.stamps[i, j])
                        if (bool(m.occupied[slot]) and int(m.count[slot]) == 1 and
                            int(m.last_write[slot]) == stamp and bool(m.mother_valid[slot]) and
                            int(m.mother_stamp[slot]) == stamp):
                            self.db.execute('INSERT OR REPLACE INTO evidence VALUES (?,?,?)',
                                            (rid, j, self.hot[str(slot)]['id']))
                self._set_meta('hot', self.hot)
        except BaseException:
            self.hot = old_hot
            raise

    @torch.no_grad()
    def best_relation(self, source, chunk_size=256):
        """Exact current-encoder search at decision time, bounded GPU scratch.

        Cost grows with configured cold capacity. This deliberately makes no
        constant-time or stale-embedding-index claim.
        """
        model = self.engine.model
        query = F.normalize(model.encoder(source), dim=-1)
        best = None
        scoped = hasattr(model.causal_relations, 'scope_axes')
        cursor = self.db.execute('''SELECT r.id,r.condition,p.payload FROM relations r
            LEFT JOIN relation_program p ON p.relation_id=r.id ORDER BY r.id'''
            if scoped else 'SELECT id,condition FROM relations ORDER BY id')
        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            x = np.stack([np.frombuffer(r[1], dtype=self.schema['source']['dtype']) for r in rows])
            keys = torch.as_tensor(x, device=source.device, dtype=source.dtype)
            scores = (query @ F.normalize(model.encoder(keys), dim=-1).T)[0]
            if scoped:
                raw_source = source[0].detach().cpu().numpy()
                for j, row in enumerate(rows):
                    if row[2] is None:
                        continue
                    metadata = json.loads(row[2])
                    valid = np.asarray(metadata['scope_valid'], dtype=bool)
                    if not valid.any():
                        continue
                    axes = np.asarray(metadata['scope_axes'], dtype=raw_source.dtype)[valid]
                    bounds = np.asarray(metadata['scope_bounds'], dtype=raw_source.dtype)[valid]
                    signs = np.asarray(metadata['scope_signs'], dtype=raw_source.dtype)[valid]
                    projection = axes @ raw_source
                    tolerance = np.finfo(raw_source.dtype).eps*(1+abs(projection)+abs(bounds))*8
                    if (signs*(projection-bounds) > tolerance).any():
                        scores[j] = -torch.inf
            if not bool(torch.isfinite(scores).any()):
                continue
            value, at = scores.max(0)
            item = (float(value), rows[int(at)][0])
            if best is None or item[0] > best[0]:
                best = item
        self.counters['scans'] += 1
        return best

    def _protected(self):
        """The delayed actor/next-boundary and retained question are still live."""
        from dataclasses import is_dataclass
        episodes, relations = set(), set()

        def visit(value):
            if is_dataclass(value) and hasattr(value, 'identity') and hasattr(value, 'slots'):
                relations.add(int(value.identity))
                episodes.update(value.slots[value.valid].detach().cpu().tolist())
            elif isinstance(value, dict):
                if isinstance(value.get('slot'), int) and value['slot'] >= 0:
                    episodes.add(value['slot'])
                if torch.is_tensor(value.get('slots')):
                    slots = value['slots']
                    valid = value.get('valid', torch.ones_like(slots, dtype=torch.bool))
                    episodes.update(slots[valid].detach().cpu().reshape(-1).tolist())
                for v in value.values():
                    visit(v)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    visit(v)

        for name in ('prev', 'internal_pending', '_retained_question_receipt'):
            visit(getattr(self.engine, name, None))
        return episodes, relations

    @torch.no_grad()
    def warm_relation(self, rid):
        """Reinstall a cold relation and its exact evidence in finite hot slots."""
        model, m = self.engine.model, self.engine.model.memory
        bank = model.causal_relations
        if m._read_participation:
            raise RuntimeError('cannot page a workset with unsettled read participation')
        row = self.db.execute('SELECT generation,version,condition FROM relations WHERE id=?', (rid,)).fetchone()
        if row is None:
            raise KeyError(rid)
        edges = self.db.execute('''SELECT ordinal,episode_id,payload FROM evidence JOIN episodes
            ON episodes.id=evidence.episode_id WHERE relation_id=? ORDER BY ordinal''', (rid,)).fetchall()
        if len(edges) > bank.evidence_budget:
            raise ValueError('disk relation exceeds the live evidence budget')
        pinned, protected_relations = self._protected()
        live = torch.nonzero(bank.identity == rid).flatten().tolist()
        candidates = [i for i in range(bank.capacity) if int(bank.identity[i]) not in protected_relations]
        if not live and not candidates:
            return False
        index = live[0] if live else min(candidates, key=lambda i: (int(bank.identity[i]) > 0, int(bank.last_use[i]), i))
        mapping = {v['id']: int(k) for k, v in self.hot.items()
                   if self._signature(int(k)) == v['signature']}
        chosen = {eid: mapping[eid] for _, eid, _ in edges if eid in mapping}
        pinned.update(chosen.values())
        victims = sorted((i for i in range(m.capacity) if i not in pinned),
                         key=lambda i: (bool(m.occupied[i]), int(m.last_touch[i]), i))
        missing = list(dict.fromkeys(eid for _, eid, _ in edges if eid not in chosen))
        if len(victims) < len(missing):
            return False
        payloads = {eid: payload for _, eid, payload in edges}
        # All referenced rows are fetched before any workset writes.
        decoded = {eid: self._unpack(payloads[eid]) for eid in missing}
        for eid, slot in zip(missing, victims):
            fields = decoded[eid]
            next_version = max(int(m.read_credit_version[slot]), int(fields['read_credit_version']))+1
            for name, target in self.fields.items():
                target[slot].copy_(torch.as_tensor(fields[name], device=target.device, dtype=target.dtype))
            m.read_credit_version[slot] = next_version
            m.last_touch[slot] = m.clock
            # Preserve the anchor's actual question/outcome. Nested question
            # slots remain legacy HOT addresses: their existing read ports
            # validate each stamp and reject unavailable parents. An expired
            # parent receipt must not invalidate the anchor's own observation.
            self.hot[str(slot)] = dict(id=eid, signature=self._signature(slot))
            chosen[eid] = slot
        bank.condition[index].copy_(torch.as_tensor(np.frombuffer(row[2], dtype=self.schema['source']['dtype']).copy(),
                                                    device=bank.condition.device, dtype=bank.condition.dtype))
        bank.identity[index], bank.generation[index], bank.version[index] = rid, row[0], row[1]
        if hasattr(bank, 'program_owner'):
            from scripts.ib_local.generative_relations import EXTRA_FIELDS
            metadata = self.db.execute('SELECT payload FROM relation_program WHERE relation_id=?', (rid,)).fetchone()
            if metadata is not None:
                for name, value in json.loads(metadata[0]).items():
                    field = getattr(bank, name)
                    field[index].copy_(torch.as_tensor(value, device=field.device, dtype=field.dtype))
            else:
                for name in EXTRA_FIELDS:
                    getattr(bank, name)[index].zero_()
                bank.state(index)
        bank.valid[index].zero_()
        for ordinal, eid, _ in edges:
            slot = chosen[eid]
            bank.slots[index, ordinal] = slot
            bank.stamps[index, ordinal] = m.last_write[slot]
            bank.valid[index, ordinal] = True
        bank.last_use[index] = m.clock
        bank.next_identity.fill_(max(int(bank.next_identity),
            self.db.execute('SELECT COALESCE(MAX(id),0)+1 FROM relations').fetchone()[0]))
        self.counters['loaded_episodes'] += len(missing)
        self.counters['warmed_relations'] += 1
        return True

    @torch.no_grad()
    def prepare(self):
        """One decision boundary: archive prior facts, then recall a relation."""
        from scripts.ib_local.language_action import peek
        self.sync()
        engine, m = self.engine, self.engine.model.memory
        engine.decision.start(engine.recurrent)
        _, h = engine.decision.step(engine.pending, peek(engine.table, engine.pending))
        source = h.detach().reshape(1, -1)
        best = self.best_relation(source)
        if best is not None and best[0] >= m.threshold:
            self.warm_relation(best[1])

    def before_slot_write(self, slot):
        if not bool(self.engine.model.memory.occupied[slot]):
            return
        arrays = {k: v.detach().cpu().contiguous().numpy() for k, v in self.fields.items()}
        old_hot = json.loads(json.dumps(self.hot))
        try:
            with self.db:
                self._episode(slot, arrays)
                self._set_meta('hot', self.hot)
        except BaseException:
            self.hot = old_hot
            raise

    def checkpoint(self, output):
        self.sync()
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / ('causal_archive_'+uuid.uuid4().hex+'.tmp.sqlite')
        target = sqlite3.connect(temporary)
        try:
            self.db.backup(target)
        finally:
            target.close()
        digest = sha256(temporary)
        destination = output / ('causal_archive_'+digest+'.sqlite')
        if destination.exists():
            temporary.unlink()
        else:
            temporary.replace(destination)
        return dict(schema=1, snapshot=str(destination), sha256=digest,
                    budgets=self.budgets, role='unified episodes, utterance attachments, versioned relations')

    def stats(self):
        return dict(episodes=self.db.execute('SELECT COUNT(*) FROM episodes').fetchone()[0],
                    relations=self.db.execute('SELECT COUNT(*) FROM relations').fetchone()[0],
                    references=self.db.execute('SELECT COUNT(*) FROM evidence').fetchone()[0],
                    disk_bytes=self.path.stat().st_size, record_bytes=self.record_bytes,
                    budgets=self.budgets, **self.counters)


def attach_archive(engine, directory, *, manifest=None, **budgets):
    if not hasattr(engine.model, 'causal_relations'):
        raise ValueError('archive requires the existing causal-program body')
    archive = CausalArchive(engine, directory, manifest=manifest, **budgets)
    # Keep runtime handles outside nn.Module/state_dict and engine snapshots.
    object.__setattr__(engine.model, '_causal_archive', archive)
    object.__setattr__(engine.model.memory, '_causal_archive', archive)
    return archive
