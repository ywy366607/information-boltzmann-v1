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
    buffer = bytearray(65536)
    with Path(path).open('rb', buffering=0) as stream:
        while (size := stream.readinto(buffer)):
            digest.update(memoryview(buffer)[:size])
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
        self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.execute('PRAGMA journal_mode=WAL')
        self._episode_payload_cache = {}
        self._unknown_evidence_cache = {}
        self._retention_cache = {}
        self._relation_vars_cache = {}
        self._best_relation_cache = None
        self._hot_meta_cache = None
        self._synced_field_versions = None
        self._synced_bank_versions = None
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

    def _invalidate_sync_cache(self):
        self._synced_arrays = None
        self._synced_relations = None
        self._synced_field_versions = None
        self._synced_bank_versions = None
        self._hot_by_id = None
        self._hot_meta_cache = None

    def hot_slot_for(self, eid):
        """Return the first live hot slot for `eid`, matching archive.hot order."""
        m = self.engine.model.memory
        hot_by_id = getattr(self, '_hot_by_id', None)
        if hot_by_id is None:
            hot_by_id = {}
            for k, v in self.hot.items():
                hot_by_id.setdefault(v['id'], []).append(int(k))
            self._hot_by_id = hot_by_id
        slots = hot_by_id.get(eid)
        if not slots:
            return -1
        ver_key = (m.occupied._version, m.last_write._version,
                   m.read_credit_version._version, m.birth._version, m.count._version)
        meta_cache = getattr(self, '_hot_meta_cache', None)
        if meta_cache is not None and meta_cache[0] == ver_key:
            snap = meta_cache[1]
        else:
            snap = (
                m.occupied.detach().cpu().tolist(),
                m.last_write.detach().cpu().tolist(),
                m.read_credit_version.detach().cpu().tolist(),
                m.birth.detach().cpu().tolist(),
                m.count.detach().cpu().tolist(),
            )
            self._hot_meta_cache = (ver_key, snap)
        occ, lw, ver, birth, cnt = snap
        for slot in slots:
            entry = self.hot.get(str(slot))
            if (entry is not None and entry['id'] == eid and bool(occ[slot]) and
                    [lw[slot], ver[slot], birth[slot], cnt[slot]] == entry['signature']):
                return slot
        return -1

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

    def _episode(self, slot, arrays, packed=None, packed_offset=0):
        m = self.engine.model.memory
        # sync() already copied these fields once. Per-slot int(CUDA tensor)
        # previously issued four device synchronizations for every hot record.
        signature = [int(arrays[k][slot]) for k in
                     ('last_write','read_credit_version','birth','count')]
        old = self.hot.get(str(slot))
        payload = self._pack(slot, arrays) if packed is None else packed[slot-packed_offset].tobytes()
        touched=int(arrays['last_touch'][slot])
        import hashlib
        digest=hashlib.blake2b(payload,digest_size=16).digest()
        cache=getattr(self,'_payload_cache',None)
        if cache is None:self._payload_cache=cache={}
        if old is not None and old['signature'] == signature:
            eid = old['id']
            if cache.get(slot)==(eid,digest,touched):
                return eid
            self.db.execute('UPDATE episodes SET payload=?,touched=? WHERE id=?',(payload,touched,eid))
            self._episode_payload_cache.pop(eid, None)
        else:
            self._room('episodes')
            cursor = self.db.execute('INSERT INTO episodes(origin_slot,stamp,payload,touched) VALUES (?,?,?,?)',
                (slot, int(arrays['last_write'][slot]), payload, touched))
            eid = cursor.lastrowid
            self.hot[str(slot)] = dict(id=eid, signature=signature)
            self._hot_by_id = None
            self._episode_payload_cache.pop(eid, None)
        cache[slot]=(eid,digest,touched)
        return eid

    def _room(self, table):
        if self.db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] < self.budgets[table]:
            return
        if table == 'episodes':
            protected = [v['id'] for v in self.hot.values()]
            self.db.execute('CREATE TEMP TABLE IF NOT EXISTS pinned(id INTEGER PRIMARY KEY)')
            self.db.execute('DELETE FROM pinned')
            self.db.executemany('INSERT OR IGNORE INTO pinned VALUES (?)', ((i,) for i in protected))
            query_refs = ''
            if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='unknown_bindings'").fetchone():
                query_refs = (' AND id NOT IN (SELECT episode_id FROM unknown_evidence)'
                    ' AND id NOT IN (SELECT anchor_episode FROM unknown_bindings WHERE anchor_episode IS NOT NULL)')
            row = self.db.execute('''SELECT id FROM episodes WHERE id NOT IN (SELECT id FROM pinned)
                AND id NOT IN (SELECT episode_id FROM evidence)'''+query_refs+
                ' ORDER BY touched,id LIMIT 1').fetchone()
        else:
            live = self.engine.model.causal_relations.identity.detach().cpu().tolist()
            placeholders = ','.join('?' for _ in live)
            row = self.db.execute('SELECT id FROM relations WHERE id NOT IN ('+placeholders+
                ') ORDER BY touched,id LIMIT 1', live).fetchone()
        if row is None:
            raise RuntimeError('archive budget full: all release candidates are referenced or hot')
        self.db.execute('DELETE FROM '+table+' WHERE id=?', row)
        if table == 'episodes':
            self._episode_payload_cache.pop(row[0], None)
        else:
            self._best_relation_cache = None
            self._relation_vars_cache.clear()
            self._retention_cache.clear()

    @torch.no_grad()
    def sync(self, *, bulk=True, pack_rows=256):
        """Persist one committed boundary, with shared payloads and FK evidence."""
        m, bank = self.engine.model.memory, self.engine.model.causal_relations
        names = ('identity', 'version', 'generation', 'condition', 'last_use', 'valid', 'slots', 'stamps')
        if hasattr(bank, 'program_owner'):
            from scripts.ib_local.generative_relations import EXTRA_FIELDS
            program_names = EXTRA_FIELDS+(('support_episode_ids',) if hasattr(bank,'support_episode_ids') else ())
            names = tuple(dict.fromkeys((*names, *program_names)))
        cache = getattr(self, '_payload_cache', None)
        prev_arrays = getattr(self, '_synced_arrays', None)
        prev_rels = getattr(self, '_synced_relations', None)
        if not cache:
            prev_rels = None
        cur_field_versions = {k: v._version for k, v in self.fields.items()}
        cur_bank_versions = {name: getattr(bank, name)._version for name in names}
        prev_fver = getattr(self, '_synced_field_versions', None)
        prev_bver = getattr(self, '_synced_bank_versions', None)
        if (bulk and prev_arrays is not None and cache and getattr(self, '_synced_hot_clean', False)
                and prev_fver is not None and prev_bver is not None):
            changed_fields = [k for k, ver in cur_field_versions.items() if prev_fver.get(k) != ver]
            changed_bank = [name for name, ver in cur_bank_versions.items() if prev_bver.get(name) != ver]
            if not changed_fields and not changed_bank:
                return
        else:
            changed_fields = None
            changed_bank = None
        if changed_fields is not None and prev_arrays is not None and cache:
            arrays = dict(prev_arrays)
            for k in changed_fields:
                arrays[k] = self.fields[k].detach().cpu().contiguous().numpy()
        else:
            arrays = {k: v.detach().cpu().contiguous().numpy() for k, v in self.fields.items()}
        # One copy per whole field, rather than per relation/slot/scalar. The
        # event is paused for this synchronous transaction; no future state is
        # read and the on-disk payload/order remains byte-for-byte compatible.
        has_program = hasattr(bank, 'program_owner')
        has_support_ids = hasattr(bank, 'support_episode_ids')
        extra_names = ()
        if has_program:
            from scripts.ib_local.generative_relations import EXTRA_FIELDS
            extra_names = EXTRA_FIELDS+(('support_episode_ids',) if has_support_ids else ())
        # A bounded structured buffer packs fields in their original byte order.
        # Keep the same transaction and ascending slot order across blocks.
        record_type=np.dtype([(k,np.dtype(spec['dtype']),tuple(spec['shape']))
                               for k,spec,_,_ in self.layouts],align=False)
        if pack_rows < 1:
            raise ValueError('pack_rows must be positive')
        occupied_mask = arrays['occupied'].astype(bool)
        if prev_arrays is not None and cache:
            dirty = occupied_mask & (~prev_arrays['occupied'].astype(bool))
            check_keys = changed_fields if changed_fields is not None else list(arrays.keys())
            for k in check_keys:
                arr = arrays[k]
                prev = prev_arrays.get(k)
                if prev is None or prev.shape != arr.shape or prev.dtype != arr.dtype:
                    dirty = occupied_mask.copy()
                    break
                a_bytes = arr.view(np.uint8).reshape(len(arr), -1)
                p_bytes = prev.view(np.uint8).reshape(len(prev), -1)
                dirty |= occupied_mask & np.any(a_bytes != p_bytes, axis=1)
        else:
            dirty = occupied_mask
        dirty_slots = np.flatnonzero(dirty)
        if changed_bank is not None and not changed_bank and prev_rels is not None and has_support_ids and bulk:
            rel_updates = []
            new_rel_cache = prev_rels
            host = {}
            field = lambda name: getattr(bank, name)
        else:
            host = {name:getattr(bank, name).detach().cpu() for name in names} if bulk else {}
            field = lambda name: host[name] if bulk else getattr(bank, name)
            live_indices = torch.nonzero(field('identity') > 0).flatten().tolist()
            rel_updates = []
            new_rel_cache = {} if (has_support_ids and bulk) else None
            for i in live_indices:
                rid, version = int(field('identity')[i]), int(field('version')[i])
                gen = int(field('generation')[i])
                cond_bytes = field('condition')[i].detach().cpu().numpy().tobytes()
                touched = int(field('last_use')[i])
                if new_rel_cache is not None:
                    extra_bytes = tuple(field(name)[i].numpy().tobytes() for name in extra_names) if has_program else ()
                    ev_ids = tuple(int(eid) for eid in field('support_episode_ids')[i].tolist() if eid > 0)
                    rel_sig = (gen, version, cond_bytes, touched, extra_bytes, ev_ids)
                    new_rel_cache[rid] = rel_sig
                    if prev_rels is not None and prev_rels.get(rid) == rel_sig:
                        continue
                    rel_updates.append((i, rid, gen, version, cond_bytes, touched, ev_ids))
                else:
                    rel_updates.append((i, rid, gen, version, cond_bytes, touched, None))
        if not len(dirty_slots) and not rel_updates and getattr(self, '_synced_hot_clean', False):
            self._synced_arrays = {k: (arr.copy() if self.fields[k].device.type == 'cpu' else arr)
                                   for k, arr in arrays.items()}
            self._synced_relations = new_rel_cache
            self._synced_field_versions = cur_field_versions
            self._synced_bank_versions = cur_bank_versions
            return
        block_rows = min(m.capacity, pack_rows)
        records = np.empty(block_rows, dtype=record_type) if len(dirty_slots) else None
        packed = records.view(np.uint8).reshape(block_rows, self.record_bytes) if records is not None else None
        old_hot = json.loads(json.dumps(self.hot))
        hot_before_len = len(self.hot)
        try:
            with self.db:
                if len(dirty_slots):
                    for pos in range(0, len(dirty_slots), block_rows):
                        chunk = dirty_slots[pos:pos+block_rows]
                        count = len(chunk)
                        for k in arrays:
                            records[k][:count] = arrays[k][chunk]
                        for local_idx, slot in enumerate(chunk.tolist()):
                            self._episode(slot, arrays, packed, slot - local_idx)
                for i, rid, gen, version, cond_bytes, touched, ev_ids in rel_updates:
                    old = self.db.execute('SELECT version FROM relations WHERE id=?', (rid,)).fetchone()
                    if old is None:
                        self._room('relations')
                        self.db.execute('INSERT INTO relations VALUES (?,?,?,?,?)',
                            (rid, gen, version, cond_bytes, touched))
                    else:
                        self.db.execute('UPDATE relations SET version=?,condition=?,touched=? WHERE id=?',
                            (version, cond_bytes, touched, rid))
                    if has_program:
                        payload = {name: field(name)[i].detach().cpu().tolist()
                                   for name in extra_names}
                        self.db.execute('INSERT OR REPLACE INTO relation_program VALUES (?,?)',
                                        (rid, json.dumps(payload)))
                    # If a physical slot disappeared without a logical revision,
                    # keep the disk edge. Physical eviction is not a refutation.
                    if old is None or old[0] != version:
                        self.db.execute('DELETE FROM evidence WHERE relation_id=?', (rid,))
                    if has_support_ids:
                        self.db.execute('DELETE FROM evidence WHERE relation_id=?',(rid,))
                        ids = ev_ids if ev_ids is not None else [int(eid) for eid in field('support_episode_ids')[i].tolist() if eid>0]
                        self.db.executemany('INSERT INTO evidence VALUES (?,?,?)',
                                            ((rid,j,eid) for j,eid in enumerate(ids)))
                        continue
                    for j in torch.nonzero(field('valid')[i]).flatten().tolist():
                        slot, stamp = int(field('slots')[i, j]), int(field('stamps')[i, j])
                        if (bool(arrays['occupied'][slot]) and int(arrays['count'][slot]) == 1 and
                            int(arrays['last_write'][slot]) == stamp and bool(arrays['mother_valid'][slot]) and
                            int(arrays['mother_stamp'][slot]) == stamp):
                            self.db.execute('INSERT OR REPLACE INTO evidence VALUES (?,?,?)',
                                            (rid, j, self.hot[str(slot)]['id']))
                if not getattr(self, '_synced_hot_clean', False) or len(self.hot) != hot_before_len or self.hot != old_hot:
                    self._set_meta('hot', self.hot)
            if rel_updates:
                self._best_relation_cache = None
                rc = getattr(self, '_best_rel_row_cache', None)
                if rc is not None:
                    for _, rid, *_ in rel_updates:
                        rc.pop(rid, None)
            self._synced_arrays = {k: (arr.copy() if self.fields[k].device.type == 'cpu' else arr)
                                   for k, arr in arrays.items()}
            self._synced_relations = new_rel_cache
            self._synced_field_versions = cur_field_versions
            self._synced_bank_versions = cur_bank_versions
            self._synced_hot_clean = True
        except BaseException:
            self.hot = old_hot
            self._payload_cache={}
            self._invalidate_sync_cache()
            self._best_relation_cache = None
            self._synced_hot_clean = False
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
        row_cache = getattr(self, '_best_rel_row_cache', None)
        if row_cache is None:
            self._best_rel_row_cache = row_cache = {}
        cache_key = (scoped, chunk_size, source.dtype)
        cached_chunks = getattr(self, '_best_relation_cache', None)
        if cached_chunks is None or cached_chunks[0] != cache_key:
            chunks = []
            cursor = self.db.execute('''SELECT r.id,r.version,r.condition,p.payload FROM relations r
                LEFT JOIN relation_program p ON p.relation_id=r.id ORDER BY r.id'''
                if scoped else 'SELECT id,version,condition FROM relations ORDER BY id')
            src_dtype = self.schema['source']['dtype']
            raw_dtype = source.detach().cpu().numpy().dtype
            seen_rids = set()
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                rids = []
                xs = []
                scopes = []
                for j, row in enumerate(rows):
                    rid, ver, cond_blob = row[0], row[1], row[2]
                    seen_rids.add(rid)
                    rids.append(rid)
                    entry = row_cache.get(rid)
                    if entry is None or entry[0] != ver or entry[1] != (scoped, raw_dtype):
                        x_row = np.frombuffer(cond_blob, dtype=src_dtype).copy()
                        sc = None
                        if scoped and row[3] is not None:
                            metadata = json.loads(row[3])
                            valid = np.asarray(metadata['scope_valid'], dtype=bool)
                            if valid.any():
                                sc = (
                                    np.asarray(metadata['scope_axes'], dtype=raw_dtype)[valid],
                                    np.asarray(metadata['scope_bounds'], dtype=raw_dtype)[valid],
                                    np.asarray(metadata['scope_signs'], dtype=raw_dtype)[valid],
                                )
                        entry = (ver, (scoped, raw_dtype), x_row, sc)
                        row_cache[rid] = entry
                    xs.append(entry[2])
                    if entry[3] is not None:
                        scopes.append((j, *entry[3]))
                chunks.append((rids, np.stack(xs), scopes))
            for stale in [k for k in row_cache if k not in seen_rids]:
                del row_cache[stale]
            self._best_relation_cache = (cache_key, chunks)
        else:
            chunks = cached_chunks[1]
        raw_source = source[0].detach().cpu().numpy() if scoped else None
        eps = np.finfo(raw_source.dtype).eps if scoped else None
        for rids, x, scopes in chunks:
            keys = torch.as_tensor(x, device=source.device, dtype=source.dtype)
            scores = (query @ F.normalize(model.encoder(keys), dim=-1).T)[0]
            if scopes:
                rejected = []
                for j, axes, bounds, signs in scopes:
                    projection = axes @ raw_source
                    tolerance = eps*(1+abs(projection)+abs(bounds))*8
                    if (signs*(projection-bounds) > tolerance).any():
                        rejected.append(j)
                if rejected:
                    scores[rejected] = -torch.inf
            if not bool(torch.isfinite(scores).any()):
                continue
            value, at = scores.max(0)
            item = (float(value), rids[int(at)])
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
        bank_id = bank.identity.detach().cpu().tolist()
        bank_use = bank.last_use.detach().cpu().tolist()
        live = [i for i, v in enumerate(bank_id) if v == rid]
        candidates = [i for i in range(bank.capacity) if bank_id[i] not in protected_relations]
        if not live and not candidates:
            return False
        index = live[0] if live else min(candidates, key=lambda i: (bank_id[i] > 0, bank_use[i], i))
        lw = m.last_write.detach().cpu().tolist()
        ver = m.read_credit_version.detach().cpu().tolist()
        birth = m.birth.detach().cpu().tolist()
        cnt = m.count.detach().cpu().tolist()
        occ = m.occupied.detach().cpu().tolist()
        touch = m.last_touch.detach().cpu().tolist()
        mapping = {v['id']: int(k) for k, v in self.hot.items()
                   if [lw[int(k)], ver[int(k)], birth[int(k)], cnt[int(k)]] == v['signature']}
        chosen = {eid: mapping[eid] for _, eid, _ in edges if eid in mapping}
        pinned.update(chosen.values())
        victims = sorted((i for i in range(m.capacity) if i not in pinned),
                         key=lambda i: (bool(occ[i]), touch[i], i))
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
        if hasattr(bank,'support_episode_ids'):
            bank.support_episode_ids[index].zero_()
        for ordinal, eid, _ in edges:
            slot = chosen[eid]
            bank.slots[index, ordinal] = slot
            bank.stamps[index, ordinal] = m.last_write[slot]
            bank.valid[index, ordinal] = True
            if hasattr(bank,'support_episode_ids'):bank.support_episode_ids[index,ordinal]=eid
        bank.last_use[index] = m.clock
        bank.next_identity.fill_(max(int(bank.next_identity),
            self.db.execute('SELECT COALESCE(MAX(id),0)+1 FROM relations').fetchone()[0]))
        self.counters['loaded_episodes'] += len(missing)
        self.counters['warmed_relations'] += 1
        if self._synced_relations is not None:
            self._synced_relations.pop(rid, None)
        if missing:
            self._hot_by_id = None
            self._hot_meta_cache = None
            self._synced_hot_clean = False
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
        slot_arrays = {k: v[slot:slot+1].detach().cpu().contiguous().numpy()
                       for k, v in self.fields.items()}
        payload = b''.join(slot_arrays[k][0].tobytes() for k, _, _, _ in self.layouts)
        signature = [int(slot_arrays[k][0]) for k in
                     ('last_write', 'read_credit_version', 'birth', 'count')]
        touched = int(slot_arrays['last_touch'][0])
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        old_hot = json.loads(json.dumps(self.hot))
        try:
            cache = getattr(self, '_payload_cache', None)
            if cache is None:
                self._payload_cache = cache = {}
            old = self.hot.get(str(slot))
            with self.db:
                if old is not None and old['signature'] == signature:
                    eid = old['id']
                    if cache.get(slot) != (eid, digest, touched):
                        self.db.execute('UPDATE episodes SET payload=?,touched=? WHERE id=?',
                                        (payload, touched, eid))
                        self._episode_payload_cache.pop(eid, None)
                else:
                    self._room('episodes')
                    cursor = self.db.execute(
                        'INSERT INTO episodes(origin_slot,stamp,payload,touched) VALUES (?,?,?,?)',
                        (slot, int(slot_arrays['last_write'][0]), payload, touched))
                    eid = cursor.lastrowid
                    self.hot[str(slot)] = dict(id=eid, signature=signature)
                    self._hot_by_id = None
                    self._episode_payload_cache.pop(eid, None)
                cache[slot] = (eid, digest, touched)
                self._set_meta('hot', self.hot)
            if getattr(self, '_synced_arrays', None) is not None:
                for k, arr in slot_arrays.items():
                    if k in self._synced_arrays:
                        self._synced_arrays[k][slot] = arr[0]
            self._synced_hot_clean = True
        except BaseException:
            self.hot = old_hot
            self._invalidate_sync_cache()
            self._synced_hot_clean = False
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
    object.__setattr__(engine.model.causal_relations,'_causal_archive',archive)
    return archive


def grow_utterance_attachment(archive):
    """Repack only the exact payload capacity, preserving every fact/edge ID.

    Runs on the private working database inside one transaction. The immutable
    source snapshot is untouched. Bounded batches avoid holding the archive in
    RAM; the causal feature schema itself is unchanged by body growth.
    """
    fields=columns(archive.engine)
    schema={k:dict(shape=list(v.shape[1:]),dtype=v.detach().cpu().numpy().dtype.str)
            for k,v in fields.items()}
    old=archive.schema
    differences=[k for k in schema if schema[k]!=old.get(k)]
    if not differences:return dict(repacked=0)
    if differences!=['@words'] or old['@words']['shape']!=[32] or schema['@words']['shape']!=[128]:
        raise ValueError('only declared 32 -> 128 retrieval attachment growth is allowed')
    layouts=[];offset=0
    for k in sorted(schema):
        spec=schema[k];size=int(np.prod(spec['shape']) if spec['shape'] else 1)*np.dtype(spec['dtype']).itemsize
        layouts.append((k,spec,offset,size));offset+=size
    count=0
    with archive.db:
        cursor=archive.db.execute('SELECT id,payload FROM episodes ORDER BY id')
        while True:
            rows=cursor.fetchmany(64)
            if not rows:break
            for identity,payload in rows:
                record=archive._unpack(payload)
                words=np.zeros(128,dtype=record['@words'].dtype)
                words[:32]=record['@words'];record['@words']=words
                packed=b''.join(record[k].tobytes() for k,_,_,_ in layouts)
                archive.db.execute('UPDATE episodes SET payload=? WHERE id=?',(packed,identity))
                count+=1
        archive._set_meta('schema',schema)
    archive.schema,archive.layouts,archive.fields=schema,layouts,fields
    archive.record_bytes=offset
    archive._episode_payload_cache.clear()
    archive._invalidate_sync_cache()
    return dict(repacked=count,old_attachment_bytes=32,new_attachment_bytes=128,
                episode_and_relation_ids_preserved=True)
