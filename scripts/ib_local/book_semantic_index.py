"""Hybrid Semantic + E-Means + IDF Book Page Retrieval Engine (`HybridBookIndex`).

Upgrades MengLi's `Hom(MengLi, Book)` navigation from brittle raw byte/1-char matching
to a two-stage, zero-GPU-VRAM hybrid book shelf:
  1. Offline page chunking (~2-4 KB UTF-8 sentence-aligned pages) encoded once with
     Google's Apache-2.0 multimodal-ready `EmbeddingGemma-2` (256-dim Matryoshka,
     quantized `int8` unit vectors on disk -> ~2 MB per 16 MB book).
  2. Balanced spherical e-means clustering (`k` topic centroids) for sub-millisecond
     hierarchical semantic candidate routing (`top_clusters` -> dense rerank).
  3. IDF-weighted sliding lexical scorer (2-4 char CJK n-grams & 3+ char English words;
     zero weight on single characters and high-frequency stopwords) both fused into
     page ranking AND used to pinpoint the exact sentence offset inside the winning page.
  4. Strict relevance gate: single-character noise or garbled fragments (`:往，滑`,
     `g gr d-`) fall below threshold and return `None` (`match=0`) so the book continues
     calmly instead of jumping to a random unrelated page.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_GEMMA_ONNX_DIR = Path(r'E:\ib_models\embeddinggemma-2-onnx')

_CJK_RUN_RE = re.compile(r'[\u4e00-\u9fff]+')
_EN_WORD_RE = re.compile(r'[a-zA-Z]{3,}')

_EN_STOPWORDS = frozenset({
    'the', 'and', 'for', 'that', 'with', 'this', 'from', 'have', 'are', 'was',
    'were', 'not', 'but', 'you', 'your', 'they', 'them', 'their', 'his', 'her',
    'she', 'him', 'its', 'our', 'out', 'all', 'one', 'two', 'can', 'will',
    'just', 'about', 'into', 'over', 'some', 'there', 'when', 'what', 'which',
    'who', 'how', 'more', 'very', 'also', 'been', 'had', 'has', 'did', 'does',
    'than', 'then', 'would', 'could', 'should', 'other', 'only', 'now', 'like',
})

_ZH_STOP_BIGRAMS = frozenset({
    '一个', '这个', '那个', '他们', '她们', '我们', '你们', '自己', '什么', '怎么',
    '因为', '所以', '但是', '如果', '已经', '可以', '没有', '不是', '就是', '还是',
    '非常', '一起', '时候', '说道',
})


def extract_lexical_terms(text: str) -> list[str]:
    """Extract informative 2-4 char CJK n-grams and 3+ char English words (NEVER 1-char noise)."""
    if not text:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for run in _CJK_RUN_RE.findall(text):
        L = len(run)
        if L < 2:
            continue
        # Slide over ALL 2, 3, 4-char CJK windows (so `场去公园里` extracts `公园里`, `去公园`, `公园`!)
        for n in (3, 2, 4):
            if L < n:
                continue
            for i in range(L - n + 1):
                g = run[i:i + n]
                if n == 2 and g in _ZH_STOP_BIGRAMS:
                    continue
                if g not in seen:
                    seen.add(g)
                    terms.append(g)
    for w in _EN_WORD_RE.findall(text.lower()):
        if w in _EN_STOPWORDS:
            continue
        if w not in seen:
            seen.add(w)
            terms.append(w)
    return terms


def chunk_utf8_corpus(raw_bytes: bytes, *, chunk_bytes: int = 2560, overlap_bytes: int = 256) -> list[tuple[int, int]]:
    """Split a UTF-8 corpus into clean character- & sentence-aligned `(start, end)` byte windows."""
    n = len(raw_bytes)
    if n == 0:
        return []
    step = max(256, int(chunk_bytes) - int(overlap_bytes))
    chunks: list[tuple[int, int]] = []
    pos = 0

    def _align_start(idx: int) -> int:
        while idx < n and (raw_bytes[idx] & 0xC0) == 0x80:
            idx += 1
        return idx

    def _align_end(idx: int) -> int:
        idx = min(n, idx)
        while idx > 0 and idx < n and (raw_bytes[idx] & 0xC0) == 0x80:
            idx -= 1
        return idx

    while pos < n:
        s = _align_start(pos)
        if s >= n:
            break
        # Nudge start forward to a clean newline/period if close by (except at byte 0).
        if s > 0:
            win = raw_bytes[s:min(n, s + 96)]
            nl = win.find(b'\n')
            dot = win.find('。'.encode('utf-8'))
            if nl >= 0 and nl < 64:
                s = _align_start(s + nl + 1)
            elif dot >= 0 and dot < 64:
                s = _align_start(s + dot + 3)
        e = _align_end(min(n, s + int(chunk_bytes)))
        if e <= s:
            break
        chunks.append((s, e))
        if e >= n:
            break
        pos = s + step
    return chunks


def fit_emeans_spherical(
    vectors: np.ndarray,
    *,
    n_clusters: int,
    n_iters: int = 12,
    balance_strength: float = 0.08,
    seed: int = 20261125,
) -> tuple[np.ndarray, np.ndarray]:
    """Balanced spherical e-means clustering on unit vectors.

    Minimizes cosine distortion with an energy-balance log-size penalty so clusters
    stay uniformly populated and sub-millisecond coarse routing never degenerates.
    Returns `(centroids [K, D] float32, assignments [N] int32)`.
    """
    x = np.asarray(vectors, dtype=np.float32)
    n, d = x.shape
    k = max(1, min(int(n_clusters), n))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-8)

    rng = np.random.default_rng(seed)
    # K-means++ style deterministic-seed initialization on the unit sphere.
    init_idx = [int(rng.integers(0, n))]
    min_dist = np.full(n, np.inf, dtype=np.float32)
    for _ in range(1, k):
        sims = x @ x[init_idx[-1]]
        dist = np.maximum(0.0, 1.0 - sims)
        min_dist = np.minimum(min_dist, dist)
        probs = min_dist.astype(np.float64)
        s = float(probs.sum())
        if s <= 1e-12:
            pick = int(rng.integers(0, n))
        else:
            probs /= s
            pick = int(rng.choice(n, p=probs))
        init_idx.append(pick)
    centroids = x[init_idx].copy()

    target_size = max(1.0, float(n) / float(k))
    counts = np.full(k, target_size, dtype=np.float32)
    assign = np.zeros(n, dtype=np.int32)

    for _ in range(max(1, int(n_iters))):
        # Penalize overcrowded clusters softly via log-energy balance term.
        penalty = float(balance_strength) * np.log(np.maximum(counts, 1.0) / target_size)
        # Process in chunks to keep peak RAM tiny.
        new_assign = np.empty(n, dtype=np.int32)
        for start in range(0, n, 2048):
            end = min(n, start + 2048)
            scores = (x[start:end] @ centroids.T) - penalty[None, :]
            new_assign[start:end] = np.argmax(scores, axis=1).astype(np.int32)
        assign = new_assign
        counts = np.bincount(assign, minlength=k).astype(np.float32)
        new_c = np.zeros((k, d), dtype=np.float32)
        np.add.at(new_c, assign, x)
        empty = counts < 0.5
        if np.any(empty):
            refill = rng.integers(0, n, size=int(empty.sum()))
            new_c[empty] = x[refill]
            counts[empty] = 1.0
        c_norms = np.linalg.norm(new_c, axis=1, keepdims=True)
        centroids = new_c / np.maximum(c_norms, 1e-8)

    return centroids.astype(np.float32), assign


class EmbeddingGemma2Onnx:
    """CPU-only EmbeddingGemma-2 Q4 ONNX encoder (256-dim Matryoshka, zero GPU VRAM)."""

    _shared_cache: dict[tuple[str, int], 'EmbeddingGemma2Onnx'] = {}

    def __init__(self, model_dir: str | Path = DEFAULT_GEMMA_ONNX_DIR, *, dim: int = 256, threads: int = 6):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = Path(model_dir).resolve()
        self.dim = int(dim)
        onnx_path = self.model_dir / 'onnx' / 'model_q4.onnx'
        tok_path = self.model_dir / 'tokenizer.json'
        if not onnx_path.exists() or not tok_path.exists():
            raise FileNotFoundError(f'EmbeddingGemma-2 ONNX files missing in {self.model_dir}')

        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(max_length=96)
        self.tokenizer.enable_padding(pad_id=0, pad_token='<pad>')

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=['CPUExecutionProvider'])
        self._empty_modal = np.zeros((0, 512), dtype=np.float32)

    @classmethod
    def get_shared(cls, model_dir: str | Path = DEFAULT_GEMMA_ONNX_DIR, *, dim: int = 256) -> 'EmbeddingGemma2Onnx':
        key = (str(Path(model_dir).resolve()), int(dim))
        inst = cls._shared_cache.get(key)
        if inst is None:
            inst = cls(model_dir, dim=dim)
            cls._shared_cache[key] = inst
        return inst

    def encode_raw(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out_chunks: list[np.ndarray] = []
        for start in range(0, len(texts), max(1, int(batch_size))):
            batch = texts[start:start + int(batch_size)]
            encs = self.tokenizer.encode_batch(batch)
            feed = dict(
                input_ids=np.array([e.ids for e in encs], dtype=np.int64),
                attention_mask=np.array([e.attention_mask for e in encs], dtype=np.int64),
                image_features=self._empty_modal,
                video_features=self._empty_modal,
                audio_features=self._empty_modal,
            )
            emb = self.session.run(['sentence_embedding'], feed)[0][:, :self.dim]
            norms = np.linalg.norm(emb, axis=-1, keepdims=True)
            out_chunks.append((emb / np.maximum(norms, 1e-8)).astype(np.float32, copy=False))
        return np.concatenate(out_chunks, axis=0)

    def encode_queries(self, queries: list[str], *, batch_size: int = 16) -> np.ndarray:
        prefixed = [f'task: search result | query: {q.strip()}' for q in queries]
        return self.encode_raw(prefixed, batch_size=batch_size)

    def encode_passages(self, passages: list[str], *, batch_size: int = 32) -> np.ndarray:
        prefixed = [f'title: none | text: {p.strip()}' for p in passages]
        return self.encode_raw(prefixed, batch_size=batch_size)


class HybridBookIndex:
    """Compact on-disk + in-RAM hybrid IDF + EmbeddingGemma-2 + e-means book retriever."""

    def __init__(
        self,
        *,
        corpus_bytes: bytes,
        chunk_starts: np.ndarray,
        chunk_ends: np.ndarray,
        vectors_i8: np.ndarray | None = None,
        centroids: np.ndarray | None = None,
        cluster_ids: np.ndarray | None = None,
        idf_table: dict[str, float] | None = None,
        postings: dict[str, np.ndarray] | None = None,
        encoder_dir: str | Path | None = None,
        dim: int = 256,
        corpus_sha256: str = '',
    ):
        self.corpus_bytes = corpus_bytes
        self.chunk_starts = np.asarray(chunk_starts, dtype=np.int64)
        self.chunk_ends = np.asarray(chunk_ends, dtype=np.int64)
        self.n_chunks = int(len(self.chunk_starts))
        self.vectors_i8 = None if vectors_i8 is None else np.asarray(vectors_i8, dtype=np.int8)
        self.centroids = None if centroids is None else np.asarray(centroids, dtype=np.float32)
        self.cluster_ids = None if cluster_ids is None else np.asarray(cluster_ids, dtype=np.int32)
        self.idf_table: dict[str, float] = idf_table or {}
        self.postings: dict[str, np.ndarray] = postings or {}
        self.encoder_dir = None if encoder_dir is None else Path(encoder_dir)
        self.dim = int(dim)
        self.corpus_sha256 = str(corpus_sha256)
        self._encoder: EmbeddingGemma2Onnx | None = None

        # Pre-index cluster member arrays for O(1) coarse routing.
        self._cluster_members: list[np.ndarray] = []
        if self.centroids is not None and self.cluster_ids is not None:
            k = int(len(self.centroids))
            order = np.argsort(self.cluster_ids, kind='mergesort')
            sorted_c = self.cluster_ids[order]
            splits = np.searchsorted(sorted_c, np.arange(k + 1))
            self._cluster_members = [order[splits[i]:splits[i + 1]] for i in range(k)]

    def _get_encoder(self) -> EmbeddingGemma2Onnx | None:
        if self.vectors_i8 is None or self.encoder_dir is None:
            return None
        if self._encoder is None:
            if (self.encoder_dir / 'onnx' / 'model_q4.onnx').exists():
                self._encoder = EmbeddingGemma2Onnx.get_shared(self.encoder_dir, dim=self.dim)
        return self._encoder

    @classmethod
    def build_from_corpus(
        cls,
        corpus_path: str | Path,
        *,
        out_prefix: str | Path | None = None,
        chunk_bytes: int = 2560,
        overlap_bytes: int = 256,
        n_clusters: int = 128,
        encoder_dir: str | Path | None = DEFAULT_GEMMA_ONNX_DIR,
        dim: int = 256,
        use_semantic: bool = True,
        batch_size: int = 32,
        progress: bool = False,
    ) -> 'HybridBookIndex':
        cpath = Path(corpus_path).resolve()
        raw = cpath.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        spans = chunk_utf8_corpus(raw, chunk_bytes=chunk_bytes, overlap_bytes=overlap_bytes)
        starts = np.array([s for s, _ in spans], dtype=np.int64)
        ends = np.array([e for _, e in spans], dtype=np.int64)
        n = len(spans)

        # Build compact IDF + inverted postings table (keeping informative mid/low-freq terms).
        df_counts: dict[str, list[int]] = {}
        texts: list[str] = []
        for idx, (s, e) in enumerate(spans):
            txt = raw[s:e].decode('utf-8', errors='ignore')
            texts.append(txt[:640])
            for term in extract_lexical_terms(txt):
                lst = df_counts.get(term)
                if lst is None:
                    df_counts[term] = [idx]
                else:
                    lst.append(idx)

        max_df = max(20, int(0.25 * n))
        idf_table: dict[str, float] = {}
        postings: dict[str, np.ndarray] = {}
        for term, doc_ids in df_counts.items():
            df = len(doc_ids)
            if df > max_df:
                continue
            idf = math.log((n + 1.0) / (df + 1.0)) + 1.0
            idf_table[term] = float(idf)
            postings[term] = np.asarray(doc_ids, dtype=np.int32)

        vectors_i8 = None
        centroids = None
        cluster_ids = None
        enc_dir_eff = None
        if use_semantic and encoder_dir is not None and (Path(encoder_dir) / 'onnx' / 'model_q4.onnx').exists():
            enc_dir_eff = Path(encoder_dir).resolve()
            enc = EmbeddingGemma2Onnx.get_shared(enc_dir_eff, dim=dim)
            all_vecs = np.empty((n, dim), dtype=np.float32)
            for s_i in range(0, n, max(1, int(batch_size))):
                e_i = min(n, s_i + int(batch_size))
                all_vecs[s_i:e_i] = enc.encode_passages(texts[s_i:e_i], batch_size=batch_size)
                if progress and ((s_i // batch_size) % 25 == 0 or e_i == n):
                    print(f'  [index:{cpath.parent.name}] encoded {e_i}/{n} pages', flush=True)
            # Quantize unit vectors to int8 ([-127, 127]) -> 4x smaller on disk & RAM, <0.002 cos error!
            vectors_i8 = np.clip(np.rint(all_vecs * 127.0), -127, 127).astype(np.int8)
            k_eff = max(1, min(int(n_clusters), max(1, n // 16)))
            centroids, cluster_ids = fit_emeans_spherical(all_vecs, n_clusters=k_eff)

        inst = cls(
            corpus_bytes=raw,
            chunk_starts=starts,
            chunk_ends=ends,
            vectors_i8=vectors_i8,
            centroids=centroids,
            cluster_ids=cluster_ids,
            idf_table=idf_table,
            postings=postings,
            encoder_dir=enc_dir_eff,
            dim=dim,
            corpus_sha256=sha,
        )
        if out_prefix is not None:
            inst.save(out_prefix)
        return inst

    def save(self, out_prefix: str | Path) -> Path:
        prefix = Path(out_prefix).resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        npz_path = prefix.with_suffix('.npz')
        meta_path = prefix.with_suffix('.json')

        # Flatten postings into contiguous CSR arrays for ultra-fast numpy load.
        terms_sorted = sorted(self.idf_table.keys())
        idf_arr = np.array([self.idf_table[t] for t in terms_sorted], dtype=np.float32)
        offsets = np.zeros(len(terms_sorted) + 1, dtype=np.int64)
        total_post = 0
        for i, t in enumerate(terms_sorted):
            total_post += len(self.postings[t])
            offsets[i + 1] = total_post
        flat_post = np.empty(total_post, dtype=np.int32)
        for i, t in enumerate(terms_sorted):
            flat_post[offsets[i]:offsets[i + 1]] = self.postings[t]

        arrays = dict(
            chunk_starts=self.chunk_starts,
            chunk_ends=self.chunk_ends,
            idf_arr=idf_arr,
            posting_offsets=offsets,
            posting_flat=flat_post,
        )
        if self.vectors_i8 is not None:
            arrays['vectors_i8'] = self.vectors_i8
        if self.centroids is not None:
            arrays['centroids'] = self.centroids
        if self.cluster_ids is not None:
            arrays['cluster_ids'] = self.cluster_ids
        np.savez(npz_path, **arrays)

        meta = dict(
            schema=1,
            corpus_sha256=self.corpus_sha256,
            n_chunks=self.n_chunks,
            dim=self.dim,
            has_semantic=(self.vectors_i8 is not None),
            n_clusters=(0 if self.centroids is None else int(len(self.centroids))),
            encoder_dir=(None if self.encoder_dir is None else str(self.encoder_dir)),
            terms=terms_sorted,
        )
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
        return npz_path

    @classmethod
    def load(
        cls,
        index_prefix: str | Path,
        corpus_bytes: bytes,
        *,
        encoder_dir: str | Path | None = DEFAULT_GEMMA_ONNX_DIR,
    ) -> 'HybridBookIndex':
        prefix = Path(index_prefix).resolve()
        npz_path = prefix.with_suffix('.npz')
        meta_path = prefix.with_suffix('.json')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        data = np.load(npz_path)

        terms = meta['terms']
        idf_arr = data['idf_arr']
        offsets = data['posting_offsets']
        flat = data['posting_flat']
        idf_table = {t: float(idf_arr[i]) for i, t in enumerate(terms)}
        postings = {t: flat[int(offsets[i]):int(offsets[i + 1])] for i, t in enumerate(terms)}

        enc_dir = encoder_dir if encoder_dir is not None else meta.get('encoder_dir')
        return cls(
            corpus_bytes=corpus_bytes,
            chunk_starts=data['chunk_starts'],
            chunk_ends=data['chunk_ends'],
            vectors_i8=(data['vectors_i8'] if 'vectors_i8' in data else None),
            centroids=(data['centroids'] if 'centroids' in data else None),
            cluster_ids=(data['cluster_ids'] if 'cluster_ids' in data else None),
            idf_table=idf_table,
            postings=postings,
            encoder_dir=enc_dir,
            dim=int(meta.get('dim', 256)),
            corpus_sha256=str(meta.get('corpus_sha256', '')),
        )

    def _pinpoint_sentence_offset(self, chunk_idx: int, matched_terms: list[str]) -> int:
        """Within a winning page chunk, pinpoint the clean sentence start containing the top term."""
        s = int(self.chunk_starts[chunk_idx])
        e = int(self.chunk_ends[chunk_idx])
        page = self.corpus_bytes[s:e]
        best_pos = -1
        for term in matched_terms[:5]:
            needle = term.encode('utf-8')
            pos = page.find(needle)
            if pos < 0:
                pos = page.lower().find(needle.lower())
            if pos >= 0:
                best_pos = pos
                break
        if best_pos <= 0:
            return s
        # Back up to nearest sentence boundary before `best_pos`.
        win_start = max(0, best_pos - 128)
        prefix = page[win_start:best_pos]
        nl = prefix.rfind(b'\n')
        cn_dot = prefix.rfind('。'.encode('utf-8'))
        en_dot = prefix.rfind(b'. ')
        cand = -1
        if nl >= 0:
            cand = max(cand, win_start + nl + 1)
        if cn_dot >= 0:
            cand = max(cand, win_start + cn_dot + 3)
        if en_dot >= 0:
            cand = max(cand, win_start + en_dot + 2)
        offset = s + (cand if cand >= 0 else best_pos)
        n = len(self.corpus_bytes)
        while offset < n and (self.corpus_bytes[offset] & 0xC0) == 0x80:
            offset += 1
        return offset

    def search(
        self,
        query_text: str,
        *,
        cursor: int = 0,
        top_clusters: int = 12,
        min_lexical_terms: int = 1,
        min_semantic_cos: float = 0.66,
        min_hybrid_score: float = 0.35,
    ) -> dict | None:
        """Search the book via IDF lexical matching + EmbeddingGemma-2 e-means semantic routing.

        Returns `None` when the query lacks informative content (single chars / garbled noise),
        preventing random unrelated page jumps.
        """
        q = str(query_text or '').strip()
        if not q:
            return None
        terms = extract_lexical_terms(q)
        # Keep only informative terms present in the corpus IDF table, ranked highest IDF first.
        scored_terms = sorted(
            [(self.idf_table[t], t) for t in terms if t in self.idf_table],
            key=lambda x: (x[0], len(x[1])),
            reverse=True,
        )

        # Strict noise gate: single-character / pure-gibberish queries with zero valid terms
        # and < 4 meaningful letters/CJK chars never trigger random jumps.
        cjk_count = sum(len(r) for r in _CJK_RUN_RE.findall(q))
        en_words = [w for w in _EN_WORD_RE.findall(q.lower()) if w not in _EN_STOPWORDS]
        if not scored_terms and cjk_count < 3 and len(en_words) < 2:
            return None

        # 1) IDF lexical scoring across postings (sub-millisecond).
        lex_scores: dict[int, float] = {}
        max_possible_idf = sum(idf for idf, _ in scored_terms[:8]) or 1.0
        for idf, term in scored_terms[:10]:
            for doc_id in self.postings[term]:
                did = int(doc_id)
                lex_scores[did] = lex_scores.get(did, 0.0) + idf

        # 2) E-means coarse semantic routing + EmbeddingGemma-2 dense rerank.
        sem_scores: dict[int, float] = {}
        enc = self._get_encoder()
        if enc is not None and self.vectors_i8 is not None and self.centroids is not None and (cjk_count >= 2 or len(en_words) >= 1):
            q_vec = enc.encode_queries([q[:160]])[0]  # [D] float32 unit vector
            c_sims = self.centroids @ q_vec
            k_pick = min(int(top_clusters), len(self.centroids))
            top_c = np.argpartition(c_sims, -k_pick)[-k_pick:]
            cand_sets = [self._cluster_members[int(cid)] for cid in top_c]
            if lex_scores:
                top_lex = sorted(lex_scores.keys(), key=lambda d: lex_scores[d], reverse=True)[:96]
                cand_sets.append(np.asarray(top_lex, dtype=np.int32))
            cands = np.unique(np.concatenate(cand_sets)) if cand_sets else np.empty(0, dtype=np.int32)
            if len(cands) > 0:
                vecs = self.vectors_i8[cands].astype(np.float32) * (1.0 / 127.0)
                sims = vecs @ q_vec
                for did, sc in zip(cands.tolist(), sims.tolist()):
                    sem_scores[int(did)] = float(sc)

        all_cands = set(lex_scores.keys()) | set(sem_scores.keys())
        if not all_cands:
            return None

        best_doc = -1
        best_score = -1.0
        best_lex_norm = 0.0
        best_sem = 0.0
        n_bytes = max(1, len(self.corpus_bytes))

        for did in all_cands:
            lx_raw = lex_scores.get(did, 0.0)
            lx_norm = min(1.0, lx_raw / max(max_possible_idf, 3.0))
            sm = sem_scores.get(did, 0.0)
            # Small forward-reading preference to avoid looping on the exact current page.
            st_pos = int(self.chunk_starts[did])
            same_page_pen = 0.12 if abs(st_pos - int(cursor)) < 1024 else 0.0
            if enc is not None and sem_scores:
                # Calibrate semantic cosine above baseline (~0.55) into [0, 1].
                sm_cal = max(0.0, min(1.0, (sm - 0.52) / 0.35))
                combined = 0.55 * sm_cal + 0.45 * lx_norm - same_page_pen
            else:
                combined = lx_norm - same_page_pen
            if combined > best_score:
                best_score = combined
                best_doc = did
                best_lex_norm = lx_norm
                best_sem = sm

        if best_doc < 0:
            return None

        # Relevance gate: refuse low-confidence noise jumps!
        has_strong_lex = (best_lex_norm >= 0.30 and len(scored_terms) >= min_lexical_terms)
        has_strong_sem = (best_sem >= min_semantic_cos)
        if best_score < min_hybrid_score and not (has_strong_lex or has_strong_sem):
            return None
        if not has_strong_lex and not has_strong_sem:
            return None

        top_terms = [t for _, t in scored_terms[:5]]
        start_offset = self._pinpoint_sentence_offset(best_doc, top_terms)
        return dict(
            start=int(start_offset),
            chunk_index=int(best_doc),
            score=round(float(best_score), 4),
            semantic_cos=round(float(best_sem), 4),
            lexical_norm=round(float(best_lex_norm), 4),
            matched_term=(top_terms[0] if top_terms else ''),
            method=('hybrid_gemma2_emeans_idf' if sem_scores else 'idf_lexical'),
        )


def main():
    import argparse
    p = argparse.ArgumentParser(description='Build Hybrid EmbeddingGemma-2 + e-means + IDF book index')
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--out-prefix', type=Path, required=True)
    p.add_argument('--chunk-bytes', type=int, default=2560)
    p.add_argument('--overlap-bytes', type=int, default=256)
    p.add_argument('--clusters', type=int, default=128)
    p.add_argument('--encoder-dir', type=Path, default=DEFAULT_GEMMA_ONNX_DIR)
    p.add_argument('--no-semantic', action='store_true')
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    idx = HybridBookIndex.build_from_corpus(
        args.corpus,
        out_prefix=args.out_prefix,
        chunk_bytes=args.chunk_bytes,
        overlap_bytes=args.overlap_bytes,
        n_clusters=args.clusters,
        encoder_dir=(None if args.no_semantic else args.encoder_dir),
        use_semantic=(not args.no_semantic),
        batch_size=args.batch_size,
        progress=True,
    )
    print(f'[DONE] {args.out_prefix}: chunks={idx.n_chunks}, terms={len(idx.idf_table)}, semantic={idx.vectors_i8 is not None}', flush=True)


if __name__ == '__main__':
    main()
