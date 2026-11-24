"""Tests for HybridBookIndex (EmbeddingGemma-2 + balanced spherical e-means + IDF lexical gate)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.ib_local.book_semantic_index import (
    DEFAULT_GEMMA_ONNX_DIR,
    HybridBookIndex,
    extract_lexical_terms,
    fit_emeans_spherical,
)
from scripts.ib_local.unified_world_port import SidecarCorpusStream


def test_extract_lexical_terms_slides_cjk_and_rejects_single_char_noise():
    # Real v10 failure case: `场去公园里` MUST yield `公园里` / `去公园` / `公园` (not just `场去`)!
    t1 = extract_lexical_terms('场去公园里')
    assert '公园里' in t1
    assert '去公园' in t1
    assert '公园' in t1

    # Real v10 failure case: `:往，滑` (single isolated chars) MUST yield ZERO terms!
    assert extract_lexical_terms(':往，滑') == []
    assert extract_lexical_terms('你') == []
    assert extract_lexical_terms('the and for') == []

    # Real v10 failure case: `天游戏快跳舞好吧` yields `游戏` and `跳舞`.
    t2 = extract_lexical_terms('名叫苏的周了。天游戏快跳舞好吧')
    assert '游戏' in t2 and '跳舞' in t2


def test_emeans_spherical_balanced_clustering():
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((240, 32)).astype(np.float32)
    centroids, assign = fit_emeans_spherical(vecs, n_clusters=8, n_iters=10)
    assert centroids.shape == (8, 32)
    assert assign.shape == (240,)
    # Unit-norm centroids & every cluster populated.
    norms = np.linalg.norm(centroids, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    counts = np.bincount(assign, minlength=8)
    assert int(counts.min()) >= 10


def test_hybrid_book_index_idf_save_load_and_noise_gate(tmp_path: Path):
    corpus = tmp_path / 'book.bin'
    pages = [
        '小熊在森林里找蜂蜜，蜜蜂嗡嗡叫着飞过了小溪。\n' * 12,
        '莉莉和朋友去公园里荡秋千、玩滑梯，阳光照在草地上。\n' * 12,
        '银行国际汇款需要经过代理行结算，手续费按比例计算。\n' * 12,
        '小鸟展开翅膀飞过蓝天和白云，落在大树枝头唱歌。\n' * 12,
        'Tim shared his red bicycle with Lily on the sunny meadow.\n' * 12,
    ]
    corpus.write_text(''.join(pages), encoding='utf-8')
    idx_prefix = tmp_path / 'hybrid_gemma2_index'
    idx = HybridBookIndex.build_from_corpus(
        corpus,
        out_prefix=idx_prefix,
        chunk_bytes=512,
        overlap_bytes=64,
        n_clusters=4,
        use_semantic=False,
    )
    reloaded = HybridBookIndex.load(idx_prefix, corpus.read_bytes(), encoder_dir=None)
    assert reloaded.n_chunks == idx.n_chunks

    # `场去公园里` hits the park page (`公园里`), NOT honey or bank!
    hit = reloaded.search('场去公园里')
    assert hit is not None
    snippet = corpus.read_bytes()[hit['start']:hit['start'] + 120].decode('utf-8', 'ignore')
    assert '公园里' in snippet

    # Single-char / garbled noise is strictly rejected (zero random jumps!).
    assert reloaded.search(':往，滑') is None
    assert reloaded.search('g gr d-') is None
    assert reloaded.search('i') is None

    # SidecarCorpusStream auto-loads sibling index & routes cleanly.
    stream = SidecarCorpusStream(corpus)
    assert stream.hybrid_index is not None
    resp, _, info = stream.take('场去公园里'.encode('utf-8'), 128, 128)
    assert info['match'] == 1
    assert '公园里' in bytes(resp).decode('utf-8', 'ignore')
    _, _, info_noise = stream.take(':往，滑'.encode('utf-8'), 128, 128)
    assert info_noise['match'] == 0


@pytest.mark.skipif(
    not (DEFAULT_GEMMA_ONNX_DIR / 'onnx' / 'model_q4.onnx').exists(),
    reason='Local EmbeddingGemma-2 Q4 ONNX model not installed',
)
def test_hybrid_book_index_gemma2_semantic_synonym_recall(tmp_path: Path):
    """Proves pure semantic recall via EmbeddingGemma-2 + e-means even with ZERO lexical overlap!"""
    corpus = tmp_path / 'semantic_book.bin'
    pages = [
        '莉莉和朋友去公园里荡秋千、玩滑梯，阳光暖暖地照在绿草地上。\n' * 8,
        '一只小麻雀展开羽毛飞上高高的蓝天，穿过软绵绵的白云唱起歌。\n' * 8,
        '银行国际汇款需要经过代理行核对账目和汇率，手续费按比例计算。\n' * 8,
        'The astronaut put on a silver helmet and stepped onto the quiet moon.\n' * 8,
    ]
    corpus.write_text(''.join(pages), encoding='utf-8')
    idx = HybridBookIndex.build_from_corpus(
        corpus,
        out_prefix=tmp_path / 'hybrid_gemma2_index',
        chunk_bytes=480,
        overlap_bytes=48,
        n_clusters=2,
        use_semantic=True,
    )
    # Cross-word semantic query: ZERO exact 2-char lexical overlap with `小麻雀/羽毛/蓝天`,
    # yet EmbeddingGemma-2 + e-means still retrieves the bird-in-sky page!
    hit = idx.search('会飞的雀鸟在空中翱翔')
    assert hit is not None
    assert hit['method'] == 'hybrid_gemma2_emeans_idf'
    assert hit['semantic_cos'] > 0.66
    snippet = corpus.read_bytes()[hit['start']:hit['start'] + 160].decode('utf-8', 'ignore')
    assert '麻雀' in snippet or '蓝天' in snippet
