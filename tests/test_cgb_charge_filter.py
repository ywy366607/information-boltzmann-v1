import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "information_boltzmann" / "core"))
from cgb_charge_filter import ChargeFilter, OnlineNgram  # noqa: E402

TEXT = b"the cat sat on the mat. the dog sat on the log. " * 40


def test_prediction_and_charge_are_normalized():
    m = ChargeFilter(16, seed=0)
    for b in TEXT[:200]:
        p = m.predict()
        assert abs(float(p.sum()) - 1) < 1e-4
        nll = m.observe(b)
        assert nll == -math.log(float(p[b])) or abs(nll + math.log(float(p[b]))) < 1e-5
        assert abs(float(m.alpha.sum()) - 1) < 1e-5


def test_masses_stay_exact_integers_and_row_sums_match():
    m = ChargeFilter(16, seed=0, row_cap=4096)
    for b in TEXT:
        m.observe(b)
    for M, sums in ((m.A, m.R), (m.B, m.S)):
        assert torch.equal(M, M.round()) and float(M.min()) >= 1
        assert torch.equal(M.sum(1), sums)
        assert float(sums.max()) <= 4096 + m.deposit * m.samples


def test_frozen_twin_changes_no_mass_but_keeps_filtering():
    m = ChargeFilter(16, seed=0)
    m.learning = False
    A, B = m.A.clone(), m.B.clone()
    alphas = []
    for b in TEXT[:50]:
        m.observe(b)
        alphas.append(m.alpha.clone())
    assert torch.equal(A, m.A) and torch.equal(B, m.B)
    assert not torch.equal(alphas[0], alphas[-1])


def test_learning_beats_its_frozen_start_on_repetitive_text():
    learner, frozen = ChargeFilter(16, seed=0), ChargeFilter(16, seed=0)
    frozen.learning = False
    tail = len(TEXT) // 2
    lo = [learner.observe(b) for b in TEXT][tail:]
    lf = [frozen.observe(b) for b in TEXT][tail:]
    assert sum(lo) < 0.7 * sum(lf)


def test_online_ngram_is_prequential():
    g = OnlineNgram(1)
    first = g.observe(97)
    assert abs(first - math.log(256)) < 1e-9
    g.observe(97)
    assert g.observe(97) < first
