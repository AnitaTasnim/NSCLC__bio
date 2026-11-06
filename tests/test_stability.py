import numpy as np
import pytest

from pmtrd.metrics import stability as ST


def test_jaccard_basic():
    assert ST.jaccard("ab", "ab") == 1.0
    assert ST.jaccard("ab", "cd") == 0.0
    assert ST.jaccard(["a", "b", "c"], ["b", "c", "d"]) == pytest.approx(0.5)


def test_pairwise_requires_two():
    with pytest.raises(ValueError):
        ST.pairwise_jaccard([["a"]])


def test_expected_jaccard_matches_simulation():
    rng = np.random.default_rng(1)
    k, n = 5, 40
    sims = [ST.jaccard(rng.choice(n, k, replace=False), rng.choice(n, k, replace=False)) for _ in range(20000)]
    assert ST.expected_jaccard_random(k, n) == pytest.approx(np.mean(sims), abs=0.005)


def test_phi_identical_is_one():
    p = [list("abcde")] * 4
    assert ST.nogueira_phi(p, d=100) == pytest.approx(1.0)


def test_phi_random_is_near_zero():
    rng = np.random.default_rng(2)
    d, k, M = 200, 10, 10
    vals = [ST.nogueira_phi([list(rng.choice(d, k, replace=False)) for _ in range(M)], d) for _ in range(300)]
    assert np.mean(vals) == pytest.approx(0.0, abs=0.01)


def test_phi_bounds_input():
    with pytest.raises(ValueError):
        ST.nogueira_phi([list("ab")], d=10)
    with pytest.raises(ValueError):
        ST.nogueira_phi([list("abc"), list("cde")], d=4)


def test_kuncheva_identical_and_unequal():
    assert ST.kuncheva([list("abc"), list("abc")], d=50) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ST.kuncheva([list("abc"), list("ab")], d=50)


def test_jackknife_and_frequency():
    p = [list("abc"), list("abd"), list("abe"), list("abc")]
    j = ST.jackknife(ST.nogueira_phi, p, d=20)
    assert j["se"] >= 0 and j["lo"] <= j["estimate"] <= j["hi"]
    f = ST.selection_frequency(p)
    assert f["a"] == 1.0 and f["c"] == 0.5 and list(f)[0] in ("a", "b")
