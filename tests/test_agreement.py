import numpy as np
import pytest

from pmtrd.metrics import agreement as AG


def test_identical_replicates_agree_perfectly():
    S = np.tile(np.linspace(0.1, 0.9, 20), (4, 1))
    assert AG.pairwise_spearman(S) == pytest.approx(1.0)
    L = (S > 0.5).astype(int)
    assert AG.pairwise_discrepancy(L) == 0.0 and AG.ambiguity(L) == 0.0


def test_ambiguity_counts_conflicting_samples():
    L = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1]])
    assert AG.ambiguity(L) == pytest.approx(0.5)
    assert AG.pairwise_discrepancy(L) == pytest.approx((1 + 1 + 2) / 3 / 4)


def test_identity_share():
    rng = np.random.default_rng(6)
    base = rng.uniform(0.2, 0.8, 200)
    noise = rng.normal(0, 0.3, (10, 200))
    fixed = 1 / (1 + np.exp(-(np.log(base / (1 - base)) + noise)))
    same = AG.identity_share(fixed, fixed, B=200)
    assert same["share"] == pytest.approx(0.0, abs=1e-12)
    more = 1 / (1 + np.exp(-(np.log(base / (1 - base)) + 2 * noise)))     # 4x the variance
    s = AG.identity_share(more, fixed, B=200)
    assert s["share"] == pytest.approx(0.75, abs=0.05) and s["lo"] <= s["share"] <= s["hi"]


def test_shape_checks():
    with pytest.raises(ValueError):
        AG.pairwise_spearman(np.ones((1, 5)))
