"""Control experiments with the real simulation pipeline (generator, selectors, classifiers): each control
has a component whose exact value is known, which the estimator must recover within Monte Carlo error."""
import os
import tempfile

import numpy as np
import pytest

from pmtrd.crossed import simulate as sm
from pmtrd.crossed.estimator import components, moments

CFG = {"block": 10, "n_signal": 20, "prevalence": 0.6, "k": 10, "fixed_seed": 42,
       "classifiers": {"logistic": {"C": 1.0},
                       "xgboost": {"n_estimators": 40, "learning_rate": 0.1, "max_depth": 3, "subsample": 0.8,
                                   "colsample_bytree": 0.8, "device": "cpu"}}}
DATA = {"n": 120, "p": 200, "rho": 0.5, "auc": 0.9, "f": 0.8}


def _design(selector, clf, R, rep, clf_over=None, M=2):
    cfg = dict(CFG)
    if clf_over:
        cfg = {**CFG, "classifiers": {**CFG["classifiers"], clf: {**CFG["classifiers"][clf], **clf_over}}}
    st = sm.Setting(cfg, DATA, selector, clf)
    X, y = st.T(0)
    Xh, _ = st.gen.sample(80, np.random.default_rng(1000 + rep), stratified=False)
    cache = sm.PanelCache(os.path.join(tempfile.mkdtemp(), "p.npz"))
    ids = [5000 + rep * R + j for j in range(R)]
    cells = np.argwhere(~np.eye(R, dtype=bool))
    seed_cells = np.array([(r, (r + 1) % R) for r in range(R)])
    return sm.build_design(st, 0, X, y, ids, cells, seed_cells, M, Xh, cache, tag=R)[0]


def _mean_se(selector, clf, key, N=25, R=7, **kw):
    vals = []
    for rep in range(N):
        c = components(_design(selector, clf, R, rep, **kw))
        vals.append(c.s2[key])
    vals = np.array(vals)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(N)


def test_deterministic_classifier_has_no_seed_variance():
    d = _design("mw", "logistic", 6, 0)
    assert np.all(moments(d)["D_seed"] == 0)


def test_deterministic_xgboost_configuration_has_no_seed_variance():
    d = _design("mw", "xgboost", 5, 0, clf_over={"subsample": 1.0, "colsample_bytree": 1.0})
    assert np.allclose(moments(d)["D_seed"], 0, atol=1e-10)


def test_xgboost_with_subsampling_has_seed_variance():
    d = _design("mw", "xgboost", 5, 0)
    assert components(d).s2["E"] > 0


@pytest.mark.parametrize("key", ["Q", "SQ", "gamma"])
def test_fixed_panel_control(key):
    """The panel never changes: panel identity, interaction and coupling are exactly 0."""
    m, se = _mean_se("fixed", "logistic", key)
    assert abs(m) <= 4 * se + 1e-10


def test_random_panel_control():
    """Panels drawn independently of the data: coupling is exactly 0, panel identity is not."""
    m, se = _mean_se("random", "logistic", "gamma")
    assert abs(m) <= 4 * se
    q, qse = _mean_se("random", "logistic", "Q")
    assert q > 4 * qse
