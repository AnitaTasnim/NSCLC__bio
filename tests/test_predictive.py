import numpy as np
import pytest

from pmtrd.metrics import predictive as PR


def test_wilson_known_value():
    lo, hi = PR.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    with pytest.raises(ValueError):
        PR.wilson(11, 10)


def test_calibration_of_calibrated_predictions():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    c = PR.calibration(y, p)
    assert c["intercept"] == pytest.approx(0, abs=0.06)
    assert c["slope"] == pytest.approx(1, abs=0.06)


def test_calibration_detects_shift():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    q = 1 / (1 + np.exp(-(PR.logit(p) - 1.0)))       # predictions shifted down by 1 logit
    assert PR.calibration(y, q)["intercept"] == pytest.approx(1.0, abs=0.08)


def test_brier_and_checks():
    assert PR.brier([0, 1], [0.0, 1.0]) == 0.0
    with pytest.raises(ValueError):
        PR.brier([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError):
        PR.auc([0, 1], [0.1, np.nan])


def test_youden_and_operating_point():
    y = np.array([0, 0, 0, 1, 1, 1]); p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = PR.youden_threshold(y, p)
    assert 0.3 < t <= 0.7
    op = PR.operating_point(y, p, t)
    assert op["sens"] == 1.0 and op["spec"] == 1.0 and op["tp"] == 3 and op["tn"] == 3


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(5)
    y = np.r_[np.ones(100, int), np.zeros(100, int)]; p = np.r_[rng.normal(1, 1, 100), rng.normal(0, 1, 100)]
    lo, hi = PR.stratified_bootstrap(y, p, PR.auc, B=300)
    assert lo < PR.auc(y, p) < hi
