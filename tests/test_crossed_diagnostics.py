"""Tests of the post-hoc interval diagnostic (pmtrd/crossed/diagnostics.py)."""
import numpy as np

from pmtrd.crossed.diagnostics import crossed_jackknife, jackknife, leak_demo
from pmtrd.crossed.estimator import CrossedData
from pmtrd.crossed.synthetic import make_model


def test_pure_panel_cancellation_and_bootstrap_leak():
    """z_rq = phi_q: S and SQ are exactly 0 with uniform draw weights, but not under multiplicity weights."""
    r = leak_demo(R=10, B=500, seed=1)
    assert abs(r["S_uniform"]) < 1e-12 and abs(r["SQ_uniform"]) < 1e-12
    assert r["S_boot_sd"] > 0.01


def test_jackknife_preserves_exact_zero():
    """Deleting one draw keeps uniform weights, so the jackknife SE of S is exactly 0 for a pure panel effect."""
    rng = np.random.default_rng(2)
    R = 9
    phi = rng.standard_normal(R)
    d = CrossedData(grid=np.broadcast_to(phi[None, :, None], (R, R, 3)).astype(float).copy())
    est, se = jackknife(d)
    assert abs(est[0]) < 1e-12 and se[0] < 1e-10


def test_crossed_jackknife_shapes():
    d = make_model("interaction", P=20).simulate(8, np.random.default_rng(3), n_seed_cells=4, M=2)
    est, se = crossed_jackknife(d, G=5)
    assert est.shape == se.shape == (7,)
    assert np.all(se[:6] >= 0)
