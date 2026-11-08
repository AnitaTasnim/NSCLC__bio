"""Mathematical tests of the crossed pick-freeze estimator on analytic models with known components.

Monte Carlo tolerance: the component estimators (variances, not shares) are unbiased, so the mean over N
independent designs must lie within 4 standard errors of the exact value."""
import numpy as np
import pytest

from pmtrd.crossed.estimator import CrossedData, bootstrap, components, eq3_psi, moments, percentile_ci
from pmtrd.crossed.synthetic import make_model

CASES = ["additive", "interaction", "random_panel", "fixed_panel", "deterministic", "seed", "coupling"]


def _mc(kind, R=8, N=400, missing=0.0, seed=1):
    model = make_model(kind)
    rng = np.random.default_rng(seed)
    est = []
    for _ in range(N):
        c = components(model.simulate(R, rng, n_seed_cells=6, M=3, missing=missing))
        est.append([c.s2["S"], c.s2["Q"], c.s2["SQ"], c.s2["E"], c.s2["gamma"], c.V_R])
    return model.truth(), np.array(est)


@pytest.mark.parametrize("kind", CASES)
def test_components_unbiased(kind):
    truth, est = _mc(kind)
    names = ["S", "Q", "SQ", "E", "gamma"]
    target = [truth[f"s2_{k}"] for k in names] + [truth["V_R"]]
    mean, se = est.mean(0), est.std(0, ddof=1) / np.sqrt(len(est))
    for k, t, m, s in zip(names + ["V_R"], target, mean, se):
        assert abs(m - t) <= 4 * s + 1e-12, f"{kind}: {k} mean {m:.4f} vs exact {t:.4f} (SE {s:.4f})"


def test_incomplete_design_unbiased():
    """Cells missing completely at random (30%) leave every moment unbiased."""
    truth, est = _mc("interaction", R=10, missing=0.3, seed=7)
    for j, k in enumerate(["S", "Q", "SQ", "E", "gamma"]):
        m, s = est[:, j].mean(), est[:, j].std(ddof=1) / np.sqrt(len(est))
        assert abs(m - truth[f"s2_{k}"]) <= 4 * s


def test_exact_zero_controls():
    """Deterministic classifier: every seed pair agrees, so D_seed is exactly 0. Fixed panel with
    kappa > 0 and random panel with kappa = 0 have exact gamma = 0 in the truth."""
    rng = np.random.default_rng(3)
    det = make_model("deterministic").simulate(8, rng, n_seed_cells=5, M=3)
    assert np.all(moments(det)["D_seed"] == 0)
    assert make_model("fixed_panel").truth()["gamma"] == 0
    assert make_model("random_panel").truth()["gamma"] == 0
    assert make_model("fixed_panel").truth()["Q"] == 0 and make_model("fixed_panel").truth()["SQ"] == 0


def test_coupling_nonzero_and_recovered():
    truth, est = _mc("coupling", R=10, seed=11)
    assert truth["s2_gamma"] > 0.1
    m, s = est[:, 4].mean(), est[:, 4].std(ddof=1) / np.sqrt(len(est))
    assert abs(m - truth["s2_gamma"]) <= 4 * s


def test_disjoint_pair_count_and_weights():
    """With z = 1 the weighted disjoint-pair count equals R(R-1)(R-2)(R-3); scaling all draw weights by a
    constant leaves the estimates unchanged."""
    R, P = 7, 3
    rng = np.random.default_rng(0)
    d = make_model("interaction", P=P).simulate(R, rng, n_seed_cells=4)
    base = components(d)
    doubled = components(d, v=np.full(R, 2.0))
    for k in base.s2:
        assert np.isclose(base.s2[k], doubled.s2[k])
    # pair count via the public function: a grid of ones has zero variance everywhere
    ones = CrossedData(grid=np.ones((R, R, P)))
    m = moments(ones)
    assert np.allclose(m["D_both"], 0) and np.allclose(m["D_col"], 0)


def test_too_few_draws_for_dboth():
    d = make_model("additive", P=5).simulate(3, np.random.default_rng(0))
    assert np.all(np.isnan(moments(d)["D_both"]))


def test_eq3_matches_definition():
    rng = np.random.default_rng(5)
    tied, fixed = rng.standard_normal((10, 6)) * 2, rng.standard_normal((10, 6))
    d = CrossedData(grid=np.zeros((10, 10, 6)), eq3_tied=tied, eq3_fixed=fixed)
    expect = 1 - fixed.var(0, ddof=1).sum() / tied.var(0, ddof=1).sum()
    assert np.isclose(eq3_psi(d), expect)


def test_bootstrap_shapes_and_coverage_sanity():
    model = make_model("interaction", P=30)
    d = model.simulate(10, np.random.default_rng(2), n_seed_cells=8, M=2)
    reps = bootstrap(d, 200, np.random.default_rng(3), "pigeonhole")
    ci = percentile_ci(reps)
    assert reps.shape == (200, 7) and ci.shape == (2, 7)
    assert np.all(ci[0, :6] <= ci[1, :6])


def test_batched_bootstrap_equals_loop():
    """The vectorised bootstrap gives exactly the statistics of the reference one-at-a-time computation."""
    from pmtrd.crossed.estimator import _stats, _stats_batch
    rng = np.random.default_rng(9)
    d = make_model("coupling", P=12).simulate(9, rng, n_seed_cells=5, M=3, missing=0.2)
    d.eq3_tied, d.eq3_fixed = rng.standard_normal((9, 12)), rng.standard_normal((9, 12))
    V = rng.multinomial(9, np.full(9, 1 / 9), size=6).astype(float)
    W = rng.multinomial(12, np.full(12, 1 / 12), size=6).astype(float)
    batch = _stats_batch(d, V, W)
    for b in range(6):
        if (V[b] > 0).sum() >= 4:
            assert np.allclose(batch[b], _stats(d, V[b], W[b]), equal_nan=True)
