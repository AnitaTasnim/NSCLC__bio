"""Unit tests for the revision-experiment helpers (pmtrd/revision)."""
import numpy as np
import pandas as pd

from pmtrd.metrics import stability as ST
from pmtrd.revision.e11 import _tmm_factor, tmm_factors
from pmtrd.revision.e18 import phi_c
from pmtrd.revision.e19 import ambiguity, pair_tensor, weighted_pair_mean


def test_phi_c_identity_is_nogueira():
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(60)]
    panels = [list(rng.choice(genes, 8, replace=False)) for _ in range(6)]
    used = sorted(set().union(*map(set, panels)))
    assert np.isclose(phi_c(panels, 60, np.eye(len(used)), used, 0.0), ST.nogueira_phi(panels, 60))


def test_pair_mean_and_ambiguity():
    L = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0]])
    d = pair_tensor(L)
    # pairs (0,1): 1/4, (0,2): 2/4, (1,2): 1/4 -> mean 1/3
    assert np.isclose(weighted_pair_mean(d, np.ones(3), np.ones(4)), 1 / 3)
    assert np.isclose(ambiguity(L), 0.5)
    # ambiguity of two replicates equals their disagreement
    assert np.isclose(ambiguity(L[:2]), weighted_pair_mean(pair_tensor(L[:2]), np.ones(2), np.ones(4)))


def test_tmm_factor_properties():
    rng = np.random.default_rng(1)
    C = pd.DataFrame(rng.poisson(30, (5, 400)))
    assert np.isclose(_tmm_factor(C.values[0], C.values[0]), 1.0)
    f = tmm_factors(C, np.arange(3))
    assert np.isclose(np.exp(np.mean(np.log(f.values[:3]))), 1.0)
