import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def toy():
    """Synthetic count-like data for interface tests only (never used for results): 160 samples x 300
    genes, the first 12 genes shifted in class 1."""
    rng = np.random.default_rng(0)
    n, d = 160, 300
    y = np.r_[np.ones(80, int), np.zeros(80, int)]
    lam = rng.gamma(5, 20, size=d)
    X = rng.poisson(lam, size=(n, d)).astype(float)
    X[y == 1, :12] *= 2.5
    genes = [f"GENE{i:03d}" for i in range(d)]
    return pd.DataFrame(X, columns=genes, index=[f"S{i}" for i in range(n)]), y
