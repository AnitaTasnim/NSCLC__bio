"""Loaders of the generalisation data sets (synthetic files; the real files are not needed)."""
import numpy as np
import pandas as pd

from pmtrd.datasets import gse68086


def test_gse68086_labels_exclusion_and_filter(tmp_path):
    cols = ["HD-1", "Control-2", "Type-Unknown-3", "NSCLC-4", "Breast-5", "HD-6", "CRC-7", "GBM-8", "Pancreas-9", "HD-10"]
    rng = np.random.default_rng(0)
    M = pd.DataFrame(rng.poisson(5, size=(4, len(cols))), columns=cols, index=["G1", "G2", "G3", "G4"])
    M.loc["G4"] = 0
    M.loc["G4", "NSCLC-4"] = 3                   # non-zero in 1 of 9 kept samples (< 20%)
    p = tmp_path / "m.txt"
    M.to_csv(p, sep="\t")
    X, y, info = gse68086(str(p), min_frac=0.20)
    assert "Type-Unknown-3" not in X.index and info["excluded"] == 1
    assert X.shape == (9, 3) and "G4" not in X.columns
    lab = dict(zip(X.index, y))
    assert lab["HD-1"] == 0 and lab["Control-2"] == 0 and lab["HD-10"] == 0
    assert lab["NSCLC-4"] == 1 and lab["GBM-8"] == 1
    assert info["positives"] == 5 and info["negatives"] == 4
