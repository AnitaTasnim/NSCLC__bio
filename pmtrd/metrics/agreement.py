"""Prediction-level agreement between models trained on different resamples (and therefore, possibly,
different gene panels), evaluated on the same held-out samples.

``S`` is an R x n matrix: row r holds the scores of replicate model r for the n held-out samples.
``L`` is an R x n binary matrix of the corresponding class decisions.

The label-based measures follow the predictive-multiplicity literature (Marx, Calmon and Ustun, 2020):
*ambiguity* is the fraction of samples that receive both labels from at least two replicates;
*disagreement* (called discrepancy in the code) is the mean pairwise fraction of samples whose labels
differ; Marx et al. define discrepancy against a reference model, which this is not.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict

import numpy as np
from scipy.stats import spearmanr

from .predictive import logit


def _mat(S) -> np.ndarray:
    S = np.asarray(S, float)
    if S.ndim != 2 or S.shape[0] < 2:
        raise ValueError("need an R x n matrix with R >= 2")
    return S


def pairwise_spearman(S) -> float:
    """Mean Spearman correlation between the score vectors of all pairs of replicates."""
    S = _mat(S)
    return float(np.mean([spearmanr(a, b).correlation for a, b in combinations(S, 2)]))


def pairwise_discrepancy(L) -> float:
    """Mean over pairs of replicates of the fraction of samples with different labels."""
    L = _mat(L).astype(int)
    return float(np.mean([np.mean(a != b) for a, b in combinations(L, 2)]))


def ambiguity(L) -> float:
    """Fraction of samples whose label is not the same for every replicate."""
    L = _mat(L).astype(int)
    return float(np.mean(L.min(0) != L.max(0)))


def per_sample_variance(P, scale: str = "logit") -> np.ndarray:
    """Across-replicate variance of each held-out sample's prediction (probabilities in, logit scale by
    default so that variance is not compressed near 0 and 1)."""
    P = _mat(P)
    X = logit(P) if scale == "logit" else P
    return X.var(axis=0, ddof=1)


def identity_share(P_reselect, P_fixed, B: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Share of across-replicate prediction variance attributable to re-selecting the panel.

    Both matrices hold predictions for the same n held-out samples from the same R training subsamples:
    ``P_reselect`` with the panel re-selected on each subsample, ``P_fixed`` with one panel held fixed.
    share = 1 - mean_i Var_fixed(i) / mean_i Var_reselect(i). A value near 0 means that gene-identity
    instability adds little to the prediction variance that training-sample variation causes anyway;
    a value near 1 means that most prediction variance comes from the changing gene identity.
    The 95% CI resamples held-out samples (columns) with replacement.
    """
    Vr = per_sample_variance(P_reselect); Vf = per_sample_variance(P_fixed)
    if Vr.shape != Vf.shape:
        raise ValueError("matrices must describe the same held-out samples")
    est = 1 - Vf.mean() / Vr.mean()
    rng = np.random.default_rng(seed); n = len(Vr); vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        vals.append(1 - Vf[i].mean() / Vr[i].mean())
    return {"share": float(est), "lo": float(np.quantile(vals, 0.025)), "hi": float(np.quantile(vals, 0.975)),
            "var_reselect": float(Vr.mean()), "var_fixed": float(Vf.mean())}
