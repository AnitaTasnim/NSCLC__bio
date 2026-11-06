"""Single-sample pathway representation used by the reliability-guided pathway model.

Each sample is scored on its own: genes are ranked within the sample (average ranks for ties), ranks are
scaled to (0, 1] and centred, and a gene set's score is the mean centred rank of its member genes
(a singscore-type statistic; Foroutan et al., 2018). The score needs no parameter estimated from other
samples, so it is unaffected by library size and needs no batch alignment, and a single new sample can be
scored on its own. Gene sets are fixed, label-free external knowledge (MSigDB v7.5.1).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu, rankdata


def load_gmt(paths: Sequence[str]) -> Dict[str, List[str]]:
    sets: Dict[str, List[str]] = {}
    for p in paths:
        with open(p) as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) < 3:
                    continue
                if f[0] in sets:
                    raise ValueError(f"duplicate gene-set name {f[0]}")
                sets[f[0]] = sorted(set(g for g in f[2:] if g))
    return sets


def filter_sets(sets: Dict[str, List[str]], universe: Sequence[str], min_size: int = 15,
                max_size: int = 500) -> Dict[str, List[str]]:
    u = set(universe)
    out = {}
    for name, genes in sets.items():
        g = [x for x in genes if x in u]
        if min_size <= len(g) <= max_size:
            out[name] = g
    return dict(sorted(out.items()))


def centred_ranks(X: pd.DataFrame) -> np.ndarray:
    """Within-sample ranks scaled to (0, 1] and centred (samples x genes)."""
    R = rankdata(X.values, axis=1, method="average")
    return (R / X.shape[1] - 0.5).astype(np.float64)


def set_scores(X: pd.DataFrame, sets: Dict[str, List[str]]) -> pd.DataFrame:
    """Score every sample on every gene set (mean centred within-sample rank of the members).
    X must contain exactly the genes the ranks should be computed over (the gene universe)."""
    genes = list(X.columns); idx = {g: i for i, g in enumerate(genes)}
    rows, cols, vals = [], [], []
    for j, (name, members) in enumerate(sets.items()):
        m = [idx[g] for g in members if g in idx]
        if not m:
            raise ValueError(f"gene set {name} has no member in X")
        rows += m; cols += [j] * len(m); vals += [1.0 / len(m)] * len(m)
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(len(genes), len(sets)))
    S = np.asarray(W.T @ centred_ranks(X).T).T          # sparse (sets x genes) @ dense (genes x samples)
    return pd.DataFrame(S, index=X.index, columns=list(sets))


def top_sets(S: pd.DataFrame, y: np.ndarray, k: int) -> List[str]:
    """The k gene sets with the smallest two-sided Mann-Whitney p-value on the given (training) samples;
    ties broken by the larger |difference in median score|, then by name."""
    y = np.asarray(y)
    p = mannwhitneyu(S.values[y == 1], S.values[y == 0], axis=0).pvalue
    eff = np.abs(np.median(S.values[y == 1], 0) - np.median(S.values[y == 0], 0))
    order = sorted(range(S.shape[1]), key=lambda i: (p[i], -eff[i], S.columns[i]))
    return [S.columns[i] for i in order[:k]]
