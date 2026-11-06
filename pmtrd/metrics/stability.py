"""Gene-identity (molecular) stability of a set of selected panels.

All functions take panels as iterables of gene names. ``d`` is the number of candidate features the
selection could have chosen from (needed by chance-corrected measures).
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.stats import hypergeom


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def pairwise_jaccard(panels: Sequence[Iterable[str]]) -> float:
    """Mean Jaccard similarity over all pairs of panels."""
    if len(panels) < 2:
        raise ValueError("need at least two panels")
    return float(np.mean([jaccard(a, b) for a, b in combinations(panels, 2)]))


def expected_jaccard_random(k: int, n: int) -> float:
    """Expected Jaccard similarity of two independent uniformly random k-subsets of n items."""
    if not 0 < k <= n:
        raise ValueError("need 0 < k <= n")
    r = np.arange(0, k + 1)
    pmf = hypergeom(n, k, k).pmf(r)
    return float(np.sum(pmf * r / (2 * k - r)))


def kuncheva(panels: Sequence[Iterable[str]], d: int) -> float:
    """Kuncheva's consistency index, averaged over pairs (defined for panels of equal size k < d)."""
    sizes = {len(set(p)) for p in panels}
    if len(sizes) != 1:
        raise ValueError("Kuncheva's index needs panels of equal size")
    k = sizes.pop()
    if not 0 < k < d:
        raise ValueError("need 0 < k < d")
    vals = [(len(set(a) & set(b)) * d - k * k) / (k * (d - k)) for a, b in combinations(panels, 2)]
    return float(np.mean(vals))


def selection_matrix(panels: Sequence[Iterable[str]], universe: Sequence[str] | None = None):
    """Binary matrix (panels x features) and the feature names; features never selected are columns only
    if a universe is given."""
    names: List[str] = list(universe) if universe is not None else sorted(set().union(*map(set, panels)))
    idx = {g: i for i, g in enumerate(names)}
    Z = np.zeros((len(panels), len(names)), dtype=np.int8)
    for r, p in enumerate(panels):
        for g in set(p):
            if g not in idx:
                raise KeyError(f"{g} is not in the universe")
            Z[r, idx[g]] = 1
    return Z, names


def nogueira_phi(panels: Sequence[Iterable[str]], d: int) -> float:
    """Nogueira, Sechidis and Brown (2018) stability estimator.

    phi = 1 - mean_f s_f^2 / [ (k/d)(1 - k/d) ],  s_f^2 = M/(M-1) p_f (1 - p_f),
    where p_f is the selection frequency of feature f over M panels and k the mean panel size.
    Equals 1 for identical panels and has expectation 0 for random panels of fixed size.
    Features never selected contribute s_f^2 = 0, so only the selected ones need to be enumerated.
    """
    M = len(panels)
    if M < 2:
        raise ValueError("need at least two panels")
    Z, _ = selection_matrix(panels)
    if Z.shape[1] > d:
        raise ValueError("more distinct selected features than d")
    p = Z.mean(0)
    s2 = M / (M - 1) * p * (1 - p)
    kbar = Z.sum(1).mean()
    denom = (kbar / d) * (1 - kbar / d)
    if denom <= 0:
        raise ValueError("mean panel size must be between 0 and d")
    return float(1 - s2.sum() / d / denom)


def jackknife(stat, panels: Sequence[Iterable[str]], **kw) -> Dict[str, float]:
    """Leave-one-panel-out jackknife estimate, standard error and normal 95% CI of a set statistic."""
    M = len(panels)
    if M < 3:
        raise ValueError("jackknife needs at least three panels")
    full = stat(panels, **kw)
    loo = np.array([stat([p for j, p in enumerate(panels) if j != i], **kw) for i in range(M)])
    se = float(np.sqrt((M - 1) / M * np.sum((loo - loo.mean()) ** 2)))
    return {"estimate": float(full), "se": se, "lo": float(full - 1.96 * se), "hi": float(full + 1.96 * se)}


def selection_frequency(panels: Sequence[Iterable[str]]) -> Dict[str, float]:
    """Fraction of panels that contain each gene (genes never selected are omitted)."""
    Z, names = selection_matrix(panels)
    return dict(sorted(zip(names, Z.mean(0).tolist()), key=lambda kv: (-kv[1], kv[0])))
