"""Predictive reliability: discrimination, calibration and operating characteristics."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 1e-6


def _check(y, p) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    if y.shape != p.shape:
        raise ValueError("y and p must have the same shape")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("y must be binary 0/1")
    if np.isnan(p).any():
        raise ValueError("predictions contain NaN")
    return y, p


def logit(p) -> np.ndarray:
    q = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(q / (1 - q))


def brier(y, p) -> float:
    y, p = _check(y, p)
    return float(np.mean((p - y) ** 2))


def calibration(y, p) -> Dict[str, float]:
    """Calibration intercept (logistic recalibration with logit(p) as offset, i.e. slope fixed at 1) and
    calibration slope (logistic regression of y on logit(p)). Ideal values 0 and 1."""
    import statsmodels.api as sm
    y, p = _check(y, p)
    lp = logit(p)
    slope = float(sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1])
    inter = float(sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0])
    return {"intercept": inter, "slope": slope, "brier": brier(y, p)}


def wilson(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


def youden_threshold(y, p) -> float:
    """Threshold maximising sensitivity + specificity - 1 (ties: the lowest threshold)."""
    y, p = _check(y, p)
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    i = int(np.flatnonzero(j == j.max())[-1])
    return float(min(thr[i], np.max(p)))


def operating_point(y, p, thr: float) -> Dict[str, float]:
    """Sensitivity and specificity at a given threshold (score >= thr is positive), with Wilson CIs."""
    y, p = _check(y, p)
    pos = p >= thr
    tp = int((pos & (y == 1)).sum()); fn = int((~pos & (y == 1)).sum())
    tn = int((~pos & (y == 0)).sum()); fp = int((pos & (y == 0)).sum())
    sl, sh = wilson(tp, tp + fn); pl, ph = wilson(tn, tn + fp)
    return {"threshold": float(thr), "sens": tp / (tp + fn), "sens_lo": sl, "sens_hi": sh,
            "spec": tn / (tn + fp), "spec_lo": pl, "spec_hi": ph, "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def auc(y, p) -> float:
    y, p = _check(y, p)
    return float(roc_auc_score(y, p))


def stratified_bootstrap(y, p, fn, B: int = 2000, seed: int = 42) -> Tuple[float, float]:
    """Percentile 95% CI of fn(y, p) over stratified bootstrap resamples."""
    y, p = _check(y, p)
    rng = np.random.default_rng(seed)
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(B):
        i = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        try:
            vals.append(fn(y[i], p[i]))
        except Exception:
            continue
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
