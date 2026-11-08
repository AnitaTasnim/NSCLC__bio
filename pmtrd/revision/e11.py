"""E11 - normalisation: A4 on GREIN counts (a, saved), CPM (b) and TMM-CPM (c), GSE89843, paper's folds.

Plan: docs/plan_E8_E19.md, section E11 (with the amendment on the linear scale).
Run:  python -m pmtrd.revision.e11 run --scale cpm|tmm      then   python -m pmtrd.revision.e11 summarise
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import selectors as SEL
from ..experiments.common import check_folds
from ..metrics import agreement as AG
from ..metrics import predictive as PR
from ..metrics import stability as ST
from .common import out_dir, path, saved, write_meta
from .units import Cache, load_dataset, subsample

R = 10


# ------------------------------------------------------------------ normalisation (per sample; TMM vs a training reference)
def cpm(C: pd.DataFrame, factors=None) -> pd.DataFrame:
    lib = C.sum(axis=1).replace(0, 1)
    if factors is not None:
        lib = lib * factors
    return C.div(lib, axis=0) * 1e6


def _tmm_factor(obs: np.ndarray, ref: np.ndarray, logratio_trim: float = 0.3, sum_trim: float = 0.05) -> float:
    """edgeR calcNormFactors(method = "TMM") for one sample against the reference (with weighting)."""
    No, Nr = obs.sum(), ref.sum()
    ok = (obs > 0) & (ref > 0)
    o, r = obs[ok].astype(float), ref[ok].astype(float)
    logR = np.log2((o / No) / (r / Nr))
    absE = (np.log2(o / No) + np.log2(r / Nr)) / 2
    v = (No - o) / No / o + (Nr - r) / Nr / r
    n = len(logR)
    loL, hiL = np.floor(n * logratio_trim) + 1, n + 1 - (np.floor(n * logratio_trim) + 1)
    loS, hiS = np.floor(n * sum_trim) + 1, n + 1 - (np.floor(n * sum_trim) + 1)
    from scipy.stats import rankdata
    rl, rs = rankdata(logR), rankdata(absE)
    keep = (rl >= loL) & (rl <= hiL) & (rs >= loS) & (rs <= hiS)
    return float(2 ** (np.sum(logR[keep] / v[keep]) / np.sum(1 / v[keep])))


def tmm_factors(C: pd.DataFrame, train: np.ndarray) -> pd.Series:
    """Reference = training sample whose upper quartile (of counts / library size) is closest to the mean
    upper quartile of the training samples (edgeR's default); factors scaled to geometric mean 1 over the
    training samples; held-out samples get factors against the same reference and the same scaling."""
    V = C.values
    lib = V.sum(1)
    f75 = np.quantile(V[train] / lib[train, None], 0.75, axis=1)
    ref = train[int(np.argmin(np.abs(f75 - f75.mean())))]
    f = np.array([_tmm_factor(V[i], V[ref]) for i in range(len(C))])
    f = f / np.exp(np.mean(np.log(f[train])))
    return pd.Series(f, index=C.index)


def transform(X: pd.DataFrame, scale: str, train: np.ndarray) -> pd.DataFrame:
    if scale == "cpm":
        return cpm(X)
    if scale == "tmm":
        return cpm(X, tmm_factors(X, train))
    raise ValueError(scale)


def run(scale: str) -> None:
    cache = Cache(path("runs", "revision", "E11", "cache", scale))
    X, y, er, _ = load_dataset("GSE89843", 1)
    folds = er.folds()
    check_folds(folds, "data/reference/R8_outer_folds.csv", X.index)
    for k, (tr, te) in enumerate(folds):
        Z = transform(X, scale, tr)
        key = f"nested_f{k}"
        if not cache.has(key):
            sel = SEL.make("six_stage", B=500, n_jobs=1, seed=42)
            g = sel.select(Z.iloc[tr], y[tr], er)
            p = sel.fit_predict(Z.iloc[tr], y[tr], Z.iloc[te], g, er)
            cache.put(key, dict(genes=g, te=te, scores=p))
            print(scale, key, len(g), round(PR.auc(y[te], p), 4), flush=True)
        fold_panel = cache.get(key)["genes"]
        for r in range(R):
            key = f"rep_f{k}_r{r:02d}"
            if cache.has(key):
                continue
            sub = subsample(tr, y, 42, k, r)
            sel = SEL.make("six_stage", B=500, n_jobs=1, seed=42)
            g = sel.select(Z.iloc[sub], y[sub], er)
            p = sel.fit_predict(Z.iloc[sub], y[sub], Z.iloc[te], g, er)
            pf = SEL.Fixed(fold_panel, n_jobs=1, seed=42).fit_predict(Z.iloc[sub], y[sub], Z.iloc[te], fold_panel, er)
            cache.put(key, dict(genes=g, te=te, scores=p, fixed_scores=pf))
            print(scale, key, len(g), round(PR.auc(y[te], p), 4), flush=True)


def summarise() -> None:
    from ..engine import T
    od = out_dir("runs", "revision", "E11")
    X, y, er, _ = load_dataset("GSE89843", 1)
    d = X.shape[1]
    four = pd.read_csv(path("data", "reference", "R4_nested_oof.csv")).oof_prob.values
    a, lo, hi, _ = T.delong_ci(y, four)
    st = saved("GSE89843", "stability.csv").set_index("method")
    ish = saved("GSE89843", "identity_share.csv").set_index("method")
    rows = [dict(scale="counts (a, saved)", nested_auc=a, lo=lo, hi=hi, phi=st.phi["six_stage"],
                 psi=ish.share["six_stage"], psi_lo=ish.lo["six_stage"], psi_hi=ish.hi["six_stage"])]
    for scale in ("cpm", "tmm"):
        cache = Cache(path("runs", "revision", "E11", "cache", scale))
        if not cache.has(f"rep_f4_r{R - 1:02d}"):
            print("incomplete:", scale)
            continue
        oof = np.full(len(y), np.nan)
        phis, Pr, Pf, sizes = [], [], [], []
        for k in range(5):
            u = cache.get(f"nested_f{k}")
            oof[u["te"]] = u["scores"]
            sizes.append(len(u["genes"]))
            reps = [cache.get(f"rep_f{k}_r{r:02d}") for r in range(R)]
            phis.append(ST.nogueira_phi([q["genes"] for q in reps], d))
            Pr.append(np.vstack([q["scores"] for q in reps]))
            Pf.append(np.vstack([q["fixed_scores"] for q in reps]))
        a, lo, hi, _ = T.delong_ci(y, oof)
        dl = T.delong_paired(y, oof, four)
        res = AG.identity_share(np.hstack(Pr), np.hstack(Pf), seed=42)
        rows.append(dict(scale=scale, nested_auc=a, lo=lo, hi=hi, diff_vs_counts=dl["diff"], diff_lo=dl["lo95"],
                         diff_hi=dl["hi95"], p_vs_counts=dl["p"], phi=float(np.mean(phis)), psi=res["share"],
                         psi_lo=res["lo"], psi_hi=res["hi"], fold_panel_sizes=";".join(map(str, sizes))))
    pd.DataFrame(rows).to_csv(path(od, "E11.csv"), index=False)
    write_meta(od, experiment="E11")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "summarise"])
    ap.add_argument("--scale", choices=["cpm", "tmm"])
    a = ap.parse_args()
    run(a.scale) if a.action == "run" else summarise()


if __name__ == "__main__":
    main()
