"""E9 - selection bagging, reduced design fixed in docs/plan_E8_E19.md (section E9).

A4 (six_stage) only; partitions seed 42 (the paper's / saved folds; saved replicates 0-9 reused) and seed 43
(new); 25 subsample selections per outer fold; ensembles of the first 10 and of all 25 replicate models.

Run:  python -m pmtrd.revision.e9 run --datasets Singh Gordon GSE68086 GSE89843 [--n-jobs 1]
      python -m pmtrd.revision.e9 summarise
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from .common import SAVED, fold_matrices, out_dir, path, saved, write_meta
from .units import Cache, load_dataset, outer_folds, run_unit, subsample

PARTITIONS = (42, 43)
NREP = 25
SEED_SEL = 42
DATASETS = ["Singh", "Gordon", "GSE68086", "GSE89843"]


def cache_for(ds: str) -> Cache:
    return Cache(path("runs", "bagging", "cache", ds))


def run(datasets: List[str], n_jobs: int) -> None:
    for ds in datasets:
        X, y, er, _ = load_dataset(ds, n_jobs)
        cache = cache_for(ds)
        for part in PARTITIONS:
            folds = outer_folds(ds, y, er, part)
            if part == 42:
                saved_f = {f["fold"] - 1: f for f in fold_matrices(ds, "six_stage")}
                fixed = saved(ds, "panels.csv")
                fixed = {int(r.fold) - 1: r.genes.split(";") for r in fixed[(fixed.method == "fixed") & (fixed.rep == 0)].itertuples()}
            for k, (tr, te) in enumerate(folds):
                if part == 42:     # check the folds are those of the saved run
                    ids = list(X.index[te])
                    if sorted(ids) != sorted(saved_f[k]["samples"]):
                        raise RuntimeError(f"{ds}: fold {k + 1} of partition 42 differs from the saved run")
                key = f"p{part}_f{k}_frozen"
                if not cache.has(key):
                    genes = fixed[k] if part == 42 else None
                    u = run_unit(X, y, er, tr, te, "six_stage", seed=SEED_SEL, n_jobs=n_jobs, genes=genes)
                    u["test"] = te
                    cache.put(key, u)
                    print(ds, key, u["seconds"], round(u["auc"], 3), flush=True)
                for r in range(NREP):
                    key = f"p{part}_f{k}_r{r:02d}"
                    if cache.has(key):
                        continue
                    if part == 42 and r < 10:
                        f = saved_f[k]
                        order = [list(f["samples"]).index(s) for s in X.index[te]]
                        from ..metrics import predictive as PR
                        u = dict(genes=None, scores=f["S"][r][order], thr=float(f["thr"][r]), seconds=0.0,
                                 saved=True)
                        u["auc"] = PR.auc(y[te], u["scores"])
                    else:
                        sub = subsample(tr, y, part, k, r)
                        u = run_unit(X, y, er, sub, te, "six_stage", seed=SEED_SEL, n_jobs=n_jobs)
                    u["test"] = te
                    cache.put(key, u)
                    print(ds, key, u["seconds"], round(u["auc"], 3), flush=True)


def calib(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    import statsmodels.api as sm
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    try:
        slope = float(sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1])
        icpt = float(sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0])
    except Exception:
        slope = icpt = np.nan
    return {"cal_intercept": icpt, "cal_slope": slope, "brier": float(np.mean((p - y) ** 2))}


def summarise() -> None:
    from ..metrics import predictive as PR
    od = out_dir("runs", "bagging")
    fold_rows, ds_rows = [], []
    for ds in DATASETS:
        cache = cache_for(ds)
        if not cache.has("p43_f4_r24"):
            print("incomplete:", ds)
            continue
        X, y, er, _ = load_dataset(ds, 1)
        n = len(y)
        for part in PARTITIONS:
            pooled = {B: np.full(n, np.nan) for B in (10, 25)}
            singles = np.full((NREP, n), np.nan)
            frozen = np.full(n, np.nan)
            for k in range(5):
                us = [cache.get(f"p{part}_f{k}_r{r:02d}") for r in range(NREP)]
                fz = cache.get(f"p{part}_f{k}_frozen")
                te = us[0]["test"]
                S = np.vstack([u["scores"] for u in us])
                thr = np.array([u["thr"] for u in us])
                yk = y[te]
                frozen[te] = fz["scores"]
                singles[:, te] = S
                row = {"dataset": ds, "partition": part, "fold": k + 1, "n_test": len(te), "n_train": n - len(te),
                       "auc_frozen": PR.auc(yk, fz["scores"])}
                for B in (10, 25):
                    bag = S[:B].mean(0)
                    pooled[B][te] = bag
                    row[f"auc_bag{B}"] = PR.auc(yk, bag)
                    row[f"auc_single_mean{B}"] = float(np.mean([PR.auc(yk, s) for s in S[:B]]))
                    row[f"gain{B}"] = row[f"auc_bag{B}"] - row[f"auc_single_mean{B}"]
                    row[f"bag{B}_minus_frozen"] = row[f"auc_bag{B}"] - row["auc_frozen"]
                la = S[:10].mean(0) >= thr[:10].mean()
                lb = S[10:20].mean(0) >= thr[10:20].mean()
                row["ensemble_disagreement_10v10"] = float(np.mean(la != lb))
                fold_rows.append(row)
            r = {"dataset": ds, "partition": part}
            for B in (10, 25):
                r.update({f"bag{B}_{k}": v for k, v in calib(y, pooled[B]).items()})
                r[f"auc_bag{B}_pooled"] = PR.auc(y, pooled[B])
            sc = [calib(y, s) for s in singles]
            r.update({f"single_{k}": float(np.mean([c[k] for c in sc])) for k in sc[0]})
            r.update({f"frozen_{k}": v for k, v in calib(y, frozen).items()})
            ds_rows.append(r)
    F = pd.DataFrame(fold_rows)
    F.to_csv(path(od, "E9_folds.csv"), index=False)
    pd.DataFrame(ds_rows).to_csv(path(od, "E9_calibration.csv"), index=False)
    # corrected resampled t-test over the 10 fold-level differences (Nadeau & Bengio 2003)
    tests = []
    for ds, d in F.groupby("dataset"):
        J = len(d)
        corr = 1 / J + float((d.n_test / d.n_train).mean())
        for col in ("gain10", "gain25", "bag10_minus_frozen", "bag25_minus_frozen"):
            x = d[col].values
            m, v = x.mean(), x.var(ddof=1)
            t = m / np.sqrt(corr * v) if v > 0 else np.nan
            p = 2 * stats.t.sf(abs(t), J - 1) if np.isfinite(t) else np.nan
            half = stats.t.ppf(0.975, J - 1) * np.sqrt(corr * v)
            tests.append({"dataset": ds, "quantity": col, "J": J, "mean": m, "lo": m - half, "hi": m + half,
                          "t": t, "p": p})
    pd.DataFrame(tests).to_csv(path(od, "E9_tests.csv"), index=False)
    write_meta(od, experiment="E9", partitions=PARTITIONS, nrep=NREP)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "summarise"])
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--n-jobs", type=int, default=1)
    a = ap.parse_args()
    if a.action == "run":
        run(a.datasets, a.n_jobs)
    else:
        summarise()


if __name__ == "__main__":
    main()
