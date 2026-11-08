"""E15 - reproducibility of the A4 selection across seeds and computing environments (GSE89843).

Plan: docs/plan_E8_E19.md, section E15 (with the amendment on what a seed sets).
Environment A: n_jobs = 2 (joblib processes), BLAS threads as set by the caller.
Environment B: n_jobs = 1, run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1.

Run:  python -m pmtrd.revision.e15 full --env A          (seeds 0-20 on the full cohort)
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pmtrd.revision.e15 full --env B
      python -m pmtrd.revision.e15 nested                  (seeds 1-5, paper's folds, environment A)
      python -m pmtrd.revision.e15 summarise
"""
from __future__ import annotations

import argparse
import functools
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from ..metrics import predictive as PR
from ..metrics import stability as ST
from .common import out_dir, path, write_meta
from .units import Cache, load_dataset

SEEDS = list(range(0, 21))          # 0 = the paper's setting (reference); 1-20 as planned
NESTED_SEEDS = [1, 2, 3, 4, 5]


def _boot_offset(b: int, Xb: np.ndarray, yb: np.ndarray, offset: int = 0) -> np.ndarray:
    """Engine's Stage-5 resample with the random stream shifted: RandomState(offset + b)."""
    idx = np.random.RandomState(offset + b).choice(len(yb), len(yb), replace=True)
    Xs = StandardScaler().fit_transform(Xb[idx])
    m = LassoCV(cv=3, max_iter=3000, tol=1e-2, alphas=20, random_state=offset + b)
    m.fit(Xs, yb[idx])
    return m.coef_ != 0


def select_a4(X, y, seed: int, n_jobs: int):
    from ..engine import S
    original = S._boot_one
    S._boot_one = functools.partial(_boot_offset, offset=1000 * seed)
    try:
        return list(S.phase1_select(X, y, B=500, n_jobs=n_jobs, seed=42 if seed == 0 else seed))
    finally:
        S._boot_one = original


def full(env: str) -> None:
    nj = 2 if env == "A" else 1
    cache = Cache(path("runs", "seeds", "cache"))
    X, y, run, _ = load_dataset("GSE89843", nj)
    for s in SEEDS:
        key = f"full_env{env}_seed{s:02d}"
        if cache.has(key):
            continue
        import time
        t0 = time.time()
        g = select_a4(X, y, s, nj)
        cache.put(key, dict(genes=g, seconds=round(time.time() - t0, 1),
                            threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}))
        print(key, len(g), round(time.time() - t0), flush=True)


def nested() -> None:
    cache = Cache(path("runs", "seeds", "cache"))
    X, y, run, _ = load_dataset("GSE89843", 2)
    folds = run.folds()
    for s in NESTED_SEEDS:
        for k, (tr, te) in enumerate(folds):
            key = f"nested_seed{s:02d}_f{k}"
            if cache.has(key):
                continue
            g = select_a4(X.iloc[tr], y[tr], s, 2)
            p = run.xgb(n_jobs=2).fit(X.iloc[tr][g].values, y[tr]).predict_proba(X.iloc[te][g].values)[:, 1]
            cache.put(key, dict(genes=g, te=te, scores=p))
            print(key, len(g), round(PR.auc(y[te], p), 3), flush=True)


def summarise() -> None:
    from ..engine import T
    cache = Cache(path("runs", "seeds", "cache"))
    od = out_dir("runs", "seeds")
    X, y, run, _ = load_dataset("GSE89843", 2)
    P27 = run.panel_in(X)
    rows = []
    panels = {}
    for env in ("A", "B"):
        for s in SEEDS:
            key = f"full_env{env}_seed{s:02d}"
            if not cache.has(key):
                continue
            u = cache.get(key)
            panels[(env, s)] = u["genes"]
            rows.append(dict(kind="full", env=env, seed=s, n_genes=len(u["genes"]),
                             jaccard_P27=ST.jaccard(u["genes"], P27), seconds=u["seconds"],
                             genes=";".join(u["genes"])))
    df = pd.DataFrame(rows)
    summ = []
    for env in ("A", "B"):
        ps = [panels[(env, s)] for s in SEEDS if s > 0 and (env, s) in panels]
        if len(ps) >= 2:
            summ.append(dict(kind="between_seeds", env=env, pairwise_jaccard=ST.pairwise_jaccard(ps),
                             phi=ST.nogueira_phi(ps, X.shape[1]), n=len(ps)))
    same = [s for s in SEEDS if ("A", s) in panels and ("B", s) in panels]
    summ.append(dict(kind="A_vs_B_same_seed", identical=int(sum(panels[("A", s)] == panels[("B", s)] for s in same)),
                     n=len(same), mean_jaccard=float(np.mean([ST.jaccard(panels[("A", s)], panels[("B", s)])
                                                              for s in same])) if same else np.nan))
    # nested
    n = len(y)
    for s in NESTED_SEEDS:
        if not all(cache.has(f"nested_seed{s:02d}_f{k}") for k in range(5)):
            continue
        oof = np.full(n, np.nan)
        fp = []
        for k in range(5):
            u = cache.get(f"nested_seed{s:02d}_f{k}")
            oof[u["te"]] = u["scores"]
            fp.append(u["genes"])
        a, lo, hi, _ = T.delong_ci(y, oof)
        summ.append(dict(kind="nested", seed=s, auc=a, lo=lo, hi=hi, fold_sizes=";".join(str(len(g)) for g in fp)))
    for k in range(5):
        ps = [cache.get(f"nested_seed{s:02d}_f{k}")["genes"] for s in NESTED_SEEDS if cache.has(f"nested_seed{s:02d}_f{k}")]
        if len(ps) >= 2:
            summ.append(dict(kind="nested_fold_panels_across_seeds", fold=k + 1,
                             pairwise_jaccard=ST.pairwise_jaccard(ps), n=len(ps)))
    df.to_csv(path(od, "E15_panels.csv"), index=False)
    pd.DataFrame(summ).to_csv(path(od, "E15.csv"), index=False)
    write_meta(od, experiment="E15", seeds=SEEDS, nested_seeds=NESTED_SEEDS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["full", "nested", "summarise"])
    ap.add_argument("--env", choices=["A", "B"], default="A")
    a = ap.parse_args()
    {"full": lambda: full(a.env), "nested": nested, "summarise": summarise}[a.action]()


if __name__ == "__main__":
    main()
