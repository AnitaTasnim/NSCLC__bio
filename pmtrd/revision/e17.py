"""E17 - pathway circularity: do gene sets replicate better than genes at matched list size? (Vumc halves)

Plan: docs/plan_E8_E19.md, section E17.   Run:  python -m pmtrd.revision.e17 run | summarise
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

from .. import selectors as SEL
from ..config import resolve
from ..metrics import stability as ST
from ..pathway import filter_sets, load_gmt, set_scores, top_sets
from .common import out_dir, path, write_meta
from .units import Cache, load_dataset

NSPLIT = 25
MS = (27, 50)
LIBS = ["data/inputs/genesets/MSigDB_Hallmark_v7.5.1.gmt", "data/inputs/genesets/Reactome_MSigDB_v7.5.1.gmt"]


def half_split(idx: np.ndarray, y: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    a = []
    for c in (0, 1):
        pos = idx[y[idx] == c]
        a.append(rng.choice(pos, len(pos) // 2, replace=False))
    A = np.sort(np.concatenate(a))
    return A, np.setdiff1d(idx, A)


def tests(V: np.ndarray, y: np.ndarray):
    p = mannwhitneyu(V[y == 1], V[y == 0], axis=0).pvalue
    s = np.sign(V[y == 1].mean(0) - V[y == 0].mean(0))
    return np.nan_to_num(p, nan=1.0), s


def replication(pA, sA, pB, sB, m):
    top = np.argsort(pA, kind="stable")[:m]
    q = multipletests(pB[top], method="fdr_bh")[1]
    return float(np.mean((q < 0.05) & (sA[top] == sB[top])))


def kuncheva_pair(a, b, d):
    k = min(len(a), len(b))
    return ST.kuncheva([a[:k], b[:k]], d)


def run() -> None:
    from ..engine import T  # noqa: F401
    od = out_dir("results", "pathway")
    cache = Cache(path("results", "pathway", "cache_E17"))
    from .units import load_discovery
    data = load_discovery(2)
    X, y, er, site = data.X, data.y, data.run, data.site.values
    vumc = np.flatnonzero(site == er.cfg["vumc_site"])
    Xe = pd.read_parquet(resolve("data/inputs/gse68086/GSE68086_counts_symbols.parquet"))
    universe = sorted(set(X.columns) & set(Xe.columns) - {"__sym"})
    sets = filter_sets(load_gmt([resolve(q) for q in LIBS]), universe, 15, 500)
    S = set_scores(X[universe], sets)
    Lg = er.logX(X)[universe]                     # log2-CPM (CPM over all genes of the sample)
    for s in range(1, NSPLIT + 1):
        key = f"split{s:02d}"
        if cache.has(key):
            continue
        A, B = half_split(vumc, y, s)
        res = {"A": A, "B": B}
        for h, idx in (("A", A), ("B", B)):
            res[f"gene_{h}"] = tests(Lg.values[idx], y[idx])
            res[f"set_{h}"] = tests(S.values[idx], y[idx])
            res[f"a4_{h}"] = SEL.make("six_stage", B=500, n_jobs=2, seed=42).select(X.iloc[idx], y[idx], er)
            res[f"pw_{h}"] = top_sets(S.iloc[idx], y[idx], 27)
        cache.put(key, res)
        print(key, len(res["a4_A"]), len(res["a4_B"]), flush=True)
    write_meta(od, experiment="E17", n_vumc=len(vumc), universe=len(universe), n_sets=len(sets))


def summarise() -> None:
    cache = Cache(path("results", "pathway", "cache_E17"))
    X, y, er, _ = load_dataset("GSE89843", 2)
    Xe = pd.read_parquet(resolve("data/inputs/gse68086/GSE68086_counts_symbols.parquet"))
    universe = sorted(set(X.columns) & set(Xe.columns) - {"__sym"})
    n_sets = len(filter_sets(load_gmt([resolve(q) for q in LIBS]), universe, 15, 500))
    rows = []
    for s in range(1, NSPLIT + 1):
        r = cache.get(f"split{s:02d}")
        row = {"split": s}
        for m in MS:
            for lvl in ("gene", "set"):
                (pA, sA), (pB, sB) = r[f"{lvl}_A"], r[f"{lvl}_B"]
                row[f"{lvl}_rep_m{m}"] = 0.5 * (replication(pA, sA, pB, sB, m) + replication(pB, sB, pA, sA, m))
        row["a4_kuncheva"] = kuncheva_pair(r["a4_A"], r["a4_B"], X.shape[1])
        row["a4_jaccard"] = ST.jaccard(r["a4_A"], r["a4_B"])
        row["pw_kuncheva"] = kuncheva_pair(r["pw_A"], r["pw_B"], n_sets)
        row["pw_jaccard"] = ST.jaccard(r["pw_A"], r["pw_B"])
        rows.append(row)
    D = pd.DataFrame(rows)
    D.to_csv(path("results", "pathway", "E17_splits.csv"), index=False)
    out = []
    for m in MS:
        g, st_ = D[f"gene_rep_m{m}"], D[f"set_rep_m{m}"]
        w = wilcoxon(st_, g)
        out.append(dict(comparison=f"replication rate, sets - genes, m={m}", gene_median=g.median(),
                        set_median=st_.median(), median_diff=(st_ - g).median(), wilcoxon_stat=w.statistic,
                        p=w.pvalue))
    w = wilcoxon(D.pw_kuncheva, D.a4_kuncheva)
    out.append(dict(comparison="chance-corrected overlap (Kuncheva), pathway top-27 - A4", gene_median=D.a4_kuncheva.median(),
                    set_median=D.pw_kuncheva.median(), median_diff=(D.pw_kuncheva - D.a4_kuncheva).median(),
                    wilcoxon_stat=w.statistic, p=w.pvalue))
    pd.DataFrame(out).to_csv(path("results", "pathway", "E17.csv"), index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "summarise"])
    a = ap.parse_args()
    run() if a.action == "run" else summarise()


if __name__ == "__main__":
    main()
