"""Nested benchmark on the paper's outer folds: every selector is run once per outer training fold T_k
and scored on the untouched held-out fold H_k. The six-stage reference predictions are the paper's
(four-term: ``R4_nested_oof.csv``; five-term: ``FR_nested_oof_fi.csv``), so that the comparison is on
identical folds and identical samples.

Pre-declared primary comparisons: six-stage (four-term) versus each baseline on pooled out-of-fold AUC,
paired DeLong test, Holm correction over the baselines. Gene-identity stability across the five fold
panels (Nogueira phi with leave-one-fold-out jackknife CI) is reported for every method.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .. import selectors as SEL
from ..config import resolve
from ..engine import Dataset, T
from ..metrics import predictive as PR
from ..metrics import stability as ST
from .common import UnitCache, check_folds, reference_fold_panels

DEFAULTS = dict(methods=["mw_topk", "l1_topk", "stability", "mrmr_ifs_svm", "random"], k=27,
                n_random=20, folds_file="data/reference/R8_outer_folds.csv",
                fold_panels_file="data/reference/R4_nested_folds.csv",
                six_oof="data/reference/R4_nested_oof.csv", five_oof="data/reference/FR_nested_oof_fi.csv",
                five_folds="data/reference/FR_nested_folds.csv")


def main(run, data: Dataset, params: Dict, mode: str = "all") -> None:
    p = {**DEFAULTS, **(params or {})}
    X, y, er = data.X, data.y, data.run
    folds = er.folds(); check_folds(folds, p["folds_file"], X.index)
    cache = UnitCache(run); nj = run.cfg.get("n_jobs", 2)
    if mode in ("all", "run"):
        jobs = [(m, 0) for m in p["methods"] if m != "random"] + [("random", i) for i in range(p["n_random"])]
        for m, rep in jobs:
            key = f"{m}_{rep:02d}"
            if cache.has(key):
                continue
            oof = np.full(len(y), np.nan); lab = np.zeros(len(y), int); panels = []
            for k, (tr, te) in enumerate(folds):
                sel = SEL.make(m, k=p["k"], n_jobs=nj, seed=(20_000 + 100 * rep + k) if m == "random" else run.cfg["seed"])
                g = sel.select(X.iloc[tr], y[tr], er)
                oof[te] = sel.fit_predict(X.iloc[tr], y[tr], X.iloc[te], g, er)
                lab[te] = oof[te] >= sel.inner_threshold(X.iloc[tr], y[tr], g, er)
                panels.append(g)
            cache.put(key, dict(method=m, rep=rep, oof=oof, labels=lab, panels=panels))
            run.log("benchmark_method", method=m, rep=rep, auc=round(PR.auc(y, oof), 4),
                    sizes=[len(q) for q in panels])
    if mode not in ("all", "summarise"):
        return
    res = cache.all(); d = X.shape[1]
    six = pd.read_csv(resolve(p["six_oof"])); five = pd.read_csv(resolve(p["five_oof"]))
    assert list(six["sample"]) == list(X.index), "reference predictions are not in sample order"
    ref_panels = reference_fold_panels(p["fold_panels_file"])
    five_panels = {int(r.fold) - 1: r.panel.split(";") for r in pd.read_csv(resolve(p["five_folds"])).itertuples()}
    entries = [("six_stage_four_term", six.oof_prob.values, [ref_panels[k] for k in range(5)]),
               ("six_stage_five_term", five["oof"].values, [five_panels[k] for k in range(5)])]
    for m in p["methods"]:
        if m == "random":
            continue
        r = res[f"{m}_00"]; entries.append((m, r["oof"], r["panels"]))
    rows = []
    for name, oof, panels in entries:
        a, lo, hi, _ = T.delong_ci(y, oof)
        phi = ST.jackknife(ST.nogueira_phi, panels, d=d)
        rows.append(dict(method=name, auc=a, auc_lo=lo, auc_hi=hi, phi=phi["estimate"], phi_lo=phi["lo"],
                         phi_hi=phi["hi"], pairwise_jaccard=ST.pairwise_jaccard(panels),
                         sizes=";".join(str(len(q)) for q in panels)))
    rand = [res[f"random_{i:02d}"] for i in range(p["n_random"]) if f"random_{i:02d}" in res]
    if rand:
        ra = [PR.auc(y, r["oof"]) for r in rand]
        rows.append(dict(method="random (median of %d)" % len(ra), auc=float(np.median(ra)), auc_lo=float(np.min(ra)),
                         auc_hi=float(np.max(ra)), phi=float(np.median([ST.nogueira_phi(r["panels"], d) for r in rand])),
                         phi_lo=np.nan, phi_hi=np.nan,
                         pairwise_jaccard=float(np.median([ST.pairwise_jaccard(r["panels"]) for r in rand])),
                         sizes="27"))
    D = pd.DataFrame(rows)
    # primary comparisons, Holm-adjusted
    comp = []
    for name, oof, _ in entries[1:]:
        dl = T.delong_paired(y, six.oof_prob.values, oof)
        comp.append(dict(comparison=f"six_stage_four_term - {name}", delta=dl["diff"], lo=dl["lo95"], hi=dl["hi95"], p=dl["p"]))
    C = pd.DataFrame(comp)
    C["p_holm"] = multipletests(C.p, method="holm")[1]
    D.to_csv(run.path("results", "benchmark.csv"), index=False); C.to_csv(run.path("results", "comparisons.csv"), index=False)
    run.summary({"benchmark": D.set_index("method").round(4).to_dict(orient="index"),
                 "comparisons": C.set_index("comparison").round(5).to_dict(orient="index")})
    run.log("benchmark_done", methods=len(D))
