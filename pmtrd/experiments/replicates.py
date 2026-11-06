"""Replicate study: does gene-identity instability make the predictions unstable?

Design (all within the paper's five outer folds, which are verified against ``R8_outer_folds.csv``):
for outer fold k with training set T_k and untouched held-out set H_k, draw R stratified subsamples
S_kr of T_k (fraction ``frac``). On each S_kr, every selector chooses a panel and trains its classifier;
the model scores H_k, and a decision threshold is learned on S_kr by inner five-fold CV. A panel-fixed
arm trains the same classifier on S_kr with the paper's four-term fold panel P_k (selected on T_k, never
on H_k), so that its predictions vary only through the training samples.

Outputs (results/):
    units.csv          one row per (fold, replicate, method): panel size, held-out AUC, threshold, seconds
    panels.csv         every selected panel
    scores.csv         held-out scores (long format)
    stability.csv      per method: Nogueira phi (jackknife CI), pairwise Jaccard, Kuncheva (equal sizes)
    agreement.csv      per method: pairwise Spearman of scores, label discrepancy, ambiguity, AUC spread
    identity_share.csv share of prediction variance due to re-selecting the panel (six-stage vs fixed)
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import pandas as pd

from .. import selectors as SEL
from ..engine import Dataset
from ..metrics import agreement as AG
from ..metrics import predictive as PR
from ..metrics import stability as ST
from .common import UnitCache, check_folds, reference_fold_panels, stratified_subsample

DEFAULTS = dict(R=10, frac=0.8, methods=["six_stage", "mw_topk", "l1_topk", "stability", "mrmr_ifs_svm", "random"],
                k=27, B=500, folds_file="data/reference/R8_outer_folds.csv",
                fold_panels_file="data/reference/R4_nested_folds.csv")


def run_units(run, data: Dataset, p: Dict) -> None:
    X, y, er = data.X, data.y, data.run
    folds = er.folds()
    check_folds(folds, p["folds_file"], X.index)
    fixed = reference_fold_panels(p["fold_panels_file"])
    cache = UnitCache(run)
    nj = run.cfg.get("n_jobs", 2)
    for k, (tr, te) in enumerate(folds):
        for r in range(p["R"]):
            key = f"unit_f{k}_r{r:02d}"
            if cache.has(key):
                continue
            rng = np.random.default_rng(run.cfg["seed"] * 1000 + 100 * k + r)
            sub = stratified_subsample(tr, y, p["frac"], rng)
            Xs, ys, Xh, yh = X.iloc[sub], y[sub], X.iloc[te], y[te]
            out = {"fold": k, "rep": r, "train": sub, "test": te, "methods": {}}
            arms = [(m, SEL.make(m, k=p["k"], B=p["B"], n_jobs=nj,
                                 seed=(10_000 + 100 * k + r) if m == "random" else run.cfg["seed"]))
                    for m in p["methods"]]
            arms.append(("fixed", SEL.Fixed(fixed[k], n_jobs=nj, seed=run.cfg["seed"])))
            for name, sel in arms:
                t0 = time.time()
                genes = sel.select(Xs, ys, er)
                scores = sel.fit_predict(Xs, ys, Xh, genes, er)
                thr = sel.inner_threshold(Xs, ys, genes, er)
                out["methods"][name] = dict(genes=genes, scores=scores, thr=thr, info=dict(sel.info),
                                            auc=PR.auc(yh, scores), seconds=round(time.time() - t0, 1))
                run.log("unit_method", fold=k + 1, rep=r, method=name, n=len(genes),
                        auc=round(out["methods"][name]["auc"], 4), s=out["methods"][name]["seconds"])
            cache.put(key, out)


def summarise(run, data: Dataset, p: Dict) -> Dict:
    y = data.y; d = data.X.shape[1]
    units = [v for k, v in UnitCache(run).all().items() if k.startswith("unit_")]
    if not units:
        raise RuntimeError("no units in the cache")
    rows, prow, srow = [], [], []
    for u in units:
        for m, v in u["methods"].items():
            rows.append(dict(fold=u["fold"] + 1, rep=u["rep"], method=m, n_genes=len(v["genes"]), auc=v["auc"],
                             threshold=v["thr"], seconds=v["seconds"], **{f"info_{a}": b for a, b in v["info"].items()}))
            prow.append(dict(fold=u["fold"] + 1, rep=u["rep"], method=m, genes=";".join(v["genes"])))
            for i, s in zip(u["test"], v["scores"]):
                srow.append(dict(fold=u["fold"] + 1, rep=u["rep"], method=m, sample=data.X.index[i], y=int(y[i]),
                                 score=float(s), label=int(s >= v["thr"])))
    U = pd.DataFrame(rows); U.to_csv(run.path("results", "units.csv"), index=False)
    pd.DataFrame(prow).to_csv(run.path("results", "panels.csv"), index=False)
    Sc = pd.DataFrame(srow); Sc.to_csv(run.path("results", "scores.csv"), index=False)
    methods = list(dict.fromkeys(U.method))
    stab, agr = [], []
    for m in methods:
        phis, jac, kun, sp, dis, amb, sd = [], [], [], [], [], [], []
        allp = []
        for k in sorted(U.fold.unique()):
            us = sorted([u for u in units if u["fold"] + 1 == k and m in u["methods"]], key=lambda u: u["rep"])
            if len(us) < 3:
                continue
            panels = [u["methods"][m]["genes"] for u in us]; allp.append(panels)
            phis.append(ST.nogueira_phi(panels, d)); jac.append(ST.pairwise_jaccard(panels))
            if len({len(set(q)) for q in panels}) == 1:
                kun.append(ST.kuncheva(panels, d))
            Smat = np.vstack([u["methods"][m]["scores"] for u in us])
            Lmat = np.vstack([u["methods"][m]["scores"] >= u["methods"][m]["thr"] for u in us]).astype(int)
            sp.append(AG.pairwise_spearman(Smat)); dis.append(AG.pairwise_discrepancy(Lmat)); amb.append(AG.ambiguity(Lmat))
            sd.append(np.std([u["methods"][m]["auc"] for u in us], ddof=1))
        if not phis:
            continue
        # jackknife over replicates within folds: leave replicate r out of every fold
        R = min(len(ps) for ps in allp)
        loo = [np.mean([ST.nogueira_phi([q for j, q in enumerate(ps) if j != r], d) for ps in allp]) for r in range(R)]
        se = float(np.sqrt((R - 1) / R * np.sum((np.array(loo) - np.mean(loo)) ** 2)))
        phi = float(np.mean(phis))
        sizes = U[U.method == m].n_genes
        stab.append(dict(method=m, phi=phi, phi_lo=phi - 1.96 * se, phi_hi=phi + 1.96 * se,
                         pairwise_jaccard=float(np.mean(jac)), kuncheva=float(np.mean(kun)) if kun else np.nan,
                         size_min=int(sizes.min()), size_max=int(sizes.max()), size_median=float(sizes.median())))
        a = U[U.method == m].auc
        bag = np.full(len(y), np.nan)          # selection-bagged ensemble: mean of the R replicate scores
        for k in sorted(U.fold.unique()):
            us = [u for u in units if u["fold"] + 1 == k and m in u["methods"]]
            bag[us[0]["test"]] = np.vstack([u["methods"][m]["scores"] for u in us]).mean(0)
        ok = ~np.isnan(bag)
        agr.append(dict(method=m, bagged_auc=PR.auc(y[ok], bag[ok]), auc_mean=float(a.mean()), auc_sd_within_fold=float(np.mean(sd)),
                        spearman=float(np.mean(sp)), discrepancy=float(np.mean(dis)), ambiguity=float(np.mean(amb))))
    stab = pd.DataFrame(stab); agr = pd.DataFrame(agr)
    stab.to_csv(run.path("results", "stability.csv"), index=False); agr.to_csv(run.path("results", "agreement.csv"), index=False)
    # identity share: six-stage (and every XGBoost selector) against the panel-fixed arm, all held-out samples
    ish = []
    for m in [m for m in methods if m not in ("fixed", "mrmr_ifs_svm")]:
        Pr, Pf = [], []
        for k in sorted(U.fold.unique()):
            us = sorted([u for u in units if u["fold"] + 1 == k], key=lambda u: u["rep"])
            Pr.append(np.vstack([u["methods"][m]["scores"] for u in us]))
            Pf.append(np.vstack([u["methods"]["fixed"]["scores"] for u in us]))
        if min(len(x) for x in Pr) < 2:
            continue
        res = AG.identity_share(np.hstack(Pr), np.hstack(Pf), seed=run.cfg["seed"])
        ish.append(dict(method=m, **res))
    ish = pd.DataFrame(ish, columns=["method", "share", "lo", "hi", "var_reselect", "var_fixed"]); ish.to_csv(run.path("results", "identity_share.csv"), index=False)
    six = ish[ish.method == "six_stage"].iloc[0] if (ish.method == "six_stage").any() else None
    summ = {"n_units": len(units), "d": d,
            "stability": stab.set_index("method").round(4).to_dict(orient="index"),
            "agreement": agr.set_index("method").round(4).to_dict(orient="index"),
            "identity_share_six_stage": None if six is None else {k: float(six[k]) for k in ("share", "lo", "hi")}}
    run.summary(summ)
    return summ


def main(run, data: Dataset, params: Dict, mode: str = "all") -> None:
    p = {**DEFAULTS, **(params or {})}
    run.log("params", **{k: v for k, v in p.items()})
    if mode in ("all", "run"):
        run_units(run, data, p)
    if mode in ("all", "summarise"):
        s = summarise(run, data, p)
        run.log("summary", identity_share=s["identity_share_six_stage"])
