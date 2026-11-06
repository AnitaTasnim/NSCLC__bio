"""Reliability-guided pathway model, evaluated with PMTRD on the paper's folds.

Motivation (from the molecular-reliability results): gene identity is unstable whereas pathway-level
signals replicate. The model therefore represents every sample by single-sample gene-set scores
(``pmtrd.pathway``) and selects pathways instead of genes. Everything data-dependent is learned inside
the training data: the k gene sets with the smallest Mann-Whitney p-value on the training samples, and an
XGBoost classifier with the paper's settings. Gene sets are fixed external knowledge.

Evaluated exactly like the gene panel:
    nested      pooled out-of-fold AUC on the paper's outer folds; paired DeLong against the four-term and
                five-term six-stage nested predictions; calibration; training-derived thresholds;
                identity stability of the five fold set lists (Nogueira phi, d = number of gene sets)
    replicates  the replicate design of ``replicates`` (same subsample seeds): phi, AUC spread, agreement,
                identity share psi against the fold's fixed set list
    site        nested CV inside Vumc (the only mixed-class site) and a Vumc-trained model applied to the
                samples of all other site codes, as for the gene panel
    locked      trained once on all 633 samples and applied unchanged to GSE68086: AUC (bootstrap CI),
                calibration intercept and slope, Brier score, sensitivity and specificity at the discovery
                threshold; compared with the locked 27-gene model (paper, log2-CPM, no alignment)
    availability external AUC when a fraction of all measured genes is missing in GSE68086 (ranks
                recomputed over the remaining genes; the trained model unchanged)
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..config import resolve
from ..engine import Dataset, T
from ..metrics import agreement as AG
from ..metrics import predictive as PR
from ..metrics import stability as ST
from ..pathway import filter_sets, load_gmt, set_scores, top_sets
from .common import check_folds, stratified_subsample

DEFAULTS = dict(libraries=["data/inputs/genesets/MSigDB_Hallmark_v7.5.1.gmt",
                           "data/inputs/genesets/Reactome_MSigDB_v7.5.1.gmt"],
                k=27, min_size=15, max_size=500, R=10, frac=0.8,
                external_expr="data/inputs/gse68086/GSE68086_counts_symbols.parquet",
                external_meta="data/inputs/gse68086/GSE68086_meta.csv",
                missing_fractions=[0.1, 0.2, 0.3, 0.5], missing_draws=50,
                folds_file="data/reference/R8_outer_folds.csv", four_oof="data/reference/R4_nested_oof.csv",
                five_oof="data/reference/FR_nested_oof_fi.csv")


def _fit(er, S, y, cols):
    return er.xgb().fit(S[cols].values, y)


def _inner_thr(er, S, y, cols, seed):
    oof = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(S, y):
        oof[b] = _fit(er, S.iloc[a], y[a], cols).predict_proba(S.iloc[b][cols].values)[:, 1]
    return PR.youden_threshold(y, oof), oof


def main(run, data: Dataset, params: Dict, mode: str = "all") -> None:
    p = {**DEFAULTS, **(params or {})}
    X, y, er, seed = data.X, data.y, data.run, run.cfg["seed"]
    folds = er.folds(); check_folds(folds, p["folds_file"], X.index)
    Xe = pd.read_parquet(resolve(p["external_expr"]))
    if "__sym" in Xe.columns:
        Xe = Xe.drop(columns="__sym")
    me = pd.read_csv(resolve(p["external_meta"])).set_index("sample")
    Xe = Xe.loc[me.index]; ye = me["label"].values.astype(int)
    universe = sorted(set(X.columns) & set(Xe.columns))
    sets = filter_sets(load_gmt([resolve(q) for q in p["libraries"]]), universe, p["min_size"], p["max_size"])
    run.log("pathway_setup", universe=len(universe), sets=len(sets), external=len(ye))
    S = set_scores(X[universe], sets); Se = set_scores(Xe[universe], sets)
    S.to_csv(run.path("results", "discovery_set_scores.csv.gz"))
    d = S.shape[1]; out = {}

    # ---------------------------------------------------------------- nested
    oof = np.zeros(len(y)); lab = np.zeros(len(y), int); lists = []
    for k, (tr, te) in enumerate(folds):
        cols = top_sets(S.iloc[tr], y[tr], p["k"])
        m = _fit(er, S.iloc[tr], y[tr], cols)
        oof[te] = m.predict_proba(S.iloc[te][cols].values)[:, 1]
        thr, _ = _inner_thr(er, S.iloc[tr], y[tr], cols, seed)
        lab[te] = oof[te] >= thr; lists.append(cols)
        run.log("nested_fold", fold=k + 1, auc=round(PR.auc(y[te], oof[te]), 4))
    a, lo, hi, _ = T.delong_ci(y, oof)
    four = pd.read_csv(resolve(p["four_oof"])).oof_prob.values; five = pd.read_csv(resolve(p["five_oof"]))["oof"].values
    d4 = T.delong_paired(y, oof, four); d5 = T.delong_paired(y, oof, five)
    cal = PR.calibration(y, oof)
    tp = int(((lab == 1) & (y == 1)).sum()); fn = int(((lab == 0) & (y == 1)).sum())
    tn = int(((lab == 0) & (y == 0)).sum()); fp = int(((lab == 1) & (y == 0)).sum())
    phi = ST.jackknife(ST.nogueira_phi, lists, d=d)
    out["nested"] = dict(auc=a, lo=lo, hi=hi, delta_vs_four=d4["diff"], delta_vs_four_lo=d4["lo95"],
                         delta_vs_four_hi=d4["hi95"], p_vs_four=d4["p"], delta_vs_five=d5["diff"], p_vs_five=d5["p"],
                         **cal, sens=tp / (tp + fn), sens_ci=PR.wilson(tp, tp + fn), spec=tn / (tn + fp),
                         spec_ci=PR.wilson(tn, tn + fp), phi=phi["estimate"], phi_lo=phi["lo"], phi_hi=phi["hi"],
                         jaccard=ST.pairwise_jaccard(lists), n_sets=d, universe=len(universe))
    pd.DataFrame({"sample": X.index, "y": y, "oof": oof, "label": lab}).to_csv(run.path("results", "nested_oof.csv"), index=False)
    pd.DataFrame({"fold": range(1, 6), "sets": [";".join(c) for c in lists]}).to_csv(run.path("results", "nested_sets.csv"), index=False)

    # ---------------------------------------------------------------- replicates
    reps = {"reselect": {}, "fixed": {}}; aucs = {"reselect": [], "fixed": []}; R_lists = {}
    for k, (tr, te) in enumerate(folds):
        for r in range(p["R"]):
            rng = np.random.default_rng(seed * 1000 + 100 * k + r)
            sub = stratified_subsample(tr, y, p["frac"], rng)
            cols = top_sets(S.iloc[sub], y[sub], p["k"]); R_lists.setdefault(k, []).append(cols)
            for arm, c in (("reselect", cols), ("fixed", lists[k])):
                sc = _fit(er, S.iloc[sub], y[sub], c).predict_proba(S.iloc[te][c].values)[:, 1]
                thr, _ = _inner_thr(er, S.iloc[sub], y[sub], c, seed)
                reps[arm].setdefault(k, []).append((sc, thr)); aucs[arm].append(PR.auc(y[te], sc))
    phis = [ST.nogueira_phi(R_lists[k], d) for k in R_lists]; Rn = p["R"]
    loo = [np.mean([ST.nogueira_phi([q for j, q in enumerate(R_lists[k]) if j != r], d) for k in R_lists]) for r in range(Rn)]
    se = float(np.sqrt((Rn - 1) / Rn * np.sum((np.array(loo) - np.mean(loo)) ** 2)))
    Pr = np.hstack([np.vstack([s for s, _ in reps["reselect"][k]]) for k in sorted(reps["reselect"])])
    Pf = np.hstack([np.vstack([s for s, _ in reps["fixed"][k]]) for k in sorted(reps["fixed"])])
    psi = AG.identity_share(Pr, Pf, seed=seed)
    sp = np.mean([AG.pairwise_spearman(np.vstack([s for s, _ in reps["reselect"][k]])) for k in reps["reselect"]])
    amb = np.mean([AG.ambiguity(np.vstack([s >= t for s, t in reps["reselect"][k]]).astype(int)) for k in reps["reselect"]])
    sd = np.mean([np.std(aucs["reselect"][k * Rn:(k + 1) * Rn], ddof=1) for k in range(5)])
    bag = np.zeros(len(y))
    for k, (tr, te) in enumerate(folds):
        bag[te] = np.vstack([s for s, _ in reps["reselect"][k]]).mean(0)
    out["replicates"] = dict(phi=float(np.mean(phis)), phi_lo=float(np.mean(phis) - 1.96 * se), phi_hi=float(np.mean(phis) + 1.96 * se),
                             jaccard=float(np.mean([ST.pairwise_jaccard(R_lists[k]) for k in R_lists])),
                             auc_mean=float(np.mean(aucs["reselect"])), auc_sd_within_fold=float(sd), spearman=float(sp),
                             ambiguity=float(amb), psi=psi["share"], psi_lo=psi["lo"], psi_hi=psi["hi"],
                             bagged_auc=PR.auc(y, bag))

    # ---------------------------------------------------------------- site (as for the gene panel)
    v = er.vumc; Sv, yv = S.loc[v], y[v]
    oofv = np.zeros(int(v.sum()))
    for a_, b_ in er.folds(yv):
        c = top_sets(Sv.iloc[a_], yv[a_], p["k"])
        oofv[b_] = _fit(er, Sv.iloc[a_], yv[a_], c).predict_proba(Sv.iloc[b_][c].values)[:, 1]
    av, lov, hiv, _ = T.delong_ci(yv, oofv)
    cv_ = top_sets(Sv, yv, p["k"])
    pt = _fit(er, Sv, yv, cv_).predict_proba(S.loc[~v][cv_].values)[:, 1]
    at, lot, hit, _ = T.delong_ci(y[~v], pt)
    out["site"] = dict(vumc_nested_auc=av, vumc_nested_ci=[lov, hiv], vumc_n=int(v.sum()),
                       transfer_auc=at, transfer_ci=[lot, hit], transfer_n=int((~v).sum()))
    run.log("site", vumc_nested=round(av, 4), transfer=round(at, 4))

    # ---------------------------------------------------------------- locked external model
    cols = top_sets(S, y, p["k"])
    m = _fit(er, S, y, cols)
    thr, _ = _inner_thr(er, S, y, cols, seed)
    pe = m.predict_proba(Se[cols].values)[:, 1]
    cal_e = PR.calibration(ye, pe); op = PR.operating_point(ye, pe, thr)
    out["locked"] = dict(auc=PR.auc(ye, pe), auc_ci=PR.stratified_bootstrap(ye, pe, PR.auc, B=2000, seed=seed),
                         **cal_e, int_ci=PR.stratified_bootstrap(ye, pe, lambda a_, b_: PR.calibration(a_, b_)["intercept"], B=2000, seed=seed),
                         slope_ci=PR.stratified_bootstrap(ye, pe, lambda a_, b_: PR.calibration(a_, b_)["slope"], B=2000, seed=seed),
                         brier_ci=PR.stratified_bootstrap(ye, pe, PR.brier, B=2000, seed=seed),
                         threshold=thr, sens=op["sens"], sens_ci=[op["sens_lo"], op["sens_hi"]], spec=op["spec"],
                         spec_ci=[op["spec_lo"], op["spec_hi"]], mean_pred=float(pe.mean()), prevalence=float(ye.mean()),
                         sets=cols)
    pd.DataFrame({"sample": Xe.index, "y": ye, "p": pe}).to_csv(run.path("results", "external_predictions.csv"), index=False)

    # ---------------------------------------------------------------- gene availability in the external cohort
    rng = np.random.default_rng(seed); rows = [dict(fraction=0.0, draw=0, auc=out["locked"]["auc"])]
    for f in p["missing_fractions"]:
        for i in range(p["missing_draws"]):
            keep = sorted(rng.choice(universe, int(round((1 - f) * len(universe))), replace=False))
            ks = set(keep)
            sub_sets = {c: [g for g in sets[c] if g in ks] for c in cols}
            if not all(sub_sets.values()):
                continue
            Sx = set_scores(Xe[keep], sub_sets)
            rows.append(dict(fraction=f, draw=i + 1, auc=PR.auc(ye, m.predict_proba(Sx[cols].values)[:, 1])))
    A = pd.DataFrame(rows); A.to_csv(run.path("results", "external_gene_availability.csv"), index=False)
    out["availability"] = A.groupby("fraction").auc.agg(["median", lambda v: v.quantile(.05), "min", "count"]).rename(
        columns={"<lambda_0>": "q05"}).reset_index().to_dict(orient="records")
    run.summary(out)
    run.log("pathway_done", nested_auc=round(a, 4), locked_auc=round(out["locked"]["auc"], 4), psi=round(psi["share"], 3))
