"""Predictive reliability inside the discovery cohort: calibration and operating characteristics of the
nested-CV models, with every threshold learned from training data only.

For each outer fold k and each version (fixed panel; four-term and five-term fold panels), the classifier
is re-trained on T_k exactly as in the paper (which reproduces the paper's out-of-fold predictions; this is
checked), a Youden threshold is learned by inner five-fold CV on T_k, and the held-out fold H_k is scored.
Reported per fold and pooled: calibration intercept and slope, Brier score, sensitivity and specificity
(Wilson CIs) at the training-derived threshold.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..config import resolve
from ..engine import Dataset
from ..metrics import predictive as PR
from .common import check_folds, reference_fold_panels

DEFAULTS = dict(folds_file="data/reference/R8_outer_folds.csv", fold_panels_file="data/reference/R4_nested_folds.csv",
                five_folds="data/reference/FR_nested_folds.csv", six_oof="data/reference/R4_nested_oof.csv")


def _inner_thr(er, Xtr, ytr, seed):
    oof = np.zeros(len(ytr))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, ytr):
        oof[te] = er.xgb().fit(Xtr[tr], ytr[tr]).predict_proba(Xtr[te])[:, 1]
    return PR.youden_threshold(ytr, oof)


def main(run, data: Dataset, params: Dict, mode: str = "all") -> None:
    p = {**DEFAULTS, **(params or {})}
    X, y, er = data.X, data.y, data.run
    folds = er.folds(); check_folds(folds, p["folds_file"], X.index)
    four = reference_fold_panels(p["fold_panels_file"])
    five = {int(r.fold) - 1: r.panel.split(";") for r in pd.read_csv(resolve(p["five_folds"])).itertuples()}
    fixed = er.panel_in(X)
    versions = {"fixed_panel": {k: fixed for k in range(5)}, "four_term": four, "five_term": five}
    rows, pooled = [], {}
    six_ref = pd.read_csv(resolve(p["six_oof"])).oof_prob.values
    for v, panels in versions.items():
        oof = np.zeros(len(y)); lab = np.zeros(len(y), int)
        for k, (tr, te) in enumerate(folds):
            g = panels[k]
            m = er.xgb().fit(X.iloc[tr][g].values, y[tr])
            oof[te] = m.predict_proba(X.iloc[te][g].values)[:, 1]
            thr = _inner_thr(er, X.iloc[tr][g].values, y[tr], run.cfg["seed"])
            lab[te] = oof[te] >= thr
            c = PR.calibration(y[te], oof[te]); op = PR.operating_point(y[te], oof[te], thr)
            rows.append(dict(version=v, fold=k + 1, auc=PR.auc(y[te], oof[te]), **c,
                             **{q: op[q] for q in ("threshold", "sens", "spec")}))
            run.log("fold", version=v, fold=k + 1, auc=round(rows[-1]["auc"], 4), thr=round(thr, 3))
        if v == "four_term":
            dev = float(np.max(np.abs(oof - six_ref)))
            run.log("reproduction_check", max_abs_diff_vs_paper=dev)
            if dev > 1e-6:
                raise RuntimeError(f"four-term predictions differ from the paper's by {dev}")
        c = PR.calibration(y, oof)
        tp = int(((lab == 1) & (y == 1)).sum()); fn = int(((lab == 0) & (y == 1)).sum())
        tn = int(((lab == 0) & (y == 0)).sum()); fp = int(((lab == 1) & (y == 0)).sum())
        sl, sh = PR.wilson(tp, tp + fn); pl, ph = PR.wilson(tn, tn + fp)
        pooled[v] = dict(auc=PR.auc(y, oof), **c, sens=tp / (tp + fn), sens_lo=sl, sens_hi=sh,
                         spec=tn / (tn + fp), spec_lo=pl, spec_hi=ph,
                         int_ci=PR.stratified_bootstrap(y, oof, lambda a, b: PR.calibration(a, b)["intercept"], B=1000),
                         slope_ci=PR.stratified_bootstrap(y, oof, lambda a, b: PR.calibration(a, b)["slope"], B=1000),
                         brier_ci=PR.stratified_bootstrap(y, oof, PR.brier, B=1000))
    F = pd.DataFrame(rows); F.to_csv(run.path("results", "folds.csv"), index=False)
    agg = F.groupby("version").agg(**{f"{c}_{s}": (c, s) for c in ("auc", "intercept", "slope", "brier", "sens", "spec")
                                      for s in ("mean", "std", "min", "max")}).reset_index()
    agg.to_csv(run.path("results", "fold_spread.csv"), index=False)
    pd.DataFrame(pooled).T.to_csv(run.path("results", "pooled.csv"))
    run.summary({"pooled": pooled, "fold_spread": agg.set_index("version").round(4).to_dict(orient="index")})
