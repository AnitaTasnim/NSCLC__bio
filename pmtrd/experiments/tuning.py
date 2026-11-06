"""Hyperparameter sensitivity of the nested estimate: XGBoost tuned inside each outer training fold.

The paper uses fixed XGBoost settings whose origin is not documented. Here, for the four-term fold panel of
each outer fold, an inner five-fold CV on T_k chooses among ``grid`` settings by AUC; the chosen model is
re-trained on T_k and scores H_k. The pooled out-of-fold AUC with tuning is compared with the fixed-setting
AUC (paired DeLong). The held-out fold is never used for the choice.
"""
from __future__ import annotations

import itertools
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from ..config import resolve
from ..engine import Dataset, T
from .common import check_folds, reference_fold_panels

DEFAULTS = dict(grid={"n_estimators": [100, 300, 600], "max_depth": [2, 4, 6], "learning_rate": [0.03, 0.1]},
                folds_file="data/reference/R8_outer_folds.csv", fold_panels_file="data/reference/R4_nested_folds.csv",
                six_oof="data/reference/R4_nested_oof.csv")


def main(run, data: Dataset, params: Dict, mode: str = "all") -> None:
    p = {**DEFAULTS, **(params or {})}
    X, y, er = data.X, data.y, data.run
    folds = er.folds(); check_folds(folds, p["folds_file"], X.index)
    panels = reference_fold_panels(p["fold_panels_file"])
    base = dict(er.cfg["xgb_params"]); nj = run.cfg.get("n_jobs", 2)
    keys = sorted(p["grid"]); combos = [dict(zip(keys, v)) for v in itertools.product(*(p["grid"][k] for k in keys))]
    oof = np.zeros(len(y)); rows = []
    for k, (tr, te) in enumerate(folds):
        g = panels[k]; Xtr, ytr = X.iloc[tr][g].values, y[tr]
        inner = list(StratifiedKFold(5, shuffle=True, random_state=run.cfg["seed"]).split(Xtr, ytr))
        scores = []
        for c in combos:
            prm = {**base, **c}; o = np.zeros(len(ytr))
            for a, b in inner:
                o[b] = XGBClassifier(**prm, n_jobs=nj, verbosity=0).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
            scores.append(roc_auc_score(ytr, o))
        best = combos[int(np.argmax(scores))]
        m = XGBClassifier(**{**base, **best}, n_jobs=nj, verbosity=0).fit(Xtr, ytr)
        oof[te] = m.predict_proba(X.iloc[te][g].values)[:, 1]
        default_inner = scores[combos.index({k2: base[k2] for k2 in keys})] if {k2: base[k2] for k2 in keys} in combos else np.nan
        rows.append(dict(fold=k + 1, chosen=best, inner_auc_best=max(scores), inner_auc_default=default_inner,
                         heldout_auc=roc_auc_score(y[te], oof[te])))
        run.log("fold", fold=k + 1, chosen=best, inner_best=round(max(scores), 4))
    ref = pd.read_csv(resolve(p["six_oof"])).oof_prob.values
    a, lo, hi, _ = T.delong_ci(y, oof); dl = T.delong_paired(y, oof, ref)
    pd.DataFrame(rows).to_csv(run.path("results", "folds.csv"), index=False)
    pd.DataFrame({"sample": X.index, "y": y, "oof_tuned": oof, "oof_fixed": ref}).to_csv(run.path("results", "oof.csv"), index=False)
    run.summary({"tuned_auc": a, "tuned_ci": [lo, hi], "fixed_auc": T.delong_ci(y, ref)[0],
                 "delta_tuned_minus_fixed": dl["diff"], "delta_ci": [dl["lo95"], dl["hi95"]], "p": dl["p"],
                 "chosen": [r["chosen"] for r in rows]})
