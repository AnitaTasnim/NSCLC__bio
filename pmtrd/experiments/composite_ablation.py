"""Ablation and weight uncertainty of the Stage-6 composite, inside the folds of the five-term run.

Stages 1-5 of every fold are taken from ``FR_nested.pkl`` (computed on T_k only), so the composite can be
re-weighted without re-running the selection. Scenarios:

    default      0.20 / 0.15 / 0.15 / 0.30 / (PPI 0) / 0.10
    drop_<term>  one term removed, the others unchanged (ranking is invariant to re-normalisation)
    equal        equal weights over the five active terms
    dirichlet    ``draws`` weight vectors from a flat Dirichlet over the five active terms

Reported: pooled out-of-fold AUC, mean Jaccard similarity of the fold panels with the default fold panels,
and Nogueira phi of the five fold panels.
"""
from __future__ import annotations

import pickle
from typing import Dict

import numpy as np
import pandas as pd

from ..config import resolve
from ..engine import Dataset, S, T
from ..metrics import predictive as PR
from ..metrics import stability as ST

DEFAULTS = dict(draws=200, fr_nested="data/reference/FR_nested.pkl")
TERMS = ("stat", "effect", "lasso", "boot", "prior")


def main(run, data: Dataset, params: Dict, mode: str = "all") -> None:
    import fold_recompute as FR   # engine module
    p = {**DEFAULTS, **(params or {})}
    X, y, er = data.X, data.y, data.run; d = X.shape[1]
    R = pickle.load(open(resolve(p["fr_nested"]), "rb"))
    W0 = dict(S.DEFAULT_WEIGHTS)

    def evaluate(W):
        oof = np.zeros(len(y)); panels = []
        for f in sorted(R):
            r = R[f]; g = FR.rescore(r["details"], r["fi"], None, weights=W)
            m = er.xgb().fit(X.iloc[r["train"]][g].values, y[r["train"]])
            oof[r["test"]] = m.predict_proba(X.iloc[r["test"]][g].values)[:, 1]
            panels.append(g)
        return oof, panels

    oof0, P0 = evaluate(W0)
    for f in sorted(R):
        if P0[f] != list(R[f]["panel"]):
            raise RuntimeError(f"default weights do not reproduce the five-term panel of fold {f + 1}")
    rows = []

    def record(name, W, draw=0):
        oof, panels = (oof0, P0) if name == "default" else evaluate(W)
        dl = T.delong_paired(y, oof, oof0) if name != "default" else {"diff": 0.0, "p": 1.0}
        rows.append(dict(scenario=name, draw=draw, auc=PR.auc(y, oof), delta_vs_default=dl["diff"], p=dl["p"],
                         jaccard_vs_default=float(np.mean([ST.jaccard(a, b) for a, b in zip(panels, P0)])),
                         phi=ST.nogueira_phi(panels, d), **{f"w_{t}": W[t] for t in TERMS}))

    record("default", W0)
    for t in TERMS:
        record(f"drop_{t}", {**W0, t: 0.0})
    record("equal", {**W0, **{t: 0.2 for t in TERMS}})
    rng = np.random.default_rng(run.cfg["seed"])
    for i in range(p["draws"]):
        w = rng.dirichlet(np.ones(len(TERMS)))
        record("dirichlet", {**W0, **dict(zip(TERMS, w))}, draw=i + 1)
    D = pd.DataFrame(rows); D.to_csv(run.path("results", "scenarios.csv"), index=False)
    dr = D[D.scenario == "dirichlet"]
    run.summary({"default_auc": float(D.iloc[0].auc), "default_phi": float(D.iloc[0].phi),
                 "single_term": D[D.scenario.str.startswith("drop_") | (D.scenario == "equal")]
                 .set_index("scenario")[["auc", "delta_vs_default", "p", "jaccard_vs_default", "phi"]].round(4).to_dict(orient="index"),
                 "dirichlet": {"auc_median": float(dr.auc.median()), "auc_q05": float(dr.auc.quantile(.05)),
                               "auc_q95": float(dr.auc.quantile(.95)), "auc_min": float(dr.auc.min()), "auc_max": float(dr.auc.max()),
                               "jaccard_median": float(dr.jaccard_vs_default.median()),
                               "jaccard_q05": float(dr.jaccard_vs_default.quantile(.05)),
                               "phi_median": float(dr.phi.median())}})
