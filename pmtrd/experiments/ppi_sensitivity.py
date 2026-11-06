"""How much could the STRING PPI term, which cannot be recomputed here, change the nested estimate?

The PPI term gives each of the fold's 30 MRMR genes its STRING weighted degree among those genes,
divided by the maximum, with weight 0.10 in the composite. Its only data-dependent input, the MRMR list,
was computed inside each training fold of the five-term run (``FR_nested.pkl``). Without STRING access we
bound its effect by re-scoring Stage 6 of every fold under four scenarios:

    none         the five-term analysis (PPI omitted)
    empirical    the PPI values observed for the published panel's MRMR genes (three non-zero values:
                 1.00, 0.61 and 0.43; the rest 0) assigned at random to the fold's MRMR genes; ``draws`` draws
    all_max      every MRMR gene scores 1 (the largest possible uniform shift)
    adversarial  every MRMR gene NOT already in the fold's five-term panel scores 1 (the largest possible
                 displacement of the panel)

For each scenario the fold panels are re-scored, the classifier re-trained on T_k and the held-out fold
scored; we report the pooled out-of-fold AUC and the Jaccard similarity with the five-term fold panels.
"""
from __future__ import annotations

import pickle
from typing import Dict

import numpy as np
import pandas as pd

from ..config import resolve
from ..engine import Dataset, T
from ..metrics import predictive as PR
from ..metrics import stability as ST

DEFAULTS = dict(draws=100, fr_nested="data/reference/FR_nested.pkl", empirical=[1.0, 0.609989, 0.42508])


def _rescore(r, ppi):
    import fold_recompute as FR   # engine module
    return FR.rescore(r["details"], r["fi"], ppi)


def main(run, data: Dataset, params: Dict, mode: str = "all") -> None:
    p = {**DEFAULTS, **(params or {})}
    X, y, er = data.X, data.y, data.run
    R = pickle.load(open(resolve(p["fr_nested"]), "rb"))
    base_panels = {f: _rescore(R[f], None) for f in R}
    for f in R:
        if base_panels[f] != list(R[f]["panel"]):
            raise RuntimeError(f"re-scoring does not reproduce the five-term panel of fold {f + 1}")
    rng = np.random.default_rng(run.cfg["seed"])

    def evaluate(ppi_by_fold):
        oof = np.zeros(len(y)); jac = []
        for f in sorted(R):
            r = R[f]; g = _rescore(r, ppi_by_fold[f])
            m = er.xgb().fit(X.iloc[r["train"]][g].values, y[r["train"]])
            oof[r["test"]] = m.predict_proba(X.iloc[r["test"]][g].values)[:, 1]
            jac.append(ST.jaccard(g, base_panels[f]))
        return PR.auc(y, oof), float(np.mean(jac)), oof

    rows = []
    a0, j0, oof0 = evaluate({f: None for f in R}); rows.append(dict(scenario="none", draw=0, auc=a0, jaccard_vs_five_term=j0))
    a, j, _ = evaluate({f: {g: 1.0 for g in R[f]["mrmr"]} for f in R}); rows.append(dict(scenario="all_max", draw=0, auc=a, jaccard_vs_five_term=j))
    a, j, oof_adv = evaluate({f: {g: (0.0 if g in base_panels[f] else 1.0) for g in R[f]["mrmr"]} for f in R})
    rows.append(dict(scenario="adversarial", draw=0, auc=a, jaccard_vs_five_term=j))
    for d in range(p["draws"]):
        ppi = {}
        for f in R:
            vals = list(p["empirical"]) + [0.0] * (len(R[f]["mrmr"]) - len(p["empirical"]))
            ppi[f] = dict(zip(R[f]["mrmr"], rng.permutation(vals)))
        a, j, _ = evaluate(ppi); rows.append(dict(scenario="empirical", draw=d + 1, auc=a, jaccard_vs_five_term=j))
    D = pd.DataFrame(rows); D.to_csv(run.path("results", "scenarios.csv"), index=False)
    emp = D[D.scenario == "empirical"]
    dl = T.delong_paired(y, oof_adv, oof0)
    summ = {"none_auc": a0,
            "all_max": D[D.scenario == "all_max"].iloc[0][["auc", "jaccard_vs_five_term"]].to_dict(),
            "adversarial": {**D[D.scenario == "adversarial"].iloc[0][["auc", "jaccard_vs_five_term"]].to_dict(),
                            "delta_vs_none": dl["diff"], "p": dl["p"]},
            "empirical": {"auc_median": float(emp.auc.median()), "auc_min": float(emp.auc.min()), "auc_max": float(emp.auc.max()),
                          "jaccard_median": float(emp.jaccard_vs_five_term.median()),
                          "jaccard_min": float(emp.jaccard_vs_five_term.min())},
            "mrmr_in_panel": {int(f) + 1: len(set(R[f]["mrmr"]) & set(base_panels[f])) for f in R}}
    run.summary(summ); run.log("ppi_done", **{k: v for k, v in summ.items() if k != "mrmr_in_panel"})
