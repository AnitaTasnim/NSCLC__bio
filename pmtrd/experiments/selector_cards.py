"""Technical-reliability card for every selector: does a stable or accurate panel also transfer?

For each selector, the panel chosen on all 633 discovery samples is audited on the technical dimension
with the paper's protocols:
    site_dominant   fraction of panel genes whose site effect is at least their disease effect
                    (genome-wide table of the paper, ``R1_genomewide_site_vs_disease.csv``)
    vumc_sig        fraction of panel genes differentially expressed inside Vumc (genome-wide q < 0.05)
    vumc_nested     nested five-fold CV inside Vumc, selection re-run in every training fold
    transfer        panel selected and model trained on Vumc only, applied to all other site codes
    coverage        fraction of panel genes with reads in GSE68086
    external        locked model on log2-CPM of the panel genes, trained on all 633 samples and applied to
                    GSE68086 without alignment: AUC (stratified bootstrap CI) and calibration intercept.
                    The paper's XGBoost classifier is used for every panel (also for the mRMR-IFS-SVM panel),
                    so that only the gene list differs.
"Six-stage (paper)" uses the published 27-gene panel for the full-cohort and external analyses and the
four-term procedure for the within-Vumc analyses, as in the paper.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .. import selectors as SEL
from ..config import resolve
from ..engine import Dataset, T
from ..metrics import predictive as PR

DEFAULTS = dict(methods=["six_stage", "mw_topk", "l1_topk", "stability", "mrmr_ifs_svm"], k=27, B=500,
                genomewide="data/reference/R1_genomewide_site_vs_disease.csv",
                external_expr="data/inputs/gse68086/GSE68086_counts_symbols.parquet",
                external_meta="data/inputs/gse68086/GSE68086_meta.csv")


def main(run, data: Dataset, params: Dict, mode: str = "all") -> None:
    p = {**DEFAULTS, **(params or {})}
    X, y, er, seed = data.X, data.y, data.run, run.cfg["seed"]; nj = run.cfg.get("n_jobs", 2)
    gw = pd.read_csv(resolve(p["genomewide"])).set_index("gene")
    Ce = pd.read_parquet(resolve(p["external_expr"])).drop(columns="__sym", errors="ignore")
    me = pd.read_csv(resolve(p["external_meta"])).set_index("sample"); Ce = Ce.loc[me.index]; ye = me.label.values.astype(int)
    shared = sorted(set(X.columns) & set(Ce.columns))
    C = X[shared].clip(lower=0).round(); Cx = Ce[shared].clip(lower=0).round()
    # library sizes as in the paper's external analysis: genes with reads in both cohorts plus the panel
    base = ((C.sum(axis=0) > 0) & (Cx.sum(axis=0) > 0)) | C.columns.isin(er.panel_in(X))
    libd = C.loc[:, base].sum(axis=1).replace(0, 1); libe = Cx.loc[:, base].sum(axis=1).replace(0, 1)
    Ld = np.log2(C.div(libd, axis=0) * 1e6 + 1.0); Le = np.log2(Cx.div(libe, axis=0) * 1e6 + 1.0)
    v = er.vumc; Xv, yv = X.loc[v], y[v]
    rows = []
    for m in p["methods"]:
        sel = SEL.make(m, k=p["k"], B=p["B"], n_jobs=nj, seed=seed)
        full = er.panel_in(X) if m == "six_stage" else sel.select(X, y, er)
        # ---- within Vumc: nested CV and transfer to the other site codes
        oofv = np.zeros(int(v.sum()))
        for a, b in er.folds(yv):
            g = sel.select(Xv.iloc[a], yv[a], er)
            oofv[b] = sel.fit_predict(Xv.iloc[a], yv[a], Xv.iloc[b], g, er)
        av, lov, hiv, _ = T.delong_ci(yv, oofv)
        gv = sel.select(Xv, yv, er)
        pt = sel.fit_predict(Xv, yv, X.loc[~v], gv, er)
        at, lot, hit, _ = T.delong_ci(y[~v], pt)
        # ---- external, locked, log2-CPM, no alignment
        G = [g for g in full if g in shared]
        measurable = [g for g in G if Cx[g].sum() > 0]
        mdl = er.xgb().fit(Ld[G].values, y)
        pe = mdl.predict_proba(Le[G].values)[:, 1]
        ae = PR.auc(ye, pe); lo, hi = PR.stratified_bootstrap(ye, pe, PR.auc, B=2000, seed=seed)
        cal = PR.calibration(ye, pe)
        dom = gw.reindex(full).dominant
        rows.append(dict(method=m, n_genes=len(full), site_dominant=float((dom == "site").mean()),
                         vumc_sig=float((gw.reindex(full).q_vumc_genomewide < 0.05).mean()),
                         vumc_nested_auc=av, vumc_lo=lov, vumc_hi=hiv, transfer_auc=at, transfer_lo=lot, transfer_hi=hit,
                         coverage=len(measurable) / len(full), external_auc=ae, external_lo=lo, external_hi=hi,
                         external_intercept=cal["intercept"], external_slope=cal["slope"], panel=";".join(full)))
        run.log("card", method=m, vumc=round(av, 4), transfer=round(at, 4), external=round(ae, 4),
                site_dominant=round(rows[-1]["site_dominant"], 3), coverage=round(rows[-1]["coverage"], 3))
    D = pd.DataFrame(rows); D.to_csv(run.path("results", "cards.csv"), index=False)
    run.summary({"cards": D.drop(columns="panel").set_index("method").round(4).to_dict(orient="index")})
