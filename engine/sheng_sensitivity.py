#!/usr/bin/env python3
"""Sensitivity of the re-implemented comparator (Sheng et al., mRMR-IFS-SVM) to its unpublished
implementation choices, on the same outer folds as the head-to-head comparison.

    python sheng_sensitivity.py config.json run
    python sheng_sensitivity.py config.json summarise

One setting is changed at a time from the base configuration used in the paper (expression pre-filter
to 4,722 genes, mRMR MID on data discretised at mean +/- 0.5 SD, IFS over 1-150 genes chosen by
inner five-fold MCC, linear SVM with C = 1). The mRMR ranking of each training fold is computed once
per (discretisation, scheme) pair and shared by the SVM variants.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tep_analysis as T  # noqa: E402

VARIANTS = {
    "base": {},
    "C = 0.1": dict(svm_C=0.1),
    "C = 10": dict(svm_C=10.0),
    "RBF kernel": dict(svm_kernel="rbf"),
    "discretisation mean +/- 1 SD": dict(discretize_alpha=1.0),
    "mRMR MIQ scheme": dict(mrmr_scheme="MIQ"),
    "IFS chosen by accuracy": dict(ifs_metric="accuracy"),
    "IFS limited to 1-50 genes": dict(ifs_max=50),
}


def ranking(L, y, sc):
    order = T.mrmr_rank(L.values, y, K=sc["mrmr_K"], scheme=sc["mrmr_scheme"], alpha=sc["discretize_alpha"])
    return list(L.columns[order])


def fit_variant(L, y, ranked, sc, seed):
    folds = list(StratifiedKFold(sc["inner_folds"], shuffle=True, random_state=seed).split(L, y))
    best = (-np.inf, None)
    for k in range(1, min(sc["ifs_max"], len(ranked)) + 1, sc["ifs_step"]):
        Xk = L[ranked[:k]].values; yhat = np.zeros(len(y), int)
        for tr, te in folds:
            m = make_pipeline(StandardScaler(), SVC(kernel=sc["svm_kernel"], C=sc["svm_C"])).fit(Xk[tr], y[tr])
            yhat[te] = m.predict(Xk[te])
        val = matthews_corrcoef(y, yhat) if sc["ifs_metric"] == "mcc" else (yhat == y).mean()
        if val > best[0] + 1e-12:
            best = (val, k)
    genes = ranked[:best[1]]
    final = make_pipeline(StandardScaler(), SVC(kernel=sc["svm_kernel"], C=sc["svm_C"])).fit(L[genes].values, y)
    return genes, final


def main():
    cfg = T.load_config(sys.argv[1]); what = sys.argv[2]
    out = cfg["out_dir"]; cdir = os.path.join(out, "cache"); tdir = os.path.join(out, "tables")
    os.makedirs(cdir, exist_ok=True); os.makedirs(tdir, exist_ok=True); os.makedirs(os.path.join(out, "macros"), exist_ok=True)
    d = T.load_data(cfg); run = T.Run(cfg, d)
    y = d.y.values; folds = run.folds(); seed = cfg["seed"]
    part = os.path.join(cdir, "SH_sens.pkl")
    R = pickle.load(open(part, "rb")) if os.path.exists(part) else {}
    if what == "run":
        ranks = {}
        for name, change in VARIANTS.items():
            if name in R:
                continue
            sc = dict(cfg["sheng"]); sc.update(change)
            t0 = time.time(); oof = np.full(len(y), np.nan); yhat = np.zeros(len(y), int); sizes = []
            for f, (tr, te) in enumerate(folds):
                Ltr = run.logX(d.X.iloc[tr]); Ltr = Ltr.loc[:, Ltr.var(axis=0) > 0]
                keep = Ltr.mean(0).sort_values(ascending=False).index[: sc["prefilter_n"]]
                Ltr = Ltr[keep]
                key = (f, sc["discretize_alpha"], sc["mrmr_scheme"])
                if key not in ranks:
                    ranks[key] = ranking(Ltr, y[tr], sc)
                genes, mdl = fit_variant(Ltr, y[tr], ranks[key], sc, seed)
                Lte = run.logX(d.X.iloc[te])[genes].values
                oof[te] = mdl.decision_function(Lte); yhat[te] = mdl.predict(Lte); sizes.append(len(genes))
            R[name] = dict(oof=oof, yhat=yhat, sizes=sizes, seconds=time.time() - t0)
            pickle.dump(R, open(part, "wb"))
            run.log(f"  [{name}] AUC {roc_auc_score(y, oof):.4f}, sizes {sizes} ({time.time() - t0:.0f}s)")
    else:
        six = pd.read_csv(os.path.join(os.path.dirname(HERE), "results", "tables", "R4_nested_oof.csv"))["oof_prob"].values
        rows = []
        for name in VARIANTS:
            if name not in R:
                continue
            r = R[name]; a, lo, hi, _ = T.delong_ci(y, r["oof"]); dl = T.delong_paired(y, six, r["oof"])
            rows.append(dict(variant=name, auc=a, lo=lo, hi=hi, mcc=matthews_corrcoef(y, r["yhat"]),
                             balanced_accuracy=balanced_accuracy_score(y, r["yhat"]),
                             panel_sizes=";".join(map(str, r["sizes"])),
                             delta_six_minus_variant=dl["diff"], delta_lo=dl["lo95"], delta_hi=dl["hi95"], p=dl["p"]))
        D = pd.DataFrame(rows); D.to_csv(os.path.join(tdir, "SH_comparator_sensitivity.csv"), index=False)
        M = {"ShSensN": str(len(D)), "ShSensMin": T.f4(D.auc.min()), "ShSensMax": T.f4(D.auc.max()),
             "ShSensAllHigher": str(int((D.delta_six_minus_variant < 0).sum())),
             "ShSensSigHigher": str(int(((D.delta_six_minus_variant < 0) & (D.p < 0.05)).sum())),
             "ShSensMinVariant": T.tex_escape(D.sort_values("auc").variant.iloc[0]),
             "ShSensDeltaRange": f"{-100 * D.delta_six_minus_variant.max():.2f}--{-100 * D.delta_six_minus_variant.min():.2f}",
             "ShSensSizeRange": f"{min(min(R[v]['sizes']) for v in D.variant)}--{max(max(R[v]['sizes']) for v in D.variant)}",
             "ShSensMaxP": T.pfmt(D.p.max())}
        json.dump(M, open(os.path.join(out, "macros", "shsens.json"), "w"), indent=1)
        lines = [r"\begin{tabular}{lccc}", r"\toprule",
                 r"Comparator setting & Nested AUC (95\% CI) & Panel sizes & $\Delta$AUC, pp ($p$) \\",
                 r"\midrule"]
        for r in rows:
            lines.append(f"{T.tex_escape(r['variant'])} & {T.f4(r['auc'])} ({T.f4(r['lo'])}--{T.f4(r['hi'])}) & "
                         f"{r['panel_sizes'].replace(';', ', ')} & {T.pp(r['delta_six_minus_variant'])} ({T.pfmt(r['p'])}) \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(tdir, "SH_comparator_sensitivity.tex"), "w").write("\n".join(lines) + "\n")
        print(json.dumps(M, indent=1)); print(D.to_string(index=False))


if __name__ == "__main__":
    main()
