"""E8 - size- and classifier-matched benchmark on the paper's outer folds (GSE89843).

Plan: docs/plan_E8_E19.md, section E8.   Run:  python -m pmtrd.revision.e8
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests

from .. import selectors as SEL
from ..config import resolve
from ..experiments.common import check_folds, reference_fold_panels
from .common import out_dir, path, saved, write_meta
from .units import Cache, load_dataset

PRIMARY = ["mw_topk@27/xgb", "l1_topk@27/xgb", "stability@27/xgb", "mrmr@27/xgb", "mrmr@own/svm"]


def mrmr_ranking(X, y, run):
    """The ranking and IFS size of the engine's mRMR-IFS-SVM (same code path, ranking kept)."""
    T = __import__("tep_analysis")
    sc = run.cfg["sheng"]
    L = run.logX(X)
    L = L.loc[:, L.var(axis=0) > 0]
    if L.shape[1] > sc["prefilter_n"]:
        L = L[L.mean(0).sort_values(ascending=False).index[: sc["prefilter_n"]]]
    order = T.mrmr_rank(L.values, np.asarray(y).astype(int), K=sc["mrmr_K"], scheme=sc["mrmr_scheme"],
                        alpha=sc["discretize_alpha"])
    ranked = list(L.columns[order])
    folds = list(StratifiedKFold(sc["inner_folds"], shuffle=True, random_state=run.cfg["seed"]).split(L, y))
    curve = []
    for k in range(1, min(sc["ifs_max"], len(ranked)) + 1, sc["ifs_step"]):
        Xk = L[ranked[:k]].values
        yhat = np.zeros(len(y), int)
        for tr, te in folds:
            m = make_pipeline(StandardScaler(), SVC(kernel=sc["svm_kernel"], C=sc["svm_C"])).fit(Xk[tr], y[tr])
            yhat[te] = m.predict(Xk[te])
        curve.append((k, matthews_corrcoef(y, yhat)))
    curve = pd.DataFrame(curve, columns=["k", "mcc"])
    kbest = int(curve.sort_values(["mcc", "k"], ascending=[False, True]).iloc[0].k)
    return ranked, kbest


def svm_scores(Xtr, ytr, Xte, genes, run):
    sc = run.cfg["sheng"]
    m = make_pipeline(StandardScaler(), SVC(kernel=sc["svm_kernel"], C=sc["svm_C"]))
    m.fit(run.logX(Xtr)[genes].values, ytr)
    return m.decision_function(run.logX(Xte)[genes].values)


def xgb_scores(Xtr, ytr, Xte, genes, run):
    return run.xgb(n_jobs=2).fit(Xtr[genes].values, ytr).predict_proba(Xte[genes].values)[:, 1]


def main() -> None:
    od = out_dir("runs", "revision", "E8")
    cache = Cache(path(od, "cache"))
    X, y, run, _ = load_dataset("GSE89843", 2)
    folds = run.folds()
    check_folds(folds, "data/reference/R8_outer_folds.csv", X.index)
    a4 = reference_fold_panels("data/reference/R4_nested_folds.csv")
    six = pd.read_csv(resolve("data/reference/R4_nested_oof.csv"))
    assert list(six["sample"]) == list(X.index)
    n = len(y)
    arms = {}
    for k, (tr, te) in enumerate(folds):
        key = f"fold{k}"
        if not cache.has(key):
            Xtr, ytr, Xte = X.iloc[tr], y[tr], X.iloc[te]
            ranked, kbest = mrmr_ranking(Xtr, ytr, run)
            panels = {"A4@own": a4[k], "mrmr@own": ranked[:kbest], "mrmr@27": ranked[:27]}
            for m in ("mw_topk", "l1_topk", "stability"):
                panels[f"{m}@27"] = SEL.make(m, k=27, n_jobs=2, seed=42).select(Xtr, ytr, run)
                panels[f"{m}@mrmrsize"] = SEL.make(m, k=kbest, n_jobs=2, seed=42).select(Xtr, ytr, run)
            sc = {}
            for name, g in panels.items():
                sc[f"{name}/xgb"] = xgb_scores(Xtr, ytr, Xte, g, run)
                sc[f"{name}/svm"] = svm_scores(Xtr, ytr, Xte, g, run)
            cache.put(key, dict(te=te, panels=panels, scores=sc, kbest=kbest))
            print("fold", k + 1, "kbest", kbest, flush=True)
        u = cache.get(key)
        for name, s in u["scores"].items():
            arms.setdefault(name, np.full(n, np.nan))[u["te"]] = s
    T = __import__("tep_analysis")
    # check: A4/xgb recomputed from the fold panels equals the paper's reference out-of-fold predictions
    ref = six.oof_prob.values
    check = float(np.max(np.abs(arms["A4@own/xgb"] - ref)))
    rows = []
    for name, oof in arms.items():
        a, lo, hi, _ = T.delong_ci(y, oof)
        dl = T.delong_paired(y, ref, oof) if name != "A4@own/xgb" else None
        rows.append(dict(arm=name, auc=a, lo=lo, hi=hi,
                         sizes=";".join(str(len(cache.get(f"fold{k}")["panels"][name.split("/")[0]])) for k in range(5)),
                         diff_A4xgb_minus_arm=None if dl is None else dl["diff"],
                         diff_lo=None if dl is None else dl["lo95"], diff_hi=None if dl is None else dl["hi95"],
                         p=None if dl is None else dl["p"], primary=name in PRIMARY))
    D = pd.DataFrame(rows).sort_values("auc", ascending=False)
    prim = D[D.primary].copy()
    prim["p_holm"] = multipletests(prim.p.astype(float), method="holm")[1]
    D = D.merge(prim[["arm", "p_holm"]], on="arm", how="left")
    # Phi and psi from the saved replicate study where the arm matches it (k = 27, XGBoost)
    st = saved("GSE89843", "stability.csv").set_index("method")
    ish = saved("GSE89843", "identity_share.csv").set_index("method")
    match = {"A4@own/xgb": "six_stage", "mw_topk@27/xgb": "mw_topk", "l1_topk@27/xgb": "l1_topk",
             "stability@27/xgb": "stability", "mrmr@own/svm": "mrmr_ifs_svm"}
    D["replicate_phi"] = D.arm.map(lambda a: st.phi.get(match.get(a), np.nan))
    D["replicate_psi"] = D.arm.map(lambda a: ish.share.get(match.get(a), np.nan))
    D.to_csv(path(od, "E8.csv"), index=False)
    write_meta(od, experiment="E8", a4_reference_check_max_abs_diff=check,
               kbest=[cache.get(f"fold{k}")["kbest"] for k in range(5)])
    print("A4 reference check (max |diff|):", check)


if __name__ == "__main__":
    main()
