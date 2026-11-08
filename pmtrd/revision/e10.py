"""E10 - external transfer of the panel, 18 versus 27 genes (GSE89843 -> 94 independent GSE68086 samples).

Plan: docs/plan_E8_E19.md, section E10 (with the amendment on the 18-gene model).
Run:  python -m pmtrd.revision.e10
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import out_dir, path, read_wide_parquet, write_meta

SEED = 10
NBOOT = 2000
NNULL = 1000
MEAS_FRAC = 0.10


class _XGBRun:
    """The parts of the engine's Run object used here (classifier factory), without loading the data twice."""

    def __init__(self, cfg):
        self.cfg = cfg

    def xgb(self, n_jobs=2):
        from xgboost import XGBClassifier
        return XGBClassifier(**dict(self.cfg["xgb_params"]), n_jobs=n_jobs, verbosity=0)

    def log(self, *a):
        pass


def external_data():
    """Discovery and external counts and labels, read directly (the engine loader needs about 4 GB), and
    log2-CPM over the genes quantified in both cohorts as in the paper's external analysis: shared genes with
    reads in both cohorts, plus the panel genes. Sample order and labels follow the engine's loader."""
    from types import SimpleNamespace
    from ..engine import T
    cfg = T.load_config(path("engine", "config.json"))
    meta = T._load_meta(path("data", "inputs", "GSE89843_meta.csv"), cfg)
    counts = read_wide_parquet(path("data", "inputs", "GSE89843_expr.parquet"))
    ids = [i for i in meta.index if i in counts.index]
    counts = counts.loc[ids].astype(float)
    y = T.parse_labels(meta.loc[ids, T._pick_col(meta, cfg.get("label_col"), ["label"])]).values.astype(int)
    X = counts.loc[:, (counts != 0).any(axis=0)]                 # the engine's zero-count filter
    me = T._load_meta(path("data", "inputs", "gse68086", "GSE68086_meta.csv"), cfg)
    Xe = read_wide_parquet(path("data", "inputs", "gse68086", "GSE68086_counts_symbols.parquet"))
    Xe = Xe.drop(columns=[c for c in Xe.columns if c == "__sym"])
    Xe = Xe.loc[[i for i in me.index if i in Xe.index]].astype(float)
    ye = T.parse_labels(me.loc[Xe.index, T._pick_col(me, cfg.get("label_col"), ["label"])]).values.astype(int)
    P27 = [g for g in cfg["panel"] if g in X.columns]
    C, Ce = counts, Xe
    shared = sorted(set(C.columns) & set(Ce.columns))
    C = C[shared].clip(lower=0).round().astype(np.int64)
    Ce = Ce[shared].clip(lower=0).round().astype(np.int64)
    ok = ((C.sum(0) > 0) & (Ce.sum(0) > 0)) | C.columns.isin(P27)
    C, Ce = C.loc[:, ok], Ce.loc[:, ok]
    d = SimpleNamespace(X=X, y=pd.Series(y, index=X.index), X_ext=Xe, y_ext=pd.Series(ye, index=Xe.index))
    return d, _XGBRun(cfg), P27, C, Ce, T.logcpm(C), T.logcpm(Ce)


def boot_ci(y, p, rng, B=NBOOT):
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(B):
        i = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        vals.append(roc_auc_score(y[i], p[i]))
    return np.quantile(vals, [0.025, 0.975])


def paired_diff(y, p1, p2, rng, B=NBOOT):
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(B):
        i = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        vals.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    vals = np.array(vals)
    return np.quantile(vals, [0.025, 0.975]), float(2 * min((vals <= 0).mean(), (vals >= 0).mean()))


def calib(y, p):
    import statsmodels.api as sm
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    slope = float(sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1])
    icpt = float(sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0])
    return icpt, slope, float(np.mean((p - y) ** 2))


def main() -> None:
    from .. import selectors as SEL
    od = out_dir("runs", "external")
    rng = np.random.default_rng(SEED)
    d, run, P27, C, Ce, Ld, Le = external_data()
    y, ye = d.y.values.astype(int), d.y_ext.values.astype(int)
    frac_nz = (Ce > 0).mean(0)
    any_read = [g for g in P27 if Ce[g].sum() > 0]
    meas = set(frac_nz.index[frac_nz >= MEAS_FRAC])
    P_meas = [g for g in P27 if g in meas]
    fit = lambda G, A, B_: run.xgb(n_jobs=2).fit(A[G].values, y).predict_proba(B_[G].values)[:, 1]  # noqa: E731

    preds, info = {}, {}
    preds["a1_locked27"] = fit(P27, Ld, Le)
    Le_imp = Le.copy()
    for g in P27:
        if g not in meas:
            Le_imp[g] = float(Ld[g].median())
    preds["a2_locked27_median_imputed"] = fit(P27, Ld, Le_imp)
    preds["b_retrained18_anyread"] = fit(any_read, Ld, Le)
    preds["b2_retrained_measurable10pct"] = fit(P_meas, Ld, Le)
    # (c) A4 re-derived on all 633 discovery samples, candidate genes restricted to measurable ones
    Xm = d.X[[g for g in d.X.columns if g in meas]]
    a4 = SEL.make("six_stage", B=500, n_jobs=2, seed=42).select(Xm, y, run)
    a4 = [g for g in a4 if g in Ld.columns]
    preds["c_A4_measurable"] = fit(a4, Ld, Le)
    info.update(n_shared=int(Ld.shape[1]), panel27=P27, any_read=any_read, measurable_panel=P_meas,
                A4_measurable_panel=a4, n_measurable=len(meas))
    # null: random panels of the size of (b), from the discovery Tier-2 pool restricted to measurable genes
    _, pool = SEL.tier2(d.X, y)
    pool = [g for g in pool.index if g in meas and g in Ld.columns]
    k = len(any_read)
    null = np.array([roc_auc_score(ye, fit(list(rng.choice(pool, k, replace=False)), Ld, Le)) for _ in range(NNULL)])
    pd.DataFrame({"null_auc": null}).to_csv(path(od, "E10_null.csv"), index=False)
    rows = []
    for name, p in preds.items():
        auc = roc_auc_score(ye, p)
        lo, hi = boot_ci(ye, p, rng)
        icpt, slope, brier = calib(ye, p)
        r = dict(model=name, n_genes={"a1_locked27": 27, "a2_locked27_median_imputed": 27,
                                       "b_retrained18_anyread": len(any_read),
                                       "b2_retrained_measurable10pct": len(P_meas), "c_A4_measurable": len(a4)}[name],
                 auc=auc, auc_lo=lo, auc_hi=hi, cal_intercept=icpt, cal_slope=slope, brier=brier)
        if name != "a1_locked27":
            (dl, dh), pp = paired_diff(ye, p, preds["a1_locked27"], rng)
            r.update(diff_vs_a1=auc - roc_auc_score(ye, preds["a1_locked27"]), diff_lo=dl, diff_hi=dh, diff_p=pp)
        if name in ("b_retrained18_anyread", "c_A4_measurable", "b2_retrained_measurable10pct"):
            r["null_p"] = float((1 + (null >= auc).sum()) / (NNULL + 1))
            r["null_median"] = float(np.median(null))
        rows.append(r)
        print(name, round(auc, 4), flush=True)
    pd.DataFrame(rows).to_csv(path(od, "E10.csv"), index=False)
    pd.DataFrame(preds, index=d.X_ext.index).assign(y=ye).to_csv(path(od, "E10_predictions.csv"))
    json.dump(info, open(path(od, "E10_info.json"), "w"), indent=1)
    write_meta(od, experiment="E10", seed=SEED, published_auc_27=0.8481, published_auc_18=0.8444)


if __name__ == "__main__":
    main()
