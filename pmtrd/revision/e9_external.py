"""E9, external part - a 25-member A4 ensemble trained on 80% subsamples of all 633 discovery samples and applied
to the 94 GSE68086 samples, against the locked 27-gene model (same log2-CPM matrix as E10).

Plan: docs/plan_E8_E19.md, section E9 (with the amendment on the external part).
Run:  python -m pmtrd.revision.e9_external run      then   python -m pmtrd.revision.e9_external summarise
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import out_dir, path, write_meta
from .e10 import boot_ci, calib, external_data, paired_diff
from .units import Cache, subsample

NREP = 25
FOLD_INDEX = 9          # the paper's subsample seed scheme at a fold index outside 0-4
SEED = 9


def run() -> None:
    from .. import selectors as SEL
    cache = Cache(path("runs", "bagging", "cache", "external"))
    d, xgbrun, P27, C, Ce, Ld, Le = external_data()
    y = d.y.values.astype(int)
    n = len(y)
    for r in range(NREP):
        key = f"member_r{r:02d}"
        if cache.has(key):
            continue
        sub = subsample(np.arange(n), y, 42, FOLD_INDEX, r)
        genes = SEL.make("six_stage", B=500, n_jobs=1, seed=42).select(d.X.iloc[sub], y[sub], xgbrun)
        G = [g for g in genes if g in Ld.columns]
        m = xgbrun.xgb(n_jobs=1).fit(Ld.iloc[sub][G].values, y[sub])
        p = m.predict_proba(Le[G].values)[:, 1]
        cache.put(key, dict(sub=sub, genes=list(genes), used=G, dropped=[g for g in genes if g not in G], scores=p))
        print(key, len(genes), len(G), round(roc_auc_score(d.y_ext.values.astype(int), p), 4), flush=True)


def summarise() -> None:
    od = out_dir("runs", "bagging")
    cache = Cache(path("runs", "bagging", "cache", "external"))
    pr = pd.read_csv(path("runs", "external", "E10_predictions.csv"), index_col=0)
    ye = pr["y"].values.astype(int)
    locked = pr["a1_locked27"].values
    ms = [cache.get(f"member_r{r:02d}") for r in range(NREP)]
    S = np.vstack([m["scores"] for m in ms])
    ens = S.mean(0)
    rng = np.random.default_rng(SEED)
    member_auc = np.array([roc_auc_score(ye, s) for s in S])
    # paired bootstrap of the gain (ensemble AUC - mean member AUC), stratified
    i1, i0 = np.flatnonzero(ye == 1), np.flatnonzero(ye == 0)
    gains = []
    for _ in range(2000):
        i = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        gains.append(roc_auc_score(ye[i], ens[i]) - np.mean([roc_auc_score(ye[i], s[i]) for s in S]))
    gains = np.array(gains)
    lo, hi = boot_ci(ye, ens, rng)
    (dl, dh), dp = paired_diff(ye, ens, locked, rng)
    icpt, slope, brier = calib(ye, ens)
    li, ls, lb = calib(ye, locked)
    rows = [dict(model="ensemble25_A4", auc=roc_auc_score(ye, ens), auc_lo=lo, auc_hi=hi,
                 mean_member_auc=member_auc.mean(), member_auc_min=member_auc.min(), member_auc_max=member_auc.max(),
                 gain=roc_auc_score(ye, ens) - member_auc.mean(), gain_lo=np.quantile(gains, 0.025),
                 gain_hi=np.quantile(gains, 0.975),
                 diff_vs_locked=roc_auc_score(ye, ens) - roc_auc_score(ye, locked), diff_lo=dl, diff_hi=dh,
                 diff_p=dp, cal_intercept=icpt, cal_slope=slope, brier=brier,
                 genes_per_member_median=float(np.median([len(m["genes"]) for m in ms])),
                 dropped_per_member_median=float(np.median([len(m["dropped"]) for m in ms]))),
            dict(model="locked27 (E10 a1)", auc=roc_auc_score(ye, locked), cal_intercept=li, cal_slope=ls, brier=lb)]
    pd.DataFrame(rows).to_csv(path(od, "E9_external.csv"), index=False)
    pd.DataFrame({"r": range(NREP), "auc": member_auc, "n_genes": [len(m["genes"]) for m in ms],
                  "n_used": [len(m["used"]) for m in ms], "dropped": [";".join(m["dropped"]) for m in ms],
                  "genes": [";".join(m["genes"]) for m in ms]}).to_csv(path(od, "E9_external_members.csv"),
                                                                        index=False)
    write_meta(od, name="metadata_E9_external.json", experiment="E9_external", nrep=NREP, fold_index=FOLD_INDEX, seed=SEED)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "summarise"])
    a = ap.parse_args()
    run() if a.action == "run" else summarise()


if __name__ == "__main__":
    main()
