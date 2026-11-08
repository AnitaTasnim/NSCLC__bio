"""E16 - unit of inference: which reliability quantity predicts the bagging gain within a data set?
EXPLORATORY (5 selectors per data set). Plan: docs/plan_E8_E19.md, section E16.

Run (after E18):  python -m pmtrd.revision.e16
"""
from __future__ import annotations

import shutil
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import SAVED, XGB_SELECTORS, out_dir, path, saved, write_meta

SEED = 16
NPERM = 20000
PREDICTORS = ["psi", "V_R", "phi", "phi_C"]


def table() -> pd.DataFrame:
    e18 = pd.read_csv(path("runs", "revision", "E18", "E18.csv"))
    rows = []
    for ds in SAVED:
        ag = saved(ds, "agreement.csv").set_index("method")
        ish = saved(ds, "identity_share.csv").set_index("method")
        st = saved(ds, "stability.csv").set_index("method")
        e = e18[e18.dataset == ds].set_index("method")
        for m in XGB_SELECTORS:
            rows.append(dict(dataset=ds, method=m, gain=ag.bagged_auc[m] - ag.auc_mean[m], psi=ish.share[m],
                             V_R=ish.var_reselect[m], phi=st.phi[m], phi_C=e["phi_C_0.5"][m]))
    return pd.DataFrame(rows)


def mean_within_rho(df: pd.DataFrame, x: str, perm=None) -> float:
    vals = []
    for i, (ds, d) in enumerate(df.groupby("dataset", sort=True)):
        xv = d[x].values if perm is None else d[x].values[perm[i]]
        vals.append(spearmanr(xv, d.gain.values).correlation)
    return float(np.mean(vals))


def loso(df: pd.DataFrame, x: str) -> tuple:
    err, base = [], []
    for ds, d in df.groupby("dataset"):
        xv, g = d[x].values, d.gain.values
        for i in range(len(d)):
            m = np.ones(len(d), bool); m[i] = False
            b = np.polyfit(xv[m], g[m], 1)
            err.append(abs(np.polyval(b, xv[i]) - g[i]))
            base.append(abs(g[m].mean() - g[i]))
    return float(np.mean(err)), float(np.mean(base))


def main() -> None:
    import statsmodels.formula.api as smf
    od = out_dir("runs", "revision", "E16")
    df = table()
    df.to_csv(path(od, "E16_data.csv"), index=False)
    rng = np.random.default_rng(SEED)
    groups = df.groupby("dataset", sort=True)
    sizes = [len(d) for _, d in groups]
    perms = [[rng.permutation(n) for n in sizes] for _ in range(NPERM)]
    rows = []
    for x in PREDICTORS:
        obs = mean_within_rho(df, x)
        null = np.array([mean_within_rho(df, x, p) for p in perms])
        p = float((1 + (np.abs(null) >= abs(obs) - 1e-12).sum()) / (NPERM + 1))
        per_ds = {f"rho_{ds}": spearmanr(d[x], d.gain).correlation for ds, d in groups}
        d2 = df.assign(z=(df[x] - df[x].mean()) / df[x].std(ddof=1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = smf.mixedlm("gain ~ z", d2, groups=d2["dataset"]).fit(reml=True)
                coef, lo, hi, pm = fit.params["z"], *fit.conf_int().loc["z"].values, fit.pvalues["z"]
                conv = bool(fit.converged)
            except Exception:
                coef = lo = hi = pm = np.nan; conv = False
        mae, mae0 = loso(df, x)
        rows.append(dict(predictor=x, mean_within_rho=obs, perm_p=p, **per_ds, mixed_coef_per_sd=coef,
                         mixed_lo=lo, mixed_hi=hi, mixed_p=pm, mixed_converged=conv, loso_mae=mae,
                         loso_mae_baseline=mae0, loso_mae_ratio=mae / mae0))
        print(x, round(obs, 3), round(p, 4), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(path(od, "E16.csv"), index=False)
    shutil.copy(path(od, "E16.csv"), path("runs", "export", "E16.csv"))
    write_meta(od, experiment="E16", seed=SEED, nperm=NPERM, note="exploratory")


if __name__ == "__main__":
    main()
