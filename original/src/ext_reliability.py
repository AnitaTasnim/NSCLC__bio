#!/usr/bin/env python3
"""Locked-model reporting for the independent platelet cohort GSE68086.

    python ext_reliability.py config_ext.json           # all analyses and the figure
    python ext_reliability.py config_ext.json figure    # redraw the figure from the saved tables

The 27-gene model is trained once on all 633 discovery samples (log2-CPM over the genes shared by both
cohorts, the paper's primary external setting) and is then applied unchanged:
  S0  no alignment; S2  reference-batch ComBat-seq (discovery = fixed reference, model untouched).
Reported for both: AUC, Brier score, calibration intercept (logistic recalibration with the linear
predictor as offset) and slope, each with 2,000-resample stratified bootstrap CIs; sensitivity and
specificity at the discovery-fixed Youden threshold with Wilson CIs; and binned calibration curves.

Feature availability. Nine panel genes have no reads in GSE68086. Two analyses show how much the
locked model depends on gene availability: (i) "locked + imputed": the 27-gene model is applied with
k randomly chosen panel genes set to their discovery median (k = 0, 2, ..., 18; 100 draws each), and
(ii) "re-trained on available genes": the model is re-trained on the discovery data using only the
remaining genes and then applied (same draws). Genes already unmeasurable in GSE68086 are always
counted as missing; the extra genes removed are drawn from the 18 measurable ones.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tep_analysis as T  # noqa: E402


def calib(ye, p):
    import statsmodels.api as sm
    q = np.clip(p, 1e-6, 1 - 1e-6); lp = np.log(q / (1 - q))
    slope = float(sm.GLM(ye, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1])
    inter = float(sm.GLM(ye, np.ones((len(ye), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0])
    return inter, slope, float(np.mean((p - ye) ** 2))


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def boot_stats(ye, p, B=2000, seed=42):
    rng = np.random.default_rng(seed); i1, i0 = np.where(ye == 1)[0], np.where(ye == 0)[0]
    A, I, S, Br = [], [], [], []
    for _ in range(B):
        i = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]
        A.append(roc_auc_score(ye[i], p[i]))
        try:
            a, b, c = calib(ye[i], p[i]); I.append(a); S.append(b); Br.append(c)
        except Exception:
            pass
    q = lambda v: (float(np.quantile(v, .025)), float(np.quantile(v, .975)))
    return dict(auc=q(A), intercept=q(I), slope=q(S), brier=q(Br))


def main(cfg_path):
    cfg = T.load_config(cfg_path); d = T.load_data(cfg); run = T.Run(cfg, d)
    out = cfg["out_dir"]; os.makedirs(os.path.join(out, "macros"), exist_ok=True)
    y, ye = d.y.values, d.y_ext.values
    P27 = run.panel_in(d.X)
    C, Ce = d.counts, d.counts_ext
    shared = sorted(set(C.columns) & set(Ce.columns))
    C = C[shared].clip(lower=0).round().astype(np.int64); Ce = Ce[shared].clip(lower=0).round().astype(np.int64)
    ok = ((C.sum(0) > 0) & (Ce.sum(0) > 0)) | C.columns.isin(P27)
    C, Ce = C.loc[:, ok], Ce.loc[:, ok]
    Ld, Le = T.logcpm(C), T.logcpm(Ce)
    top = Ld.mean(0).sort_values(ascending=False).index[:2000]
    zero_any = [g for g in C.columns if C[g].sum() == 0 or Ce[g].sum() == 0]
    keep = sorted((set(top) | set(P27)) - set(zero_any))
    Cr, Cer = T._reduce_with_rest(C.drop(columns=zero_any), keep), T._reduce_with_rest(Ce.drop(columns=zero_any), keep)
    _, Re_ = T._combat_seq(Cr, Cer, ref_batch=0)
    for z in [g for g in zero_any if g in P27]:
        Re_[z] = Ce[z].values
    LRe = T.logcpm(Re_)
    G = [g for g in P27 if g in Ld.columns]
    undetected = [g for g in G if Ce[g].sum() == 0]
    measurable = [g for g in G if g not in undetected]
    mdl = run.xgb().fit(Ld[G].values, y)
    preds = {"S0": mdl.predict_proba(Le[G].values)[:, 1], "S2": mdl.predict_proba(LRe[G].values)[:, 1]}
    oof, _ = T.cv_oof(Ld[G].values, y, run.folds(), run.xgb); thr = T.youden_threshold(y, oof)
    M, rows, curves = {}, [], []
    for s, p in preds.items():
        inter, slope, brier = calib(ye, p); bs = boot_stats(ye, p)
        tp = int(((p >= thr) & (ye == 1)).sum()); fn = int(((p < thr) & (ye == 1)).sum())
        tn = int(((p < thr) & (ye == 0)).sum()); fp = int(((p >= thr) & (ye == 0)).sum())
        sl, sh = wilson(tp, tp + fn); pl, ph = wilson(tn, tn + fp)
        rows.append(dict(setting=s, auc=roc_auc_score(ye, p), auc_lo=bs["auc"][0], auc_hi=bs["auc"][1],
                         brier=brier, brier_lo=bs["brier"][0], brier_hi=bs["brier"][1],
                         cal_intercept=inter, int_lo=bs["intercept"][0], int_hi=bs["intercept"][1],
                         cal_slope=slope, slope_lo=bs["slope"][0], slope_hi=bs["slope"][1],
                         threshold=thr, sens=tp / (tp + fn), sens_lo=sl, sens_hi=sh,
                         spec=tn / (tn + fp), spec_lo=pl, spec_hi=ph, mean_pred=float(p.mean()), prevalence=float(ye.mean())))
        bins = pd.qcut(p, 5, duplicates="drop")
        for b, g in pd.DataFrame({"p": p, "y": ye, "b": bins}).groupby("b", observed=True):
            lo, hi = wilson(int(g.y.sum()), len(g))
            curves.append(dict(setting=s, mean_pred=g.p.mean(), obs=g.y.mean(), lo=lo, hi=hi, n=len(g)))
        key = {"S0": "None", "S2": "Ref"}[s]; r = rows[-1]
        M[f"Lk{key}Brier"] = f"{brier:.3f} ({bs['brier'][0]:.3f}--{bs['brier'][1]:.3f})"
        M[f"Lk{key}Int"] = f"{inter:.2f} ({bs['intercept'][0]:.2f} to {bs['intercept'][1]:.2f})".replace("-", "$-$").replace("$-$$-$", "--")
        M[f"Lk{key}Slope"] = f"{slope:.2f} ({bs['slope'][0]:.2f}--{bs['slope'][1]:.2f})"
        M[f"Lk{key}Sens"] = f"{r['sens']:.3f} ({sl:.3f}--{sh:.3f})"; M[f"Lk{key}Spec"] = f"{r['spec']:.3f} ({pl:.3f}--{ph:.3f})"
        M[f"Lk{key}MeanPred"] = f"{p.mean():.2f}"
    M["LkNoneIntShort"] = M["LkNoneInt"].split(" (")[0]; M["LkRefIntShort"] = M["LkRefInt"].split(" (")[0]
    M["LkPrev"] = f"{ye.mean():.2f}"; M["LkThr"] = f"{thr:.3f}"
    R = pd.DataFrame(rows); R.to_csv(os.path.join(out, "tables", "LK_locked_model_external.csv"), index=False)
    Cv = pd.DataFrame(curves); Cv.to_csv(os.path.join(out, "tables", "LK_calibration_bins.csv"), index=False)
    # ---------------- feature availability
    med = Ld[G].median(0)
    rng = np.random.default_rng(42); fa = []
    NDRAW = 100
    for k in range(0, len(measurable) + 1, 2):
        for rep in range(NDRAW if 0 < k < len(measurable) else 1):
            extra = list(rng.choice(measurable, k, replace=False)) if k else []
            miss = set(undetected) | set(extra)
            Xi = Le[G].copy()
            for g in miss:
                Xi[g] = med[g]
            a_imp = roc_auc_score(ye, mdl.predict_proba(Xi.values)[:, 1])
            avail = [g for g in G if g not in miss]
            a_re = roc_auc_score(ye, run.xgb(n_jobs=1).fit(Ld[avail].values, y).predict_proba(Le[avail].values)[:, 1]) if len(avail) >= 2 else np.nan
            fa.append(dict(extra_missing=k, n_available=len(avail), auc_locked_imputed=a_imp, auc_retrained=a_re))
    F = pd.DataFrame(fa); F.to_csv(os.path.join(out, "tables", "LK_feature_availability.csv"), index=False)
    S = F.groupby("n_available").agg(imp=("auc_locked_imputed", "median"), imp_lo=("auc_locked_imputed", lambda v: v.quantile(.05)),
                                     re=("auc_retrained", "median"), re_lo=("auc_retrained", lambda v: v.quantile(.05))).reset_index()
    S.to_csv(os.path.join(out, "tables", "LK_feature_availability_summary.csv"), index=False)
    full = S[S.n_available == len(measurable)].iloc[0]
    half = S.iloc[(S.n_available - len(measurable) // 2).abs().argmin()]
    M["LkFaDraws"] = str(NDRAW); M["LkMeasN"] = str(len(measurable)); M["LkUndetN"] = str(len(undetected))
    M["LkFaHalfN"] = str(int(half.n_available))
    M["LkFaHalfImp"] = f"{half.imp:.3f}"; M["LkFaHalfRe"] = f"{half.re:.3f}"
    M["LkFaHalfImpLo"] = f"{half.imp_lo:.3f}"; M["LkFaHalfReLo"] = f"{half.re_lo:.3f}"
    M["LkFaFullImp"] = f"{full.imp:.3f}"; M["LkFaFullRe"] = f"{full.re:.3f}"
    json.dump(M, open(os.path.join(out, "macros", "locked.json"), "w"), indent=1)
    draw(out, Cv, S)
    print(R.round(3).to_string(index=False)); print(S.round(3).to_string(index=False)); print(json.dumps(M, indent=1))


def draw(out, Cv, S):
    """Figure: (A) binned calibration, (B) external AUC against the number of available panel genes."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for s, col, lab in [("S0", "#b5532a", "No alignment"), ("S2", "#2a6f97", "Reference-batch ComBat-seq")]:
        c = Cv[Cv.setting == s]
        ax[0].errorbar(c.mean_pred, c.obs, yerr=[(c.obs - c.lo).clip(lower=0), (c.hi - c.obs).clip(lower=0)],
                       fmt="o-", color=col, capsize=3, label=lab)
    ax[0].plot([0, 1], [0, 1], ls="--", color="grey", lw=1); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1.02)
    ax[0].set_xlabel("Mean predicted probability (quintiles)"); ax[0].set_ylabel("Observed NSCLC fraction (95% CI)")
    ax[0].set_title("A  Calibration in GSE68086 (locked model)", loc="left", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False, loc="lower right")
    S = S[S.n_available > 0]
    ax[1].plot(S.n_available, S.imp, "o-", color="#b5532a", label="Locked model, missing genes imputed (median)")
    ax[1].plot(S.n_available, S.imp_lo, ls="--", lw=1, color="#b5532a", label="5th percentile")
    ax[1].plot(S.n_available, S.re, "s-", color="#2a6f97", label="Re-trained on available genes (median)")
    ax[1].plot(S.n_available, S.re_lo, ls="--", lw=1, color="#2a6f97", label="5th percentile")
    ax[1].axhline(0.5, ls=":", color="grey"); ax[1].set_ylim(0.4, 0.9)
    ax[1].set_xticks(list(S.n_available.astype(int)))
    ax[1].set_xlabel("Panel genes available in GSE68086 (of 18 measurable)"); ax[1].set_ylabel("External AUC")
    ax[1].set_title("B  Dependence on gene availability", loc="left", fontsize=10)
    ax[1].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(out, "figures", f"fig_locked_external.{e}"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "figure":     # redraw from the saved tables
        o = T.load_config(sys.argv[1])["out_dir"]
        draw(o, pd.read_csv(os.path.join(o, "tables", "LK_calibration_bins.csv")),
             pd.read_csv(os.path.join(o, "tables", "LK_feature_availability_summary.csv")))
    else:
        main(sys.argv[1])
