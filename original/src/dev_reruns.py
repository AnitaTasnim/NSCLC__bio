#!/usr/bin/env python3
"""Seeded re-runs of the four supporting analyses of the development phase.

    python dev_reruns.py config.json ablation    # group and leave-one-out ablation (~2 min)
    python dev_reruns.py config.json featcount   # feature-count curve, genes ranked by Mann-Whitney (~15 min)
    python dev_reruns.py config.json rfecv       # RFECV ceiling and Meinshausen-Buhlmann stability selection (~15 min)

All three use the same five folds (seed 42, GEO accession order), the same classifier as the fixed-panel estimate and
the un-normalised counts, and, like the fixed-panel estimate, they are conditional on genes chosen from all samples.
The RFECV and stability-selection settings follow rfecv_ceiling() and stability_selection_mb() of the authors'
nestedcvleakagefix.ipynb. The "novel" genes are the seven genes without prior cancer literature as classified in the
Phase-1 literature screen (phaase1_extended_1.ipynb)."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tep_analysis as T  # noqa: E402
from selection_pipeline import stage1_mw  # noqa: E402

NOVEL = ["LCE3B", "SPHAR", "NTNG2", "TMEM39B", "OR52B6", "PRSS42", "KRTAP22-1"]
TOP5_SHAP = ["P2RY10", "HSD17B3", "MGAT5", "CXCL14", "PTMS"]
GREY, ACCENT, INK = "#9AA5B1", "#C0392B", "#333333"


def setup(cfg_path):
    cfg = T.load_config(cfg_path); d = T.load_data(cfg); run = T.Run(cfg, d)
    out = cfg["out_dir"]
    for s in ("macros", "tables", "figures"):
        os.makedirs(os.path.join(out, s), exist_ok=True)
    return cfg, d, run, out


def save(out, name, m):
    json.dump(m, open(os.path.join(out, "macros", f"{name}.json"), "w"), indent=1)
    print(name, json.dumps(m, indent=0))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)


def pp(x):
    return f"{x:+.2f}".replace("-", "$-$")


def ablation(cfg_path):
    cfg, d, run, out = setup(cfg_path)
    y = d.y.values; P = run.panel_in(d.X); folds = run.folds()
    fac = lambda: run.xgb(n_jobs=2)
    full, fa = T.cv_oof(d.X[P].values, y, folds, fac)
    auc_full = T.roc_auc_score(y, full)
    rows = []

    def drop(name, genes, group):
        keep = [g for g in P if g not in set(genes)]
        o, f = T.cv_oof(d.X[keep].values, y, folds, fac)
        r = T.delong_paired(y, o, full)
        rows.append(dict(group=group, removed=name, n_genes=len(keep), auc=r["auc1"], delta_pp=100 * r["diff"],
                         lo_pp=100 * r["lo95"], hi_pp=100 * r["hi95"], p=r["p"], fold_mean=np.mean(f)))

    drop("top-five SHAP genes", TOP5_SHAP, "group")
    drop("seven novel genes", NOVEL, "group")
    drop("three contamination-prone genes", ["KRTAP22-1", "OR52B6", "LCE3B"], "group")
    for g in P:
        drop(g, [g], "single")
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, "tables", "DEV_ablation.csv"), index=False)
    S = R[R.group == "single"].sort_values("delta_pp")
    G = R[R.group == "group"].set_index("removed")
    worst = S.iloc[0]
    n_sig_drop = int(((S.hi_pp < 0)).sum())
    m = dict(AblFullAUC=f"{auc_full:.4f}",
             AblTopFiveDelta=pp(G.loc["top-five SHAP genes", "delta_pp"]),
             AblTopFiveCI=f"{pp(G.loc['top-five SHAP genes', 'lo_pp'])} to {pp(G.loc['top-five SHAP genes', 'hi_pp'])}",
             AblTopFiveP=T.pmac(G.loc["top-five SHAP genes", "p"]),
             AblNovelDelta=pp(G.loc["seven novel genes", "delta_pp"]),
             AblNovelCI=f"{pp(G.loc['seven novel genes', 'lo_pp'])} to {pp(G.loc['seven novel genes', 'hi_pp'])}",
             AblNovelP=T.pmac(G.loc["seven novel genes", "p"]),
             AblWorstGene=T.tex_escape(worst.removed), AblWorstDelta=pp(worst.delta_pp),
             AblPtwoDelta=pp(float(S.set_index("removed").loc["P2RY10", "delta_pp"])),
             AblSingleMin=pp(S.delta_pp.min()), AblSingleMax=pp(S.delta_pp.max()),
             AblSingleNsig=str(n_sig_drop), AblNovelGenes=", ".join(NOVEL))
    save(out, "ablation", m)
    # figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axs[0]
    g = R[R.group == "group"].iloc[::-1]
    yy = np.arange(len(g))
    ax.barh(yy, g.delta_pp, color=GREY, height=0.5)
    ax.errorbar(g.delta_pp, yy, xerr=[g.delta_pp - g.lo_pp, g.hi_pp - g.delta_pp], fmt="none", ecolor=INK, lw=1, capsize=3)
    ax.axvline(0, c=INK, lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels([f"without {t}" for t in g.removed], fontsize=8)
    ax.set_xlabel("Change in AUC (percentage points, 95% CI)", fontsize=8)
    ax.set_title("A  Groups of genes removed", fontsize=10, loc="left", fontweight="bold")
    style(ax)
    ax = axs[1]
    s = S.iloc[::-1]
    yy = np.arange(len(s))
    ax.errorbar(s.delta_pp, yy, xerr=[s.delta_pp - s.lo_pp, s.hi_pp - s.delta_pp], fmt="o", ms=3.5, color=GREY,
                ecolor=GREY, lw=0.8)
    ax.axvline(0, c=INK, lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels(s.removed, fontsize=6.5)
    ax.set_xlabel("Change in AUC (percentage points, 95% CI)", fontsize=8)
    ax.set_title("B  One gene removed", fontsize=10, loc="left", fontweight="bold")
    style(ax)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(out, "figures", f"DEV_ablation.{e}"), dpi=200, bbox_inches="tight")


def featcount(cfg_path):
    cfg, d, run, out = setup(cfg_path)
    y = d.y.values; P = run.panel_in(d.X); folds = run.folds()
    mw = stage1_mw(d.X, y)
    ranked = mw.assign(_g=mw.index).sort_values(["padj", "pvalue", "_g"]).index.tolist()
    sets = [("10", ranked[:10]), ("27 (panel)", P), ("50", ranked[:50]), ("100", ranked[:100]), ("500", ranked[:500]),
            (f"{d.X.shape[1]:,}", list(d.X.columns))]
    rows = []
    for lab, genes in sets:
        t0 = time.time()
        o, f = T.cv_oof(d.X[genes].values, y, folds, lambda: run.xgb(n_jobs=2))
        a, lo, hi, _ = T.delong_ci(y, o)
        rows.append(dict(label=lab, n_genes=len(genes), auc=a, lo=lo, hi=hi, fold_mean=np.mean(f), fold_sd=np.std(f, ddof=1)))
        print(lab, round(a, 4), f"{time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, "tables", "DEV_feature_count.csv"), index=False)
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Genes & AUC (95\% CI) & Fold mean $\pm$ SD \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['label'].replace(',', '{,}')} & {r['auc']:.4f} ({r['lo']:.4f}--{r['hi']:.4f}) & "
                     f"{r['fold_mean']:.4f} $\\pm$ {r['fold_sd']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tables", "DEV_feature_count.tex"), "w").write("\n".join(lines) + "\n")
    best = R.loc[R.auc.idxmax()]; pan = R[R.label == "27 (panel)"].iloc[0]
    m = dict(FcTenAUC=f"{R.auc.iloc[0]:.4f}", FcPanelAUC=f"{pan.auc:.4f}", FcFiftyAUC=f"{R.auc.iloc[2]:.4f}",
             FcHundredAUC=f"{R.auc.iloc[3]:.4f}", FcFiveHundredAUC=f"{R.auc.iloc[4]:.4f}", FcAllAUC=f"{R.auc.iloc[5]:.4f}",
             FcAllN=f"{d.X.shape[1]:,}".replace(",", "{,}"), FcBestLabel=str(best.label).replace(",", "{,}"),
             FcBestAUC=f"{best.auc:.4f}", FcPanelPctOfBest=f"{100 * pan.auc / best.auc:.1f}")
    save(out, "featcount", m)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.fill_between(R.n_genes, R.lo, R.hi, color=GREY, alpha=0.25, lw=0)
    ax.plot(R.n_genes, R.auc, "-o", color=GREY, ms=4, lw=1.5)
    ax.plot([pan.n_genes], [pan.auc], "o", color=ACCENT, ms=7, zorder=5)
    ax.annotate(f"27-gene panel\n{pan.auc:.3f}", (pan.n_genes, pan.auc), textcoords="offset points", xytext=(8, -26),
                fontsize=8, color=ACCENT)
    ax.set_xscale("log"); ax.set_xticks(R.n_genes); ax.set_xticklabels([str(n) if n < 1000 else f"{n:,}" for n in R.n_genes])
    ax.set_xlabel("Number of genes (top genes by Mann-Whitney p, except the panel)", fontsize=8)
    ax.set_ylabel("Five-fold CV AUC (95% CI)", fontsize=8)
    style(ax)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(out, "figures", f"DEV_feature_count.{e}"), dpi=200, bbox_inches="tight")


def rfecv(cfg_path, n_sub=200):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    cfg, d, run, out = setup(cfg_path)
    y = d.y.values; P = run.panel_in(d.X)
    mw = stage1_mw(d.X, y)
    top200 = mw.assign(_g=mw.index).sort_values(["padj", "pvalue", "_g"]).index[:200].tolist()
    pool = sorted(set(top200) | set(P))
    Xs = StandardScaler().fit_transform(d.X[pool].values)
    cv5 = StratifiedKFold(5, shuffle=True, random_state=42)
    curve_csv = os.path.join(out, "tables", "DEV_rfecv_curve.csv")
    if os.path.exists(curve_csv) and "--force" not in sys.argv:      # RFECV takes ~15 min; reuse a finished run
        cur = pd.read_csv(curve_csv)
        nfeat, mean, sd = cur.n_features.values, cur.mean_auc.values, cur.sd_auc.values
    else:
        t0 = time.time()
        sel = RFECV(GradientBoostingClassifier(random_state=42), step=1, cv=cv5, scoring="roc_auc", n_jobs=2,
                    min_features_to_select=5).fit(Xs, y)
        print(f"RFECV {time.time() - t0:.0f}s", flush=True)
        res = sel.cv_results_
        nfeat = np.asarray(res.get("n_features", np.arange(5, len(pool) + 1)))
        mean, sd = np.asarray(res["mean_test_score"]), np.asarray(res["std_test_score"])
        pd.DataFrame({"n_features": nfeat, "mean_auc": mean, "sd_auc": sd}).to_csv(curve_csv, index=False)
    i = int(np.argmax(mean)); k = int(nfeat[i])                       # RFECV's choice: first maximum of the mean AUC
    pidx = [pool.index(g) for g in P]
    pan = cross_val_score(GradientBoostingClassifier(random_state=42), Xs[:, pidx], y, cv=cv5, scoring="roc_auc", n_jobs=2)
    # Meinshausen-Buhlmann stability selection on the same pool (200 half-samples, LASSO on 0/1 labels)
    Xp = d.X[pool].values; n = len(y); cnt = np.zeros(len(pool))
    t0 = time.time()
    for b in range(n_sub):
        idx = np.random.RandomState(b).choice(n, int(n * 0.5), replace=False)
        Xb = StandardScaler().fit_transform(Xp[idx])
        try:                                                           # scikit-learn >= 1.7 replaced n_alphas by alphas=int
            las = LassoCV(cv=3, max_iter=3000, tol=1e-2, alphas=20, n_jobs=2, random_state=b)
        except TypeError:
            las = LassoCV(cv=3, max_iter=3000, tol=1e-2, n_alphas=20, n_jobs=2, random_state=b)
        las.fit(Xb, y[idx])
        cnt += las.coef_ != 0
    print(f"stability selection {time.time() - t0:.0f}s", flush=True)
    pi = pd.Series(cnt / n_sub, index=pool).sort_values(ascending=False)
    pi.rename("pi").to_csv(os.path.join(out, "tables", "DEV_stability_selection_pi.csv"))
    pip = pi.reindex(P)
    m = dict(RfecvPool=str(len(pool)), RfecvK=str(k), RfecvAUC=f"{mean[i]:.4f}", RfecvSD=f"{sd[i]:.4f}",
             RfecvPanelAUC=f"{pan.mean():.4f}", RfecvPanelSD=f"{pan.std(ddof=1):.4f}",
             RfecvPanelPct=f"{100 * pan.mean() / mean[i]:.1f}", RfecvGenePct=f"{100 * 27 / k:.1f}",
             SsN=str(n_sub), SsPanelN=str(int((pip >= 0.60).sum())),
             SsPanelBelow=", ".join(T.tex_escape(g) for g in pip[pip < 0.60].sort_values().index) or "none",
             SsPoolN=str(int((pi >= 0.60).sum())))
    save(out, "rfecv", m)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.fill_between(nfeat, mean - sd, mean + sd, color=GREY, alpha=0.25, lw=0)
    ax.plot(nfeat, mean, color=GREY, lw=1.2)
    ax.plot([k], [mean[i]], "o", color=INK, ms=6)
    ax.annotate(f"RFECV optimum: {k} genes, {mean[i]:.3f}", (k, mean[i]), textcoords="offset points", xytext=(6, -16), fontsize=8)
    ax.axhline(pan.mean(), color=ACCENT, lw=1.2, ls="--")
    ax.text(nfeat.max(), pan.mean() - 0.006, f"27-gene panel {pan.mean():.3f}", color=ACCENT, fontsize=8, ha="right", va="top")
    ax.set_xlabel("Number of genes retained by RFECV", fontsize=8)
    ax.set_ylabel("Five-fold CV AUC (mean ± SD)", fontsize=8)
    style(ax)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(out, "figures", f"DEV_rfecv.{e}"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    {"ablation": ablation, "featcount": featcount, "rfecv": rfecv}[sys.argv[2]](sys.argv[1])
