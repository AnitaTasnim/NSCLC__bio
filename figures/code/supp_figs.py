"""Supplementary figures S1-S14 (S8 and S14 are conceptual and live in concept_figs.py)."""
from __future__ import annotations

import os
import re
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import data as D
import style as S
from style import INK, MUTED, FAINT, RULE, GRID

CODE = os.path.join(D.PKG, "code")


@lru_cache(maxsize=1)
def engine():
    """The discovery data through the analysis engine (same loading as every analysis in the paper)."""
    if D.REPO:
        sys.path.insert(0, D.REPO)
        from pmtrd import config as C
        from pmtrd.engine import T
        ec = T.load_config(C.engine_config(C.load(os.path.join(D.REPO, "configs", "replicates.json"))))
        where = D.REPO
    else:
        sys.path.insert(0, CODE)
        import tep_analysis as T
        ec = T.load_config(os.path.join(CODE, "config.json"))
        where = CODE
    cwd = os.getcwd(); os.chdir(where)
    try:
        d = T.load_data(ec); run = T.Run(ec, d)
    finally:
        os.chdir(cwd)
    return T, ec, d, run


def panel_table():
    """Development-run selection profile of the 27 genes (Supplementary Table tab:panel)."""
    if D.PANEL_CSV:
        return pd.read_csv(D.PANEL_CSV)
    s = open(os.path.join(D.MAN, "supp", "s02b_moved.tex")).read()
    blk = s[s.index("\\label{tab:panel}"):]
    blk = blk[blk.index("\\midrule") + 8: blk.index("\\bottomrule")]
    rows = []
    for line in blk.strip().split("\\\\"):
        f = [x.strip() for x in line.split("&")]
        if len(f) == 5:
            rows.append(dict(gene=re.sub(r"\$.*?\$", "", f[0]).strip(), direction=f[1], boot=float(f[2]),
                             pi_dev=float(f[3]), pi_seed=float(f[4])))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- S1
def figS01_stabprof(out):
    T = panel_table().sort_values("pi_dev")
    fig, ax = plt.subplots(figsize=(S.HALF + 0.9, 4.1))
    y = np.arange(len(T))
    ax.hlines(y, T[["boot", "pi_dev"]].min(axis=1), T[["boot", "pi_dev"]].max(axis=1), color="#d5dbe2", lw=1.6)
    ax.plot(T.boot, y, "o", color=S.SEL["six_stage"], ms=4, label="bootstrap frequency (500 resamples)")
    ax.plot(T.pi_dev, y, "D", color=S.NSCLC, mfc="white", mew=1.0, ms=3.6,
            label="stability-selection probability $\\pi$ (200 half-samples)")
    ax.axvline(0.6, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(0.603, len(T) - 0.3, "0.60 threshold", fontsize=6.2, color=MUTED, va="bottom")
    ax.set_yticks(y); ax.set_yticklabels(T.gene, fontsize=6.6); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.55, 1.01); ax.set_ylim(-0.7, len(T) + 0.6)
    ax.set_xlabel("Selection frequency or probability (development run)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.01), ncol=1, fontsize=6.6, handletextpad=0.3)
    S.xgrid(ax)
    return S.save(fig, out, "FigS01_stabprof")


# --------------------------------------------------------------------------------------------- S2
def figS02_heatmap(out):
    P = pd.read_csv(os.path.join(D.V2, "data", "reference", "DEFINITIVE_biomarker_panel.csv")).rename(
        columns={"Unnamed: 0": "gene"}).sort_values("composite", ascending=False)
    cols = [("stat_score", "Statistical\nsignificance"), ("effect_score", "Effect\nsize"), ("lasso_score", "LASSO\nscore"),
            ("ppi_score", "PPI\ncentrality"), ("boot_score", "Bootstrap\nstability"), ("composite", "Composite\nscore")]
    V = P[[c for c, _ in cols]].values
    fig = plt.figure(figsize=(S.HALF + 0.9, 4.9))
    ax = fig.add_axes([0.24, 0.02, 0.74, 0.86])
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("b", ["#f3f6fa", "#a9c1db", "#4d7aa6", "#1e4a7a"])
    cm2 = LinearSegmentedColormap.from_list("t", ["#f2f8f6", "#9ccbc4", "#3f958a", "#155f55"])
    n = len(P)
    for i in range(n):
        for j in range(len(cols)):
            v = V[i, j]; c = (cm2 if j == len(cols) - 1 else cm)(v if j < len(cols) - 1 else (v - 0.5) / 0.35)
            ax.add_patch(plt.Rectangle((j + 0.04, n - 1 - i + 0.06), 0.92, 0.88, color=c, lw=0))
            dark = (v if j < len(cols) - 1 else (v - 0.5) / 0.35) > 0.55
            ax.text(j + 0.5, n - 1 - i + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=5.9,
                    color="white" if dark else INK)
        ax.text(-0.1, n - 1 - i + 0.5, P.gene.iloc[i], ha="right", va="center", fontsize=6.4)
    for j, (_, lab) in enumerate(cols):
        ax.text(j + 0.5, n + 0.25, lab, ha="center", va="bottom", fontsize=6.4, linespacing=1.1,
                fontweight="bold" if j == len(cols) - 1 else "normal")
    ax.set_xlim(0, len(cols)); ax.set_ylim(0, n + 1.6); ax.axis("off")
    fig.text(0.98, -0.01, "Scores 0–1; rows ordered by the development composite score.", ha="right",
             fontsize=6.2, color=MUTED)
    return S.save(fig, out, "FigS02_heatmap")


# --------------------------------------------------------------------------------------------- S3
def figS03_contam(out):
    from scipy.stats import pearsonr
    T, cfg, d, run = engine()
    X = d.X; y = np.asarray(d.y)
    genes = ["LCE3B", "KRTAP22-1", "OR52B6"]; markers = ["KRT1", "KRT10", "FLG", "DSC1"]
    fig, ax = plt.subplots(3, 4, figsize=(S.FULL, 4.3), gridspec_kw=dict(hspace=0.55, wspace=0.32))
    for i, g in enumerate(genes):
        for j, m in enumerate(markers):
            a = ax[i, j]
            for cls, col in ((0, S.HEALTHY), (1, S.NSCLC)):
                k = y == cls
                a.scatter(X[m].values[k], X[g].values[k], s=3, color=col, alpha=0.45, lw=0, rasterized=True)
            r, p = pearsonr(X[g], X[m])
            strong = r > 0.2
            a.text(0.97, 0.94, f"r = {r:.3f}\np = {p:.1e}".replace("e-0", "e−").replace("e-", "e−"),
                   transform=a.transAxes, ha="right", va="top", fontsize=6.0,
                   color=INK if strong else MUTED, fontweight="bold" if strong else "normal", linespacing=1.2)
            if i == 2:
                a.set_xlabel(f"{m} (counts)", fontsize=6.8)
            if j == 0:
                a.set_ylabel(f"{g} (counts)", fontsize=6.8)
            a.tick_params(labelsize=6)
    fig.legend(handles=[Line2D([], [], marker="o", color=S.HEALTHY, ls="", ms=3.5, label="healthy"),
                        Line2D([], [], marker="o", color=S.NSCLC, ls="", ms=3.5, label="NSCLC")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.text(0.01, 0.94, "Suspect panel genes (rows) against skin-contamination markers (columns); Pearson r on "
             "counts, all 633 samples", fontsize=7.2, va="top", color=MUTED)
    return S.save(fig, out, "FigS03_contam")


# --------------------------------------------------------------------------------------------- S4
def figS04_shap(out):
    T = D.tab("minor_shap_out_of_fold").sort_values("mean_abs_shap_oof")
    fig, ax = plt.subplots(figsize=(S.HALF + 0.6, 4.0))
    y = np.arange(len(T))
    ax.hlines(y, 0, T.mean_abs_shap_oof, color=S.DIM_M["P"], lw=2.0)
    ax.plot(T.mean_abs_shap_oof, y, "o", color=S.SEL["six_stage"], ms=4, label="held-out folds")
    ax.plot(T.mean_abs_shap_insample, y, "|", color=INK, ms=6, mew=1.1, label="in-sample (training data)")
    ax.set_yticks(y); ax.set_yticklabels(T.gene, fontsize=6.6); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.set_xlim(0, None); ax.set_ylim(-0.7, len(T) - 0.3)
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.legend(loc="lower right")
    S.xgrid(ax)
    return S.save(fig, out, "FigS04_shap")


# --------------------------------------------------------------------------------------------- S5
def figS05_featcount(out):
    T = D.tab("DEV_feature_count")
    fig, ax = plt.subplots(figsize=(S.HALF + 0.4, 2.3))
    x = T.n_genes.values
    ax.plot(x, T.auc, "-", color=FAINT, lw=1.0, zorder=1)
    for _, r in T.iterrows():
        panel = r.n_genes == 27
        col = S.NSCLC if panel else S.SEL["six_stage"]
        ax.errorbar(r.n_genes, r.auc, yerr=[[r.auc - r.lo], [r.hi - r.auc]], fmt="o", color=col, ms=4.2,
                    elinewidth=0.9, capsize=0, mec="white", mew=0.5, zorder=3)
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{int(v):,}" for v in x])
    ax.minorticks_off()
    ax.annotate("27-gene panel", (27, float(T[T.n_genes == 27].auc.iloc[0])), xytext=(-14, 9),
                textcoords="offset points", fontsize=6.4, color=S.NSCLC)
    ax.set_xlabel("Number of genes (log scale)"); ax.set_ylabel("Pooled out-of-fold AUC (95% CI)")
    S.ygrid(ax)
    return S.save(fig, out, "FigS05_featcount")


# --------------------------------------------------------------------------------------------- S6
def figS06_ablation(out):
    T = D.tab("DEV_ablation")
    gr = T[T.group == "group"]; ge = T[T.group != "group"].sort_values("delta_pp")
    fig = plt.figure(figsize=(S.FULL, 3.3))
    a = fig.add_axes([0.13, 0.62, 0.30, 0.28])
    lab = {"top-five SHAP genes": "top 5 SHAP genes", "seven novel genes": "7 novel genes"}
    for i, (_, r) in enumerate(gr.iterrows()):
        yy = len(gr) - 1 - i
        col = S.NSCLC if r.hi_pp < 0 else S.SEL["six_stage"]
        a.plot([r.lo_pp, r.hi_pp], [yy, yy], color=col, lw=1.2); a.plot(r.delta_pp, yy, "o", color=col, ms=4)
    a.set_yticks(range(len(gr))); a.set_yticklabels([lab.get(v, v) for v in gr.removed][::-1], fontsize=6.6)
    a.axvline(0, color=MUTED, lw=0.7, ls=(0, (2, 2))); a.tick_params(axis="y", length=0)
    a.set_xlabel("AUC change (percentage points)"); S.xgrid(a)
    a.set_ylim(-0.6, len(gr) - 0.4)
    fig.text(0.01, 0.95, "A   Gene groups removed", fontsize=8.5, fontweight="bold", va="bottom")
    h = (len(ge) + 1) // 2
    for k, part in enumerate((ge.iloc[:h], ge.iloc[h:])):
        b = fig.add_axes([0.56 + k * 0.23, 0.13, 0.17, 0.77])
        for i, (_, r) in enumerate(part.iterrows()):
            yy = len(part) - 1 - i
            col = S.NSCLC if r.hi_pp < 0 else (S.SEL["six_stage"])
            b.plot([r.lo_pp, r.hi_pp], [yy, yy], color=col, lw=1.0); b.plot(r.delta_pp, yy, "o", color=col, ms=3.2)
        b.set_yticks(range(len(part))); b.set_yticklabels(part.removed.tolist()[::-1], fontsize=6.3)
        b.tick_params(axis="y", length=0); b.axvline(0, color=MUTED, lw=0.7, ls=(0, (2, 2)))
        b.set_xlim(-1.4, 1.4); b.set_ylim(-0.6, len(part) - 0.4); b.set_xlabel("AUC change (pp)"); S.xgrid(b)
    fig.text(0.47, 0.95, "B   Single genes removed", fontsize=8.5, fontweight="bold", va="bottom")
    fig.text(0.01, 0.30, "Red: the 95% CI excludes zero.\nDifferences from the full panel,\n"
             "paired DeLong CIs, same folds.", fontsize=6.4, color=MUTED, va="top", linespacing=1.35)
    return S.save(fig, out, "FigS06_ablation")


# --------------------------------------------------------------------------------------------- S7
def figS07_de(out):
    from PIL import Image
    T, cfg, d, run = engine()
    sys.path.insert(0, CODE)
    if D.REPO:
        from pmtrd.engine import S as SP
    else:
        import selection_pipeline as SP
    r = SP.stage1_mw(d.X, np.asarray(d.y)); a_ = r.log2FC.abs()
    t1 = (r.padj < 0.05) & (a_ > 1); t2 = (r.padj < 0.05) & (a_ > 0.5) & ~t1; t3 = (r.padj < 0.10) & (a_ > 0.3) & ~t1 & ~t2
    ns = ~(t1 | t2 | t3)
    panel = pd.read_csv(os.path.join(D.V2, "data", "reference", "DEFINITIVE_biomarker_panel.csv"))["Unnamed: 0"]
    fig = plt.figure(figsize=(S.FULL, 3.2))
    a = fig.add_axes([0.07, 0.13, 0.40, 0.75])
    yv = -np.log10(r.padj.clip(lower=1e-300))
    layers = [(ns, "#d3d8de", "not significant", 2),
              (t3, "#f0b37e", f"tier 3: q < 0.10, |log$_2$FC| > 0.3 ({int((t1 | t2 | t3).sum()):,} incl. tiers 1\u20132)", 2.5),
              (t2, "#d9653b", f"tier 2: q < 0.05, |log$_2$FC| > 0.5 ({int((t1 | t2).sum()):,} incl. tier 1)", 2.5),
              (t1, "#8e2b1c", f"tier 1: q < 0.05, |log$_2$FC| > 1 ({int(t1.sum()):,})", 3)]
    hs = {}
    for m, col, lab, sz in layers:
        hs[lab] = a.scatter(r.log2FC[m], yv[m], s=sz, color=col, lw=0, rasterized=True, label=lab)
    pp = r.loc[r.index.isin(panel)]
    hp = a.scatter(pp.log2FC, -np.log10(pp.padj), s=14, color=INK, edgecolor="white", lw=0.5, zorder=4,
                   label="27-gene panel")
    for gname, row in pp.sort_values("padj").head(4).iterrows():
        a.annotate(gname, (row.log2FC, -np.log10(row.padj)), xytext=(-6, 0), textcoords="offset points",
                   fontsize=5.8, ha="right", va="center", color=INK)
    a.axhline(-np.log10(0.05), color=MUTED, lw=0.6, ls=(0, (2, 2)))
    a.set_xlabel("log$_2$ fold change (NSCLC vs healthy)"); a.set_ylabel("$-\\log_{10}$ BH-adjusted p")
    a.legend(handles=[hs[l[2]] for l in layers[::-1]] + [hp], loc="upper right", fontsize=5.8, markerscale=2.2,
             handletextpad=0.2, borderaxespad=0.1)
    a.set_xlim(-16, 16)
    fig.text(0.01, 0.92, f"A   Mann–Whitney, all {len(d.y)} samples ({len(r):,} genes)", fontsize=8.5,
             fontweight="bold", va="bottom")
    im = Image.open(D.DE_IMAGE).convert("RGB")
    W_, H_ = im.size
    crop = im.crop((int(W_ * 0.535), int(H_ * 0.085), W_, H_))
    b = fig.add_axes([0.53, 0.0, 0.47, 0.88]); b.imshow(np.asarray(crop)); b.axis("off")
    fig.text(0.53, 0.92, "B   Count model (DESeq2), development analysis", fontsize=8.5, fontweight="bold",
             va="bottom")
    return S.save(fig, out, "FigS07_de")


# --------------------------------------------------------------------------------------------- S9
def figS09_roc(out):
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
    T, cfg, d, run = engine()
    P = run.panel_in(d.X); X, y = d.X[P].values, d.y.values
    oof, fa = T.cv_oof(X, y, run.folds(), run.xgb)
    auc, lo, hi, _ = T.delong_ci(y, oof)
    M = D.macros()
    assert abs(auc - D.num(M["CoreFixedAUC"])) < 5e-5, (auc, M["CoreFixedAUC"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL, 2.75), gridspec_kw=dict(wspace=0.3))
    for k, (_, te) in enumerate(run.folds()):
        fpr, tpr, _ = roc_curve(y[te], oof[te]); a.plot(fpr, tpr, color="#c3cad3", lw=0.8)
    fpr, tpr, _ = roc_curve(y, oof)
    a.plot(fpr, tpr, color=S.SEL["six_stage"], lw=1.6)
    a.plot([0, 1], [0, 1], color=FAINT, lw=0.7, ls=(0, (2, 2)))
    a.text(0.97, 0.08, f"pooled out-of-fold AUC {auc:.3f}\n95% CI {lo:.3f}–{hi:.3f}\n"
                       f"fold AUCs {min(fa):.3f}–{max(fa):.3f} (grey)", ha="right", va="bottom", fontsize=6.5,
           transform=a.transAxes, linespacing=1.35)
    a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate"); a.set_aspect("equal")
    S.panel(a, "A", "ROC curve")
    pr, rc, _ = precision_recall_curve(y, oof); ap = average_precision_score(y, oof)
    b.plot(rc, pr, color=S.SEL["six_stage"], lw=1.6)
    b.axhline(y.mean(), color=FAINT, lw=0.7, ls=(0, (2, 2)))
    b.text(0.02, y.mean() - 0.03, f"prevalence {y.mean():.3f}", fontsize=6.4, color=MUTED, va="top")
    b.text(0.03, 0.08, f"AUPRC {ap:.3f}", fontsize=6.5, transform=b.transAxes)
    b.set_ylim(0, 1.02); b.set_xlabel("Recall"); b.set_ylabel("Precision"); b.set_aspect("equal")
    S.panel(b, "B", "Precision–recall curve")
    return S.save(fig, out, "FigS09_roc")


# --------------------------------------------------------------------------------------------- S10
def figS10_pca(out):
    from sklearn.decomposition import PCA
    T, cfg, d, run = engine()
    L = np.log2(d.X + 1.0)
    pc = PCA(2, svd_solver="full").fit(L.values)
    Z = pc.transform(L.values); ev = pc.explained_variance_ratio_
    site = pd.Series(np.asarray(d.site), index=d.X.index); y = np.asarray(d.y)

    def eta2(v, g):
        m = v.mean(); ss = ((v - m) ** 2).sum()
        return sum((g == k).sum() * (v[g == k].mean() - m) ** 2 for k in np.unique(g)) / ss
    stats = dict(site=[eta2(Z[:, i], site.values) for i in (0, 1)], label=[eta2(Z[:, i], y) for i in (0, 1)],
                 ev=list(ev))
    pd.Series({"PcaSiteOne": f"{100 * stats['site'][0]:.1f}", "PcaSiteTwo": f"{100 * stats['site'][1]:.1f}",
               "PcaLabelOne": f"{100 * stats['label'][0]:.1f}", "PcaLabelTwo": f"{100 * stats['label'][1]:.1f}",
               "PcaEvOne": f"{100 * ev[0]:.1f}", "PcaEvTwo": f"{100 * ev[1]:.1f}"}).to_json(
        os.path.join(out, "pca_stats.json"))
    with open(os.path.join(out, "figure_macros.tex"), "w") as fh:
        fh.write("% written by code/figures_v7/supp_figs.py (PCA of log2(x+1) values of all genes, 633 samples)\n")
        for k, v in (("PcaSiteOne", stats["site"][0]), ("PcaSiteTwo", stats["site"][1]), ("PcaLabelOne", stats["label"][0]),
                     ("PcaLabelTwo", stats["label"][1]), ("PcaEvOne", ev[0]), ("PcaEvTwo", ev[1])):
            fh.write(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{100 * v:.1f}}}\n")
    major = site.value_counts(); major = [s for s in major.index if major[s] >= 9]
    pal = ["#2a78d6", "#eb6834", "#14967a", "#9b59b6", "#c49a00"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL, 3.3), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.08))
    oth = ~site.isin(major).values
    a.scatter(Z[oth, 0], Z[oth, 1], s=5, color="#c3cad3", lw=0, label=f"other codes (n = {int(oth.sum())})")
    for s_, c in zip(major, pal):
        k = (site == s_).values
        a.scatter(Z[k, 0], Z[k, 1], s=5, color=c, lw=0, alpha=0.8, label=f"{s_} (n = {int(k.sum())})")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=6.2, markerscale=2, handletextpad=0.1,
             columnspacing=0.8)
    for cls, c, nm in ((0, S.HEALTHY, "healthy"), (1, S.NSCLC, "NSCLC")):
        k = y == cls
        b.scatter(Z[k, 0], Z[k, 1], s=5, color=c, lw=0, alpha=0.65, label=f"{nm} (n = {int(k.sum())})")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.2, markerscale=2, handletextpad=0.1)
    for ax_, nm, key in ((a, "Collection site", "site"), (b, "Disease status", "label")):
        ax_.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% of variance)")
        ax_.text(0.0, 1.0, f"{nm.lower()} explains {100 * stats[key][0]:.1f}% of PC1 and {100 * stats[key][1]:.1f}% of PC2",
                 transform=ax_.transAxes, fontsize=6.4, color=MUTED, va="bottom")
    a.set_ylabel(f"PC2 ({100 * ev[1]:.1f}% of variance)")
    S.panel(a, "A", "Collection site", y=1.07); S.panel(b, "B", "Disease status", y=1.07)
    return S.save(fig, out, "FigS10_pca")


# --------------------------------------------------------------------------------------------- S11
def figS11_sitescores(out):
    T = D.tab("R1_scores_by_site")
    groups = list(dict.fromkeys(T.site))
    fig, ax = plt.subplots(figsize=(S.FULL * 0.75, 2.9))
    rows = []
    for g in groups:
        for cls in (1, 0):
            q = T[(T.site == g) & (T.y == cls)]
            if len(q):
                rows.append((g, cls, q.p.values))
    rng = np.random.default_rng(3)
    for i, (g, cls, v) in enumerate(rows):
        yy = len(rows) - 1 - i
        col = S.NSCLC if cls == 1 else S.HEALTHY
        if len(v) >= 5:
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.add_patch(plt.Rectangle((q1, yy - 0.28), q3 - q1, 0.56, fc=col, alpha=0.18, ec=col, lw=0.8))
            ax.plot([med, med], [yy - 0.28, yy + 0.28], color=col, lw=1.4)
        ax.scatter(v, yy + rng.uniform(-0.2, 0.2, len(v)), s=4, color=col, alpha=0.55, lw=0, zorder=3)
        lab = g.replace(" (within-site CV)", " (within-site CV)")
        ax.text(-0.02, yy, f"{lab} · {'NSCLC' if cls else 'healthy'} (n = {len(v)})", ha="right", va="center",
                fontsize=6.5, transform=ax.get_yaxis_transform())
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Predicted probability of NSCLC (Vumc-trained model)")
    S.xgrid(ax)
    ax.legend(handles=[Patch(fc=S.NSCLC, alpha=0.5, label="NSCLC"), Patch(fc=S.HEALTHY, alpha=0.5, label="healthy")],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    return S.save(fig, out, "FigS11_sitescores")


# --------------------------------------------------------------------------------------------- S12
def figS12_harm(out):
    T = D.tab("R3sim_alignment_within_vumc")
    sets = [("auc_none", "no alignment", "#5b6675", "o"), ("auc_percohort_z", "per-cohort standardisation", S.NSCLC, "s"),
            ("auc_ref_combat_frozen", "reference-batch ComBat-seq", S.SEL["six_stage"], "D"),
            ("auc_joint_combat_retrained", "joint ComBat-seq + re-training", S.SEL["stability"], "^")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL, 2.6), gridspec_kw=dict(wspace=0.3))
    prev = sorted(T.prevalence.unique())
    for k, (c, lab, col, mk) in enumerate(sets):
        dx = (k - 1.5) * 0.022
        g = T.groupby("prevalence")[c]
        m, lo, hi = g.mean(), g.quantile(0.1), g.quantile(0.9)
        a.errorbar(np.array(prev) + dx, m, yerr=[m - lo, hi - m], fmt=mk, color=col, ms=3.8, elinewidth=0.8,
                   capsize=0, mec="white", mew=0.4, label=lab)
        if c != "auc_none":
            dd = (T[c] - T.auc_none) * 100
            gg = dd.groupby(T.prevalence)
            m2, lo2, hi2 = gg.mean(), gg.quantile(0.1), gg.quantile(0.9)
            b.errorbar(np.array(prev) + dx, m2, yerr=[m2 - lo2, hi2 - m2], fmt=mk, color=col, ms=3.8, elinewidth=0.8,
                       capsize=0, mec="white", mew=0.4)
    b.axhline(0, color=MUTED, lw=0.7, ls=(0, (2, 2)))
    for ax_ in (a, b):
        ax_.set_xticks(prev); ax_.set_xlabel("NSCLC prevalence of the 50-sample 'external' cohort"); S.ygrid(ax_)
    a.set_ylabel("AUC (mean, 10th–90th percentile)"); b.set_ylabel("Change from no alignment (AUC points)")
    S.panel(a, "A", "Discrimination"); S.panel(b, "B", "Effect of the alignment step")
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12), fontsize=6.6, handletextpad=0.3)
    return S.save(fig, out, "FigS12_harm")


# --------------------------------------------------------------------------------------------- S13
def figS13_rfecv(out):
    T = D.tab("DEV_rfecv_curve"); M = D.macros()
    fig, ax = plt.subplots(figsize=(S.HALF + 0.9, 2.4))
    ax.fill_between(T.n_features, T.mean_auc - T.sd_auc, T.mean_auc + T.sd_auc, color=S.DIM_L["P"], lw=0)
    ax.plot(T.n_features, T.mean_auc, color=S.SEL["six_stage"], lw=1.3)
    k, ka = D.num(M["RfecvK"]), D.num(M["RfecvAUC"]); pa = D.num(M["RfecvPanelAUC"])
    ax.plot(k, ka, "D", color=S.DIM["M"], ms=5, mec="white", mew=0.5, zorder=3)
    ax.annotate(f"RFECV optimum: {int(k)} genes, AUC {ka:.3f}", (k, ka), xytext=(8, 6), textcoords="offset points",
                fontsize=6.5, color=S.DIM["M"])
    ax.plot(27, pa, "s", color=S.NSCLC, ms=4.5, mec="white", mew=0.5, zorder=3)
    ax.annotate(f"27-gene panel: AUC {pa:.3f}", (27, pa), xytext=(8, -12), textcoords="offset points", fontsize=6.5,
                color=S.NSCLC)
    ax.set_xlabel(f"Genes retained (of {M['RfecvPool']})"); ax.set_ylabel("Five-fold AUC (mean ± SD)")
    ax.set_xlim(0, T.n_features.max() + 2); S.ygrid(ax)
    return S.save(fig, out, "FigS13_rfecv")


ALL = [figS01_stabprof, figS02_heatmap, figS03_contam, figS04_shap, figS05_featcount, figS06_ablation, figS07_de,
       figS09_roc, figS10_pca, figS11_sitescores, figS12_harm, figS13_rfecv]
