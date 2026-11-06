"""Main-text data figures (3-11), drawn from the saved results."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import data as D
import style as S
from style import INK, MUTED, FAINT, GRID, RULE


# --------------------------------------------------------------------------------------------- Figure 3
def fig_forest(out):
    T = D.tab("ALL_AUCs_with_95CI")
    short = ["Fixed panel, five-fold CV", "Fixed panel, log$_2$-CPM input", "Six-stage selection nested in CV",
             "mRMR–IFS–SVM, nested (same folds)",
             "Site-stratified five-fold CV", "Vumc only, five-fold CV", "Vumc only, selection nested",
             "Train outside Vumc → test Vumc", "Train in Vumc → test other sites",
             "Development configuration (22 genes)", "Log$_2$-CPM, no alignment", "Reference-batch ComBat-seq",
             "Joint ComBat-seq + re-training", "Single-sample ranks", "QC-filtered 24-gene panel",
             "Vumc-trained panel", "mRMR–IFS–SVM, no alignment",
             "GSE40791 (lung tissue)", "GSE68465 (lung tissue)"]
    groups = [("Discovery cohort  ·  GSE89843, n = 633", range(0, 4), "P"),
              ("Site-aware validation  ·  GSE89843", range(4, 9), "T"),
              ("Independent platelet cohort  ·  GSE68086, n = 94", range(9, 17), "P"),
              ("Transfer to lung tissue", range(17, 19), "T")]
    rows = []
    for g, idx, dim in groups:
        rows.append(("h", g, dim))
        rows += [("r", i, dim) for i in idx]
    n = len(rows)
    fig = plt.figure(figsize=(S.FULL, 0.215 * n + 0.55))
    ax = fig.add_axes([0.40, 0.06, 0.38, 0.88])
    y = n - 1
    for kind, v, dim in rows:
        if kind == "h":
            ax.axhspan(y - 0.5, y + 0.5, xmin=-0.76, xmax=1.62, color=S.DIM_L[dim], lw=0, clip_on=False, zorder=0)
            ax.text(-0.745, y, v, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7.5,
                    fontweight="bold", color=S.DIM[dim])
        else:
            r = T.iloc[v]
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=S.DIM[dim], lw=1.3, solid_capstyle="round", zorder=2)
            ax.plot(r.auc, y, "o", ms=4.2, color=S.DIM[dim], mec="white", mew=0.6, zorder=3)
            ax.text(-0.73, y, short[v], transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7.2)
            ax.text(1.04, y, f"{r.auc:.3f}", transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=7.2, color=INK)
            ax.text(1.20, y, f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]", transform=ax.get_yaxis_transform(), ha="left",
                    va="center", fontsize=7.0, color=MUTED)
        y -= 1
    ax.axvline(0.5, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(0.3, 1.0); ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks([0.3, 0.5, 0.7, 0.9, 1.0]); ax.set_xlabel("AUC with 95% CI")
    S.xgrid(ax)
    ax.text(-0.73, n - 0.45, "Evaluation setting", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=7.5, fontweight="bold")
    ax.text(1.04, n - 0.45, "AUC", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=7.5,
            fontweight="bold")
    ax.text(1.20, n - 0.45, "95% CI", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=7.5, fontweight="bold")
    return S.save(fig, out, "Fig03_forest")


# --------------------------------------------------------------------------------------------- Figure 4
def fig_locked(out):
    M = D.macros()
    C = D.tab("LK_calibration_bins"); F = D.tab("LK_feature_availability_summary")
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL, 2.55), gridspec_kw=dict(wspace=0.32))
    a.plot([0, 1], [0, 1], color=FAINT, lw=0.8, ls=(0, (3, 2)), zorder=1)
    spec = {"S0": ("No alignment", S.SEL["six_stage"], "o", -0.006), "S2": ("Reference-batch ComBat-seq", S.NSCLC, "s", 0.006)}
    for k, (lab, c, mk, dx) in spec.items():
        q = C[C.setting == k]
        a.errorbar(q.mean_pred + dx, q.obs, yerr=[q.obs - q.lo, q.hi - q.obs], fmt=mk + "-", color=c, ms=3.8,
                   lw=1.1, elinewidth=0.8, capsize=0, mec="white", mew=0.5, label=lab, zorder=3)
    a.set_xlim(-0.02, 1.0); a.set_ylim(-0.02, 1.02); a.set_aspect("equal")
    a.set_xlabel("Mean predicted probability (quintile)"); a.set_ylabel("Observed NSCLC fraction")
    a.legend(loc="upper left", handlelength=1.4, borderaxespad=0.1)
    ints = (M["LkNoneInt"].split(" ")[0], M["LkRefInt"].split(" ")[0])
    a.text(0.97, 0.03, f"calibration intercept\n{ints[0]} (no alignment)\n{ints[1]} (ComBat-seq)",
           transform=a.transAxes, ha="right", va="bottom", fontsize=6.4, color=MUTED, linespacing=1.35)
    S.ygrid(a); S.panel(a, "A", "Calibration in GSE68086")
    b.plot(F.n_available, F.imp, "-o", color=S.SEL["six_stage"], ms=3, lw=1.3, label="Locked model, missing genes imputed")
    b.plot(F.n_available, F.re, "-s", color=S.NSCLC, ms=3, lw=1.3, label="Re-trained on the available genes")
    b.plot(F.n_available, F.imp_lo, color=S.SEL["six_stage"], lw=0.8, ls=(0, (3, 2)))
    b.plot(F.n_available, F.re_lo, color=S.NSCLC, lw=0.8, ls=(0, (3, 2)))
    b.plot([], [], color=MUTED, lw=0.8, ls=(0, (3, 2)), label="5th percentile over draws")
    b.axhline(0.5, color=FAINT, lw=0.7, ls=(0, (2, 2)))
    b.set_xlabel("Panel genes measurable in GSE68086"); b.set_ylabel("External AUC")
    b.set_xticks(F.n_available.values[::2] if len(F) > 9 else F.n_available.values)
    b.set_ylim(0.4, 1.0); b.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02))
    S.ygrid(b); S.panel(b, "B", "Gene availability")
    return S.save(fig, out, "Fig04_locked")


# --------------------------------------------------------------------------------------------- Figure 5
def fig_nested(out):
    M = D.macros(); N = D.tab("R4_nested_folds")
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL, 2.4), gridspec_kw=dict(wspace=0.3))
    refs = [(D.num(M["CoreFixedAUC"]), "fixed panel", (0, ())), (D.num(M["NestedAUC"]), "nested, pooled", (0, (4, 2))),
            (D.num(M["ExtTwentySevenNoneAUC"]), "GSE68086", (0, (1, 1.5)))]
    for v, lab, ls in refs:
        a.axhline(v, color=MUTED, lw=0.8, ls=ls, zorder=1)
        below = lab.startswith("nested")
        a.text(5.45, v + (-0.003 if below else 0.002), f"{lab} {v:.3f}", fontsize=6.5, color=MUTED,
               va="top" if below else "bottom", ha="right")
    a.plot(N.fold, N.auc, "o", color=S.SEL["six_stage"], ms=5, mec="white", mew=0.6, zorder=3)
    a.set_xlim(0.5, 5.5); a.set_ylim(0.8, 1.0); a.set_xticks(range(1, 6))
    a.set_xlabel("Outer fold"); a.set_ylabel("Held-out AUC")
    S.ygrid(a); S.panel(a, "A", "Discrimination of the re-derived panels")
    wmin, wmax, wmean = D.num(M["WeightJMin"]), D.num(M["WeightJMax"]), D.num(M["WeightJMean"])
    b.axhspan(wmin, wmax, color=S.DIM_M["M"], lw=0, zorder=0)
    b.axhline(wmean, color=S.DIM["M"], lw=0.9)
    b.text(0.62, wmax + 0.02, "composite weights perturbed (±20%)", fontsize=6.5, color=S.DIM["M"], va="bottom")
    ej = D.num(M["StabExpJ"])
    b.axhline(ej, color=MUTED, lw=0.8, ls=(0, (1, 1.5)))
    b.text(0.62, ej + 0.02, "two random 27-gene sets", fontsize=6.5, color=MUTED, va="bottom")
    b.plot(N.fold, N.jaccard_vs_original, "o", color=S.SEL["six_stage"], ms=5, mec="white", mew=0.6, zorder=3)
    b.text(5.4, N.jaccard_vs_original.mean() + 0.08, "patients resampled\n(fold panels)", fontsize=6.5,
           color=S.SEL["six_stage"], ha="right", va="bottom")
    b.set_xlim(0.5, 5.5); b.set_ylim(-0.02, 1.05); b.set_xticks(range(1, 6))
    b.set_xlabel("Outer fold"); b.set_ylabel("Jaccard similarity with 27-gene panel")
    S.ygrid(b); S.panel(b, "B", "Gene identity of the re-derived panels")
    return S.save(fig, out, "Fig05_nested")


# --------------------------------------------------------------------------------------------- Figure 6
def fig_random(out):
    M = D.macros(); R = D.tab("R9_random_panels")
    cur = D.num(M.get("CoreFixedAUC", "0.9626"))
    fig, axs = plt.subplots(1, 2, figsize=(S.FULL, 2.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    bins = np.arange(0.55, 1.0001, 0.01)
    marks = []
    for ax, kind, lab, letter in ((axs[0], "unmatched", "Random 27-gene panels", "A"),
                                  (axs[1], "matched", "Random panels matched on effect size", "B")):
        v = R[R.kind == kind].auc_foldmean
        ax.hist(v, bins=bins, color=S.DIM_M["M"], ec="white", lw=0.4)
        ax.axvline(cur, color=S.NSCLC, lw=1.2)
        marks.append(ax.text(cur - 0.006, 1, f"27-gene panel\n{cur:.3f}", ha="right", va="top", fontsize=6.6,
                             color=S.NSCLC))
        ax.text(0.56, 0.95, f"n = {len(v):,}\nmedian {np.median(v):.3f}\nmaximum {v.max():.3f}",
                transform=ax.get_xaxis_transform(), va="top", fontsize=6.6, color=MUTED, linespacing=1.35)
        ax.set_xlim(0.55, 1.0); ax.set_xlabel("Mean five-fold AUC")
        S.ygrid(ax); S.panel(ax, letter, lab)
    axs[0].set_ylabel("Number of panels")
    ymax = max(a.get_ylim()[1] for a in axs)
    for ax, t in zip(axs, marks):
        ax.set_ylim(0, ymax); t.set_y(ymax * 0.95)
    return S.save(fig, out, "Fig06_random")


# --------------------------------------------------------------------------------------------- Figure 7
def fig_pathstab(out):
    Sm = D.tab("PS_split_half_summary"); G = D.tab("PS_panel_gene_replication"); P = D.tab("PS_gene_set_replication")
    fig = plt.figure(figsize=(S.FULL, 4.7))
    def title(x, y, t):
        fig.text(x, y, t, fontsize=8.5, fontweight="bold", va="bottom", ha="left")
    a = fig.add_axes([0.33, 0.745, 0.65, 0.19])
    cats = [("gene_top27_J", "Top 27 genes", S.SEL["six_stage"]), ("gene_top200_J", "Top 200 genes", S.SEL["six_stage"]),
            ("unadj_path_sig_J", "Significant gene sets", S.DIM["M"]),
            ("adj_path_sig_J", "Significant gene sets (contamination-adjusted)", S.DIM["M"])]
    rng = np.random.default_rng(1)
    for i, (c, lab, col) in enumerate(cats):
        yy = len(cats) - 1 - i
        v = Sm[c].values
        a.scatter(v, yy + rng.uniform(-0.16, 0.16, len(v)), s=7, color=col, alpha=0.5, lw=0)
        a.plot([np.median(v)] * 2, [yy - 0.3, yy + 0.3], color=INK, lw=1.5)
    a.set_yticks(range(len(cats))); a.set_yticklabels([c[1] for c in cats][::-1], fontsize=7)
    a.tick_params(axis="y", length=0); a.spines["left"].set_visible(False)
    a.set_xlim(0, 1); a.set_ylim(-0.6, len(cats) - 0.4)
    a.set_xlabel(f"Jaccard similarity between the two halves ({len(Sm)} splits; bar = median)")
    S.xgrid(a)
    title(0.01, 0.955, "A   Agreement between two independent halves of Vumc")
    b = fig.add_axes([0.11, 0.115, 0.33, 0.475])
    T = G[G.expressed_in_vumc].sort_values(["frac_splits_rep_both", "frac_halves_sig"], ascending=[True, True])
    yy = np.arange(len(T))
    b.hlines(yy, 0, T.frac_splits_rep_both, color=S.DIM_M["P"], lw=2.2)
    b.plot(T.frac_splits_rep_both, yy, "o", color=S.SEL["six_stage"], ms=3.6)
    b.set_yticks(yy); b.set_yticklabels(T.gene, fontsize=6.6); b.tick_params(axis="y", length=0)
    b.set_xlim(0, 1.05); b.set_ylim(-0.7, len(T) - 0.3); b.set_xlabel("Fraction of splits significant in both halves")
    ne = G[~G.expressed_in_vumc].gene.tolist()
    fig.text(0.01, 0.005, f"Not testable in Vumc ({len(ne)} genes): " + ", ".join(ne) + ".", fontsize=6.2,
             color=MUTED, ha="left", va="bottom")
    S.xgrid(b)
    title(0.01, 0.615, f"B   Panel genes ({len(T)} testable)")
    c = fig.add_axes([0.73, 0.115, 0.25, 0.40])
    sets = [("REACTOME_PLATELET_ACTIVATION_SIGNALING_AND_AGGREGATION", "Platelet activation"),
            ("REACTOME_RESPONSE_TO_ELEVATED_PLATELET_CYTOSOLIC_CA2", "Platelet cytosolic Ca$^{2+}$"),
            ("REACTOME_HEMOSTASIS", "Haemostasis"), ("HALLMARK_COAGULATION", "Coagulation")]
    for i, (term, lab) in enumerate(sets):
        yv = len(sets) - 1 - i
        for rk, dy, mk, fc in (("unadj", 0.14, "o", S.DIM["M"]), ("adj", -0.14, "D", "white")):
            q = P[(P.ranking == rk) & (P.Term == term)]
            v = float(q.frac_splits_up_both.iloc[0]) if len(q) else np.nan
            c.hlines(yv + dy, 0, v, color=S.DIM_M["M"], lw=2.2)
            c.plot(v, yv + dy, mk, ms=4, color=S.DIM["M"], mfc=fc, mew=1.0)
    c.set_yticks(range(len(sets))); c.set_yticklabels([s_[1] for s_ in sets][::-1], fontsize=7)
    c.tick_params(axis="y", length=0)
    c.set_xlim(0, 1.05); c.set_ylim(-0.6, len(sets) - 0.4)
    c.set_xlabel("Fraction of splits enriched in both halves")
    fig.legend(handles=[Line2D([], [], marker="o", color=S.DIM["M"], ls="", ms=4, label="unadjusted ranking"),
                        Line2D([], [], marker="D", color=S.DIM["M"], mfc="white", ls="", ms=4,
                               label="contamination-adjusted ranking")], loc="upper left",
               bbox_to_anchor=(0.53, 0.60), ncol=1, fontsize=6.6, handletextpad=0.3)
    S.xgrid(c)
    title(0.53, 0.615, "C   Platelet gene sets")
    return S.save(fig, out, "Fig07_pathstab")


# --------------------------------------------------------------------------------------------- Figure 8
def fig_genesite(out):
    G = D.tab("R1_genomewide_site_vs_disease"); P = D.tab("R1_panel_site_vs_disease")
    t2 = G.tier2.astype(str).str.lower().eq("true")
    G = G[t2 & ~G.gene.isin(P.gene)]
    fig, ax = plt.subplots(figsize=(S.HALF + 0.6, 3.1))
    ax.scatter(G.eps2_site_max, G.eps2_disease_vumc, s=2, color="#c3cad3", lw=0, alpha=0.6, rasterized=True,
               zorder=1)
    lim = max(G.eps2_site_max.max(), G.eps2_disease_vumc.max(), P.eps2_site_max.max(), P.eps2_disease_vumc.max()) * 1.04
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.text(lim * 0.97, lim * 0.93, "site = disease", rotation=0, fontsize=6.4, color=MUTED, ha="right")
    dis = P[P.dominant == "disease"]; site = P[P.dominant != "disease"]
    ax.scatter(site.eps2_site_max, site.eps2_disease_vumc, s=16, facecolor="white", edgecolor=S.NSCLC, lw=0.9,
               marker="s", zorder=3, label=f"Panel gene, site-dominant ({len(site)})")
    ax.scatter(dis.eps2_site_max, dis.eps2_disease_vumc, s=18, color=S.SEL["six_stage"], edgecolor="white",
               lw=0.5, zorder=4, label=f"Panel gene, disease-dominant ({len(dis)})")
    offs = {"PTMS": (5, 2), "P2RY10": (5, -3), "HSD17B3": (5, -1), "MGAT5": (5, -2), "TNFRSF17": (4, 4),
            "PCSK5": (2, 7)}
    for _, r in P[P.gene.isin(offs)].iterrows():
        ax.annotate(r.gene, (r.eps2_site_max, r.eps2_disease_vumc), xytext=offs[r.gene], textcoords="offset points",
                    fontsize=6.4, color=INK)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Largest within-class site effect ($\\varepsilon^2$)")
    ax.set_ylabel("Disease effect within Vumc ($\\varepsilon^2$)")
    ax.scatter([], [], s=6, color="#c3cad3", label=f"other Tier 2 genes ({len(G):,})")
    ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    S.ygrid(ax)
    return S.save(fig, out, "Fig08_genesite")


# --------------------------------------------------------------------------------------------- Figure 9
def fig_cards(out):
    M = D.export_json()
    rows = [("Six-stage (paper)", "Four", "Six", "Six"), ("Mann–Whitney top 27", "Mw", "Mw", "Mw"),
            ("L1-logistic top 27", "Lone", "Lone", "Lone"), ("Stability selection", "Stab", "Stab", "Stab"),
            ("mRMR–IFS–SVM", "Mrmr", "Mrmr", "Mrmr")]
    cols = [("Nested\nAUC", "P", +1, 3), ("External\nAUC", "P", +1, 3), ("|External\nintercept|", "P", -1, 2),
            ("$\\hat\\Phi$\nfolds", "M", +1, 3), ("$\\hat\\Phi$\nreplicates", "M", +1, 3), ("$\\psi$", "M", -1, 3),
            ("Vumc\nnested AUC", "T", +1, 3), ("Vumc →\nother sites", "T", +1, 3),
            ("Site-\ndominant (%)", "T", -1, 0), ("Measurable\n(%)", "T", +1, 0)]
    V, names = [], []
    for name, b, r, c in rows:
        names.append(name)
        V.append([D.num(M.get(f"VxBm{b}AUC")), D.num(M.get(f"VxCd{c}Ext")), abs(D.num(M.get(f"VxCd{c}ExtInt"))),
                  D.num(M.get(f"VxBm{b}Phi")), D.num(M.get(f"VxRp{r}Phi")), D.num(M.get(f"VxRp{r}Psi")),
                  D.num(M.get(f"VxCd{c}Vumc")), D.num(M.get(f"VxCd{c}Transfer")), D.num(M.get(f"VxCd{c}SiteDom")),
                  D.num(M.get(f"VxCd{c}Cov"))])
    names.append("Pathway model")
    V.append([D.num(M.get("VxPwAUC")), D.num(M.get("VxPwExtAUC")), abs(D.num(str(M.get("VxPwExtInt", "")).split(" (")[0])),
              D.num(M.get("VxPwPhi")), D.num(M.get("VxPwRpPhi")), D.num(M.get("VxPwRpPsi")), D.num(M.get("VxPwVumcAUC")),
              D.num(M.get("VxPwTransferAUC")), np.nan, 100.0])
    V = np.array(V, float); nr, nc = V.shape
    R = np.full_like(V, np.nan)
    for j, (_, _, sgn, _) in enumerate(cols):
        ok = ~np.isnan(V[:, j]); R[ok, j] = pd.Series(-sgn * V[ok, j]).rank(method="min").values
    ramps = {"P": ["#2b5c8a", "#4d7aa6", "#7c9fc2", "#a9c1db", "#cfdcec", "#e9eff7"],
             "M": ["#1f7a6d", "#3f958a", "#6cb0a6", "#9ccbc4", "#c7e2de", "#e6f2f0"],
             "T": ["#6e4f96", "#8a6eae", "#a891c5", "#c5b5da", "#ded4ea", "#f0ebf6"]}
    fig = plt.figure(figsize=(S.FULL, 2.75))
    ax = fig.add_axes([0.185, 0.04, 0.81, 0.70])
    for i in range(nr):
        y0 = nr - 1 - i - (0.18 if i == nr - 1 else 0)
        for j in range(nc):
            dim = cols[j][1]
            if np.isnan(V[i, j]):
                ax.add_patch(plt.Rectangle((j + .04, y0 + .06), .92, .88, color="#f1f2f4", lw=0))
                ax.text(j + .5, y0 + .5, "–", ha="center", va="center", fontsize=7.5, color=FAINT)
                continue
            k = int(R[i, j]) - 1
            colr = ramps[dim][min(k, 5)]
            ax.add_patch(plt.Rectangle((j + .04, y0 + .06), .92, .88, color=colr, lw=0))
            txt = f"{V[i, j]:.{cols[j][3]}f}"
            ax.text(j + .5, y0 + .5, txt, ha="center", va="center", fontsize=7.2,
                    color="white" if k <= 1 else INK, fontweight="bold" if k == 0 else "normal")
        ax.text(-0.12, y0 + .5, names[i], ha="right", va="center", fontsize=7.4,
                fontweight="bold" if i == 0 else "normal")
    for j, (lab, dim, sgn, _) in enumerate(cols):
        ax.text(j + .5, nr + 0.12, lab, ha="center", va="bottom", fontsize=6.5, color=INK, linespacing=1.15)
    for dim, a0, a1 in (("P", 0, 3), ("M", 3, 6), ("T", 6, 10)):
        ax.plot([a0 + .06, a1 - .06], [nr + 1.02, nr + 1.02], color=S.DIM[dim], lw=1.6, clip_on=False,
                solid_capstyle="butt")
        ax.text((a0 + a1) / 2, nr + 1.12, S.DIM_NAME[dim], ha="center", va="bottom", fontsize=7.8,
                fontweight="bold", color=S.DIM[dim])
    ax.set_xlim(0, nc); ax.set_ylim(-0.25, nr + 1.6); ax.axis("off")
    fig.text(0.99, -0.02, "Shade = rank within the column (darkest = best; bold = best value). Lower is better for "
             "|external intercept|, ψ and site-dominant genes.", ha="right", fontsize=6.4, color=MUTED)
    return S.save(fig, out, "Fig09_cards")


# --------------------------------------------------------------------------------------------- Figure 10
def fig_link(out):
    d = D.run_results("replicates")
    St = pd.read_csv(os.path.join(d, "stability.csv")); A = pd.read_csv(os.path.join(d, "agreement.csv"))
    I = pd.read_csv(os.path.join(d, "identity_share.csv"))
    T = St.merge(A, on="method").merge(I[["method", "share", "lo", "hi"]], on="method", how="left")
    P = json.load(open(os.path.join(D.latest("pathway_model"), "summary.json")))["replicates"]
    Sc = pd.read_csv(os.path.join(d, "scores.csv"))
    fig, ax = plt.subplots(2, 2, figsize=(S.FULL, 5.0), gridspec_kw=dict(hspace=0.48, wspace=0.34))
    a, b, c, e = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    order = ["six_stage", "mw_topk", "l1_topk", "stability", "mrmr_ifs_svm", "random"]
    offA = {"six_stage": (5, -9), "mw_topk": (0, -12), "l1_topk": (-6, 6), "stability": (-6, -11),
            "mrmr_ifs_svm": (7, 1), "random": (5, 2), "pathway": (5, -8)}
    offA["stability"] = (6, -10)
    haA = {"mw_topk": "center", "l1_topk": "right"}
    offB = {"six_stage": (6, -2), "mw_topk": (-6, 6), "l1_topk": (-5, -10), "stability": (6, 2), "random": (6, -2),
            "pathway": (6, -3)}
    haB = {"mw_topk": "right", "l1_topk": "right"}
    for m in order:
        r = T[T.method == m].iloc[0]; col = S.SEL[m]; mk = S.SEL_MARK[m]
        a.errorbar(r.phi, r.auc_mean, yerr=r.auc_sd_within_fold, xerr=[[r.phi - r.phi_lo], [r.phi_hi - r.phi]],
                   fmt=mk, color=col, ms=4.5, elinewidth=0.8, capsize=0, mec="white", mew=0.5, zorder=3)
        a.annotate(S.SEL_SHORT[m], (r.phi, r.auc_mean), xytext=offA[m], textcoords="offset points", fontsize=6.6,
                   color=col, ha=haA.get(m, "left"))
        if not np.isnan(r.share):
            b.errorbar(r.phi, r.share, yerr=[[r.share - r.lo], [r.hi - r.share]], fmt=mk, color=col, ms=4.5,
                       elinewidth=0.8, capsize=0, mec="white", mew=0.5, zorder=3)
            b.annotate(S.SEL_SHORT[m], (r.phi, r.share), xytext=offB[m], textcoords="offset points", fontsize=6.6,
                       color=col, ha=haB.get(m, "left"))
            g = 100 * (r.bagged_auc - r.auc_mean)
            e.plot(r.share, g, mk, color=col, ms=5, mec="white", mew=0.5, zorder=3)
            e.annotate(S.SEL_SHORT[m], (r.share, g), xytext=(6, -2), textcoords="offset points", fontsize=6.6,
                       color=col)
    col = S.SEL["pathway"]
    a.errorbar(P["phi"], P["auc_mean"], yerr=P["auc_sd_within_fold"], fmt="*", color=col, ms=7, elinewidth=0.8,
               capsize=0, mec="white", mew=0.4, zorder=3)
    a.annotate("Pathway model", (P["phi"], P["auc_mean"]), xytext=offA["pathway"], textcoords="offset points",
               fontsize=6.6, color=col)
    b.errorbar(P["phi"], P["psi"], yerr=[[P["psi"] - P["psi_lo"]], [P["psi_hi"] - P["psi"]]], fmt="*", color=col,
               ms=7, elinewidth=0.8, capsize=0, mec="white", mew=0.4, zorder=3)
    b.annotate("Pathway model\n(gene sets)", (P["phi"], P["psi"]), xytext=offB["pathway"], textcoords="offset points",
               fontsize=6.6, color=col, va="top")
    a.set_xlim(-0.04, 1.02); a.set_xlabel("Identity stability $\\hat\\Phi$"); a.set_ylabel("Mean held-out AUC")
    a.set_ylim(0.76, 0.99)
    S.ygrid(a); S.panel(a, "A", "Discrimination vs identity stability")
    b.set_xlim(-0.04, 1.02); b.set_ylim(0, 1); b.set_xlabel("Identity stability $\\hat\\Phi$")
    b.set_ylabel("$\\psi$ (share of prediction variance)")
    S.ygrid(b); S.panel(b, "B", "Identity stability and $\\psi$")
    lg = lambda p_: np.log(np.clip(p_, 1e-6, 1 - 1e-6) / (1 - np.clip(p_, 1e-6, 1 - 1e-6)))
    v = {}
    for m in ("six_stage", "fixed"):
        s_ = Sc[Sc.method == m].copy(); s_["lg"] = lg(s_.score)
        v[m] = s_.groupby("sample").lg.var(ddof=1)
    j = pd.concat(v, axis=1).dropna()
    yb = Sc.drop_duplicates("sample").set_index("sample").y.reindex(j.index)
    for k_, nm, cc in ((0, "Healthy", S.HEALTHY), (1, "NSCLC", S.NSCLC)):
        q = j[yb == k_]
        c.scatter(q.fixed, q.six_stage, s=5, alpha=0.55, color=cc, lw=0, label=f"{nm} ({len(q)})", rasterized=True)
    lo_, hi_ = max(j.min().min(), 1e-3) / 1.5, j.max().max() * 1.5
    c.plot([lo_, hi_], [lo_, hi_], color=MUTED, lw=0.7, ls=(0, (3, 2)))
    c.text(hi_ / 1.6, hi_ / 3.6, "equal\nvariance", fontsize=6.2, color=MUTED, ha="right", va="top")
    c.set_xscale("log"); c.set_yscale("log"); c.set_xlim(lo_, hi_); c.set_ylim(lo_, hi_)
    above = float((j.six_stage > j.fixed).mean())
    c.set_xlabel("Variance of predicted log-odds, panel fixed"); c.set_ylabel("Variance, panel re-selected")
    c.legend(loc="upper left", markerscale=2, handletextpad=0.2)
    c.text(0.97, 0.04, f"{100 * above:.0f}% of patients above the line", transform=c.transAxes, ha="right",
           fontsize=6.5, color=MUTED)
    S.panel(c, "C", f"Per held-out patient (n = {len(j)})")
    e.set_xlabel("$\\psi$"); e.set_ylabel("Gain from selection bagging (AUC points)")
    e.set_xlim(0.45, 1.0); S.ygrid(e); S.panel(e, "D", "$\\psi$ and the gain from bagging")
    return S.save(fig, out, "Fig10_link")


# --------------------------------------------------------------------------------------------- Figure 11
def fig_general(out):
    G = pd.read_csv(os.path.join(D.V2, "runs", "export", "V2_generalisation.csv"))
    M = D.export_json()
    order = ["GSE89843", "GSE68086", "Singh", "Gordon"]
    task = {"GSE89843": "NSCLC vs healthy donors\nplatelet RNA-seq, n = 633",
            "GSE68086": "six cancer types vs healthy\nplatelet RNA-seq, n = 280",
            "Singh": "prostate tumour vs normal\nmicroarray, n = 102",
            "Gordon": "mesothelioma vs lung adeno-\ncarcinoma; microarray, n = 181"}
    fig, ax = plt.subplots(2, 4, figsize=(S.FULL, 3.95), sharex="row", sharey="row",
                           gridspec_kw=dict(hspace=0.78, wspace=0.12))
    sels = ["six_stage", "mw_topk", "l1_topk", "stability", "random"]
    for k, ds in enumerate(order):
        g = G[G.dataset == ds]
        a, b = ax[0, k], ax[1, k]
        for m in sels:
            r = g[g.method == m]
            if r.empty or np.isnan(r.psi.iloc[0]):
                continue
            r = r.iloc[0]
            a.plot(r.phi, r.psi, S.SEL_MARK[m], color=S.SEL[m], ms=4.6, mec="white", mew=0.5, zorder=3)
            b.plot(r.psi, 100 * r.gain, S.SEL_MARK[m], color=S.SEL[m], ms=4.6, mec="white", mew=0.5, zorder=3)
        fx = g[g.method == "fixed"].iloc[0]
        b.plot(0, 100 * fx.gain, "P", color=S.SEL["fixed"], ms=4.6, mec="white", mew=0.4, zorder=3)
        b.axhline(0, color=RULE, lw=0.6, zorder=1)
        a.set_title(f"{ds}", fontsize=7.6, fontweight="bold", loc="left", pad=22)
        a.text(0, 1.03, task[ds], transform=a.transAxes, fontsize=6.0, color=MUTED, va="bottom", linespacing=1.25)
        S.ygrid(a); S.ygrid(b)
        a.set_xlim(-0.05, 1.0); a.set_ylim(0, 1); b.set_xlim(-0.05, 1.0)
        a.set_xticks([0, 0.5, 1]); b.set_xticks([0, 0.5, 1])
    ax[0, 0].set_ylabel("$\\psi$"); ax[1, 0].set_ylabel("Bagging gain\n(AUC points)")
    for k in range(4):
        ax[0, k].set_xlabel("Identity stability $\\hat\\Phi$", fontsize=6.8)
        ax[1, k].set_xlabel("$\\psi$", fontsize=6.8)
    rp = M.get("VxGenRhoPhiPsi", "").replace("$-$", "\u2212"); rw = M.get("VxGenRhoPhiPsiWithin", "").replace("$-$", "\u2212")
    fig.text(0.01, 0.985, f"A   Identity stability and \u03c8   (Spearman {rp} over all pairs; {rw} within data sets)",
             fontsize=8.3, fontweight="bold", va="top")
    fig.text(0.01, 0.50, f"B   \u03c8 and the gain from selection bagging   (Spearman {M.get('VxGenRhoPsiGain', '')} over all "
             f"pairs; {M.get('VxGenRhoPsiGainWithin', '')} within data sets)", fontsize=8.3, fontweight="bold", va="top")
    hs = [Line2D([], [], marker=S.SEL_MARK[m], color=S.SEL[m], ls="", ms=4.8, mec="white", mew=0.5,
                 label=S.SEL_SHORT[m]) for m in sels]
    hs.append(Line2D([], [], marker="P", color=S.SEL["fixed"], ls="", ms=4.6, label="Panel fixed (\u03c8 = 0)"))
    fig.legend(handles=hs, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04), fontsize=6.8,
               handletextpad=0.3, columnspacing=1.3)
    fig.subplots_adjust(top=0.84, bottom=0.14)
    return S.save(fig, out, "Fig11_general")


ALL = [fig_forest, fig_locked, fig_nested, fig_random, fig_pathstab, fig_genesite, fig_cards, fig_link, fig_general]
