"""Conceptual figures: the PMTRD framework (Figure 1), the psi design (Figure 2), the six selection stages
(Figure S8) and the exploratory classifier benchmark (Figure S14). Coordinates are in inches on the page.
Counts are the values reported in the Methods and the Supplementary Methods."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import data as D
import style as S
from style import INK, MUTED, FAINT, RULE


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def card(ax, x, y, w, h, title, dim=None, fc=None, ec=None, title_size=7.8, head_h=0.27, sym=None):
    """Rounded box with a tinted header strip."""
    ec = ec or (S.DIM[dim] if dim else RULE)
    fc = fc or (S.DIM_L[dim] if dim else "white")
    S.box(ax, x, y, w, h, fc="white", ec=ec, lw=0.8, r=0.05)
    S.box(ax, x, y + h - head_h, w, head_h, fc=fc, ec="none", lw=0, r=0.05, z=1.5)
    ax.plot([x + 0.01, x + w - 0.01], [y + h - head_h] * 2, color=ec, lw=0.5, zorder=1.6)
    ax.text(x + 0.09, y + h - head_h / 2, title, ha="left", va="center", fontsize=title_size, fontweight="bold",
            color=S.DIM[dim] if dim else INK, zorder=3)
    if sym:
        ax.text(x + w - 0.09, y + h - head_h / 2, sym, ha="right", va="center", fontsize=8.5,
                color=S.DIM[dim] if dim else INK, zorder=3)


def bullets(ax, x, y, items, dy=0.175, size=6.7, color=INK, dot=None):
    """Bulleted lines; an item that starts with '~' continues the previous bullet (no dot)."""
    for k, t in enumerate(items):
        yy = y - k * dy
        if not t.startswith("~"):
            ax.plot(x, yy, "o", ms=2.0, color=dot or MUTED, zorder=3)
        ax.text(x + 0.08, yy, t.lstrip("~"), ha="left", va="center", fontsize=size, color=color, zorder=3)


# --------------------------------------------------------------------------------------------- Figure 1
def fig_framework(out):
    W, H = S.FULL, 4.2
    fig, ax = canvas(W, H)
    top = H - 0.08
    # discovery cohort
    card(ax, 0.02, top - 1.18, 1.72, 1.18, "Discovery cohort $D$")
    bullets(ax, 0.12, top - 0.45, ["GSE89843, platelet RNA-seq", "633 samples: 402 NSCLC, 231 healthy",
                                   "28,089 genes", "16 site codes; only Vumc", "~contains both classes"],
            dy=0.155, size=6.5)
    S.arrow(ax, 1.76, top - 0.6, 1.9, top - 0.6)
    # selection procedure
    x0, w0 = 1.92, 3.3
    card(ax, x0, top - 1.18, w0, 1.18, "Fold-recomputable selection procedure $\\mathcal{A}$")
    chips = ["feature\nimportance", "Mann–\nWhitney", "LASSO ∩\nelastic net", "coding\nfilter",
             "mRMR;\nPPI term", "bootstrap\nstability", "composite\nscore"]
    cw, gap = 0.42, 0.04
    for k, t in enumerate(chips):
        cx = x0 + 0.08 + k * (cw + gap)
        S.box(ax, cx, top - 0.9, cw, 0.5, fc="#f3f5f8", ec=RULE, lw=0.6, r=0.04, z=2)
        ax.text(cx + cw / 2, top - 0.5, f"{k}", ha="center", va="center", fontsize=6.4, fontweight="bold",
                color=INK, zorder=3)
        ax.text(cx + cw / 2, top - 0.72, t, ha="center", va="center", fontsize=5.5, color=INK, linespacing=1.05,
                zorder=3)
        if k < len(chips) - 1:
            S.arrow(ax, cx + cw + 0.003, top - 0.65, cx + cw + gap - 0.003, top - 0.65, lw=0.6, ms=3.5)
    ax.text(x0 + 0.08, top - 1.05, "every data-dependent step re-run inside each outer training fold; random steps seeded",
            fontsize=6.0, color=MUTED, va="center")
    S.arrow(ax, x0 + w0 + 0.02, top - 0.6, x0 + w0 + 0.2, top - 0.6)
    # panel
    x1 = x0 + w0 + 0.22
    card(ax, x1, top - 1.18, W - x1 - 0.02, 1.18, "Candidate panel")
    ax.text(x1 + 0.12, top - 0.64, "27", fontsize=17, fontweight="bold", color=INK, va="center")
    ax.text(x1 + 0.5, top - 0.64, "genes, selected\non all 633\nsamples", fontsize=6.3, color=INK, va="center",
            linespacing=1.2)
    ax.text(x1 + 0.12, top - 1.05, "the object assessed", fontsize=6.0, color=MUTED, va="center")
    # bus to the three dimensions
    yb = top - 1.42
    ax.plot([x0 + w0 / 2, x0 + w0 / 2], [top - 1.18, yb], color=MUTED, lw=0.8)
    cols = [(0.02, "P"), (2.2, "M"), (4.38, "T")]
    cwid = 2.1
    ax.plot([cols[0][0] + cwid / 2, cols[2][0] + cwid / 2], [yb, yb], color=MUTED, lw=0.8)
    ybox_top = yb - 0.16
    hbox = 1.78
    content = {
        "P": ("Predictive reliability", "$\\mathcal{R}_P$", "Does discrimination survive honest evaluation?",
              ["nested AUC; selection optimism", "comparator on identical folds", "locked model in GSE68086 (n = 94)",
               "calibration, sensitivity, specificity", "~at a pre-set threshold"]),
        "M": ("Molecular reliability", "$\\mathcal{R}_M$", "Do the selected features replicate?",
              ["identity stability: Jaccard, Nogueira $\\hat\\Phi$", "random and matched null panels",
               "gene and pathway replication", "~in split halves of one site",
               "$\\psi$: does re-selection move predictions?"]),
        "T": ("Technical reliability", "$\\mathcal{R}_T$", "Does the signal survive technical variation?",
              ["site × label test; within-site nested AUC", "site-free transfer; site vs disease per gene",
               "contamination: QC panels (TOST)", "harmonisation leakage audit",
               "quantification coverage; gene availability"]),
    }
    for x, dim in cols:
        S.arrow(ax, x + cwid / 2, yb, x + cwid / 2, ybox_top + 0.01, color=MUTED, lw=0.8)
        name, sym, q, items = content[dim]
        card(ax, x, ybox_top - hbox, cwid, hbox, name, dim=dim, sym=sym)
        ax.text(x + 0.09, ybox_top - 0.44, q, fontsize=6.5, color=S.DIM[dim], fontstyle="italic", va="center")
        bullets(ax, x + 0.13, ybox_top - 0.70, items, dy=0.205, size=6.4, dot=S.DIM[dim])
    # reliability card
    yc = ybox_top - hbox - 0.2
    for x, dim in cols:
        S.arrow(ax, x + cwid / 2, ybox_top - hbox, x + cwid / 2, yc + 0.01, color=MUTED, lw=0.8)
    S.box(ax, 0.02, yc - 0.52, W - 0.04, 0.52, fc="#f6f7f9", ec=RULE, lw=0.8, r=0.05)
    ax.text(0.14, yc - 0.26, "PMTRD reliability card", fontsize=8, fontweight="bold", va="center")
    for k, dim in enumerate("PMT"):
        cx = 1.85 + k * 0.62
        S.box(ax, cx, yc - 0.4, 0.52, 0.28, fc=S.DIM_L[dim], ec=S.DIM[dim], lw=0.7, r=0.04, z=2)
        ax.text(cx + 0.26, yc - 0.26, f"$\\mathcal{{R}}_{dim}$", fontsize=8, color=S.DIM[dim], ha="center",
                va="center", zorder=3)
    ax.text(3.82, yc - 0.18, "every quantity with an estimator and a 95% confidence interval", fontsize=6.6,
            va="center")
    ax.text(3.82, yc - 0.36, "reported side by side, not combined into one score", fontsize=6.6, color=MUTED,
            va="center")
    return S.save(fig, out, "Fig01_framework")


# --------------------------------------------------------------------------------------------- Figure 2
def fig_psidesign(out, M=None):
    d = D.run_results("replicates")
    Sc = pd.read_csv(os.path.join(d, "scores.csv")); U = pd.read_csv(os.path.join(d, "units.csv"))
    thr = U.set_index(["fold", "rep", "method"]).threshold
    Sc["thr"] = [thr[(f, r, m)] for f, r, m in zip(Sc.fold, Sc.rep, Sc.method)]
    Sc["lab"] = (Sc.score >= Sc.thr).astype(int)
    W, H = S.FULL, 5.25
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0.44, 1, 0.56]); ax.set_xlim(0, W); ax.set_ylim(0, H * 0.56); ax.axis("off")
    h = H * 0.56
    ax.text(0.02, h - 0.1, "A   The control-arm design", fontsize=8.5, fontweight="bold", va="top")
    ax.text(W - 0.02, h - 0.1, "repeated in each of the 5 outer folds", fontsize=6.5, color=MUTED, va="top", ha="right")
    # training fold and subsamples
    card(ax, 0.02, 0.62, 1.38, 1.98, "Outer fold $k$")
    for k in range(5):
        S.box(ax, 0.13 + k * 0.235, 2.0, 0.205, 0.17, fc="#c9d3df" if k < 4 else S.DIM_M["T"], ec="none", r=0.02, z=2)
    ax.text(0.13, 1.88, "training set $T_k$  |  held-out $H_k$", fontsize=6.2, color=MUTED, va="center")
    for k in range(3):
        S.box(ax, 0.16 + k * 0.05, 1.08 - k * 0.05, 0.42, 0.42, fc="#eef2f7", ec="#b9c7d8", lw=0.6, r=0.03, z=2 + k)
    ax.text(0.72, 1.33, "$R = 10$ stratified\nsubsamples", fontsize=6.4, va="center", linespacing=1.2)
    ax.text(0.72, 1.0, "$S_{k1},\\dots,S_{kR}$\n(80% of $T_k$)", fontsize=6.4, va="center", linespacing=1.2)
    ax.text(0.13, 0.75, "same subsamples in both arms", fontsize=6.0, color=MUTED, va="center", fontstyle="italic")
    # arms
    ax0, aw = 1.66, 2.38
    for yy, ttl, line1, line2, col, fc in ((1.68, "Re-selected arm", "select the panel again on every subsample",
                                            "$P_{kr}=\\mathcal{A}(S_{kr})$, train $f_{kr}$ on $S_{kr}$", S.ARM_RE,
                                            "#e8f1fc"),
                                           (0.62, "Panel-fixed arm (control)", "keep the panel chosen on $T_k$",
                                            "$P_{k}=\\mathcal{A}(T_{k})$, train $g_{kr}$ on $S_{kr}$", S.ARM_FIX,
                                            "#f1f1f1")):
        S.box(ax, ax0, yy, aw, 0.92, fc="white", ec=col, lw=0.9, r=0.05)
        S.box(ax, ax0, yy + 0.92 - 0.26, aw, 0.26, fc=fc, ec="none", r=0.05, z=1.5)
        ax.text(ax0 + 0.1, yy + 0.92 - 0.13, ttl, fontsize=7.4, fontweight="bold", color=col, va="center", zorder=3)
        ax.text(ax0 + 0.1, yy + 0.44, line1, fontsize=6.4, va="center", color=INK)
        ax.text(ax0 + 0.1, yy + 0.2, line2, fontsize=6.6, va="center", color=INK)
        ax.text(ax0 + aw - 0.1, yy + 0.2, "× 10", fontsize=6.6, va="center", ha="right", color=MUTED)
    S.arrow(ax, 1.40, 1.62, ax0 - 0.01, 2.1); S.arrow(ax, 1.40, 1.52, ax0 - 0.01, 1.08)
    # held-out patients
    hx = 4.32
    card(ax, hx, 0.62, 0.9, 1.98, "Held-out $H_k$", dim="T")
    for k in range(5):
        ax.plot(hx + 0.17 + k * 0.14, 2.05, "o", ms=3.4, color=S.DIM["T"], zorder=3)
        ax.plot([hx + 0.17 + k * 0.14] * 2, [1.82, 1.98], color=S.DIM["T"], lw=0.9, zorder=3)
    ax.text(hx + 0.45, 1.55, "untouched", fontsize=6.3, ha="center", color=S.DIM["T"], va="center")
    ax.text(hx + 0.45, 1.15, "10 + 10\npredictions\nper patient", fontsize=6.4, ha="center", va="center",
            linespacing=1.2)
    S.arrow(ax, ax0 + aw + 0.01, 2.14, hx - 0.01, 1.9); S.arrow(ax, ax0 + aw + 0.01, 1.08, hx - 0.01, 1.3)
    # outputs
    ox = 5.42
    outs = [("$\\psi = 1 - V_{\\mathrm{fixed}}\\,/\\,V_{\\mathrm{re\\!-\\!selected}}$",
             "$V=\\sum_i \\mathrm{Var}_r[\\mathrm{logit}\\ \\hat p_{ir}]$"),
            ("Ambiguity", "patients given both labels"), ("Selection bagging", "mean of the 10 re-selected\nmodels")]
    for k, (t1, t2) in enumerate(outs):
        yy = 2.0 - k * 0.66
        S.box(ax, ox, yy, W - ox - 0.02, 0.56, fc="#f6f7f9", ec=RULE, lw=0.7, r=0.04)
        ax.text(ox + 0.07, yy + 0.38, t1, fontsize=6.6 if k else 6.4, fontweight="normal" if k == 0 else "bold",
                va="center")
        ax.text(ox + 0.07, yy + 0.16 if "\n" not in t2 else yy + 0.12, t2, fontsize=6.0, color=MUTED, va="center",
                linespacing=1.1)
        S.arrow(ax, hx + 0.9 + 0.01, 1.6, ox - 0.01, yy + 0.28, lw=0.7, ms=5)
    ax.text(0.02, 0.3, "ψ near 0: re-selection adds little beyond the variation caused by new training samples;"
            " near 1: the changing genes drive the predictions.", fontsize=6.4, color=MUTED, va="center")
    # B: patients
    g = Sc[Sc.method.isin(["six_stage", "fixed"])]
    st = g.groupby(["sample", "method"]).lab.agg(["mean", "nunique"]).unstack()
    y_ = g.drop_duplicates("sample").set_index("sample").y
    cand = pd.DataFrame({"bal": (st[("mean", "six_stage")] - 0.5).abs(),
                         "ok": (st[("nunique", "six_stage")] == 2) & (st[("nunique", "fixed")] == 1)})
    cand = cand[cand.ok].join(y_.rename("y")).rename_axis("sample").reset_index().sort_values(["bal", "sample"])
    picks = {1: list(cand[cand.y == 1].head(3)["sample"]), 0: list(cand[cand.y == 0].head(3)["sample"])}
    fig.text(0.02 / W, 0.405, "B   Held-out predictions of six patients", fontsize=8.5, fontweight="bold", va="top")
    fig.text(1 - 0.02 / W, 0.405, f"{len(cand)} of {g['sample'].nunique()} held-out patients split by re-selection "
             "and not by the fixed panel; 3 per class shown", fontsize=6.4, color=MUTED, va="top", ha="right")
    for j, (cls, nm, who) in enumerate(((1, "NSCLC patients", "Patient"), (0, "Healthy donors", "Donor"))):
        b = fig.add_axes([0.085 + j * 0.48, 0.115, 0.40, 0.235])
        for i, smp in enumerate(picks[cls]):
            for k, (m, col) in enumerate((("six_stage", S.ARM_RE), ("fixed", S.ARM_FIX))):
                q = g[(g["sample"] == smp) & (g.method == m)]
                yy = -(i * 1.0 + k * 0.34)
                b.scatter(q.score[q.lab == 1], [yy] * int((q.lab == 1).sum()), s=15, color=col, lw=0, zorder=3)
                b.scatter(q.score[q.lab == 0], [yy] * int((q.lab == 0).sum()), s=15, facecolors="white",
                          edgecolors=col, linewidths=0.9, zorder=3)
            b.axhspan(-i - 0.6, -i + 0.26, color="#f4f6f8" if i % 2 == 0 else "white", lw=0, zorder=0)
        b.set_yticks([-(i * 1.0 + 0.17) for i in range(3)]); b.set_yticklabels([f"{who} {i + 1}" for i in range(3)])
        b.tick_params(axis="y", length=0); b.spines["left"].set_visible(False)
        b.set_xlim(0, 1); b.set_ylim(-2.6, 0.26)
        b.set_xlabel("Predicted probability of NSCLC")
        b.set_title(nm, fontsize=7.4, fontweight="bold", loc="left", pad=3)
        S.xgrid(b)
    hs = [Line2D([], [], marker="o", color=S.ARM_RE, ls="", ms=4, label="re-selected arm (upper row)"),
          Line2D([], [], marker="o", color=S.ARM_FIX, ls="", ms=4, label="panel-fixed arm (lower row)"),
          Line2D([], [], marker="o", color=MUTED, ls="", ms=4, label="filled: classified NSCLC"),
          Line2D([], [], marker="o", color=MUTED, mfc="white", ls="", ms=4, label="open: classified healthy")]
    fig.legend(handles=hs, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=6.6,
               handletextpad=0.25, columnspacing=1.4)
    return S.save(fig, out, "Fig02_psidesign")


# --------------------------------------------------------------------------------------------- Figure S8
def fig_funnel(out):
    W, H = S.FULL, 3.35
    fig, ax = canvas(W, H)
    def stage(x, y, w, h, num, title, big, sub, dim="P"):
        S.box(ax, x, y, w, h, fc="white", ec=S.DIM[dim], lw=0.8, r=0.05)
        ax.add_patch(plt.Circle((x + 0.17, y + h - 0.17), 0.1, color=S.DIM[dim], zorder=3))
        ax.text(x + 0.17, y + h - 0.17, str(num), ha="center", va="center", fontsize=6.6, color="white",
                fontweight="bold", zorder=4)
        ax.text(x + 0.33, y + h - 0.17, title, ha="left", va="center", fontsize=7.2, fontweight="bold",
                color=S.DIM[dim])
        if big:
            ax.text(x + 0.12, y + h - 0.47, big, ha="left", va="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(x + 0.12, y + 0.17, sub, ha="left", va="center", fontsize=6.1, color=MUTED, linespacing=1.2)
    top = H - 0.05
    S.box(ax, 0.02, top - 0.95, 1.05, 0.95, fc="#f6f7f9", ec=RULE, lw=0.8, r=0.05)
    ax.text(0.12, top - 0.17, "Input", fontsize=7.2, fontweight="bold", va="center")
    ax.text(0.12, top - 0.47, "28,089", fontsize=12, fontweight="bold", va="center")
    ax.text(0.12, top - 0.78, "genes; GSE89843,\nn = 633", fontsize=6.1, color=MUTED, va="center", linespacing=1.2)
    w = 1.68
    xs = [1.2, 2.98, 4.76]
    st = [(1, "Mann\u2013Whitney filter", "3,557", "tier 2: q < 0.05, |log$_2$FC| > 0.5"),
          (2, "Sparse consensus", "119", "LASSO (120) \u2229 elastic net (128)"),
          (3, "Coding filter", "94", "non-coding and isoform symbols removed")]
    prev = 1.07
    for x, (n, t, b, sub) in zip(xs, st):
        S.arrow(ax, prev + 0.01, top - 0.47, x - 0.01, top - 0.47)
        stage(x, top - 0.95, w, 0.95, n, t, b, sub)
        prev = x + w
    # parallel evidence streams on the 94-gene pool
    y2 = top - 2.05
    ax.plot([xs[2] + w / 2, xs[2] + w / 2], [top - 0.95, top - 1.05], color=MUTED, lw=0.8)
    ax.plot([1.6, xs[2] + w / 2], [top - 1.05, top - 1.05], color=MUTED, lw=0.8)
    for xx in (1.6, 4.5):
        S.arrow(ax, xx, top - 1.05, xx, y2 + 0.82, color=MUTED, lw=0.8)
    S.box(ax, 0.02, y2, 3.05, 0.8, fc="white", ec=S.DIM["M"], lw=0.8, r=0.05)
    ax.add_patch(plt.Circle((0.19, y2 + 0.63), 0.1, color=S.DIM["M"], zorder=3))
    ax.text(0.19, y2 + 0.63, "4", ha="center", va="center", fontsize=6.6, color="white", fontweight="bold", zorder=4)
    ax.text(0.35, y2 + 0.63, "mRMR ranking; PPI evidence", fontsize=7.2, fontweight="bold", color=S.DIM["M"],
            va="center")
    ax.text(0.14, y2 + 0.36, "rank all 94 genes \u2192 top 30 \u2192 STRING weighted degree (score \u2265 0.4)",
            fontsize=6.3, va="center")
    ax.text(0.14, y2 + 0.15, "supplies the PPI term only; not a gene filter", fontsize=6.1, color=MUTED, va="center")
    S.box(ax, 3.27, y2, W - 3.29, 0.8, fc="white", ec=S.DIM["M"], lw=0.8, r=0.05)
    ax.add_patch(plt.Circle((3.44, y2 + 0.63), 0.1, color=S.DIM["M"], zorder=3))
    ax.text(3.44, y2 + 0.63, "5", ha="center", va="center", fontsize=6.6, color="white", fontweight="bold", zorder=4)
    ax.text(3.6, y2 + 0.63, "Bootstrap stability", fontsize=7.2, fontweight="bold", color=S.DIM["M"], va="center")
    ax.text(3.39, y2 + 0.36, "500 LASSO refits on the 94 genes; keep $f_g \\geq 0.60$", fontsize=6.3, va="center")
    ax.text(3.39, y2 + 0.15, "73 stable genes", fontsize=6.6, fontweight="bold", color=INK, va="center")
    # composite
    y3 = 0.05
    for xx in (1.6, 4.5):
        S.arrow(ax, xx, y2, xx, y3 + 0.86, color=MUTED, lw=0.8)
    S.box(ax, 0.02, y3, W - 0.04, 0.84, fc="white", ec=S.DIM["T"], lw=0.8, r=0.05)
    ax.add_patch(plt.Circle((0.19, y3 + 0.67), 0.1, color=S.DIM["T"], zorder=3))
    ax.text(0.19, y3 + 0.67, "6", ha="center", va="center", fontsize=6.6, color="white", fontweight="bold", zorder=4)
    ax.text(0.35, y3 + 0.67, "Composite score of the 73 stable genes; artefact QC", fontsize=7.2, fontweight="bold",
            color=S.DIM["T"], va="center")
    wts = [("statistical", 0.20), ("effect", 0.15), ("LASSO", 0.15), ("bootstrap", 0.30), ("PPI", 0.10), ("FI", 0.10)]
    x = 0.14; tw = 3.5
    for k, (nm, v) in enumerate(wts):
        S.box(ax, x, y3 + 0.2, tw * v - 0.02, 0.28, fc=S.DIM_M["T"] if k % 2 == 0 else S.DIM_L["T"], ec="none", r=0.02, z=2)
        ax.text(x + (tw * v - 0.02) / 2, y3 + 0.34, f"{nm} {int(round(v * 100))}%", ha="center", va="center",
                fontsize=5.8, color=INK, zorder=3)
        x += tw * v
    S.arrow(ax, x + 0.05, y3 + 0.34, x + 0.28, y3 + 0.34)
    S.box(ax, x + 0.3, y3 + 0.17, 0.62, 0.34, fc="#f6f7f9", ec=RULE, lw=0.6, r=0.04)
    ax.text(x + 0.61, y3 + 0.34, "top 30", ha="center", va="center", fontsize=6.6)
    S.arrow(ax, x + 0.94, y3 + 0.34, x + 1.13, y3 + 0.34)
    S.box(ax, x + 1.15, y3 + 0.12, W - x - 1.2, 0.44, fc=S.DIM_L["T"], ec=S.DIM["T"], lw=0.8, r=0.04)
    ax.text(x + 1.15 + (W - x - 1.2) / 2, y3 + 0.4, "27-gene panel", ha="center", va="center", fontsize=8,
            fontweight="bold", color=S.DIM["T"])
    ax.text(x + 1.15 + (W - x - 1.2) / 2, y3 + 0.22, "3 isoform entries removed", ha="center", va="center",
            fontsize=5.8, color=MUTED)
    ax.text(W - 0.04, y3 + 0.67, "FI: feature-importance consensus (stage 0)", ha="right", va="center", fontsize=6.0,
            color=MUTED)
    return S.save(fig, out, "FigS08_funnel")


# --------------------------------------------------------------------------------------------- Figure S14
def fig_benchmark(out):
    W, H = S.FULL, 3.1
    fig, ax = canvas(W, H)
    ax.text(0.02, H - 0.08, "Inputs", fontsize=7.6, fontweight="bold", va="top")
    ax.text(1.72, H - 0.08, "Models (stratified five-fold CV, same folds)", fontsize=7.6, fontweight="bold", va="top")
    ax.text(5.06, H - 0.08, "Evaluation", fontsize=7.6, fontweight="bold", va="top")
    S.box(ax, 0.02, 1.95, 1.4, 0.62, fc=S.DIM_L["P"], ec=S.DIM["P"], lw=0.8, r=0.05)
    ax.text(0.12, 2.38, "27-gene panel", fontsize=7.2, fontweight="bold", color=S.DIM["P"], va="center")
    ax.text(0.12, 2.12, "633 samples", fontsize=6.3, color=MUTED, va="center")
    S.box(ax, 0.02, 0.7, 1.4, 0.62, fc=S.DIM_L["M"], ec=S.DIM["M"], lw=0.8, r=0.05)
    ax.text(0.12, 1.13, "Transcriptome", fontsize=7.2, fontweight="bold", color=S.DIM["M"], va="center")
    ax.text(0.12, 0.87, "28,089 genes", fontsize=6.3, color=MUTED, va="center")
    models = [("XGBoost", "baseline and Optuna-tuned (50 trials)", "P"),
              ("BioGAT", "graph attention on an expanded PPI graph; panel-node pooling", "P"),
              ("VAE + XGBoost", "128-dimensional latent space; log1p-normalised input", "M"),
              ("scVI-inspired VAE + XGBoost", "negative-binomial likelihood; 32-dimensional latent space", "M"),
              ("GeneTransformer", "embeddings of the top 200 ranked gene tokens", "M"),
              ("Ensemble", "soft vote: XGBoost (weight 0.6) + GeneTransformer", "T")]
    mx, mw, mh = 1.72, 3.05, 0.34
    ys = [H - 0.55 - k * 0.42 for k in range(len(models))]
    for (nm, sub, dim), yy in zip(models, ys):
        S.box(ax, mx, yy - mh / 2, mw, mh, fc="white", ec=S.DIM[dim], lw=0.8, r=0.04)
        ax.text(mx + 0.1, yy + 0.055, nm, fontsize=6.9, fontweight="bold", color=S.DIM[dim], va="center")
        ax.text(mx + 0.1, yy - 0.085, sub, fontsize=5.9, color=MUTED, va="center")
    for yy in ys[:2]:
        S.arrow(ax, 1.42, 2.26, mx - 0.01, yy, lw=0.7, ms=5)
    for yy in ys[2:5]:
        S.arrow(ax, 1.42, 1.01, mx - 0.01, yy, lw=0.7, ms=5)
    S.arrow(ax, mx + mw + 0.03, (ys[0] + ys[-1]) / 2, 5.04, (ys[0] + ys[-1]) / 2, lw=0.8)
    S.box(ax, 5.06, 0.45, W - 5.08, 2.0, fc="#f6f7f9", ec=RULE, lw=0.8, r=0.05)
    ev = [("Primary endpoint", "ROC AUC"), ("Interval", "1,000-repeat bootstrap"), ("Folds", "stratified five-fold"),
          ("Explanation", "TreeSHAP (XGBoost only)")]
    for k, (a_, b_) in enumerate(ev):
        yy = 2.2 - k * 0.45
        ax.text(5.16, yy, a_, fontsize=6.0, color=MUTED, va="center")
        ax.text(5.16, yy - 0.17, b_, fontsize=6.9, color=INK, va="center", fontweight="bold" if k == 0 else "normal")
    ax.text(0.02, 0.12, "Development analysis with an earlier, unseeded version of the code; it informed the choice of "
            "classifier only.", fontsize=6.1, color=MUTED, va="center")
    return S.save(fig, out, "FigS14_benchmark")


ALL = [fig_framework, fig_psidesign, fig_funnel, fig_benchmark]
