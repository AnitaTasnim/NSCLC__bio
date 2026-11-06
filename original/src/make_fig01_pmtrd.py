#!/usr/bin/env python3
"""Figure 1: the PMTRD framework (schematic)."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = sys.argv[1] if len(sys.argv) > 1 else "fig01_pmtrd"
fig, ax = plt.subplots(figsize=(11, 6.2)); ax.set_xlim(0, 110); ax.set_ylim(0, 61); ax.axis("off")


def box(x, y, w, h, title, lines, fc, ec, tfs=9.5, lfs=7.6, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    ax.add_patch(FancyBboxPatch((x, y + h - 5.2), w, 5.2, boxstyle="round,pad=0.4,rounding_size=1.2", fc=ec, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center", fontsize=tfs, color=tc, weight="bold")
    for i, l in enumerate(lines):
        ax.text(x + 1.5, y + h - 8.3 - i * 3.15, l, ha="left", va="center", fontsize=lfs, color="#1b1b1b")


def arrow(x0, y0, x1, y1, c="#555555"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2))


# left column: data and selection procedure A
box(1, 37, 27, 22, "Discovery cohort D", ["GSE89843 platelet RNA-seq", "n = 633 (402 NSCLC / 231 healthy)",
    "16 site codes; 15 single-class", "only Vumc contains both classes", "28,089 genes"], "#f3f6f9", "#41607a")
box(1, 1, 27, 33, "Selection procedure $\\mathcal{A}$", ["fold-recomputable six stages:", "0  feature-importance screen",
    "1  Mann-Whitney Tier 2", "2  LASSO and elastic-net consensus", "3  protein-coding filter",
    "4  MRMR; STRING PPI term", "5  bootstrap stability (B = 500)", "6  composite score $\\rightarrow$ 27 genes"], "#f3f6f9", "#41607a")
arrow(14.5, 37, 14.5, 34.6)
# three dimensions
cols = [("Predictive reliability $\\mathcal{R}_P$", "#fbf1ec", "#b5532a",
         ["nested AUC, $\\mathcal{A}$ re-run in every fold", "selection optimism (fixed - nested)", "comparator on identical folds",
          "  + implementation sensitivity", "locked model in GSE68086 (n = 94)", "calibration intercept, slope, Brier",
          "sensitivity, specificity at fixed threshold"]),
        ("Molecular reliability $\\mathcal{R}_M$", "#eef4f8", "#2a6f97",
         ["identity stability: Jaccard vs chance,", "  Nogueira $\\hat\\Phi$", "random and matched null panels",
          "gene-level replication $\\rho_g$", "pathway-level replication $\\rho_S$", "  (25 split halves of one site,",
          "  with contamination adjustment)"]),
        ("Technical reliability $\\mathcal{R}_T$", "#f1f5ee", "#4d7a3a",
         ["site x label test; within-site nested AUC", "site-free transfer to 15 site codes",
          "gene-level site vs disease effect", "contamination: QC panels (TOST),", "  adjusted model",
          "harmonisation leakage audit", "quantification coverage; gene availability"])]
for i, (t, fc, ec, lines) in enumerate(cols):
    x = 33 + i * 25.5
    box(x, 27, 24, 32, t, lines, fc, ec, tfs=9.0, lfs=7.4)
ax.annotate("", xy=(32.4, 40), xytext=(28.6, 17), arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.2))
# bottom band: output
ax.add_patch(FancyBboxPatch((33, 12), 75, 9.2, boxstyle="round,pad=0.4,rounding_size=1.2", fc="#2b2b2b", ec="#2b2b2b"))
ax.text(70.5, 18.1, "PMTRD reliability card: every quantity with an estimator and a 95% CI", ha="center",
        va="center", fontsize=9, color="white", weight="bold")
ax.text(70.5, 14.5, "dimensions are reported side by side, not combined into one score", ha="center", va="center",
        fontsize=8, color="#dddddd")
for i in range(3):
    arrow(45 + i * 25.5, 26.8, 45 + i * 25.5, 21.9, c="#888888")
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(f"{out}.{e}", dpi=300, bbox_inches="tight")
print("written", out)
