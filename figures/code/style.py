"""One visual style for every figure of the paper and the supplement.

Sizes are the printed sizes on an A4 page with 2.2 cm margins (text width 16.6 cm = 6.54 in), so the
figures are included at 100% and the type is 7-8.5 pt in print. Fonts are embedded (TrueType).
Colours have fixed meanings across all figures:
  dimensions  predictive / molecular / technical
  selectors   one colour per selector, the same in every figure (with a marker shape as a second cue)
  classes     NSCLC / healthy
"""
from __future__ import annotations

import os

import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

FULL, TWOTHIRDS, HALF = 6.5, 4.4, 3.2          # inches

INK, MUTED, FAINT = "#1d2733", "#5b6675", "#8c96a3"
GRID, RULE, PANEL = "#e6e9ee", "#b9c1cb", "#f6f7f9"

DIM = {"P": "#2b5c8a", "M": "#1f7a6d", "T": "#6e4f96"}             # predictive, molecular, technical
DIM_L = {"P": "#eaf0f7", "M": "#e7f3f0", "T": "#f1ecf7"}           # pale fills
DIM_M = {"P": "#c9d8ea", "M": "#c4e2db", "T": "#dcd1ea"}           # mid fills
DIM_NAME = {"P": "Predictive", "M": "Molecular", "T": "Technical"}

SEL = {"six_stage": "#2a78d6", "mw_topk": "#eb6834", "l1_topk": "#14967a", "stability": "#9b59b6",
       "mrmr_ifs_svm": "#7a5c3a", "random": "#8c96a3", "fixed": "#2b2b2b", "pathway": "#c49a00"}
SEL_MARK = {"six_stage": "o", "mw_topk": "s", "l1_topk": "^", "stability": "D", "mrmr_ifs_svm": "v",
            "random": "X", "fixed": "P", "pathway": "*"}
SEL_NAME = {"six_stage": "Six-stage (paper)", "mw_topk": "Mann–Whitney top 27", "l1_topk": "L1-logistic top 27",
            "stability": "Stability selection", "mrmr_ifs_svm": "mRMR–IFS–SVM", "random": "Random panels",
            "fixed": "Panel fixed", "pathway": "Pathway model"}
SEL_SHORT = {"six_stage": "Six-stage", "mw_topk": "Mann–Whitney", "l1_topk": "L1-logistic",
             "stability": "Stability sel.", "mrmr_ifs_svm": "mRMR–IFS–SVM", "random": "Random",
             "fixed": "Panel fixed", "pathway": "Pathway model"}

NSCLC, HEALTHY = "#b8432f", "#3f6e9a"
ARM_RE, ARM_FIX = SEL["six_stage"], "#3b3b3b"


def setup():
    plt.rcParams.update({
        "font.family": "FreeSans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "axes.edgecolor": RULE, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
        "mathtext.fontset": "stixsans", "axes.titlelocation": "left",
        "axes.titleweight": "bold", "axes.titlepad": 6, "figure.dpi": 100,
    })


def panel(ax, letter, title=None, x=-0.01, y=1.0, size=8.5):
    """Panel letter plus a short title, left-aligned above the axes."""
    ax.set_title("")
    txt = f"{letter}" if not title else f"{letter}   {title}"
    ax.text(x, y, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=size, fontweight="bold",
            color=INK)


def ygrid(ax):
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)


def xgrid(ax):
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)


def box(ax, x, y, w, h, fc="white", ec=RULE, lw=0.8, r=0.06, z=1, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z,
                       linestyle=ls)
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=0.9, style="-|>", z=3, ms=7):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=ms),
                zorder=z)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300 if ext == "png" else None, bbox_inches="tight",
                    pad_inches=0.03)
    plt.close(fig)
    return os.path.join(out, f"{name}.pdf")
