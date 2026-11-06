#!/usr/bin/env python3
"""Forest plot and table of the main AUCs with 95% CIs (Figure 'forest', Table S1).
Reads the macro files written by the analysis steps.   python make_forest.py <out_dir>"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
M = {}
for f in sorted(os.listdir(os.path.join(out, "macros"))):
    if f.endswith(".json"):
        M.update(json.load(open(os.path.join(out, "macros", f))))


def num(k):
    return float(M[k])


def rng(k):
    lo, hi = M[k].split("--")
    return float(lo), float(hi)


rows = [
    ("Discovery cohort (GSE89843, n = 633)", None),
    ("27-gene panel, fixed-panel 5-fold CV", ("CoreFixedAUC", "CoreFixedCI", "DeLong")),
    ("27-gene panel, fixed-panel CV, log2-CPM input", ("CoreCpmAUC", "CoreCpmCI", "DeLong")),
    ("Six-stage selection nested in CV (B = 500)", ("NestedAUC", "NestedCI", "DeLong")),
    ("mRMR-IFS-SVM (Sheng et al.), nested, same folds", ("HhShengAUC", "HhShengCI", "DeLong")),
    ("Site-aware audit (GSE89843)", None),
    ("Site-stratified 5-fold CV", ("CoreSiteStratAUC", "CoreSiteStratCI", "DeLong")),
    ("Vumc only, 5-fold CV (site constant)", ("CoreVumcAUC", "CoreVumcCI", "DeLong")),
    ("Vumc only, selection nested in CV", ("VumcNestedAUC", "VumcNestedCI", "DeLong")),
    ("Held-out site: train other sites, test Vumc", ("CoreHeldOutAUC", "CoreHeldOutCI", "DeLong")),
    ("Vumc-derived panel, Vumc-trained -> 15 other site codes", ("VumcTransferAUC", "VumcTransferCI", "DeLong")),
    ("Independent platelet cohort (GSE68086, n = 94)", None),
    ("Development configuration (22 genes, un-normalised, per-cohort scaling)", ("ExtOrigReproAUC", "ExtOrigReproCI", "bootstrap")),
    ("27-gene panel, log2-CPM, no alignment", ("ExtTwentySevenNoneAUC", "ExtTwentySevenNoneCI", "bootstrap")),
    ("27-gene panel, reference-batch ComBat-seq", ("ExtTwentySevenRefAUC", "ExtTwentySevenRefCI", "bootstrap")),
    ("27-gene panel, joint ComBat-seq + re-trained model", ("ExtTwentySevenJointAUC", "ExtTwentySevenJointCI", "bootstrap")),
    ("27-gene panel, single-sample ranks", ("ExtTwentySevenRankAUC", "ExtTwentySevenRankCI", "bootstrap")),
    ("QC-filtered 24-gene panel, no alignment", ("ExtQcTwentyFourNoneAUC", "ExtQcTwentyFourNoneCI", "bootstrap")),
    ("Vumc-derived panel (Vumc-trained), no alignment", ("ExtVumcNoneAUC", "ExtVumcNoneCI", "bootstrap")),
    ("mRMR-IFS-SVM (Sheng et al.), no alignment", ("HhExtShengAUC", "HhExtShengCI", "bootstrap")),
    ("Lung tissue", None),
    ("GSE40791, platelet-trained model", (0.4334, (0.3536, 0.5137), "bootstrap")),
    ("GSE68465, platelet-trained model", (0.5256, (0.4348, 0.6099), "bootstrap")),
]
cols = {"Discovery": "#1F618D", "Site-aware": "#7D3C98", "Independent": "#C0392B", "Lung": "#7F8C8D"}
items, table = [], []
for lab, spec in rows:
    if spec is None:
        items.append(("H", lab, None)); continue
    if isinstance(spec[0], str):
        if spec[0] not in M:
            continue
        a, (lo, hi), meth = num(spec[0]), rng(spec[1]), spec[2]
    else:
        a, (lo, hi), meth = spec
    items.append(("R", lab, (a, lo, hi)))
    table.append((lab, a, lo, hi, meth))

fig, ax = plt.subplots(figsize=(7.4, 0.30 * len(items) + 0.9))
y = 0; yt, yl = [], []; colour = "#333"
for kind, lab, v in items:
    if kind == "H":
        colour = next(c for k, c in cols.items() if lab.startswith(k))
        ax.text(0.30, y, lab, fontsize=8, fontweight="bold", va="center", color=colour)
        y += 1; continue
    a, lo, hi = v
    ax.plot([lo, hi], [y, y], c=colour, lw=1.6); ax.plot(a, y, "o", c=colour, ms=4.5)
    ax.text(1.005, y, f"{a:.3f} ({lo:.3f}-{hi:.3f})", fontsize=6.5, va="center")
    yt.append(y); yl.append(lab); y += 1
ax.axvline(.5, ls=":", c="k", lw=.8)
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.8); ax.invert_yaxis()
ax.set_xlim(.3, 1.0); ax.set_xlabel("AUC (95% CI)")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(os.path.join(out, "figures", f"forest_all_aucs.{e}"), dpi=300, bbox_inches="tight")
# LaTeX table for the supplement
lines = ["\\begin{tabular}{@{}p{9.0cm}rl@{}}", "\\toprule", "Analysis & AUC (95\\% CI) & CI method \\\\", "\\midrule"]
for kind, lab, v in items:
    if kind == "H":
        lines.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\textit{{{lab}}}}} \\\\")
    else:
        a, lo, hi = v
        meth = [t[4] for t in table if t[0] == lab][0]
        lines.append(f"{lab.replace('->', '$\\to$')} & {a:.4f} ({lo:.4f}--{hi:.4f}) & {meth} \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(out, "tables", "ALL_AUCs_with_95CI.tex"), "w").write("\\footnotesize\n" + "\n".join(lines) + "\n")
import csv
with open(os.path.join(out, "tables", "ALL_AUCs_with_95CI.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["analysis", "auc", "ci_lo", "ci_hi", "ci_method"]); w.writerows(table)
print(f"forest: {len(table)} estimates")
