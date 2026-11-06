#!/usr/bin/env python3
"""Final version of the random-panel figure (R9).
    python redraw_r9.py <out_dir>
Reads <out>/tables/R9_random_panels.csv and the curated panel's AUC (macro RandCurated) and writes
<out>/figures/R9_random_panels.{pdf,png}."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

out = sys.argv[1]
r = pd.read_csv(os.path.join(out, "tables", "R9_random_panels.csv"))
cur = float(json.load(open(os.path.join(out, "macros", "r9.json")))["RandCurated"])
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.3))
for ax, kind, ttl in [(axs[0], "unmatched", "A  Random 27-gene panels"),
                      (axs[1], "matched", "B  Expression/variance-matched panels")]:
    a = r.loc[r.kind == kind, "auc_foldmean"].astype(float)
    ax.hist(a, bins=60, color="#B8B8B8", lw=0)
    ax.axvline(cur, c="#C0392B", lw=2)
    top = ax.get_ylim()[1]
    ax.text(cur - 0.004, top * 0.93, f"27-gene panel\n{cur:.3f}", color="#C0392B", fontsize=8, ha="right", va="top")
    n_ge = int((a >= cur).sum())
    ax.text(0.02, 0.95, f"n = {len(a):,}\nmean {a.mean():.3f}\nmax {a.max():.3f}\n{n_ge} at or above\nthe panel",
            transform=ax.transAxes, fontsize=8, va="top")
    ax.set_xlim(min(0.6, a.min() - 0.01), max(1.0, cur + 0.02))
    ax.set_title(ttl, fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel("Five-fold CV AUC (mean of folds)")
    ax.set_ylabel("Number of panels")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(os.path.join(out, "figures", f"R9_random_panels.{e}"), dpi=200, bbox_inches="tight")
print("R9 figure written; curated", cur)
