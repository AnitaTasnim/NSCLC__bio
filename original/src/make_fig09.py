#!/usr/bin/env python3
"""Figure 9: ROC and precision-recall curves of the fixed 27-gene panel from the seeded five-fold CV.
    python make_fig09.py <config.json>
Uses the out-of-fold predictions saved by run_steps.py (cache/run_cache.pkl, key oof_P27) and the seeded folds."""
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tep_analysis as T  # noqa: E402

cfg = T.load_config(sys.argv[1]); d = T.load_data(cfg); run = T.Run(cfg, d)
out = cfg["out_dir"]
y = d.y.values
oof = pickle.load(open(os.path.join(out, "cache", "run_cache.pkl"), "rb"))["oof_P27"]
folds = run.folds()
grid = np.linspace(0, 1, 201)
INK, GREY, ACCENT = "#333333", "#9AA5B1", "#1F618D"
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
ax = axs[0]
tprs, fa = [], []
for k, (_, te) in enumerate(folds):
    fpr, tpr, _ = roc_curve(y[te], oof[te])
    fa.append(roc_auc_score(y[te], oof[te]))
    ax.plot(fpr, tpr, color=GREY, lw=1, alpha=0.9, label="Single folds" if k == 0 else None)
    t = np.interp(grid, fpr, tpr); t[0] = 0.0; tprs.append(t)
fpr, tpr, _ = roc_curve(y, oof)
auc, lo, hi, _ = T.delong_ci(y, oof)
ax.plot(fpr, tpr, color=ACCENT, lw=2.2, label=f"Pooled out-of-fold: AUC {auc:.3f} (95% CI {lo:.3f}–{hi:.3f})")
ax.plot([0, 1], [0, 1], ls=":", color=INK, lw=0.8)
ax.text(0.97, 0.12, f"Fold AUCs {min(fa):.3f}–{max(fa):.3f}\nmean {np.mean(fa):.3f} ± {np.std(fa, ddof=1):.3f}",
        ha="right", fontsize=8, color=INK)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("A  ROC, stratified five-fold CV", fontsize=10, loc="left", fontweight="bold")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.22), fontsize=7.5, frameon=False)
ax = axs[1]
prec, rec, _ = precision_recall_curve(y, oof)
ap = average_precision_score(y, oof)
ax.plot(rec, prec, color=ACCENT, lw=2.2, label=f"Pooled out-of-fold: AUPRC {ap:.3f}")
ax.axhline(y.mean(), ls=":", color=INK, lw=0.8)
ax.text(0.02, y.mean() - 0.04, f"prevalence {y.mean():.3f}", fontsize=8, color=INK)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("B  Precision–recall", fontsize=10, loc="left", fontweight="bold")
ax.legend(loc="lower left", fontsize=7.5, frameon=False)
for a in axs:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.tick_params(labelsize=8)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(os.path.join(out, "figures", f"fig09_roc_pr_seeded.{e}"), dpi=200, bbox_inches="tight")
print("fig09 written", round(auc, 4), np.round(fa, 4), round(ap, 4))
