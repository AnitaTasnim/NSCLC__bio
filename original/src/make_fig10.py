#!/usr/bin/env python3
"""Figure 10 (nested CV): A) held-out AUC per outer fold with reference lines;
B) Jaccard of each fold panel with the published panel, against weight perturbation and chance.
    python make_fig10.py <out_dir> [external_auc]"""
import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

out = sys.argv[1]
ext = float(sys.argv[2]) if len(sys.argv) > 2 else None
cache = pickle.load(open(os.path.join(out, "cache", "run_cache.pkl"), "rb"))
res = cache["nested_six_stage"]
import pandas as pd
meta = pd.read_csv(os.path.join(out, "tables", "R4_nested_oof.csv"))
y = meta.y.values
fixed = roc_auc_score(y, cache["oof_P27"])
nested = roc_auc_score(y, res["oof"])
P = pd.read_csv(os.path.join(out, "tables", "R1_panel_site_vs_disease.csv")).gene.tolist()
jac = [len(set(P) & set(p)) / len(set(P) | set(p)) for p in res["panels"]]
rec = pd.read_csv(os.path.join(out, "tables", "R4_gene_recurrence.csv")) if os.path.exists(
    os.path.join(out, "tables", "R4_gene_recurrence.csv")) else None
import json
def _macro(name, default):
    for fn in os.listdir(os.path.join(out, "macros")):
        try:
            m = json.load(open(os.path.join(out, "macros", fn)))
        except Exception:
            continue
        if name in m:
            return m[name]
    return default
exp_j = float(_macro("StabExpJ", "0.0037"))
# composite-weight perturbation (+/-20%, 2,000 draws) from extra_analyses.py weights
w_mean, w_lo, w_hi = (float(_macro(k, d)) for k, d in [("WeightJMean", "nan"), ("WeightJMin", "nan"), ("WeightJMax", "nan")])
f = np.arange(1, len(jac) + 1)
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4))
ax = axs[0]
ax.plot(f, res["fold_aucs"], "o", c="#1F618D", ms=7)
ax.axhline(fixed, c="#555", lw=1.2); ax.text(5.6, fixed, f"fixed panel {fixed:.3f}", fontsize=8, va="center", color="#333")
ax.axhline(nested, c="#555", ls="--", lw=1.2); ax.text(5.6, nested - 0.006, f"nested, pooled {nested:.3f}", fontsize=8, va="center", color="#333")
lo = 0.70
if ext is not None:
    ax.axhline(ext, c="#777", ls=":", lw=1.2); ax.text(5.6, ext, f"GSE68086 {ext:.3f}", fontsize=8, va="center", color="#333")
ax.set_ylim(lo, 1.0); ax.set_xlim(0.5, 5.4); ax.set_xticks(f)
ax.set_xlabel("Nested outer fold"); ax.set_ylabel("AUC on held-out fold")
ax.set_title("A  Performance is stable", fontsize=11, loc="left", fontweight="bold")
ax = axs[1]
ax.axhspan(w_lo, w_hi, color="#C0392B", alpha=0.12, lw=0)
ax.axhline(w_mean, c="#C0392B", lw=1.2)
ax.text(5.6, w_mean - 0.04, f"weight perturbation\nmean {w_mean:.3f} ({w_lo:.2f}–{w_hi:.2f})", fontsize=8, va="center", color="#C0392B")
ax.plot(f, jac, "o", c="#7D3C98", ms=7)
ax.axhline(np.mean(jac), c="#7D3C98", ls="--", lw=1.2)
ax.text(5.6, np.mean(jac) + 0.02, f"patient resampling\nmean {np.mean(jac):.3f}", fontsize=8, va="center", color="#7D3C98")
ax.axhline(exp_j, c="k", ls=":", lw=1)
ax.text(5.6, 0.06, f"random 27-gene\nsets {exp_j:.3f}", fontsize=8, va="center")
ax.set_ylim(0, 1.0); ax.set_xlim(0.5, 5.4); ax.set_xticks(f)
ax.set_xlabel("Nested outer fold"); ax.set_ylabel("Jaccard with 27-gene panel")
ax.set_title("B  Gene identity is not", fontsize=11, loc="left", fontweight="bold")
for a in axs:
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
fig.tight_layout()
for ext_ in ("pdf", "png"):
    fig.savefig(os.path.join(out, "figures", f"fig10_nested_simplified.{ext_}"), dpi=200, bbox_inches="tight")
print("fig10 written", np.round(res["fold_aucs"], 4), np.round(jac, 3), round(fixed, 4), round(nested, 4))
