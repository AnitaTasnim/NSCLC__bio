"""Redraw the within-site alignment experiment figure from its table.  python redraw_r3sim.py <out_dir>"""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
out = sys.argv[1]
df = pd.read_csv(os.path.join(out, "tables", "R3sim_alignment_within_vumc.csv"))
lab = {"auc_none": "no alignment", "auc_percohort_z": "each cohort standardised on itself",
       "auc_joint_combat_retrained": "joint ComBat-seq + re-trained model",
       "auc_ref_combat_frozen": "reference-batch ComBat-seq, frozen model"}
sty = {"auc_none": ("#555555", "s"), "auc_percohort_z": ("#C0392B", "o"),
       "auc_joint_combat_retrained": ("#7D3C98", "^"), "auc_ref_combat_frozen": ("#1F618D", "D")}
cols = [c for c in lab if c in df]
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.4))
off = dict(zip(cols, np.linspace(-0.03, 0.03, len(cols))))
for c in cols:
    g = df.groupby("prevalence")[c]
    mu, lo, hi = g.mean(), g.quantile(.1), g.quantile(.9)
    x = mu.index.values + off[c]
    axs[0].errorbar(x, mu.values, yerr=[mu.values - lo.values, hi.values - mu.values], fmt=sty[c][1], c=sty[c][0],
                    ms=4, lw=1, capsize=2, label=lab[c])
axs[0].set_ylim(0.80, 1.0); axs[0].set_xticks([0.2, 0.4, 0.6, 0.8])
axs[0].set_xlabel("NSCLC prevalence of the 'external' part"); axs[0].set_ylabel("AUC (mean, 10th-90th pct.)")
axs[0].set_title("A  AUC by case mix, no true batch effect", fontsize=9, loc="left")
axs[0].legend(frameon=False, fontsize=6.5, loc="lower left")
for c in [c for c in cols if c != "auc_none"]:
    dd = 100 * (df[c] - df.auc_none)
    g = dd.groupby(df.prevalence)
    mu, lo, hi = g.mean(), g.quantile(.1), g.quantile(.9)
    x = mu.index.values + off[c]
    axs[1].errorbar(x, mu.values, yerr=[mu.values - lo.values, hi.values - mu.values], fmt=sty[c][1], c=sty[c][0],
                    ms=4, lw=1, capsize=2, label=lab[c])
axs[1].axhline(0, c="k", lw=.7, ls=":"); axs[1].set_ylim(-8, 8); axs[1].set_xticks([0.2, 0.4, 0.6, 0.8])
axs[1].set_xlabel("NSCLC prevalence of the 'external' part"); axs[1].set_ylabel("AUC change vs no alignment (pp)")
axs[1].set_title("B  Change caused by the alignment step", fontsize=9, loc="left")
for a in axs:
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(os.path.join(out, "figures", f"R3sim_alignment_within_vumc.{e}"), dpi=300, bbox_inches="tight")
print("ok")
