"""Redraw the gene-level site-versus-disease figure with non-overlapping labels.
    python redraw_r1_effects.py <out_dir>"""
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adjustText import adjust_text
out = sys.argv[1]
gw = pd.read_csv(os.path.join(out, "tables", "R1_genomewide_site_vs_disease.csv"))
pan = pd.read_csv(os.path.join(out, "tables", "R1_panel_site_vs_disease.csv"))
t2 = gw[gw.tier2 & ~gw.gene.isin(pan.gene)]
fig, ax = plt.subplots(figsize=(5.6, 4.8))
ax.scatter(t2.eps2_site_max, t2.eps2_disease_vumc, s=5, c="#C8C8C8", alpha=.6, label="other Tier-2 genes", rasterized=True)
dis = pan.dominant == "disease"
ax.scatter(pan.eps2_site_max[dis], pan.eps2_disease_vumc[dis], s=30, c="#C0392B", edgecolor="k", lw=.4, zorder=3, label="panel gene, disease-dominant")
ax.scatter(pan.eps2_site_max[~dis], pan.eps2_disease_vumc[~dis], s=30, c="#F5B7B1", edgecolor="k", lw=.4, zorder=3, label="panel gene, site-dominant")
clus = (pan.eps2_site_max < 0.06) & (pan.eps2_disease_vumc < 0.05)
texts = [ax.text(r.eps2_site_max, r.eps2_disease_vumc, r.gene, fontsize=6.5) for r in pan[~clus].itertuples()]
if clus.any():
    cd = pan[clus & (pan.dominant == "disease")].gene.tolist(); cs = pan[clus & (pan.dominant != "disease")].gene.tolist()
    lab = "near origin:\n" + ", ".join(cd) + (" (disease-dominant)\n" if cd else "") + ", ".join(cs) + (" (site-dominant)" if cs else "")
    ax.annotate(lab, xy=(0.02, 0.02), xytext=(0.015, 0.40), fontsize=6.5, color="#333",
                arrowprops=dict(arrowstyle="-", color="#666", lw=0.4), wrap=True)
lim = 0.72
ax.plot([0, lim], [0, lim], ls="--", c="k", lw=.8)
ax.set_xlim(-0.01, lim); ax.set_ylim(-0.01, lim)
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#666", lw=0.4), expand=(1.2, 1.5))
ax.set_xlabel("Site effect, larger within-class $\\varepsilon^2$")
ax.set_ylabel("Disease effect within Vumc, $\\varepsilon^2$")
ax.text(.97, .22, "site-dominant", transform=ax.transAxes, ha="right", fontsize=8, color="#555")
ax.text(.03, .97, "disease-dominant", transform=ax.transAxes, va="top", fontsize=8, color="#555")
ax.legend(frameon=False, fontsize=7, loc="upper right")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for e in ("pdf", "png"):
    fig.savefig(os.path.join(out, "figures", f"R1_site_vs_disease_effects.{e}"), dpi=300, bbox_inches="tight")
print("redrawn")
