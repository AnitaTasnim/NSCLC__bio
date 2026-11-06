"""Add GSE68086 AUCs (log2-CPM, no alignment, frozen model) to the QC-panel table of R5.
    python r5_external_table.py config_ext.json"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tep_analysis as T
cfg = T.load_config(sys.argv[1]); cfg["out_dir"] = cfg["out_dir"]
d = T.load_data(cfg); run = T.Run(cfg, d)
out = cfg["out_dir"]
y, ye = d.y.values, d.y_ext.values
P = run.panel_in(d.X)
shared = sorted(set(d.counts.columns) & set(d.counts_ext.columns))
C = d.counts[shared].clip(lower=0).round(); Ce = d.counts_ext[shared].clip(lower=0).round()
ok = ((C.sum(0) > 0) & (Ce.sum(0) > 0)) | C.columns.isin(P)
Ld, Le = T.logcpm(C.loc[:, ok]), T.logcpm(Ce.loc[:, ok])
cmp_ = pd.read_csv(os.path.join(out, "tables", "compare_R5_contamination_panels.csv"))
scr = pd.read_csv(os.path.join(out, "tables", "R5_contamination_screen_all27.csv"))
flagged = scr.loc[scr.contamination_flags.fillna("") != "", "gene"].tolist()
sus = ["KRTAP22-1", "OR52B6", "LCE3B"]
panels = {"P27": P, "P26": [g for g in P if g != "KRTAP22-1"], "P24": [g for g in P if g not in sus]}
unfl = [g for g in P if g not in set(sus) | set(flagged)]
if len(unfl) >= 5:
    panels[f"P{len(unfl)}unflagged"] = unfl
rows, M = [], {}
for name, G in panels.items():
    p = run.xgb().fit(Ld[G].values, y).predict_proba(Le[G].values)[:, 1]
    a, lo, hi = T.bootstrap_auc_ci(ye, p, 2000, 42)
    ra = cmp_[(cmp_.panel == name) & (cmp_.scope == "all")].iloc[0]
    rv = cmp_[(cmp_.panel == name) & (cmp_.scope == "vumc")].iloc[0]
    dl = "reference" if name == "P27" else f"{T.pp(ra.delta_vs_ref)} ({T.pp(ra.delta_lo95)}, {T.pp(ra.delta_hi95)})"
    eq = "--" if name == "P27" else ("yes" if ra.equivalent else "no")
    rows.append([name.replace("unflagged", " (unflagged)"), str(int(ra.n_genes)), f"{T.f4(ra.auc_rep_mean)} $\\pm$ {T.f4(ra.auc_rep_sd)}",
                 T.tex_stack(*dl.split(" (", 1)[:1], *(["(" + dl.split(" (", 1)[1]] if " (" in dl else [])), eq, T.f4(rv.auc_rep_mean), T.tex_auc_ci(a, lo, hi)])
    if name.endswith("unflagged"):
        M["QcUnflaggedN"] = str(len(G)); M["QcUnflaggedAUC"] = T.f4(ra.auc_rep_mean); M["QcUnflaggedExtAUC"] = T.f4(a)
        M["QcUnflaggedGenes"] = ", ".join(G)
T.write_tex_table(os.path.join(out, "tables", "R5_contamination_panels.tex"), rows,
                  ["Panel", "Genes", T.tex_stack("CV AUC", "(mean $\\pm$ SD)"), T.tex_stack("$\\Delta$AUC vs 27", "pp (95\\% CI)"),
                   T.tex_stack("Equi-", "valent"), T.tex_stack("Vumc-only", "AUC"), T.tex_stack("GSE68086 AUC", "(95\\% CI)")],
                  "lcccccc", "\\footnotesize")
json.dump(M, open(os.path.join(out, "macros", "r5ext.json"), "w"), indent=1)
print(rows, M)
