#!/usr/bin/env python3
"""R7 pathway analysis, one gene-set library per process (keeps memory low).

    python r7_gsea.py config.json rank                      # write the within-Vumc ranking
    python r7_gsea.py config.json lib <gmt> [perm] [maxsize] # pre-ranked GSEA for one library
    python r7_gsea.py config.json ora                       # over-representation of the panel genes
    python r7_gsea.py config.json combine                   # merge results, write macros
    python r7_gsea.py config.json rank_adj                  # sensitivity ranking adjusted for contamination scores
    python r7_gsea.py config.json lib <gmt> 1000 500 adj    # ... GSEA on the adjusted ranking
    python r7_gsea.py config.json combine_adj

Ranking statistic: signed -log10 Mann-Whitney p-value, NSCLC versus healthy within Vumc (site constant),
on log2-CPM values of the expressed genes (CPM >= 1 in at least as many Vumc samples as the smaller class,
the filterByExpr logic of edgeR). The over-representation test of the panel uses all genes as background."""
import gc
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tep_analysis as T  # noqa: E402


def main():
    cfg = T.load_config(sys.argv[1]); out = cfg["out_dir"]; what = sys.argv[2]
    tdir = os.path.join(out, "tables"); os.makedirs(tdir, exist_ok=True)
    rnk_path = os.path.join(tdir, "R7_vumc_ranked_genes.rnk")
    if what == "rank":
        d = T.load_data(cfg); run = T.Run(cfg, d)
        y = d.y.values; v = run.vumc
        # expression filter (edgeR filterByExpr logic): CPM >= 1 in at least as many Vumc samples as the smaller group
        Xv = d.X.loc[v]; cpm = Xv.div(Xv.sum(axis=1), axis=0) * 1e6
        n_min = int(min((y[v] == 1).sum(), (y[v] == 0).sum()))
        keep = (cpm >= 1).sum(axis=0) >= n_min
        L = run.logX().loc[:, keep.values]
        print(f"expression filter: CPM >= 1 in >= {n_min} of {int(v.sum())} Vumc samples -> {int(keep.sum())} genes")
        json.dump({"n_genes_ranked": int(keep.sum()), "n_min": n_min, "n_vumc": int(v.sum())},
                  open(os.path.join(tdir, "R7_filter.json"), "w"))
        u = stats.mannwhitneyu(L.values[v & (y == 1)], L.values[v & (y == 0)], axis=0)
        auc = u.statistic / ((v & (y == 1)).sum() * (v & (y == 0)).sum())
        stat = -np.log10(np.clip(u.pvalue, 1e-300, 1)) * np.sign(auc - 0.5)
        rnk = pd.Series(stat, index=L.columns).replace([np.inf, -np.inf], np.nan).dropna()
        rnk = rnk[~rnk.index.duplicated()].sort_values(ascending=False)
        rnk.to_csv(rnk_path, sep="\t", header=False)
        # background for the over-representation test: every gene in the matrix (any gene could have been selected)
        pd.Series(list(d.X.columns)).to_csv(os.path.join(tdir, "R7_background_genes.txt"), index=False, header=False)
        print("ranked", len(rnk))
    elif what == "rank_adj":
        # sensitivity: disease t-statistic adjusted for the contamination marker scores (leukocyte, skin, erythroid)
        d = T.load_data(cfg); run = T.Run(cfg, d)
        y = d.y.values; v = run.vumc
        rnk0 = pd.read_csv(rnk_path, sep="\t", header=None, index_col=0).iloc[:, 0]
        Lall = run.logX()
        S = T.marker_scores(Lall.loc[v])
        L = Lall.loc[v, list(rnk0.index)]
        Z = np.column_stack([np.ones(int(v.sum())), y[v], S["leukocyte"].values, S["skin"].values, S["erythroid"].values])
        B, *_ = np.linalg.lstsq(Z, L.values, rcond=None)
        R = L.values - Z @ B; dfree = Z.shape[0] - Z.shape[1]
        s2 = (R ** 2).sum(0) / dfree; c = np.linalg.inv(Z.T @ Z)[1, 1]
        t = B[1] / np.sqrt(s2 * c); pv = 2 * stats.t.sf(np.abs(t), dfree)
        stat = -np.log10(np.clip(pv, 1e-300, 1)) * np.sign(t)
        rnk = pd.Series(stat, index=L.columns).replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        rnk.to_csv(rnk_path.replace(".rnk", "_adj.rnk"), sep="\t", header=False)
        rho = stats.spearmanr(rnk, rnk0.reindex(rnk.index)).statistic
        json.dump({"rho_with_unadjusted": float(rho)}, open(os.path.join(tdir, "R7_adj_info.json"), "w"))
        print("adjusted ranking", len(rnk), "rho with unadjusted:", round(rho, 3))
    elif what == "lib":
        import gseapy as gp
        gmt = sys.argv[3]; perm = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
        maxs = int(sys.argv[5]) if len(sys.argv) > 5 else 500
        variant = sys.argv[6] if len(sys.argv) > 6 else ""
        rp = rnk_path.replace(".rnk", f"_{variant}.rnk") if variant else rnk_path
        rnk = pd.read_csv(rp, sep="\t", header=None, index_col=0).iloc[:, 0]
        pre = gp.prerank(rnk=rnk, gene_sets=gmt, permutation_num=perm, min_size=15, max_size=maxs,
                         seed=cfg["seed"], threads=1, outdir=None, verbose=False)
        r = pre.res2d.copy(); r["library"] = os.path.basename(gmt).replace(".gmt", "")
        r["permutations"] = perm; r["max_size"] = maxs
        sub = "R7adj_prerank_" if variant == "adj" else "R7_prerank_"
        r.to_csv(os.path.join(tdir, f"{sub}{r['library'].iloc[0]}.csv"), index=False)
        import resource
        print(gmt, len(r), "peak RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
    elif what == "ora":
        import gseapy as gp
        P = [g for g in cfg["panel"]]
        bg = pd.read_csv(os.path.join(tdir, "R7_background_genes.txt"), header=None).iloc[:, 0].tolist()
        ora = []
        for gmt in cfg["gene_sets"]:
            e = gp.enrich(gene_list=P, gene_sets=gmt, background=bg, outdir=None, verbose=False)
            r = e.results.copy(); r["library"] = os.path.basename(gmt).replace(".gmt", ""); ora.append(r)
        o = pd.concat(ora).sort_values("Adjusted P-value")
        o.to_csv(os.path.join(tdir, "R7_panel_ORA.csv"), index=False)
        print("ORA sets with adjusted p < 0.05:", int((o["Adjusted P-value"] < 0.05).sum()))
    elif what == "combine":
        parts = [pd.read_csv(os.path.join(tdir, f)) for f in sorted(os.listdir(tdir)) if f.startswith("R7_prerank_") and f.endswith(".csv") and f != "R7_prerank_gsea_vumc.csv"]
        res = pd.concat(parts)
        fdr = [c for c in res.columns if "FDR" in c][0]
        res[fdr] = pd.to_numeric(res[fdr], errors="coerce"); res["NES"] = pd.to_numeric(res["NES"], errors="coerce")
        res = res.sort_values([fdr, "NES"], ascending=[True, False])
        res.to_csv(os.path.join(tdir, "R7_prerank_gsea_vumc.csv"), index=False)
        sig = res[res[fdr] < 0.05]
        M = {"GseaNsig": str(len(sig))}
        fj = os.path.join(tdir, "R7_filter.json")
        if os.path.exists(fj):
            fl = json.load(open(fj)); M["GseaNgenes"] = f"{fl['n_genes_ranked']:,}".replace(",", "{,}"); M["GseaNmin"] = str(fl["n_min"])
        M["GseaNtested"] = f"{len(res):,}".replace(",", "{,}")
        for lib, nm in [("MSigDB_Hallmark", "Hallmark"), ("Reactome", "Reactome"), ("GO_BP", "GoBp")]:
            M[f"Gsea{nm}Ntested"] = f"{int(res.library.str.startswith(lib).sum()):,}".replace(",", "{,}")
        for lib, nm in [("MSigDB_Hallmark", "Hallmark"), ("Reactome", "Reactome"), ("GO_BP", "GoBp")]:
            s_ = sig[sig.library.str.startswith(lib)]
            M[f"Gsea{nm}Nsig"] = str(len(s_))
        clean = lambda t: T.tex_escape(t.replace("HALLMARK_", "").replace("REACTOME_", "").replace("GOBP_", "").replace("_", " ").lower())
        h = sig[sig.library.str.startswith("MSigDB_Hallmark")]
        M["GseaHallUp"] = "; ".join(clean(t) for t in h[h.NES > 0].sort_values("NES", ascending=False).Term.head(4)) or "none"
        M["GseaHallDown"] = "; ".join(clean(t) for t in h[h.NES < 0].sort_values("NES").Term.head(4)) or "none"
        up = sig[sig.NES > 0].sort_values("NES", ascending=False); dn = sig[sig.NES < 0].sort_values("NES")
        M["GseaTopUp"] = "; ".join(clean(t) for t in up.Term.head(3)) or "none"
        M["GseaTopDown"] = "; ".join(clean(t) for t in dn.Term.head(3)) or "none"
        M["GseaNup"] = str(int((sig.NES > 0).sum())); M["GseaNdown"] = str(int((sig.NES < 0).sum()))
        p_ora = os.path.join(tdir, "R7_panel_ORA.csv")
        if os.path.exists(p_ora):
            o = pd.read_csv(p_ora); M["OraNsig"] = str(int((o["Adjusted P-value"] < 0.05).sum()))
        os.makedirs(os.path.join(out, "macros"), exist_ok=True)
        json.dump(M, open(os.path.join(out, "macros", "r7.json"), "w"), indent=1)
        print(json.dumps(M, indent=0))
        print(sig[["library", "Term", "NES", fdr]].head(40).to_string(index=False))
    if what == "combine_adj":
        parts = [pd.read_csv(os.path.join(tdir, f)) for f in sorted(os.listdir(tdir)) if f.startswith("R7adj_prerank_") and f.endswith(".csv")]
        res = pd.concat(parts); fdr = [c for c in res.columns if "FDR" in c][0]
        res[fdr] = pd.to_numeric(res[fdr], errors="coerce"); res["NES"] = pd.to_numeric(res["NES"], errors="coerce")
        res = res.sort_values([fdr, "NES"], ascending=[True, False])
        res.to_csv(os.path.join(tdir, "R7_prerank_gsea_vumc_contamination_adjusted.csv"), index=False)
        un = pd.read_csv(os.path.join(tdir, "R7_prerank_gsea_vumc.csv"))
        un = un[un.library.isin(res.library.unique())]
        sig_a = res[res[fdr] < 0.05]; sig_u = un[un[fdr] < 0.05]
        plate = lambda df: df[df.Term.str.contains("PLATELET|COAGULATION|HEMOSTASIS", regex=True)]
        clean = lambda t: T.tex_escape(t.replace("HALLMARK_", "").replace("REACTOME_", "").replace("GOBP_", "").replace("_", " ").lower())
        up = sig_a[sig_a.NES > 0].sort_values("NES", ascending=False); dn = sig_a[sig_a.NES < 0].sort_values("NES")
        M = {"GseaAdjNsig": str(len(sig_a)), "GseaAdjNsigUnadjSameLibs": str(len(sig_u)),
             "GseaAdjNup": str(int((sig_a.NES > 0).sum())), "GseaAdjNdown": str(int((sig_a.NES < 0).sum())),
             "GseaAdjPlateletSig": str(len(plate(sig_a[sig_a.NES > 0]))), "GseaUnadjPlateletSig": str(len(plate(sig_u[sig_u.NES > 0]))),
             "GseaAdjTopUp": "; ".join(clean(t) for t in up.Term.head(3)) or "none",
             "GseaAdjTopDown": "; ".join(clean(t) for t in dn.Term.head(3)) or "none"}
        info = os.path.join(tdir, "R7_adj_info.json")
        if os.path.exists(info):
            M["GseaAdjRho"] = f"{json.load(open(info))['rho_with_unadjusted']:.2f}"
        fmtq = lambda q: "$q<0.001$" if q < 0.001 else f"$q={q:.3f}$"
        cellq = lambda q: "$<$0.001" if q < 0.001 else f"{q:.3f}"
        for nm, term in [("PlAct", "REACTOME_PLATELET_ACTIVATION_SIGNALING_AND_AGGREGATION"), ("Coag", "HALLMARK_COAGULATION"),
                         ("PlAgg", "GOBP_PLATELET_AGGREGATION"), ("TrElong", "REACTOME_EUKARYOTIC_TRANSLATION_ELONGATION"),
                         ("Allo", "HALLMARK_ALLOGRAFT_REJECTION"), ("Bcr", "GOBP_B_CELL_RECEPTOR_SIGNALING_PATHWAY"),
                         ("Myc", "HALLMARK_MYC_TARGETS_V1"), ("Splice", "REACTOME_MRNA_SPLICING")]:
            for tag, df in [("Unadj", un), ("Adj", res)]:
                r = df[df.Term == term]
                if len(r):
                    M[f"Gsea{nm}{tag}NES"] = f"{float(r.NES.iloc[0]):.2f}".replace("-", "$-$")
                    M[f"Gsea{nm}{tag}Q"] = fmtq(float(r[fdr].iloc[0]))
                    M[f"Gsea{nm}{tag}Cell"] = f"{M[f'Gsea{nm}{tag}NES']} ({cellq(float(r[fdr].iloc[0]))})"
        for lib, nm in [("MSigDB_Hallmark", "Hallmark"), ("Reactome", "Reactome"), ("GO_BP", "GoBp")]:
            M[f"GseaAdj{nm}Nsig"] = str(int(sig_a.library.str.startswith(lib).sum()))
        both = set(sig_a.Term) & set(sig_u.Term)
        M["GseaAdjOverlap"] = str(len(both))
        json.dump(M, open(os.path.join(out, "macros", "r7adj.json"), "w"), indent=1)
        print(json.dumps(M, indent=0)); print(sig_a[["library", "Term", "NES", fdr]].head(25).to_string(index=False))
    gc.collect()


if __name__ == "__main__":
    main()
