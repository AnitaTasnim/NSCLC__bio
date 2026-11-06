#!/usr/bin/env python3
"""Is the biology more reproducible than the gene list?  Split-half replication within Vumc.

    python pathway_stability.py config.json full                     # rank-based GSEA on the full Vumc rankings
    python pathway_stability.py config.json run [n_splits] [perm]    # default 25 splits, 500 permutations
    python pathway_stability.py config.json summarise

Design. Within Vumc (one site, so site cannot drive any difference) the samples are split at random into two
disjoint, class-stratified halves, 25 times (seeds 0..24). The two halves of a split contain different patients,
so agreement between them is replication in independent patients. Each half is ranked as in r7_gsea.py
(signed -log10 Mann-Whitney p-value of log2-CPM, NSCLC vs healthy, same expressed genes) and, as a sensitivity
analysis, with the disease t-statistic adjusted for the leukocyte, skin and erythroid marker scores.

Gene-set test. The within-Vumc rankings are strongly skewed (about 86% of genes rank lower in NSCLC, a compositional
effect of the higher platelet-RNA fraction in patient samples), and the skew differs between halves. The weighted
GSEA statistic (weight = 1) is normalised against a permutation null that depends on this skew and can return
degenerate results (NES = +/-1, p = 1) for sets with a large ES. Pre-ranked GSEA is therefore run here with the
classic unweighted Kolmogorov-Smirnov statistic (weight = 0), which uses ranks only (GSEApy, Hallmark and Reactome,
15-500 genes, seed 42).

Compared between the two halves of each split:
  gene level    - Jaccard of the top-27 and top-200 genes by |statistic|; Spearman of the gene statistics;
                  replication (BH q < 0.05, same direction, both halves) of each of the 27 panel genes
  pathway level - Jaccard of the significant gene sets (q < 0.05), Spearman of NES; replication of the platelet sets
"""
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

KEY_SETS = {
    "PlAct": "REACTOME_PLATELET_ACTIVATION_SIGNALING_AND_AGGREGATION",
    "PlCa": "REACTOME_RESPONSE_TO_ELEVATED_PLATELET_CYTOSOLIC_CA2",
    "Hemo": "REACTOME_HEMOSTASIS",
    "Coag": "HALLMARK_COAGULATION",
}
LIBS = ["MSigDB_Hallmark_v7.5.1.gmt", "Reactome_MSigDB_v7.5.1.gmt"]
RANKINGS = ("unadj", "adj")


def mw_rank(L, yv):
    u = stats.mannwhitneyu(L[yv == 1], L[yv == 0], axis=0)
    auc = u.statistic / ((yv == 1).sum() * (yv == 0).sum())
    return -np.log10(np.clip(u.pvalue, 1e-300, 1)) * np.sign(auc - 0.5), u.pvalue


def adj_rank(L, yv, S):
    Z = np.column_stack([np.ones(len(yv)), yv, S[:, 0], S[:, 1], S[:, 2]])
    B, *_ = np.linalg.lstsq(Z, L, rcond=None)
    R = L - Z @ B; dfree = Z.shape[0] - Z.shape[1]
    s2 = (R ** 2).sum(0) / dfree; c = np.linalg.inv(Z.T @ Z)[1, 1]
    t = B[1] / np.sqrt(s2 * c); pv = 2 * stats.t.sf(np.abs(t), dfree)
    return -np.log10(np.clip(pv, 1e-300, 1)) * np.sign(t)


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else np.nan


def setup(cfg):
    d = T.load_data(cfg); r = T.Run(cfg, d)
    y = d.y.values; v = r.vumc
    Xv = d.X.loc[v]; cpm = Xv.div(Xv.sum(axis=1), axis=0) * 1e6
    n_min = int(min((y[v] == 1).sum(), (y[v] == 0).sum()))
    keep = ((cpm >= 1).sum(axis=0) >= n_min).values
    Lall = r.logX().loc[v]
    S = T.marker_scores(Lall)[["leukocyte", "skin", "erythroid"]].values
    L = Lall.loc[:, keep]
    print(f"Vumc: {L.shape[0]} samples ({int(y[v].sum())} NSCLC), {L.shape[1]} expressed genes")
    return L, y[v], S


def gsea(st, gmts, perm, cfg):
    import gseapy as gp
    st = pd.Series(st).replace([np.inf, -np.inf], np.nan).dropna()
    st = st[~st.index.duplicated()].sort_values(ascending=False)
    out = []
    for gmt in gmts:
        pre = gp.prerank(rnk=st, gene_sets=gmt, permutation_num=perm, weight=0, min_size=15, max_size=500,
                         seed=42, threads=cfg.get("n_jobs", 2), outdir=None, verbose=False)
        g = pre.res2d[["Term", "ES", "NES", "NOM p-val", "FDR q-val"]].copy()
        g["library"] = os.path.basename(gmt).replace(".gmt", ""); out.append(g)
    g = pd.concat(out)
    for c in ("ES", "NES", "NOM p-val", "FDR q-val"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    return g


def full(cfg, perm):
    tdir = os.path.join(cfg["out_dir"], "tables"); os.makedirs(tdir, exist_ok=True)
    L, yv, S = setup(cfg); genes = np.array(L.columns)
    gmts = [os.path.join(os.path.dirname(cfg["gene_sets"][0]), g) for g in LIBS]
    s_un, _ = mw_rank(L.values, yv); s_ad = adj_rank(L.values, yv, S)
    ref = os.path.join(os.path.dirname(HERE), "results", "tables", "R7_vumc_ranked_genes.rnk")
    if os.path.exists(ref):
        r0 = pd.read_csv(ref, sep="\t", header=None, index_col=0).iloc[:, 0]
        print("ranking vs published: Spearman", round(stats.spearmanr(pd.Series(s_un, index=genes).reindex(r0.index), r0).statistic, 6))
    print(f"skew of the full-Vumc ranking: {(s_un < 0).mean():.3f} of genes lower in NSCLC; adjusted {(s_ad < 0).mean():.3f}")
    res = []
    for nm, st in (("unadj", s_un), ("adj", s_ad)):
        g = gsea(pd.Series(st, index=genes), gmts, perm, cfg); g["ranking"] = nm; res.append(g)
    res = pd.concat(res); res.to_csv(os.path.join(tdir, "PS_full_vumc_rank_based_gsea.csv"), index=False)
    json.dump({"frac_lower": float((s_un < 0).mean()), "frac_lower_adj": float((s_ad < 0).mean())},
              open(os.path.join(tdir, "PS_skew.json"), "w"))
    print(res[res.Term.isin(KEY_SETS.values())].to_string(index=False))


def run(cfg, n_splits, perm):
    tdir = os.path.join(cfg["out_dir"], "tables"); os.makedirs(tdir, exist_ok=True)
    part, gpart, spart = (os.path.join(tdir, f) for f in ("PS_halves.csv", "PS_gsea_halves.csv", "PS_gene_stats.csv"))
    L, yv, S = setup(cfg); genes = np.array(L.columns); Lv = L.values
    P = [g for g in cfg["panel"] if g in L.columns]
    done = set(pd.read_csv(part).split.unique()) if os.path.exists(part) else set()
    gmts = [os.path.join(os.path.dirname(cfg["gene_sets"][0]), g) for g in LIBS]
    idx1, idx0 = np.where(yv == 1)[0], np.where(yv == 0)[0]
    for s in range(n_splits):
        if s in done:
            continue
        rs = np.random.RandomState(s)
        a1 = rs.permutation(idx1); a0 = rs.permutation(idx0)
        halves = (("A", np.r_[a1[: len(a1) // 2], a0[: len(a0) // 2]]), ("B", np.r_[a1[len(a1) // 2:], a0[len(a0) // 2:]]))
        rows, grows, srows = [], [], []
        for h, ix in halves:
            st, p = mw_rank(Lv[ix], yv[ix])
            q = pd.Series(T.bh(p), index=genes)
            sr = pd.Series(st, index=genes)
            top = sr.abs().sort_values(ascending=False)
            rows.append({"split": s, "half": h, "n": len(ix), "n_nsclc": int(yv[ix].sum()),
                         "frac_lower": float((st < 0).mean()), "n_sig_genes": int((q < 0.05).sum()),
                         "top27": ";".join(top.index[:27]), "top200": ";".join(top.index[:200]),
                         "panel_sig_up": ";".join(g for g in P if q[g] < 0.05 and sr[g] > 0),
                         "panel_sig_down": ";".join(g for g in P if q[g] < 0.05 and sr[g] < 0)})
            srows.append(sr.rename(f"s{s}{h}"))
            for nm, stt in (("unadj", st), ("adj", adj_rank(Lv[ix], yv[ix], S[ix]))):
                g = gsea(pd.Series(stt, index=genes), gmts, perm, cfg)
                g["split"] = s; g["half"] = h; g["ranking"] = nm; grows.append(g)
        pd.DataFrame(rows).to_csv(part, mode="a", header=not os.path.exists(part), index=False)
        pd.concat(grows).to_csv(gpart, mode="a", header=not os.path.exists(gpart), index=False)
        pd.DataFrame(srows).to_csv(spart, mode="a", header=not os.path.exists(spart))
        print(f"split {s} done", flush=True); gc.collect()


def summarise(cfg):
    out = cfg["out_dir"]; tdir = os.path.join(out, "tables")
    H = pd.read_csv(os.path.join(tdir, "PS_halves.csv")).fillna("")
    G = pd.read_csv(os.path.join(tdir, "PS_gsea_halves.csv"))
    Sg = pd.read_csv(os.path.join(tdir, "PS_gene_stats.csv"), index_col=0)
    P = list(cfg["panel"])
    n = H.split.nunique(); nh = 2 * n
    rows = []
    for s, h in H.groupby("split"):
        a, b = h[h.half == "A"].iloc[0], h[h.half == "B"].iloc[0]
        rec = {"split": s,
               "gene_top27_J": jacc(a.top27.split(";"), b.top27.split(";")),
               "gene_top200_J": jacc(a.top200.split(";"), b.top200.split(";")),
               "gene_stat_rho": stats.spearmanr(Sg.loc[f"s{s}A"], Sg.loc[f"s{s}B"], nan_policy="omit").statistic}
        up_a, up_b = set(filter(None, a.panel_sig_up.split(";"))), set(filter(None, b.panel_sig_up.split(";")))
        dn_a, dn_b = set(filter(None, a.panel_sig_down.split(";"))), set(filter(None, b.panel_sig_down.split(";")))
        rec["panel_rep_both"] = len(up_a & up_b) + len(dn_a & dn_b)
        for rk in RANKINGS:
            ga = G[(G.split == s) & (G.half == "A") & (G.ranking == rk)].set_index("Term")
            gb = G[(G.split == s) & (G.half == "B") & (G.ranking == rk)].set_index("Term")
            common = ga.index.intersection(gb.index)
            sa, sb = set(ga.index[ga["FDR q-val"] < 0.05]), set(gb.index[gb["FDR q-val"] < 0.05])
            rec[f"{rk}_path_sig_J"] = jacc(sa, sb)
            rec[f"{rk}_path_nes_rho"] = stats.spearmanr(ga.loc[common, "NES"], gb.loc[common, "NES"]).statistic
            for k, t in KEY_SETS.items():
                for hh, g in (("A", ga), ("B", gb)):
                    rec[f"{rk}_{k}_{hh}"] = bool(t in g.index and g.loc[t, "NES"] > 0 and g.loc[t, "FDR q-val"] < 0.05)
                    rec[f"{rk}_{k}_{hh}_NES"] = float(g.loc[t, "NES"]) if t in g.index else np.nan
        rows.append(rec)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(tdir, "PS_split_half_summary.csv"), index=False)
    # replication of every gene set: significant with the same sign in both halves of a split
    rep = []
    for (rk, t), g in G.groupby(["ranking", "Term"]):
        q = g.pivot_table(index="split", columns="half", values="FDR q-val")
        e = g.pivot_table(index="split", columns="half", values="NES")
        rep.append({"ranking": rk, "Term": t, "library": g.library.iloc[0],
                    "frac_halves_sig_up": float(((q < 0.05) & (e > 0)).values.mean()),
                    "frac_splits_up_both": float(((q < 0.05) & (e > 0)).all(axis=1).mean()),
                    "frac_splits_down_both": float(((q < 0.05) & (e < 0)).all(axis=1).mean()),
                    "median_NES": float(g.NES.median()), "min_NES": float(g.NES.min())})
    rep = pd.DataFrame(rep).sort_values(["ranking", "frac_splits_up_both"], ascending=[False, False])
    rep.to_csv(os.path.join(tdir, "PS_gene_set_replication.csv"), index=False)
    # per-gene replication of the 27 panel genes
    tested = [g for g in P if g in Sg.columns]
    prow = []
    for g in P:
        up = H.panel_sig_up.str.split(";").apply(lambda x: g in x); dn = H.panel_sig_down.str.split(";").apply(lambda x: g in x)
        H_ = H.assign(up=up, dn=dn)
        both = H_.groupby("split").apply(lambda d: bool(d.up.all() or d.dn.all()), include_groups=False)
        top = H.top27.str.split(";").apply(lambda x: g in x)
        prow.append({"gene": g, "expressed_in_vumc": g in tested, "frac_halves_sig": float((up | dn).mean()), "frac_splits_rep_both": float(both.mean()),
                     "frac_halves_top27": float(top.mean())})
    PR = pd.DataFrame(prow).sort_values("frac_splits_rep_both", ascending=False)
    PR.to_csv(os.path.join(tdir, "PS_panel_gene_replication.csv"), index=False)
    gtop = {}
    for s, h in H.groupby("split"):
        for g in set(h[h.half == "A"].top27.iloc[0].split(";")) & set(h[h.half == "B"].top27.iloc[0].split(";")):
            gtop[g] = gtop.get(g, 0) + 1
    gene_rep = pd.Series(gtop, dtype=float) / n
    med_iqr = lambda x: f"{np.median(x):.2f} (IQR {np.percentile(x, 25):.2f}--{np.percentile(x, 75):.2f})"
    f2 = lambda x: f"{x:.2f}"
    sk = json.load(open(os.path.join(tdir, "PS_skew.json"))) if os.path.exists(os.path.join(tdir, "PS_skew.json")) else {}
    M = {"PsNsplits": str(n), "PsNhalves": str(nh), "PsNhalf": str(int(H.n.median())),
         "PsNhalfNsclc": str(int(H.n_nsclc.median())),
         "PsGeneTopJ": med_iqr(R.gene_top27_J), "PsGeneTopJMed": f2(R.gene_top27_J.median()),
         "PsGeneTopTwoHundredJ": med_iqr(R.gene_top200_J), "PsGeneTopTwoHundredJMed": f2(R.gene_top200_J.median()),
         "PsGeneRho": med_iqr(R.gene_stat_rho), "PsGeneRhoMed": f2(R.gene_stat_rho.median()),
         "PsTopGenesEver": str(len(set(";".join(H.top27).split(";")))),
         "PsGenesAlwaysTop": str(int((gene_rep >= 1).sum())),
         "PsGenesHalfTop": str(int((gene_rep >= 0.5).sum())),
         "PsPanelRepMed": str(int(np.median(R.panel_rep_both))),
         "PsPanelRepRange": f"{int(R.panel_rep_both.min())}--{int(R.panel_rep_both.max())}",
         "PsPanelAlwaysRep": str(int((PR.frac_splits_rep_both >= 1).sum())),
         "PsPanelNeverRep": str(int((PR.frac_splits_rep_both == 0).sum())),
         "PsPanelInTopAny": str(int((PR.frac_halves_top27 > 0).sum())),
         "PsPanelTested": str(len(tested)), "PsPanelNotExpressed": str(len(P) - len(tested)),
         "PsPanelNeverRepTested": str(int(((PR.frac_splits_rep_both == 0) & PR.expressed_in_vumc).sum())),
         "PsPanelSomeRep": str(int(((PR.frac_splits_rep_both > 0) & (PR.frac_splits_rep_both < 1)).sum())),
         "PsFracLower": f"{100 * sk.get('frac_lower', np.nan):.0f}",
         "PsHalfFracLower": f"{100 * H.frac_lower.min():.0f}--{100 * H.frac_lower.max():.0f}"}
    for rk, tag in (("unadj", ""), ("adj", "Adj")):
        M[f"PsPath{tag}J"] = med_iqr(R[f"{rk}_path_sig_J"]); M[f"PsPath{tag}JMed"] = f2(R[f"{rk}_path_sig_J"].median())
        M[f"PsPath{tag}Rho"] = med_iqr(R[f"{rk}_path_nes_rho"]); M[f"PsPath{tag}RhoMed"] = f2(R[f"{rk}_path_nes_rho"].median())
        for k in KEY_SETS:
            M[f"Ps{tag}{k}Halves"] = f"{int(R[f'{rk}_{k}_A'].sum() + R[f'{rk}_{k}_B'].sum())}/{nh}"
            M[f"Ps{tag}{k}Both"] = f"{int((R[f'{rk}_{k}_A'] & R[f'{rk}_{k}_B']).sum())}/{n}"
            nes = pd.concat([R[f"{rk}_{k}_A_NES"], R[f"{rk}_{k}_B_NES"]])
            M[f"Ps{tag}{k}NES"] = f"{nes.median():.2f} ({nes.min():.2f}--{nes.max():.2f})"
    fv = os.path.join(tdir, "PS_full_vumc_rank_based_gsea.csv")
    if os.path.exists(fv):
        F = pd.read_csv(fv)
        for rk, tag in (("unadj", "Un"), ("adj", "Adj")):
            f = F[F.ranking == rk]
            M[f"RbFull{tag}Nsig"] = str(int((f["FDR q-val"] < 0.05).sum())); M[f"RbFull{tag}Ntested"] = str(len(f))
            for k, t in KEY_SETS.items():
                r_ = f[f.Term == t]
                if len(r_):
                    qq = float(r_["FDR q-val"].iloc[0])
                    M[f"RbFull{tag}{k}"] = f"NES {float(r_.NES.iloc[0]):.2f}, " + ("$q<0.001$" if qq < 0.001 else f"$q={qq:.3f}$")
            for k, t in (("Myc", "HALLMARK_MYC_TARGETS_V1"), ("Allo", "HALLMARK_ALLOGRAFT_REJECTION"),
                         ("Splice", "REACTOME_MRNA_SPLICING"), ("TrElong", "REACTOME_EUKARYOTIC_TRANSLATION_ELONGATION")):
                r_ = f[f.Term == t]
                if len(r_):
                    qq = float(r_["FDR q-val"].iloc[0])
                    M[f"RbFull{tag}{k}"] = f"NES {float(r_.NES.iloc[0]):.2f}".replace("-", "$-$") + ", " + ("$q<0.001$" if qq < 0.001 else f"$q={qq:.3f}$")
    os.makedirs(os.path.join(out, "macros"), exist_ok=True)
    json.dump(M, open(os.path.join(out, "macros", "pathstab.json"), "w"), indent=1)
    print(json.dumps(M, indent=1))
    print(rep[rep.Term.isin(KEY_SETS.values())].to_string(index=False))
    print(PR.to_string(index=False))
    fig(R, rep, PR, out, M)


def fig(R, rep, PR, out, M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fdir = os.path.join(out, "figures"); os.makedirs(fdir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 5.6), gridspec_kw={"width_ratios": [1.0, 1.05]})
    data = [R.gene_top27_J, R.gene_top200_J, R.unadj_path_sig_J, R.adj_path_sig_J]
    labs = ["Top-27\ngenes", "Top-200\ngenes", "Gene sets\nq < 0.05", "Gene sets\nq < 0.05\n(adjusted)"]
    cols = ["#b5532a", "#d9925f", "#2a6f97", "#6aa5c8"]
    bp = ax[0].boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
    for p_, c in zip(bp["boxes"], cols):
        p_.set_facecolor(c); p_.set_alpha(0.8)
    for m in bp["medians"]:
        m.set_color("black")
    rs = np.random.RandomState(0)
    for i, x in enumerate(data, 1):
        ax[0].scatter(i + rs.uniform(-0.12, 0.12, len(x)), x, s=8, color="black", alpha=0.55, zorder=3)
    ax[0].set_xticks(range(1, 5)); ax[0].set_xticklabels(labs, fontsize=9)
    ax[0].set_ylabel("Jaccard between the two halves", fontsize=10); ax[0].set_ylim(0, 1)
    ax[0].set_title(f"A  Agreement between halves ({M['PsNsplits']} splits)", fontsize=10, loc="left")
    # panel B: replication in both halves, panel genes vs platelet gene sets
    names = {"REACTOME_PLATELET_ACTIVATION_SIGNALING_AND_AGGREGATION": "Platelet activation (Reactome)",
             "REACTOME_RESPONSE_TO_ELEVATED_PLATELET_CYTOSOLIC_CA2": "Platelet cytosolic Ca$^{2+}$ (Reactome)",
             "REACTOME_HEMOSTASIS": "Haemostasis (Reactome)", "HALLMARK_COAGULATION": "Coagulation (Hallmark)"}
    r_un = rep[(rep.ranking == "unadj") & rep.Term.isin(names)].set_index("Term").loc[list(names)]
    r_ad = rep[(rep.ranking == "adj") & rep.Term.isin(names)].set_index("Term").loc[list(names)]
    pr = PR.sort_values("frac_splits_rep_both", ascending=True)
    labels = [g + ("" if e else " (n.e.)") for g, e in zip(pr.gene, pr.expressed_in_vumc)] + [""] + [names[t] for t in names][::-1]
    vals_g = list(pr.frac_splits_rep_both)
    yy_g = np.arange(len(vals_g))
    ax[1].barh(yy_g, vals_g, color="#b5532a", alpha=0.85, height=0.7, label="Panel gene (BH q < 0.05, same direction)")
    yy_s = np.arange(len(vals_g) + 1, len(vals_g) + 1 + len(names))
    ax[1].barh(yy_s + 0.18, list(r_un.frac_splits_up_both)[::-1], color="#2a6f97", height=0.36, label="Gene set, unadjusted")
    ax[1].barh(yy_s - 0.18, list(r_ad.frac_splits_up_both)[::-1], color="#6aa5c8", height=0.36, label="Gene set, contamination-adjusted")
    ax[1].set_yticks(list(yy_g) + [len(vals_g)] + list(yy_s)); ax[1].set_yticklabels(labels, fontsize=7.5)
    ax[1].set_xlim(0, 1.0); ax[1].set_xlabel("Fraction of splits replicated in both halves", fontsize=9.5)
    ax[1].set_title("B  Replicated in both halves", fontsize=10, loc="left")
    ax[1].legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(fdir, f"fig_pathway_stability.{e}"), dpi=300, bbox_inches="tight")
    print("figure written")


if __name__ == "__main__":
    cfg = T.load_config(sys.argv[1]); what = sys.argv[2]
    if what == "full":
        full(cfg, int(sys.argv[3]) if len(sys.argv) > 3 else 1000)
    elif what == "run":
        run(cfg, int(sys.argv[3]) if len(sys.argv) > 3 else 25, int(sys.argv[4]) if len(sys.argv) > 4 else 500)
    else:
        summarise(cfg)
