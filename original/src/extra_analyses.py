#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extra_analyses.py - the smaller analyses that are not part of tep_analysis.py.
Each writes its numbers to <out>/macros/<name>.json (merged by `run_steps.py ... merge`).

    python extra_analyses.py config.json repro        # seeded re-run of the original pipeline (~20 min)
    python extra_analyses.py config.json libsize      # library size and log2-CPM sensitivity
    python extra_analyses.py config.json r0extra      # fold mean/SD of the site-aware regimes, AUPRC
    python extra_analyses.py config.json nestedextra  # Jaccard summaries of the nested panels, Tier-2 sizes
    python extra_analyses.py config.json subsvumc     # substitution test with site held constant
    python extra_analyses.py config.json r5extra      # re-selection without suspect genes vs. same pipeline
Run `repro` first (the last two use its output) and `nestedextra`/`subsvumc` after the `nested` step.
"""
import itertools
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tep_analysis as T  # noqa: E402
from selection_pipeline import phase1_select, stage1_mw  # noqa: E402


def setup(cfg_path):
    cfg = T.load_config(cfg_path)
    d = T.load_data(cfg)
    run = T.Run(cfg, d)
    out = cfg["out_dir"]
    os.makedirs(os.path.join(out, "macros"), exist_ok=True)
    return cfg, d, run, out


def save(out, name, m):
    json.dump(m, open(os.path.join(out, "macros", f"{name}.json"), "w"), indent=1)
    print(name, m)


def panel_csv(cfg):
    return cfg.get("panel_csv", "inputs/DEFINITIVE_biomarker_panel.csv")


def repro(cfg_path):
    cfg, d, run, out = setup(cfg_path)
    y = d.y.values
    pub = pd.read_csv(panel_csv(cfg), index_col=0)
    P = set(pub.index)
    res = {}
    for tag, kw in [("full_ppi_prior", dict(use_ppi=True, ppi_scores=pub["ppi_score"].to_dict(), use_prior=True)),
                    ("fold_recomputable", dict(use_ppi=False, use_prior=False))]:
        panel, det = phase1_select(d.X, y, B=500, n_jobs=cfg.get("sel_n_jobs", 2), return_details=True, verbose=True, **kw)
        det["scores"].to_csv(os.path.join(out, "tables", f"replica_scores_{tag}.csv"))
        det["stability"].to_csv(os.path.join(out, "tables", f"replica_stability_{tag}.csv"))
        res[tag] = dict(panel=panel, log={k: v for k, v in det["log"].items() if k != "mrmr"})
    json.dump(res, open(os.path.join(out, "tables", "replica_full.json"), "w"), indent=1)
    a = res["full_ppi_prior"]; L = a["log"]; S = set(a["panel"])
    m = dict(ReproLassoN=str(L["stage2_lasso_n"]), ReproEnetN=str(L["stage2_enet_n"]), ReproConsN=str(L["stage2_n"]),
             ReproCodingN=str(L["stage3_n"]), ReproStableN=str(L["stage5_n"]), ReproOverlap=str(len(P & S)),
             ReproJaccard=f"{len(P & S) / len(P | S):.3f}", ReproNew=", ".join(sorted(S - P)) or "none",
             ReproMissing=", ".join(sorted(P - S)) or "none")
    F = set(res["fold_recomputable"]["panel"])
    m.update(FoldRecOverlap=str(len(P & F)), FoldRecJaccard=f"{len(P & F) / len(P | F):.3f}", FoldRecN=str(len(F)),
             FoldRecNew=", ".join(sorted(F - P)) or "none", FoldRecMissing=", ".join(sorted(P - F)) or "none")
    st = pd.read_csv(os.path.join(out, "tables", "replica_stability_full_ppi_prior.csv"), index_col=0).iloc[:, 0]
    m["ReproBootCorr"] = f"{np.corrcoef(pub['boot_stability'], st.reindex(pub.index).fillna(0))[0, 1]:.2f}"
    save(out, "repro", m)


def libsize(cfg_path):
    cfg, d, run, out = setup(cfg_path)
    y = d.y.values; v = run.vumc; P = run.panel_in(d.X)
    lib = d.X.sum(axis=1).values
    L = run.logX()[P].values
    f3 = lambda x: f"{x:.3f}"; f4 = lambda x: f"{x:.4f}"
    o1, _ = T.cv_oof(d.X.loc[v, P].values, y[v], run.folds(y[v]), run.xgb)
    o2, _ = T.cv_oof(L[v], y[v], run.folds(y[v]), run.xgb)
    dv = T.delong_paired(y[v], o1, o2)
    o1a, _ = T.cv_oof(d.X[P].values, y, run.folds(), run.xgb)
    o2a, _ = T.cv_oof(L, y, run.folds(), run.xgb)
    da = T.delong_paired(y, o1a, o2a)
    o3, _ = T.cv_oof(np.c_[L, np.log2(lib)], y, run.folds(), run.xgb)
    pm = lambda p: "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"
    save(out, "libsize", dict(
        LibAUCAll=f3(T.roc_auc_score(y, lib)), LibAUCVumc=f3(T.roc_auc_score(y[v], lib[v])),
        CpmVumcAUC=f4(dv["auc2"]), RawVumcAUC=f4(dv["auc1"]), CpmVumcP=pm(dv["p"]),
        CpmVumcDelta=f"{100 * (dv['auc1'] - dv['auc2']):.1f}",
        CpmDelta=f"{100 * (da['auc1'] - da['auc2']):.1f}", CpmDeltaCI=f"{100 * da['lo95']:.1f} to {100 * da['hi95']:.1f}",
        CpmDeltaP=pm(da["p"]), CpmPlusLibAUC=f4(T.roc_auc_score(y, o3))))


def r0extra(cfg_path):
    cfg, d, run, out = setup(cfg_path)
    r0 = pd.read_csv(os.path.join(out, "tables", "R0_core_estimates.csv")).set_index("analysis")
    f4 = lambda x: f"{x:.4f}"
    m = {}
    for key, name in [("Site-stratified 5-fold CV", "CoreSiteStrat"), ("Vumc-only 5-fold CV", "CoreVumc"),
                      ("Fixed-panel 5-fold CV", "CoreFixed")]:
        m[name + "FoldMean"] = f4(r0.loc[key, "fold_mean"]); m[name + "FoldSD"] = f4(r0.loc[key, "fold_sd"])
    c = pickle.load(open(os.path.join(out, "cache", "run_cache.pkl"), "rb"))
    m["CoreFixedAUPRC"] = f4(average_precision_score(d.y.values, c["oof_P27"]))
    save(out, "r0extra", m)


def nestedextra(cfg_path):
    cfg, d, run, out = setup(cfg_path)
    c = pickle.load(open(os.path.join(out, "cache", "run_cache.pkl"), "rb"))
    res = c["nested_six_stage"]
    rep = json.load(open(os.path.join(out, "tables", "replica_full.json")))
    F = set(rep["fold_recomputable"]["panel"])
    P = set(pd.read_csv(panel_csv(cfg), index_col=0).index)
    jF = [len(F & set(p)) / len(F | set(p)) for p in res["panels"]]
    jP = [len(P & set(p)) / len(P | set(p)) for p in res["panels"]]
    jpair = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in itertools.combinations(res["panels"], 2)]
    y = d.y.values
    t2 = []
    for tr, _ in run.folds():
        mw = stage1_mw(d.X.iloc[tr], y[tr])
        t2.append(int(((mw.padj < 0.05) & (mw.log2FC.abs() > 0.5)).sum()))
    save(out, "nestedextra", dict(
        NestedJRecMean=f"{np.mean(jF):.3f}", NestedJRecMin=f"{min(jF):.3f}", NestedJRecMax=f"{max(jF):.3f}",
        NestedJPairMean=f"{np.mean(jpair):.3f}", NestedJMin=f"{min(jP):.3f}", NestedJMax=f"{max(jP):.3f}",
        NestedTierTwoRange=f"{min(t2):,}--{max(t2):,}".replace(",", "{,}")))


def subsvumc(cfg_path):
    cfg, d, run, out = setup(cfg_path)
    c = pickle.load(open(os.path.join(out, "cache", "run_cache.pkl"), "rb"))
    panels = c["nested_six_stage"]["panels"]
    P = run.panel_in(d.X); y = d.y.values; v = run.vumc
    Lv = run.logX().loc[v]; yv = y[v]
    resid = Lv - Lv.groupby(yv).transform("mean")
    subs = sorted({g for p in panels for g in p if g not in P})
    mx = np.abs(T.rank_corr_matrix(resid[subs].values, resid[P].values)).max(1)
    rng = np.random.default_rng(42)
    pool = [g for g in T.tier2_genes(run) if g not in set(P) | set(subs)]
    null = rng.choice(pool, size=min(1000, len(pool)), replace=False)
    nmx = np.abs(T.rank_corr_matrix(resid[list(null)].values, resid[P].values)).max(1)
    pd.DataFrame({"gene": subs, "max_abs_rho_within_vumc": mx}).to_csv(
        os.path.join(out, "tables", "R4_substitutes_within_vumc.csv"), index=False)
    save(out, "subsvumc", dict(SubVumcCorrPct=f"{100 * np.mean(mx >= 0.5):.0f}", SubVumcNullPct=f"{100 * np.mean(nmx >= 0.5):.0f}",
                               SubVumcMedRho=f"{np.median(mx):.2f}", SubVumcNullMedRho=f"{np.median(nmx):.2f}",
                               SubVumcMWP=T.pmac(stats.mannwhitneyu(mx, nmx, alternative="greater").pvalue)))


def r5extra(cfg_path):
    cfg, d, run, out = setup(cfg_path)
    rep = json.load(open(os.path.join(out, "tables", "replica_full.json")))
    F = set(rep["fold_recomputable"]["panel"])
    R = set(pd.read_csv(os.path.join(out, "tables", "R5_reselected_panel.csv")).gene)
    save(out, "r5extra", dict(QcReselectFoldRecJaccard=f"{len(F & R) / len(F | R):.3f}", QcReselectFoldRecOverlap=str(len(F & R)),
                              QcReselectVsFoldRecNew=", ".join(sorted(R - F)) or "none",
                              QcReselectVsFoldRecLost=", ".join(sorted(F - R)) or "none"))


def weights(cfg_path, n_draws=2000):
    """Composite-weight sensitivity on the Stage-6 sub-scores of the seeded full-cohort re-run (73 stable genes).
    Each weight is multiplied by an independent uniform factor in [1-a, 1+a] and the vector is renormalised to sum 1;
    the panel is then re-formed exactly as in Stage 6 (top 30 by composite, artefact QC). a = 0.2 (main) and 0.5."""
    from selection_pipeline import DEFAULT_WEIGHTS, is_clean_gene
    cfg = T.load_config(cfg_path); out = cfg["out_dir"]
    os.makedirs(os.path.join(out, "macros"), exist_ok=True)
    sc = pd.read_csv(os.path.join(out, "tables", "replica_scores_full_ppi_prior.csv"), index_col=0)
    cols = ["stat_score", "effect_score", "lasso_score", "boot_score", "ppi_score", "ml_prior"]
    w0 = np.array([DEFAULT_WEIGHTS[k] for k in ["stat", "effect", "lasso", "boot", "ppi", "prior"]])
    S = sc[cols].values; genes = np.array(sc.index)
    pub = set(pd.read_csv(panel_csv(cfg), index_col=0).index)

    def panel_for(w):
        comp = S @ w
        order = np.lexsort((genes, -comp))          # composite descending, ties by name (as in phase1_select)
        return [g for g in genes[order][:30] if is_clean_gene(g)]

    base = set(panel_for(w0))
    m = {"WeightN": f"{n_draws:,}".replace(",", "{,}"), "WeightBaseVsPub": f"{len(base & pub) / len(base | pub):.3f}",
         "WeightWorkingN": str(len(genes))}
    rows = []
    for a, tag in [(0.2, ""), (0.5, "Wide")]:
        rng = np.random.default_rng(42)
        panels = []
        for _ in range(n_draws):
            w = w0 * rng.uniform(1 - a, 1 + a, size=len(w0)); w = w / w.sum()
            panels.append(set(panel_for(w)))
        jpub = np.array([len(p & pub) / len(p | pub) for p in panels])
        jbase = np.array([len(p & base) / len(p | base) for p in panels])
        M = np.array([[g in p for g in genes] for p in panels], dtype=float)
        inter = M @ M.T; sz = M.sum(1); uni = sz[:, None] + sz[None, :] - inter
        iu = np.triu_indices(len(panels), 1); jpair = (inter[iu] / uni[iu])
        incl = pd.Series(M.mean(0), index=genes)
        pin = incl.reindex(sorted(pub)).fillna(0.0)
        prep = incl.reindex(sorted(pub & base))          # published genes that the re-run also selects at the original weights
        m.update({f"Weight{tag}JMean": f"{jpub.mean():.3f}", f"Weight{tag}JMin": f"{jpub.min():.3f}",
                  f"Weight{tag}JMax": f"{jpub.max():.3f}", f"Weight{tag}JBaseMean": f"{jbase.mean():.3f}",
                  f"Weight{tag}JPairMean": f"{jpair.mean():.3f}", f"Weight{tag}MinIncl": f"{100 * pin.min():.0f}",
                  f"Weight{tag}MinInclGene": T.tex_escape(pin.idxmin()),
                  f"Weight{tag}NbelowHalf": str(int((pin < 0.5).sum())),
                  f"Weight{tag}MinInclRep": f"{100 * prep.min():.0f}", f"Weight{tag}MinInclRepGene": T.tex_escape(prep.idxmin()),
                  f"Weight{tag}NbelowHalfRep": str(int((prep < 0.5).sum())),
                  f"Weight{tag}Identical": f"{100 * np.mean([p == base for p in panels]):.0f}"})
        rows.append(pd.DataFrame({"perturbation": f"+/-{int(a * 100)}%", "gene": genes, "inclusion": incl.values,
                                  "in_published_panel": [g in pub for g in genes]}))
        if a == 0.2:
            json.dump({"jaccard_vs_published": jpub.tolist()}, open(os.path.join(out, "tables", "R4_weight_jaccards.json"), "w"))
    pd.concat(rows).to_csv(os.path.join(out, "tables", "R4_weight_sensitivity_inclusion.csv"), index=False)
    save(out, "weights", m)


if __name__ == "__main__":
    {"repro": repro, "libsize": libsize, "r0extra": r0extra, "nestedextra": nestedextra,
     "subsvumc": subsvumc, "r5extra": r5extra, "weights": weights}[sys.argv[2]](sys.argv[1])
