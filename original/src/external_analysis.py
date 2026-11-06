#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
external_analysis.py - all GSE68086 analyses in one place.

  A  reproduction of the ORIGINAL external analysis (22 genes; XGBoost 200 trees, depth 3;
     log2(x+1); each cohort standardised on its own samples) and two variants: the 24-gene
     subset with USP27X and PRSS42 (both present under the Ensembl-75 annotation that GSE68086
     uses) and the 22 genes without per-cohort standardisation
  B  the main classifier (300 trees, depth 4) on log2-CPM of genes shared by both cohorts,
     for several panels, under: S0 no alignment; S3 each cohort standardised on its own samples
     (= location/scale alignment with all external samples); S2 reference-batch ComBat-seq
     (discovery fixed, frozen model); S1a joint ComBat-seq + re-trained model; S5 single-sample
     within-sample ranks
  C  operating characteristics and calibration at a threshold fixed in the discovery cohort
  D  other models in GSE68086: Vumc-derived panel (Vumc-trained), mRMR-IFS-SVM (Sheng et al.)
The full harmonisation audit with add-on calibration, case-mix and random-panel controls is
tep_analysis.r3_harmonisation_audit (run with run_steps.py ... r3).

    python external_analysis.py config_ext.json
"""
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tep_analysis as T  # noqa: E402

warnings.filterwarnings("ignore")

PANEL_24 = ['P2RY10', 'MGAT5', 'FCER1A', 'HSD17B3', 'PCSK5', 'JOSD1', 'USP27X', 'C1QA', 'DUOXA1',
            'HOXC12', 'NTNG2', 'ITGB3BP', 'LY6G6D', 'PTMS', 'TMEM39B', 'OR52B6', 'DDAH1', 'PRSS42',
            'TM4SF1', 'MFN2', 'GJB2', 'CXCL14', 'TYMS', 'TNFRSF17']
ORIG22 = [g for g in PANEL_24 if g not in ("USP27X", "PRSS42")]


def orig_model():
    return XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=3, subsample=0.8,
                         colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=2, verbosity=0)


def boot(ye, p, B=2000, seed=42):
    a, lo, hi = T.bootstrap_auc_ci(ye, p, B, seed)
    return a, lo, hi


def write_external_table(out, res, n_det):
    """LaTeX table: one row per panel, one column per alignment setting, AUC over its 95% CI."""
    b = res[res.block == "B main classifier"]
    order_c = ["S0 no alignment", "S3 each cohort standardised", "S2 reference-batch ComBat-seq",
               "S1a joint ComBat-seq, re-trained", "S5 within-sample ranks"]
    labels = {"P27": "27-gene panel", "P24qc": "QC-filtered 24-gene panel", "P22orig": "22-gene subset",
              "Vumc-derived": "Vumc-derived panel (Vumc-trained)"}
    trows = []
    for key in ["P27", "P24qc", "P22orig", "detected", "Vumc-derived"]:
        sub = b[b.panel.str.startswith(key) | b.panel.str.contains(key)]
        if sub.empty:
            continue
        pname = sub.panel.iloc[0]
        nm = pname.split(" (")[0]
        ng = pname.split("(")[1].split()[0]
        lab = labels.get(nm, f"{n_det} genes with reads in GSE68086")
        cells = []
        for c_ in order_c:
            r = sub[sub.setting == c_]
            cells.append(T.tex_auc_ci(r.auc.iloc[0], r.lo.iloc[0], r.hi.iloc[0]) if len(r) else "--")
        two = {"27-gene panel": ["27-gene panel"], "QC-filtered 24-gene panel": ["QC-filtered", "24-gene panel"],
               "22-gene subset": ["22-gene subset"],
               "Vumc-derived panel (Vumc-trained)": ["Vumc-derived panel", "(Vumc-trained)"]}
        trows.append([T.tex_stack(*two.get(lab, [f"{n_det} genes with", "reads in GSE68086"]), align="l"), ng] + cells)
    T.write_tex_table(os.path.join(out, "tables", "R2_external_all_settings.tex"), trows,
                      ["Panel", "Genes", T.tex_stack("S0", "none"), T.tex_stack("S3", "per-cohort"),
                       T.tex_stack("S2", "reference", "ComBat-seq"), T.tex_stack("S1a", "joint", "ComBat-seq"),
                       T.tex_stack("S5", "ranks")], "lcccccc", "\\footnotesize")


def main(cfg_path):
    t0 = time.time()
    cfg = T.load_config(cfg_path)
    d = T.load_data(cfg)
    run = T.Run(cfg, d)
    out = cfg["out_dir"]
    c = pickle.load(open(os.path.join(out, "cache", "run_cache.pkl"), "rb"))
    y, ye = d.y.values, d.y_ext.values
    P27 = run.panel_in(d.X)
    M, rows, aucs = {}, [], []

    def reg(level, name, p, method="bootstrap"):
        a, lo, hi = boot(ye, p)
        aucs.append(dict(level=level, analysis=name, auc=a, ci_lo=lo, ci_hi=hi, ci_method=method,
                         n=len(ye), fold_sd=None, note=""))
        return a, lo, hi

    # ------------------------------------------------------------------ A: development configuration
    undetected = [g for g in P27 if g in d.X_ext.columns and (d.X_ext[g] > 0).sum() == 0]
    M["ExtUndetected"] = ", ".join(undetected) if undetected else "none"
    M["ExtUndetectedN"] = str(len(undetected))
    Xtr = np.log2(d.X[ORIG22].values + 1); sc = StandardScaler().fit(Xtr)
    m = orig_model().fit(sc.transform(Xtr), y)
    Xte = np.log2(d.X_ext[ORIG22].values + 1)
    pA1 = m.predict_proba(StandardScaler().fit_transform(Xte))[:, 1]
    pA3 = m.predict_proba(sc.transform(Xte))[:, 1]
    a1 = reg("Independent platelet cohort", "GSE68086 development configuration (22 genes, each cohort standardised on itself)", pA1)
    a3 = reg("Independent platelet cohort", "GSE68086 development model, 22 genes, discovery scaling (no alignment)", pA3)
    G24 = [g for g in PANEL_24]
    Xtr24 = np.log2(d.X[G24].values + 1); sc24 = StandardScaler().fit(Xtr24)
    m24 = orig_model().fit(sc24.transform(Xtr24), y)
    pA2 = m24.predict_proba(StandardScaler().fit_transform(np.log2(d.X_ext[G24].values + 1)))[:, 1]
    a2 = reg("Independent platelet cohort", "GSE68086 development model, 24 genes incl. USP27X and PRSS42, each cohort standardised", pA2)
    acc_half = float(((pA1 >= .5).astype(int) == ye).mean())
    M.update(ExtOrigReproAUC=T.f4(a1[0]), ExtOrigReproCI=T.ci(a1[1], a1[2]), ExtOrigReproAcc=T.f3(acc_half),
             ExtOrigNoAlignAUC=T.f4(a3[0]), ExtOrigNoAlignCI=T.ci(a3[1], a3[2]),
             ExtOrigTwentyFourAUC=T.f4(a2[0]), ExtOrigTwentyFourCI=T.ci(a2[1], a2[2]))
    rows += [dict(block="A development model", panel="22 (development)", setting="each cohort standardised", auc=a1[0], lo=a1[1], hi=a1[2]),
             dict(block="A development model", panel="22 (development)", setting="discovery scaling, no alignment", auc=a3[0], lo=a3[1], hi=a3[2]),
             dict(block="A development model", panel="24 (+USP27X, PRSS42)", setting="each cohort standardised", auc=a2[0], lo=a2[1], hi=a2[2])]
    print("A:", rows[-3:], flush=True)

    # ------------------------------------------------------------------ B: main classifier, log-CPM
    # counts on the genes shared by both cohorts; panel genes are kept even if one cohort has no reads
    Craw, Ceraw = d.counts, d.counts_ext
    shared = sorted(set(Craw.columns) & set(Ceraw.columns))
    C = Craw[shared].clip(lower=0).round().astype(np.int64)
    Ce = Ceraw[shared].clip(lower=0).round().astype(np.int64)
    QC24 = [g for g in P27 if g not in ("KRTAP22-1", "OR52B6", "LCE3B")]
    DET = [g for g in P27 if g not in undetected]
    panels = {"P27": P27, "P24qc": QC24, "P22orig": ORIG22, f"P{len(DET)}detected": DET}
    if c.get("vumc_panel"):
        panels["Vumc-derived"] = [g for g in c["vumc_panel"] if g in shared]
    allp = sorted(set().union(*panels.values()))
    ok = ((C.sum(0) > 0) & (Ce.sum(0) > 0)) | C.columns.isin(allp)
    C, Ce = C.loc[:, ok], Ce.loc[:, ok]
    M["ExtSharedGenes"] = f"{C.shape[1]:,}".replace(",", "{,}")
    Ld, Le = T.logcpm(C), T.logcpm(Ce)
    # ComBat-seq once (panel genes + 2,000 most expressed + rest), joint and reference;
    # genes without reads in one cohort cannot be adjusted and are carried over unchanged
    top = Ld.mean(0).sort_values(ascending=False).index[:2000]
    zero_any = [g for g in C.columns if C[g].sum() == 0 or Ce[g].sum() == 0]
    keep = sorted((set(top) | set(allp)) - set(zero_any))
    Cr, Cer = T._reduce_with_rest(C.drop(columns=zero_any), keep), T._reduce_with_rest(Ce.drop(columns=zero_any), keep)
    tc = time.time()
    Jd, Je = T._combat_seq(Cr, Cer, ref_batch=None)
    _, Re_ = T._combat_seq(Cr, Cer, ref_batch=0)
    print(f"ComBat-seq done ({time.time() - tc:.0f}s)", flush=True)
    for z in [g for g in zero_any if g in allp]:
        Jd[z], Je[z], Re_[z] = C[z].values, Ce[z].values, Ce[z].values
    # library sizes for log-CPM after ComBat: use the adjusted retained genes + REST (same universe as before)
    LJd, LJe, LRe = T.logcpm(Jd), T.logcpm(Je), T.logcpm(Re_)
    v = run.vumc
    expressed = None
    fd = ((2 ** Ld - 1) > 1).mean(0); fe = ((2 ** Le - 1) > 1).mean(0)
    expressed = fd.index[(fd >= .2) & (fe >= .2)].tolist()
    Rd, Re = T.within_sample_ranks(Ld, expressed), T.within_sample_ranks(Le, expressed)
    for name, G in panels.items():
        G = [g for g in G if g in Ld.columns]
        trm = v if name == "Vumc-derived" else np.ones(len(y), bool)
        mdl = run.xgb().fit(Ld.loc[trm, G].values, y[trm])
        p0 = mdl.predict_proba(Le[G].values)[:, 1]
        p3 = mdl.predict_proba(T._loc_scale(Ld.loc[trm, G], Le[G], Le[G]).values)[:, 1]
        p2 = mdl.predict_proba(LRe[G].values)[:, 1]
        pj = T.fit_predict(LJd.loc[trm, G].values, y[trm], LJe[G].values, run.xgb)
        Gr = [g for g in G if g in expressed]
        pr = T.fit_predict(Rd.loc[trm, Gr].values, y[trm], Re[Gr].values, run.xgb) if len(Gr) >= 3 else None
        for sname, p in [("S0 no alignment", p0), ("S3 each cohort standardised", p3),
                         ("S2 reference-batch ComBat-seq", p2), ("S1a joint ComBat-seq, re-trained", pj),
                         ("S5 within-sample ranks", pr)]:
            if p is None:
                continue
            a, lo, hi = boot(ye, p)
            rows.append(dict(block="B main classifier", panel=f"{name} ({len(G)} genes)", setting=sname, auc=a, lo=lo, hi=hi))
            if name in ("P27", "P24qc") or sname.startswith(("S0", "S2")):
                aucs.append(dict(level="Independent platelet cohort", analysis=f"GSE68086 {name}: {sname}",
                                 auc=a, ci_lo=lo, ci_hi=hi, ci_method="bootstrap", n=len(ye), fold_sd=None, note=""))
        key = {"P27": "TwentySeven", "P24qc": "QcTwentyFour", "P22orig": "TwentyTwo",
               f"P{len(DET)}detected": "Detected", "Vumc-derived": "Vumc"}[name]
        for sname, p in [("None", p0), ("Std", p3), ("Ref", p2), ("Joint", pj), ("Rank", pr)]:
            if p is None:
                continue
            a, lo, hi = boot(ye, p)
            M[f"Ext{key}{sname}AUC"] = T.f4(a); M[f"Ext{key}{sname}CI"] = T.ci(lo, hi)
        M[f"Ext{key}N"] = str(len(G))
        if name == "P27":
            p_main = {"S0": p0, "S3": p3, "S2": p2}
            gene_main = G
        print(name, "done", flush=True)
    M["ExtDetectedN"] = str(len(DET))
    # ------------------------------------------------------------------ attribution of the external gain
    #  original classifier configuration on log2-CPM, and main classifier on un-normalised log2(x+1)
    M["ExtAttrOrigCfgCpm"] = T.f4(roc_auc_score(ye, orig_model().fit(Ld[ORIG22].values, y).predict_proba(Le[ORIG22].values)[:, 1]))
    M["ExtAttrRevCfgRawTwentySeven"] = T.f4(roc_auc_score(ye, run.xgb().fit(np.log2(d.X[P27].values + 1), y)
                                                          .predict_proba(np.log2(d.X_ext[P27].values + 1))[:, 1]))
    M["ExtAttrRevCfgRawTwentyTwo"] = T.f4(roc_auc_score(ye, run.xgb().fit(np.log2(d.X[ORIG22].values + 1), y)
                                                         .predict_proba(np.log2(d.X_ext[ORIG22].values + 1))[:, 1]))
    M["ExtRankGenes"] = ", ".join([g for g in P27 if g in expressed]); M["ExtRankGenesN"] = str(len([g for g in P27 if g in expressed]))
    # ------------------------------------------------------------------ C: operating characteristics (P27)
    oof, _ = T.cv_oof(Ld[gene_main].values, y, run.folds(), run.xgb)
    thr = T.youden_threshold(y, oof)
    M["ExtThr"] = T.f3(thr)
    ops = []
    for sname, p in p_main.items():
        for rule, t in [("discovery Youden threshold", thr), ("0.5", 0.5)]:
            mm = T.binary_metrics(ye, p, t); mm.update(setting=sname, rule=rule)
            rng = np.random.default_rng(42); bs = []
            for _ in range(1000):
                i = rng.choice(len(ye), len(ye))
                if ye[i].min() == ye[i].max():
                    continue
                q = T.binary_metrics(ye[i], p[i], t); bs.append((q["sensitivity"], q["specificity"]))
            bs = np.array(bs)
            mm["sens_lo"], mm["sens_hi"] = np.quantile(bs[:, 0], [.025, .975])
            mm["spec_lo"], mm["spec_hi"] = np.quantile(bs[:, 1], [.025, .975])
            try:
                import statsmodels.api as sm
                lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
                mm["calib_slope"] = float(sm.GLM(ye, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1])
                mm["calib_intercept"] = float(sm.GLM(ye, np.ones((len(ye), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0])
            except Exception:
                pass
            mm["brier"] = float(np.mean((p - ye) ** 2))
            ops.append(mm)
    ops = pd.DataFrame(ops)
    ops.to_csv(os.path.join(out, "tables", "R2_external_operating_characteristics.csv"), index=False)
    for sname, key in [("S0", "None"), ("S2", "Ref"), ("S3", "Std")]:
        r = ops[(ops.setting == sname) & (ops.rule == "discovery Youden threshold")].iloc[0]
        h = ops[(ops.setting == sname) & (ops.rule == "0.5")].iloc[0]
        M[f"Ext{key}Sens"] = T.f3(r.sensitivity); M[f"Ext{key}Spec"] = T.f3(r.specificity)
        M[f"Ext{key}SensCI"] = T.ci(r.sens_lo, r.sens_hi, 3); M[f"Ext{key}SpecCI"] = T.ci(r.spec_lo, r.spec_hi, 3)
        M[f"Ext{key}PPV"] = T.f3(r.ppv); M[f"Ext{key}NPV"] = T.f3(r.npv); M[f"Ext{key}BalAcc"] = T.f3(r.balanced_accuracy)
        M[f"Ext{key}AccHalf"] = T.f3(h.accuracy)
        if "calib_slope" in r:
            M[f"Ext{key}CalInt"] = T.f2(r.calib_intercept); M[f"Ext{key}CalSlope"] = T.f2(r.calib_slope)
        M[f"Ext{key}Brier"] = T.f3(r.brier)
    # ------------------------------------------------------------------ D: Sheng et al. model in GSE68086
    genes_s, _ = T.sheng_mrmr_ifs_svm(d.X, y, run)
    gs = [g for g in genes_s if g in Ld.columns]
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC
    svm = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)).fit(Ld[gs].values, y)
    ps0 = svm.decision_function(Le[gs].values)
    ps2 = svm.decision_function(LRe[[g for g in gs if g in LRe.columns]].values) if all(g in LRe.columns for g in gs) else None
    a = boot(ye, ps0); M["HhExtShengAUC"] = T.f4(a[0]); M["HhExtShengCI"] = T.ci(a[1], a[2])
    M["HhExtShengNgenes"] = f"{len(gs)}/{len(genes_s)}"
    rows.append(dict(block="D other models", panel=f"mRMR-IFS-SVM ({len(gs)} genes)", setting="S0 no alignment", auc=a[0], lo=a[1], hi=a[2]))
    aucs.append(dict(level="Independent platelet cohort", analysis="GSE68086 mRMR-IFS-SVM (Sheng et al.): S0 no alignment",
                     auc=a[0], ci_lo=a[1], ci_hi=a[2], ci_method="bootstrap", n=len(ye), fold_sd=None, note=""))
    de = T.delong_paired(ye, p_main["S0"], ps0)
    M["HhExtDelta"] = T.pp(de["diff"]); M["HhExtP"] = T.pmac(de["p"])
    if ps2 is not None:
        a = boot(ye, ps2); M["HhExtShengRefAUC"] = T.f4(a[0]); M["HhExtShengRefCI"] = T.ci(a[1], a[2])
        rows.append(dict(block="D other models", panel=f"mRMR-IFS-SVM ({len(gs)} genes)", setting="S2 reference-batch ComBat-seq", auc=a[0], lo=a[1], hi=a[2]))
        de2 = T.delong_paired(ye, p_main["S2"], ps2)
        M["HhExtRefDelta"] = T.pp(de2["diff"]); M["HhExtRefP"] = T.pmac(de2["p"])
    # ------------------------------------------------------------------ write
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "tables", "R2_external_all_settings.csv"), index=False)
    write_external_table(out, res, len(DET))
    os.makedirs(os.path.join(out, "macros"), exist_ok=True); os.makedirs(os.path.join(out, "aucs"), exist_ok=True)
    json.dump(M, open(os.path.join(out, "macros", "external.json"), "w"), indent=1)
    json.dump(aucs, open(os.path.join(out, "aucs", "external.json"), "w"), indent=1, default=float)
    print(res.round(4).to_string(index=False))
    print(json.dumps(M, indent=0))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[1])
