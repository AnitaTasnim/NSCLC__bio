"""Write the LaTeX macros and tables that the revised manuscript reads its audit numbers from.

Every number of the pre-registered audit (E1-E4, E8-E19) that the paper quotes is written here from the saved
result files; nothing is typed by hand. Output: runs/revision/paper/revision_macros.tex and tab_*.tex, which are
copied to the manuscript's results/ folder.

Run:  python -m pmtrd.revision.paper_macros
"""
from __future__ import annotations

import collections
import json
import os
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .common import path

OUT = path("runs", "revision", "paper")
M: dict[str, str] = {}


def r(x, d=2):
    """Round half up on the decimal representation, so that 0.725 -> 0.73."""
    q = Decimal(1).scaleb(-d)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def s(x, d=2):
    """Signed, with a proper minus sign."""
    v = r(x, d)
    if v.startswith("-"):
        return "$-$" + v[1:]
    return "+" + v


def u(x, d=2):
    """Unsigned, with a proper minus sign."""
    v = r(x, d)
    return "$-$" + v[1:] if v.startswith("-") else v


def pts(x, d=2):
    return s(100 * x, d)


def ci_pts(lo, hi, d=2):
    return f"{pts(lo, d)} to {pts(hi, d)}"


def pv(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 0.0001:
        return "$<$0.0001"
    if p < 0.001:
        return r(p, 4)
    if p < 0.1:
        return r(p, 3)
    return r(p, 2)


def pct(x, d=1):
    return r(100 * x, d)


def put(k, v):
    assert k not in M, k
    M[k] = v


def e8():
    d = pd.read_csv(path("runs", "revision", "E8", "E8.csv")).set_index("arm")
    for key, arm in [("Mrmr", "mrmr@27/xgb"), ("MrmrSvm", "mrmr@own/svm"), ("Stab", "stability@27/xgb"),
                     ("LOne", "l1_topk@27/xgb"), ("Mw", "mw_topk@27/xgb")]:
        x = d.loc[arm]
        put(f"AuBm{key}AUC", r(x.auc, 4))
        put(f"AuBm{key}Diff", pts(-x.diff_A4xgb_minus_arm))
        put(f"AuBm{key}CI", ci_pts(-x.diff_hi, -x.diff_lo))
        put(f"AuBm{key}P", pv(x.p_holm))
    put("AuBmAFourAUC", r(d.loc["A4@own/xgb"].auc, 4))
    put("AuBmAFourSvmAUC", r(d.loc["A4@own/svm"].auc, 4))
    put("AuBmMwPsi", r(d.loc["mw_topk@27/xgb"].replicate_psi, 3))


def e9():
    t = pd.read_csv(path("runs", "bagging", "E9_tests.csv"))
    f = pd.read_csv(path("runs", "bagging", "E9_folds.csv"))
    names = {"GSE89843": "Disc", "GSE68086": "Pan", "Singh": "Singh", "Gordon": "Gordon"}
    for ds, key in names.items():
        for q, qk in [("gain10", "GainTen"), ("gain25", "GainTwentyFive"), ("bag10_minus_frozen", "BagTenFrozen"),
                      ("bag25_minus_frozen", "BagFrozen")]:
            x = t[(t.dataset == ds) & (t.quantity == q)].iloc[0]
            put(f"AuBg{qk}{key}", pts(x["mean"]))
            put(f"AuBg{qk}{key}CI", ci_pts(x.lo, x.hi))
            put(f"AuBg{qk}{key}P", pv(x.p))
        put(f"AuBgEnsDis{key}", pct(f[f.dataset == ds].ensemble_disagreement_10v10.mean()))
    dg = t.pivot(index="dataset", columns="quantity", values="mean")
    inc = (dg.gain25 - dg.gain10)
    put("AuBgBTwentyFiveMin", pts(inc.min()))
    put("AuBgBTwentyFiveMax", pts(inc.max()))
    c = pd.read_csv(path("runs", "bagging", "E9_calibration.csv"))
    put("AuBgSlopeBagMin", r(min(c.bag10_cal_slope.min(), c.bag25_cal_slope.min())))
    put("AuBgSlopeBagMax", r(max(c.bag10_cal_slope.max(), c.bag25_cal_slope.max())))
    put("AuBgSlopeSingleMin", r(c.single_cal_slope.min()))
    put("AuBgSlopeSingleMax", r(c.single_cal_slope.max()))
    x = c[(c.dataset == "GSE89843") & (c.partition == 42)].iloc[0]
    put("AuBgBrierBagDisc", r(x.bag25_brier, 3))
    put("AuBgBrierSingleDisc", r(x.single_brier, 3))
    put("AuBgBrierFrozenDisc", r(x.frozen_brier, 3))
    p = pd.read_csv(path("runs", "bagging", "E9_posthoc.csv"))
    pp = p[p.kind == "pooled_published_definition"].set_index(["partition", "B"])
    for (part, B), key in [((42, 10), "PaperTen"), ((43, 10), "NewTen"), ((42, 25), "PaperTwentyFive"),
                           ((43, 25), "NewTwentyFive")]:
        put(f"AuBgPooled{key}AUC", r(pp.loc[(part, B)].auc_bag_pooled, 4))
        put(f"AuBgPooled{key}Gain", pts(pp.loc[(part, B)].gain_pooled))
    ph = p[p.kind != "pooled_published_definition"].set_index(["B", "comparator"])
    for B, bk in [(10, "Ten"), (25, "TwentyFive")]:
        for comp, ck in [("mrmr@27/xgb", "Mrmr"), ("mrmr@own/svm", "MrmrSvm"), ("A4@own/xgb", "Frozen")]:
            x = ph.loc[(B, comp)]
            put(f"AuBgPh{bk}{ck}Diff", pts(x["diff"]))
            put(f"AuBgPh{bk}{ck}CI", ci_pts(x.diff_lo, x.diff_hi))
            put(f"AuBgPh{bk}{ck}P", pv(x.p))
        put(f"AuBgPh{bk}AUC", r(ph.loc[(B, "mrmr@27/xgb")].auc_bag_pooled, 4))
    x = pd.read_csv(path("runs", "bagging", "E9_external.csv"))
    e, lk = x.iloc[0], x.iloc[1]
    put("AuExEnsAUC", r(e.auc, 3))
    put("AuExEnsCI", f"{r(e.auc_lo, 3)}--{r(e.auc_hi, 3)}")
    put("AuExMemberMean", r(e.mean_member_auc, 3))
    put("AuExMemberMin", r(e.member_auc_min, 3))
    put("AuExMemberMax", r(e.member_auc_max, 3))
    put("AuExGain", pts(e.gain))
    put("AuExGainCI", ci_pts(e.gain_lo, e.gain_hi))
    put("AuExDiff", pts(e.diff_vs_locked))
    put("AuExDiffCI", ci_pts(e.diff_lo, e.diff_hi))
    put("AuExDiffP", pv(e.diff_p))
    put("AuExIcpt", r(e.cal_intercept))
    put("AuExSlope", r(e.cal_slope))
    put("AuExBrier", r(e.brier, 3))
    put("AuExLockedBrier", r(lk.brier, 3))
    put("AuExLockedIcpt", r(lk.cal_intercept))
    put("AuExLockedSlope", r(lk.cal_slope))
    put("AuExGenes", str(int(e.genes_per_member_median)))
    put("AuExDropped", str(int(e.dropped_per_member_median)))


def e10():
    d = pd.read_csv(path("runs", "external", "E10.csv")).set_index("model")
    for m, k in [("a1_locked27", "Locked"), ("a2_locked27_median_imputed", "Imputed"),
                 ("b_retrained18_anyread", "Eighteen"), ("b2_retrained_measurable10pct", "Sixteen"),
                 ("c_A4_measurable", "Measurable")]:
        x = d.loc[m]
        put(f"AuTr{k}AUC", r(x.auc, 3))
        put(f"AuTr{k}CI", f"{r(x.auc_lo, 3)}--{r(x.auc_hi, 3)}")
        put(f"AuTr{k}Genes", str(int(x.n_genes)))
        if m != "a1_locked27":
            put(f"AuTr{k}Diff", pts(x.diff_vs_a1))
            put(f"AuTr{k}P", pv(x.diff_p))
        if not np.isnan(x.null_p):
            put(f"AuTr{k}NullP", pv(x.null_p))
    put("AuTrNullMedian", r(d.null_median.dropna().iloc[0], 3))
    q = pd.read_csv(path("runs", "external", "E10_published_null_posthoc.csv")).set_index("model")
    put("AuTrPubNullMedian", r(q.null_median.iloc[0], 3))
    put("AuTrPubNullLockedP", pv(q.loc["a1_locked27"].empirical_p))
    put("AuTrPubNullEighteenP", pv(q.loc["b_retrained18_anyread"].empirical_p))
    put("AuTrPubNullMeasurableP", pv(q.loc["c_A4_measurable"].empirical_p))


def e11():
    d = pd.read_csv(path("runs", "revision", "E11", "E11.csv")).set_index("scale")
    for sc, k in [("cpm", "Cpm"), ("tmm", "Tmm")]:
        x = d.loc[sc]
        put(f"AuNm{k}AUC", r(x.nested_auc, 4))
        put(f"AuNm{k}Diff", pts(x.diff_vs_counts))
        put(f"AuNm{k}CI", ci_pts(x.diff_lo, x.diff_hi))
        put(f"AuNm{k}P", pv(x.p_vs_counts))
        put(f"AuNm{k}Psi", r(x.psi, 3))
        put(f"AuNm{k}Phi", r(x.phi, 3))
        put(f"AuNm{k}DiffAbs", r(abs(100 * x.diff_vs_counts), 1))


def e15():
    d = pd.read_csv(path("runs", "seeds", "E15.csv"))
    p = pd.read_csv(path("runs", "seeds", "E15_panels.csv"))
    full = p[(p.kind == "full") & (p.env == "A")]
    s0 = full[full.seed == 0].iloc[0]
    rest = full[full.seed > 0]
    put("AuSdSeeds", str(len(rest)))
    put("AuSdZeroJ", r(s0.jaccard_P27, 3))
    put("AuSdJMean", r(rest.jaccard_P27.mean(), 3))
    put("AuSdJMin", r(rest.jaccard_P27.min(), 3))
    put("AuSdJMax", r(rest.jaccard_P27.max(), 3))
    put("AuSdSizeMin", str(int(rest.n_genes.min())))
    put("AuSdSizeMax", str(int(rest.n_genes.max())))
    bs = d[(d.kind == "between_seeds") & (d.env == "A")].iloc[0]
    put("AuSdPairJ", r(bs.pairwise_jaccard, 3))
    put("AuSdPhi", r(bs.phi, 3))
    ident = d[d.kind == "A_vs_B_same_seed"].iloc[0]
    put("AuSdIdentical", f"{int(ident.identical)} of {int(ident.n)}")
    c = collections.Counter(g for gs in rest.genes for g in gs.split(";"))
    n = len(rest)
    put("AuSdAllSeeds", str(sum(v == n for v in c.values())))
    put("AuSdHalfSeeds", str(sum(v >= n / 2 for v in c.values())))
    put("AuSdAnySeed", str(len(c)))
    ne = d[d.kind == "nested"].sort_values("seed")
    put("AuSdNestedMin", r(ne.auc.min(), 3))
    put("AuSdNestedMax", r(ne.auc.max(), 3))
    put("AuSdNestedN", str(len(ne)))
    core = ["P2RY10", "C1QA", "DDAH1", "HSD17B3", "PCSK5", "TNFRSF17", "USP27X"]
    put("AuSdCoreAll", str(sum(c[g] == n for g in core)))


def e16():
    d = pd.read_csv(path("runs", "export", "E16.csv")).set_index("predictor")
    for k, key in [("psi", "Psi"), ("V_R", "VR"), ("phi", "Phi"), ("phi_C", "PhiC")]:
        x = d.loc[k]
        put(f"AuPg{key}Rho", u(x.mean_within_rho))
        put(f"AuPg{key}Loso", r(x.loso_mae_ratio))
        put(f"AuPg{key}PermP", pv(x.perm_p))


def e17():
    d = pd.read_csv(path("results", "pathway", "E17.csv"))
    k = d[d.comparison.str.startswith("chance-corrected")].iloc[0]
    put("AuPwAFourK", r(k.gene_median, 3))
    put("AuPwSetK", r(k.set_median, 3))
    put("AuPwKP", pv(k.p))
    rr = d[d.comparison.str.contains("m=27")].iloc[0]
    put("AuPwRepGenes", pct(rr.gene_median, 0))
    put("AuPwRepSets", pct(rr.set_median, 0))
    t = pd.read_csv(path("results", "pathway", "E17_posthoc_toplists.csv"))
    put("AuPwTopGeneK", r(t.gene_top27_kuncheva.median(), 3))
    put("AuPwTopSetK", r(t.set_top27_kuncheva.median(), 3))
    put("AuPwTopP", pv(wilcoxon(t.set_top27_kuncheva, t.gene_top27_kuncheva).pvalue))
    put("AuPwSplits", str(len(t)))
    put("AuPwTopGeneJ", r(t.gene_top27_jaccard.median(), 2))
    put("AuPwTopSetJ", r(t.set_top27_jaccard.median(), 2))


def e18():
    d = pd.read_csv(path("runs", "export", "E18.csv"))
    g = d[d.dataset == "GSE89843"].set_index("method")
    put("AuCsAFourPhiC", r(g.loc["six_stage", "phi_C_0.5"], 2))
    put("AuCsAFourPhiCCI", f"{r(g.loc['six_stage', 'phi_C_0.5_lo'], 2)}--{r(g.loc['six_stage', 'phi_C_0.5_hi'], 2)}")
    put("AuCsAFourPhi", r(g.loc["six_stage", "phi_identity"], 2))
    put("AuCsAFourJ", r(g.loc["six_stage", "pairwise_jaccard"], 3))
    put("AuCsMwPhiC", r(g.loc["mw_topk", "phi_C_0.5"], 2))
    put("AuCsRandomMatched", r(g.loc["random", "matched_overlap_0.5"], 2))
    put("AuCsAFourMatched", r(g.loc["six_stage", "matched_overlap_0.5"], 2))


def e19():
    d = pd.read_csv(path("runs", "export", "E19.csv"))
    g = d[d.dataset == "GSE89843"].set_index("method")
    a = g.loc["six_stage"]
    put("AuAmD", pct(a.D))
    put("AuAmDCI", f"{pct(a.D_lo)}--{pct(a.D_hi)}")
    put("AuAmATwo", pct(a.A2))
    put("AuAmATen", pct(a.A10))
    put("AuAmFixedD", pct(g.loc["fixed", "D"]))
    put("AuAmRandomD", pct(g.loc["random", "D"]))
    put("AuAmMwD", pct(g.loc["mw_topk", "D"]))
    put("AuAmEnsFive", pct(a.ensemble_disagreement))
    six = d[d.method == "six_stage"].set_index("dataset")
    fix = d[d.method == "fixed"].set_index("dataset")
    for ds, key in [("GSE68086", "Pan"), ("Singh", "Singh"), ("Gordon", "Gordon")]:
        put(f"AuAmD{key}", pct(six.loc[ds, "D"]))
        put(f"AuAmFixedD{key}", pct(fix.loc[ds, "D"]))
    put("AuAmDMin", pct(six.D.min(), 0))
    put("AuAmDMax", pct(six.D.max(), 0))
    put("AuAmFixedMin", pct(fix.D.min(), 0))
    put("AuAmFixedMax", pct(fix.D.max(), 0))


def e1_4():
    v = json.load(open(path("runs", "crossed_sim", "v1", "summary", "verdict.json")))
    put("AuSmCells", str(v["cells"]))
    put("AuSmPassed", str(v["passed"]))
    s_ = pd.read_csv(path("runs", "crossed_sim", "v1", "summary", "summary.csv"))
    e = s_[s_.stat == "eq3"]
    ten = e[e.R == 10]
    put("AuSmCovMin", pct(ten.coverage_patients.min(), 0))
    put("AuSmCovMax", pct(ten.coverage_patients.max(), 0))
    put("AuSmCovMedian", pct(ten.coverage_patients.median(), 0))
    put("AuSmPigMin", pct(ten.coverage_pigeonhole.min(), 0))
    put("AuSmPigMax", pct(ten.coverage_pigeonhole.max(), 0))
    put("AuSmBiasMax", r(e.bias.abs().max(), 3))
    put("AuSmQgMax", r(e.eq3_minus_Qgamma_truth.abs().max(), 2))


def multiple_testing():
    d = pd.read_csv(path("runs", "revision", "multiple_testing.csv"))
    put("AuMtN", str(int(d.n_tests_all.iloc[0])))
    def one(fam, test):
        x = d[d.family.str.startswith(fam) & (d.test == test)].iloc[0]
        return x
    for key, fam, test in [("Mrmr", "E8", "mrmr@27/xgb"), ("MrmrSvm", "E8", "mrmr@own/svm"),
                           ("Stab", "E8", "stability@27/xgb"), ("Tmm", "E11", "tmm"),
                           ("Ext", "E9 external", "ensemble25 - locked"),
                           ("Sixteen", "E10 paired bootstrap incl", "b2_retrained_measurable10pct"),
                           ("SinghGain", "E9 corrected", "Singh:gain10")]:
        x = one(fam, test)
        put(f"AuMt{key}All", pv(x.p_holm_all_tests))
        put(f"AuMt{key}Fam", pv(x.p_holm_within_family))
    lost = d.drop_duplicates(["test", "p_raw"])
    lost = lost[(lost["significant_raw_0.05"]) & (~lost["significant_holm_all_0.05"])]
    put("AuMtLostAll", str(len(lost)))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for f in (e1_4, e8, e9, e10, e11, e15, e16, e17, e18, e19, multiple_testing):
        f()
    with open(os.path.join(OUT, "revision_macros.tex"), "w") as fh:
        fh.write("% Auto-generated by python -m pmtrd.revision.paper_macros - do not edit by hand.\n")
        for k in sorted(M):
            fh.write(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{M[k]}}}\n")
    json.dump(M, open(os.path.join(OUT, "revision_macros.json"), "w"), indent=1, sort_keys=True)
    print(len(M), "macros ->", os.path.join(OUT, "revision_macros.tex"))


if __name__ == "__main__":
    main()
