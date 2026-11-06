#!/usr/bin/env python3
"""Compare this analysis's estimates with the authors' second, independently written analysis notebook
(TEP_Reviewer_Revision_Run.ipynb, PAPER mode; results folder PAPER_<hash>/).

    python crosscheck_independent.py <out_dir> <independent_results_dir>

Writes <out>/tables/S_crosscheck_independent.{csv,tex} and <out>/macros/crosscheck.json.
The second notebook used a different code base, a four-score variant of the selector (no STRING and G106 terms,
B = 100 bootstrap resamples per outer fold), log2(x + 1) inputs, and 100-tree models for its random-panel controls."""
import glob
import json
import os
import sys

import pandas as pd


def main(out, ind):
    M = {}
    for f in glob.glob(os.path.join(out, "macros", "*.json")):
        M.update(json.load(open(f)))
    perf = pd.read_csv(os.path.join(ind, "all_performance_results.csv"))
    perf["key"] = perf["model"] + "|" + perf["analysis_folder"]

    def theirs(model, folder, ci=True):
        r = perf[(perf.model == model) & (perf.analysis_folder == folder)].iloc[0]
        return f"{r.auc:.4f} ({r.auc_ci_low:.4f}--{r.auc_ci_high:.4f})" if ci else f"{r.auc:.4f}"

    def mine(a, c=None):
        if a not in M:
            return "pending"
        return f"{M[a]} ({M[c]})" if c and c in M else M[a]

    pdiff = pd.read_csv(os.path.join(ind, "nested_paired_differences.csv")).set_index("comparison")
    sh = pdiff.loc["sheng_style minus nested_baseline"]
    r1 = json.load(open(os.path.join(ind, "R1_site_effects", "interpretation.json")))
    r9, r9m = {}, {}
    for kind, nm in [("unmatched", "Rand"), ("matched", "Match")]:
        s = pd.read_csv(os.path.join(ind, f"R9_random_{kind}", "summary.csv")).iloc[0]
        r9[kind] = f"{int(s.exceedances)}/{int(s.n_panels):,} (mean {s.random_mean_auc:.3f})".replace(",", "{,}")
        r9m.update({f"Xc{nm}ExceedTheirs": str(int(s.exceedances)), f"Xc{nm}NTheirs": f"{int(s.n_panels):,}".replace(",", "{,}"),
                    f"Xc{nm}MeanTheirs": f"{s.random_mean_auc:.3f}"})
    ora = pd.read_csv(os.path.join(ind, "R7_enrichment", "all_tested_terms.csv"), low_memory=False)
    qcol = "q_global" if "q_global" in ora.columns else [c for c in ora.columns if c.startswith("q")][0]
    n_ora = int((pd.to_numeric(ora[qcol], errors="coerce") < 0.05).sum())
    hh = M.get("HhDelta", "").replace("$-$", "")
    rows = [
        ("Fixed 27-gene panel, five-fold CV (genes selected on all samples)",
         mine("CoreFixedAUC", "CoreFixedCI"), theirs("panel27", "R2_R5_fixed_panel")),
        ("22-gene subset (without LCE3B, USP27X, SPHAR, PRSS42, KRTAP22-1)",
         mine("PanelSharedAUC", "PanelSharedCI"), theirs("legacy22", "R2_R5_fixed_panel")),
        ("The five omitted genes alone",
         mine("PanelMissingAUC", "PanelMissingCI"), theirs("omitted5_only", "R2_R5_fixed_panel")),
        ("Selection repeated inside every training fold (nested CV)",
         mine("NestedAUC", "NestedCI"), theirs("all_baseline_four_score", "R1_R5_nested_all_baseline")),
        ("Selection and nested CV within Vumc only",
         mine("VumcNestedAUC", "VumcNestedCI"), theirs("vumc_baseline_four_score", "R1_R5_nested_vumc_baseline")),
        ("mRMR--IFS--SVM comparator, same outer folds",
         mine("HhShengAUC", "HhShengCI"), theirs("Sheng_style_nested_mRMR_IFS_SVM", "R8_sheng_style")),
        ("Comparator minus selection pipeline (percentage points)",
         (f"+{hh} ({M['HhDeltaCI'].replace('$-$', '').split(' to ')[1]} to {M['HhDeltaCI'].replace('$-$', '').split(' to ')[0]})"
          if hh else "pending"), f"+{100 * sh.delta_auc:.2f} ({100 * sh.delta_ci_low:.2f} to {100 * sh.delta_ci_high:.2f})"),
        ("Panel genes with a diagnosis effect after adjustment for site",
         f"{M.get('GeneAdjSig', '?')}/27", f"{r1['disease_q_below_05']}/27"),
        ("GSE68086, 22 genes, without library-size normalisation",
         mine("ExtOrigNoAlignAUC", "ExtOrigNoAlignCI"), theirs("legacy22", "R2_R5_external")),
        ("Random 27-gene panels reaching the curated panel's AUC",
         f"{M['RandB']}/{M['RandN']} (mean {M['RandMean'][:5]})" if "RandB" in M else "pending", r9["unmatched"]),
        ("Expression/variance-matched panels reaching it",
         f"{M['MatchB']}/{M['MatchN']} (mean {M['MatchMean'][:5]})" if "MatchB" in M else "pending", r9["matched"]),
        ("Gene sets over-represented among the panel genes (FDR $<0.05$)",
         M.get("OraNsig", "pending"), str(n_ora)),
    ]
    df = pd.DataFrame(rows, columns=["Estimate", "This analysis", "Independent notebook"])
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    df.to_csv(os.path.join(out, "tables", "S_crosscheck_independent.csv"), index=False)
    lines = [r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.42\linewidth}>{\raggedright\arraybackslash}p{0.235\linewidth}>{\raggedright\arraybackslash}p{0.235\linewidth}}", r"\toprule",
             r"Estimate & This analysis & Independent notebook \\", r"\midrule"]
    lines += [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tables", "S_crosscheck_independent.tex"), "w").write("\n".join(lines) + "\n")
    m = {"XcFixedTheirs": theirs("panel27", "R2_R5_fixed_panel", ci=False),
         "XcNestedTheirs": theirs("all_baseline_four_score", "R1_R5_nested_all_baseline", ci=False),
         "XcVumcNestedTheirs": theirs("vumc_baseline_four_score", "R1_R5_nested_vumc_baseline", ci=False),
         "XcShengTheirs": theirs("Sheng_style_nested_mRMR_IFS_SVM", "R8_sheng_style", ci=False),
         "XcShengDeltaTheirs": f"{100 * sh.delta_auc:.2f}", "XcRandTheirs": r9["unmatched"], "XcMatchTheirs": r9["matched"],
         "XcExtNoNormTheirs": theirs("legacy22", "R2_R5_external", ci=False), "XcSiteAdjTheirs": str(r1["disease_q_below_05"]),
         "XcOraTheirs": str(n_ora), "XcShengDeltaOurs": M.get("HhDelta", "").replace("$-$", "").replace("-", "")}
    m.update(r9m)
    json.dump(m, open(os.path.join(out, "macros", "crosscheck.json"), "w"), indent=1)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
