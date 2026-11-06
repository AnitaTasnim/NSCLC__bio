"""Collect the latest completed run of every experiment into manuscript macros, LaTeX tables and figures.

    python -m pmtrd.export <out_dir>

Writes ``<out_dir>/macros_v2.json`` (every number the manuscript quotes from these experiments),
``V2_*.tex`` tables and ``fig_reliability_link.{pdf,png}``. Only completed runs are read; an experiment
without a completed run is skipped and reported.
"""
from __future__ import annotations

import json
import re
import os
import sys

import numpy as np
import pandas as pd

from .config import ROOT

RUNS = os.path.join(ROOT, "runs")
NAMES = {"six_stage": "Six-stage (paper)", "six_stage_four_term": "Six-stage, four-term (paper)",
         "six_stage_five_term": "Six-stage, five-term", "mw_topk": "Mann--Whitney top 27",
         "l1_topk": "L1-logistic top 27", "stability": "Stability selection", "mrmr_ifs_svm": "mRMR--IFS--SVM",
         "random": "Random 27 (Tier 2)", "fixed": "Panel fixed (control arm)"}


def latest(exp):
    f = os.path.join(RUNS, exp, "LATEST")
    if not os.path.exists(f):
        return None
    d = os.path.join(RUNS, exp, open(f).read().strip())
    meta = json.load(open(os.path.join(d, "metadata.json")))
    return d if meta["status"] == "completed" else None


def latest_in(sub, exp):
    f = os.path.join(RUNS, sub, exp, "LATEST")
    if not os.path.exists(f):
        return None
    d = os.path.join(RUNS, sub, exp, open(f).read().strip())
    return d if json.load(open(os.path.join(d, "metadata.json")))["status"] == "completed" else None


def f4(x): return f"{x:.4f}"
def f3(x): return f"{x:.3f}"
def f2(x): return f"{x:.2f}"
def pp(x): return f"{100 * x:+.2f}".replace("-", "$-$")
def pp1(x): return f"{100 * x:.1f}"


def pmac(p):
    if p < 0.001:
        return "$p < 0.001$"
    return f"$p = {p:.3f}$"


def ci(lo, hi, f=f3):
    return f"{f(lo)}--{f(hi)}".replace("--$-$", " to $-$").replace("-$-$", " to $-$")


def signed2(x):
    return f"{x:.2f}".replace("-", "$-$")


def main(out):
    os.makedirs(out, exist_ok=True)
    M, missing = {}, []
    # ---------------------------------------------------------------- column order
    d = latest("column_order")
    if d:
        s = json.load(open(os.path.join(d, "summary.json")))
        M.update(VxCoPanel=f4(s["panel_order_auc"]), VxCoMin=f4(s["min"]), VxCoMax=f4(s["max"]),
                 VxCoSD=f4(s["sd"]), VxCoN=str(s["random_orders"]))
    else:
        missing.append("column_order")
    # ---------------------------------------------------------------- internal calibration
    d = latest("internal_calibration")
    if d:
        s = json.load(open(os.path.join(d, "summary.json")))
        F = pd.read_csv(os.path.join(d, "results", "folds.csv"))
        for v, tag in (("fixed_panel", "Fix"), ("four_term", "Four"), ("five_term", "Five")):
            r = s["pooled"][v]
            M[f"VxIc{tag}Int"] = f"{signed2(r['intercept'])} ({signed2(r['int_ci'][0])} to {signed2(r['int_ci'][1])})"
            M[f"VxIc{tag}Slope"] = f"{f2(r['slope'])} ({f2(r['slope_ci'][0])}--{f2(r['slope_ci'][1])})"
            M[f"VxIc{tag}Brier"] = f"{f3(r['brier'])} ({f3(r['brier_ci'][0])}--{f3(r['brier_ci'][1])})"
            M[f"VxIc{tag}Sens"] = f"{f3(r['sens'])} ({f3(r['sens_lo'])}--{f3(r['sens_hi'])})"
            M[f"VxIc{tag}Spec"] = f"{f3(r['spec'])} ({f3(r['spec_lo'])}--{f3(r['spec_hi'])})"
            g = F[F.version == v]
            M[f"VxIc{tag}SensRange"] = f"{f3(g.sens.min())}--{f3(g.sens.max())}"
            M[f"VxIc{tag}SpecRange"] = f"{f3(g.spec.min())}--{f3(g.spec.max())}"
            M[f"VxIc{tag}SlopeRange"] = f"{f2(g.slope.min())}--{f2(g.slope.max())}"
            M[f"VxIc{tag}IntRange"] = f"{signed2(g.intercept.min())} to {signed2(g.intercept.max())}"
        rows = [r"\begin{tabular}{lccccc}", r"\toprule",
                r"Version & Calibration intercept & Calibration slope & Brier score & Sensitivity & Specificity \\", r"\midrule"]
        for v, lab in (("fixed_panel", "Fixed 27-gene panel"), ("four_term", "Nested, four-term"), ("five_term", "Nested, five-term")):
            t = {"fixed_panel": "Fix", "four_term": "Four", "five_term": "Five"}[v]
            rows.append(f"{lab} & {M[f'VxIc{t}Int']} & {M[f'VxIc{t}Slope']} & {M[f'VxIc{t}Brier']} & {M[f'VxIc{t}Sens']} & {M[f'VxIc{t}Spec']} \\\\")
            rows.append(f"\\quad fold range & {M[f'VxIc{t}IntRange']} & {M[f'VxIc{t}SlopeRange']} & & {M[f'VxIc{t}SensRange']} & {M[f'VxIc{t}SpecRange']} \\\\")
        rows += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(out, "V2_internal_calibration.tex"), "w").write("\n".join(rows) + "\n")
    else:
        missing.append("internal_calibration")
    # ---------------------------------------------------------------- PPI sensitivity
    d = latest("ppi_sensitivity")
    if d:
        s = json.load(open(os.path.join(d, "summary.json")))
        D = pd.read_csv(os.path.join(d, "results", "scenarios.csv"))
        e = s["empirical"]
        M.update(VxPpNone=f4(s["none_auc"]), VxPpDraws=str(int((D.scenario == "empirical").sum())),
                 VxPpEmpMed=f4(e["auc_median"]), VxPpEmpRange=f"{f4(e['auc_min'])}--{f4(e['auc_max'])}",
                 VxPpEmpJMed=f3(e["jaccard_median"]), VxPpEmpJMin=f3(e["jaccard_min"]),
                 VxPpAdvAUC=f4(s["adversarial"]["auc"]), VxPpAdvDelta=pp(s["adversarial"]["delta_vs_none"]),
                 VxPpAdvP=pmac(s["adversarial"]["p"]), VxPpAdvJ=f3(s["adversarial"]["jaccard_vs_five_term"]),
                 VxPpMaxAUC=f4(s["all_max"]["auc"]), VxPpMaxJ=f3(s["all_max"]["jaccard_vs_five_term"]))
    else:
        missing.append("ppi_sensitivity")
    # ---------------------------------------------------------------- composite ablation
    d = latest("composite_ablation")
    if d:
        s = json.load(open(os.path.join(d, "summary.json")))
        D = pd.read_csv(os.path.join(d, "results", "scenarios.csv"))
        M["VxCaDefault"] = f4(s["default_auc"]); M["VxCaDraws"] = str(int((D.scenario == "dirichlet").sum()))
        lab = {"drop_stat": "Stat", "drop_effect": "Effect", "drop_lasso": "Lasso", "drop_boot": "Boot",
               "drop_prior": "Prior", "equal": "Equal"}
        for k, t in lab.items():
            r = s["single_term"][k]
            M[f"VxCa{t}AUC"] = f4(r["auc"]); M[f"VxCa{t}Delta"] = pp(r["delta_vs_default"])
            M[f"VxCa{t}P"] = pmac(r["p"]); M[f"VxCa{t}J"] = f3(r["jaccard_vs_default"])
        dr = s["dirichlet"]
        M.update(VxCaDirMed=f4(dr["auc_median"]), VxCaDirQ=f"{f4(dr['auc_q05'])}--{f4(dr['auc_q95'])}",
                 VxCaDirRange=f"{f4(dr['auc_min'])}--{f4(dr['auc_max'])}", VxCaDirJMed=f3(dr["jaccard_median"]),
                 VxCaDirJLo=f3(dr["jaccard_q05"]))
        dd = D[D.scenario.str.startswith("drop_")]
        M["VxCaJMin"] = f3(dd.jaccard_vs_default.min()); M["VxCaJMax"] = f3(dd.jaccard_vs_default.max())
        M["VxCaMaxAbsDelta"] = f"{100 * dd.delta_vs_default.abs().max():.2f}"; M["VxCaMinP"] = f"{dd.p.min():.3f}"
        nsig = int(((D.scenario.str.startswith("drop_")) & (D.p < 0.05)).sum())
        M["VxCaNsig"] = str(nsig)
        rows = [r"\begin{tabular}{lcccc}", r"\toprule",
                r"Composite & Nested AUC & $\Delta$AUC vs default (pp) & Paired DeLong & Jaccard with default fold panels \\", r"\midrule",
                f"Default weights & {M['VxCaDefault']} & -- & -- & 1.000 \\\\"]
        names = {"Stat": "Without significance term", "Effect": "Without effect-size term", "Lasso": "Without LASSO term",
                 "Boot": "Without bootstrap-stability term", "Prior": "Without feature-importance term", "Equal": "Equal weights"}
        for t, nm in names.items():
            rows.append(f"{nm} & {M[f'VxCa{t}AUC']} & {M[f'VxCa{t}Delta']} & {M[f'VxCa{t}P']} & {M[f'VxCa{t}J']} \\\\")
        rows.append(f"Random weights, {M['VxCaDraws']} draws: median (5th--95th pct.) & {M['VxCaDirMed']} ({M['VxCaDirQ']}) & & & {M['VxCaDirJMed']} \\\\")
        rows += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(out, "V2_composite_ablation.tex"), "w").write("\n".join(rows) + "\n")
    else:
        missing.append("composite_ablation")
    # ---------------------------------------------------------------- tuning
    d = latest("tuning")
    if d:
        s = json.load(open(os.path.join(d, "summary.json")))
        M.update(VxTuAUC=f4(s["tuned_auc"]), VxTuCI=f"{f4(s['tuned_ci'][0])}--{f4(s['tuned_ci'][1])}",
                 VxTuFixed=f4(s["fixed_auc"]), VxTuDelta=pp(s["delta_tuned_minus_fixed"]),
                 VxTuDeltaCI=f"{pp(s['delta_ci'][0])} to {pp(s['delta_ci'][1])}", VxTuP=pmac(s["p"]))
        ch = s["chosen"]
        M["VxTuChosen"] = "; ".join(f"{c['n_estimators']} trees, depth {c['max_depth']}, rate {c['learning_rate']}" for c in ch)
        M["VxTuNdefault"] = str(sum(1 for c in ch if (c["n_estimators"], c["max_depth"], c["learning_rate"]) == (300, 4, 0.03)))
    else:
        missing.append("tuning")
    # ---------------------------------------------------------------- benchmark
    d = latest("benchmark")
    if d:
        B = pd.read_csv(os.path.join(d, "results", "benchmark.csv")); C = pd.read_csv(os.path.join(d, "results", "comparisons.csv"))
        key = {"six_stage_four_term": "Four", "six_stage_five_term": "Five", "mw_topk": "Mw", "l1_topk": "Lone",
               "stability": "Stab", "mrmr_ifs_svm": "Mrmr"}
        for _, r in B.iterrows():
            k = key.get(r.method, "Rand" if str(r.method).startswith("random") else None)
            if k is None:
                continue
            M[f"VxBm{k}AUC"] = f4(r.auc); M[f"VxBm{k}CI"] = f"{f4(r.auc_lo)}--{f4(r.auc_hi)}"
            M[f"VxBm{k}Phi"] = f3(r.phi); M[f"VxBm{k}J"] = f3(r.pairwise_jaccard)
        for _, r in C.iterrows():
            m = r.comparison.split(" - ")[1]; k = key[m]
            M[f"VxBm{k}Delta"] = pp(r.delta); M[f"VxBm{k}P"] = pmac(r.p); M[f"VxBm{k}PHolm"] = pmac(r.p_holm)
        rows = [r"\begin{tabular}{lcccccc}", r"\toprule",
                r"Selector & Nested AUC (95\% CI) & $\Delta$AUC (pp) & Holm $p$ & $\hat\Phi$ (95\% CI) & Jaccard & Genes per fold \\", r"\midrule"]
        for _, r in B.iterrows():
            nm = NAMES.get(r.method, r.method)
            if str(r.method).startswith("random"):
                nm = "Random 27 (Tier 2), median of " + str(r.method).split("of ")[1].rstrip(")")
                rows.append(f"{nm} & {f4(r.auc)} (range {f4(r.auc_lo)}--{f4(r.auc_hi)}) & & & {f3(r.phi)} & {f3(r.pairwise_jaccard)} & 27 \\\\")
                continue
            cr = C[C.comparison.str.endswith(" - " + r.method)]
            dl = "--" if cr.empty else pp(-cr.iloc[0].delta)
            ph = "--" if cr.empty else pmac(cr.iloc[0].p_holm)
            rows.append(f"{nm} & {f4(r.auc)} ({f4(r.auc_lo)}--{f4(r.auc_hi)}) & {dl} & {ph} & {f3(r.phi)} ({f3(r.phi_lo)}--{f3(r.phi_hi)}) & "
                        f"{f3(r.pairwise_jaccard)} & {str(r.sizes).replace(';', ', ')} \\\\")
        rows += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(out, "V2_benchmark.tex"), "w").write("\n".join(rows) + "\n")
    else:
        missing.append("benchmark")
    # ---------------------------------------------------------------- replicates
    d = latest("replicates")
    if d:
        S = pd.read_csv(os.path.join(d, "results", "stability.csv")); A = pd.read_csv(os.path.join(d, "results", "agreement.csv"))
        I = pd.read_csv(os.path.join(d, "results", "identity_share.csv")); U = pd.read_csv(os.path.join(d, "results", "units.csv"))
        T = S.merge(A, on="method").merge(I[["method", "share", "lo", "hi"]], on="method", how="left")
        key = {"six_stage": "Six", "mw_topk": "Mw", "l1_topk": "Lone", "stability": "Stab", "mrmr_ifs_svm": "Mrmr",
               "random": "Rand", "fixed": "Fixed"}
        M["VxRpUnits"] = str(U.groupby(["fold", "rep"]).ngroups); M["VxRpR"] = str(U.rep.nunique())
        for _, r in T.iterrows():
            k = key[r.method]
            M[f"VxRp{k}Phi"] = f3(r.phi); M[f"VxRp{k}PhiCI"] = f"{f3(r.phi_lo)}--{f3(r.phi_hi)}"
            M[f"VxRp{k}J"] = f3(r.pairwise_jaccard); M[f"VxRp{k}AUC"] = f4(r.auc_mean); M[f"VxRp{k}AUCsd"] = f4(r.auc_sd_within_fold)
            M[f"VxRp{k}Rho"] = f3(r.spearman); M[f"VxRp{k}Disc"] = f3(r.discrepancy); M[f"VxRp{k}Amb"] = f3(r.ambiguity)
            M[f"VxRp{k}Sizes"] = f"{int(r.size_min)}--{int(r.size_max)}"
            if "bagged_auc" in r and not np.isnan(r.bagged_auc):
                M[f"VxRp{k}Bag"] = f4(r.bagged_auc)
            M[f"VxRp{k}AmbPct"] = f"{100 * r.ambiguity:.0f}"
            if not np.isnan(r.share):
                M[f"VxRp{k}Psi"] = f3(r.share); M[f"VxRp{k}PsiCI"] = f"{f3(r.lo)}--{f3(r.hi)}"
        order = ["six_stage", "mw_topk", "l1_topk", "stability", "mrmr_ifs_svm", "random", "fixed"]
        rows = [r"\begin{tabular}{lcccccccc}", r"\toprule",
                r"Selector & $\hat\Phi$ (95\% CI) & Jaccard & Held-out AUC & AUC SD & Bagged AUC & Spearman $\rho$ & Ambiguity & $\psi$ (95\% CI) \\",
                r"\midrule"]
        for m in order:
            r = T[T.method == m]
            if r.empty:
                continue
            r = r.iloc[0]
            psi = "--" if np.isnan(r.share) else f"{f3(r.share)} ({f3(r.lo)}--{f3(r.hi)})"
            phi = "1 (fixed)" if m == "fixed" else f"{f3(r.phi)} ({f3(r.phi_lo)}--{f3(r.phi_hi)})"
            jac = "1" if m == "fixed" else f3(r.pairwise_jaccard)
            rows.append(f"{NAMES[m]} & {phi} & {jac} & {f4(r.auc_mean)} & {f4(r.auc_sd_within_fold)} & {f4(r.bagged_auc)} & "
                        f"{f3(r.spearman)} & {f3(r.ambiguity)} & {psi} \\\\")
        rows += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(out, "V2_replicates.tex"), "w").write("\n".join(rows) + "\n")
        # selection bagging: mean of the R replicate scores per held-out sample, against single nested models
        import pickle
        from .engine import T as ENG
        Sc = pd.read_csv(os.path.join(d, "results", "scores.csv"))
        ref = pd.read_csv(os.path.join(ROOT, "data", "reference", "R4_nested_oof.csv")).set_index("sample")
        bdir = latest("benchmark")
        mr = pickle.load(open(os.path.join(bdir, "cache", "mrmr_ifs_svm_00.pkl"), "rb"))["oof"] if bdir else None
        gains = []
        for m in ["six_stage", "mw_topk", "l1_topk", "stability", "random", "mrmr_ifs_svm"]:
            b = Sc[Sc.method == m].groupby("sample").score.mean().reindex(ref.index)
            yy = ref.y.values
            if m == "six_stage":
                dl = ENG.delong_paired(yy, b.values, ref.oof_prob.values)
                M["VxBagSixVsNested"] = pp(dl["diff"]); M["VxBagSixVsNestedCI"] = f"{pp(dl['lo95'])} to {pp(dl['hi95'])}"
                M["VxBagSixVsNestedP"] = pmac(dl["p"])
                if mr is not None:
                    dm = ENG.delong_paired(yy, b.values, mr)
                    M["VxBagSixVsMrmr"] = pp(dm["diff"]); M["VxBagSixVsMrmrP"] = pmac(dm["p"])
            r = T[T.method == m].iloc[0]
            gains.append(dict(method=m, gain=float(r.bagged_auc - r.auc_mean), psi=float(r.share) if not np.isnan(r.share) else np.nan))
            M[f"VxRp{key[m]}BagGain"] = pp(r.bagged_auc - r.auc_mean)
        G = pd.DataFrame(gains).dropna()
        from scipy.stats import spearmanr
        rho = spearmanr(G.psi, G.gain)
        M["VxBagPsiRho"] = f2(rho.correlation); M["VxBagPsiN"] = str(len(G))
        G.to_csv(os.path.join(out, "V2_bagging_gain.csv"), index=False)
        figure(d, T, out)
        psi_design(d, out, M)
    else:
        missing.append("replicates")
    # ---------------------------------------------------------------- pathway model (+ GO BP sensitivity)
    for exp_dir, tag in ((latest("pathway_model"), "Pw"), (latest_in("sensitivity", "pathway_model"), "PwGo")):
        if not exp_dir:
            missing.append(f"pathway_model ({tag})"); continue
        s = json.load(open(os.path.join(exp_dir, "summary.json")))
        n, r, st, lk = s["nested"], s["replicates"], s["site"], s["locked"]
        M[f"Vx{tag}AUC"] = f4(n["auc"]); M[f"Vx{tag}CI"] = f"{f4(n['lo'])}--{f4(n['hi'])}"
        M[f"Vx{tag}DeltaFour"] = pp(n["delta_vs_four"]); M[f"Vx{tag}DeltaFourCI"] = f"{pp(n['delta_vs_four_lo'])} to {pp(n['delta_vs_four_hi'])}"
        M[f"Vx{tag}PFour"] = pmac(n["p_vs_four"]); M[f"Vx{tag}Nsets"] = f"{n['n_sets']:,}".replace(",", "{,}")
        M[f"Vx{tag}Universe"] = f"{n['universe']:,}".replace(",", "{,}")
        M[f"Vx{tag}Int"] = signed2(n["intercept"]); M[f"Vx{tag}Slope"] = f2(n["slope"]); M[f"Vx{tag}Brier"] = f3(n["brier"])
        M[f"Vx{tag}Sens"] = f"{f3(n['sens'])} ({f3(n['sens_ci'][0])}--{f3(n['sens_ci'][1])})"
        M[f"Vx{tag}Spec"] = f"{f3(n['spec'])} ({f3(n['spec_ci'][0])}--{f3(n['spec_ci'][1])})"
        M[f"Vx{tag}Phi"] = f3(n["phi"]); M[f"Vx{tag}PhiCI"] = f"{f3(n['phi_lo'])}--{f3(n['phi_hi'])}"; M[f"Vx{tag}J"] = f3(n["jaccard"])
        M[f"Vx{tag}RpPhi"] = f3(r["phi"]); M[f"Vx{tag}RpPhiCI"] = f"{f3(r['phi_lo'])}--{f3(r['phi_hi'])}"; M[f"Vx{tag}RpJ"] = f3(r["jaccard"])
        M[f"Vx{tag}RpAUC"] = f4(r["auc_mean"]); M[f"Vx{tag}RpAUCsd"] = f4(r["auc_sd_within_fold"]); M[f"Vx{tag}RpRho"] = f3(r["spearman"])
        M[f"Vx{tag}RpAmb"] = f3(r["ambiguity"]); M[f"Vx{tag}RpPsi"] = f3(r["psi"]); M[f"Vx{tag}RpPsiCI"] = f"{f3(r['psi_lo'])}--{f3(r['psi_hi'])}"
        M[f"Vx{tag}RpBag"] = f4(r["bagged_auc"])
        M[f"Vx{tag}VumcAUC"] = f4(st["vumc_nested_auc"]); M[f"Vx{tag}VumcCI"] = f"{f4(st['vumc_nested_ci'][0])}--{f4(st['vumc_nested_ci'][1])}"
        M[f"Vx{tag}TransferAUC"] = f4(st["transfer_auc"]); M[f"Vx{tag}TransferCI"] = f"{f4(st['transfer_ci'][0])}--{f4(st['transfer_ci'][1])}"
        M[f"Vx{tag}ExtAUC"] = f4(lk["auc"]); M[f"Vx{tag}ExtCI"] = f"{f4(lk['auc_ci'][0])}--{f4(lk['auc_ci'][1])}"
        M[f"Vx{tag}ExtInt"] = f"{signed2(lk['intercept'])} ({signed2(lk['int_ci'][0])} to {signed2(lk['int_ci'][1])})"
        M[f"Vx{tag}ExtSlope"] = f"{f2(lk['slope'])} ({f2(lk['slope_ci'][0])}--{f2(lk['slope_ci'][1])})"
        M[f"Vx{tag}ExtBrier"] = f"{f3(lk['brier'])} ({f3(lk['brier_ci'][0])}--{f3(lk['brier_ci'][1])})"
        M[f"Vx{tag}ExtSens"] = f"{f3(lk['sens'])} ({f3(lk['sens_ci'][0])}--{f3(lk['sens_ci'][1])})"
        M[f"Vx{tag}ExtSpec"] = f"{f3(lk['spec'])} ({f3(lk['spec_ci'][0])}--{f3(lk['spec_ci'][1])})"
        M[f"Vx{tag}ExtThr"] = f3(lk["threshold"]); M[f"Vx{tag}ExtMeanPred"] = f2(lk["mean_pred"])
        M[f"Vx{tag}ExtGap"] = pp(n["auc"] - lk["auc"])
        for a in s["availability"]:
            if a["fraction"] > 0:
                q = {10: "Ten", 20: "Twenty", 30: "Thirty", 50: "Fifty"}[int(round(100 * a["fraction"]))]
                M[f"Vx{tag}Av{q}"] = f4(a["median"]); M[f"Vx{tag}Av{q}Lo"] = f4(a["q05"])
        if tag == "Pw":
            nice = [c.replace("REACTOME_", "").replace("HALLMARK_", "").replace("_", " ").lower() for c in lk["sets"]]
            open(os.path.join(out, "V2_pathway_sets.txt"), "w").write("\n".join(lk["sets"]) + "\n")
            M["VxPwTopSets"] = "; ".join(nice[:5])
    # ---------------------------------------------------------------- technical cards + combined table
    d = latest("selector_cards")
    if d:
        Cd = pd.read_csv(os.path.join(d, "results", "cards.csv"))
        key = {"six_stage": "Six", "mw_topk": "Mw", "l1_topk": "Lone", "stability": "Stab", "mrmr_ifs_svm": "Mrmr"}
        for _, r in Cd.iterrows():
            k = key[r.method]
            M[f"VxCd{k}Vumc"] = f4(r.vumc_nested_auc); M[f"VxCd{k}VumcCI"] = f"{f4(r.vumc_lo)}--{f4(r.vumc_hi)}"
            M[f"VxCd{k}Transfer"] = f4(r.transfer_auc); M[f"VxCd{k}TransferCI"] = f"{f4(r.transfer_lo)}--{f4(r.transfer_hi)}"
            M[f"VxCd{k}SiteDom"] = f"{100 * r.site_dominant:.0f}"; M[f"VxCd{k}Cov"] = f"{100 * r.coverage:.0f}"
            M[f"VxCd{k}VumcSig"] = f"{100 * r.vumc_sig:.0f}"
            M[f"VxCd{k}Ext"] = f4(r.external_auc); M[f"VxCd{k}ExtCI"] = f"{f4(r.external_lo)}--{f4(r.external_hi)}"
            M[f"VxCd{k}ExtInt"] = signed2(r.external_intercept); M[f"VxCd{k}N"] = str(int(r.n_genes))
        combined(out, M)
        cards_profile(out, M)
    else:
        missing.append("selector_cards")
    generalisation(out, M, missing)
    for k in [k for k in M if k.startswith("VxGen") and "Rho" in k and not k.endswith("P")]:
        M[k] = re.sub(r"(?<![0-9$])-(?=[0-9])", "$-$", M[k])      # typographic minus in correlations
    json.dump(M, open(os.path.join(out, "macros_v2.json"), "w"), indent=1)
    print(f"{len(M)} macros; missing experiments: {missing}")


def combined(out, M):
    """One row per model, one column per PMTRD quantity (macros only, so the table matches the text)."""
    rows = [r"\begin{tabular}{l ccc ccc cccc}", r"\toprule",
            r" & \multicolumn{3}{c}{Predictive} & \multicolumn{3}{c}{Molecular} & \multicolumn{4}{c}{Technical} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-11}",
            r"Model & Nested AUC & External AUC & Ext.\ intercept & $\hat\Phi$ folds & $\hat\Phi$ replicates & $\psi$ & "
            r"Vumc nested AUC & Vumc$\rightarrow$other sites & Site-dominant (\%) & Coverage (\%) \\", r"\midrule"]
    spec = [("Six-stage (paper)", "Four", "Six", "Six"), ("Mann--Whitney top 27", "Mw", "Mw", "Mw"),
            ("L1-logistic top 27", "Lone", "Lone", "Lone"), ("Stability selection", "Stab", "Stab", "Stab"),
            ("mRMR--IFS--SVM", "Mrmr", "Mrmr", "Mrmr")]
    g = lambda k: M.get(k, "--")
    for name, b, r, c in spec:
        rows.append(f"{name} & {g(f'VxBm{b}AUC')} & {g(f'VxCd{c}Ext')} & {g(f'VxCd{c}ExtInt')} & {g(f'VxBm{b}Phi')} & "
                    f"{g(f'VxRp{r}Phi')} & {g(f'VxRp{r}Psi')} & {g(f'VxCd{c}Vumc')} & {g(f'VxCd{c}Transfer')} & "
                    f"{g(f'VxCd{c}SiteDom')} & {g(f'VxCd{c}Cov')} \\\\")
    rows.append(r"\midrule")
    rows.append(f"Pathway model (reliability-guided) & {g('VxPwAUC')} & {g('VxPwExtAUC')} & {g('VxPwExtInt').split(' (')[0]} & "
                f"{g('VxPwPhi')} & {g('VxPwRpPhi')} & {g('VxPwRpPsi')} & {g('VxPwVumcAUC')} & {g('VxPwTransferAUC')} & -- & 100 \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "V2_cards.tex"), "w").write("\n".join(rows) + "\n")


def figure(d, T, out):
    """Four panels: (A) identity stability vs discrimination, (B) identity stability vs psi,
    (C) per-sample prediction variance with the panel re-selected vs fixed, (D) psi vs selection-bagging gain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Sc = pd.read_csv(os.path.join(d, "results", "scores.csv"))
    col = {"six_stage": "#2a6f97", "mw_topk": "#6d6d6d", "l1_topk": "#9c6b30", "stability": "#3d8b5a",
           "mrmr_ifs_svm": "#b5532a", "random": "#a9a9a9", "pathway": "#7b3f8c"}
    lab = {"six_stage": "Six-stage", "mw_topk": "Mann–Whitney", "l1_topk": "L1-logistic", "stability": "Stability sel.",
           "mrmr_ifs_svm": "mRMR–IFS–SVM", "random": "Random", "pathway": "Pathway model (sets)"}
    off = {"six_stage": (8, -10), "mw_topk": (-62, 8), "l1_topk": (-58, 6), "stability": (8, -11),
           "mrmr_ifs_svm": (-40, 9), "random": (8, 4), "pathway": (-60, -14)}
    offb = {**off, "l1_topk": (-20, -16), "mw_topk": (-62, 8)}
    pw = latest("pathway_model")
    P = json.load(open(os.path.join(pw, "summary.json")))["replicates"] if pw else None
    fig, ax = plt.subplots(2, 2, figsize=(10.4, 8.0))
    a, b, c, e = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    for _, r in T.iterrows():
        if r.method == "fixed":
            continue
        m = r.method
        a.errorbar(r.phi, r.auc_mean, xerr=[[r.phi - r.phi_lo], [r.phi_hi - r.phi]], yerr=r.auc_sd_within_fold,
                   fmt="o", color=col[m], capsize=3, ms=7)
        a.annotate(lab[m], (r.phi, r.auc_mean), textcoords="offset points", xytext=off[m], fontsize=8)
        if not np.isnan(r.share):
            b.errorbar(r.phi, r.share, xerr=[[r.phi - r.phi_lo], [r.phi_hi - r.phi]], yerr=[[r.share - r.lo], [r.hi - r.share]],
                       fmt="o", color=col[m], capsize=3, ms=7)
            b.annotate(lab[m], (r.phi, r.share), textcoords="offset points", xytext=offb[m], fontsize=8)
            e.scatter(r.share, 100 * (r.bagged_auc - r.auc_mean), color=col[m], s=45, zorder=3)
            e.annotate(lab[m], (r.share, 100 * (r.bagged_auc - r.auc_mean)), textcoords="offset points",
                       xytext=(7, -3), fontsize=8)
    if P:
        a.errorbar(P["phi"], P["auc_mean"], yerr=P["auc_sd_within_fold"], fmt="s", color=col["pathway"], capsize=3, ms=7)
        a.annotate(lab["pathway"], (P["phi"], P["auc_mean"]), textcoords="offset points", xytext=off["pathway"], fontsize=8)
        b.errorbar(P["phi"], P["psi"], yerr=[[P["psi"] - P["psi_lo"]], [P["psi_hi"] - P["psi"]]], fmt="s",
                   color=col["pathway"], capsize=3, ms=7)
        b.annotate(lab["pathway"], (P["phi"], P["psi"]), textcoords="offset points", xytext=off["pathway"], fontsize=8)
    for q in (a, b):
        q.set_xlim(-0.05, 1.02); q.set_xlabel("Identity stability, Nogueira $\\hat\\Phi$ (95% CI)")
    a.set_ylabel("Held-out AUC (mean; bar = SD across replicates)")
    a.set_title("A  Discrimination does not track identity stability", loc="left", fontsize=10)
    b.set_ylabel("$\\psi$, share of prediction variance due to re-selection")
    b.set_ylim(0, 1); b.set_title("B  Unstable identity makes predictions unstable", loc="left", fontsize=10)
    lg = lambda p_: np.log(np.clip(p_, 1e-6, 1 - 1e-6) / (1 - np.clip(p_, 1e-6, 1 - 1e-6)))
    v = {}
    for m in ("six_stage", "fixed"):
        s_ = Sc[Sc.method == m].copy(); s_["lg"] = lg(s_.score)
        v[m] = s_.groupby("sample").lg.var(ddof=1)
    j = pd.concat(v, axis=1).dropna()
    yb = Sc.drop_duplicates("sample").set_index("sample").y.reindex(j.index)
    for k_, nm, cc in ((1, "NSCLC", "#b5532a"), (0, "healthy", "#2a6f97")):
        q = j[yb == k_]
        c.scatter(q.fixed, q.six_stage, s=8, alpha=.5, color=cc, label=nm)
    lo_, hi_ = max(j.min().min(), 1e-3) / 1.5, j.max().max() * 1.5
    c.plot([lo_, hi_], [lo_, hi_], ls="--", color="grey", lw=1)
    c.set_xscale("log"); c.set_yscale("log"); c.set_xlim(lo_, hi_); c.set_ylim(lo_, hi_)
    c.set_xlabel("Variance of predicted log-odds, panel fixed"); c.set_ylabel("Variance, panel re-selected (six-stage)")
    c.set_title("C  Per held-out patient (633 samples)", loc="left", fontsize=10); c.legend(fontsize=8, frameon=False, loc="upper left")
    e.set_xlabel("$\\psi$"); e.set_ylabel("Gain from selection bagging (AUC, percentage points)")
    e.set_xlim(0.45, 0.95); e.set_title("D  Bagging helps most where $\\psi$ is high", loc="left", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_reliability_link.{ext}"), dpi=300, bbox_inches="tight")



def _num(v):
    """Macro string -> float (handles $-$ and missing values)."""
    try:
        return float(str(v).replace("$-$", "-").replace("{,}", "").split(" ")[0])
    except ValueError:
        return np.nan


def cards_profile(out, M):
    """Heat map of the PMTRD comparison (same macros as V2_cards.tex): every column ranked best to worst."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    rows = [("Six-stage (paper)", "Four", "Six", "Six"), ("Mann–Whitney top 27", "Mw", "Mw", "Mw"),
            ("L1-logistic top 27", "Lone", "Lone", "Lone"), ("Stability selection", "Stab", "Stab", "Stab"),
            ("mRMR–IFS–SVM", "Mrmr", "Mrmr", "Mrmr")]
    cols = [("Nested\nAUC", "P", +1), ("External\nAUC", "P", +1), ("|External\nintercept|", "P", -1),
            ("$\\hat\\Phi$\nfolds", "M", +1), ("$\\hat\\Phi$\nreplicates", "M", +1), ("$\\psi$", "M", -1),
            ("Vumc\nnested AUC", "T", +1), ("Vumc →\nother sites", "T", +1), ("Site-\ndominant %", "T", -1),
            ("Coverage\n%", "T", +1)]
    V, names = [], []
    for name, b, r, c in rows:
        names.append(name)
        V.append([_num(M.get(f"VxBm{b}AUC")), _num(M.get(f"VxCd{c}Ext")), abs(_num(M.get(f"VxCd{c}ExtInt"))),
                  _num(M.get(f"VxBm{b}Phi")), _num(M.get(f"VxRp{r}Phi")), _num(M.get(f"VxRp{r}Psi")),
                  _num(M.get(f"VxCd{c}Vumc")), _num(M.get(f"VxCd{c}Transfer")), _num(M.get(f"VxCd{c}SiteDom")),
                  _num(M.get(f"VxCd{c}Cov"))])
    names.append("Pathway model")
    V.append([_num(M.get("VxPwAUC")), _num(M.get("VxPwExtAUC")), abs(_num(str(M.get("VxPwExtInt", "")).split(" (")[0])),
              _num(M.get("VxPwPhi")), _num(M.get("VxPwRpPhi")), _num(M.get("VxPwRpPsi")), _num(M.get("VxPwVumcAUC")),
              _num(M.get("VxPwTransferAUC")), np.nan, 100.0])
    V = np.array(V, dtype=float)
    R = np.full_like(V, np.nan)
    for j, (_, _, sgn) in enumerate(cols):
        v = V[:, j]; ok = ~np.isnan(v)
        R[ok, j] = pd.Series(-sgn * v[ok]).rank(method="average").values   # 1 = best
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
    cmap = LinearSegmentedColormap.from_list("blue", ramp[::-1])             # rank 1 -> darkest
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    nr, nc = V.shape
    for i in range(nr):
        for j in range(nc):
            x0, y0 = j, nr - 1 - i
            if np.isnan(V[i, j]):
                ax.add_patch(plt.Rectangle((x0 + .03, y0 + .05), .94, .9, color="#f0efec", lw=0))
                ax.text(x0 + .5, y0 + .5, "–", ha="center", va="center", fontsize=9, color="#6d6d6d")
                continue
            t = (R[i, j] - 1) / (np.nanmax(R[:, j]) - 1) if np.nanmax(R[:, j]) > 1 else 0
            colr = cmap(t)
            ax.add_patch(plt.Rectangle((x0 + .03, y0 + .05), .94, .9, color=colr, lw=0))
            j_ = cols[j][0]
            val = V[i, j]
            txt = f"{val:.0f}" if "%" in j_ else (f"{val:.2f}" if "intercept" in j_ else f"{val:.3f}")
            best = R[i, j] == np.nanmin(R[:, j])
            ax.text(x0 + .5, y0 + .5, txt, ha="center", va="center", fontsize=9,
                    color="white" if t < 0.45 else "#1a1a1a", fontweight="bold" if best else "normal")
    ax.set_xlim(0, nc); ax.set_ylim(-0.75, nr + 0.05)
    ax.set_yticks([nr - 1 - i + .5 for i in range(nr)]); ax.set_yticklabels(names, fontsize=9)
    ax.set_xticks([j + .5 for j in range(nc)]); ax.set_xticklabels([c[0] for c in cols], fontsize=8.5)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axhline(1.0, color="white", lw=4)          # separate the pathway model
    for lab, a0, a1 in (("Predictive", 0, 3), ("Molecular", 3, 6), ("Technical", 6, 10)):
        ax.annotate("", xy=(a0 + .08, -0.25), xytext=(a1 - .08, -0.25), xycoords="data",
                    arrowprops=dict(arrowstyle="-", color="#6d6d6d", lw=1.2), annotation_clip=False)
        ax.text((a0 + a1) / 2, -0.55, lab, ha="center", va="center", fontsize=9, color="#3d3d3d")
    fig.text(0.99, 0.005, "Darker = better rank within the column (bold: best). Higher is better, except for "
             "|external intercept|, ψ and site-dominant %.", ha="right", fontsize=7.5, color="#6d6d6d")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_cards_profile.{ext}"), dpi=300, bbox_inches="tight")
    M["VxCardsBestCount"] = ", ".join(f"{n}: {int(np.nansum(R[i] == np.nanmin(R, axis=0)))}" for i, n in enumerate(names))


def psi_design(d, out, M):
    """(A) The control-arm design behind psi; (B) illustrative held-out patients of the discovery cohort."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    Sc = pd.read_csv(os.path.join(d, "results", "scores.csv"))
    U = pd.read_csv(os.path.join(d, "results", "units.csv"))
    thr = U.set_index(["fold", "rep", "method"]).threshold
    Sc["thr"] = [thr[(f, r, m)] for f, r, m in zip(Sc.fold, Sc.rep, Sc.method)]
    Sc["lab"] = (Sc.score >= Sc.thr).astype(int)
    fig = plt.figure(figsize=(10.4, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.12)
    a = fig.add_subplot(gs[0]); a.set_xlim(0, 10.4); a.set_ylim(0, 4.85); a.axis("off")
    ink, mut = "#1a1a1a", "#6d6d6d"
    def box(x, y, w, h, txt, fc="#f4f7fb", ec="#9ec5f4", fs=8.6):
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
        a.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=ink)
    def arrow(x0, y0, x1, y1):
        a.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=mut, lw=1.2))
    box(0.05, 2.75, 2.15, 1.15, "Outer fold $k$\ntraining set $T_k$\n(held-out fold $H_k$ untouched)")
    box(0.05, 0.75, 2.15, 1.25, "$R=10$ stratified 80%\nsubsamples $S_{k1},\\dots,S_{kR}$\nof $T_k$")
    arrow(1.12, 2.75, 1.12, 2.0)
    box(2.85, 2.55, 2.55, 1.35, "Re-selected arm\nselect panel $\\mathcal{A}(S_{kr})$\ntrain model $f_{kr}$ on $S_{kr}$",
        fc="#e8f0fb", ec="#256abf")
    box(2.85, 0.6, 2.55, 1.35, "Panel-fixed arm\nfixed panel $P_k=\\mathcal{A}(T_k)$\ntrain model $g_{kr}$ on $S_{kr}$",
        fc="#f5f2ec", ec="#9c6b30")
    arrow(2.2, 1.45, 2.85, 3.15); arrow(2.2, 1.3, 2.85, 1.25)
    box(6.0, 1.45, 1.75, 1.75, "Score every\nheld-out patient $i$\nin $H_k$:\n$R$ predictions\nper arm")
    arrow(5.4, 3.2, 6.0, 2.6); arrow(5.4, 1.25, 6.0, 2.0)
    box(8.25, 3.05, 2.1, 1.3, "$\\psi = 1-\\frac{\\sum_i\\, \\mathrm{Var}_r\\, g_{kr}(x_i)}{\\sum_i\\, \\mathrm{Var}_r\\, f_{kr}(x_i)}$"
        "\nVar: variance of the\npredicted log-odds", fs=8.0)
    box(8.25, 1.85, 2.1, 1.05, "Ambiguity: share of\npatients given both labels\n(training-set thresholds)", fs=8.2)
    box(8.25, 0.5, 2.1, 1.05, "Selection bagging:\nmean of the $R$\nre-selected models", fs=8.2)
    for yy in (3.7, 2.37, 1.02):
        arrow(7.75, 2.32, 8.25, yy)
    a.text(0.0, 4.72, "A  The control-arm design: what re-selecting the genes adds to the variation caused by the training samples",
           fontsize=10, ha="left", va="center")
    # (B) illustrative patients: re-selected arm gives both labels, panel-fixed arm one; most balanced split first
    b = fig.add_subplot(gs[1])
    g = Sc[Sc.method.isin(["six_stage", "fixed"])]
    st = g.groupby(["sample", "method"]).lab.agg(["mean", "nunique"]).unstack()
    y_ = g.drop_duplicates("sample").set_index("sample").y
    cand = pd.DataFrame({"bal": (st[("mean", "six_stage")] - 0.5).abs(),
                         "ok": (st[("nunique", "six_stage")] == 2) & (st[("nunique", "fixed")] == 1)})
    cand = cand[cand.ok].join(y_.rename("y")).rename_axis("sample").reset_index().sort_values(["bal", "sample"])
    pick = list(cand[cand.y == 1].head(3)["sample"]) + list(cand[cand.y == 0].head(3)["sample"])
    M["VxDesignNcand"] = str(len(cand)); M["VxDesignNheld"] = str(g["sample"].nunique())
    yt, yl = [], []
    for i, smp in enumerate(pick):
        for k, (m, nm) in enumerate((("six_stage", "re-selected"), ("fixed", "panel fixed"))):
            q = g[(g["sample"] == smp) & (g.method == m)]
            yy = -(i * 2.6 + k)
            colr = "#256abf" if m == "six_stage" else "#9c6b30"
            b.scatter(q.score[q.lab == 1], [yy] * int((q.lab == 1).sum()), s=42, color=colr, zorder=3)
            b.scatter(q.score[q.lab == 0], [yy] * int((q.lab == 0).sum()), s=42, facecolors="white", edgecolors=colr,
                      linewidths=1.4, zorder=3)
            yt.append(yy); yl.append(f"{'NSCLC' if y_[smp] == 1 else 'Healthy'} patient {i % 3 + 1}, {nm}")
    b.set_yticks(yt); b.set_yticklabels(yl, fontsize=8)
    b.set_xlim(0, 1); b.set_xlabel("Predicted probability of NSCLC from each of the 10 re-derived models")
    b.grid(axis="x", color="#e6e6e6", lw=0.8); b.set_axisbelow(True)
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)
    b.scatter([], [], s=42, color="#256abf", label="re-selected arm")
    b.scatter([], [], s=42, color="#9c6b30", label="panel-fixed arm")
    b.scatter([], [], s=42, color="#6d6d6d", label="filled: classified NSCLC")
    b.scatter([], [], s=42, facecolors="white", edgecolors="#6d6d6d", linewidths=1.4, label="open: classified healthy")
    b.legend(fontsize=8, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.17))
    b.set_title("B  Six illustrative held-out patients: re-selecting the genes splits their classification", loc="left",
                fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_psi_design.{ext}"), dpi=300, bbox_inches="tight")


GEN = [("GSE89843", "Disc", None), ("GSE68086", "Gse", "gse68086_pancancer"), ("Singh", "Singh", "singh_prostate"),
       ("Gordon", "Gordon", "gordon_lung")]
GEN_TASK = {"GSE89843": "NSCLC vs healthy, platelet RNA-seq", "GSE68086": "six cancers vs healthy, platelet RNA-seq",
            "Singh": "prostate tumour vs normal, microarray", "Gordon": "mesothelioma vs adenocarcinoma, microarray"}


def generalisation(out, M, missing):
    """Replicate-study summaries of every data set, pooled relationships, table and figure."""
    from scipy.stats import spearmanr
    rows = []
    for label, tag, sub in GEN:
        d = latest("replicates") if sub is None else latest_in(os.path.join("generalisation", sub), "replicates_generic")
        if not d:
            missing.append(f"generalisation {label}"); continue
        S = pd.read_csv(os.path.join(d, "results", "stability.csv")); A = pd.read_csv(os.path.join(d, "results", "agreement.csv"))
        I = pd.read_csv(os.path.join(d, "results", "identity_share.csv"))
        T = S.merge(A, on="method").merge(I[["method", "share", "lo", "hi"]], on="method", how="left")
        info = json.load(open(os.path.join(d, "results", "dataset.json"))) if os.path.exists(os.path.join(d, "results", "dataset.json")) else \
            dict(n=633, positives=402, negatives=231, features=27949, task="NSCLC vs healthy donors", source="GEO GSE89843")
        for _, r in T.iterrows():
            rows.append(dict(dataset=label, tag=tag, method=r.method, phi=r.phi, psi=r.share, psi_lo=r.lo, psi_hi=r.hi,
                             auc=r.auc_mean, bag=r.bagged_auc, gain=r.bagged_auc - r.auc_mean, amb=r.ambiguity, n=info["n"],
                             features=info["features"], task=info["task"], pos=info["positives"], neg=info["negatives"]))
    if not rows:
        return
    G = pd.DataFrame(rows); G.to_csv(os.path.join(out, "V2_generalisation.csv"), index=False)
    xg = G[(G.method != "fixed") & (G.method != "mrmr_ifs_svm") & G.psi.notna()]
    r1 = spearmanr(xg.phi, xg.psi); r2 = spearmanr(xg.psi, xg.gain)
    M["VxGenN"] = str(len(xg)); M["VxGenNds"] = str(G.dataset.nunique())
    M["VxGenRhoPhiPsi"] = f2(r1.correlation); M["VxGenRhoPhiPsiP"] = pmac(r1.pvalue)
    M["VxGenRhoPsiGain"] = f2(r2.correlation); M["VxGenRhoPsiGainP"] = pmac(r2.pvalue)
    r3 = spearmanr(xg.phi, xg.gain)
    M["VxGenRhoPhiGain"] = f2(r3.correlation); M["VxGenRhoPhiGainP"] = pmac(r3.pvalue)
    within, wphi, wphig = [], [], []
    for ds, g in xg.groupby("dataset"):
        within.append(spearmanr(g.psi, g.gain).correlation); wphi.append(spearmanr(g.phi, g.psi).correlation)
        wphig.append(spearmanr(g.phi, g.gain).correlation)
    rng = lambda v: f"{min(v):.2f}" if np.isclose(min(v), max(v)) else f"{min(v):.2f} to {max(v):.2f}"
    M["VxGenRhoPsiGainWithin"] = rng(within); M["VxGenRhoPhiPsiWithin"] = rng(wphi); M["VxGenRhoPhiGainWithin"] = rng(wphig)
    # psi versus Phi as guides to the bagging gain: |rho(psi, gain)| - |rho(Phi, gain)|, with a bootstrap that
    # resamples selectors within data sets (exploratory; not pre-specified)
    xb = xg.reset_index(drop=True); brng = np.random.default_rng(42); dd = []
    for _ in range(5000):
        idx = np.concatenate([brng.choice(g.index, len(g), replace=True) for _, g in xb.groupby("dataset")])
        sb = xb.loc[idx]
        if sb.psi.nunique() < 3:
            continue
        dd.append(abs(spearmanr(sb.psi, sb.gain).correlation) - abs(spearmanr(sb.phi, sb.gain).correlation))
    dd = np.array(dd); dd = dd[~np.isnan(dd)]
    M["VxGenDiffRho"] = f2(abs(r2.correlation) - abs(r3.correlation))
    M["VxGenDiffRhoCI"] = f"{np.percentile(dd, 2.5):.2f} to {np.percentile(dd, 97.5):.2f}".replace("-", "$-$")
    nge = sum(abs(spearmanr(g.psi, g.gain).correlation) >= abs(spearmanr(g.phi, g.gain).correlation) - 1e-12
              for _, g in xg.groupby("dataset"))
    M["VxGenPsiGeqPhiN"] = str(nge)
    # selectors other than random panels only (is the relation carried by the random reference alone?)
    xn = xg[xg.method != "random"]
    M["VxGenPsiNrMin"] = f3(xn.psi.min()); M["VxGenPsiNrMax"] = f3(xn.psi.max())
    rr = xg[xg.method == "random"]; M["VxGenRandPsiMin"] = f3(rr.psi.min()); M["VxGenRandPsiMax"] = f3(rr.psi.max())
    for (ds, tag), g in G.groupby(["dataset", "tag"]):
        nr = g[~g.method.isin(["random", "fixed"])]
        M[f"VxGen{tag}GainNrMax"] = pp(nr.gain.max()); M[f"VxGen{tag}AUCNrMin"] = f3(nr.auc.min())
        M[f"VxGen{tag}SixPhiRank"] = str(int((nr.phi < nr[nr.method == "six_stage"].phi.iloc[0]).sum()) + 1)
        M[f"VxGen{tag}NrPhiMin"] = f3(nr.phi.min()); M[f"VxGen{tag}NrPhiMax"] = f3(nr.phi.max())
    r4 = spearmanr(xn.psi, xn.gain); r5 = spearmanr(xn.phi, xn.psi); r6 = spearmanr(xn.phi, xn.gain)
    M["VxGenNnr"] = str(len(xn))
    M["VxGenNrRhoPsiGain"] = f2(r4.correlation); M["VxGenNrRhoPsiGainP"] = pmac(r4.pvalue)
    M["VxGenNrRhoPhiPsi"] = f2(r5.correlation); M["VxGenNrRhoPhiPsiP"] = pmac(r5.pvalue)
    M["VxGenNrRhoPhiGain"] = f2(r6.correlation); M["VxGenNrRhoPhiGainP"] = pmac(r6.pvalue)
    G.assign(rho_note="").to_csv(os.path.join(out, "V2_generalisation.csv"), index=False)
    for (ds, tag), g in G.groupby(["dataset", "tag"]):
        six = g[g.method == "six_stage"].iloc[0]; fx = g[g.method == "fixed"].iloc[0]
        M[f"VxGen{tag}SixPhi"] = f3(six.phi); M[f"VxGen{tag}SixPsi"] = f3(six.psi); M[f"VxGen{tag}SixAUC"] = f4(six.auc)
        M[f"VxGen{tag}SixAmb"] = f3(six.amb); M[f"VxGen{tag}FixedAmb"] = f3(fx.amb); M[f"VxGen{tag}SixGain"] = pp(six.gain)
        M[f"VxGen{tag}FixedGain"] = pp(fx.gain); M[f"VxGen{tag}SixAmbPct"] = f"{100 * six.amb:.0f}"
        M[f"VxGen{tag}FixedAmbPct"] = f"{100 * fx.amb:.0f}"
        M[f"VxGen{tag}SixPsiCI"] = f"{f3(six.psi_lo)}--{f3(six.psi_hi)}"
        rs = g[g.method == "random"].iloc[0]; M[f"VxGen{tag}RandPsi"] = f3(rs.psi)
        xs = g[(g.method != "fixed") & (g.method != "mrmr_ifs_svm") & g.psi.notna()]
        M[f"VxGen{tag}PsiMin"] = f3(xs.psi.min()); M[f"VxGen{tag}PsiMax"] = f3(xs.psi.max())
        M[f"VxGen{tag}RhoPsiGain"] = f2(spearmanr(xs.psi, xs.gain).correlation)
        M[f"VxGen{tag}RhoPhiPsi"] = f2(spearmanr(xs.phi, xs.psi).correlation)
        M[f"VxGen{tag}N"] = str(int(six.n)); M[f"VxGen{tag}Features"] = f"{int(six.features):,}".replace(",", "{,}")
        mw = g[g.method == "mw_topk"].iloc[0]; M[f"VxGen{tag}MwPhi"] = f3(mw.phi); M[f"VxGen{tag}MwPsi"] = f3(mw.psi)
    # table: one block per data set
    nice = {"six_stage": "Six-stage", "mw_topk": "Mann--Whitney", "l1_topk": "L1-logistic", "stability": "Stability sel.",
            "mrmr_ifs_svm": "mRMR--IFS--SVM", "random": "Random", "fixed": "Panel fixed"}
    lines = [r"\begin{tabular}{@{}p{2mm}lcccccc@{}}", r"\toprule",
             r" & Selector & $\hat\Phi$ & $\psi$ (95\% CI) & Held-out AUC & Bagged AUC & Gain (pp) & Ambiguity \\", r"\midrule"]
    first = True
    for ds in [x[0] for x in GEN]:
        g = G[G.dataset == ds]
        if g.empty:
            continue
        if not first:
            lines.append(r"\midrule")
        first = False
        g0 = g.iloc[0]
        feats = f"{int(g0.features):,}".replace(",", "{,}")
        head = f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{ds}: {GEN_TASK[ds]}; $n={int(g0.n)}$, {feats} features}}}} \\\\"
        for i, m in enumerate(["six_stage", "mw_topk", "l1_topk", "stability", "mrmr_ifs_svm", "random", "fixed"]):
            r = g[g.method == m]
            if r.empty:
                continue
            r = r.iloc[0]
            psi = "--" if np.isnan(r.psi) else f"{f3(r.psi)} ({f3(r.psi_lo)}--{f3(r.psi_hi)})"
            phi = "1" if m == "fixed" else f3(r.phi)
            if i == 0:
                lines.append(head)
            lines.append(f" & {nice[m]} & {phi} & {psi} & {f4(r.auc)} & {f4(r.bag)} & {pp(r.gain)} & {f3(r.amb)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "V2_generalisation.tex"), "w").write("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.2))
    colors = {"GSE89843": "#2a6f97", "GSE68086": "#b5532a", "Singh": "#3d8b5a", "Gordon": "#7b3f8c"}
    marks = {"six_stage": "o", "mw_topk": "s", "l1_topk": "^", "stability": "D", "random": "x"}
    for _, r in xg.iterrows():
        ax[0].scatter(r.phi, r.psi, color=colors[r.dataset], marker=marks[r.method], s=45)
        ax[1].scatter(r.psi, 100 * r.gain, color=colors[r.dataset], marker=marks[r.method], s=45)
    for _, r in G[G.method == "fixed"].iterrows():      # panel-fixed arm: psi = 0 by construction
        ax[1].scatter(0.0, 100 * r.gain, color=colors[r.dataset], marker="P", s=55)
    for ds, c in colors.items():
        ax[0].scatter([], [], color=c, label=ds)
    for m, mk in marks.items():
        ax[1].scatter([], [], color="grey", marker=mk, label=nice[m].replace("--", "\u2013"))
    ax[1].scatter([], [], color="grey", marker="P", label="Panel fixed ($\\psi=0$)")
    ax[0].set_xlabel("Identity stability, Nogueira $\\hat\\Phi$"); ax[0].set_ylabel("$\\psi$, prediction variance due to re-selection")
    ax[0].set_title(f"A  Identity stability and $\\psi$ (Spearman {r1.correlation:.2f})".replace("-", "\u2212"), loc="left", fontsize=10)
    ax[1].set_xlabel("$\\psi$"); ax[1].set_ylabel("Gain from selection bagging (AUC, pp)")
    ax[1].set_title(f"B  $\\psi$ and the gain from bagging (Spearman {r2.correlation:.2f})", loc="left", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False, title="Data set", title_fontsize=8); ax[1].legend(fontsize=8, frameon=False, title="Selector", title_fontsize=8)
    ax[0].set_ylim(-0.05, 1.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_generalisation.{ext}"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(RUNS, "export"))
