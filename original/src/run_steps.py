#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_steps.py - run the analyses step by step, with everything persisted to disk so a
step can be re-run (or a crashed session resumed) without repeating the others.

    python run_steps.py config.json r0 r1t r1g nested r1v r1n r8 r4 r2e r5 r6 shap
    python run_steps.py config.json r9            # long: 10,000 + 10,000 random panels
    python run_steps.py config.json merge         # write results_macros.tex + AUC table/figure

Each step writes <out>/macros/<step>.json and <out>/aucs/<step>.json; `merge` combines them.
The selection function is the seeded replica of the original Phase-1 code (selection_pipeline.py),
in the "fold-recomputable" form (no STRING PPI term, no G106 term) so that nested CV is free of
leakage.
"""
import json
import os
import pickle
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tep_analysis as T                      # noqa: E402
from selection_pipeline import make_select_fn      # noqa: E402

CACHE_KEYS = ["oof_P27", "oof_vumc_P27", "ext_shared", "ext_missing", "p_ext_frozen", "vumc_panel",
              "nested_six_stage", "nested_sheng", "r1_panel_table", "_tier2", "nested_vumc"]


def load_run(cfg_path):
    cfg = T.load_config(cfg_path)
    data = T.load_data(cfg)
    sel = make_select_fn(B_default=cfg["selection"]["boot_B"], n_jobs=cfg.get("sel_n_jobs", 2))
    run = T.Run(cfg, data, sel)
    for d in ("macros", "aucs"):
        os.makedirs(os.path.join(run.out, d), exist_ok=True)
    ck = os.path.join(run.out, "cache", "run_cache.pkl")
    if os.path.exists(ck):
        run.cache.update(pickle.load(open(ck, "rb")))
    return run


def save_cache(run):
    ck = os.path.join(run.out, "cache", "run_cache.pkl")
    old = pickle.load(open(ck, "rb")) if os.path.exists(ck) else {}
    old.update({k: run.cache[k] for k in CACHE_KEYS if k in run.cache})
    pickle.dump(old, open(ck + ".tmp", "wb"))
    os.replace(ck + ".tmp", ck)


# ---------------------------------------------------------------------------------------
# extra step: nested CV of the full cohort with the replica (macros + table + figure)
# ---------------------------------------------------------------------------------------
def step_nested(run):
    d = run.d
    y = d.y.values
    P = run.panel_in(d.X)
    folds = run.folds()
    res = run.cache.get("nested_six_stage") or T.nested_cv(run, d.X, y, folds, "six_stage")
    run.cache["nested_six_stage"] = res
    a, lo, hi, _ = T.delong_ci(y, res["oof"])
    fa = np.array(res["fold_aucs"])
    jac = [T.jaccard(P, p) for p in res["panels"]]
    run.register_auc("Within cohort", "Nested 5-fold CV (selection re-run in every outer training fold)",
                     a, lo, hi, "DeLong", len(y), sd=fa.std(ddof=1))
    run.macro("NestedAUC", T.f4(a)); run.macro("NestedCI", T.ci(lo, hi))
    run.macro("NestedFoldMean", T.f4(fa.mean())); run.macro("NestedFoldSD", T.f4(fa.std(ddof=1)))
    run.macro("NestedFoldMin", T.f4(fa.min())); run.macro("NestedFoldMax", T.f4(fa.max()))
    run.macro("NestedJMean", T.f3(np.mean(jac))); run.macro("NestedJSD", T.f3(np.std(jac, ddof=1)))
    run.macro("NestedSizes", ", ".join(str(len(p)) for p in res["panels"]))
    base = run.cache.get("oof_P27")
    if base is not None:
        fixed = T.roc_auc_score(y, base)
        run.macro("NestedGap", f"{100 * (fixed - a):.1f}")
        dp = T.delong_paired(y, base, res["oof"])
        run.macro("NestedGapCI", f"{T.pp(dp['lo95'])} to {T.pp(dp['hi95'])}")
        run.macro("NestedGapP", T.pmac(dp["p"]))
    rows = []
    for i, (f_auc, p, j) in enumerate(zip(fa, res["panels"], jac), 1):
        rows.append([str(i), T.f4(f_auc), str(len(p)), T.f3(j)])
    rows.append(["Mean $\\pm$ SD", f"{T.f4(fa.mean())} $\\pm$ {T.f4(fa.std(ddof=1))}", "--",
                 f"{T.f3(np.mean(jac))} $\\pm$ {T.f3(np.std(jac, ddof=1))}"])
    rows.append(["Pooled out-of-fold", f"{T.f4(a)} ({T.ci(lo, hi)})", "--", "--"])
    T.write_tex_table(os.path.join(run.out, "tables", "R4_nested_folds.tex"), rows,
                      ["Outer fold", "AUC", "Panel size", "Jaccard vs.\\ 27-gene panel"], "lccc", "\\small")
    pd.DataFrame({"fold": range(1, len(fa) + 1), "auc": fa, "n_genes": [len(p) for p in res["panels"]],
                  "jaccard_vs_original": jac, "genes": [";".join(p) for p in res["panels"]]}
                 ).to_csv(os.path.join(run.out, "tables", "R4_nested_folds.csv"), index=False)
    pd.DataFrame({"sample": d.X.index, "y": y, "oof_prob": res["oof"]}).to_csv(
        os.path.join(run.out, "tables", "R4_nested_oof.csv"), index=False)
    json.dump(res["panels"], open(os.path.join(run.out, "tables", "R4_nested_fold_panels.json"), "w"), indent=1)
    run.log(f"[nested] pooled AUC {a:.4f} [{lo:.4f}-{hi:.4f}], folds {np.round(fa, 4).tolist()}, "
            f"Jaccard {np.round(jac, 3).tolist()}")
    return res


def step_nested_fig(run):
    """Simplified nested figure (Figure 10)."""
    d = run.d
    res = run.cache["nested_six_stage"]
    P = run.panel_in(d.X)
    y = d.y.values
    jac = [T.jaccard(P, p) for p in res["panels"]]
    fixed = T.roc_auc_score(y, run.cache["oof_P27"])
    nested = T.roc_auc_score(y, res["oof"])
    T.fig_nested_summary(res["fold_aucs"], jac, fixed, nested, None,
                         os.path.join(run.out, "figures", "fig10_nested_simplified.pdf"))
    T.fig_nested_summary(res["fold_aucs"], jac, fixed, nested, None,
                         os.path.join(run.out, "figures", "fig10_nested_simplified.png"))


STEPS = {
    "r0": lambda r: T.r0_core_estimates(r),
    "r1t": lambda r: T.r1_site_table(r),
    "r1g": lambda r: T.r1_gene_effects(r),
    "nested": step_nested,
    "nestedfig": step_nested_fig,
    "r1v": lambda r: T.r1_vumc_rederivation(r),
    "r1n": lambda r: T.r1_vumc_nested(r),
    "r2e": lambda r: T.r2_panel_equivalence(r),
    "r2o": lambda r: T.r2_external_operating(r),
    "r3": lambda r: T.r3_harmonisation_audit(r),
    "r8": lambda r: T.r8_head_to_head(r),
    "r4": lambda r: T.r4_stability(r),
    "r5": lambda r: T.r5_contamination(r),
    "r6": lambda r: T.r6_covariates(r),
    # pathway analysis: use r7_gsea.py (one gene-set library per process; the in-process version needs > 6 GB of RAM)
    "r7": lambda r: sys.exit("Run the pathway analysis with r7_gsea.py (see README.md)."),
    "r9": lambda r: T.r9_random_panels(r),
    "r10": lambda r: T.r10_tissue(r),
    "shap": lambda r: T.shap_out_of_fold(r),
}


def merge(out_dir):
    macros, aucs = {}, []
    for f in sorted(os.listdir(os.path.join(out_dir, "macros"))):
        macros.update(json.load(open(os.path.join(out_dir, "macros", f))))
    order = list(STEPS)
    for f in sorted(os.listdir(os.path.join(out_dir, "aucs")),
                    key=lambda s: order.index(s[:-5]) if s[:-5] in order else 99):
        aucs += json.load(open(os.path.join(out_dir, "aucs", f)))
    path = os.path.join(out_dir, "results_macros.tex")
    with open(path, "w") as fh:
        fh.write("% Auto-generated by run_steps.py merge - do not edit by hand.\n")
        fh.write(f"% {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for k in sorted(macros):
            fh.write(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{macros[k]}}}\n")
    print(f"merged {len(macros)} macros -> {path}")
    if aucs:
        class _R:  # minimal stand-in for auc_summary
            pass
        r = _R(); r.aucs = aucs; r.out = out_dir
        r.table = lambda df, name, index=False: df.to_csv(os.path.join(out_dir, "tables", name + ".csv"), index=index)
        r.fig = lambda fig, name: [fig.savefig(os.path.join(out_dir, "figures", f"{name}.{e}"), dpi=300,
                                               bbox_inches="tight") for e in ("pdf", "png")]
        T.auc_summary(r)
        print(f"AUC summary: {len(aucs)} rows")


def main():
    cfg_path, steps = sys.argv[1], sys.argv[2:]
    if steps == ["merge"]:
        merge(T.load_config(cfg_path)["out_dir"]); return
    run = load_run(cfg_path)
    for s in steps:
        t = time.time()
        run.macros, run.aucs = {}, []
        run.log(f"##### step {s} started {time.strftime('%H:%M:%S')}")
        try:
            STEPS[s](run)
            ok = True
        except Exception as e:
            run.log(f"!! step {s} failed: {e!r}\n{traceback.format_exc()}")
            ok = False
        json.dump(run.macros, open(os.path.join(run.out, "macros", f"{s}.json"), "w"), indent=1)
        json.dump(run.aucs, open(os.path.join(run.out, "aucs", f"{s}.json"), "w"), indent=1, default=float)
        if s != "r9":
            save_cache(run)
        run.log(f"##### step {s} {'done' if ok else 'FAILED'} in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()
