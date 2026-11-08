"""Multiple-testing check across the revision experiments (E8-E19).

For each family of hypothesis tests it lists the raw p-values, the correction the plan pre-specified (if any) and,
as a labelled sensitivity analysis, Holm-adjusted p-values within the family. Nothing here changes a planned
analysis; the planned (un)corrected p-values are reported alongside.

Run:  python -m pmtrd.revision.multiple_testing
"""
from __future__ import annotations

import os

import pandas as pd
from statsmodels.stats.multitest import multipletests

from .common import path


def _family(name, planned, labels, p):
    p = pd.Series(p, index=labels, dtype=float)
    ok = p.notna()
    holm = pd.Series(float("nan"), index=labels)
    if ok.any():
        holm[ok] = multipletests(p[ok].values, method="holm")[1]
    return pd.DataFrame({"family": name, "planned_correction": planned, "test": labels, "p_raw": p.values,
                         "p_holm_within_family": holm.values})


def main() -> None:
    fams = []
    e8 = pd.read_csv(path("runs", "revision", "E8", "E8.csv"))
    pr = e8[e8.primary]
    f = _family("E8 primary comparisons (paired DeLong)", "Holm over 5 (pre-specified)", list(pr.arm), pr.p.values)
    f["p_holm_reported"] = pr.p_holm.values
    fams.append(f)
    e10 = pd.read_csv(path("runs", "external", "E10.csv")).set_index("model")
    pl = ["a2_locked27_median_imputed", "b_retrained18_anyread", "c_A4_measurable"]
    fams.append(_family("E10 paired bootstrap vs locked model (planned)", "none specified", pl,
                        e10.loc[pl, "diff_p"].values))
    pl2 = pl + ["b2_retrained_measurable10pct"]
    fams.append(_family("E10 paired bootstrap incl. sensitivity model b'", "none specified", pl2,
                        e10.loc[pl2, "diff_p"].values))
    nl = ["b_retrained18_anyread", "c_A4_measurable"]
    fams.append(_family("E10 random-panel null (empirical p)", "none specified", nl, e10.loc[nl, "null_p"].values))
    e11 = pd.read_csv(path("runs", "revision", "E11", "E11.csv")).dropna(subset=["p_vs_counts"])
    fams.append(_family("E11 paired DeLong vs counts", "none (descriptive by plan)", list(e11.scale),
                        e11.p_vs_counts.values))
    e16 = pd.read_csv(path("runs", "export", "E16.csv"))
    fams.append(_family("E16 stratified permutation (exploratory)", "none specified", list(e16.predictor),
                        e16.perm_p.values))
    p9 = path("runs", "bagging", "E9_tests.csv")
    if os.path.exists(p9):
        e9 = pd.read_csv(p9)
        fams.append(_family("E9 corrected resampled t-tests (4 data sets x 4 quantities)", "none specified",
                            list(e9.dataset + ":" + e9.quantity), e9.p.values))
    p9x = path("runs", "bagging", "E9_external.csv")
    if os.path.exists(p9x):
        x = pd.read_csv(p9x).iloc[:1]
        fams.append(_family("E9 external paired bootstrap vs locked", "none (single test)", ["ensemble25 - locked"],
                            x.diff_p.values))
    p17 = path("results", "pathway", "E17.csv")
    if os.path.exists(p17):
        e17 = pd.read_csv(p17)
        fams.append(_family("E17 Wilcoxon signed-rank over splits", "none specified", list(e17.comparison),
                            e17.p.values))
    out = pd.concat(fams, ignore_index=True)
    out["significant_raw_0.05"] = out.p_raw < 0.05
    out["significant_holm_0.05"] = out.p_holm_within_family < 0.05
    out["holm_changes_conclusion"] = out["significant_raw_0.05"] != out["significant_holm_0.05"]
    out.to_csv(path("runs", "revision", "multiple_testing.csv"), index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
