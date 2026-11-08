"""E9, post hoc and descriptive (not in docs/plan_E8_E19.md; written before the E9 summaries were seen).

1. Reproduction check: on the paper's folds (partition 42) with the 10 saved replicates, the published pooled
   quantities of the six-stage bagging claim (pooled bagged AUC 0.9630; gain +2.83 = pooled bagged AUC minus the
   mean single-unit AUC over the 50 units) are recomputed from the E9 cache. The same pooled quantities are
   reported for B = 25 and for partition 43.
2. Comparison the audit asked for but the plan did not carry (deviation D7): pooled bagged A4 against the mRMR
   arms of E8 on the same folds (paired DeLong). Descriptive; no multiplicity correction; not a planned test.

Run:  python -m pmtrd.revision.e9_posthoc      (needs the E9 GSE89843 cache and the E8 cache)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine import T
from .common import out_dir, path, write_meta
from .e9 import NREP, cache_for
from .units import Cache, load_dataset


def main() -> None:
    from ..metrics import predictive as PR
    od = out_dir("runs", "bagging")
    X, y, _, _ = load_dataset("GSE89843", 1)
    n = len(y)
    cache = cache_for("GSE89843")
    rows, pooled = [], {}
    for part in (42, 43):
        for B in (10, 25):
            bag = np.full(n, np.nan)
            unit_auc = []
            for k in range(5):
                us = [cache.get(f"p{part}_f{k}_r{r:02d}") for r in range(B)]
                te = us[0]["test"]
                S = np.vstack([u["scores"] for u in us])
                bag[te] = S.mean(0)
                unit_auc += [PR.auc(y[te], s) for s in S]
            assert not np.isnan(bag).any()
            pooled[(part, B)] = bag
            a = PR.auc(y, bag)
            rows.append(dict(kind="pooled_published_definition", partition=part, B=B, auc_bag_pooled=a,
                             mean_unit_auc=float(np.mean(unit_auc)), gain_pooled=a - float(np.mean(unit_auc)),
                             n_units=len(unit_auc)))
    # E8 arms on the paper's folds (partition 42)
    e8 = Cache(path("runs", "revision", "E8", "cache"))
    arms = {}
    for k in range(5):
        u = e8.get(f"fold{k}")
        for name in ("mrmr@27/xgb", "mrmr@own/svm", "A4@own/xgb"):
            arms.setdefault(name, np.full(n, np.nan))[u["te"]] = u["scores"][name]
    for B in (10, 25):
        for name in ("mrmr@27/xgb", "mrmr@own/svm", "A4@own/xgb"):
            d = T.delong_paired(y, pooled[(42, B)], arms[name])
            rows.append(dict(kind="posthoc_bagged_A4_vs_E8_arm", partition=42, B=B, comparator=name,
                             auc_bag_pooled=d["auc1"], auc_comparator=d["auc2"], diff=d["diff"],
                             diff_lo=d["lo95"], diff_hi=d["hi95"], p=d["p"]))
    out = pd.DataFrame(rows)
    out.to_csv(path(od, "E9_posthoc.csv"), index=False)
    write_meta(od, name="metadata_E9_posthoc.json", experiment="E9_posthoc", nrep=NREP,
               note="post hoc and descriptive; not part of the pre-registered plan")
    print(out.to_string())


if __name__ == "__main__":
    main()
