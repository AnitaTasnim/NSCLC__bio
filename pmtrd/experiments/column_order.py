"""Numerical noise floor of the fixed-panel AUC: XGBoost with column subsampling depends on the order of
the input columns. The fixed 27-gene panel is evaluated on the paper's folds in ``n_orders`` random column
orders (and in panel order, the paper's setting)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..engine import Dataset, T
from ..metrics import predictive as PR

DEFAULTS = dict(n_orders=20)


def main(run, data: Dataset, params: Dict, mode: str = "all") -> None:
    p = {**DEFAULTS, **(params or {})}
    X, y, er = data.X, data.y, data.run
    P = er.panel_in(X); folds = er.folds(); rng = np.random.default_rng(run.cfg["seed"])
    rows = []
    for i in range(p["n_orders"] + 1):
        order = P if i == 0 else list(rng.permutation(P))
        oof, _ = T.cv_oof(X[order], y, folds, er.xgb)
        rows.append(dict(order=i, panel_order=(i == 0), auc=PR.auc(y, oof)))
    D = pd.DataFrame(rows); D.to_csv(run.path("results", "orders.csv"), index=False)
    r = D[~D.panel_order].auc
    run.summary({"panel_order_auc": float(D.iloc[0].auc), "random_orders": int(len(r)), "mean": float(r.mean()),
                 "sd": float(r.std(ddof=1)), "min": float(r.min()), "max": float(r.max())})
