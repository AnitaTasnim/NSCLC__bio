"""The replicate study on another data set: do the reliability findings generalise?

Same design as ``replicates`` (R stratified subsamples of every outer training fold; every selector chooses
a panel on the subsample and scores the untouched held-out fold; thresholds by inner CV on the subsample;
a panel-fixed arm), on a data set named in ``params.dataset``. The fixed arm's panel is the six-stage
panel selected on the whole outer training fold (computed here, as the paper's fold panels were). Outer
folds: stratified five-fold, seed 42. The classifier is the paper's XGBoost; mRMR-IFS-SVM uses its SVM.
Outputs and summaries are those of ``replicates`` plus ``dataset.json``.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from .. import selectors as SEL
from ..datasets import load as load_dataset
from ..engine import validate_inputs
from ..metrics import predictive as PR
from .common import UnitCache, stratified_subsample
from .replicates import summarise

DEFAULTS = dict(R=10, frac=0.8, k=27, B=500,
                methods=["six_stage", "mw_topk", "l1_topk", "stability", "mrmr_ifs_svm", "random"])


class GenericRun:
    """The parts of the engine's Run object that the selectors use, for an arbitrary data set."""

    def __init__(self, engine_cfg: Dict, counts: bool):
        self.cfg = engine_cfg; self.counts = counts

    def xgb(self, n_jobs=None):
        return XGBClassifier(**dict(self.cfg["xgb_params"]), n_jobs=n_jobs or 2, verbosity=0)

    def folds(self, y):
        return list(StratifiedKFold(5, shuffle=True, random_state=self.cfg["seed"]).split(np.zeros(len(y)), y))

    def logX(self, X):
        X = X.clip(lower=0)
        if self.counts:
            X = X.div(X.sum(axis=1).replace(0, 1), axis=0) * 1e6
        return np.log2(X + 1.0)

    def log(self, *a):
        pass


def main(run, data, params: Dict, mode: str = "all") -> None:
    p = {**DEFAULTS, **(params or {})}
    X, y, info = load_dataset(p["dataset"])
    validate_inputs(X, y)
    json.dump(info, open(run.path("results", "dataset.json"), "w"), indent=1)
    run.log("dataset", **info)
    er = GenericRun(data.cfg, counts=(info["scale"] == "read counts"))
    nj = run.cfg.get("n_jobs", 2); seed = run.cfg["seed"]
    folds = er.folds(y); cache = UnitCache(run)
    if mode in ("all", "run"):
        fixed = {}
        for k, (tr, te) in enumerate(folds):
            key = f"foldpanel_f{k}"
            if not cache.has(key):
                cache.put(key, SEL.make("six_stage", B=p["B"], n_jobs=nj, seed=seed).select(X.iloc[tr], y[tr], er))
            fixed[k] = cache.get(key)
        for k, (tr, te) in enumerate(folds):
            for r in range(p["R"]):
                key = f"unit_f{k}_r{r:02d}"
                if cache.has(key):
                    continue
                rng = np.random.default_rng(seed * 1000 + 100 * k + r)
                sub = stratified_subsample(tr, y, p["frac"], rng)
                Xs, ys, Xh, yh = X.iloc[sub], y[sub], X.iloc[te], y[te]
                out = {"fold": k, "rep": r, "train": sub, "test": te, "methods": {}}
                arms = [(m, SEL.make(m, k=p["k"], B=p["B"], n_jobs=nj,
                                     seed=(10_000 + 100 * k + r) if m == "random" else seed)) for m in p["methods"]]
                arms.append(("fixed", SEL.Fixed(fixed[k], n_jobs=nj, seed=seed)))
                for name, sel in arms:
                    t0 = time.time()
                    genes = sel.select(Xs, ys, er)
                    scores = sel.fit_predict(Xs, ys, Xh, genes, er)
                    thr = sel.inner_threshold(Xs, ys, genes, er)
                    out["methods"][name] = dict(genes=genes, scores=scores, thr=thr, info=dict(sel.info),
                                                auc=PR.auc(yh, scores), seconds=round(time.time() - t0, 1))
                cache.put(key, out)
                run.log("unit", fold=k + 1, rep=r, **{m: round(v["auc"], 3) for m, v in out["methods"].items()})
    if mode in ("all", "summarise"):
        s = summarise(run, SimpleNamespace(X=X, y=y), p)
        run.log("summary", identity_share=s["identity_share_six_stage"])
