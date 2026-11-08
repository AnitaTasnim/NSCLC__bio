"""Resumable selection units shared by E8-E11, E15 and E17: one pickle per unit, so a stopped run resumes."""
from __future__ import annotations

import json
import os
import pickle
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .. import selectors as SEL
from ..experiments.common import stratified_subsample
from ..metrics import predictive as PR
from .common import SAVED, path


class Cache:
    def __init__(self, d: str):
        self.d = d
        os.makedirs(d, exist_ok=True)

    def _p(self, key: str) -> str:
        return os.path.join(self.d, key + ".pkl")

    def has(self, key: str) -> bool:
        return os.path.exists(self._p(key))

    def get(self, key: str):
        return pickle.load(open(self._p(key), "rb"))

    def put(self, key: str, value) -> None:
        tmp = self._p(key) + ".tmp"
        pickle.dump(value, open(tmp, "wb"))
        os.replace(tmp, self._p(key))


# ------------------------------------------------------------------ data sets
def load_discovery(n_jobs: int = 1):
    """GSE89843 exactly as the engine's loader builds it (same samples, order, labels, sites, zero-count
    filter and Run object), but reading the wide parquet column-chunk-wise: the engine's loader needs about
    4 GB of temporary memory, which two concurrent jobs on this machine cannot afford."""
    from types import SimpleNamespace
    from ..config import engine_config, load as load_cfg
    from ..engine import T
    from .common import read_wide_parquet
    cfg = load_cfg(path("configs", "base.json"))
    cfg["n_jobs"] = n_jobs
    ec = T.load_config(engine_config(cfg))
    meta = T._load_meta(ec["discovery_meta"], ec)
    X = read_wide_parquet(ec["discovery_expr"])
    common = [s for s in meta.index if s in X.index]
    X, meta = X.loc[common].astype(float), meta.loc[common]
    lab = T._pick_col(meta, ec.get("label_col"), ["label", "y", "condition", "group", "class", "disease",
                                                    "patient group", "status"])
    y = T.parse_labels(meta[lab]).rename("y")
    tc = T._pick_col(meta, ec.get("title_col"), ["title", "sample_title", "geo_title"])
    site = meta[tc].map(lambda t: T.site_from_title(t, ec["site_regex"])).rename("site")
    X = X.loc[:, (X != 0).any(axis=0)]
    d = T.TEPData(X=X, y=y, site=site, meta=meta)
    run = T.Run(ec, d)
    return SimpleNamespace(X=d.X, y=y.values.astype(int), site=site, run=run, cfg=ec)


def load_dataset(name: str, n_jobs: int = 1):
    """(X on the processed scale used by the selectors, y, engine-like run object, counts flag)."""
    if name == "GSE89843":
        data = load_discovery(n_jobs)
        return data.X, data.y, data.run, True
    from ..datasets import load as load_ds
    from ..experiments.replicates_generic import GenericRun
    from ..engine import T
    cfg = json.load(open(path(SAVED[name], "config.json")))
    X, y, info = load_ds(cfg["params"]["dataset"])
    ec = T.load_config(path("engine", "config.json"))
    ec["seed"] = cfg["seed"]
    er = GenericRun(ec, counts=(info["scale"] == "read counts"))
    return X, np.asarray(y), er, info["scale"] == "read counts"


def outer_folds(name: str, y: np.ndarray, er, seed: int) -> List:
    if name == "GSE89843":
        return er.folds(seed=seed)
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


# ------------------------------------------------------------------ one selection unit
def run_unit(X: pd.DataFrame, y: np.ndarray, er, train: np.ndarray, test: Optional[np.ndarray], selector: str,
             k: int = 27, B: int = 500, seed: int = 42, n_jobs: int = 1, genes: Optional[Sequence[str]] = None,
             threshold: bool = True, Xtest: Optional[pd.DataFrame] = None) -> Dict:
    """Select on X[train] (unless genes are given), train the selector's classifier on X[train], score the test
    samples (X[test], or Xtest), and learn a threshold by inner CV on the training samples."""
    t0 = time.time()
    sel = SEL.Fixed(genes, n_jobs=n_jobs, seed=seed) if genes is not None else \
        SEL.make(selector, k=k, B=B, n_jobs=n_jobs, seed=seed)
    Xs, ys = X.iloc[train], y[train]
    g = sel.select(Xs, ys, er)
    Xh = Xtest if Xtest is not None else X.iloc[test]
    scores = sel.fit_predict(Xs, ys, Xh, g, er)
    thr = sel.inner_threshold(Xs, ys, g, er) if threshold else np.nan
    out = dict(genes=list(g), scores=np.asarray(scores, float), thr=float(thr), info=dict(sel.info),
               seconds=round(time.time() - t0, 1))
    if test is not None and Xtest is None:
        out["auc"] = PR.auc(y[test], scores)
    return out


def subsample(train: np.ndarray, y: np.ndarray, seed: int, k: int, r: int, frac: float = 0.8) -> np.ndarray:
    """The paper's subsample scheme: rng(seed * 1000 + 100 k + r)."""
    return stratified_subsample(train, y, frac, np.random.default_rng(seed * 1000 + 100 * k + r))
