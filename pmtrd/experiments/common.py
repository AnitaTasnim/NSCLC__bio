"""Helpers shared by the experiments."""
from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..config import resolve


def stratified_subsample(idx: np.ndarray, y: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Stratified subsample without replacement of the positions ``idx`` (labels y[idx])."""
    out = []
    for c in (0, 1):
        pos = idx[y[idx] == c]
        out.append(rng.choice(pos, int(round(frac * len(pos))), replace=False))
    return np.sort(np.concatenate(out))


def reference_fold_panels(path: str) -> Dict[int, List[str]]:
    """Four-term nested-CV fold panels of the paper (``R4_nested_folds.csv``), keyed by fold 0..K-1."""
    df = pd.read_csv(resolve(path))
    return {int(r.fold) - 1: r.genes.split(";") for r in df.itertuples()}


def check_folds(folds, path: str, samples: pd.Index) -> None:
    """Verify that the engine's outer folds are the folds used in the paper (``R8_outer_folds.csv``)."""
    ref = pd.read_csv(resolve(path))
    col = "fold" if "fold" in ref.columns else ref.columns[-1]
    sid = ref.columns[0]
    ref = ref.set_index(sid)[col]
    for k, (_, te) in enumerate(folds):
        got = set(samples[te]); want = set(ref.index[ref.values == ref.values.min() + k])
        if got != want:
            raise RuntimeError(f"outer fold {k + 1} differs from the paper's fold assignment")


class UnitCache:
    """One pickle per experimental unit, so that an interrupted run resumes where it stopped."""

    def __init__(self, run):
        self.dir = run.path("cache")

    def _p(self, key: str) -> str:
        return os.path.join(self.dir, f"{key}.pkl")

    def has(self, key: str) -> bool:
        return os.path.exists(self._p(key))

    def get(self, key: str) -> Any:
        return pickle.load(open(self._p(key), "rb"))

    def put(self, key: str, value: Any) -> None:
        tmp = self._p(key) + ".tmp"
        pickle.dump(value, open(tmp, "wb")); os.replace(tmp, self._p(key))

    def all(self) -> Dict[str, Any]:
        return {f[:-4]: pickle.load(open(os.path.join(self.dir, f), "rb"))
                for f in sorted(os.listdir(self.dir)) if f.endswith(".pkl")}
