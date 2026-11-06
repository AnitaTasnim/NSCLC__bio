"""Bridge to the verified analysis engine (``engine/``), which is imported unchanged."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import ROOT, engine_config

_ENGINE = os.path.join(ROOT, "engine")
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

import tep_analysis as T  # noqa: E402
import selection_pipeline as S  # noqa: E402


@dataclass
class Dataset:
    X: pd.DataFrame            # samples x genes, processed scale (as used by the selection)
    y: np.ndarray              # 0 healthy, 1 NSCLC
    site: pd.Series
    run: Any                   # engine Run object (classifier factory, folds, log transform)
    cfg: Dict[str, Any]        # engine configuration


def validate_inputs(X: pd.DataFrame, y: np.ndarray) -> None:
    if X.shape[0] != len(y):
        raise ValueError(f"{X.shape[0]} samples in X but {len(y)} labels")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("labels must be 0/1")
    if X.isna().any().any():
        raise ValueError("expression matrix contains missing values")
    if (X.values < 0).any():
        raise ValueError("expression matrix contains negative values (expected counts or processed counts)")
    if X.columns.duplicated().any():
        raise ValueError("duplicate gene names in the expression matrix")
    if min(np.bincount(y)) < 10:
        raise ValueError("fewer than 10 samples in one class")


def load(cfg: Dict[str, Any]) -> Dataset:
    ec = T.load_config(engine_config(cfg))
    d = T.load_data(ec)
    run = T.Run(ec, d)
    y = d.y.values.astype(int)
    validate_inputs(d.X, y)
    return Dataset(X=d.X, y=y, site=d.site, run=run, cfg=ec)
