"""Additional public datasets for testing whether the reliability findings generalise.

Every loader returns (X, y, info): X samples x features (non-negative, no missing values), y 0/1, and a
dict describing the preprocessing. All preprocessing is label-free and fixed in advance.

    gse68086      GEO GSE68086 (Best et al. 2015): platelet RNA-seq read counts of healthy donors and
                  patients with six tumour types. Healthy = sample names starting with "HD-" or "Control-";
                  cancer = every other sample except the five named "Type-Unknown-*", which are excluded.
                  Genes with a non-zero count in fewer than 10% of samples are dropped. Raw counts are used,
                  as for the discovery cohort.
    datamicroarray  classic Affymetrix benchmark sets distributed with the R package datamicroarray
                  (github.com/ramhiser/datamicroarray): Singh et al. 2002 (prostate tumour vs normal) and
                  Gordon et al. 2002 (malignant pleural mesothelioma vs lung adenocarcinoma). Linear-scale
                  intensities; values below 1 are set to 1 (MAS-type intensities can be negative).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import resolve


def gse68086(path: str, min_frac: float = 0.10) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    M = pd.read_csv(resolve(path), sep="\t", index_col=0, low_memory=False)
    names = list(M.columns)
    healthy = [s for s in names if s.startswith(("HD-", "Control-"))]
    unknown = [s for s in names if s.startswith("Type-Unknown")]
    keep = [s for s in names if s not in unknown]
    X = M[keep].T.astype(float)
    X = X.loc[:, (X > 0).mean(axis=0) >= min_frac]
    y = np.array([0 if s in healthy else 1 for s in X.index])
    info = dict(source="GEO GSE68086", task="cancer (six tumour types) vs healthy donors", n=len(y),
                positives=int(y.sum()), negatives=int((1 - y).sum()), features=X.shape[1],
                excluded=len(unknown), scale="read counts")
    return X, y, info


def datamicroarray(path: str, name: str, positive: str, floor: float = 1.0) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    import rdata
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conv = rdata.conversion.convert(rdata.parser.parse_file(resolve(path)))
    d = conv[name]
    x = d["x"]
    if hasattr(x, "dims"):                      # xarray (feature names in the second coordinate)
        cols = [str(c) for c in x.coords[x.dims[1]].values] if x.dims[1] in x.coords else None
        X = pd.DataFrame(np.asarray(x, dtype=float), columns=cols)
    else:
        X = pd.DataFrame(np.asarray(x, dtype=float), columns=[str(c) for c in x.columns])
    X.columns = [f"F{c}" if str(c)[:1].isdigit() else str(c) for c in X.columns]
    X.index = [f"{name}_{i + 1}" for i in range(len(X))]
    labels = pd.Series(np.asarray(d["y"]).astype(str))
    if positive not in set(labels):
        raise ValueError(f"positive class {positive!r} not in {sorted(set(labels))}")
    y = (labels == positive).astype(int).values
    n_floor = int((X.values < floor).sum())
    X = X.clip(lower=floor)
    X = X.loc[:, X.var(axis=0) > 0]
    info = dict(source=f"datamicroarray::{name}", task=f"{positive} vs {sorted(set(labels) - {positive})[0]}",
                n=len(y), positives=int(y.sum()), negatives=int((1 - y).sum()), features=X.shape[1],
                floored_values=n_floor, scale="linear intensities")
    return X, y, info


def load(spec: Dict) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    kind = spec["loader"]
    if kind == "gse68086":
        return gse68086(spec["path"], spec.get("min_frac", 0.10))
    if kind == "datamicroarray":
        return datamicroarray(spec["path"], spec["name"], spec["positive"], spec.get("floor", 1.0))
    raise ValueError(f"unknown loader {kind!r}")
