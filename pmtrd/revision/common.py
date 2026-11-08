"""Shared helpers for the revision experiments (E8-E11, E15-E19): saved replicate studies and outputs."""
from __future__ import annotations

import json
import os
import platform
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import ROOT

# The four saved replicate studies (paper's folds, R = 10, f = 0.8, same settings).
SAVED = {
    "GSE89843": "runs/replicates/20261105T021622Z_f933d1c2",
    "Gordon": "runs/generalisation/gordon_lung/replicates_generic/20261105T081127Z_53be0d9c",
    "GSE68086": "runs/generalisation/gse68086_pancancer/replicates_generic/20261105T082557Z_91104506",
    "Singh": "runs/generalisation/singh_prostate/replicates_generic/20261105T075820Z_d4d93e0f",
}
SELECTORS = ["six_stage", "mw_topk", "l1_topk", "stability", "mrmr_ifs_svm", "random"]
XGB_SELECTORS = ["six_stage", "mw_topk", "l1_topk", "stability", "random"]


def path(*p: str) -> str:
    return os.path.join(ROOT, *p)


def saved(dataset: str, name: str) -> pd.DataFrame:
    return pd.read_csv(path(SAVED[dataset], "results", name))


def fold_matrices(dataset: str, method: str) -> List[Dict]:
    """Per outer fold: scores S (R x n), labels L (R x n), thresholds (R), y (n), sample ids."""
    sc = saved(dataset, "scores.csv")
    un = saved(dataset, "units.csv")
    sc = sc[sc.method == method]
    out = []
    for k in sorted(sc.fold.unique()):
        d = sc[sc.fold == k]
        S = d.pivot(index="rep", columns="sample", values="score").sort_index()
        L = d.pivot(index="rep", columns="sample", values="label").loc[S.index, S.columns]
        y = d.drop_duplicates("sample").set_index("sample").y.loc[S.columns].values
        thr = un[(un.fold == k) & (un.method == method)].set_index("rep").threshold.loc[S.index].values
        out.append(dict(fold=int(k), S=S.values, L=L.values.astype(int), thr=thr, y=y, samples=list(S.columns)))
    return out


def out_dir(*p: str) -> str:
    d = path(*p)
    os.makedirs(d, exist_ok=True)
    return d


def write_meta(d: str, **extra) -> None:
    import importlib
    vers = {}
    for m in ("numpy", "scipy", "pandas", "sklearn", "xgboost", "statsmodels"):
        try:
            vers[m] = importlib.import_module(m).__version__
        except Exception:
            vers[m] = None
    json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(),
               "platform": platform.platform(), "cpu_count": os.cpu_count(), "packages": vers, **extra},
              open(os.path.join(d, "metadata.json"), "w"), indent=1, default=str)


def read_wide_parquet(p: str, chunk: int = 2000) -> "pd.DataFrame":
    """Read a wide samples x genes parquet column-chunk by column-chunk into one float64 array (pandas'
    default conversion of 28,000 columns needs about 4 GB of temporary memory)."""
    import pyarrow.parquet as pq
    f = pq.ParquetFile(p)
    meta = f.schema_arrow.pandas_metadata or {}
    idx = [c for c in meta.get("index_columns", []) if isinstance(c, str)]
    cols = [n for n in f.schema_arrow.names if n not in idx]
    index = [str(v) for v in f.read(columns=idx).column(0).to_pylist()] if idx else None
    out = np.empty((f.metadata.num_rows, len(cols)), dtype=np.float64)
    for a in range(0, len(cols), chunk):
        t = f.read(columns=cols[a:a + chunk])
        for j, c in enumerate(t.columns):
            out[:, a + j] = c.to_numpy(zero_copy_only=False)
    return pd.DataFrame(out, index=index, columns=cols)


def export_cache(cache_dir: str, out_path: str) -> int:
    """Raw unit outputs of a resumable cache (one pickle per unit) written as JSON lines, so that the raw
    results are kept in a readable, version-controlled form (arrays become lists)."""
    import gzip
    import pickle

    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v

    n = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wt") as fh:
        for f in sorted(os.listdir(cache_dir)):
            if f.endswith(".pkl"):
                u = pickle.load(open(os.path.join(cache_dir, f), "rb"))
                fh.write(json.dumps({"unit": f[:-4], **conv(u if isinstance(u, dict) else {"value": u})}) + "\n")
                n += 1
    return n
