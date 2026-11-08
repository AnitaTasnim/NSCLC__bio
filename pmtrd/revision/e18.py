"""E18 - correlation-aware stability of the saved replicate panels (saved panels plus expression).

Phi_C (Sechidis et al. 2019) = 1 - tr(C S) / tr(C Sigma0):
    Z        M x d selection indicators of the M replicate panels of one outer fold
    S        unbiased covariance of Z: S_fg = M/(M-1) (p_fg - p_f p_g)
    Sigma0   covariance of Z when every panel is a uniformly random subset of its size:
             diagonal kbar/d (1 - kbar/d); off-diagonal E[k(k-1)]/(d(d-1)) - (kbar/d)^2
    C        C_ff = 1; C_fg = |rho_fg| if |rho_fg| >= theta, else 0 (Spearman, outer training fold)
With C = I this is Nogueira's Phi. Matched overlap (defined here, descriptive): for panels A and B, the
fraction of genes of A that have a partner in B (the same gene, or |rho| >= theta), averaged both ways.

Plan: docs/plan_E8_E19.md, section E18.   Run:  python -m pmtrd.revision.e18
"""
from __future__ import annotations

import json
import shutil
from itertools import combinations
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .common import SAVED, SELECTORS, out_dir, path, saved, write_meta

THETAS = (0.3, 0.5, 0.7)
PRIMARY = 0.5


# ------------------------------------------------------------------ data on the selectors' scale
def load_log_scale(dataset: str):
    """Samples x genes on the log scale used for statistics, labels, and the outer folds of the saved run."""
    if dataset == "GSE89843":
        from ..config import load as load_cfg
        from ..engine import load
        data = load(load_cfg(path("configs", "base.json")))
        folds = data.run.folds()
        return data.run.logX(data.X), data.y, folds
    from sklearn.model_selection import StratifiedKFold
    from ..datasets import load as load_ds
    from ..experiments.replicates_generic import GenericRun
    cfg = json.load(open(path(SAVED[dataset], "config.json")))
    X, y, info = load_ds(cfg["params"]["dataset"])
    er = GenericRun({"seed": cfg["seed"]}, counts=(info["scale"] == "read counts"))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=cfg["seed"]).split(np.zeros(len(y)), y))
    return er.logX(X), np.asarray(y), folds


def spearman_scores(L: pd.DataFrame) -> np.ndarray:
    """Columns rank-transformed and standardised so that G.T @ G / n is the Spearman matrix."""
    R = rankdata(L.values, axis=0).astype(np.float32)
    R -= R.mean(0)
    sd = R.std(0)
    sd[sd == 0] = np.inf                       # constant genes: correlation 0 with everything
    return R / sd


def offdiag_sums(G: np.ndarray, thetas: Sequence[float], block: int = 1500) -> Dict[float, float]:
    """Sum over ordered pairs f != g of |rho_fg| 1{|rho_fg| >= theta}, for every theta."""
    n, d = G.shape
    tot = {t: 0.0 for t in thetas}
    for a in range(0, d, block):
        Rb = np.abs(G[:, a:a + block].T @ G) / n          # (b, d)
        idx = np.arange(a, min(a + block, d))
        Rb[np.arange(len(idx)), idx] = 0.0
        for t in thetas:
            tot[t] += float(Rb[Rb >= t].sum())
    return tot


# ------------------------------------------------------------------ stability measures
def phi_c(panels: List[List[str]], d: int, Csub: np.ndarray, genes: List[str], T: float) -> float:
    idx = {g: i for i, g in enumerate(genes)}
    M = len(panels)
    Z = np.zeros((M, len(genes)))
    for r, p in enumerate(panels):
        for g in set(p):
            Z[r, idx[g]] = 1
    S = np.cov(Z, rowvar=False, ddof=1)                    # unbiased, M/(M-1) (p_fg - p_f p_g)
    k = Z.sum(1)
    kbar, kk = k.mean(), np.mean(k * (k - 1))
    v0 = kbar / d * (1 - kbar / d)
    c0 = kk / (d * (d - 1)) - (kbar / d) ** 2
    num = float(np.sum(Csub * S))
    den = d * v0 + c0 * T
    return 1 - num / den


def matched_overlap(panels: List[List[str]], R: pd.DataFrame, theta: float) -> float:
    vals = []
    for a, b in combinations(panels, 2):
        ab = np.mean([(ga in set(b)) or (np.abs(R.loc[ga, b].values) >= theta).any() for ga in a])
        ba = np.mean([(gb in set(a)) or (np.abs(R.loc[a, gb].values) >= theta).any() for gb in b])
        vals.append((ab + ba) / 2)
    return float(np.mean(vals))


def main() -> None:
    od = out_dir("runs", "revision", "E18")
    rows = []
    for ds in SAVED:
        L, y, folds = load_log_scale(ds)
        d = L.shape[1]
        pn = saved(ds, "panels.csv")
        fold_info = {}
        for k, (tr, te) in enumerate(folds):
            G = spearman_scores(L.iloc[tr])
            T = offdiag_sums(G, THETAS)
            sel = sorted(set(";".join(pn[pn.fold == k + 1].genes.dropna()).split(";")) - {""})
            pos = [L.columns.get_loc(g) for g in sel]
            Rsel = pd.DataFrame((G[:, pos].T @ G[:, pos]) / G.shape[0], index=sel, columns=sel)
            fold_info[k + 1] = (T, Rsel)
            print(ds, "fold", k + 1, {t: round(v) for t, v in T.items()}, flush=True)
        for m in SELECTORS:
            if m not in set(pn.method):
                continue
            per_fold = {}
            for f, (T, Rsel) in fold_info.items():
                panels = [p.split(";") for p in pn[(pn.fold == f) & (pn.method == m)].sort_values("rep").genes]
                per_fold[f] = panels
            row = {"dataset": ds, "method": m, "d": d}
            for t in THETAS:
                est, jk = [], []
                for f, panels in per_fold.items():
                    T, Rsel = fold_info[f]
                    genes = sorted(set().union(*map(set, panels)))
                    Rg = Rsel.loc[genes, genes].values
                    C = np.where(np.abs(Rg) >= t, np.abs(Rg), 0.0)
                    np.fill_diagonal(C, 1.0)
                    est.append(phi_c(panels, d, C, genes, T[t]))
                row[f"phi_C_{t}"] = float(np.mean(est))
                # leave-one-replicate-out jackknife over folds (as for Phi)
                Rn = min(len(p) for p in per_fold.values())
                loo = []
                for r in range(Rn):
                    vals = []
                    for f, panels in per_fold.items():
                        T, Rsel = fold_info[f]
                        pp = [q for j, q in enumerate(panels) if j != r]
                        genes = sorted(set().union(*map(set, pp)))
                        Rg = Rsel.loc[genes, genes].values
                        C = np.where(np.abs(Rg) >= t, np.abs(Rg), 0.0)
                        np.fill_diagonal(C, 1.0)
                        vals.append(phi_c(pp, d, C, genes, T[t]))
                    loo.append(np.mean(vals))
                se = float(np.sqrt((Rn - 1) / Rn * np.sum((np.array(loo) - np.mean(loo)) ** 2)))
                row[f"phi_C_{t}_lo"], row[f"phi_C_{t}_hi"] = row[f"phi_C_{t}"] - 1.96 * se, row[f"phi_C_{t}"] + 1.96 * se
                row[f"matched_overlap_{t}"] = float(np.mean([matched_overlap(p, fold_info[f][1], t)
                                                            for f, p in per_fold.items()]))
            # check: C = I reproduces Nogueira's Phi of the saved run
            ident = []
            for f, panels in per_fold.items():
                genes = sorted(set().union(*map(set, panels)))
                ident.append(phi_c(panels, d, np.eye(len(genes)), genes, 0.0))
            row["phi_identity"] = float(np.mean(ident))
            st = saved(ds, "stability.csv").set_index("method")
            row["phi_published"] = float(st.phi.get(m, np.nan))
            row["pairwise_jaccard"] = float(st.pairwise_jaccard.get(m, np.nan))
            rows.append(row)
            print(ds, m, round(row["phi_published"], 3), round(row[f"phi_C_{PRIMARY}"], 3), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path(od, "E18.csv"), index=False)
    shutil.copy(path(od, "E18.csv"), path("runs", "export", "E18.csv"))
    write_meta(od, experiment="E18", thetas=THETAS, primary=PRIMARY, saved_runs=SAVED)


if __name__ == "__main__":
    main()
