"""E19 - does ambiguity depend on the number of replicates R? (saved scores and thresholds)

Plan: docs/plan_E8_E19.md, section E19.   Run:  python -m pmtrd.revision.e19
"""
from __future__ import annotations

import itertools
import shutil

import numpy as np
import pandas as pd

from .common import SAVED, SELECTORS, fold_matrices, out_dir, path, saved, write_meta

SEED = 19
B = 1000


def pair_tensor(L: np.ndarray) -> np.ndarray:
    """d[r, s, i] = 1 if replicates r and s label sample i differently."""
    return (L[:, None, :] != L[None, :, :]).astype(float)


def weighted_pair_mean(d: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    """Mean of d over ordered pairs of distinct replicates (weights v_r v_s) and samples (weights w_i)."""
    M = d @ w                                   # (R, R)
    V = np.outer(v, v)
    np.fill_diagonal(V, 0.0)
    return float((V * M).sum() / (V.sum() * w.sum()))


def ambiguity(L: np.ndarray) -> float:
    return float(np.mean(L.min(0) != L.max(0)))


def ambiguity_curve(L: np.ndarray, rng: np.random.Generator, max_subsets: int = 200) -> dict:
    R = L.shape[0]
    out = {}
    for m in range(2, R + 1):
        combos = list(itertools.combinations(range(R), m))
        if len(combos) > max_subsets:
            idx = rng.choice(len(combos), max_subsets, replace=False)
            combos = [combos[i] for i in idx]
        out[m] = float(np.mean([ambiguity(L[list(c)]) for c in combos]))
    return out


def ensemble_disagreement(S: np.ndarray, thr: np.ndarray) -> float:
    R = S.shape[0]
    vals = []
    for c in itertools.combinations(range(R), R // 2):
        if 0 not in c:                          # each unordered split once
            continue
        a, b = list(c), [r for r in range(R) if r not in c]
        la = S[a].mean(0) >= thr[a].mean()
        lb = S[b].mean(0) >= thr[b].mean()
        vals.append(np.mean(la != lb))
    return float(np.mean(vals))


def analyse(dataset: str, method: str, rng: np.random.Generator) -> dict:
    F = fold_matrices(dataset, method)
    D, Dc, A10, ens, curves = [], [], [], [], []
    tensors = []
    for f in F:
        d = pair_tensor(f["L"])
        Lc = (f["S"] >= np.median(f["thr"])).astype(int)
        dc = pair_tensor(Lc)
        tensors.append((d, dc, f["L"]))
        R, n = f["L"].shape
        D.append(weighted_pair_mean(d, np.ones(R), np.ones(n)))
        Dc.append(weighted_pair_mean(dc, np.ones(R), np.ones(n)))
        A10.append(ambiguity(f["L"]))
        ens.append(ensemble_disagreement(f["S"], f["thr"]))
        curves.append(ambiguity_curve(f["L"], rng))
    est = {"D": np.mean(D), "D_c": np.mean(Dc), "A10": np.mean(A10), "ensemble_disagreement": np.mean(ens)}
    est["threshold_share"] = 1 - est["D_c"] / est["D"] if est["D"] > 0 else np.nan
    # pigeonhole bootstrap for D, D_c and the threshold share; patients only for A(10) (plan amendment)
    bs = {"D": [], "D_c": [], "threshold_share": [], "A10": []}
    for _ in range(B):
        Db, Dcb, Ab = [], [], []
        for d, dc, L in tensors:
            R, n = L.shape
            v = rng.multinomial(R, np.full(R, 1 / R)).astype(float)
            while (v > 0).sum() < 2:
                v = rng.multinomial(R, np.full(R, 1 / R)).astype(float)
            w = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
            Db.append(weighted_pair_mean(d, v, w))
            Dcb.append(weighted_pair_mean(dc, v, w))
            Ab.append(float(np.sum(w * (L.min(0) != L.max(0))) / w.sum()))
        bs["D"].append(np.mean(Db)); bs["D_c"].append(np.mean(Dcb)); bs["A10"].append(np.mean(Ab))
        bs["threshold_share"].append(1 - np.mean(Dcb) / np.mean(Db) if np.mean(Db) > 0 else np.nan)
    row = {"dataset": dataset, "method": method, **{k: float(v) for k, v in est.items()}}
    for k, vals in bs.items():
        row[f"{k}_lo"], row[f"{k}_hi"] = np.nanquantile(vals, [0.025, 0.975])
    curve = pd.DataFrame(curves).mean(0)
    for m, a in curve.items():
        row[f"A{m}"] = float(a)
    return row


def main() -> None:
    rng = np.random.default_rng(SEED)
    od = out_dir("runs", "revision", "E19")
    rows = []
    for ds in SAVED:
        agr = saved(ds, "agreement.csv").set_index("method")
        for m in SELECTORS + ["fixed"]:
            if m not in set(saved(ds, "units.csv").method):
                continue
            r = analyse(ds, m, rng)
            r["published_ambiguity"] = float(agr.ambiguity.get(m, np.nan)) if m in agr.index else np.nan
            r["published_discrepancy"] = float(agr.discrepancy.get(m, np.nan)) if m in agr.index else np.nan
            rows.append(r)
            print(ds, m, round(r["D"], 4), round(r["A10"], 4), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path(od, "E19.csv"), index=False)
    shutil.copy(path(od, "E19.csv"), path("runs", "export", "E19.csv"))
    write_meta(od, experiment="E19", seed=SEED, bootstrap_B=B, saved_runs=SAVED)


if __name__ == "__main__":
    main()
