"""Simulation study E1-E4 of the crossed pick-freeze estimator (docs/crossed_simulation.md).

One *unit* of work is either the reference truth for one simulated training set T (R = 200 draws, sparse
cyclic crossed design, M = 5 seeds on 40 cells) or one estimator replicate (R draws, complete crossed grid,
fresh held-out patients). Units are independent, deterministic given the root seed, written as JSON files
and skipped when they already exist, so a run can be interrupted and resumed.

Usage:
    python -m pmtrd.crossed.simulate --config configs/crossed_sim.json --out runs/crossed_sim --worker 0 --workers 2
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.stats import norm

from .estimator import STAT_NAMES, CrossedData, bootstrap, components, eq3_psi, percentile_ci


# ----------------------------------------------------------------------------------------------- seeds
def seed_for(*keys: int) -> int:
    """Deterministic 31-bit seed from integer keys (independent streams via SeedSequence)."""
    return int(np.random.SeedSequence([int(k) % (2 ** 32) for k in keys]).generate_state(1)[0] % (2 ** 31 - 1))


def stable_hash(obj) -> int:
    return zlib.crc32(json.dumps(obj, sort_keys=True).encode())


# ------------------------------------------------------------------------------------------- generator
@dataclass
class BlockGaussian:
    """Two Gaussian classes; p features in blocks of size `block` with within-block correlation rho; the
    first `n_signal` features (whole blocks) are shifted by delta in class 1. delta is set so that the AUC of
    the Bayes-optimal linear rule, Phi(Delta / sqrt 2) with Delta the Mahalanobis distance, equals auc."""

    p: int
    rho: float
    auc: float
    block: int = 10
    n_signal: int = 20
    prevalence: float = 0.6

    def __post_init__(self) -> None:
        if self.p % self.block or self.n_signal % self.block:
            raise ValueError("p and n_signal must be multiples of block")
        Delta = np.sqrt(2) * norm.ppf(self.auc)
        per_block = self.block / (1 + (self.block - 1) * self.rho)        # 1' Sigma_b^-1 1
        self.delta = Delta / np.sqrt((self.n_signal // self.block) * per_block)

    def sample(self, n: int, rng: np.random.Generator, stratified: bool = True):
        n1 = int(round(self.prevalence * n)) if stratified else rng.binomial(n, self.prevalence)
        y = np.r_[np.ones(n1, int), np.zeros(n - n1, int)]
        nb = self.p // self.block
        common = rng.standard_normal((n, nb, 1))
        X = (np.sqrt(self.rho) * common + np.sqrt(1 - self.rho) * rng.standard_normal((n, nb, self.block)))
        X = X.reshape(n, self.p)
        X[y == 1, : self.n_signal] += self.delta
        return X, y


# ------------------------------------------------------------------------------------------- selectors
def _auc_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(X, 0, kind="stable"), 0, kind="stable") + 1.0
    n1 = y.sum()
    U = ranks[y == 1].sum(0) - n1 * (n1 + 1) / 2
    return U / (n1 * (len(y) - n1))


def select(name: str, X: np.ndarray, y: np.ndarray, k: int, xi: int) -> np.ndarray:
    """Panel of k feature indices. mw: largest |AUC - 0.5| (deterministic). l1: L1-logistic on standardised
    features at the strongest penalty that keeps at least k features, top k by |beta| (liblinear seeded by
    xi). random: k features drawn with seed xi, independent of the data."""
    p = X.shape[1]
    if name == "fixed":                                   # control: the same panel whatever the data
        return np.arange(k)
    if name == "random":
        return np.sort(np.random.default_rng(xi).choice(p, k, replace=False))
    if name == "mw":
        s = np.abs(_auc_scores(X, y) - 0.5)
        return np.sort(np.argsort(-s, kind="stable")[:k])
    if name == "l1":
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import l1_min_c
        mu, sd = X.mean(0), X.std(0) + 1e-12
        Xs = (X - mu) / sd
        C = l1_min_c(Xs, y, loss="log") * 1.05
        coef = np.zeros(p)
        for _ in range(60):
            m = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C, random_state=xi, max_iter=500, tol=1e-4)
            coef = m.fit(Xs, y).coef_.ravel()
            if (coef != 0).sum() >= k:
                break
            C *= 1.25
        order = np.lexsort((np.arange(p), -np.abs(coef)))
        return np.sort(order[:k])
    raise ValueError(name)


# ------------------------------------------------------------------------------------------ classifiers
def fit_logodds(clf: Dict, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, seed: int) -> np.ndarray:
    """Held-out log-odds (decision function / output margin, no clipping)."""
    if clf["name"] == "logistic":
        from sklearn.linear_model import LogisticRegression
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        m = LogisticRegression(C=clf.get("C", 1.0), max_iter=2000).fit((Xtr - mu) / sd, ytr)
        return m.decision_function((Xte - mu) / sd)
    if clf["name"] == "xgboost":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=clf["n_estimators"], learning_rate=clf["learning_rate"],
                              max_depth=clf["max_depth"], subsample=clf["subsample"],
                              colsample_bytree=clf["colsample_bytree"], random_state=seed, n_jobs=1,
                              tree_method="hist", device=clf.get("device", "cpu"), eval_metric="logloss")
        m.fit(Xtr, ytr)
        return m.predict(Xte, output_margin=True)
    raise ValueError(clf["name"])


def stratified_subsample(y: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    idx = []
    for c in (0, 1):
        pool = np.flatnonzero(y == c)
        idx.append(rng.choice(pool, int(round(frac * len(pool))), replace=False))
    return np.sort(np.concatenate(idx))


# ------------------------------------------------------------------------------------------- the study
class Setting:
    """One data configuration x selector x classifier, with seeds derived from the configuration itself."""

    def __init__(self, cfg: Dict, data: Dict, selector: str, clf_name: str):
        self.cfg, self.data, self.selector = cfg, data, selector
        self.clf = {"name": clf_name, **cfg["classifiers"][clf_name]}
        self.key = stable_hash({**{k: data[k] for k in ("n", "p", "rho", "auc", "f")}, "root": cfg.get("root_seed", 0)})
        self.gen = BlockGaussian(p=data["p"], rho=data["rho"], auc=data["auc"], block=cfg["block"],
                                 n_signal=cfg["n_signal"], prevalence=cfg["prevalence"])
        self.sel_code = {"mw": 1, "l1": 2, "random": 3, "fixed": 4}[selector]
        self.clf_code = {"logistic": 1, "xgboost": 2}[clf_name]
        self.deterministic = clf_name == "logistic"

    def T(self, t: int):
        return self.gen.sample(self.data["n"], np.random.default_rng(seed_for(self.key, t, 0)))

    def draw(self, t: int, y: np.ndarray, d: int):
        S = stratified_subsample(y, self.data["f"], np.random.default_rng(seed_for(self.key, t, 1, d)))
        return S, seed_for(self.key, t, 2, self.sel_code, d)


class PanelCache:
    """Panels per (setting data, selector, T) shared by both classifiers; saved as .npz."""

    def __init__(self, path: str):
        self.path = path
        self.panels: Dict[int, np.ndarray] = {}
        if os.path.exists(path):
            z = np.load(path)
            self.panels = {int(k): z[k] for k in z.files}
        self.dirty = False

    def get(self, d: int, make):
        if d not in self.panels:
            self.panels[d] = make()
            self.dirty = True
        return self.panels[d]

    def save(self) -> None:
        if self.dirty:
            tmp = self.path + ".tmp.npz"
            np.savez(tmp, **{str(k): v for k, v in self.panels.items()})
            os.replace(tmp, self.path)
            self.dirty = False


def build_design(st: Setting, t: int, X, y, draw_ids: Sequence[int], cells: np.ndarray, seed_cells: np.ndarray,
                 M: int, Xh: np.ndarray, cache: PanelCache, tag: int) -> tuple:
    """Fit every requested cell and return (CrossedData, n_fits). cells: (n, 2) off-diagonal (r, q) pairs in
    local draw indices; the diagonal (tied pipeline) is always fitted; seed_cells get M seeds each; Eq. 3
    arms use the fixed seed m0 and the panel selected on the whole of T."""
    R = len(draw_ids)
    k = st.cfg["k"]
    subs, panels = [], []
    for d in draw_ids:
        S, xi = st.draw(t, y, d)
        subs.append(S)
        panels.append(cache.get(d, lambda S=S, xi=xi: select(st.selector, X[S], y[S], k, xi)))
    P0 = cache.get(-1, lambda: select(st.selector, X, y, k, seed_for(st.key, t, 3, st.sel_code)))
    grid = np.full((R, R, Xh.shape[0]), np.nan)
    n_fits = 0

    def fit(r, q, m):
        nonlocal n_fits
        n_fits += 1
        seed = seed_for(st.key, t, 6, st.sel_code, st.clf_code, tag, draw_ids[r], draw_ids[q], m)
        S, Q = subs[r], panels[q]
        return fit_logodds(st.clf, X[S][:, Q], y[S], Xh[:, Q], seed)

    for r, q in cells:
        grid[r, q] = fit(r, q, 0)
    for r in range(R):
        grid[r, r] = fit(r, r, 0)
    seed_vals = np.empty((len(seed_cells), M, Xh.shape[0]))
    for i, (r, q) in enumerate(seed_cells):
        seed_vals[i, 0] = grid[r, q]
        for m in range(1, M):
            seed_vals[i, m] = grid[r, q] if st.deterministic else fit(r, q, m)
    m0 = st.cfg["fixed_seed"]

    def fit_fixed_seed(r, Q):
        nonlocal n_fits
        n_fits += 1
        S = subs[r]
        return fit_logodds(st.clf, X[S][:, Q], y[S], Xh[:, Q], m0)

    tied_fixed = np.array([grid[r, r] if st.deterministic else fit_fixed_seed(r, panels[r]) for r in range(R)])
    fixed = np.array([fit_fixed_seed(r, P0) for r in range(R)])
    data = CrossedData(grid=grid.astype(np.float32), seed_cells=np.asarray(seed_cells).reshape(-1, 2),
                       seed_vals=seed_vals, eq3_tied=tied_fixed, eq3_fixed=fixed)
    return data, n_fits


def summarise(data: CrossedData) -> Dict:
    c = components(data, chunk=100)
    sh = c.shares()
    return {"V_R": c.V_R, "D_col": c.D_col, "D_row": c.D_row, "D_both": c.D_both, "D_seed": c.D_seed,
            "s2": c.s2, "share": {k: sh[k] for k in STAT_NAMES[:-1]}, "eq3": eq3_psi(data)}


def run_truth(st: Setting, t: int, cache: PanelCache) -> Dict:
    tc = st.cfg["truth"]
    X, y = st.T(t)
    Xh, _ = st.gen.sample(tc["n_patients"], np.random.default_rng(seed_for(st.key, t, 4)), stratified=False)
    R, D = tc["R"], tc["offsets"][st.clf["name"]]
    cells = np.array([(r, (r + o) % R) for o in range(1, D + 1) for r in range(R)])
    seed_cells = cells[: tc["seed_cells"]]
    data, n_fits = build_design(st, t, X, y, list(range(R)), cells, seed_cells, tc["M"], Xh, cache, tag=0)
    out = summarise(data)
    # grouped jackknife over draws (G groups): Monte Carlo SE of the truth itself
    G = tc["jackknife_groups"]
    groups = np.array_split(np.arange(R), G)
    jk = []
    for g in groups:
        keep = np.ones(R, bool)
        keep[g] = False
        v = keep.astype(float)
        c = components(data, v=v, chunk=100)
        sh = c.shares()
        jk.append([sh[k] for k in STAT_NAMES[:-1]] + [eq3_psi(data, v=v)])
    jk = np.array(jk)
    out["mc_se"] = dict(zip(STAT_NAMES, np.sqrt((G - 1) / G * ((jk - jk.mean(0)) ** 2).sum(0))))
    out["n_fits"] = n_fits
    return out


def run_rep(st: Setting, t: int, R: int, rep: int, cache: PanelCache, rng_boot: np.random.Generator) -> Dict:
    X, y = st.T(t)
    Xh, _ = st.gen.sample(st.data["n"], np.random.default_rng(seed_for(st.key, t, 5, R, rep)), stratified=False)
    base = st.cfg["rep_draw_base"][str(R)]
    ids = [base + rep * R + j for j in range(R)]
    cells = np.argwhere(~np.eye(R, dtype=bool))
    seed_cells = np.array([(r, (r + 1) % R) for r in range(R)])
    data, n_fits = build_design(st, t, X, y, ids, cells, seed_cells, 2, Xh, cache, tag=R)
    out = summarise(data)
    B = st.cfg["bootstrap_B"]
    for mode in ("pigeonhole", "draws", "patients"):
        reps = bootstrap(data, B, rng_boot, mode)
        ci = percentile_ci(reps)
        out[f"ci_{mode}"] = {k: [float(ci[0, j]), float(ci[1, j])] for j, k in enumerate(STAT_NAMES)}
    out["n_fits"] = n_fits
    return out


# ------------------------------------------------------------------------------------------- planning
def data_configs(cfg: Dict) -> Dict[str, Dict]:
    """Base configuration plus one-factor-at-a-time variations (the fractional design)."""
    base = cfg["base"]
    out = {"base": dict(base)}
    for fac, values in cfg["variations"].items():
        for v in values:
            d = dict(base)
            d[fac] = v
            out[f"{fac}={v}"] = d
    return out


def plan_units(cfg: Dict) -> List[Dict]:
    units = []
    for name, data in data_configs(cfg).items():
        for clf_name, spec in cfg["design"].items():
            if name not in spec["configs"] and spec["configs"] != "all":
                continue
            nT = spec["n_T_p10000"] if data["p"] >= 10000 else spec["n_T"]
            Rs = spec["R_p10000"] if data["p"] >= 10000 else (spec["R_base"] if name == "base" else spec["R_other"])
            for sel in cfg["selectors"]:
                for t in range(nT):
                    group = f"{name}|{sel}|T{t}"
                    units.append(dict(config=name, data=data, selector=sel, clf=clf_name, T=t, kind="truth",
                                      R=cfg["truth"]["R"], rep=0, group=group))
                    for R, nrep in Rs.items():
                        for rep in range(nrep):
                            units.append(dict(config=name, data=data, selector=sel, clf=clf_name, T=t,
                                              kind="rep", R=int(R), rep=rep, group=group))
    return units


def unit_path(out: str, u: Dict) -> str:
    safe = u["config"].replace("=", "_")
    return os.path.join(out, "units", f"{safe}__{u['selector']}__{u['clf']}__T{u['T']}__{u['kind']}_R{u['R']}_{u['rep']:03d}.json")


def environment() -> Dict:
    import importlib
    vers = {}
    for mod in ("numpy", "scipy", "sklearn", "xgboost"):
        try:
            vers[mod] = importlib.import_module(mod).__version__
        except Exception:  # pragma: no cover
            vers[mod] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "processor": platform.processor(), "packages": vers}


def run(cfg_path: str, out: str, worker: int = 0, workers: int = 1, only: Optional[Iterable[str]] = None,
        max_minutes: Optional[float] = None) -> None:
    cfg = json.load(open(cfg_path))
    os.makedirs(os.path.join(out, "units"), exist_ok=True)
    os.makedirs(os.path.join(out, "panels"), exist_ok=True)
    meta = os.path.join(out, "metadata.json")
    if not os.path.exists(meta):
        json.dump({"config": cfg, "config_hash": stable_hash(cfg), "environment": environment(),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, open(meta, "w"), indent=1)
    units = plan_units(cfg)
    if only:
        only = set(only)
        units = [u for u in units if u["clf"] in only or u["config"] in only]
    groups = sorted({u["group"] for u in units})
    mine = {g for i, g in enumerate(groups) if i % workers == worker}   # a group (panel cache) has one owner
    t_start = time.time()
    log = open(os.path.join(out, f"log_worker{worker}.txt"), "a")
    current_group, cache = None, None
    for u in units:
        if u["group"] not in mine:
            continue
        path = unit_path(out, u)
        if os.path.exists(path):
            continue
        if max_minutes and (time.time() - t_start) / 60 > max_minutes:
            log.write(f"{time.strftime('%H:%M:%S')} stopping: time budget reached\n")
            break
        if u["group"] != current_group:
            if cache:
                cache.save()
            safe = u["group"].replace("|", "__").replace("=", "_")
            cache = PanelCache(os.path.join(out, "panels", safe + ".npz"))
            current_group = u["group"]
        st = Setting(cfg, u["data"], u["selector"], u["clf"])
        t0, c0 = time.time(), time.process_time()
        if u["kind"] == "truth":
            res = run_truth(st, u["T"], cache)
        else:
            rng_boot = np.random.default_rng(seed_for(st.key, u["T"], 7, st.sel_code, st.clf_code, u["R"], u["rep"]))
            res = run_rep(st, u["T"], u["R"], u["rep"], cache, rng_boot)
        res.update({k: u[k] for k in ("config", "selector", "clf", "T", "kind", "R", "rep")})
        res["data"] = u["data"]
        res["seconds"] = time.time() - t0
        res["cpu_seconds"] = time.process_time() - c0
        cache.save()
        tmp = path + ".tmp"
        json.dump(res, open(tmp, "w"), default=float)
        os.replace(tmp, path)
        log.write(f"{time.strftime('%H:%M:%S')} {os.path.basename(path)} {res['seconds']:.1f}s fits={res['n_fits']}\n")
        log.flush()
    if cache:
        cache.save()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/crossed_sim.json")
    ap.add_argument("--out", default="runs/crossed_sim")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="restrict to classifier names or configuration names")
    ap.add_argument("--max-minutes", type=float, default=None, help="stop cleanly after this many minutes")
    a = ap.parse_args(argv)
    run(a.config, a.out, a.worker, a.workers, a.only, a.max_minutes)


if __name__ == "__main__":
    main()
